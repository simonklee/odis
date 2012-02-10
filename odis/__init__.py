# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re
import redis
import time
import datetime
import itertools
import functools

from . import config
from .utils import safe_bytestr, safe_unicode, s

try:
    import odisconfig
    for attr in dir(config):
        if attr.startswith('__'):
            continue
        try:
            setattr(config, attr, getattr(odisconfig, attr))
        except AttributeError:
            pass
except ImportError:
    pass

EMPTY_VALUES = (None, '', [], (), {})
CHUNK_SIZE = 50

class Redis(redis.StrictRedis):
    RESPONSE_CALLBACKS = redis.client.dict_merge(
        redis.StrictRedis.RESPONSE_CALLBACKS,
        {
            'EVAL': lambda r: itertools.imap(redis.client.pairs_to_dict, iter(r)),
        }
    )

    hgetallrel_lua = '''
        local keys, argv = KEYS, ARGV
        local obj_key = table.remove(keys, 1)
        local pks = argv
        local index = {}
        local r = {}
        local extra_keys = {}
        local next = next

        i = 1
        while i < #keys do
            table.insert(extra_keys, {member=keys[i], fmt=keys[i + 1]})
            i = i + 2
        end

        for i, pk in ipairs(pks) do
            local key = obj_key:gsub('{pk}', pk)
            local obj = redis.call('hgetall', key)

            --ignore empty results
            if next(obj) ~= nil then
                table.insert(r, obj)

                --populate the index of keys location in the objects table, called once.
                if #index == 0 and #extra_keys > 0 then
                    for i, key in ipairs(obj) do
                        index[key] = i + 1
                    end
                end

                --for every extra related key we want to retrieve we add it after
                --the current item.
                for i, v in ipairs(extra_keys) do
                    local key = v.fmt:gsub('{pk}', tostring(obj[index[v.member]]))
                    table.insert(r, redis.call('hgetall', key))
                end
            end
        end

        return r
        '''

    def hgetallrel(self, name, pks, fieldkeys=None):
        if fieldkeys:
            extra = list(itertools.chain.from_iterable(fieldkeys))
        else:
            extra = []

        return self.execute_command('EVAL', self.hgetallrel_lua, len(extra) + 1, name, *extra + pks)

r = Redis(**config.REDIS_DATABASE)

class ValidationError(Exception):
    'An error raised on validation'

class FieldError(Exception):
    'An error raised on wrong field type'

class EmptyError(Exception):
    'An error raised on no result'

class EMPTY:
    pass

class Collection(object):
    METHODS = ()

    def __init__(self, key, db=None, model=None, map_res=False, callback=None):
        self.key = key
        self.db = db or r

        if map_res and not (model or callback):
            raise ValueError('need callback or model paramater to perform map on results')

        self.map_res = map_res
        self.model = model
        self.callback = callback or self.map_pk_to_model if map_res else None

        for attr in self.METHODS:
            setattr(self, attr, functools.partial(getattr(self.db, attr), self.key))

    def _clone(self, key):
        return self.__class__(
            key,
            db=self.db,
            model=self.model,
            map_res=self.map_res,
            callback=self.callback)

    def flush(self):
        self.db.delete(self.key)

    def add(self, *args, **kwargs):
        raise NotImplementedError

    def delete(self, *members):
        raise NotImplementedError

    def map_pk_to_model(self, pk):
        data = self.db.hgetall(self.model.key_for('obj', pk=pk))
        return self.model().from_dict(data, to_python=True)

class Set(Collection):
    METHODS = ('sadd', 'scard', 'sdiff', 'sdiffstore', 'sinter', 'sinterstore', 'sismember',
        'smembers', 'smove', 'spop', 'srandmember', 'srem', 'sunion', 'sunionstore')

    def __len__(self):
        return self.db.scard(self.key)

    def __iter__(self):
        if self.map_res:
            return itertools.imap(self.callback, iter(self.db.smembers(self.key)))

        return iter(self.db.smembers(self.key))

    def __contains__(self, value):
        return self.db.sismember(self.key, value)

    def __repr__(self):
        return "<%s %s(%s)>" % (self.__class__.__name__, self.key, self.db.smembers(self.key))

    def diff(self, target, keys):
        return self._associative_op(self.db.sdiffstore, target, keys)

    def inter(self, target, keys):
        return self._associative_op(self.db.sinterstore, target, keys)

    def add(self, *members):
        return self.db.sadd(self.key, *members)

    def replace(self, *members):
        'replace all members in a set with new members'
        p = self.db.pipeline(transaction=True)
        p.delete(self.key)
        p.sadd(self.key, *members)
        p.execute()

    def all(self):
        return iter(self)

    def delete(self, *members):
        return self.db.srem(self.key, *members)

    def _associative_op(self, command, target, keys, expire=60):
        if len(keys) == 0:
            return self

        # seperate input
        keys = [self.key] + keys

        # then do the command
        command(target, keys)

        # key is just temporary
        self.db.expire(target, expire)

        # return the result as a new Set
        return self._clone(target)

class SortedSet(Collection):
    METHODS = ('zadd', 'zcard', 'zcount', 'zincrby', 'zinterstore', 'zrange',
        'zrangebyscore', 'zrank', 'zrem', 'zremrangebyrank',
        'zremrangebyscore', 'zrevrange', 'zrevrangebyscore', 'zrevrank',
        'zscore', 'zunionstore')

    def __init__(self, *args, **kwargs):
        self.desc = kwargs.pop('desc', False)
        super(SortedSet, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if self.desc:
            func = self.db.zrevrange
        else:
            func = self.db.zrange

        if isinstance(key, (slice)):
            start = key.start or 0

            if key.stop and key.stop > 0:
                stop = key.stop - 1
            else:
                stop = key.stop or -1

            if self.map_res:
                return map(self.callback, func(self.key, start, stop))

            return func(self.key, start, stop)
        else:
            obj = func(self.key, key, key)[0]

            if self.map_res:
                return self.callback(obj)

            return obj

    def __len__(self):
        if not hasattr(self, '_len'):
            self._len = self.db.zcard(self.key)
        return self._len

    def __iter__(self):
        if self.map_res:
            return itertools.imap(self.callback, self.db.zrange(self.key, 0, -1))

        return iter(self.db.zrange(self.key, 0, -1))

    def __reversed__(self):
        if self.map_res:
            return itertools.imap(self.callback, self.db.zrevrange(self.key, 0, -1))

        return self.db.zrevrange(self.key, 0, -1).__iter__()

    def __repr__(self):
        return "<%s '%s'>" % (self.__class__.__name__, self.key)

    def _clone(self, key):
        c = super(SortedSet, self)._clone(key)
        c.desc = self.desc
        return c

    def _associative_op(self, command, target, keys, expire=60):
        # seperate input
        keys = [self.key] + keys

        # then do the command
        command(target, keys)

        # key is just temporary
        self.db.expire(target, expire)

        # return the result as a new Set
        return self._clone(target)

    def add(self, score, member):
        self.db.zadd(self.key, score, member)

    def all(self):
        return iter(self)

    def delete(self, *members):
        return self.db.zrem(self.key, *members)

    def inter(self, target, keys):
        return self._associative_op(self.db.zinterstore, target, keys)

    def diff(self, target, keys):
        return self._associative_op(self.db.zdiffstore, target, keys)

    #@timeit
    def sort(self, by, **opts):
        if len(by) == 0:
            by = (self.key, )

        return self.db.zinterstore(self.key, by, **opts)

    def expireat(self, timestamp):
        return self.db.expireat(self.key, timestamp)

class QueryKey(object):
    def __init__(self, model, base_key=None):
        self.model = model
        self.base_key = base_key or model.key_for('all')

    def match_and_parse_keys(self, keys, data):
        'returns only ok keys in a dict where the value is the score field'
        p = self.model._db.pipeline()
        good = {}
        async = []

        # figure out if a key pattern matches the data. Some keys require
        # to be evaluted by checking pk in a different set, we do this with
        # pipe for all keys which require it, after the first run through.
        for k in keys:
            ok, parts, async_eval = self.match_and_parse_key(k, data, pipe=p)

            if async_eval:
                async.append(k)

            if ok:
                good[k] = parts[3][0].split(':')[1]

        # remove any key which did not evalute after the
        # async evaluation of keys which required it.
        for i, ok in enumerate(p.execute()):
            if not ok:
                try:
                    del good[async[i]]
                except KeyError:
                    pass

        return good

    def match_and_parse_key(self, key, data, pipe=None):
        db = pipe or self.model._db
        parts = self.parse_key(key)
        base_ok, async_eval = self._match_base_part(parts[0][0], data, db)

        if not base_ok:
            ok = False
        elif not self._match_associative_parts(parts[1], data, False):
            ok = False
        elif not self._match_associative_parts(parts[2], data, True):
            ok = False
        else:
            ok = True

        return ok, parts, (async_eval and not pipe is None)

    def _match_base_part(self, base, data, db):
        parts = base.split(':')

        if len(parts) == 4 and self.base_key == parts[3]:
            # if db is a redis pipe, running pipe.execute() returns
            # a list of [true, true, false, ... ] for each key you
            # match.
            if db.sismember(base, data['pk']):
                return True, True
        elif self.base_key == parts[0]:
            return True, False

        return False, False

    def _match_associative_parts(self, parts, data, negate=False):
        for part in parts:
            model, field, value = part.split(':')

            if (data[field] == value) is negate:
                return False

        return True

    def field_keys(self, data):
        keys = []
        sorted_keys = data.keys()
        sorted_keys.sort()

        for k in sorted_keys:
            keys.append(self.model.key_for('index', field=k, value=data[k]))

        return keys

    def parse_key(self, key):
        parts = [('^', []), ('-', []), ('+', []), ('~', [])]

        for op, l in parts:
            key = self._parse_key_part(key, op, l)

        return [part for op, part in reversed(parts)]

    def _parse_key_part(self, key, sep, res):
        a, op, b = key.rpartition(sep)

        if op == '':
            return key
        else:
            res.insert(0, b)
            return self._parse_key_part(a, sep, res)

    def build_key(self, inter, diff, sort):
        'The key is defined based on the model, inter, diff and sorting'
        key = '~' + self.base_key
        sorted_inter = self.field_keys(inter)
        sorted_diff = self.field_keys(diff)

        if len(inter) > 0:
            key = key + '+' + '+'.join(sorted_inter)

        if len(diff) > 0:
            key = key + '-' + '-'.join(sorted_diff)

        if len(sort) > 0:
            key = key + '^' + sort

        return key

    def build_and_field_keys(self, key_type, opts):
        if key_type == 'inter':
            return self.build_key(opts, {}, ''), self.field_keys(opts)
        elif key_type == 'diff':
            return self.build_key({}, opts, ''), self.field_keys(opts)
        raise ValueError('key_type is unknown')

class Query(object):
    def __init__(self, model, base_key):
        self.res = None
        self.pos = 0
        self.off = 0
        self.limit = None
        self.diff = {}
        self.inter = {}
        self.includes = []
        self.sort_by = ''
        self.desc = False
        self.model = model
        self.base_key = base_key
        self.db = model._db
        self._hits = 0

    def _clone(self):
        obj = self.__class__(self.model, self.base_key)
        obj.diff = self.diff.copy()
        obj.inter = self.inter.copy()
        obj.includes = self.includes[:]
        obj.sort_by = self.sort_by
        obj.limit = self.limit
        obj.off = self.off
        obj.desc = self.desc
        obj._hits = self._hits
        return obj

    def count(self):
        'count = MIN(N - off, limit - off)'
        self.execute_query()
        n = max(0, len(self.res) - self.off)

        if self.limit:
            n = min(n, self.limit - self.off)

        return n

    def execute_query(self):
        if self.res:
            return

        if len(self.sort_by) == 0:
            self.add_sorting(self.model.key_for('zindex', field='pk'))

        key = QueryKey(self.model, self.base_key).build_key(self.inter, self.diff, self.sort_by)
        self.res = SortedSet(key, desc=self.desc)
        timestamp = int(time.time() + 604800.0)

        if self.db.zadd(self.model.key_for('queries'), timestamp, self.res.key) == 0:
            self.res.expireat(timestamp)
            return

        self._hits = self._hits + 1

        # base key for query
        base = Set(self.base_key)

        # do set intersect on base set and inter keys
        target, keys = QueryKey(self.model, base.key).build_and_field_keys('inter', self.inter)
        intersected = base.inter(target, keys)

        # do set sym difference on result of intersection with diff keys
        target, keys = QueryKey(self.model, intersected.key).build_and_field_keys('diff', self.diff)
        diffed = intersected.diff(target, keys)

        # only sort will create the final sorted set
        self.res.sort({diffed.key:0, self.sort_by:1})
        self.res.expireat(timestamp)

    #@timeit
    def new_chunk(self, start, stop):
        if len(self.includes) > 0:
            # do query using lua scripts to retrieve extra objects
            # for each obj.
            return self.db.hgetallrel(self.model._keys['obj'], self.res[start:stop], self.includes)

        p = self.db.pipeline()

        for pk in self.res[start:stop]:
            p.hgetall(self.model.key_for('obj', pk=pk))

        return p.execute()

    def fetch_values(self):
        self.execute_query()
        self.pos = max(self.off, self.pos)

        while self.pos < self.off + self.count():
            end = self.pos + CHUNK_SIZE

            if self.limit: #and self.limit >= self.count(): prevent endless loop
                end = min(end, self.limit)

            val = self.new_chunk(self.pos, end)
            self.pos = end
            return val

        return []

    def result_iter(self):
        for chunks in iter(self.fetch_values, []):
            for chunk in chunks:
                yield chunk

    def is_bound(self):
        return self.off != 0 or self.limit != None

    def add_bounds(self, off=0, limit=None):
        self.off = off
        self.limit = limit

    def add_query(self, negate, **kwargs):
        if negate:
            self.diff.update(**kwargs)
        else:
            self.inter.update(**kwargs)

        return self

    def add_sorting(self, field, desc=False):
        self.sort_by = field
        self.desc = desc

    def add_includes(self, *fieldkeys):
        self.includes = fieldkeys

class QuerySet(object):
    def __init__(self, model, key=None, query=None):
        self.model = model
        self.base_key = key or model.key_for('all')
        self.query = query or Query(model, base_key=self.base_key)
        self._cache = None
        self._iter = None
        self._includes = []

    def all(self):
        return self.filter()

    def filter(self, **kwargs):
        return self._filter_or_exclude(False, **kwargs)

    def exclude(self, **kwargs):
        return self._filter_or_exclude(True, **kwargs)

    def _filter_or_exclude(self, negate, **kwargs):
        assert not self.query.is_bound(), 'cannot filter once slice has been taken'
        c = self._clone()
        c.query.add_query(negate, **kwargs)
        return c

    def get(self, **kwargs):
        assert len(kwargs) == 1, 'only possible to get() by single key=value lookup'
        key, value = kwargs.popitem()

        if key == 'pk':
            pk = value
        else:
            if not key in self.model._indices:
                raise FieldError('`%s` is not a valid index field.' % key)

            pk = r.srandmember(self.model.key_for('index', field=key, value=value))

        if not pk or not r.sismember(self.base_key, pk):
            raise EmptyError('`%s(%s=%s)` returned an empty result' %
                    (self.model.__name__, key, pk or value))

        data = r.hgetall(self.model.key_for('obj', pk=pk))
        return self.model().from_dict(data, to_python=True)

    def desc(self):
        c = self._clone()
        c.query.desc = True
        return c

    def order(self, field, *args, **kwargs):
        if field.startswith('-'):
            field, desc = field[1:], True
        else:
            desc = False

        if field in self.model._zindices:
            c = self._clone()
            field = self.model.key_for('zindex', field=field)
        else:
            raise FieldError('Sortable only by sorted set indexed values')

        c.query.add_sorting(field, desc)
        return c

    def include(self, *fields):
        fieldkeys = []
        models = []

        for field in fields:
            if field not in self.model._fks:
                raise FieldError('%s is not a ForeignField' % field)

            model = self.model._fields[field].model
            fieldkeys.append((field, model._keys['obj']))
            models.append((field, model))

        c = self._clone()
        c._includes = models
        c.query.add_includes(*fieldkeys)
        return c

    def iterator(self):
        i = 0

        for data in self.query.result_iter():
            if i == 0:
                d = self.model().from_dict(data, to_python=True)
            elif data:
                # data might be an empty dict — at which point we dont want
                # to do anything with it.
                field, model = self._includes[i - 1]
                setattr(d, field, model().from_dict(data, to_python=True))

            if i != len(self._includes):
                i = i + 1
            else:
                i = 0
                yield d

    def count(self):
        return len(self._clone())

    def __len__(self):
        return self.query.count()

    def __getitem__(self, k):
        if self._cache:
            if self._iter:
                if isinstance(k, slice):
                    off = k.stop if k.stop else None
                else:
                    off = k + 1

                if len(self._cache) < off:
                    self._cache_fill(num=off - len(self._cache))

            return self._cache[k]

        qs = self._clone()

        if isinstance(k, slice):
            qs.query.add_bounds(off=k.start or 0, limit=k.stop)
            return qs
        else:
            qs.query.add_bounds(off=k, limit=k+1)
            return list(qs)[0]

    def __iter__(self):
        if self._cache is None:
            self._cache = []
            self._iter = self.iterator()

        if self._iter:
            return self._cache_iter()

        return iter(self._cache)

    def _flush_local_cache(self):
        self._cache = None
        self._iter = None
        self.query = self.query._clone()

    def _clone(self, *args, **kwargs):
        c = self.__class__(self.model, key=self.base_key, query=self.query._clone())
        c._includes = self._includes[:]
        return c

    def _cache_iter(self):
        pos = 0

        while 1:
            upper = len(self._cache)

            while pos < upper:
                yield self._cache[pos]
                pos = pos + 1

            if not self._iter:
                raise StopIteration

            if len(self._cache) <= pos:
                self._cache_fill()

    def _cache_fill(self, num=None):
        try:
            for i in range(num or CHUNK_SIZE):
                self._cache.append(self._iter.next())
        except StopIteration:
            self._iter = None

class Field(object):
    msg = {
        'type': 'invalid type "%s"',
        'nil': 'unexpected nil value',
        'unique': '`%s` not unique',
        'choice': '`%s` invalid choice',
    }

    def __init__(self,
            verbose_name=None,
            index=False,
            unique=False,
            choices=None,
            nil=False,
            default=EMPTY):
        '''
        `verbose_name`:  A verbose name for the field.
        `index`:         Key for value maps to `pk`. lookup by value possible.
        `unique`:        Only one model with a given value.
        `nil`:           Allow nil value.
        `default`:       Set to default value if otherwise empty. '''
        self.verbose_name = verbose_name
        self.unique = unique
        self.index = index or unique
        self.nil = nil
        self.default = default
        self.choices = choices or []

    def __set__(self, instance, value):
        return setattr(instance, '_' + self.name, value)

    def __get__(self, instance, owner):
        attr = '_' + self.name

        if hasattr(instance, attr):
            return getattr(instance, attr)
        elif self.default != EMPTY:
            setattr(instance, attr, self.default)
            return self.default

        return None

    def clean(self, instance, value):
        if not self.is_empty(value):
            value = self.to_python(value)

        self.validate(instance, value)
        return value

    def validate(self, instance, value):
        if self.nil and self.is_empty(value):
            return

        if self.is_empty(value):
            raise ValidationError(self.msg['nil'])

        if self.unique and not self.is_unique(instance, value):
            raise ValidationError(self.msg['unique'] % value)

        if self.choices:
            for v, name in self.choices:
                if v == value:
                    return

            raise ValidationError(self.msg['choice'] % value)

    def is_empty(self, value):
        return value in EMPTY_VALUES

    def is_unique(self, instance, value):
        '''Check uniqueness on all fields with `unique=True`'''
        key = instance.key_for('index', field=self.name, value=value)
        return (instance.pk and instance._db.sismember(key, instance.pk) \
                or instance._db.scard(key) == 0)

    def to_python(self, value):
        'to_python should recieve a non-empty value'
        return value

    def to_db(self, value):
        return value

class CharField(Field):
    def to_python(self, value):
        try:
            return safe_unicode(value)
        except UnicodeEncodeError, e:
            raise ValidationError(self.msg['type'] % e.message)

    def to_db(self, value):
        return safe_bytestr(value)

class ZField(Field):
    def __init__(self, zindex=False, **kwargs):
        super(ZField, self).__init__(**kwargs)
        self.zindex = zindex

class IntegerField(ZField):
    def to_python(self, value):
        try:
            return int(value)
        except (TypeError, ValueError), e:
            raise ValidationError(self.msg['type'] % e.message)

    def to_db(self, value):
        return safe_bytestr(value)

class ForeignField(IntegerField):
    def __init__(self, model, **kwargs):
        super(ForeignField, self).__init__(**kwargs)
        self.model = model

    def __set__(self, instance, value):
        if isinstance(value, self.model):
            setattr(instance, '_' + self.name + '_obj', value)
            value = value.pk

        return super(ForeignField, self).__set__(instance, value)

    def to_python(self, value):
        if isinstance(value, self.model):
            return value

        return super(ForeignField, self).to_python(value)

class DateTimeField(ZField):
    def __init__(self, now=False, **kwargs):
        super(DateTimeField, self).__init__(**kwargs)
        self.now = now

    def __get__(self, instance, owner):
        value = super(DateTimeField, self).__get__(instance, owner)

        if self.is_empty(value) and self.now:
            value = datetime.datetime.now()

        return value

    def to_python(self, value):
        if isinstance(value, datetime.datetime):
            # make sure we always use the same precision 1 == 100000
            value.replace(microsecond=int(str(value.microsecond).ljust(6, '0')))
            return value
        elif isinstance(value, datetime.date):
            return datetime.datetime(value.year, value.month, value.day)
        try:
            return datetime.datetime.fromtimestamp(float(value))
        except TypeError, e:
            raise ValidationError(self.msg['type'] % e.message)

    def to_db(self, value):
        return u'%d.%d' % (time.mktime(value.timetuple()), value.microsecond)

class DateField(ZField):
    def __init__(self, now=False, **kwargs):
        super(DateField, self).__init__(**kwargs)
        self.now = now

    def __get__(self, instance, owner):
        value = super(DateField, self).__get__(instance, owner)

        if self.is_empty(value) and self.now:
            value = datetime.datetime.today()

        return value

    def to_python(self, value):
        if isinstance(value, datetime.datetime):
            return value.date()
        elif isinstance(value, datetime.date):
            return value
        try:
            return datetime.date.fromtimestamp(float(value))
        except TypeError, e:
            raise ValidationError(self.msg['type'] % e.message)

    def to_db(self, value):
        return u'%f' % time.mktime(value.timetuple())

class CollectionField(object):
    def __init__(self, verbose_name=None, model=None, coerce=None):
        self.verbose_name = verbose_name
        self.model = model
        self.coerce = coerce

    def __set__(self, instance, value):
        return setattr(instance, '_' + self.name, value)

    def __get__(self, instance, owner):
        assert instance, 'can only be called on instance obj'
        attr = '_' + self.name

        if not hasattr(instance, attr):
            setattr(instance, attr, self.name)

        return getattr(instance, attr)

class BaseSetField(CollectionField):
    def __get__(self, instance, owner):
        field = super(BaseSetField, self).__get__(instance, owner)
        key = instance.key_for(self.key_type, pk=instance.pk, field=field)

        if self.model:
            return self.datastructure(key, model=self.model, map_res=True)
        elif self.coerce:
            return self.datastructure(key, map_res=True, callback=self.coerce)

        return self.datastructure(key)

class SetField(BaseSetField):
    'set of values which can be pks and map to a model or type'
    datastructure = Set
    key_type = 'set'

class SortedSetField(BaseSetField):
    'sorted set of values which can be pks and map to a model or type'
    datastructure = SortedSet
    key_type = 'sortedset'

class RelField(CollectionField):
    'sorted set of pks which map to a different model and expose a RelQuerySet'
    key_type='rel'

    def __init__(self, model, *args, **kwargs):
        super(RelField, self).__init__(model=model, **kwargs)

    def __get__(self, instance, owner):
        field = super(RelField, self).__get__(instance, owner)
        model = self.model
        key = instance.key_for(
            self.key_type,
            pk=instance.pk,
            field=field,
            other=model.key_for('all'))
        ds = Set(key)

        class RelQuerySet(QuerySet):
            def add(self, *objs):
                'one or more `(score, obj)`'
                for obj in objs:
                    if not isinstance(obj, model):
                        raise TypeError('invalid model')

                    ds.add(obj.pk)
                    obj.flush_query_cache()

            def delete(self, *objs):
                for obj in objs:
                    if not isinstance(obj, model):
                        raise TypeError('invalid model')

                    ds.delete(obj.pk)
                    obj.flush_query_cache()

                self._flush_local_cache()

            def replace(self, *objs):
                'replace all members in a set with new members'
                self.delete(*self.all())
                self.model._db.delete(key)
                self.add(*objs)

        return RelQuerySet(model, key=key)

class BaseModel(type):
    def __new__(meta, name, bases, attrs):
        attrs['pk'] = IntegerField(nil=True, zindex=True, index=True)
        cls = super(BaseModel, meta).__new__(meta, name, bases, attrs)
        cls._fields = {}
        cls._coll_fields = {}
        cls._fks = []
        cls._indices = []
        cls._zindices = []
        cls._db = r

        if config.REDIS_PREFIX:
            cls._namespace = config.REDIS_PREFIX + '_' + name
        else:
            cls._namespace = name

        cls._keys = {
            'pk': cls._namespace + '_pk',
            'all' : cls._namespace + '_all',
            'obj' : cls._namespace + ':{pk}',
            'index': cls._namespace + '_index:{field}:{value}',
            'zindex': cls._namespace + '_zindex:{field}',
            'indices': cls._namespace + ':{pk}_indices',
            'queries': cls._namespace + '_cached_queries',
            'set': cls._namespace + ':{pk}_set:{field}',
            'sortedset': cls._namespace + ':{pk}_sortedset:{field}',
            'rel': cls._namespace + ':{pk}_relset:{field}:{other}'}

        for k, v in attrs.iteritems():
            if isinstance(v, Field):
                cls._fields[k] = v
                v.name = k

                if v.index:
                    cls._indices.append(k)

                if getattr(v, 'zindex', False):
                    cls._zindices.append(k)
            elif isinstance(v, CollectionField):
                cls._coll_fields[k] = v
                v.name = k

            if isinstance(v, ForeignField):
                cls._fks.append(k)

        if 'obj' in attrs:
            cls.obj = attrs['obj'](cls)
        else:
            cls.obj = QuerySet(cls)
        return cls

class Model(object):
    __metaclass__ = BaseModel

    def __init__(self, **kwargs):
        self.from_dict(kwargs)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.as_dict() == other.as_dict()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return u'<%s %s>' % (self.__class__.__name__, self.as_dict())

    @classmethod
    def key_for(cls, name, **kwargs):
        return safe_bytestr(cls._keys[name].format(**kwargs))

    @property
    def key(self):
        return self.key_for('obj', pk=self.pk)

    @property
    def errors(self):
        return getattr(self, '_errors', {})

    def is_valid(self):
        self._errors = {}

        for name, field in self._fields.items():
            v = getattr(self, name)

            if field.nil and field.is_empty(v):
                continue
            try:
                setattr(self, name, field.clean(self, v))
            except ValidationError, e:
                self._errors[name] = e.message

        try:
            self.validate()
        except ValidationError, e:
            self._errors['__all__'] = e.message

        if self._errors:
            return False
        return True

    def validate(self):
        '''custom validation'''

    def save(self):
        if not self.is_valid():
            return False

        if self.pk is None:
            setattr(self, 'pk', self._db.incr(self.key_for('pk')))

        p = self._db.pipeline()
        self.delete(write=False, pipe=p, clean_cache=False)
        self.update_query_cache(write=False, pipe=p)
        self.write(write=False, pipe=p)
        p.execute()
        return True

    def delete(self, write=True, pipe=None, clean_cache=True):
        p = pipe or self._db.pipeline()
        # first we delete hash
        p.delete(self.key)
        # rm from index for model
        p.srem(self.key_for('all'), self.pk)
        # make sure we delete indices not more in use.
        pattern = re.compile(r'%s_zindex\:' % self._namespace)
        indices_key = self.key_for('indices', pk=self.pk)

        for k in self._db.smembers(indices_key):
            if pattern.search(k):
                p.zrem(k, self.pk)
            else:
                p.srem(k, self.pk)

        p.delete(indices_key)

        if clean_cache:
            self.flush_query_cache(pk=self.pk)

        if write:
            p.execute()

    def write(self, write=True, pipe=None):
        p = pipe or self._db.pipeline()
        data = self.as_dict(to_db=True)
        # then we set the new data
        p.hmset(self.key, data)
        # add pk to index for model
        p.sadd(self.key_for('all'), self.pk)

        # add all indexed keys to their index
        indices_key = self.key_for('indices', pk=self.pk)

        for k in self._indices:
            key = self.key_for('index', field=k, value=data[k])
            p.sadd(key, self.pk)
            p.sadd(indices_key, key)

        # add all sorted set indexed keys to their index
        for k in self._zindices:
            key = self.key_for('zindex', field=k)
            p.zadd(key, data[k], self.pk)
            p.sadd(indices_key, key)

        if write:
            p.execute()

    def update_query_cache(self, write=True, pipe=None):
        'we simply update the score for the query instead of flushing it'
        p = pipe or self._db.pipeline()
        data = self.as_dict(to_db=True)
        keys = self._db.zrange(self.key_for('queries'), 0, -1)
        qk = QueryKey(self)
        good = qk.match_and_parse_keys(keys, data)

        for k in keys:
            if k in good:
                p.zadd(k, data[good[k]], self.pk)
            else:
                p.zrem(k, self.pk)

        if write:
            p.execute()

    def flush_query_cache(self, pk=None, write=True, pipe=None):
        p = pipe or self._db.pipeline()
        queries = self._db.zrange(self.key_for('queries'), 0, -1)

        if pk:
            for k in queries:
                p.zrem(k, pk)
        else:
            for k in queries:
                p.delete(k)

            # del query cache key
            p.delete(self.key_for('queries'))

        if write:
            p.execute()

    def from_dict(self, data, to_python=False):
        for name, field in self._fields.items():
            if name in data:
                if to_python:
                    setattr(self, name, field.to_python(data[name]))
                else:
                    setattr(self, name, data[name])
        return self

    def as_dict(self, to_db=False):
        data = {}

        for k, f in self._fields.items():
            v = getattr(self, k)

            if to_db:
                if f.nil and f.is_empty(v):
                    continue

                data[k] = f.to_db(v)
            else:
                data[k] = v

        return data
