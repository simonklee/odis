from __future__ import absolute_import

import re
import redis
import time
import datetime

from . import config

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

r = redis.StrictRedis(**config.REDIS_DATABASE)

EMPTY_VALUES = (None, '', [], (), {})
CHUNK_SIZE = 50

class ValidationError(Exception):
    'An error raised on validation'

class FieldError(Exception):
    'An error raised on validation'

class EmptyError(Exception):
    'An error raised on validation'

class EMPTY:
    pass

class Collection(object):
    '''Create a collection object saved in Redis.
    `key` the redis key for collection.
    `db` or `pipe` must be provided'''
    def __init__(self, model, key):
        self.model = model
        self.db = model._db

        if not self.db:
            raise Exception('No connection specified')

        self.key = key

        for attr in self.METHODS:
            setattr(self, attr, getattr(self.db, attr))

    def clear(self):
        self.db.delete(self.key)

    def sort(self, **options):
        return self.db.sort(self.key, **options)

    def sort_by(self, by, **options):
        options['by'] = self.model.key_for('obj', pk='*->%s' % by)
        get = options.pop('get', None)

        if get:
            if isinstance(get, (tuple, list)):
                get = (self.model.key_for('obj', pk='*->%s' % get[0]), '#')
            options['get'] = get

        return self.db.sort(self.key, **options)

    METHODS = ()

class Set(Collection):
    def __len__(self):
        return self.db.scard(self.key)

    def __iter__(self):
        return iter(self.db.smembers(self.key))
        #return itertools.imap(self._map, self.db.smembers(self.key))

    def __contains__(self, value):
        return self.db.sismember(self.key, value)

    def __repr__(self):
        return "<%s '%s' %s(%s)>" % (self.__class__.__name__, self.model.__name__, self.key, self.db.smembers(self.key))

    #def _map(self, pk):
    #    data = self.db.hgetall(self.model.key_for('obj', pk=pk))
    #    return self.model().from_dict(data, to_python=True)

    def diff(self, **kwargs):
        return self._apply_sort(
            self.db.sdiffstore,
            QueryKey(self.model).build_key({}, kwargs, ''),
            kwargs)

    def inter(self, **kwargs):
        return self._apply_sort(
            self.db.sinterstore,
            QueryKey(self.model).build_key(kwargs, {}, ''),
            kwargs)

    def _apply_sort(self, command, target, opts, expire=60):
        # seperate input
        keys = [self.key] + QueryKey(self.model).field_keys(opts)

        # then do the command
        command(target, *keys)

        # key is just temporary
        self.db.expire(target, expire)

        # return the result as a new Set
        return Set(self.model, target)

    METHODS = ('sadd', 'scard', 'sdiff', 'sdiffstore', 'sinter', 'sinterstore', 'sismember',
        'smembers', 'smove', 'spop', 'srandmember', 'srem', 'sunion', 'sunionstore')

class Index(Set):
    def inter(self, **kwargs):
        keys = QueryKey(self.model).field_keys(kwargs)

        if len(keys) == 0:
            return self
        elif len(keys) > 1:
            return super(Index, self).inter(**kwargs)
        else:
            return Index(self.model, keys[0])

    def diff(self, **kwargs):
        keys = QueryKey(self.model).field_keys(kwargs)

        if len(keys) == 0:
            return self
        else:
            return super(Index, self).diff(**kwargs)

class SortedSet(Collection):
    def __getitem__(self, key):
        if getattr(self, 'sort_desc', False):
            func = self.db.zrevrange
        else:
            func = self.db.zrange

        if isinstance(key, (slice)):
            start = key.start or 0

            if key.stop and key.stop > 0:
                stop = key.stop - 1
            else:
                stop = key.stop or -1

            return func(self.key, start, stop)
        else:
            return func(self.key, key, key)[0]

    def __len__(self):
        if not hasattr(self, '_len'):
            self._len = self.zcard(self.key)
        return self._len

    def __iter__(self):
        return self.zrange(self.key, 0, -1).__iter__()

    def __reversed__(self):
        return self.zrevrange(self.key, 0, -1).__iter__()

    def __repr__(self):
        return "<%s '%s'>" % (self.__class__.__name__, self.key)

    def sort(self, by, **opts):
        if len(by) == 0:
            by = (self.key, )

        self.sort_desc = opts.pop('desc', False)
        return self.db.zinterstore(self.key, by, **opts)

    def expireat(self, timestamp):
        return self.db.expireat(self.key, timestamp)

    METHODS = ('zadd', 'zcard', 'zcount', 'zincrby', 'zinterstore', 'zrange',
        'zrangebyscore', 'zrank', 'zrem', 'zremrangebyrank',
        'zremrangebyscore', 'zrevrange', 'zrevrangebyscore', 'zrevrank',
        'zscore', 'zunionstore')

class QueryKey(object):
    def __init__(self, model):
        self.model = model

    def match_and_parse_key(self, key, data):
        parts = self.parse_key(key)

        if not self._match_key_parts(parts[1], data, False):
            ok = False
        elif not self._match_key_parts(parts[2], data, True):
            ok = False
        elif self.model.key_for('all') != parts[0][0]:
            ok = False
        else:
            ok = True

        return ok, parts

    def _match_key_parts(self, parts, data, negate=False):
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
        key = '~' + self.model.key_for('all')
        sorted_inter = self.field_keys(inter)
        sorted_diff = self.field_keys(diff)

        if len(inter) > 0:
            key = key + '+' + '+'.join(sorted_inter)

        if len(diff) > 0:
            key = key + '-' + '-'.join(sorted_diff)

        if len(sort) > 0:
            key = key + '^' + sort

        return key

class Query(object):
    def __init__(self, model):
        self.res = None
        self.index = 0
        self.diff = {}
        self.inter = {}
        self.sort_by = ''
        self.sort_opts = {}
        self.model = model
        self.db = model._db
        self._hits = 0

    def _clone(self):
        obj = self.__class__(self.model)
        obj.diff = self.diff.copy()
        obj.inter = self.inter.copy()
        obj.sort_by = self.sort_by
        obj.sort_opts = self.sort_opts.copy()
        obj._hits = self._hits
        return obj

    def count(self):
        self.do_query()
        return len(self.res)

    def do_query(self):
        if self.res:
            return

        key = QueryKey(self.model).build_key(self.inter, self.diff, self.sort_by)
        self.res = SortedSet(self.model, key)
        timestamp = int(time.time() + 604800.0)

        if self.db.zadd(self.model.key_for('queries'), timestamp, self.res.key) == 0:
            self.res.expireat(timestamp)
            return

        self._hits = self._hits + 1
        base = Index(self.model, self.model.key_for('all'))
        intersected = base.inter(**self.inter)
        diffed = intersected.diff(**self.diff)
        self.apply_sort(diffed)
        self.res.expireat(timestamp)

    def apply_sort(self, index):
        if len(self.sort_by) == 0:
            self.add_sorting(self.model.key_for('zindex', field='pk'), {})

        self.res.sort([index.key, self.sort_by], **self.sort_opts)

    def new_chunk(self, start, stop):
        p = self.db.pipeline()

        for pk in self.res[start:stop]:
            p.hgetall(self.model.key_for('obj', pk=pk))

        return p.execute()

    def fetch_values(self):
        self.do_query()

        while self.index < self.count():
            val = self.new_chunk(self.index, self.index + CHUNK_SIZE)
            self.index = self.index + CHUNK_SIZE
            return val

        return []

    def result_iter(self):
        for chunks in iter(self.fetch_values, []):
            for data in chunks:
                yield data

    def add_query(self, negate, **kwargs):
        if negate:
            self.diff.update(**kwargs)
        else:
            self.inter.update(**kwargs)

        return self

    def add_sorting(self, field, opts):
        self.sort_by = field
        self.sort_opts = opts

class QuerySet(object):
    def __init__(self, model, query=None):
        self.model = model
        self.query = query or Query(model)
        self._cache = None
        self._iter = None

    def filter(self, **kwargs):
        c = self._clone()
        c.query.add_query(False, **kwargs)
        return c

    def exclude(self, **kwargs):
        c = self._clone()
        c.query.add_query(True, **kwargs)
        return c

    def order(self, field, *args, **kwargs):
        opts = kwargs.copy()

        if field.startswith('-'):
            field, opts['desc'] = field[1:], True
        else:
            opts['desc'] = False

        if field in self.model._zindices:
            c = self._clone()
            field = self.model.key_for('zindex', field=field)
        else:
            raise ValidationError('Sortable only by sorted set indexed values')

        c.query.add_sorting(field, opts)
        return c

    def iterator(self):
        for data in self.query.result_iter():
            yield self.model().from_dict(data, to_python=True)

    def __len__(self):
        return self.query.count()

    def __getitem__(self, key):
        pass

    def __iter__(self):
        if self._cache is None:
            self._cache = []
            self._iter = self.iterator()

        if self._iter:
            return self._cache_iter()

        return iter(self._cache)

    def _clone(self, *args, **kwargs):
        return self.__class__(self.model, query=self.query._clone(*args, **kwargs))

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

    def _cache_fill(self):
        try:
            for i in range(CHUNK_SIZE):
                self._cache.append(self._iter.next())
        except StopIteration:
            self._iter = None

class Field(object):
    def __init__(self, index=False, unique=False, nil=False, default=EMPTY):
        '''An attribute field on a model.

        `index`: if set the value is used create an additional key which maps to
        `pk`. This makes it possible to find pk by this field attribute.'''
        self.unique = unique
        self.index = index or unique
        self.nil = nil
        self.default = default

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

    def validate(self, instance, value):
        if self.nil and self.is_empty(value):
            return
        elif self.is_empty(value):
            raise ValidationError('`%s` unexpected nil value' % self.name)
        try:
            setattr(instance, self.name, self.to_python(value))
        except TypeError, e:
            raise ValidationError('`%s` invalid type "%s"' % (self.name, e.message))

        if self.unique and not self.is_unique(instance, value):
            raise ValidationError('%s `%s` not unique' % (self.name, value))

    def is_empty(self, value):
        return value in EMPTY_VALUES

    def is_unique(self, instance, value):
        '''Check uniqueness on all fields with `unique=True`'''
        key = instance.key_for('index', field=self.name, value=value)
        return (instance.pk and instance._db.sismember(key, instance.pk) \
                or instance._db.scard(key) == 0)

    def to_python(self, value):
        return value

    def to_db(self, value):
        return value

class ZField(Field):
    def __init__(self, zindex=False, **kwargs):
        super(ZField, self).__init__(**kwargs)
        self.zindex = zindex

class IntegerField(ZField):
    def to_python(self, value):
        return int(value)

    def to_db(self, value):
        return unicode(value)

class DateTimeField(ZField):
    def __init__(self, auto_now_add=False, **kwargs):
        super(DateTimeField, self).__init__(**kwargs)
        self.auto_now_add = auto_now_add

    def validate(self, instance, value):
        if self.is_empty(value) and self.auto_now_add:
            value = datetime.datetime.now()
            setattr(instance, self.name, value)

        super(DateTimeField, self).validate(instance, value)

    def to_python(self, value):
        if not isinstance(value, datetime.datetime):
            return datetime.datetime.fromtimestamp(float(value))

        return value

    def to_db(self, value):
        return u'%d.%s' % (time.mktime(value.timetuple()), str(value.microsecond).ljust(6, '0'))

class DateField(ZField):
    def to_python(self, value):
        if not isinstance(value, datetime.date):
            return datetime.date.fromtimestamp(float(value))

        return value

    def to_db(self, value):
        return u'%f' % time.mktime(value.timetuple())

class BaseModel(type):
    def __new__(meta, name, bases, attrs):
        attrs['pk'] = IntegerField(nil=True, zindex=True)
        cls = super(BaseModel, meta).__new__(meta, name, bases, attrs)
        cls._fields = {}
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
            'queries': cls._namespace + '_cached_queries'}

        for k, v in attrs.iteritems():
            if isinstance(v, Field):
                cls._fields[k] = v
                v.name = k

                if v.index:
                    cls._indices.append(k)

                if getattr(v, 'zindex', False):
                    cls._zindices.append(k)

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
        return cls._keys[name].format(**kwargs)

    @property
    def key(self):
        return self.key_for('obj', pk=self.pk)

    def is_valid(self):
        self._errors = {}

        for name, field in self._fields.items():
            try:
                field.validate(self, getattr(self, name))
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

        self.write()
        return True

    def write(self):
        data = self.as_dict(to_db=True)
        p = self._db.pipeline()
        # first we delete prior data
        p.delete(self.key)
        # then we set the new data
        p.hmset(self.key, data)
        # add pk to index for model
        p.sadd(self.key_for('all'), self.pk)

        # make sure we delete indices not more in use.
        indices_key = self.key_for('indices', pk=data['pk'])
        pattern = re.compile(r'%s_zindex\:' % self._namespace)

        for k in self._db.smembers(indices_key):
            if pattern.search(k):
                p.zrem(k, data['pk'])
            else:
                p.srem(k, data['pk'])

        # add all indexed keys to their index
        for k in self._indices:
            key = self.key_for('index', field=k, value=data[k])
            p.sadd(key, data['pk'])
            p.sadd(indices_key, key)

        # add all sorted set indexed keys to their index
        for k in self._zindices:
            key = self.key_for('zindex', field=k)
            p.zadd(key, data[k], data['pk'])
            p.sadd(indices_key, key)

        # flush query cache for model
        for k in self._db.zrange(self.key_for('queries'), 0, -1):
            p.delete(k)

        # del query cache key
        p.delete(self.key_for('queries'))
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

        for name, field in self._fields.items():
            if to_db:
                data[name] = field.to_db(getattr(self, name))
            else:
                data[name] = getattr(self, name)

        return data
