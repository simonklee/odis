from __future__ import absolute_import

import redis
import time
import datetime
import functools
import itertools

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
CHUNK_SIZE = 100

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
            setattr(self, attr, functools.partial(getattr(self.db, attr), self.key))

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
        """``x.__len__() <==> len(x)``"""
        return self.scard()

    def __iter__(self):
        return itertools.imap(self._map, self.smembers())

    def __contains__(self, value):
        return self.sismember(value)

    def __repr__(self):
        return "<%s '%s' %s(%s)>" % (self.__class__.__name__, self.model.__name__, self.key, self.smembers())

    def _map(self, pk):
        data = self.db.hgetall(self.model.key_for('obj', pk=pk))
        return self.model().from_dict(data, to_python=True)

    def union(self, **kwargs):
        return self._do('union', kwargs, op='*')

    def diff(self, **kwargs):
        return self._do('diff', kwargs, op='-')

    def inter(self, **kwargs):
        return self._do('inter', kwargs)

    def _do(self, name, opts, op='+', expire=60):
        # seperate input
        keys = [self.key] + QueryKey(self.model).sort_fields(opts)
        target = '~' + op.join(keys)

        # then do the command
        if name == 'diff':
            self.db.sdiffstore(target, *keys)
        elif name == 'inter':
            self.db.sinterstore(target, *keys)
        elif name == 'union':
            self.db.sunionstore(target, *keys)
        else:
            raise ValueError('invalid name `%s`' % name)

        # key is just temporary
        self.db.expire(target, expire)

        # return the result as a new Set
        return Set(self.model, target)

    METHODS = ('sadd', 'scard', 'sdiff', 'sdiffstore', 'sinter', 'sinterstore', 'sismember',
        'smembers', 'smove', 'spop', 'srandmember', 'srem', 'sunion', 'sunionstore')

class Index(Set):
    def inter(self, **kwargs):
        keys = QueryKey(self.model).sort_fields(kwargs)

        if len(keys) == 0:
            return self
        elif len(keys) > 1:
            return super(Index, self).inter(**kwargs)
        else:
            return Index(self.model, keys[0])

    def diff(self, **kwargs):
        keys = QueryKey(self.model).sort_fields(kwargs)

        if len(keys) == 0:
            return self
        else:
            return super(Index, self).diff(**kwargs)

class QueryKey(object):
    def __init__(self, model):
        self.model = model

    def sort_fields(self, data):
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
        '''The key is predefined based on the model,
        intersection, difference and sorting.'''
        key = '~' + self.model.key_for('all')
        sorted_inter = self.sort_fields(inter)
        sorted_diff = self.sort_fields(diff)

        if len(inter) > 0:
            key = key + '+' + '+'.join(sorted_inter)

        if len(diff) > 0:
            key = key + '-' + '-'.join(sorted_diff)

        if len(sort) > 0:
            key = key + '^' + sort

        return key

class Query(object):
    def __init__(self, model):
        self.key = None
        self.llen = None
        self.index = 0
        self.diff = {}
        self.inter = {}
        self.sort = ''
        self.sort_opts = {}
        self.model = model
        self.db = model._db

    def _clone(self, zclone=False):
        if zclone:
            obj = ZQuery(self.model)
            print "zquery"
        else:
            obj = self.__class__(self.model)

        obj.diff = self.diff.copy()
        obj.inter = self.inter.copy()
        obj.sort = self.sort
        obj.sort_opts = self.sort_opts.copy()
        return obj

    def count(self):
        if not self.llen:
            self.do_query()
            self.llen = self.db.llen(self.key)

        return self.llen

    def do_query(self):
        if self.key:
            return

        base = Index(self.model, self.model.key_for('all'))
        intersected = base.inter(**self.inter)
        diffed = intersected.diff(**self.diff)
        self.key = QueryKey(self.model).build_key(self.inter, self.diff, self.sort)
        self.do_sort(diffed)

    def do_sort(self, index):
        self.sort_opts['store'] = self.key
        index.sort(**self.sort_opts)

    def new_chunk(self, start, stop):
        p = self.db.pipeline()

        for pk in self.db.lrange(self.key, start, stop):
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
        self.sort = field
        self.sort_opts = opts

class ZQuery(Query):
    def count(self):
        if not self.llen:
            self.do_query()
            self.llen = self.db.zcard(self.key)

        return self.llen

    def do_sort(self, index):
        self.db.zinterstore(self.key, [self.sort])

    def new_chunk(self, start, stop):
        p = self.db.pipeline()

        for pk in self.db.zrange(self.key, start, stop):
            p.hgetall(self.model.key_for('obj', pk=pk))

        return p.execute()

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

        if opts.pop('zindex', None) and field in self.model._zindices:
            c = self._clone(zclone=True)
            field = self.model.key_for('zindex', field=field)
        elif field in self.model._fields:
            c = self._clone()
            field = self.model.key_for('obj', pk='*->%s' % field)
            opts['by'] = field
        else:
            raise ValidationError('currently only sortable by fields on this model')

        c.query.add_sorting(field, opts)
        return c

    def iterator(self):
        self._resolve_order()

        for data in self.query.result_iter():
            yield self.model().from_dict(data, to_python=True)

    def __len__(self):
        self._resolve_order()
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

    def _resolve_order(self):
        if len(self.query.sort) > 0:
            return

        self.query = self.query._clone()
        self.query.add_sorting(self.model.key_for('obj', pk='*->pk'), {})

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
        attrs['pk'] = IntegerField(nil=True)
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
            'zindex': cls._namespace + '_zindex:{field}'}

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
        # update other indexes
        # TODO: make sure we delete indices not more in use.
        for k in self._indices:
            p.sadd(self.key_for('index', field=k, value=data[k]), data['pk'])

        for k in self._zindices:
            r.zadd(self.key_for('zindex', field=k), data[k], data['pk'])

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

#class SortedSet(Collection):
#    def __getitem__(self, s):
#        if isinstance(s, slice):
#            start = s.start or 0
#            stop = s.stop or -1
#            stop = stop - 1
#            return self.zrange(start, stop)
#        else:
#            return self.zrange(s, s)[0]
#
#    def __len__(self):
#        """``x.__len__() <==> len(x)``"""
#        return self.zcard(self.key)
#
#    def __iter__(self):
#        return self.members.__iter__()
#
#    def __reversed__(self):
#        return self.zrevrange(0, -1).__iter__()
#
#    def __repr__(self):
#        return "<%s '%s' %s>" % (self.__class__.__name__, self.key, self.members)
#
#    @property
#    def members(self):
#        return self.zrange(0, -1)
#
#    METHODS = ('zadd', 'zcard', 'zcount', 'zincrby', 'zinterstore', 'zrange',
#        'zrangebyscore', 'zrank', 'zrem', 'zremrangebyrank',
#        'zremrangebyscore', 'zrevrange', 'zrevrangebyscore', 'zrevrank',
#        'zscore', 'zunionstore')