from __future__ import absolute_import

import redis
import time
import datetime
import functools

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

class ValidationError(Exception):
    'An error raised on validation'

class FieldError(Exception):
    'An error raised on validation'

class EmptyError(Exception):
    'An error raised on validation'

class Field(object):
    def __init__(self, index=False, unique=False, nil=False, default=None):
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
        if hasattr(instance, '_' + self.name):
            return getattr(instance, '_' + self.name)
        return None

    def validate(self, value, model_instance):
        if not self.nil and self.is_empty(value):
            raise ValidationError('`%s` nil value' % self.name)

    def is_empty(self, value):
        return value in EMPTY_VALUES

    def to_python(self, value):
        return value

    def to_db(self, value):
        return value

class IntegerField(Field):
    def to_python(self, value):
        return int(value)

    def to_db(self, value):
        return unicode(value)

class DateTimeField(Field):
    def to_python(self, value):
        if not isinstance(value, datetime.datetime):
            return datetime.datetime.fromtimestamp(float(value))
        return value

    def to_db(self, value):
        return u'%d.%d' % (time.mktime(value.timetuple()), value.microsecond)

class DateField(Field):
    def to_python(self, value):
        if not isinstance(value, datetime.date):
            return datetime.date.fromtimestamp(float(value))
        return value

    def to_db(self, value):
        return u'%f' % time.mktime(value.timetuple())

class Query(object):
    def __init__(self, cls):
        self.cls = cls

class Manager(object):
    def __init__(self, cls):
        self._cls = cls

    def get(self, **kwargs):
        key, value = kwargs.popitem()

        if key == 'pk':
            pk = value
        else:
            if not key in self._cls._lookup:
                raise FieldError('`%s` is not a valid lookup field. fields \
                        are `%`' % (key, self._cls._lookup))
            pk = r.get(self._cls.key('lookup', field=key, key=value))

        data = r.hgetall(self._cls.key('obj', pk=pk))

        if not data:
            raise EmptyError('`%s(%s=%s)` returned an empty result' %
                    (self._cls.__name__, key, pk or value))

        return self._cls().from_dict(data, to_python=True)

class BaseModel(type):
    def __new__(meta, name, bases, attrs):
        attrs['pk'] = IntegerField(nil=True)
        cls = super(BaseModel, meta).__new__(meta, name, bases, attrs)
        cls._fields = {}
        cls._indices = []

        if config.REDIS_PREFIX:
            cls._namespace = config.REDIS_PREFIX + '_' + name
        else:
            cls._namespace = name

        cls._keys = {
            'pk': cls._namespace + '_pk',
            'all' : cls._namespace + '_all',
            'obj' : cls._namespace + ':{pk}',
            'index': cls._namespace + '_index:{field}:{value}'}

        for k, v in attrs.iteritems():
            if isinstance(v, Field):
                cls._fields[k] = v
                v.name = k

                if v.index:
                    cls._indices.append(k)

        cls.obj = Manager(cls)
        return cls

class Model(object):
    __metaclass__ = BaseModel

    def __init__(self, **kwargs):
        self.from_dict(kwargs)

    def __repr__(self):
        return u'<%s %s>' % (self.__class__.__name__, self.as_dict())

    @classmethod
    def key(cls, name, **kwargs):
        return cls._keys[name].format(**kwargs)

    def is_valid(self):
        try:
            self.clean()
            self.validate()
            self.validate_unique()
        except ValidationError:
            return False
        return True

    def clean(self):
        '''Calls to_python on all fields which do not contain empty values
        and have `field.nil` set to True'''
        for name, field in self._fields.items():
            raw = getattr(self, name)

            if field.nil and raw in EMPTY_VALUES:
                continue

            setattr(self, name, field.to_python(raw))

    def validate(self):
        '''Calls validate() on all fields'''
        self._errors = {}

        for name, field in self._fields.items():
            try:
                field.validate(getattr(self, name), self)
            except ValidationError, e:
                self._errors[name] = e.message

        if self._errors:
            raise ValidationError(self._errors)

    def validate_unique(self):
        '''Check uniqueness on all fields with `unique=True`'''
        self._errors = getattr(self, '_errors', {})
        data = self.as_dict()

        for name, field in self._fields.items():
            if not field.unique or attr in self._errors:
                continue

            key = self.key('index', field=name, value=data[name])

            if (self.pk and r.sismember(key, self.pk) or r.scard(key) > 0):
                self._errors[name] = ValidationError('%s `%s` not unique' % (name, data[name])).message

        if self._errors:
            raise ValidationError(self._errors)

    def save(self):
        if not self.is_valid():
            return False

        if self.pk is None:
            self.pk = r.incr(self.key('pk'))

        data = self.as_dict(to_db=True)
        p = r.pipeline()
        p.hmset(self.key('obj', pk=self.pk), data)
        p.sadd(self.key('all'), self.pk)

        for k in self._indices:
            p.sadd(self.key('index', field=k, value=data[k]), data['pk'])

        p.execute()
        return True

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

class Collection(object):
    '''Create a collection object saved in Redis.
    `key` the redis key for collection.
    `db` or `pipe` must be provided'''
    def __init__(self, model, key, db=None, pipe=None):
        self.model = model
        self.db = pipe or db

        if not self.db:
            raise Exception('No connection specified')

        self.key = key

        for attr in self.METHODS:
            setattr(self, attr, functools.partial(getattr(self.db, attr), self.key))

    def clear(self):
        self.db.delete(self.key)

    METHODS = ()

class Set(Collection):
    def __len__(self):
        """``x.__len__() <==> len(x)``"""
        return self.scard()

    def __iter__(self):
        return self.smembers().__iter__()

    def __contains__(self, value):
        return self.sismember(value)

    def __repr__(self):
        return "<%s %s('%s' %s)>" % (self.__class__.__name__, self.model.__name__, self.key, self.smembers())

    def find(self, **kwargs):
        # seperate input
        keys = self.keys(kwargs)
        target = '~' + '+'.join(keys)

        # then do a sinterstore
        self.db.sinterstore(target, *keys)
        self.db.expire(target, 60)

        # return the result as a new Set
        return Set(self.model, target, db=self.db)

    def keys(self, data):
        return [self.model.key('index', field=k, value=v) for k, v in data.items()]

    METHODS = ('sadd', 'scard', 'sdiff', 'sdiffstore', 'sinter', 'sinterstore', 'sismember',
        'smembers', 'smove', 'spop', 'srandmember', 'srem', 'sunion', 'sunionstore')

class Index(Set):
    def find(self, **kwargs):
        pass

class SortedSet(Collection):
    def __getitem__(self, s):
        if isinstance(s, slice):
            start = s.start or 0
            stop = s.stop or -1
            stop = stop - 1
            return self.zrange(start, stop)
        else:
            return self.zrange(s, s)[0]

    def __len__(self):
        """``x.__len__() <==> len(x)``"""
        return self.zcard(self.key)

    def __iter__(self):
        return self.members.__iter__()

    def __reversed__(self):
        return self.zrevrange(0, -1).__iter__()

    def __repr__(self):
        return "<%s '%s' %s>" % (self.__class__.__name__, self.key, self.members)

    @property
    def members(self):
        return self.zrange(0, -1)

    METHODS = ('zadd', 'zcard', 'zcount', 'zincrby', 'zinterstore', 'zrange',
        'zrangebyscore', 'zrank', 'zrem', 'zremrangebyrank',
        'zremrangebyscore', 'zrevrange', 'zrevrangebyscore', 'zrevrank',
        'zscore', 'zunionstore')
