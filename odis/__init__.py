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

class EMPTY:
    pass

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
        return (instance.pk and r.sismember(key, instance.pk) or r.scard(key) == 0)

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
            pk = r.get(self._cls.key_for('lookup', field=key, key=value))

        data = r.hgetall(self._cls.key_for('obj', pk=pk))

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
            self.pk = r.incr(self.key_for('pk'))

        self.write()
        return True

    def write(self):
        data = self.as_dict(to_db=True)
        p = r.pipeline()
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
        return self.smembers().__iter__()

    def __contains__(self, value):
        return self.sismember(value)

    def __repr__(self):
        return "<%s '%s' %s(%s)>" % (self.__class__.__name__, self.model.__name__, self.key, self.smembers())

    def exclude(self, **kwargs):
        return self._inter(kwargs, diff=True)

    def filter(self, **kwargs):
        return self._inter(kwargs)

    def _inter(self, options, expire=60, diff=False):
        # seperate input
        keys = [self.key] + self._keys(options)
        target = '~' + '+'.join(keys)

        # then do the intersection
        if diff:
            self.db.sdiffstore(target, *keys)
        else:
            self.db.sinterstore(target, *keys)

        # key is just temporary
        self.db.expire(target, expire)

        # return the result as a new Set
        return Set(self.model, target, db=self.db)

    def _keys(self, data):
        return [self.model.key_for('index', field=k, value=v) for k, v in data.items()]

    METHODS = ('sadd', 'scard', 'sdiff', 'sdiffstore', 'sinter', 'sinterstore', 'sismember',
        'smembers', 'smove', 'spop', 'srandmember', 'srem', 'sunion', 'sunionstore')

class Index(Set):
    def filter(self, **kwargs):
        keys = self._keys(kwargs)

        if len(keys) == 0:
            return self
        elif len(keys) > 1:
            return super(Index, self).find(**kwargs)
        else:
            return Set(self.model, keys[0], db=self.db)

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
