import random
import time

from tests.tests import Foo
from odis import Index

i = Index(Foo, Foo.key_for('all'))
db = i.db
CHUNK_SIZE = 100

class Chunked(object):
    def __init__(self):
        self.cached = None
        self.llen = None
        self.index = 0

    def count(self):
        if not self.llen:
            self.do_sort()
            self.llen = db.llen(self.cached)

        return self.llen

    def do_sort(self):
        if not self.cached:
            self.cached = 'cached'
            db.sort('foo', store=self.cached)

    def new_chunk(self, start, stop):
        return db.lrange(self.cached, start, stop)

    def fetchvalues(self):
        self.do_sort()

        while self.index < self.count():
            val = self.new_chunk(self.index, self.index + CHUNK_SIZE)
            self.index = self.index + CHUNK_SIZE
            return val

        return []

    def result_iter(self):
        for chunks in iter(self.fetchvalues, []):
            for pk in chunks:
                yield pk

class ZChunked(Chunked):
    #def count(self):
    #    if not self.llen:
    #        self.llen = db.llen(self.cached)

    #    return self.llen

    def do_sort(self):
        if not self.cached:
            self.cached = 'zcached'
            db.sort('bar', store=self.cached)

    #def new_chunk(self, start, stop):
    #    return db.zrange(self.cached, start, stop)

class Query(object):
    def __init__(self):
        self.c = Chunked()
        self._cache = []
        self._iter = self.c.result_iter()

    def __iter__(self):
        return self._cache_iter()

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


    def __len__(self):
        return self.c.count()


#db.delete('foo')
#db.delete('bar')
#
#p = db.pipeline()
#
#for i in range(10000, 1000000):
#    p.sadd('foo', i)
#    p.zadd('bar', i, i)
#
#p.execute()

#db.flushdb()
#p = db.pipeline()
#
#for i in range(100000):
#    data = {
#        'pk': i,
#        'active': random.randint(0, 1),
#        'username': random.choice(['foo', 'bar', 'qux']) }
#    p.sadd('bar:all', i)
#    p.hmset('bar:%d' % i, data)
#    p.sadd('bar:username:%s' % data['username'], i)
#    p.sadd('bar:type:%s' % random.choice(['V', 'P', 'A']), i)
#    p.zadd('bar:active', data['active'], i)
#
#p.execute()

#db.flushdb()
#p = db.pipeline()
#
#for i in range(10000):
#    data = {
#        'pk': i,
#        'active': random.randint(0, 1),
#        'username': random.choice(['foo', 'bar', 'qux']),
#        'type': random.choice(['V', 'P', 'A'])}
#    p.sadd('baz:all', i)
#    p.hmset('baz:%d' % i, data)
#    p.sadd('baz:username:%s' % data['username'], i)
#    p.sadd('baz:type:%s' % data['type'], i)
#    p.zadd('baz:active', data['active'], i)
#p.execute()
#
#db.flushdb()
#p = db.pipeline()
#
#for i in range(100):
#    data = {
#        'pk': i,
#        'active': random.randint(0, 1),
#        'username': random.choice(['foo', 'bar', 'qux']),
#        'type': random.choice(['V', 'P', 'A'])}
#    p.sadd('foo:all', i)
#    p.hmset('foo:%d' % i, data)
#    p.sadd('foo:username:%s' % data['username'], i)
#    p.sadd('foo:type:%s' % data['type'], i)
#    p.zadd('foo:active', data['active'], i)
#
#p.execute()
total = time.time()

start2 = time.time()
print db.sunionstore('foo+bar', ['baz:username:qux', 'baz:type:P'])
print 'foo+bar %.3fms' % (time.time() - start2)

#start2 = time.time()
#print db.zinterstore('foo+bar+active', ['foo+bar', 'foo:active'])
#print 'foo+bar+active %.3fms' % (time.time() - start2)

#print db.zinterstore('qux', {'foo+bar+active':0, 'bar:active': 1})
#print db.zinterstore('qux', {'foo+bar+active':0, 'bar:active': 1})
start2 = time.time()
db.sort('foo+bar', start=0, num=200, by='baz:*->active', store='qux')
print 'sort %.3fms' % (time.time() - start2)

start2 = time.time()
print db.lrange('qux',  0, 200)
#print db.zrange('foo+bar+active',  0, 20)
print '%.3fms' % (time.time() - start2)

print 'total %.3fms' % (time.time() - total)

#start2 = time.time()
#q = Query()
#
#for v in q:
#    if int(v) >= 1000:
#        break
#
#print time.time() - start2
#
#start2 = time.time()
#
#for v in q:
#    if int(v) >= 1000:
#        break
#
#print time.time() - start2

#start2 = time.time()
#c = Chuncked()
#it = iter(c.fetchvalues, [])
#list(it)[:100]
#print time.time() - start2
#start2 = time.time()
#db.sort('foo', store='cc')
#llen = db.llen('cc')
#val = db.lrange('cc', 0, llen)
#print time.time() - start2
