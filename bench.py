from tests.tests import Foo
from odis import Index
import time

i = Index(Foo, Foo.key_for('all'))
db = i.db
CHUNK_SIZE = 100

db.delete('foo')
db.delete('bar')

p = db.pipeline()

for i in range(10000):
    p.sadd('foo', i)
    p.zadd('bar', i, i)

p.execute()

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
        self.c = ZChunked()
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

start2 = time.time()
q = Query()

for v in q:
    if int(v) >= 1000:
        break

print time.time() - start2

start2 = time.time()

for v in q:
    if int(v) >= 1000:
        break

print time.time() - start2

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
