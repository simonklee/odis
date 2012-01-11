import random
import time

from odis import Model, Field, ForeignField

from odis.utils import s

class Foo(Model):
    username = Field(index=True)

class Bar(Model):
    foo = ForeignField(Foo, nil=True)
    foob = ForeignField(Foo)
    fooc = ForeignField(Foo)

db = Foo._db
db.flushdb()
usernames = ['foo', 'bar', 'baz', 'qux', 'foobar', 'foobaz', 'fooqux', 'barfoo', 'barbar']

for u in range(10):
    Foo(username=random.choice(usernames), active=random.randint(0, 1)).save()

for i in range(1000):
    Bar(foob=random.randint(1, 10), fooc=random.randint(1, 10)).save()

total = time.time()
qs = Bar.obj.include('foo')
#qs = Bar.obj
print list(qs[:2])
print 'total %.3fms' % (time.time() - total)
