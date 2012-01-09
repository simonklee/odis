import random

from odis import Model, Field, RelField, SortedSetField

from odis.utils import s

class Foo(Model):
    username = Field(index=True)

class Bar(Model):
    foos = RelField(Foo)
    col = SortedSetField(Foo)

db = Foo._db
db.flushdb()
usernames = ['foo', 'bar', 'baz', 'qux', 'foobar', 'foobaz', 'fooqux', 'barfoo', 'barbar']

for u in range(10):
    Foo(username=random.choice(usernames), active=random.randint(0, 1)).save()

s()
obj = Bar()
obj.save()
f1 = Foo.obj.get(pk=1)
f2 = Foo.obj.get(pk=2)
f3 = Foo.obj.get(pk=3)
obj.foos.add(f1)
obj.foos.add(f2)
print list(obj.foos.desc())
f4 = Foo(username=random.choice(usernames), active=random.randint(0, 1))
f4.save()


#obj.foos.add((1.1, f3))
#qs = obj.foos
#print list(qs)
#qs.delete(f3)
#print list(qs)

obj.col.zadd(1.0, 1)
obj.col.zadd(1.2, 2)

print obj.col[1:]
