import random
import time

from odis import Model, CharField, RelField, IntegerField

from odis.utils import s

class Foo(Model):
    username = CharField(index=True, unique=True)
    active = IntegerField(index=True)

class Bar(Model):
    users = RelField(Foo)

db = Foo._db
db.flushdb()

for u in ['foo', 'bar', 'baz', 'qux']:
    Foo(username=u, active=random.randint(0, 1)).save()

obj = Bar()
obj.save()

print obj.as_dict()
print obj.as_dict(to_db=True)
total = time.time()
qs = Bar.obj.all()
#qs = Bar.obj
obj = qs[0]
print 'total %.3fms' % (time.time() - total)
s()
