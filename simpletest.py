import random
import time

from odis import Model, Field, DateTimeField, IntegerField
from odis.utils import s

class Foo(Model):
    username = Field(index=True)
    active = IntegerField(zindex=True, index=True, default=1)
    created_at = DateTimeField(zindex=True, auto_now_add=True)

db = Foo._db
#db.flushdb()
#usernames = ['foo', 'bar', 'baz', 'qux', 'foobar', 'foobaz', 'fooqux', 'barfoo', 'barbar']
#
#for u in range(40000):
#    Foo(username=random.choice(usernames), active=random.randint(0, 1)).save()

total = time.time()
#for username in usernames:
#    list(Foo.obj.filter(username=username))

list(Foo.obj.filter(active=1, username='foo')[:10])
#list(Foo.obj.filter(active=1))
#list(Foo.obj.order('active'))
#list(Foo.obj.order('pk'))

save = time.time()
obj = Foo.obj.get(username='foo')
obj.username = 'simon'
obj.save()
print 'save %.3fms' % (time.time() - save)
print 'total %.3fms' % (time.time() - total)
