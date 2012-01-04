import random
from odis import Model, Field, DateTimeField, IntegerField
from odis.utils import s

class Foo(Model):
    username = Field(index=True, unique=True)
    active = IntegerField(index=True, default=1)
    created_at = DateTimeField(zindex=True, auto_now_add=True)

db = Foo._db
db.flushdb()
usernames = ['foo', 'bar', 'baz', 'qux', 'foobar', 'foobaz', 'fooqux', 'barfoo', 'barbar']

for u in range(100):
    Foo(username=random.choice(usernames), active=random.randint(0, 1)).save()

qs = Foo.obj.order('-created_at')
s()
