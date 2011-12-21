import unittest
import datetime

from models import Model, Field, DateTimeField, r, SortedSet, EmptyError

class Foo(Model):
    username = Field(index=True, lookup=True)
    created_at = DateTimeField(index=True)

class ModelsTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dbkeys(self):
        f = Foo(username='foo', created_at=datetime.datetime.now())
        pre = f._options['prefix'] + '_' + f._options['name']

        self.assertEquals(f.key('pk'), pre + '_pk')
        self.assertEquals(f.key('obj', pk=1), pre + ':1')
        self.assertEquals(f.key('lookup', field='username', key=f.username), pre + '_lookup:username:' + f.username)
        self.assertEquals(f.key('index', field='created_at'), pre + '_index:created_at')

    def test_is_valid(self):
        r.flushdb()
        f = Foo(username='foo', created_at=datetime.datetime.now())
        self.assertEquals(f.is_valid(), True)
        self.assertEquals(f.save(), True)

        f = Foo(username='foo', created_at=datetime.datetime.now())
        self.assertEquals(f.is_valid(), False)
        self.assertEquals(f.save(), False)

        f = Foo(username='bar', created_at=datetime.datetime.now())
        self.assertEquals(f.is_valid(), True)
        self.assertEquals(f.save(), True)

    def test_pk(self):
        r.flushdb()

        a = Foo(username='foo', created_at=datetime.datetime.now())
        b = Foo(username='bar', created_at=datetime.datetime.now())
        c = Foo(username='bar', created_at=datetime.datetime.now())
        d = Foo(username='baz', created_at=datetime.datetime.now())

        self.assertEquals(a.save(), True)
        self.assertEquals(b.save(), True)
        self.assertEquals(c.save(), False)
        self.assertEquals(d.save(), True)
        self.assertEquals([a.pk, b.pk, d.pk], [1, 2, 3])

    def test_pack(self):
        f = Foo()
        data = {'username': 'foo', 'created_at': 1}
        f.from_dict(data)
        data['pk'] = None
        self.assertEquals(f.as_dict(), data)

class ManagerTestCase(unittest.TestCase):
    def setUp(self):
        r.flushdb()

        for name in ['foo', 'bar', 'baz']:
            Foo(username=name, created_at=datetime.datetime.now()).save()

    def tearDown(self):
        pass

    def test_get(self):
        f = Foo.obj.get(pk=1)
        self.assertEquals(f.pk, 1)
        self.assertEquals(f.username, 'foo')

        f = Foo.obj.get(username='foo')
        self.assertEquals(f.pk, 1)
        self.assertEquals(f.username, 'foo')

        self.assertRaises(EmptyError, Foo.obj.get, pk=4)
        self.assertRaises(EmptyError, Foo.obj.get, username='qux')

class TypeTestCase(unittest.TestCase):
    def setUp(self):
        r.flushdb()

        for name in ['foo', 'bar', 'baz']:
            Foo(username=name, created_at=datetime.datetime.now()).save()

    def tearDown(self):
        pass

    def test_sortedset(self):
        ss = SortedSet(Foo.key('index', field='pk'), db=r)

        for i, k in enumerate(ss):
            self.assertEquals(int(k), i+1)

def main():
    """Runs the testsuite as command line application."""
    suite1 = unittest.TestLoader().loadTestsFromTestCase(ModelsTestCase)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TypeTestCase)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(ManagerTestCase)
    suite = unittest.TestSuite([suite1, suite2, suite3])
    unittest.TextTestRunner(verbosity=3).run(suite)
