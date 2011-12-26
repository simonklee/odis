import unittest
import datetime
import sys
import os

from odis import Model, Field, DateTimeField, r, Set, Index, IntegerField

from redisco import models as remod

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'odisconfig.py'))

class Foo(Model):
    username = Field(index=True, unique=True)
    active = IntegerField(index=True, default=1)
    created_at = DateTimeField()

class Bar(remod.Model):
    username = remod.Attribute(required=True)
    created_at = remod.DateTimeField(auto_now_add=True)

class ModelsTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    #def test_compare(self):
    #    r.flushdb()
    #    for i in range(0, 100):
    #        bb = Bar(username='%s' % i)
    #        bb.save()

    #    for i in range(0, 100):
    #        ff = Foo(username='%s' % i, created_at=datetime.datetime.now())
    #        ff.save()

    #    print 'a'

    def test_dbkeys(self):
        f = Foo(username='foo', created_at=datetime.datetime.now())
        self.assertEqual(f.key_for('pk'), f._namespace + '_pk')
        self.assertEqual(f.key_for('obj', pk=1), f._namespace + ':1')
        self.assertEqual(f.key_for('index', field='username', value=f.username), f._namespace + '_index:username:' + f.username)

    def test_is_valid(self):
        r.flushdb()
        f = Foo(username='foo', created_at=datetime.datetime.now())
        self.assertEqual(f.is_valid(), True)
        self.assertEqual(f.save(), True)

        f = Foo(username='foo', created_at=datetime.datetime.now())
        self.assertEqual(f.is_valid(), False)
        self.assertEqual(len(f._errors), 1)
        self.assertEqual(f.save(), False)

        f = Foo(username='bar', created_at=datetime.datetime.now())
        self.assertEqual(f.is_valid(), True)
        self.assertEqual(f.save(), True)

    def test_cmp(self):
        tests = [
            (Foo(username='foo', pk=1), Foo(username='foo', pk=1), True, 'equal'),
            (Foo(username='bar', pk=1), Foo(username='foo', pk=1), False, 'unequal name'),
            (Foo(username='foo', pk=2), Foo(username='foo', pk=1), False, 'unequal pk'),
        ]

        for a, b, expected, msg in tests:
            self.assertEqual(a == b, expected, msg=msg)

    def test_pk(self):
        r.flushdb()

        a = Foo(username='foo', created_at=datetime.datetime.now())
        b = Foo(username='bar', created_at=datetime.datetime.now())
        c = Foo(username='bar', created_at=datetime.datetime.now())
        d = Foo(username='baz', created_at=datetime.datetime.now())

        self.assertEqual(a.save(), True)
        self.assertEqual(b.save(), True)
        self.assertEqual(c.save(), False)
        self.assertEqual(d.save(), True)
        self.assertEqual([a.pk, b.pk, d.pk], [1, 2, 3])

    def test_pack(self):
        f = Foo()
        data = {'username': 'foo', 'created_at': 1}
        f.from_dict(data)
        data['pk'] = None
        data['active'] = 1
        got = f.as_dict()
        self.assertEqual(got, data)

#class ManagerTestCase(unittest.TestCase):
#    def setUp(self):
#        r.flushdb()
#
#        for name in ['foo', 'bar', 'baz']:
#            Foo(username=name, created_at=datetime.datetime.now()).save()
#
#    def tearDown(self):
#        pass
#
#    def test_get(self):
#        f = Foo.obj.get(pk=1)
#        self.assertEqual(f.pk, 1)
#        self.assertEqual(f.username, 'foo')
#
#        f = Foo.obj.get(username='foo')
#        self.assertEqual(f.pk, 1)
#        self.assertEqual(f.username, 'foo')
#
#        self.assertRaises(EmptyError, Foo.obj.get, pk=4)
#        self.assertRaises(EmptyError, Foo.obj.get, username='qux')

class TypeTestCase(unittest.TestCase):
    def setUp(self):
        r.flushdb()

        for name in ['foo', 'bar', 'baz']:
            Foo(username=name, created_at=datetime.datetime.now()).save()

    def tearDown(self):
        pass

    def test_index(self):
        index = Index(Foo, Foo.key_for('all'), db=r)

        for i in index:
            print i
            #self.assertEqual(unicode(i) in s, True)

    def test_set(self):
        s = Set(Foo, Foo.key_for('all'), db=r)

        for i in range(1, 3):
            self.assertEqual(unicode(i) in s, True)
