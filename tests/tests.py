import unittest
import datetime
import sys
import os

import ipdb

from odis import Model, Field, DateTimeField, r, Set, SortedSet, EmptyError, IntegerField
from redisco import models as remod

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'odisconfig.py'))

class Foo(Model):
    username = Field(index=True, unique=True)
    active = IntegerField(index=True, default=0)
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
        self.assertEquals(f.key('pk'), f._namespace + '_pk')
        self.assertEquals(f.key('obj', pk=1), f._namespace + ':1')
        self.assertEquals(f.key('index', field='username', value=f.username), f._namespace + '_index:username:' + f.username)

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
        data = {'username': 'foo', 'created_at': 1, 'active': None}
        f.from_dict(data)
        data['pk'] = None
        self.assertEquals(f.as_dict(), data)

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
#        self.assertEquals(f.pk, 1)
#        self.assertEquals(f.username, 'foo')
#
#        f = Foo.obj.get(username='foo')
#        self.assertEquals(f.pk, 1)
#        self.assertEquals(f.username, 'foo')
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

    def test_set(self):
        s = Set(Foo, Foo.key('all'), db=r)

        for i in range(1, 3):
            self.assertEquals(unicode(i) in s, True)
