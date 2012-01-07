# -*- coding: utf-8 -*-
import unittest
import datetime
import sys
import base64
import os

from odis.utils import s
from odis import (Model, r, Set, Index, IntegerField, QueryKey, EmptyError, FieldError,
    Field, DateTimeField, SetField, SortedSetField, RelField)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'odisconfig.py'))

class Foo(Model):
    username = Field(index=True, unique=True)
    active = IntegerField(index=True, default=1)
    created_at = DateTimeField()

class Bar(Model):
    username = Field(index=True, unique=True)
    created_at = DateTimeField(auto_now_add=True, zindex=True)

class Baz(Model):
    username = Field(unique=True)

class Qux(Model):
    sets = SetField(Baz)
    sets_float  = SetField(callback=float)
    sortedsets  = SortedSetField(Baz)
    rel  = RelField(Baz)

class ModelsTestCase(unittest.TestCase):
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

class FieldTestCase(unittest.TestCase):
    def setUp(self):
        r.flushdb()
        self.users = []

        for name in ['foo', 'bar', 'baz', 'qux']:
            b = Baz(username=name)
            b.save()
            self.users.append(b)

    @unittest.skip('known datetime bug')
    def test_datetimefield(self):
        b1 = Bar(
            username='foo',
            created_at=datetime.datetime.fromtimestamp(1325161824.91981))
        b2 = Bar(
            username='bar',
            created_at=datetime.datetime(2011, 12, 29, 13, 30, 24, 91981))

        b2.save()
        b1.save()
        self.assertEquals(b1.created_at, b2.created_at)
        self.assertEquals(Bar.obj.get(pk=b1.pk).created_at, Bar.obj.get(pk=b2.pk).created_at)

    def test_setfield(self):
        q = Qux()
        q.save()
        q.sets.sadd(*[u.pk for u in self.users[:2]])
        self.assertEqual(list(q.sets), self.users[:2])

    def test_sortedsetfield(self):
        q = Qux()
        q.save()
        q.sortedsets.zadd(1.0, self.users[0].pk)
        q.sortedsets.zadd(0.2, self.users[1].pk)
        self.assertEqual(list(q.sortedsets), [self.users[1], self.users[0]])

    def test_relfield(self):
        q = Qux()
        q.save()
        q.rel.add(*[(u.pk, u) for u in self.users[:2]])
        self.assertEqual(list(q.rel), self.users[:2])

    def test_setswithtype(self):
        q = Qux()
        q.save()
        values = [i * 1.0 for i in range(10)]
        q.sets_float.sadd(*values)
        res = set(q.sets_float)

        for v in values:
            self.assertEqual(v in res, True)

class QueryTestCase(unittest.TestCase):
    def setUp(self):
        r.flushdb()
        self.b1 = Bar(username='foo')
        self.b2 = Bar(username='bar')
        self.b3 = Bar(username='baz')

        self.b1.save()
        self.b2.save()
        self.b3.save()

    def runtests(self, tests):
        for a, b, msg in tests:
            self.assertEqual(a, b, msg=msg)

    def test_key_parser(self):
        qk = QueryKey(Foo)
        self.runtests([
            (qk.parse_key('~S'),       [['S'], [],        [],        []],          'all'),
            (qk.parse_key('~S^S:a'),   [['S'], [],        [],        ['S:a']],     'all+sort'),
            (qk.parse_key('~S-S:d:0'), [['S'], [],        ['S:d:0'], []],          'all+diff'),
            (qk.parse_key('~S+S:a:1'), [['S'], ['S:a:1'], [],        []],          'all+inter'),
            (qk.parse_key('~S+S:a:1+S:b:2-S:d:0^S:a'), [['S'], ['S:a:1', 'S:b:2'], ['S:d:0'], ['S:a']], 'all+inter+inter+diff+sort')
        ])

    def test_key_builder(self):
        qk = QueryKey(Foo)
        self.runtests([
            (qk.build_key({}, {}, ''), '~Foo_all',  'all'),
            (qk.build_key({'username': 'foo'}, {}, ''), '~Foo_all+Foo_index:username:foo',  'all+inter'),
            (qk.build_key({}, {'username': 'foo'}, ''), '~Foo_all-Foo_index:username:foo',  'all+diff'),
            (qk.build_key({}, {}, 'Foo:username'), '~Foo_all^Foo:username',  'all+sort'),
        ])

    def test_key_matcher(self):
        qk = QueryKey(Bar)

        tests = [
            ('~Bar_all', self.b1, True),
            ('~Foo_all', self.b1, False),
            ('~Bar_all^Bar:username', self.b1, True),
            ('~Bar_all+Bar_index:username:foo', self.b1, True),
            ('~Bar_all+Bar_index:username:foo+Bar:pk:1', self.b1, True),
            ('~Bar_all+Bar_index:username:foo-Bar:pk:2', self.b1, True),
            ('~Bar_all+Bar_index:username:foo+Bar:pk:2', self.b1, False),
            ('~Bar_all+Bar_index:username:foo+Bar:pk:1^Bar:created_at', self.b1, True),
        ]

        for key, obj, expected in tests:
            ok, parts = qk.match_and_parse_key(key, obj.as_dict(to_db=True))
            self.assertEqual(ok, expected)

    def test_cache(self):
        qs = Bar.obj.filter(username='foo')
        self.assertEqual(qs._cache, None)
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs.query._hits, 1)
        self.assertEqual(len(list(qs)), 1)
        self.assertEqual(len(qs._cache), 1)
        self.assertEqual(qs.query._hits, 1)

        qs = qs.filter(username='bar')
        self.assertEqual(qs._cache, None)
        self.assertEqual(qs.query._hits, 1)
        self.assertEqual(len(list(qs)), 1)
        self.assertEqual(len(qs._cache), 1)
        self.assertEqual(qs.query._hits, 2)

        self.assertEqual(len(list(qs)), 1)
        qs = qs.filter(username='bar')
        self.assertEqual(qs._cache, None)
        self.assertEqual(len(list(qs)), 1)
        self.assertEqual(qs.query._hits, 2)

        obj = qs[0]
        obj.username = 'qux'
        obj.save()

        qs = qs.filter(username='bar')
        self.assertEqual(qs._cache, None)
        self.assertEqual(len(list(qs)), 0)
        self.assertEqual(qs.query._hits, 2)

    def test_getitem(self):
        for i, obj in enumerate(Bar.obj.filter()[1:3]):
            self.assertEqual(obj.pk, i + 2)

        self.assertEqual(Bar.obj.filter()[1:3].count(), 2)
        self.assertEqual(Bar.obj.filter()[0], self.b1)

    def test_get(self):
        self.runtests([
            (Bar.obj.get(pk=1), self.b1, 'pk=1'),
            (Bar.obj.get(username='bar'), self.b2, 'filter all, exclude username'),
        ])

        self.assertRaises(AssertionError, Bar.obj.get, pk=1, username='bar')
        self.assertRaises(EmptyError, Bar.obj.get, pk=4)
        self.assertRaises(EmptyError, Bar.obj.get, username='qux')
        self.assertRaises(FieldError, Bar.obj.get, nil=4)

    def test_filter(self):
        self.runtests([
            (list(Bar.obj.filter(username='foo')), [self.b1], 'username=foo'),
            (list(Bar.obj.filter().exclude(username='bar')), [self.b1, self.b3], 'filter all, exclude username'),
            (list(Bar.obj.filter(username='bar').exclude(username='foo')), [self.b2], 'filter all, exclude username'),
        ])

    def test_sorting(self):
        self.runtests([
            (list(Bar.obj.filter()), [self.b1, self.b2, self.b3], 'SORT by Bar:*->pk'),
            (list(Bar.obj.order('pk')), [self.b1, self.b2, self.b3], 'SORT by Bar:*->pk'),
            (list(Bar.obj.order('created_at')), [self.b1, self.b2, self.b3], 'ZINTERSTORE Bar_zindex:created_at'),
            (list(Bar.obj.order('-created_at')), [self.b3, self.b2, self.b1], 'ZINTERSTORE Bar_zindex:created_at'),
            #(list(Bar.obj.order('username', alpha=True)), [self.b2, self.b3, self.b1], 'SORT by Bar:*->username'),
        ])

    @unittest.skip('not implemented yet')
    def test_advanced_filter(self):
        self.runtests([
            (list(Bar.obj.filter(username__in=['foo', 'bar'])), [self.b1, self.b2], 'username__in=[foo, bar]'),
        ])

class TypeTestCase(unittest.TestCase):
    def setUp(self):
        r.flushdb()

    def test_index(self):
        for name in ['foo', 'bar', 'baz']:
            Bar(username=name).save()

        index = Index(Bar, Bar.key_for('all'))
        self.assertEquals(list(index.inter(username='foo')), ['1'])

    def test_set(self):
        for name in ['foo', 'bar', 'baz']:
            Foo(username=name, created_at=datetime.datetime.now()).save()

        s = Set(Foo, Foo.key_for('all'))

        for i in range(1, 3):
            self.assertEqual(unicode(i) in s, True)

class ScoreTestCase(unittest.TestCase):
    def setUp(self):
        match_replace = {
            u'Æ': u'[',
            u'Ø': u'\\',
            u'Å': u']',
            u'æ': u'{',
            u'ø': u'|',
            u'å': u'}'}

        self.rules = [self.build_pattern(pattern, replace) for pattern, replace in match_replace.items()]

    def str_dec(self, s):
        return ''.join(['%s' % ord(c) for c in s])

    def base64_dec(self, s):
        return self.str_dec(base64.b64encode(s))

    def build_pattern(self, pattern, replace):
        import re

        def matches_rule(word):
            return re.search(pattern, word, flags=re.UNICODE)

        def apply_rule(word):
            return re.sub(pattern, replace, word, flags=re.UNICODE)

        return (matches_rule, apply_rule)

    def strip(self, word):
        for m, a in self.rules:
            if m(word):
                return a(word)

    def score(self, word):
        try:

            res = ord(word[0]) * (256^3)
            res = res + ord(word[1]) * (256^2)
            res = res + ord(word[2]) * (256^1)
            res = res + ord(word[3])

        except IndexError:
            pass

        return res

    def test_trans(self):
        tests = [
            (u'å', u'}', True, u'å'),
            (u'Å', u']', True, u'Å'),
        ]

        for a, b, expected, msg in tests:
            self.assertEqual(self.strip(a) == b, expected, msg=msg)

    def test_equality(self):
        tests = [
            self.score(u'A') < self.score(u'AAA'),
            self.score(u'Z') < self.score(u'{'),
            self.score(u'Z') < self.score(u'{A'),
        ]

        for a in tests:
            self.assertEquals(a, True)
