# odis

odis - objects in [Redis](http://redis.io) for Python.

## Install odis

    $ pip install git+git://github.com/simonz05/odis.git

## Use odis

    from odis import Model, CharField, DateTimeField

    class Foo(Model):
        username = CharField(index=True, unique=True)
        created_at = DateTimeField(aut_now_add=True)


    >>> obj = Foo(username='foo')
    >>> obj.save()
    True

    >>> Foo.obj.get(username='foo')
    <Foo {'username': 'foo', 'pk': 1, 'created_at': datetime.datetime(2012, 1, 14, 0, 0)}>

    >>> Foo.obj.filter(username='foo')
    <odis.QuerySet at 0x24e50d0>

## Acknowledgment

[Ohm](http://ohm.keyvalue.org/), [redisco](https://github.com/iamteem/redisco/) and
[Django internals](https://github.com/django/django/tree/master/django/db/models)
were used to figure out how to solve similar problems.
