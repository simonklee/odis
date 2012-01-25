import sys
import time

try:
    from IPython.core.debugger import Pdb, BdbQuit_excepthook
except ImportError:
    pass

def s():
    BdbQuit_excepthook.excepthook_ori = sys.excepthook
    Pdb('LightBG').set_trace(sys._getframe().f_back)

def timeit(method):
    def wrapper(*args, **kw):
        t = time.time()
        result = method(*args, **kw)
        print '%r (%r, %r) %2.3f sec' % \
              (method.__name__, args, kw, time.time() - t)
        return result
    return wrapper

def safe_unicode(s):
    if isinstance(s, unicode):
        return s

    if not isinstance(s, basestring):
        s = unicode(str(s))

    return s.decode('utf-8')

def safe_bytestr(s):
    try:
        return str(s)
    except UnicodeEncodeError:
        return unicode(s).encode('utf-8')
