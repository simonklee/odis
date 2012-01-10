import sys

from IPython.core.debugger import Pdb, BdbQuit_excepthook

def s():
    BdbQuit_excepthook.excepthook_ori = sys.excepthook
    Pdb('LightBG').set_trace(sys._getframe().f_back)

def safe_unicode(s):
    if isinstance(s, unicode):
        return s

    return s.decode('utf-8')

def safe_bytestr(s):
    try:
        return str(s)
    except UnicodeEncodeError:
        return unicode(s).encode('utf-8')
