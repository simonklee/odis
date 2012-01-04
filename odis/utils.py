import sys

from IPython.core.debugger import Pdb, BdbQuit_excepthook

def s():
    BdbQuit_excepthook.excepthook_ori = sys.excepthook
    Pdb('LightBG').set_trace(sys._getframe().f_back)

