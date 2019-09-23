""" Trash script for non-official unit test of the modules.
You can use it has you want but keep it as simple as possible.

"""
import zeta

lol=None

for i in range(5):
    try:
        if i==2:
            assert False
        elif i==3:
            lol=zeta.data.datasets.load_sessions_raw("/Volumes/Ext/data/Zeta/", "raw_clean_32", "x")[0]
            print(lol)
        else:
            print(i)
    except AssertionError:
        print("unit test went wrong %i"%(i))
    except:
        print("something went wrong here %i"%(i))
