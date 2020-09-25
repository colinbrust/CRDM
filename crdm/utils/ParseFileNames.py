import os

def parse_fname(f):
    f = os.path.basename(f).replace('.dat', '').split('_')
    f = [x.split('-') for x in f]

    return dict(zip([x[0] for x in f], [x[-1] for x in f]))
    
