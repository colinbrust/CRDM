import os

def parse_premade(f):
    f = os.path.basename(f).replace('.dat', '').split('_')

    return dict(zip(['type', 'class', 'nMonths', 'leadTime', 'size', 'rmFeatures'], 
           [x.split('-')[-1] for x in f]))
    
