import pickle

# File that reads pickle so that it can be loaded into R with reticulate. 
def read_pickle(f):
    with open(f, 'rb') as dat:
        out = pickle.load(dat)

    return out
