import pickle

# File that reads pickle so that it can be loaded into R with reticulate and plotted.
def read_pickle(f: str):
    """
    :param f: Path to pickled file.
    :return: Python object that was pickled.
    """
    with open(f, 'rb') as dat:
        out = pickle.load(dat)

    return out
