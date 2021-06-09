import os


def make_model_dir() -> str:
    """
    :return: Creates a new directory in current working directory named 'modelX' where X is the lowest number such that
    'modelX' doesn't already exist.
    """
    i = 0
    while os.path.exists('model'+str(i)):
        i += 1

    new_dir = 'model'+str(i)
    os.mkdir(new_dir)
    return new_dir
