import os


def make_model_dir():
    i = 0
    while os.path.exists('model'+str(i)):
        i += 1

    new_dir = 'model'+str(i)
    os.mkdir(new_dir)
    os.chdir(new_dir)
