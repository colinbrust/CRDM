import numpy as np


# Class that builds on a traditional stack to facilitate sampling training data.
# Essentially, it ensures that we aren't sampling pixels that have already been sampled in the last 'max_len' weeks.
class Stack(object):

    def __init__(self, max_index, max_len):
        self.max_index = max_index
        self.max_len = max_len
        self.sample_size = self.max_index // self.max_len
        self.options = set(range(max_index))
        self.locs = []
        self.indices = set()

    def push(self, item):

        if len(self.indices.intersection(set(item))) > 0:
            raise ValueError('Values {} are already in Stack.'.format(self.indices.intersection(set(item))))

        self.locs = [item] + self.locs
        if len(self.locs) >= self.max_len:
            self.locs.pop()
        self.indices = set([x for l in self.locs for x in l])

    def get_options(self):
        return list(self.indices ^ self.options)

    def sample(self):

        return list(np.random.choice(self.get_options(), self.sample_size, False))
