import numpy as np


class SumTree:

    def __init__(self, size):
        self.steps = 0
        self.step = 0
        self.size = size
        self.tree = np.zeros(2 * size -1, dtype=np.float64)
        self.data = np.zeros(size, dtype=object)

    def add(self, p, data):
        self.data[self.step] = data
        self.update(self.size - 1 + self.step, p)

        self.step += 1

        if self.step >= self.size:
            self.step = 0
            self.steps += 1

    def update(self, idx, p):
        diff = p - self.tree[idx]
        self.tree[idx] = p

        self._propagate(idx, diff)

    def _propagate(self, idx, diff):
        parent = (idx - 1)//2

        self.tree[parent] = self.tree[parent] + diff
        if parent != 0:
            self._propagate(parent, diff)

    def get_data(self, value):
        idx = self._retrieve(0, value)
        data = self.data[idx - self.size + 1]
        return idx, data

    def _retrieve(self, idx, value):
        idx_l = 2 * idx + 1
        idx_r = idx_l + 1

        if idx_l >= len(self.tree):
            return idx

        if value - self.tree[idx_l] < 1e-8:
            return self._retrieve(idx_l, value)
        else:
            return self._retrieve(idx_r, value - self.tree[idx_l])

    def total(self):
        return self.tree[0], self.step + self.steps * self.size
