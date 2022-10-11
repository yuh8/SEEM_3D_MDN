import numpy as np


class RunningStats:

    def __init__(self, size):
        self.n = 0
        self.old_m = np.zeros(size)
        self.new_m = np.zeros(size)
        self.old_s = np.zeros(size)
        self.new_s = np.zeros(size)
        self.size = size

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else np.zeros(self.size)

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else np.zeros(self.size)

    def standard_deviation(self):
        return np.sqrt(self.variance())
