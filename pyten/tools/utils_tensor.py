import re

from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry


class IndexAdjustI(object):
    def __init__(self, raw):
        self.i = raw.i - 1
        self.j = raw.j
        self.k = raw.k
        self.val = raw.val

class IndexAdjustJ(object):
    def __init__(self, raw):
        self.i = raw.i
        self.j = raw.j - 1
        self.k = raw.k
        self.val = raw.val

class IndexAdjustK(object):
    def __init__(self, raw):
        self.i = raw.i
        self.j = raw.j
        self.k = raw.k - 1
        self.val = raw.val

class TensorEntry(object):
    def __init__(self, line):
        if type(line) is unicode:
            raw = re.split("[\s\t,]+", line.strip())
            if len(raw) != 4:
                raw = re.split("[\s\t;]+", line.strip())
            self.i = int(raw[0])
            self.j = int(raw[1])
            self.k = int(raw[2])
            if raw[3] == '':
                self.val = ''
            else:
                self.val = float(raw[3])
        elif type(line) is tuple and len(line) == 4:
            if line[3] != '':
                self.i = int(line[0])
                self.j = int(line[1])
                self.k = int(line[2])
                self.val = float(line[3])
        else:
            raise ValueError("Invalid value to create TensorEntry.")


class DistTensor(object):
    def __init__(self, entries = None, numDimX = 0, numDimY = 0, numDimZ = 0):
        self.entries = None
        self.numDimI = 0
        self.numDimJ = 0
        self.numDimK = 0
        if isinstance(entries, 'pyspark.rdd.RDD'):
            self.entries = entries
            self.numDimI = entries.map(lambda x: x.i).max() + 1
            self.numDimJ = entries.map(lambda x: x.j).max() + 1
            self.numDimK = entries.map(lambda x: x.k).max() + 1

    def unfolding(self, mode = None):
        def mapFuncI(entry):
            return MatrixEntry(entry.i, entry.k + self.numDimK * entry.j, entry.val)

        def mapFuncJ(entry):
            return MatrixEntry(entry.j, entry.i + self.numDimI * entry.k, entry.val)

        def mapFuncK(entry):
            return MatrixEntry(entry.k, entry.j + self.numDimJ * entry.i, entry.val)

        if mode == 1:
            matrix = CoordinateMatrix(self.entries.map(mapFuncI))
        elif mode == 2:
            matrix = CoordinateMatrix(self.entries.map(mapFuncJ))
        elif mode == 3:
            matrix = CoordinateMatrix(self.entries.map(mapFuncK))
        else:
            raise ValueError("The dimension index is out of the space!")

        return matrix


class TensorBlock(object):
    def __init__(self, entry=None):
        self.iIds = []
        self.jIds = []
        self.kIds = []
        self.vals = []
        self.uniqueIs = set()
        self.uniqueJs = set()
        self.uniqueKs = set()
        self.size = 0
        if entry:
            self.iIds.append(entry.i)
            self.jIds.append(entry.j)
            self.kIds.append(entry.k)
            # self.uniqueIs.add(entry.i)
            # self.uniqueJs.add(entry.j)
            # self.uniqueKs.add(entry.k)
            self.vals.append(entry.val)
            self.size += 1

    def add(self, entry):
        if entry:
            self.iIds.append(entry.i)
            self.jIds.append(entry.j)
            self.kIds.append(entry.k)
            self.vals.append(entry.val)
            # self.uniqueIs.add(entry.i)
            # self.uniqueJs.add(entry.j)
            # self.uniqueKs.add(entry.k)
            self.size += 1

    def merge(self, block):
        if block:
            self.iIds += block.iIds
            self.jIds += block.jIds
            self.kIds += block.kIds
            self.vals += block.vals
            # self.uniqueIs.union(block.uniqueIs)
            # self.uniqueJs.union(block.uniqueJs)
            # self.uniqueKs.union(block.uniqueKs)
            self.size += block.size

    def size(self):
        return self.size

    def build(self):
        # self.uniqueIs = sorted(self.uniqueIs)
        # self.uniqueJs = sorted(self.uniqueJs)
        # self.uniqueKs = sorted(self.uniqueKs)
        self.uniqueIs = sorted(set(self.iIds))
        self.uniqueJs = sorted(set(self.jIds))
        self.uniqueKs = sorted(set(self.kIds))

    # def __iter__(self):
    #     for x in [self.iIds, self.jIds, self.kIds, self.vals]:
    #         yield x


class CustomizedPartitioner(object):
    def __init__(self, numPartitioner = 1):
        self.numPartitioner = numPartitioner

    def getPartition(self, val):
        return int(val) % self.numPartitioner