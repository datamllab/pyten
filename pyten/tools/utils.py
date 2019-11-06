import re
import math
from scipy.linalg import blas
from scipy.optimize import nnls


class Value(object):
    def __init__(self, line):
        raw = re.split("[\s\t,]+", line.strip())
        self.x = int(raw[0])
        self.y = int(raw[1])
        self.val = float(raw[2])


class valueBlock(object):
    def __init__(self, Value=None):
        self.xIds = []
        self.yIds = []
        self.vals = []
        self.size = 0
        if Value:
            self.xIds.append(Value.x)
            self.yIds.append(Value.y)
            self.vals.append(Value.val)
            self.size += 1

    def add(self, value):
        if value:
            self.xIds.append(value.x)
            self.yIds.append(value.y)
            self.vals.append(value.val)
            self.size += 1

    def merge(self, block):
        if block:
            self.xIds += block.xIds
            self.yIds += block.yIds
            self.vals += block.vals
            self.size += block.size

    def size(self):
        return self.size


class localIndexEncoder(object):
    def __init__(self, numBlocks = 1):
        self.numBlocks = numBlocks
        self.numLocalIndexBits = min(31, 31-int(math.log(numBlocks-1, 2)))
        self.localIndexMask = (1 << self.numLocalIndexBits) - 1

    def encode(self, blockId, localIdx):
        if blockId < self.numBlocks and (localIdx & ~self.localIndexMask) == 0:
            return (blockId << self.numLocalIndexBits) | localIdx

    def getblockId(self, encoded):
        return encoded >> self.numLocalIndexBits

    def getlocalIdx(self, encoded):
        return encoded & self.localIndexMask


class customizedPartitioner(object):
    def __init__(self, numPartitioner = 1):
        self.numPartitioner = numPartitioner

    def getPartition(self, val):
        return int(val) % self.numPartitioner


class inBlock(object):
    def __init__(self, uniqueSrcIds, dstPtrs, dstEncodedIndices, values):
        self.uniqueSrcIds = uniqueSrcIds
        self.dstPtrs = dstPtrs
        self.dstEncodedIndices = dstEncodedIndices
        self.vals = values

    def size(self):
        return len(self.values)


class uncompressedInBlock(object):
    def __init__(self, srcIds, dstEncodedIndices, values):
        self.srcIds = srcIds
        self.dstEncodedIndices = dstEncodedIndices
        self.vals = values

    def compress(self):
        dataSorted = sorted(zip(self.srcIds, self.dstEncodedIndices, self.vals))
        srcIds, dstEncodedIndices, values = zip(*dataSorted)
        del dataSorted
        uniqueSrcIds = sorted(set(srcIds))
        uniqueSrcIdsDict = dict([(Id, 0) for Id in uniqueSrcIds])
        for Id in srcIds:
            uniqueSrcIdsDict[Id] += 1
        dstPtrs = [0]
        sumV = 0
        for Id in uniqueSrcIds:
            sumV += uniqueSrcIdsDict[Id]
            dstPtrs.append(sumV)
        return inBlock(list(uniqueSrcIds), dstPtrs, dstEncodedIndices, values)


class uncompressedInBlockBuilder(object):
    def __init__(self, encoder=None):
        self.srcIds = []
        self.dstEncodedIndices = []
        self.vals = []
        self.encoder = encoder

    def add(self, v):
        dstBlockId, srcIds, dstLocalIndices, values = v
        self.srcIds += srcIds
        self.vals += values
        if self.encoder:
            self.dstEncodedIndices += [self.encoder.encode(dstBlockId, x) for x in dstLocalIndices]

    def build(self):
        return uncompressedInBlock(self.srcIds, self.dstEncodedIndices, self.vals)


class normalEquation(object):
    def __init__(self, rank=1):
        self.k = rank
        self.triK = self.k * (self.k + 1) / 2
        self.ata = [0] * self.triK
        self.atb = [0] * self.k
        self.upper = "U"

    def dspr(self, uplo, n, alpha, x, incx, A):
        idx = 0
        for i in range(n):
            for j in range(i,n):
                A[idx] += x[i] * x[j]
                idx += 1

    def add(self, a, b, c=1.0):
        self.dspr(self.upper, self.k, 1.0, a, 1, self.ata)
        if b != 0:
            self.atb = blas.daxpy(a, self.atb, a=c*b)

    def merge(self, ne):
        self.ata = blas.daxpy(self.ata, ne.ata)
        self.atb = blas.daxpy(self.atb, ne.atb)

    def reset(self):
        self.ata = [0] * self.triK
        self.atb = [0] * self.k


class NNLS(object):
    def __init__(self, rank=1):
        self.rank = rank
        self.ata = [[0] * self.rank for _ in range(self.rank)]

    def constructSymmeticAtA(self, triAtA, regParam):
        pos = 0
        for i in range(self.rank):
            for j in range(i, self.rank):
                self.ata[i][j] = triAtA[pos]
                self.ata[j][i] = triAtA[pos]
                if i==j:
                    self.ata[i][j] += regParam
                pos += 1

    def solve(self, ne, regParam):
        self.constructSymmeticAtA(ne.ata, regParam)
        x = nnls(self.ata, ne.atb)
        ne.reset()
        return list(x[0])


if __name__ == "__main__":
    A = [[1,3],[3,9]]
    b = [1,3]
    print nnls(A,b)
    print blas.daxpy([1,3], [1,1], a = 0.5)
    solver = NNLS(3)
    solver.constructSymmeticAtA([1,4,5,2,6,3], 0.5)
    print solver.ata

    encoder = localIndexEncoder(5)

    for i in range(5):
        for j in range(5):
            encoded = encoder.encode(i, j)
            print encoded, (i,j)==(encoder.getblockId(encoded),encoder.getlocalIdx(encoded))
