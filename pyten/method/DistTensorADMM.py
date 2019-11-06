import re
import os
import sys
import math
import time
import numpy as np
import pyten.tenclass

from pyspark import SparkContext, SparkConf
from pyspark import StorageLevel

from pyten.tools.utils_tensor import *


class DistTensorADMM(object):
    def __init__(self):
        self.dir_data = None
        self.I = None
        self.J = None
        self.K = None
        self.size = None
        self.nnz = None
        self.numIBlocks = 5
        self.numJBlocks = 5
        self.numKBlocks = 5
        self.rank = 10
        self.maxIter = 10
        self.tol = 0.0001
        self.regLambda = 0.001
        self.regAlpha = 0
        self.regEta = 0.01   # 0.0001
        self.rho = 1.05         # 1.1
        self.maxEta = sys.maxint
        self.intermediateRDDStorageLevel = StorageLevel.MEMORY_AND_DISK
        self.finalRDDStorageLevel = StorageLevel.MEMORY_AND_DISK
        self.iFactors = None
        self.jFactors = None
        self.kFactors = None
        self.Us = None
        self.lambdaVals = None
        self.ktensor = None

    def initFactors(self, outBlocks, label='factor'):
        def generateRandomNumbers(count=1, seed=None):
            seed = seed if seed is not None else int(os.urandom(4).encode('hex'), 16)
            rs = np.random.RandomState(seed)
            return rs.normal(0, 1, count)

        def mapValuesFunc(_):
            factor = [abs(_) for _ in generateRandomNumbers(rank)]
            nrm = math.sqrt(sum([x * x for x in factor]))
            factor = [x / nrm for x in factor]
            return factor

        rank = self.rank
        if label == 'factor':
            initFactors = outBlocks.mapValues(mapValuesFunc)\
                                   .persist(self.intermediateRDDStorageLevel)
        elif label == 'variable':
            initFactors = outBlocks.mapValues(lambda x: [0 for _ in range(rank)]) \
                                   .persist(self.intermediateRDDStorageLevel)
        else:
            raise ValueError("InitFactor: cannot recognize label (factor, variable)")
        return initFactors

    def makeBlock(self, Blocks, mode, partitioner):
        def flatMapFunc(kv_pair):
            blockId, tensorblock = kv_pair
            if mode == 0:
                for iId in tensorblock.uniqueIs:
                    yield iId, set([blockId])
            elif mode == 1:
                for jId in tensorblock.uniqueJs:
                    yield jId, set([blockId])
            elif mode == 2:
                for kId in tensorblock.uniqueKs:
                    yield kId, set([blockId])
            else:
                raise ValueError("Invalid mode number!")

        outBlocks = Blocks.flatMap(flatMapFunc)\
                          .reduceByKey(lambda x, y: x.union(y), partitioner.numPartitioner)\
                          .persist(self.intermediateRDDStorageLevel)
        # print outBlocks.collect()
        return outBlocks

    def computeAtA(self, factors):
        def flatMapFunc(kp_pair):
            index, factor = kp_pair
            for i in range(len(factor)):
                for j in range(len(factor)):
                    yield (i, j), factor[i] * factor[j]
        return factors.flatMap(flatMapFunc).reduceByKey(lambda x, y: x + y).collect()

    def computeAtAInverse(self, AtA1, AtA2):
        AtA = np.ones((self.rank, self.rank))
        for i in range(AtA.size):
            (x1, y1), val1 = AtA1[i]
            (x2, y2), val2 = AtA2[i]
            AtA[x1][y1] *= val1
            AtA[x2][y2] *= val2
        for i in range(self.rank):
            AtA[i][i] += (self.regEta + self.regLambda)
        AtAInverse = np.linalg.inv(AtA)
        # print AtA1
        # print AtA2
        # print AtA
        # print AtAInverse, self.regEta, self.regLambda
        return AtAInverse

    def computeFactors(self, mode, tensorBlocks, factors1, outBlock1, factors2, outBlock2, Z, Y, AtAInverse):
        def flatMapFunc(kv_pair):
            index, (blockIds, factor) = kv_pair
            for blockId in blockIds:
                yield blockId, [(index, factor)]

        def flatMapComputeFunc(kp_pair):
            blockId, (block, (factor1, factor2)) = kp_pair
            factor1 = dict(factor1)
            factor2 = dict(factor2)
            for elem in zip(block.iIds, block.jIds, block.kIds, block.vals):
                key = elem[mode]
                val = elem[3]
                modes = range(3)
                indices = [elem[idx] for idx in modes[:mode] + modes[mode+1:]]
                factors = [factor1[indices[0]], factor2[indices[1]]]
                newRow = [x[0] * x[1] * val for x in zip(*factors)]
                yield key, newRow

        def mapValuesFunc(value):
            (val, z), y = value
            z = np.array(z)
            y = np.array(y)
            res = val + self.regEta * z + y
            res = res.dot(AtAInverse)
            return res.tolist()

        out1 = outBlock1.join(factors1, outBlock1.getNumPartitions())\
                        .flatMap(flatMapFunc)\
                        .reduceByKey(lambda x, y: x + y)

        out2 = outBlock2.join(factors2, outBlock2.getNumPartitions())\
                        .flatMap(flatMapFunc)\
                        .reduceByKey(lambda x, y: x + y)

        mttkrp = tensorBlocks.join(out1.join(out2, out1.getNumPartitions()), tensorBlocks.getNumPartitions())\
                             .flatMap(flatMapComputeFunc)\
                             .reduceByKey(lambda x, y: [sum(_) for _ in zip(x, y)])

        out = mttkrp.join(Z, mttkrp.getNumPartitions()).join(Y, mttkrp.getNumPartitions())\
                    .mapValues(mapValuesFunc)

        if mode == 2:
            return out, mttkrp.persist(self.intermediateRDDStorageLevel)
        else:
            return out, None

    def computeDualVariable(self, factors, Y):
        def mapValuesFunc(value):
            val, y = value
            val = np.array(val)
            y = np.array(y)
            res = val - y / eta
            return [_ if _ >= 0 else 0 for _ in res]

        eta = self.regEta
        out = factors.join(Y, factors.getNumPartitions())\
                     .mapValues(mapValuesFunc)\
                     .persist(self.intermediateRDDStorageLevel)
        return out

    def computeLargMultiplier(self, Y, factors, Z):
        def mapValuesFunc(value):
            (y, val), z = value
            y = np.array(y)
            val = np.array(val)
            z = np.array(z)
            res = y + eta * (z - val)
            return res.tolist()

        eta = self.regEta
        out = Y.join(factors, Y.getNumPartitions()).join(Z, Y.getNumPartitions())\
               .mapValues(mapValuesFunc)\
               .persist(self.intermediateRDDStorageLevel)
        return out

    def computeNormOfEstimatedTensor(self, iAtA, jAtA, kAtA, lambdaVals):
        AtA = np.ones((self.rank, self.rank))
        for i in range(AtA.size):
            (x1, y1), val1 = iAtA[i]
            (x2, y2), val2 = jAtA[i]
            (x3, y3), val3 = kAtA[i]
            AtA[x1][y1] *= val1
            AtA[x2][y2] *= val2
            AtA[x3][y3] *= val3
        res = lambdaVals.dot(AtA).dot(lambdaVals)
        return np.sqrt(res)

    def calculateError(self, mkktrp, factors, normData, normEst, lambdaVals):
        def mapFunc(kv_pair):
            index, (factor1, factor2) = kv_pair
            return sum([x * y * z for x, y, z in zip(factor1, factor2, lambdaVals)])

        innerProd = mkktrp.join(factors, mkktrp.getNumPartitions()).map(mapFunc).sum()
        error = np.sqrt(normData * normData + normEst * normEst - 2 * innerProd)
        fit = 1 - (error / normData)
        return error, fit

    def columnNormalization(self, factors, iteration):
        def mapValuesFunc(value):
            return [val / lvsq for val, lvsq in zip(value, lambdaVals)]

        if iteration == 0:
            lambdaValSquare = factors.map(lambda x: (x[1][0]*x[1][0], x[1][1]*x[1][1]))\
                                     .reduce(lambda x, y: [sum(_) for _ in zip(x, y)])
        else:
            lambdaValSquare = factors.map(lambda x: x[1])\
                                     .reduce(lambda x, y: [max(_+(1,)) for _ in zip(x, y)])
        lambdaVals = np.sqrt(lambdaValSquare)
        if all(x==1 for x in lambdaVals):
            return lambdaVals, factors
        else:
            normFactors = factors.mapValues(mapValuesFunc) \
                                 .persist(self.intermediateRDDStorageLevel)
            return lambdaVals, normFactors

    def run(self):
        conf = SparkConf()
        conf.setAppName('TF_ALS')
        conf.set("spark.storage.memoryFraction", "0.5")
        conf.set("spark.executor.memory", "6g")

        sc = SparkContext.getOrCreate(conf=conf)
        sc.setLogLevel('WARN')
        sc.addPyFile("pyten/tools/utils_tensor.py")

        start_time = time.time()

        # load the raw data from a different types of inputs with the pre-defined partitions
        if type(self.dir_data)==np.ndarray or type(self.dir_data)==pyten.tenclass.Tensor:
            if type(self.dir_data)==pyten.tenclass.Tensor:
                self.dir_data = self.dir_data.tondarray()
            subs = np.nonzero(self.dir_data)
            vals = self.dir_data[subs]
            self.dir_data = np.stack(subs + (vals,)).T
            self.dir_data = tuple(map(tuple, self.dir_data))
            rawData = sc.parallelize(self.dir_data).map(TensorEntry)
        elif type(self.dir_data)==str:
            if self.dir_data[-3:] == 'csv':
                rawData = sc.textFile(self.dir_data)
                header = rawData.first()
                rawData = rawData.filter(lambda row: row != header)
                rawData = rawData.map(TensorEntry).filter(lambda x: x.val!='')
            elif self.dir_data[-3:] == 'txt':
                rawData = sc.textFile(self.dir_data).map(TensorEntry).filter(lambda x: x.val!='')
            else:
                raise TypeError('No Such File or Not Support for This File Type')
        else:
            raise TypeError('No Such DataType')

        self.nnz = rawData.count()
        if rawData.map(lambda x: x.i).min() == 0:
            self.I = rawData.map(lambda x: x.i).max() + 1
        else:
            self.I = rawData.map(lambda x: x.i).max()
        if rawData.map(lambda x: x.j).min() == 0:
            self.J = rawData.map(lambda x: x.j).max() + 1
        else:
            self.J = rawData.map(lambda x: x.j).max()
        if rawData.map(lambda x: x.k).min() == 0:
            self.K = rawData.map(lambda x: x.k).max() + 1
        else:
            self.K = rawData.map(lambda x: x.k).max()
        self.size = (self.I, self.J, self.K)

        normData = np.sqrt(rawData.map(lambda x: x.val * x.val).sum())

        print "Dimension: {}x{}x{} with {} non-zero elements.".format(self.I, self.J, self.K, self.nnz)

        iPartitioner = CustomizedPartitioner(self.numIBlocks)
        jPartitioner = CustomizedPartitioner(self.numJBlocks)
        kPartitioner = CustomizedPartitioner(self.numKBlocks)

        def mergeValue(agg, v):
            f = agg
            f.add(v)
            return f

        def mergeCombiner(agg1, agg2):
            f = agg1
            f.merge(agg2)
            return f

        def mapValuesFunc(value):
            value.build()
            return value

        tensorBlocks = rawData.map(lambda x: ((iPartitioner.getPartition(x.i), jPartitioner.getPartition(x.j), kPartitioner.getPartition(x.k)), x))\
                              .aggregateByKey(TensorBlock(), mergeValue, mergeCombiner)\
                              .mapValues(mapValuesFunc)\
                              .persist(self.intermediateRDDStorageLevel)

        print "Time - Process Raw Tensor Data: {} sec.".format(time.time() - start_time)

        start_time = time.time()

        # print tensorBlocks.map(lambda x: (x[0], (x[1].uniqueIs, x[1].iIds, x[1].uniqueJs, x[1].jIds, x[1].uniqueKs, x[1].kIds, x[1].vals))).collect()

        iOutBlocks = self.makeBlock(tensorBlocks, 0, iPartitioner)
        jOutBlocks = self.makeBlock(tensorBlocks, 1, jPartitioner)
        kOutBlocks = self.makeBlock(tensorBlocks, 2, kPartitioner)

        print "Time - Calculate OutBlocks: {} sec.".format(time.time() - start_time)

        start_time = time.time()

        # print iOutBlocks.mapValues(len).collect()

        iFactors = self.initFactors(iOutBlocks, 'factor')
        jFactors = self.initFactors(jOutBlocks, 'factor')
        kFactors = self.initFactors(kOutBlocks, 'factor')

        # iFactors = sc.parallelize([(0, [0.657779466584, 0.753210577024]), (1, [0.85344409143, 0.521184403837]),
        #                            (2, [0.620733799485, 0.784021396504]), (3, [0.249584029223, 0.968353144445]),
        #                            (4, [0.0350049579474, 0.99938713866]), (5, [0.826226432899, 0.563338159172]),
        #                            (6, [0.595844977618, 0.803099472449]), (7, [0.0423617006339, 0.999102340263]),
        #                            (8, [0.722785058675, 0.691072904227]), (9, [0.464777956216, 0.88542727054])],
        #                           iOutBlocks.getNumPartitions()) \
        #              .persist(self.intermediateRDDStorageLevel)
        # jFactors = sc.parallelize([(0, [0.242517274998, 0.97014708747]), (1, [0.459425350703, 0.888216385309]),
        #                            (2, [0.842437905452, 0.538793444149]), (3, [0.97519165771, 0.221362216136]),
        #                            (4, [0.997621104371, 0.0689357100062]), (5, [0.0382065853847, 0.999269861866]),
        #                            (6, [0.747664267419, 0.664076910625]), (7, [0.969993722984, 0.24312995984]),
        #                            (8, [0.126317469934, 0.991989867282]), (9, [0.618615262594, 0.785694060615])],
        #                           jOutBlocks.getNumPartitions()) \
        #              .persist(self.intermediateRDDStorageLevel)
        # kFactors = sc.parallelize([(0, [0.429745694604, 0.902949964267]), (1, [0.609245817289, 0.792981421041]),
        #                            (2, [0.859656331349, 0.51087277474]), (3, [0.79929733552, 0.600935744844]),
        #                            (4, [0.975190685523, 0.221366498976]), (5, [0.398054563557, 0.917361741317]),
        #                            (6, [0.998998433367, 0.0447451687818]), (7, [0.101122988625, 0.994873932301]),
        #                            (8, [0.224515994927, 0.974470403872]), (9, [0.868663599483, 0.495402413129])],
        #                           kOutBlocks.getNumPartitions()) \
        #              .persist(self.intermediateRDDStorageLevel)




        # Zi = self.initFactors(iOutBlocks, 'variable')
        # Zj = self.initFactors(jOutBlocks, 'variable')
        # Zk = self.initFactors(kOutBlocks, 'variable')
        Yi = self.initFactors(iOutBlocks, 'variable')
        Yj = self.initFactors(jOutBlocks, 'variable')
        Yk = self.initFactors(kOutBlocks, 'variable')

        print "Time - Initial Factor Matrices: {} sec.".format(time.time() - start_time)

        start_time = time.time()

        # print "iFactors:"
        # for x in sorted(iFactors.collect()):
        #     print str(x[1][0]) + "," + str(x[1][1]) + ";"
        # print "jFactors:"
        # for x in sorted(jFactors.collect()):
        #     print str(x[1][0]) + "," + str(x[1][1]) + ";"
        # print "kFactors:"
        # for x in sorted(kFactors.collect()):
        #     print str(x[1][0]) + "," + str(x[1][1]) + ";"

        # iAtA = self.computeAtA(iFactors)
        jAtA = self.computeAtA(jFactors)
        kAtA = self.computeAtA(kFactors)

        print "Time - Two AtA Matrices: {} sec.".format(time.time() - start_time)

        # start_time = time.time()

        errors = []
        fits = []
        fitold = 0

        # print "Initial AtA:", iAtA, jAtA, kAtA

        for iteration in range(self.maxIter):
            start_time = time.time()
            self.regEta = min(self.rho * self.regEta, self.maxEta)

            Zi = self.computeDualVariable(iFactors, Yi)
            Zj = self.computeDualVariable(jFactors, Yj)
            Zk = self.computeDualVariable(kFactors, Yk)

            # print "Zi:"
            # for x in sorted(Zi.collect()):
            #     print str(x[1][0]) + "," + str(x[1][1]) + ";"
            # print "Zj:"
            # for x in sorted(Zj.collect()):
            #     print str(x[1][0]) + "," + str(x[1][1]) + ";"
            # print "Zk:"
            # for x in sorted(Zk.collect()):
            #     print str(x[1][0]) + "," + str(x[1][1]) + ";"

            jkAtAInverse = self.computeAtAInverse(jAtA, kAtA)
            prevIFactors = iFactors
            iFactors, _ = self.computeFactors(0, tensorBlocks, jFactors, jOutBlocks, kFactors, kOutBlocks, Zi, Yi, jkAtAInverse)
            # _, iFactors = self.columnNormalization(iFactors, iteration)
            prevIFactors.unpersist()
            iFactors.persist(self.intermediateRDDStorageLevel)
            # print "jkAtAInverse and iFactors", jkAtAInverse, sorted(iFactors.collect())
            iAtA = self.computeAtA(iFactors)
            # print "Updated iAtA", iAtA
            # print "New Lambdas", _

            ikAtAInverse = self.computeAtAInverse(iAtA, kAtA)
            prevJFactors = jFactors
            jFactors, _ = self.computeFactors(1, tensorBlocks, iFactors, iOutBlocks, kFactors, kOutBlocks, Zj, Yj, ikAtAInverse)
            # _, jFactors = self.columnNormalization(jFactors, iteration)
            prevJFactors.unpersist()
            jFactors.persist(self.intermediateRDDStorageLevel)
            # print "ikAtAInverse and jFactors", ikAtAInverse, sorted(jFactors.collect())
            jAtA = self.computeAtA(jFactors)
            # print "Updated jAtA", jAtA
            # print "New Lambdas", _

            ijAtAInverse = self.computeAtAInverse(iAtA, jAtA)
            prevKFactors = kFactors
            kFactors, kmkktrp = self.computeFactors(2, tensorBlocks, iFactors, iOutBlocks, jFactors, jOutBlocks, Zk, Yk, ijAtAInverse)
            # lambdaVals, kFactors = self.columnNormalization(kFactors, iteration)
            prevKFactors.unpersist()
            kFactors.persist(self.intermediateRDDStorageLevel)
            # print "ijAtAInverse and kFactors", ijAtAInverse, sorted(kFactors.collect())
            # print "kmkktrp", kmkktrp.collect() if kmkktrp else "None"
            kAtA = self.computeAtA(kFactors)
            # print "Updated kAtA", kAtA
            # print "New Lambdas", lambdaVals

            Yi = self.computeLargMultiplier(Yi, iFactors, Zi)
            Yj = self.computeLargMultiplier(Yj, jFactors, Zj)
            Yk = self.computeLargMultiplier(Yk, kFactors, Zk)

            # print "Yi:"
            # for x in sorted(Yi.collect()):
            #     print str(x[1][0]) + "," + str(x[1][1]) + ";"
            # print "Yj:"
            # for x in sorted(Yj.collect()):
            #     print str(x[1][0]) + "," + str(x[1][1]) + ";"
            # print "Yk:"
            # for x in sorted(Yk.collect()):
            #     print str(x[1][0]) + "," + str(x[1][1]) + ";"

            lambdaVals = np.array([1 for _ in range(self.rank)])
            normEst = self.computeNormOfEstimatedTensor(iAtA, jAtA, kAtA, lambdaVals)
            error, fit = self.calculateError(kmkktrp, kFactors, normData, normEst, lambdaVals)
            errors.append(error)
            fits.append(fit)
            fitchange = abs(fit - fitold)
            fitold = fit
            print error, fit, fitchange
            if fitchange < self.tol:
                break

            print "Time - Update Factor Matrices: {} sec.".format(time.time() - start_time)

        self.iFactors = iFactors
        self.jFactors = jFactors
        self.kFactors = kFactors
        self.Us = range(3)
        self.Us[0] = np.zeros([self.I, self.rank])
        self.Us[1] = np.zeros([self.J, self.rank])
        self.Us[2] = np.zeros([self.K, self.rank])
        tmp = self.iFactors.collect()
        for i in range(self.I):
            self.Us[0][tmp[i][0] - 1] = tmp[i][1]
        tmp = self.jFactors.collect()
        for j in range(self.J):
            self.Us[1][tmp[j][0] - 1] = tmp[j][1]
        tmp = self.kFactors.collect()
        for k in range(self.K):
            self.Us[2][tmp[k][0] - 1] = tmp[k][1]

        self.ktensor = pyten.tenclass.Ktensor(lambdaVals, self.Us)

        print errors
        print fits



if __name__ == "__main__":
    TF = DistTensorADMM()
    # TF.dir_data = "tensor_1000X1000X1000_10000_10.txt"
    TF.dir_data = "tensor_10x10x10_101.txt"
    TF.run()
