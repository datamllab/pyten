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


class TensorDecompositionALS(object):
    def __init__(self, dir_data = None,
                 appName = "TensorDecomposition_ALS",
                 I = 100, J = 100, K = 100, rank = 10,
                 numIBlocks = 5, numJBlocks = 5, numKBlocks = 5,
                 maxIteration = 10, errorTolerance = 0.000001):
        self.appName = appName
        self.I = I
        self.J = J
        self.K = K
        self.size = None
        self.nnz = None
        self.dir_data = dir_data
        self.numIBlocks = numIBlocks
        self.numJBlocks = numJBlocks
        self.numKBlocks = numKBlocks
        self.rank = rank
        self.maxIter = maxIteration
        self.tol = errorTolerance
        self.intermediateRDDStorageLevel = StorageLevel.MEMORY_AND_DISK
        self.finalRDDStorageLevel = StorageLevel.MEMORY_AND_DISK
        self.iFactors = None
        self.jFactors = None
        self.kFactors = None
        self.Us = None
        self.lambdaVals = None
        self.ktensor = None


    def initFactors(self, outBlocks):
        # initialize factor matrices for a dimension
        def generateRandomNumbers(count=1, seed=None):
            # generate a seed for conducting random numbers in a distributed system
            seed = seed if seed is not None else int(os.urandom(4).encode('hex'), 16)
            rs = np.random.RandomState(seed)
            return rs.normal(0, 1, count)

        def mapValuesFunc(_):
            # map_value function for generating a normalized row in a factor matrix
            factor = [abs(_) for _ in generateRandomNumbers(rank)]
            nrm = math.sqrt(sum([x * x for x in factor]))
            factor = [x / nrm for x in factor]
            return factor

        rank = self.rank
        initFactors = outBlocks.mapValues(mapValuesFunc)\
                               .persist(self.intermediateRDDStorageLevel)
        return initFactors

    def makeBlock(self, Blocks, mode, partitioner):
        # generate the information of which blocks an index in a dimension is located in
        def flatMapFunc(kv_pair):
            # flatMap function for generating pairs (index, [blockIds]) for each of dimensions
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
        return outBlocks

    def computeAtA(self, factors):
        # calculate AtA for a factor matrix
        def flatMapFunc(kp_pair):
            # flatMap function used to generate the element-wise multiplication of two rows in a factor matrix
            index, factor = kp_pair
            for i in range(len(factor)):
                for j in range(len(factor)):
                    yield (i, j), factor[i] * factor[j]

        return factors.flatMap(flatMapFunc)\
                      .reduceByKey(lambda x, y: x + y).collect()

    def computeAtAInverse(self, AtA1, AtA2):
        # calculate the inverse of the multiplication of two factor matrices
        AtA = np.ones((self.rank, self.rank))
        for i in range(AtA.size):
            (x1, y1), val1 = AtA1[i]
            (x2, y2), val2 = AtA2[i]
            AtA[x1][y1] *= val1
            AtA[x2][y2] *= val2
        AtAInverse = np.linalg.inv(AtA)
        return AtAInverse

    def computeFactors(self, mode, tensorBlocks, factors1, outBlock1, factors2, outBlock2, AtAInverse):
        # update a factor matrix
        def flatMapFunc(kv_pair):
            # flatMap function for unpacking to (blockId, [(index, row in the factor matrix)])
            index, (blockIds, factor) = kv_pair
            for blockId in blockIds:
                yield blockId, [(index, factor)]

        def flatMapComputeFunc(kp_pair):
            # flatMap function for updating a row in the factor matrix
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
            # mapValues function for multiplying a updated row with AtAInverse
            value = np.array(value)
            res = value.dot(AtAInverse)
            return res.tolist()

        # create a RDD storing pairs (blockId, [all related (index, row in the factor matrix)]) for a dimension
        out1 = outBlock1.join(factors1, outBlock1.getNumPartitions())\
                        .flatMap(flatMapFunc)\
                        .reduceByKey(lambda x, y: x + y)

        # create a RDD storing pairs (blockId, [all related (index, row in the factor matrix)]) for a dimension
        out2 = outBlock2.join(factors2, outBlock2.getNumPartitions())\
                        .flatMap(flatMapFunc)\
                        .reduceByKey(lambda x, y: x + y)

        # calculate MTTKRP as a new RDD for the dimension by joining the previous RDDs
        mttkrp = tensorBlocks.join(out1.join(out2), tensorBlocks.getNumPartitions())\
                             .flatMap(flatMapComputeFunc)\
                             .reduceByKey(lambda x, y: [sum(_) for _ in zip(x, y)])

        # multiplying MTTKRP with AtAInverse
        out = mttkrp.mapValues(mapValuesFunc)

        if mode == 2:
            return out, mttkrp.persist(self.intermediateRDDStorageLevel)
        else:
            return out, None

    def computeNormOfEstimatedTensor(self, iAtA, jAtA, kAtA, lambdaVals):
        # compute the norm of the estimated tensor
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
        # calculate the error between the original and estimated tensors
        def mapFunc(kv_pair):
            # map function for computing the multiplication of corresponding rows in two dimensions and lambda values (1 for most cases)
            index, (factor1, factor2) = kv_pair
            return sum([x * y * z for x, y, z in zip(factor1, factor2, lambdaVals)])

        # calculate the inner product by using MTTKRP and the corresponding factor matrix
        innerProd = mkktrp.join(factors, mkktrp.getNumPartitions())\
                          .map(mapFunc)\
                          .sum()
        # calculate the error
        error = np.sqrt(normData * normData + normEst * normEst - 2 * innerProd)
        # calculate the fit
        fit = 1 - (error / normData)
        return error, fit

    def columnNormalization(self, factors, iteration):
        # column-wise normalize the factor matrix
        def mapValuesFunc(value):
            return [val / lvsq for val, lvsq in zip(value, lambdaVals)]

        if iteration == 0:
            lambdaValSquare = factors.map(lambda x: [val*val for val in x[1]])\
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

#    def readArgvs(self):
#        if len(sys.argv) != 12:
#            raise ValueError("Invalid arguments!")

#        self.appName = sys.argv[1]
#        self.dir_data= sys.argv[2]
#        self.I = int(sys.argv[3])
#        self.J = int(sys.argv[4])
#        self.K = int(sys.argv[5])
#        self.numIBlocks = int(sys.argv[6])
#        self.numJBlocks = int(sys.argv[7])
#        self.numKBlocks = int(sys.argv[8])
#        self.rank = int(sys.argv[9])
#        self.maxIter = int(sys.argv[10])
#        self.tol = float(sys.argv[11])

    def run(self):
        # start the timer for the whole process
        start_program = time.time()

        # read all arguments from run.sh
#        self.readArgvs()

        # configure Spark
        conf = SparkConf()
        conf.setAppName(self.appName)
        # conf.set("spark.storage.memoryFraction", "0.5")
        conf.set("spark.executor.memory", "4g")
        conf.set("spark.driver.memory", "4g")
        sc = SparkContext.getOrCreate(conf=conf)
        sc.setLogLevel('WARN')
        sc.addPyFile("pyten/tools/utils_tensor.py")

        print "TIME - Setting up the Spark: {} sec.".format(time.time() - start_program)

        # start the timer for processing data
        start_process_data = time.time()

        # load the raw data from a txt file
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

        # count the number of observations
        self.nnz = rawData.count()
        # compute the size for each dimension in the tensor
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
        # save the shape of the observed tensor
        self.size = (self.I, self.J, self.K)

        # calculate the norm of the observed tensor
        normData = np.sqrt(rawData.map(lambda x: x.val * x.val).sum())

        print "INFO - A tensor of the size {}x{}x{} with {} non-zero elements.".format(self.I, self.J, self.K, self.nnz)

        # initialize the partitioner for each dimension
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

        # convert the raw data to the TensorBlock
        tensorBlocks = rawData.map(lambda x: ((iPartitioner.getPartition(x.i), jPartitioner.getPartition(x.j), kPartitioner.getPartition(x.k)), x))\
                              .aggregateByKey(TensorBlock(), mergeValue, mergeCombiner)\
                              .mapValues(mapValuesFunc)\
                              .persist(self.intermediateRDDStorageLevel)

        print "TIME - Process Raw Tensor Data: {} sec.".format(time.time() - start_process_data)

        # start the timer for the initialization
        start_init = time.time()

        # record the information of which blocks an index in a dimension is located in
        iOutBlocks = self.makeBlock(tensorBlocks, 0, iPartitioner)
        jOutBlocks = self.makeBlock(tensorBlocks, 1, jPartitioner)
        kOutBlocks = self.makeBlock(tensorBlocks, 2, kPartitioner)

        # initialize all factor matrices
        iFactors = self.initFactors(iOutBlocks)
        jFactors = self.initFactors(jOutBlocks)
        kFactors = self.initFactors(kOutBlocks)

        print "TIME - Initial Factor Matrices: {} sec.".format(time.time() - start_init)

        # calculate AtA for dimensions J and K
        jAtA = self.computeAtA(jFactors)
        kAtA = self.computeAtA(kFactors)

        # initialize variables for storing errors and fits
        errors = []
        fits = []
        fitold = 0

        # start a timer for the iteration
        start_iter = time.time()

        # loop starts
        for iteration in range(self.maxIter):
            # start a timer for the inner loop
            start_inner = time.time()

            # update the factor matrix for the dimension I
            jkAtAInverse = self.computeAtAInverse(jAtA, kAtA)
            prevIFactors = iFactors
            iFactors, _ = self.computeFactors(0, tensorBlocks, jFactors, jOutBlocks, kFactors, kOutBlocks, jkAtAInverse)
            _, iFactors = self.columnNormalization(iFactors, iteration)
            prevIFactors.unpersist()
            iFactors.persist(self.intermediateRDDStorageLevel)
            # re-calculate the AtA for the dimension I
            iAtA = self.computeAtA(iFactors)

            # update the factor matrix for the dimension J
            ikAtAInverse = self.computeAtAInverse(iAtA, kAtA)
            prevJFactors = jFactors
            jFactors, _ = self.computeFactors(1, tensorBlocks, iFactors, iOutBlocks, kFactors, kOutBlocks, ikAtAInverse)
            _, jFactors = self.columnNormalization(jFactors, iteration)
            prevJFactors.unpersist()
            jFactors.persist(self.intermediateRDDStorageLevel)
            # re-calculate the AtA for the dimension J
            jAtA = self.computeAtA(jFactors)

            # update the factor matrix for the dimension K
            ijAtAInverse = self.computeAtAInverse(iAtA, jAtA)
            prevKFactors = kFactors
            kFactors, kmkktrp = self.computeFactors(2, tensorBlocks, iFactors, iOutBlocks, jFactors, jOutBlocks, ijAtAInverse)
            lambdaVals, kFactors = self.columnNormalization(kFactors, iteration)
            prevKFactors.unpersist()
            kFactors.persist(self.intermediateRDDStorageLevel)
            # re-calculate the AtA for the dimension K
            kAtA = self.computeAtA(kFactors)

            # calculate the norm of the estimated tensor
            normEst = self.computeNormOfEstimatedTensor(iAtA, jAtA, kAtA, lambdaVals)
            # calculate the error and the fit between the original and estimated tensors
            error, fit = self.calculateError(kmkktrp, kFactors, normData, normEst, lambdaVals)
            errors.append(error)
            fits.append(fit)
            # calculate the change of fit comparing with the previous one
            fitchange = abs(fit - fitold)
            fitold = fit
            print "Iteration {}: error-{}, fit-{} and fitchange-{} with {} seconds".format(iteration, error, fit, fitchange, time.time() - start_inner)
            # stopping criteria
            if fitchange < self.tol:
                break

            print "TIME - Update Factor Matrices: {} sec. in {} iterations".format(time.time() - start_inner, iteration)

        self.iFactors = iFactors
        self.jFactors = jFactors
        self.kFactors = kFactors
        self.Us = range(3)
        self.Us[0] = np.zeros([self.I, self.rank])
        self.Us[1] = np.zeros([self.J, self.rank])
        self.Us[2] = np.zeros([self.K, self.rank])
        tmp = self.iFactors.collect()
        for i in range(self.I):
            self.Us[0][tmp[i][0]-1] = tmp[i][1]
        tmp = self.jFactors.collect()
        for j in range(self.J):
            self.Us[1][tmp[j][0] - 1] = tmp[j][1]
        tmp = self.kFactors.collect()
        for k in range(self.K):
            self.Us[2][tmp[k][0] - 1] = tmp[k][1]
        self.lambdaVals = lambdaVals
        self.ktensor = pyten.tenclass.Ktensor(self.lambdaVals, self.Us)

        print "TIME - Completeing the whole iterations: {} sec.".format(time.time() - start_iter)
        print "TIME - Total computation time: {} sec.".format(time.time() - start_program)


if __name__ == "__main__":
    TF = TensorDecompositionALS()
    TF.dir_data = "tensor_10x10x10_101.txt"
    TF.run()