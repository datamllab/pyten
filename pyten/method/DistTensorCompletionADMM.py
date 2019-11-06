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


class DistTensorCompletionADMM(object):
    def __init__(self):
        self.I = None
        self.J = None
        self.K = None
        self.regLambda = 0.001
        self.regAlpha = 0
        self.regEta = 0.01
        self.rho = 1.05
        self.maxEta = sys.maxint
        self.size = None
        self.nnz = None
        self.dir_data = None
        self.numIBlocks = 5
        self.numJBlocks = 5
        self.numKBlocks = 5
        self.rank = 10
        self.maxIter = 10
        self.tol = 0.0001
        self.intermediateRDDStorageLevel = StorageLevel.MEMORY_AND_DISK
        self.finalRDDStorageLevel = StorageLevel.MEMORY_AND_DISK
        self.iFactors = None
        self.jFactors = None
        self.kFactors = None
        self.Us = None
        self.lambdaVals = None
        self.ktensor = None

    def initFactors(self, dim, sc, partitioner, label='factor'):
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

        # distribute indices of a factor matrix in the cluster
        initFactors = sc.range(dim, numSlices=partitioner.numPartitioner)\
                        .map(lambda x: (x, x))

        if label == 'factor':
            # initialize factor matrices
            initFactors = initFactors.mapValues(mapValuesFunc)\
                                     .persist(self.intermediateRDDStorageLevel)
        elif label == 'variable':
            # initialize other variables with all zeros
            initFactors = initFactors.mapValues(lambda x: [0 for _ in range(rank)]) \
                                     .persist(self.intermediateRDDStorageLevel)
        else:
            raise ValueError("InitFactor: cannot recognize label (factor, variable)")
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
        def mapFunc(kp_pair):
            # flatMap function used to calculate the element-wise multiplication of a row itself in a factor matrix
            key, val = kp_pair
            val = np.array(val)
            return np.outer(val,val).tolist()

        def reduceFunc(x, y):
            # reduce all intermediate results by adding them together
            x_np = np.array(x)
            y_np = np.array(y)
            out = x_np + y_np
            return out.tolist()

        return factors.map(mapFunc)\
                      .reduce(reduceFunc)

    def computeAtAInverse(self, AtA1, AtA2):
        # calculate the inverse of the multiplication of two factor matrices
        AtA = np.ones((self.rank, self.rank))
        for i in range(AtA.size):
            (x1, y1), val1 = AtA1[i]
            (x2, y2), val2 = AtA2[i]
            AtA[x1][y1] *= val1
            AtA[x2][y2] *= val2
        for i in range(self.rank):
            AtA[i][i] += (self.regEta + self.regLambda)
        AtAInverse = np.linalg.inv(AtA)
        return AtAInverse

    def computeUtU(self, AtA1, AtA2):
        # calculate UtU by element-wise multiplying all AtAs excluding the current dimension
        AtA1_np = np.array(AtA1)
        AtA2_np = np.array(AtA2)
        UtU = AtA1_np * AtA2_np
        return UtU.tolist()

    def computeFactors(self, mode, factors1, tensorBlocks, factors2, outBlock2, factors3, outBlock3, Z, Y, UtU, Pt, sc, iter_idx):
        # update a factor matrix
        def flatMapFunc(kv_pair):
            # flatMap function for unpacking to (blockId, [(index, row in the factor matrix)])
            index, (blockIds, factor) = kv_pair
            for blockId in blockIds:
                yield blockId, [(index, factor)]

        def flatMapComputeFunc(kp_pair):
            # flatMap function for updating a row in the factor matrix
            blockId, values = kp_pair
            tb, factor2, factor3 = None, None, None
            # set tb, factor2 and factor3 with the correct contents
            for x in values:
                if type(x) is tuple:
                    if x[0] == 2:
                        factor2 = x[1]
                    else:
                        factor3 = x[1]
                if type(x) is TensorBlock:
                    tb = x
            factor2 = dict(factor2)
            factor3 = dict(factor3)
            for elem in zip(tb.iIds, tb.jIds, tb.kIds, tb.vals):
                key = elem[mode]
                val = elem[3]
                modes = range(3)
                indices = [elem[idx] for idx in modes[:mode] + modes[mode+1:]]
                factors = [factor2[indices[0]], factor3[indices[1]]]
                newRow = [x[0] * x[1] * val for x in zip(*factors)]
                yield key, newRow

        def mapValuesFunc(value):
            # mapValues function for multiplying a updated row with AtAInverse
            val_dict = dict(value)
            if len(val_dict) != 4:
                return val_dict['factor1']
            fact1 = val_dict['factor1']
            h = val_dict['mttkrp']
            z = val_dict['Z']
            y = val_dict['Y']
            fact1 = np.array(fact1)
            h = np.array(h)
            z = np.array(z)
            y = np.array(y)
            if iter_idx == 0:
                res = h + regEta * z + y
            else:
                res = fact1.dot(UtU) + h + regEta * z + y
            res = res.dot(UtUInverse)
            return res.tolist()

        def mergeValue(f, v):
            f.append(v)
            return f

        def mergeCombiner(f1, f2):
            return f1 + f2

        def computeUtUInverse(UTU):
            # calculate the inverse of the summation of UtU, regEta and regLambda
            UtU_np = np.array(UTU)
            for i in range(rank):
                UtU_np[i][i] += (regEta + regLambda)
            UtUInverse = np.linalg.inv(UtU_np)
            return UtUInverse

        rank = self.rank
        regEta = self.regEta
        regLambda = self.regLambda
        UtUInverse = computeUtUInverse(UtU)

        # create a RDD storing pairs (blockId, [all related (index, row in the factor matrix)]) for a dimension
        out2 = outBlock2.join(factors2, outBlock2.getNumPartitions())\
                        .flatMap(flatMapFunc)\
                        .reduceByKey(lambda x, y: x + y)\
                        .mapValues(lambda x: (2,x))

        # create a RDD storing pairs (blockId, [all related (index, row in the factor matrix)]) for a dimension
        out3 = outBlock3.join(factors3, outBlock3.getNumPartitions())\
                        .flatMap(flatMapFunc)\
                        .reduceByKey(lambda x, y: x + y)\
                        .mapValues(lambda x: (3,x))

        if iter_idx == 0:
            # the observed tensor is used to calculate MTTKRP in the first iteration
            mttkrp = sc.union([tensorBlocks, out2, out3]) \
                       .partitionBy(tensorBlocks.getNumPartitions()) \
                       .aggregateByKey([], mergeValue, mergeCombiner) \
                       .flatMap(flatMapComputeFunc) \
                       .reduceByKey(lambda x, y: [sum(_) for _ in zip(x, y)])
        else:
            # the residual tensor is used to calculate MTTKRP after the first iteration
            mttkrp = sc.union([Pt, out2, out3]) \
                       .partitionBy(Pt.getNumPartitions()) \
                       .aggregateByKey([], mergeValue, mergeCombiner) \
                       .flatMap(flatMapComputeFunc) \
                       .reduceByKey(lambda x, y: [sum(_) for _ in zip(x, y)])

        factors1 = factors1.mapValues(lambda x: ('factor1',x))
        mttkrp = mttkrp.mapValues(lambda x: ('mttkrp', x))
        Z = Z.mapValues(lambda x: ('Z', x))
        Y = Y.mapValues(lambda x: ('Y', x))

        out = sc.union([factors1, mttkrp, Z, Y]) \
                .partitionBy(factors1.getNumPartitions()) \
                .aggregateByKey([], mergeValue, mergeCombiner) \
                .mapValues(mapValuesFunc)

        return out

    def computeDualVariable(self, factors, Y):
        # update the auxiliary variables
        def mapValuesFunc(value):
            val, y = value
            val = np.array(val)
            y = np.array(y)
            res = val - y / eta
            return res.tolist()

        eta = self.regEta
        out = factors.join(Y, factors.getNumPartitions())\
                     .mapValues(mapValuesFunc)\
                     .persist(self.intermediateRDDStorageLevel)
        return out

    def computeLargMultiplier(self, Y, factors, Z):
        # update the Lagrange multipliers
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

    def calculateError(self, Pt, normData, lambdaVals):
        # calculate the error by directly computing the square root of the residual tensor
        error = np.sqrt(Pt.map(lambda x: sum([val * val for val in x[1].vals])).sum())
        fit = 1 - (error / normData)
        return error, fit

    def columnNormalization(self, factors, iteration):
        # column-wise normalize the factor matrix
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

    def initResidualTensor(self, tensorBlocks):
        def mapValuesFunc(tensor_block):
            tensor_block.vals = [0 for val in tensor_block.vals]
            return tensor_block

        res = tensorBlocks.mapValues(mapValuesFunc) \
                          .persist(self.intermediateRDDStorageLevel)
        return res

    def computeResidualTensor(self, tensorBlocks, factors1, factors2, factors3, outBlock1, outBlock2, outBlock3, sc):
        # calculate the residual tensor
        def flatMapFunc(kv_pair):
            # flatMap function for unpacking to (blockId, [(index, row in the factor matrix)])
            index, (blockIds, factor) = kv_pair
            for blockId in blockIds:
                yield blockId, [(index, factor)]

        def mergeValue(f, v):
            f.append(v)
            return f

        def mergeCombiner(f1, f2):
            return f1 + f2

        def mapValuesFunc(value):
            # mapValues function for calculating the residual tensor for each observation
            tb, fact1, fact2, fact3 = None, None, None, None
            for x in value:
                if type(x) is tuple:
                    if x[0] == 1:
                        fact1 = x[1]
                    elif x[0] == 2:
                        fact2 = x[1]
                    else:
                        fact3 = x[1]
                if type(x) is TensorBlock:
                    tb = x
            fact1 = dict(fact1)
            fact2 = dict(fact2)
            fact3 = dict(fact3)
            for i in range(len(tb.vals)):
                i_idx = tb.iIds[i]
                j_idx = tb.jIds[i]
                k_idx = tb.kIds[i]
                tb.vals[i] = tb.vals[i] - sum([fact1[i_idx][r] * fact2[j_idx][r] * fact3[k_idx][r] for r in range(rank)])
            return tb

        rank = self.rank

        # create a RDD storing pairs (blockId, [all related (index, (idx_dimension, row in the factor matrix))]) for a dimension
        out1 = outBlock1.join(factors1, outBlock1.getNumPartitions())\
                        .flatMap(flatMapFunc)\
                        .reduceByKey(lambda x, y: x + y)\
                        .mapValues(lambda x: (1,x))

        # create a RDD storing pairs (blockId, [all related (index, (idx_dimension, row in the factor matrix))]) for a dimension
        out2 = outBlock2.join(factors2, outBlock2.getNumPartitions()) \
                        .flatMap(flatMapFunc) \
                        .reduceByKey(lambda x, y: x + y)\
                        .mapValues(lambda x: (2,x))

        # create a RDD storing pairs (blockId, [all related (index, (idx_dimension, row in the factor matrix))]) for a dimension
        out3 = outBlock3.join(factors3, outBlock3.getNumPartitions()) \
                        .flatMap(flatMapFunc) \
                        .reduceByKey(lambda x, y: x + y)\
                        .mapValues(lambda x: (3,x))

        Pt = sc.union([tensorBlocks, out1, out2, out3]) \
               .partitionBy(tensorBlocks.getNumPartitions()) \
               .aggregateByKey([], mergeValue, mergeCombiner) \
               .mapValues(mapValuesFunc) \
               # .persist(self.intermediateRDDStorageLevel)
        return Pt

    def readArgvs(self):
        if len(sys.argv) != 16:
            raise ValueError("Invalid arguments!")

        self.appName = sys.argv[1]
        self.dir_data= sys.argv[2]
        self.I = int(sys.argv[3])
        self.J = int(sys.argv[4])
        self.K = int(sys.argv[5])
        self.numIBlocks = int(sys.argv[6])
        self.numJBlocks = int(sys.argv[7])
        self.numKBlocks = int(sys.argv[8])
        self.rank = int(sys.argv[9])
        self.maxIter = int(sys.argv[10])
        self.tol = float(sys.argv[11])
        self.regLambda = float(sys.argv[12])
        self.regAlpha = float(sys.argv[13])
        self.regEta = float(sys.argv[14])
        self.rho = float(sys.argv[15])

    def run(self):
        # start the timer for the whole process
        start_program = time.time()

        # read all arguments from run.sh
#        self.readArgvs()

        # configure Spark
        conf = SparkConf()
        conf.setAppName('DisTenC')
        conf.set("spark.storage.memoryFraction", "0.5")
        conf.set("spark.executor.memory", "4g")
        conf.set("spark.driver.memory", "4g")
        sc = SparkContext.getOrCreate(conf=conf)
        sc.setLogLevel('WARN')
        sc.addPyFile("pyten/tools/utils_tensor.py")

        print "TIME - Setting up the Spark: {} sec.".format(time.time() - start_program)

        # start the timer for processing data
        start_process_data = time.time()

        # initialize the partitioner for each dimension
        iPartitioner = CustomizedPartitioner(self.numIBlocks)
        jPartitioner = CustomizedPartitioner(self.numJBlocks)
        kPartitioner = CustomizedPartitioner(self.numKBlocks)

        # calculate the number of partitions for the observed tensor
        numPartitions = self.numIBlocks * self.numJBlocks * self.numKBlocks

        ## load the raw data from a txt file with the pre-defined partitions
        #rawData = sc.textFile(self.dir_data, minPartitions=numPartitions)\
        #            .map(TensorEntry)

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




        # count the number of observations
        self.nnz = rawData.count()
        # compute the size for each dimension in the tensor
        if rawData.map(lambda x: x.i).min() == 0:
            self.I = rawData.map(lambda x: x.i).max() + 1
        else:
            self.I = rawData.map(lambda x: x.i).max()
            rawData = rawData.map(IndexAdjustI)
        if rawData.map(lambda x: x.j).min() == 0:
            self.J = rawData.map(lambda x: x.j).max() + 1
        else:
            self.J = rawData.map(lambda x: x.j).max()
            rawData = rawData.map(IndexAdjustJ)
        if rawData.map(lambda x: x.k).min() == 0:
            self.K = rawData.map(lambda x: x.k).max() + 1
        else:
            self.K = rawData.map(lambda x: x.k).max()
            rawData = rawData.map(IndexAdjustK)

        # save the shape of the observed tensor
        self.size = (self.I, self.J, self.K)

        normData = np.sqrt(rawData.map(lambda x: x.val * x.val).sum())

        print "INFO - A tensor of the size {}x{}x{} with {} non-zero elements.".format(self.I, self.J, self.K, self.nnz)

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
                              .partitionBy(numPartitions, lambda key: hash(key))\
                              .persist(self.intermediateRDDStorageLevel)

        print "TIME - Process Raw Tensor Data: {} sec.".format(time.time() - start_process_data)

        # start the timer for the initialization
        start_init = time.time()

        # record the information of which blocks an index in a dimension is located in
        iOutBlocks = self.makeBlock(tensorBlocks, 0, iPartitioner)
        jOutBlocks = self.makeBlock(tensorBlocks, 1, jPartitioner)
        kOutBlocks = self.makeBlock(tensorBlocks, 2, kPartitioner)

        # initialize all factor matrices
        iFactors = self.initFactors(self.I, sc, iPartitioner, 'factor')
        jFactors = self.initFactors(self.J, sc, jPartitioner, 'factor')
        kFactors = self.initFactors(self.K, sc, kPartitioner, 'factor')

        # initialize all Lagrange multipliers
        Yi = self.initFactors(self.I, sc, iPartitioner, 'variable')
        Yj = self.initFactors(self.J, sc, jPartitioner, 'variable')
        Yk = self.initFactors(self.K, sc, kPartitioner, 'variable')

        print "TIME - Initial Factor Matrices: {} sec.".format(time.time() - start_init)

        # calculate AtA for dimensions J and K
        jAtA = self.computeAtA(jFactors)
        kAtA = self.computeAtA(kFactors)

        # initialize variables for storing errors and fits
        errors = []
        fits = []
        fitold = 0

        # start a timer for the calculation of the residual tensor
        start_init_pt = time.time()

        # calculate the initial residual tensor
        Pt = self.initResidualTensor(tensorBlocks)

        print "TIME - Initialize Pt: {} sec.".format(time.time() - start_init_pt)

        # start a timer for the iteration
        start_iter = time.time()

        # loop starts
        for iteration in range(self.maxIter):
            # start a timer for the inner loop
            start_inner = time.time()

            # update the parameter eta (step size)
            self.regEta = min(self.rho * self.regEta, self.maxEta)

            # update the auxiliary variables
            start_update_z = time.time()
            Zi = self.computeDualVariable(iFactors, Yi)
            Zj = self.computeDualVariable(jFactors, Yj)
            Zk = self.computeDualVariable(kFactors, Yk)
            print "TIME: updating Z: {}".format(time.time() - start_update_z)

            # start a timer for the update in the dimension I
            start_update_fact_i = time.time()
            # update the factor matrix for the dimension I
            jkUtU = self.computeUtU(jAtA, kAtA)
            iFactors = self.computeFactors(0, iFactors, tensorBlocks, jFactors, jOutBlocks, kFactors, kOutBlocks, Zi, Yi, jkUtU, Pt, sc, iteration)
            iFactors.persist(self.intermediateRDDStorageLevel)
            # re-calculate the AtA for the dimension I
            iAtA = self.computeAtA(iFactors)
            print "TIME: updating I: {}".format(time.time() - start_update_fact_i)

            # start a timer for the update in the dimension J
            start_update_fact_j = time.time()
            # update the factor matrix for the dimension J
            ikUtU = self.computeUtU(iAtA, kAtA)
            jFactors = self.computeFactors(1, jFactors, tensorBlocks, iFactors, iOutBlocks, kFactors, kOutBlocks, Zj, Yj, ikUtU, Pt, sc, iteration)
            jFactors.persist(self.intermediateRDDStorageLevel)
            # re-calculate the AtA for the dimension J
            jAtA = self.computeAtA(jFactors)
            print "TIME: updating J: {}".format(time.time() - start_update_fact_j)

            # start a timer for the update in the dimension K
            start_update_fact_k = time.time()
            # update the factor matrix for the dimension K
            ijUtU = self.computeUtU(iAtA, jAtA)
            kFactors = self.computeFactors(2, kFactors, tensorBlocks, iFactors, iOutBlocks, jFactors, jOutBlocks, Zk, Yk, ijUtU, Pt, sc, iteration)
            kFactors.persist(self.intermediateRDDStorageLevel)
            # re-calculate the AtA for the dimension K
            kAtA = self.computeAtA(kFactors)
            print "TIME: updating K: {}".format(time.time() - start_update_fact_k)

            # start a timer for the update of the Lagrange multipliers
            start_update_y = time.time()
            # update the Lagrange multipliers
            Yi = self.computeLargMultiplier(Yi, iFactors, Zi)
            Yj = self.computeLargMultiplier(Yj, jFactors, Zj)
            Yk = self.computeLargMultiplier(Yk, kFactors, Zk)
            print "TIME: updating Y: {}".format(time.time() - start_update_y)

            # start a timer for the calculation of the residual tensor
            start_update_pt = time.time()
            # update the residual tensor
            Pt = self.computeResidualTensor(tensorBlocks, iFactors, jFactors, kFactors, iOutBlocks, jOutBlocks, kOutBlocks, sc)
            Pt.persist(self.intermediateRDDStorageLevel)
            print "TIME: updating Pt: {}".format(time.time() - start_update_pt)

            lambdaVals = np.array([1 for _ in range(self.rank)])
            # calculate the error and the fit between the original and estimated tensors
            error, fit = self.calculateError(Pt, normData, lambdaVals)
            errors.append(error)
            fits.append(fit)
            # calculate the change of fit comparing with the previous one
            fitchange = abs(fit - fitold)
            fitold = fit
            print "Iteration {}: error-{}, fit-{} and fitchange-{} with {} seconds".format(iteration, error, fit, fitchange, time.time() - start_inner)
            # stopping criteria
            if fitchange < self.tol:
                break

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

        print "TIME - Completeing the whole iterations: {} sec.".format(time.time() - start_iter)
        print "TIME - Total computation time: {} sec.".format(time.time() - start_program)


if __name__ == "__main__":
    TF = DistTensorCompletionADMM()
    TF.dir_data = "tensor_10x10x10_101.txt"
    TF.run()