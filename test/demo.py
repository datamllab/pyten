"""
Some Files For Testing
./test/syntensor.csv : A tensor of size (3, 2, 4) with missing entries.
./test/complete_syntensor.csv : The ground truth tensor of file './test/1.csv'.
./test/dedicom.csv : the symmetric tensor for testing DEDICOM method
./test/test_parafac2_1.csv; ./test/test_parafac2_2.csv; ./test/test_parafac2_3.csv
                   : multiset data (3 slices) for testing PARAFAC2 method
./test/aux_CM.csv : The auxiliary matrix of './test/1.csv' coupled on mode 1.
./test/aux_1.csv : The similarity matrix of './test/1.csv' coupled on mode 1.
./test/aux_2.csv : The similarity matrix of './test/1.csv' coupled on mode 2.
./test/aux_3.csv : The similarity matrix of './test/1.csv' coupled on mode 3.
./test/mast1.csv : A tensor of size (2, 2, 2) with missing entries.
./test/mast2.csv : The incremental part of a tensor from (2,2,2) to (3,2,4) with missing entries.
./test/testImg.png : An RGB image.
"""

""" Quick Start"""
import pyten

# Quick Start (without prior input)
[OriTensor, DeTensor, TenClass, RecTensor] = pyten.UI.helios()

# Results
print OriTensor  # Original Tensor (For scalable methods it is None)
print DeTensor.data  # Full Tensor reconstructed by decomposed matrices
print TenClass  # Final Decomposition Results e.g. Ttensor or Ktensor
print RecTensor.data  # Recovered Tensor (Completed Tensor)

""" Four UI Functions For Four Scenarios"""
import pyten

# Scenario 1: Basic Tensor completion or decomposition.
[OriTensor, DeTensor, TenClass, RecTensor] = pyten.UI.basic()

# Scenario 2: Tensor completion or decomposition with auxiliary information
[OriTensor, DeTensor, TenClass, RecTensor] = pyten.UI.auxiliary()

# Scenario 3: Dynamic/Online/Streaming Tensor completion or decomposition
[OriTensor, DeTensor, TenClass, RecTensor] = pyten.UI.dynamic()

# Scenario 4: Scalable Tensor completion or decomposition
[OriTensor, DeTensor, TenClass, RecTensor] = pyten.UI.scalable()

""" Usage of Main Functions """
'''Function: create'''
# 1. Create Tensor Completion Problem
from pyten.tools import create  # Import the problem creation function

problem = 'basic'  # Problem Definition
siz = [20, 20, 20]  # Size of the Created Synthetic Tensor
r = [4, 4, 4]  # Rank of the Created Synthetic Tensor
miss = 0.8  # Missing Percentage
tp = 'CP'  # Solution Format (Creating Method) of the Created Synthetic Tensor
[X1, Omega1, sol1] = create(problem, siz, r, miss, tp)
# X1: The created tensor object.
# Omega1: An 0-1 'numpy.ndarray' which represent the missing data. (0 for missing)
# sol1: Solution of the created problem.

'''Scenario 1: Basic Tensor Completion/Decomposition'''
# 1. Solve Synthetic Completion Problem
from pyten.tools import create  # Import the problem creation function

problem = 'basic'  # Define Problem As Basic Tensor Completion Problem
siz = [20, 20, 20]  # Size of the Created Synthetic Tensor
r = [4, 4, 4]  # Rank of the Created Synthetic Tensor
miss = 0.8  # Missing Percentage
tp = 'CP'  # Define Solution Format of the Created Synthetic Tensor As 'CP decomposition'
[X1, Omega1, sol1] = create(problem, siz, r, miss, tp)

# Basic Tensor Completion with methods: CP-ALS,Tucker-ALS, FaLRTC, SiLRTC, HaLRTC, TNCP
from pyten.method import *

r = 4  # Rank for CP-based methods
R = [4, 4, 4]  # Rank for tucker-based methods
# CP-ALS
[T1, rX1] = cp_als(X1, r, Omega1)  # if no missing data just omit Omega1 by using [T1,rX1]=cp_als.cp_als(X1,r)
# print sol1.totensor().data
# print rX1.data

# Tucker-ALS
[T2, rX2] = tucker_als(X1, R, Omega1)  # if no missing data just omit Omega1
# FalRTC, SiLRTC, HaLRTC
rX3 = falrtc(X1, Omega1)
rX4 = silrtc(X1, Omega1)
rX5 = halrtc(X1, Omega1)
# TNCP
self1 = TNCP(X1, Omega1, rank=r)
self1.run()

# Error Testing
from pyten.tools import tenerror

realX = sol1.totensor()
[Err1, ReErr11, ReErr21] = tenerror(rX1, realX, Omega1)
[Err2, ReErr12, ReErr22] = tenerror(rX2, realX, Omega1)
[Err3, ReErr13, ReErr23] = tenerror(rX3, realX, Omega1)
[Err4, ReErr14, ReErr24] = tenerror(rX4, realX, Omega1)
[Err5, ReErr15, ReErr25] = tenerror(rX5, realX, Omega1)
[Err6, ReErr16, ReErr26] = tenerror(self1.X, realX, Omega1)
print '\n', 'The Relative Error of the Six Methods are:', ReErr21, ReErr22, ReErr23, ReErr24, ReErr25, ReErr26

# 2. Real Problem - Image Recovery
import matplotlib.image as mpimg  # Use it to load image
import numpy as np

lena = mpimg.imread("./test/testImg.png")
im = np.double(np.uint8(lena * 255))
im = im[0:50, 0:50, 0:3]

from pyten.tenclass import Tensor  # Use it to construct Tensor object

X1 = Tensor(im)  # Construct Image Tensor to be Completed
X0 = X1.data.copy()
X0 = Tensor(X0)  # Save the Ground Truth
Omega1 = (im < 100) * 1.0  # Missing index Tensor
X1.data[Omega1 == 0] = 0

# Basic Tensor Completion with methods: CP-ALS, Tucker-ALS, FaLRTC, SiLRTC, HaLRTC, TNCP
from pyten.method import *

r = 10
R = [10, 10, 3]  # Rank for tucker-based methods
[T1, rX1] = cp_als(X1, r, Omega1, maxiter=1000, printitn=100)
[T2, rX2] = tucker_als(X1, R, Omega1, max_iter=100, printitn=100)
alpha = np.array([1.0, 1.0, 1e-3])
alpha = alpha / sum(alpha)
rX3 = falrtc(X1, Omega1, max_iter=100, alpha=alpha)
rX4 = silrtc(X1, Omega1, max_iter=100, alpha=alpha)
rX5 = halrtc(X1, Omega1, max_iter=100, alpha=alpha)
self1 = TNCP(X1, Omega1, rank=r)
self1.run()

# Error Testing
from pyten.tools import tenerror

realX = X0
[Err1, ReErr11, ReErr21] = tenerror(rX1, realX, Omega1)
[Err2, ReErr12, ReErr22] = tenerror(rX2, realX, Omega1)
[Err3, ReErr13, ReErr23] = tenerror(rX3, realX, Omega1)
[Err4, ReErr14, ReErr24] = tenerror(rX4, realX, Omega1)
[Err5, ReErr15, ReErr25] = tenerror(rX5, realX, Omega1)
[Err6, ReErr16, ReErr26] = tenerror(self1.X, realX, Omega1)
print '\n', 'The Relative Error of the Six Methods are:', ReErr21, ReErr22, ReErr23, ReErr24, ReErr25, ReErr26

'''Scenario 2: Tensor Completion/Decomposition with Auxiliary Information'''
# 1. Use  AirCP Method to solve Tensor Completion With Auxiliary Similarity Matrices
from pyten.method import AirCP  # Import AirCP
from pyten.tools import create  # Import the problem creation function

problem = 'auxiliary'  # Define Problem As Basic Tensor Completion Problem
siz = [20, 20, 20]  # Size of the Created Synthetic Tensor
r = [4, 4, 4]  # Rank of the Created Synthetic Tensor
miss = 0.8  # Missing Percentage
tp = 'sim'  # Define Auxiliary Information As 'Similarity Matrices'
# Construct Similarity Matrices (if 'None', then it will use the default Similarity Matrices)
# aux = [np.diag(np.ones(siz[n]-1), -1)+np.diag(np.ones(siz[n]-1), 1) for n in range(dims)]
aux = None
[X1, Omega1, sol1, sim_matrices] = create(problem, siz, r, miss, tp, aux=aux)

self = AirCP(X1, Omega1, r, sim_mats=sim_matrices)
self.run()
# self_no_aux = AirCP(X1, Omega1, r)
# self_no_aux.run()

# Error Testing
from pyten.tools import tenerror

realX = sol1.totensor()
[Err1, ReErr11, ReErr21] = tenerror(self.X, realX, Omega1)
# [Err2, ReErr12, ReErr22] = tenerror(self_no_aux.X, realX, Omega1)
print '\n', 'The Relative Error of the Two Methods are:', ReErr11


# 2. Use  CMTF Method to solve Tensor Completion With Coupled Matrices
from pyten.method import cmtf
from pyten.tools import create  # Import the problem creation function
import numpy as np

problem = 'auxiliary'  # Define Problem As Basic Tensor Completion Problem
siz = [20, 20, 20]  # Size of the Created Synthetic Tensor
r = [4, 4, 4]  # Rank of the Created Synthetic Tensor
miss = 0.8  # Missing Percentage
tp = 'couple'  # Define Auxiliary Information As 'Similarity Matrices'
# Construct Similarity Matrices (if 'None', then it will use the default Similarity Matrices)
dims = 3
[X1, Omega1, sol1, coupled_matrices] = create(problem, siz, r, miss, tp)

[T1, Rec1, V1] = cmtf(X1, coupled_matrices, [1, 2, 3], r, Omega1, maxiter=500)
# [T2, Rec2, V2] = cmtf(X1, None, None, r, Omega1, maxiter=500)
fit_coupled_matrices_1 = [np.dot(T1.Us[n], V1[n].T) for n in range(dims)]

# Error Testing
from pyten.tools import tenerror

realX = sol1.totensor()
[Err1, ReErr11, ReErr21] = tenerror(Rec1, realX, Omega1)
# [Err1, ReErr12, ReErr22] = tenerror(Rec2, realX, Omega1)
print '\n', 'The Relative Error of the Two Methods are:', ReErr11


'''Scenario 3: Dynamic Tensor Decomposition/Completion'''
from pyten.method import onlineCP, OLSGD
from pyten.tools import create  # Import the problem creation function
from pyten.tools import tenerror
import numpy as np

problem = 'dynamic'  # Define Problem As Dynamic Tensor Completion Problem
time_steps = 10  # Define the Number of Total Time Steps
siz = np.array([[1, 50, 50] for t in range(time_steps)])
r = [4, 4, 4]  # Rank of the Created Synthetic Tensor
miss = 0.8  # Missing Percentage
# Create a Dynmaic Tensor Completion Problem
[X1, Omega1, sol1, siz, time_steps] = create(problem, siz, r, miss, timestep=time_steps)

for t in range(time_steps):
    if t == 0:  # Initial Step
        print('Initial Step\n')
        self1 = OLSGD(rank=r, mu=0.01, lmbda=0.1)  # OLSGD assume time is the first mode.
        self1.update(X1[t], Omega1[t])  # Complete the initial tensor using OLSGD method.
        # onlineCP assume time is the last mode.
        self = onlineCP(X1[t].permute([1, 2, 0]), rank=r, tol=1e-8, printitn=0)  # Just decompose without completion using onlineCP
    else:
        if t==1:
            print('Update Step\n')
        self1.update(X1[t], Omega1[t])  # Update Decomposition as well as Completion using OLSGD.
        self.update(X1[t].permute([1, 2, 0]))  # Update Decomposition of onlineCP.
    # Test Current Step OLSGD Completion Error
    realX = sol1[t].totensor()
    [Err1, ReErr11, ReErr21] = tenerror(self1.recx, realX, Omega1[t])
    print 'OLSGD Recover Error at Current Step:', Err1, ReErr11, ReErr21



'''Scenario 4: Scalable Tensor Completion/Decomposition'''
# 1. Solve Synthetic Completion Problem
from pyten.tools import create  # Import the problem creation function

problem = 'basic'  # Define Problem As Basic Tensor Completion Problem
siz = [20, 20, 20]  # Size of the Created Synthetic Tensor
r = [4, 4, 4]  # Rank of the Created Synthetic Tensor
miss = 0.8  # Missing Percentage
tp = 'CP'  # Define Solution Format of the Created Synthetic Tensor As 'CP decomposition'
[X1, Omega1, sol1] = create(problem, siz, r, miss, tp)

# Basic Tensor Completion with methods: CP-ALS,Tucker-ALS, FaLRTC, SiLRTC, HaLRTC, TNCP
from pyten.method import *

r = 4  # Rank for CP-based methods
R = [4, 4, 4]  # Rank for tucker-based methods

# Distributed CP_ALS
self0 = TensorDecompositionALS()
self0.dir_data = X1  # Could also be '.csv' or '.txt' format, e.g. 'test/syntensor.csv', 'test/tensor_10x10x10_101.txt'
self0.rank = r
self0.run()

# DistTensorADMM
self1 = DistTensorADMM()
self1.dir_data = X1  # Could also be '.csv' or '.txt' format, e.g. 'test/syntensor.csv', 'test/tensor_10x10x10_101.txt'
self1.rank = r
self1.run()

# DistTensorCompletionADMM
self2 = DistTensorCompletionADMM()
self2.dir_data = X1  # Could also be '.csv' or '.txt' format, e.g. 'test/syntensor.csv', 'test/tensor_10x10x10_101.txt'
self2.rank = r
self2.run()

# Error Testing
from pyten.tools import tenerror
realX = sol1.totensor()
[Err1, ReErr11, ReErr21] = tenerror(self0.ktensor.totensor(), realX, Omega1)
[Err2, ReErr21, ReErr22] = tenerror(self1.ktensor.totensor(), realX, Omega1)
RecTensor = self2.ktensor.totensor().data*(1-Omega1)+X1.data*Omega1
[Err3, ReErr31, ReErr32] = tenerror(RecTensor, realX, Omega1)
print '\n', 'The Relative Error of the Three Distributed Methods are:', ReErr21, ReErr22, ReErr32





'''Scenario *: Other Decomposition Method'''

# 1. PARAFAC2
# Create multiset
from pyten.method import parafac2  # Import the problem creation function
from pyten.tools import create  # Import the problem creation function

problem = 'basic'  # Define Problem As Basic Tensor Completion Problem
siz = [30, 50, 40]  # Size of the Created Synthetic Tensor
r = 5  # Rank of the Created Synthetic Tensor
miss = 0  # Missing Percentage
tp = 'Parafac2'  # Define Solution Format of the Created Synthetic Tensor As 'CP decomposition'
[X1, Omega1, sol1] = create(problem, siz, r, miss, tp, share_mode_size=10)
self = parafac2.PARAFAC2(X1, r, printitn=100, maxiter=1000, tol=1e-7)
self.run()

# 2. Dedicom (Under Construction)
# from pyten.method import dedicom
# from pyten.tools import create  # Import the problem creation function

# problem = 'basic'  # Define Problem As Basic Tensor Completion Problem
# siz = [50, 50, 50]  # Size of the Created Synthetic Tensor
# r = 2  # Rank of the Created Synthetic Tensor
# miss = 0  # Missing Percentage
# tp = 'Dedicom'  # Define Solution Format of the Created Synthetic Tensor As 'CP decomposition'
# [X1, Omega1, sol1] = create(problem, siz, r, miss, tp)

# DEDICOM
# self = dedicom.DEDICOM(X1, r, printitn=0)
# self.run()
