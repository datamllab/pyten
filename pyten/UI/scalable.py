import pandas as pd
import numpy as np

import pyten.tenclass
import pyten.method
import pyten.tools


def scalable(file_name=None, function_name=None, recover=None, omega=None, r=2, tol=1e-8, maxiter=100, init='random',
          printitn=0):
    """
    Helios1 API returns CP_ALS, TUCKER_ALS, or NNCP decomposition or Recovery Result
    arg can be list, tuple, set, and array with numerical values.
    -----------
    :param file_name: {Default: None}
    :param function_name: Tensor-based Method
    :param recover: Input '1' to recover other to decompose.{Default: None}
    :param omega: Index Tensor of Obseved Entries
    :param r: The rank of the Tensor you want to use for  approximation (recover or decompose).{Default: 2}
    :param tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-4}
    :param maxiter: Maximum number of iterations {Default: 50}
    :param init: Initial guess 'random'|'nvecs'|'eigs'. {Default 'random'}
    :param printitn: Print fit every n iterations; 0 for no printing.
    -----------
    :return Ori:   Original Tensor
    :return full:  Full Tensor reconstructed by decomposed matrices
    :return Final: Decomposition Results e.g. Ttensor or Ktensor
    :return Rec:   Recovered Tensor (Completed Tensor)
    -----------
    """

    # User Interface
    if file_name is None:
        file_name = raw_input("Please input the file_name of the data: \n")
        print("\n")

    if function_name is None:
        function_name = raw_input("Please choose the method you want to use to recover data(Input one number):\n"
                                  " 1. Distributed CP(ALS)  2.Distributed CP(ADMM) 3. DisTenC  0.Exit \n")
        print("\n")
    #if recover is None:
    #    recover = raw_input("If there are missing values in the file? (Input one number)\n"
    #                        "1.Yes, recover it 2.No, just decompose (Missing entries in the original tensor will be replaced by 0) 0.Exit\n")

    # Use pandas package to load data
##    if file_name[-3:] == 'csv':
#    dat1 = pd.read_csv(file_name, delimiter=';')

    # Data preprocessing
    # First: create Sptensor
#    dat = dat1.values
#    sha = dat.shape
#    subs = dat[:, range(sha[1] - 1)]
#    subs = subs - 1
#    vals = dat[:, sha[1] - 1]
#    vals = vals.reshape(len(vals), 1)
#    siz = np.max(subs, 0)
#    siz = np.int32(siz + 1)
#    X1 = pyten.tenclass.Sptensor(subs, vals, siz)

    # Second: create Tensor object and find missing data
#    X = X1.totensor()
#    Ori = X.data
#    lstnan = np.isnan(X.data)
#    X.data = np.nan_to_num(X.data)

    # Construct omega
    #output = 1  # An output indicate flag. (Decompose: 1, Recover:2)
    Ori = None
    #if type(omega) != np.ndarray:
    #    # if True in lstnan:
    #    omega = X.data * 0 + 1
    #    omega[lstnan] = 0
    #    if recover == '1':
    #        output = 2

    # Choose method to recover or decompose
    if type(function_name) == str:
        if function_name == '1' or function_name == 'D_cp_als':
            Dals = pyten.method.TensorDecompositionALS()
            Dals.dir_data = file_name
            Dals.rank = r
            Dals.run()
            Dals.maxIter = maxiter
            Dals.tol = tol

            ######
            Final = Dals.ktensor
            Rec = None
            full = Final.totensor()
            ######

        elif function_name == '2' or function_name == 'D_ADMM':
            Dadmm = pyten.method.DistTensorADMM()
            Dadmm.dir_data = file_name
            Dadmm.rank = r
            Dadmm.run()
            Dadmm.maxIter = maxiter
            Dadmm.tol = tol

            ######
            Final = Dadmm.ktensor
            Rec = None
            full = Final.totensor()
            ######

        elif function_name == '3' or function_name == 'D_ADMM_C':
            DadmmC = pyten.method.DistTensorCompletionADMM()
            DadmmC.dir_data = file_name
            DadmmC.rank = r
            DadmmC.run()
            DadmmC.maxIter = maxiter
            DadmmC.tol = tol

            ######
            Final = DadmmC.ktensor
            #Rec = Final.totensor().data * omega + X.data * (1 - omega)
            full = Final.totensor()
            Rec = full
            ######

        elif function_name == '0':
            print 'Successfully Exit'
            return None, None, None, None
        else:
            raise ValueError('No Such Method')

    else:
        raise TypeError('No Such Method')

    # Output Result
    # [nv, nd] = subs.shape
    if function_name == 1 or function_name == 2:
            newsubs = full.tosptensor().subs
            tempvals = full.tosptensor().vals
            newfilename = file_name[:-4] + '_Decomposite' + file_name[-4:]
            #print "\n" + "The original Tensor is: "
            #print X1
            print "\n" + "The Decomposed Result is: "
            print Final
    else:
        newsubs = Rec.tosptensor().subs
        tempvals = Rec.tosptensor().vals
        newfilename = file_name[:-4] + '_Recover' + file_name[-4:]
        #print "\n" + "The original Tensor is: "
        #print Ori
        print "\n" + "The Recovered Tensor is: "
        print Rec.data

    # Return result
    return Ori, full, Final, Rec
