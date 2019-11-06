import pandas as pd
import numpy as np
import scipy

import pyten.tenclass
import pyten.method


def auxiliary(file_name=None, function_name=None, aux_mode=None, aux_file=None, recover=None, omega=None, r=2, tol=1e-8,
              maxiter=100, init='random', printitn=0):
    """
    Helios API returns decomposition or Recovery with Auxiliary Result
    arg can be list, tuple, set, and array with numerical values.
    -----------
    :param file_name: {Default: None}
    :param function_name: Tensor-based Method
    :param aux_mode: idex of modes that contains auxiliary information (either similarity info. or coupled matrices)
    :param aux_file: name of auxiliary files (contains either similarity matrices or coupled matrices)
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
        file_name = raw_input("Please input the file_name of the Tensor data:\n")
        print("\n")

    # Use pandas package to load data
    dat1 = pd.read_csv(file_name, delimiter=';')

    # Data preprocessing
    # First: create Sptensor
    dat = dat1.values
    sha = dat.shape
    subs = dat[:, range(sha[1] - 1)]
    subs = subs - 1
    vals = dat[:, sha[1] - 1]
    vals = vals.reshape(len(vals), 1)
    siz = np.max(subs, 0)
    siz = np.int32(siz + 1)
    X1 = pyten.tenclass.Sptensor(subs, vals, siz)

    # Second: create Tensor object and find missing data
    X = X1.totensor()
    Ori = X.data
    lstnan = np.isnan(X.data)
    X.data = np.nan_to_num(X.data)

    if function_name is None:
        function_name = raw_input("Please choose the method you want to use to recover data(Input one number):\n"
                                  " 1. AirCP  2.CMTF 0.Exit \n")
        print("\n")

    if function_name == '1' or function_name == 'AirCP':
        simMats = np.array([np.identity(X.shape[i]) for i in range(X.ndims)])
        if aux_mode is None:
            aux_mode = raw_input(
                "Please input all the modes that have Auxiliary Similarity Matrix "
                "(separate with space. Input 'None' if no auxiliary info.)\n")
            if aux_mode != 'None':
                for i in range((len(aux_mode) + 1) / 2):
                    Mode = int(aux_mode[i * 2])
                    FileName2 = raw_input(
                        "Please input the file_name of the Auxiliary Similarity Matrix of Mode " + str(Mode) + " :\n")
                    if FileName2 != 'None':
                        dat2 = pd.read_csv(FileName2, delimiter=';', header=None)
                        # Data preprocessing
                        Mat_dat = dat2.values
                        if Mat_dat.shape == (X1.shape[Mode - 1], X1.shape[Mode - 1]):
                            simMats[Mode - 1] = Mat_dat
                        else:
                            print('Wrong Size of Auxiliary Matrix.\n')
                print("\n")
        else:
            for i in range((len(aux_mode) + 1) / 2):
                Mode = int(aux_mode[i * 2])
                FileName2 = aux_file[i]
                if FileName2 != 'None':
                    dat2 = pd.read_csv(FileName2, delimiter=';', header=None)
                    # Data preprocessing
                    Mat_dat = dat2.values
                    if Mat_dat.shape == (X1.shape[Mode - 1], X1.shape[Mode - 1]):
                        simMats[Mode - 1] = Mat_dat
                    else:
                        print('Wrong Size of Auxiliary Matrix.\n')

    elif function_name == '2' or function_name == 'CMTF':
        CM = None
        Y = None
        if aux_mode is None:
            aux_mode = raw_input(
                "Please input all the modes that have Coupled Matrix (separate with space. "
                "Input 'None' if no coupled matrices. Allow Multiple Coupled Matrices for One Mode)\n")
            if aux_mode != 'None':
                for i in range((len(aux_mode) + 1) / 2):
                    Mode = int(aux_mode[i * 2])
                    FileName2 = raw_input("Please input the file_name of the Coupled Matrix of Mode " + str(
                        Mode) + " (Input 'None' if no auxiliary info):\n")
                    print("\n")
                    if FileName2 != 'None':
                        dat2 = pd.read_csv(FileName2, delimiter=';')
                        Mat_dat = dat2.values
                        Mat_subs = Mat_dat[:, range(2)]
                        Mat_subs = Mat_subs - 1
                        Mat_vals = Mat_dat[:, 2]
                        Mat_siz = np.max(Mat_subs, 0)
                        Mat_siz = Mat_siz + 1
                        X2 = scipy.sparse.coo_matrix((Mat_vals, (Mat_subs[:, 0], Mat_subs[:, 1])),
                                                     shape=(Mat_siz[0], Mat_siz[1]))
                        if CM is None:
                            CM = Mode
                            Y = X2.toarray()
                        else:
                            CM = [CM, Mode]
                            Y = [Y, X2.toarray()]
        else:
            for i in range((len(aux_mode) + 1) / 2):
                Mode = int(aux_mode[i * 2])
                FileName2 = aux_file[i]
                print("\n")
                if FileName2 != 'None':
                    dat2 = pd.read_csv(FileName2, delimiter=';')
                    Mat_dat = dat2.values
                    Mat_subs = Mat_dat[:, range(2)]
                    Mat_subs = Mat_subs - 1
                    Mat_vals = Mat_dat[:, 2]
                    Mat_siz = np.max(Mat_subs, 0)
                    Mat_siz = Mat_siz + 1
                    X2 = scipy.sparse.coo_matrix((Mat_vals, (Mat_subs[:, 0], Mat_subs[:, 1])),
                                                 shape=(Mat_siz[0], Mat_siz[1]))
                    if CM is None:
                        CM = Mode
                        Y = X2.toarray()
                    else:
                        CM = [CM, Mode]
                        Y = [Y, X2.toarray()]

    elif function_name == '0':
        print 'Successfully Exit'
        return None, None, None, None
    else:
        raise ValueError('No Such Method')

    if recover is None:
        recover = raw_input("If there are missing values in the file? (Input one number)\n 1. Yes, recover it  "
                            "2.No, just decompose (Missing entries in the original tensor will be replaced by 0) 0.Exit\n")

    # Construct omega
    output = 1  # An output indicate flag. (recover:1, Decompose: 0)
    if type(omega) != np.ndarray:
        omega = X.data * 0 + 1
        if recover == '1':
            omega[lstnan] = 0
            output = 2

    # Choose method to recover or decompose
    if type(function_name) == str:
        if function_name == '1' or function_name == 'AirCP':
            Omega1 = pyten.tenclass.Tensor(omega)
            self = pyten.method.AirCP(X, Omega1, r, tol, maxiter, simMats=simMats)
            self.run()
            Final = self.U
            Rec = self.X
            full = self.II.copy()
            for i in range(self.ndims):
                full = full.ttm(self.U[i], i + 1)

        elif function_name == '2' or function_name == 'CMTF':
            [Final, Rec, V] = pyten.method.cmtf(X, Y, CM, r, omega, tol, maxiter, init, printitn)
            full = Final.totensor()

        elif function_name == '0':
            print 'Successfully Exit'
            return None, None, None, None
        else:
            raise ValueError('No Such Method')

    else:
        raise TypeError('No Such Method')

    # Output Result
    [nv, nd] = subs.shape
    if output == 1:
        newsubs = full.tosptensor().subs
        tempvals = full.tosptensor().vals
        newfilename = file_name[:-4] + '_Decomposite' + file_name[-4:]
        print "\n" + "The original Tensor is: "
        print Ori
        print "\n" + "The Decomposed Result is: "
        print Final
    else:
        newsubs = Rec.tosptensor().subs
        tempvals = Rec.tosptensor().vals
        newfilename = file_name[:-4] + '_Recover' + file_name[-4:]
        print "\n" + "The original Tensor is: "
        print Ori
        print "\n" + "The Recovered Tensor is: "
        print Rec.data

    # Reconstruct
    df = dat1
    for i in range(nv):
        pos = map(sum, newsubs == subs[i])
        idx = pos.index(nd)
        temp = tempvals[idx]
        df.iloc[i, nd] = temp[0]
        # newvals.append(list(tempvals(idx)))
    df.to_csv(newfilename, sep=';', index=0)

    # Return result
    return Ori, full, Final, Rec
