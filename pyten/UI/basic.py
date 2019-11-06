import pandas as pd
import numpy as np

import pyten.tenclass
import pyten.method


def basic(file_name=None, function_name=None, recover=None, omega=None, r=2, tol=1e-8, maxiter=100, init='random',
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
        file_name = raw_input("Please input the file_name of the data: (if it is multiset data, please seperate them"
                              " to different files and input the first one)\n")
        print("\n")

    if function_name is None:
        function_name = raw_input("Please choose the method you want to use to recover data(Input one number):\n"
                                  " 1. Tucker(ALS)  2.CP(ALS) 3.TNCP(Trace Norm + ADMM, Only For Recovery) "
                                  "4.SiLRTC(Only For Recovery) 5.FaLRTC(Only For Recovery)"
                                  " 6.HaLRTC(Only For Recovery) 7. PARAFAC2 8.DEDICOM 0.Exit \n")
        print("\n")
    if recover is None:
        recover = raw_input("If there are missing values in the file? (Input one number)\n"
                            "1.Yes, recover it 2.No, just decompose (Missing entries in the original tensor will be replaced by 0) 0.Exit\n")

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

    # Construct omega
    output = 1  # An output indicate flag. (Decompose: 1, Recover:2)
    if type(omega) != np.ndarray:
        # if True in lstnan:
        omega = X.data * 0 + 1
        omega[lstnan] = 0
        if recover == '1':
            output = 2

    # Choose method to recover or decompose
    if type(function_name) == str:
        if function_name == '1' or function_name == 'tucker_als':
            [Final, Rec] = pyten.method.tucker_als(X, r, omega, tol, maxiter, init, printitn)
            full = Final.totensor()
        elif function_name == '2' or function_name == 'cp_als':
            [Final, Rec] = pyten.method.cp_als(X, r, omega, tol, maxiter, init, printitn)
            full = Final.totensor()
        elif function_name == '3' or function_name == 'TNCP':
            Omega1 = pyten.tenclass.Tensor(omega)
            NNCP = pyten.method.TNCP(X, Omega1, r, tol, maxiter)
            NNCP.run()
            Final = NNCP.U
            Rec = NNCP.X
            full = NNCP.II.copy()
            for i in range(NNCP.ndims):
                full = full.ttm(NNCP.U[i], i + 1)
        elif function_name == '4' or function_name == 'SiLRTC':
            Rec = pyten.method.silrtc(X, omega, max_iter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif function_name == '5' or function_name == 'FaLRTC':
            Rec = pyten.method.falrtc(X, omega, max_iter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif function_name == '6' or function_name == 'HaLRTC':
            Rec = pyten.method.halrtc(X, omega, max_iter=maxiter, printitn=printitn)
            full = None
            Final = None
        elif function_name == '7' or function_name == 'PARAFAC2':
            X1 = [X.data]
            multi = raw_input("Please input how many other multiset files you want to couple with the first one "
                              "(Input 'None' if no other info.) \n")
            if multi != 'None':
                for i in range(int(multi)):
                    FileName2 = raw_input(
                        "Please input the file_name of the " + str(i + 2) + " slice of the multiset data:\n")
                    if FileName2 != 'None':
                        dat2 = pd.read_csv(FileName2, delimiter=';')
                        # Data preprocessing
                        # First: create Sptensor
                        dat2v = dat2.values
                        sha2v = dat2v.shape
                        subs2v = dat2v[:, range(sha2v[1] - 1)]
                        subs2v = subs2v - 1
                        vals2 = dat2v[:, sha2v[1] - 1]
                        vals2 = vals2.reshape(len(vals2), 1)
                        siz2 = np.max(subs2v, 0)
                        siz2 = np.int32(siz2 + 1)
                        X2 = pyten.tenclass.Sptensor(subs2v, vals2, siz2)

                        # Second: create Tensor object and find missing data
                        X2 = X2.totensor()
                        # lstnan = np.isnan(X2.data)
                        X2.data = np.nan_to_num(X2.data)
                        X1.append(X2.data)
                print (X1[0].shape)
                print (X1[1].shape)
                print (X1[2].shape)
                print("\n")
            parafac = pyten.method.PARAFAC2(X1, r, maxiter=maxiter, printitn=printitn)
            parafac.run()
            Ori = parafac.X
            Final = parafac
            Rec = None
            full = parafac.fit
        elif function_name == '8' or function_name == 'DEDICOM':
            dedicom = pyten.method.DEDICOM(X, r, maxiter=maxiter, printitn=printitn)
            dedicom.run()
            Final = dedicom
            Rec = None
            full = dedicom.fit
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
        if function_name == '7':
            # newsubs = []
            # tempvals = []
            # for i in range(int(multi)+1):
            # temp = pyten.tenclass.Tensor(full[i])
            # newsubs = newsubs.append(temp.tosptensor().subs)
            # tempvals = tempvals.append(temp.tosptensor().vals)
            # newfilename = file_name[:-4] + '_Decomposite' + file_name[-4:]
            print "\n" + "The original Multiset Data is: "
            print Ori
            print "\n" + "The Decomposed Result is: "
            print Final
        else:
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
    if function_name != '7' and function_name != 'PARAFAC2':
        df = dat1
        for i in range(nv):
            pos = map(sum, newsubs == subs[i])
            idx = pos.index(nd)
            temp = tempvals[idx]
            df.iloc[i, nd] = temp[0]
            # newvals.append(list(tempvals(idx)));
        df.to_csv(newfilename, sep=';', index=0)

    # Return result
    return Ori, full, Final, Rec
