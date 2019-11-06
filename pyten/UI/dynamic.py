import pandas as pd
import numpy as np
import pyten.tenclass
import pyten.method


def dynamic(file_name=None, function_name=None, fore__file=None, save__file=None, recover=None, omega=None, r=2,
            tol=1e-8, maxiter=100, mu=0.01, lmbda=0.1, self=None):
    """
    Helios UI for dynamic scenario returns Online decomposition or Recovery Result
    arg can be list, tuple, set, and array with numerical values.
    -----------
    :param file_name: {Default: None}
    :param function_name: Tensor-based Method
    :param fore__file: former file name
    :param save__file: save file name
    :param recover: Input '1' to recover other to decompose.{Default: None}
    :param omega: Index Tensor of Obseved Entries
    :param r: The rank of the Tensor you want to use for  approximation (recover or decompose).{Default: 2}
    :param tol: Tolerance on difference in fit.(Convergence tolerance for both cp(als) or tucker(als).){Default: 1.0e-8}
    :param maxiter: Maximum number of iterations {Default: 100}
    :param mu: OLSGD parameter
    :param lmbda: OLSGD parameter
    :param self: a class of former result (MAST, OLSGD, onlinCP)
    -----------
    :return Ori:   Original Tensor
    :return full:  Full Tensor reconstructed by decomposed matrices
    :return Final: Decomposition Results e.g. Ttensor or Ktensor
    :return Rec:   Recovered Tensor (Completed Tensor)
    -----------
    """

    if file_name is None:
        file_name = raw_input("Please input the file_name of the data:\n")
        print("\n")

    if function_name is None:
        function_name = raw_input("Please choose the method you want to use (Input one number):\n"
                                  " 1. onlineCP(only for decomposition)  2.OLSGD 3.MAST 0.Exit \n")
        print("\n")

    Former_result = '2'
    if self is None and fore__file is None:
        Former_result = raw_input("If there are former decomposition or recovery result (.npy file):\n"
                                  " 1. Yes  2.No 0.Exit \n")
        if Former_result == '1':
            fore__file = raw_input("Please input the file_name of the former result:\n")
            temp = np.load(fore__file)
            self = temp.all()
        elif Former_result == '0':
            print 'Successfully Exit'
            return None, None, None, None

    elif self is None:
        Former_result = '1'
        temp = np.load(fore__file)
        self = temp.all()

    if recover is None:
        if function_name == '1':
            recover = '2'
        else:
            recover = raw_input("If there are missing values in the file? (Input one number)\n"
                                "1. Yes, recover it  2.No, just decompose "
                                "(Missing entries in the original tensor will be replaced by 0) 0.Exit\n")

    # Use pandas package to load data
    dat1 = pd.read_csv(file_name, delimiter=';')

    # Data preprocessing
    # First: create Sptensor
    dat = dat1.values
    sha = dat.shape
    subs = dat[:, range(sha[1] - 1)]
    subs -= 1
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
    output = 1  # An output indicate flag. (recover:1, Decompose: 0)
    if type(omega) != np.ndarray:
        omega = X.data * 0
        if recover == '1':
            omega[lstnan] = 0
            output = 2

    # Choose method to recover or decompose
    if type(function_name) == str:
        n = X.shape[0]
        if function_name == '1' or function_name == 'onlineCP':
            if Former_result == '1':
                self.update(X)
            else:
                self = pyten.method.onlineCP(X, r, tol)
            Final = self.As
            full = pyten.tenclass.Tensor(self.X.data[-n:, ])
            Rec = None

        elif function_name == '2' or function_name == 'OLSGD':
            Omega1 = pyten.tenclass.Tensor(omega)
            if Former_result != '1':
                self = pyten.method.OLSGD(r, mu, lmbda)
            self.update(X, Omega1, mu, lmbda)
            Final = [self.A, self.B, self.C]
            full = pyten.tenclass.Tensor(self.X.data[-n:, :, :])
            Rec = pyten.tenclass.Tensor(self.Rec.data[-n:, :, :])
        elif function_name == '3' or function_name == 'MAST':
            Omega1 = pyten.tenclass.Tensor(omega)
            idx = np.zeros([2 ** X.ndims, X.ndims])
            for i in range(2 ** X.ndims):
                temp = bin(i)[2:]
                ltemp = len(temp)
                for j in range(ltemp):
                    idx[i, X.ndims - j - 1] = int(temp[ltemp - j - 1])

            init = {}
            for i in range(2 ** X.ndims):
                init[str(idx[i, ])] = 0
            omega_block = init.copy()

            if Former_result == '1':

                # Partition
                olddims = self.newdims
                newdims = X.shape

                for i in range(2 ** X.ndims):
                    expr = str()
                    for j in range(X.ndims):
                        if idx[i, j] == 0:
                            expr += '0:int(olddims[' + str(j) + ']),'
                        else:
                            expr += '( int(olddims[' + str(j) + '])+1):X.shape[' + str(j) + '],'
                    expr = expr[:-1]
                    # if not [] in eval(expr):
                    init[str(idx[i, ])] = eval('X.data[' + expr + ']')
                    omega_block[str(idx[i, ])] = eval('Omega1.data[' + expr + ']')

                # MastCore
                self1 = pyten.method.MAST(init, omega_block, self.finalAs, olddims, newdims, idx, r)
                self1.update()

                # FinalDecompositionResult
                Final = []
                for i in range(X.ndims):
                    Final.append(self1.finalAs[i])

                full = pyten.tenclass.Ktensor(np.ones(r), Final)
                full = full.totensor()

                # Reconstruction
                Rec = pyten.tenclass.Tensor((1 - Omega1.data) * full.data + X.data * Omega1.data)

            else:
                NNCP = pyten.method.TNCP(X, Omega1, r, tol, maxiter)
                NNCP.run()
                Final = NNCP.U
                Rec = NNCP.X
                full = NNCP.II.copy()
                for i in range(NNCP.ndims):
                    full = full.ttm(NNCP.U[i], i + 1)
                self1 = pyten.method.MAST(init, omega_block, NNCP.U, np.zeros(3) + X.shape, np.zeros(3) + X.shape)

        elif function_name == '0':
            return 'Successfully Exit'
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
        # newvals.append(list(tempvals(idx)));
    df.to_csv(newfilename, sep=';', index=0)

    if save__file is None:
        SaveOption = raw_input("If you want to save the result into .npy file):\n"
                               " 1. Yes  2.No  0.Exit \n")
        if SaveOption == '1':
            save__file = raw_input("Please input the address and fileName to save the result: (end in '.npy')\n")
            if function_name == '3' or function_name == 'MAST':
                np.save(save__file, self1)
            else:
                np.save(save__file, self)

    # Return result
    return Ori, full, Final, Rec
