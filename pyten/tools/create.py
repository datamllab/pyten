import numpy as np
import pyten.tenclass


def create(problem='basic', siz=None, r=2, miss=0, tp='CP', aux=None, timestep=5, share_mode_size=10):
    """
    A function to create a Tensor decomposition or completion problem in different situations.
    Input:
    :param problem: Tensor completion/decomposition problem (basic,auxiliary,dynamic)
    :param siz: size of Tensor
    :param r: rank of Tensor
    :param miss: missing percentage of data
    :param tp: type of expect solution. (Tucker, CP; sim, couple)
    :param aux: a list of auxiliary similarity matrices or coupled matrices
    :param timestep: timesteps for dynamic situation
    :param share_mode_size: the size of the shared mode for parafac2 method
    Output:
    ten: generated Tensor;
    omega: index tensor of observed entries (0: Miss; 1:Exist);
    sol: solution.
    """

    if siz is None:
        if problem == 'dynamic':
            siz = np.ones([timestep, 3]) * 20
        else:
            siz = [20, 20, 20]

    if problem == 'dynamic':
        dims = len(siz[0])
    else:
        dims = len(siz)

    if type(r) == int:
        r = np.zeros(dims, dtype='int') + r

    if problem == 'basic':
        if tp == 'Tucker':
            # Solution Decomposition Matrices
            u = [np.random.random([siz[n], r[n]]) for n in range(dims)]
            core = pyten.tenclass.Tensor(np.random.random(r))
            sol = pyten.tenclass.Ttensor(core, u)
        elif tp == 'CP':
            # Solution Decomposition Matrices
            u = [np.random.random([siz[n], r[n]]) for n in range(dims)]
            syn_lambda = np.ones(r[0])
            sol = pyten.tenclass.Ktensor(syn_lambda, u)
        elif tp == 'Dedicom':
            # Solution Decomposition Matrices
            A = np.random.rand(siz[0], r[0])
            R = np.random.rand(r[0], r[0])
            D = np.zeros([r[0], r[0], siz[2]])
            diagonal = np.random.rand(r[0], siz[2])
            for i in range(siz[2]):
                D[:, :, i] = np.diag(diagonal[:, i])
            sol = np.zeros(siz)
            for i in range(siz[2]):
                temp_ad = A.dot(D[:, :, i])
                temp_da = D[:, :, i].dot(A.T)
                sol[:, :, i] = temp_ad.dot(R).dot(temp_da)
        elif tp == 'Parafac2':
            # Solution Decomposition Matrices
            U = [np.random.random([siz[n], r[n]]) for n in range(dims)]
            H = np.random.rand(r[0], r[0])
            S = np.random.rand(r[0], r[0], dims)
            for k in range(dims):
                S[:, :, k] = np.random.rand(r[0])
            V = np.random.rand(share_mode_size, r[0])
            sol = range(dims)
            for k in range(dims):
                sol[k] = U[k].dot(H).dot(S[:, :, k]).dot(V.T)
        else:
            raise ValueError('No Such Method.')

    elif problem == 'auxiliary':
        if tp == 'sim':
            if aux is None:
                aux = [np.diag(np.ones(siz[n] - 1), -1) + np.diag(np.ones(siz[n] - 1), 1) for n in range(dims)]
                epsilon = [np.random.random([r[n], 2]) for n in range(dims)]
                # Solution Decomposition Matrices
                tmp = []
                for n in range(dims):
                    tmp.append(np.array([range(1, siz[n] + 1), np.ones(siz[n])]).T)
                u = [np.dot(tmp[n], epsilon[n].T) for n in range(dims)]
            else:
                # Solution Decomposition Matrices
                u = [np.random.multivariate_normal(np.zeros(siz[n]), aux[n], r[n]).T for n in range(dims)]
            syn_lambda = np.ones(r[0])
            sol = pyten.tenclass.Ktensor(syn_lambda, u)
        elif tp == 'couple':
            u = [np.random.random([siz[n], r[n]]) for n in range(dims)]
            syn_lambda = np.ones(r[0])
            sol = pyten.tenclass.Ktensor(syn_lambda, u)
            if aux is None:
                aux = [np.dot(sol.Us[n], np.random.random([r[n], siz[n]])) for n in range(dims)]
        else:
            raise ValueError('Do Not Support Such Auxiliary Format.')

    elif problem == 'dynamic':
        ten = []
        omega = []
        sol = []
        for t in range(timestep):
            u = [np.random.random([siz[t, n], r[n]]) for n in range(dims)]
            syn_lambda = np.ones(r[1])
            temp_sol = pyten.tenclass.Ktensor(syn_lambda, u)
            temp_omega = (np.random.random(siz[t]) > miss) * 1
            temp_ten = temp_sol.totensor()
            temp_ten.data[temp_omega == 0] -= temp_ten.data[temp_omega == 0]
            omega.append(temp_omega)
            sol.append(temp_sol)
            ten.append(temp_ten)
        return ten, omega, sol, siz, timestep
    else:
        raise ValueError('No Such Scenario.')

    if tp == 'Dedicom':
        ten = pyten.tenclass.Tensor(sol)
        omega = None
    elif tp == 'Parafac2':
        ten = sol
        omega = None
    else:
        ten = sol.totensor()
        omega = (np.random.random(siz) > miss) * 1
        ten.data[omega == 0] -= ten.data[omega == 0]

    if problem == 'basic':
        return ten, omega, sol
    elif problem == 'auxiliary':
        return ten, omega, sol, aux
    else:
        return ten, omega, sol
