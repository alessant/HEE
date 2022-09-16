from typing import Iterable
from hypernetx.classes.hypergraph import Hypergraph
import numpy as np

from scipy import sparse
from scipy.sparse import csr_matrix

from hee.utils import matrix as mat
from sklearn.decomposition import PCA

class SpectralEmbeddingFramework:
    def __init__(self, Hg: Hypergraph, W: csr_matrix=None) -> None:
        self.H, _, _ = Hg.edges.incidence_matrix(sparse=True, index=True)
        self.Dv = sparse.diags(self.H.sum(axis=1).A1)
        self.De = sparse.diags(self.H.sum(axis=0).A1)
        self.Z = sparse.eye(self.H.shape[0])
        self.Sv = lambda H, Se: H * Se * H.transpose()
        self.W = W if W is not None else sparse.eye(self.H.shape[1])
        self.Se = lambda W: W

    def _laplacian(self) -> csr_matrix:
        return self.Z * (self.Dv - self.Sv(self.H, self.Se(self.W))) * self.Z.transpose()
    
    def fit(self, dim: int=None, indices: Iterable[int]=(1,2)) -> np.ndarray:
        L = self._laplacian()
        k = (dim if dim is not None else max(indices)) + 1
        eigenval, eigenvec = sparse.linalg.eigs(L, k=k, which='SR')
            
        s_eigenval = np.argsort(eigenval.real)
        indices = range(1, dim+1) if dim is not None else indices
        vecs = [s_eigenval[index] for index in indices]
        return eigenvec[:, vecs].real


class Zhou(SpectralEmbeddingFramework):
    def __init__(self, H: Hypergraph, W: csr_matrix=None, norm_de: bool=False) -> None:
        super().__init__(H, W)
        self.Z = mat.fast_inverse(self.Dv).sqrt()
        if norm_de:
            De_inv = mat.fast_inverse(self.De)
            self.Se = lambda W: W * De_inv

class Ren(SpectralEmbeddingFramework):
    def __init__(self, H: Hypergraph, W: csr_matrix=None) -> None:
        super().__init__(H, W)
        self.Z *= np.sqrt(2)
        self.Se = lambda W: .5 * sparse.eye(self.H.shape[1])
    
    def fit(self, dim, pca: bool=False) -> np.ndarray:
        if not pca:
            return super().fit(dim=dim)
        else:
            assert dim is not None and dim <= 6
            emb = super().fit(dim=6)
            return PCA(n_components=dim).fit_transform(emb)


class Bolla(SpectralEmbeddingFramework):
    def __init__(self, H: Hypergraph) -> None:
        super().__init__(H)
        self.Se = lambda W: mat.fast_inverse(self.De)

class Zhu(SpectralEmbeddingFramework):
    def __init__(self, H: Hypergraph, W: csr_matrix) -> None:
        super().__init__(H, W)
        De_inv = mat.fast_inverse(self.De)
        self.Se = lambda W: W * De_inv
    
class Luo(Zhou):
    def __init__(self, H: Hypergraph, W: csr_matrix=None) -> None:
        super().__init__(H, W, norm_de=True)
    
    def fit(self, X: np.ndarray, dim=None, indices=(1,2)) -> np.ndarray:
        assert dim is None or dim <= X.shape[1]
        L = self._laplacian()
        LL = X @ L.toarray() @ X.transpose()
        eigenval, eigenvec = np.linalg.eig(LL)
        #eigenval, eigenvec = sparse.linalg.eigs(L, which='SR')
        s_eigenval = np.argsort(eigenval.real)
        indices = range(1, dim+1) if dim is not None else indices
        vecs = [s_eigenval[index] for index in indices]
        return X @ eigenvec[:, vecs].real


class Rodriguez(SpectralEmbeddingFramework):
    def __init__(self, H: Hypergraph, W: csr_matrix=None) -> None:
        super().__init__(H, W)
        Dr = sparse.diags((self.H * self.W * self.H.T - self.Dv).sum(axis=1).A1)
        self.Z = mat.fast_inverse(Dr).sqrt()
        self.Sv = lambda H, Se: H * Se * H.transpose() + Dr


class Saito(SpectralEmbeddingFramework):
    def __init__(self, H: Hypergraph, W: csr_matrix=None) -> None:
        super().__init__(H, W)
        self.Z = mat.fast_inverse(self.Dv).sqrt()
        self.De -= sparse.eye(self.De.shape[1])
        De_inv = mat.fast_inverse(self.De)

        if not np.isfinite(De_inv.diagonal()).all():
            print("H has hyperedges of size 1. Modifying inf to 0.")
            d = De_inv.diagonal()
            d[np.where(np.isinf(d))] = 0
            De_inv = sparse.diags(d)

        A = self.H * De_inv * self.W * self.H.transpose()
        self.Se = lambda W: De_inv * W
        self.Sv = lambda H, Se: H * Se * H.transpose() - A.diagonal()


def get_hypergraph_laplacian(h, W=None, mode=None):
    H, id2n, id2e = h.edges.incidence_matrix(sparse=True, index=True)

    Dv = sparse.diags(H.sum(axis=1).A1)
    De = sparse.diags(H.sum(axis=0).A1)

    if W is None:
        W = sparse.eye(H.shape[1])

    I = sparse.eye(H.shape[0])

    if mode == "Zhou":
        De_inv = mat.fast_inverse(De)
        A = H * W * De_inv * H.transpose()
    elif mode == "Rod":
        A = H * W * H.transpose() - Dv
        Dv = sparse.diags(A.sum(axis=1).A1)
    elif mode == "Ren":
        return 2 * Dv - H * H.transpose()
    elif mode == "hhe":
        De_inv = mat.fast_inverse(De)
        return Dv - H * W * De_inv * H.transpose() #I - H * W * De_inv * H.transpose() #
    elif mode == "Saito":
        De = De - sparse.eye(De.shape[1])
        De_inv = mat.fast_inverse(De)

        if not np.isfinite(De_inv.diagonal()).all():
            print("H has hyperedges of size 1. Modifying inf to 0.")

            d = De_inv.diagonal()
            d[np.where(np.isinf(d))] = 0
            De_inv = sparse.diags(d)

        A1 = H * De_inv * W * H.transpose()
        A = A1 - sparse.diags(A1.diagonal())
        
    
    Dv_inv_sqrt = mat.fast_inverse(Dv).sqrt()
    
    L = Dv_inv_sqrt * (Dv - A) * Dv_inv_sqrt

    return L



def smhc(hgs, Ws=None, alphas=None): 
    # TODO check if all hgs have the same set of vertices
    V = hgs[0].shape[0]

    if Ws is None:
        Ws = [sparse.eye(h.shape[1]) for h in hgs]

    if alphas is None:
        alphas = np.array([1/len(hgs)] * len(hgs))

    #
    Ps, Pies = zip(*[get_rw_dist(h) for h in hgs])

    # compute the unique stationary distribution
    # for the mixute of random walks
    Pies = np.array(Pies)
    Pie_m = np.dot(alphas, Pies) 

    # compute the mixute of
    # transition probabilities
    ones = np.ones(V)

    Betas = Pies * np.outer(alphas, ones) / Pie_m
    P_m = np.zeros((V, V))
    
    for i in range(len(hgs)):
        P_m += Ps[i].toarray() * np.outer(Betas[i], ones)

    # laplacian
    Pi = np.diag(Pie_m)

    L = Pi - (0.5 * (np.dot(Pi, P_m) + np.dot(P_m.T, Pi)))

    return L



def get_rw_dist(h, W=None):
    # evaluating the transition probability matrix for h
    H, id2n, id2e = h.edges.incidence_matrix(sparse=True, index=True)

    Dv_inv = mat.fast_inverse(sparse.diags(H.sum(axis=1).A1))
    De_inv = mat.fast_inverse(sparse.diags(H.sum(axis=0).A1))

    if W is None:
        W = sparse.eye(H.shape[1])

    P = Dv_inv * H * W * De_inv * H.transpose()

    # evaluating the stationary distribution
    # of the random walk
    volV = sum(H.sum(axis=0).A1)
    Pie = H.sum(axis=1).A1 / volV

    return P, Pie



def HGE(h, dim, ws=None):
    # hypergraph params
    H, id2n, id2e = h.incidence_matrix(sparse=True, index=True)

    V = h.number_of_nodes()
    E = h.number_of_edges()

    if ws is None:
        ws = np.ones(H.shape[1])

    # algo params
    learn_rate = 1e-2;      # learning rate
    lr_dec_rate = 0.995;    # decreasing rate of the learning rate
    lr_dec_freq = 3;        # frequency (per epoch) of decreasing learning rate
    n_epoch = 300;          # number of maximum epoch
    eps_SC = 1e-5;          # eps for the stopping criterion

    # normalize using P-norm
    # Set normP larger than orders of all hyperedges
    P = max(H.sum(axis=0).A1) + 1 

    # Initialization
    rng = np.random.default_rng()
    X = rng.random((V, dim))
    cost = np.zeros(E)
    last_loss = np.inf
    conv = False
    dec_idx = np.floor(np.linspace(0, E, num=lr_dec_freq+1))

    # Algorithm
    for epoch in range(0, n_epoch):
        print("Epoch %d" % epoch)

        idx = np.random.permutation(E)
        
        for i in idx:
            vs = H[:, i].nonzero()[0]
            de = len(vs)
            #vs = np.where(e == 1)[0] # check non zero, o nz
            Xi = X[vs, :]

            A = np.vstack((np.ones(dim), np.cumprod(Xi[:-1], axis=0)))
            B = np.vstack((np.cumprod(np.flip(Xi[1:], 0), axis=0), np.ones(dim)))

            grad = ws[i] * (np.power(Xi, de-1) - A * B)

            # update X
            X[vs, :] = Xi - learn_rate * grad

            # clip negative coordinates
            X[vs, :] = np.max(X[vs, :], 0)

            # normalize X
            for v in vs:
                z = np.linalg.norm(X[v, :], P)

                if z == 0:
                    X[v, :] = rng.random((1, dim))
                else:
                    X[v, :] = X[v, :] * np.power(dim, 1/P) / z

            # Decrease learning rate
            if i in dec_idx:
                learn_rate = learn_rate * lr_dec_rate

        # compute loss
        for j in range(0, E):
            vs = H[:, j].nonzero()[0]

            de = len(vs)
            Xi = X[vs, :]

            cost[j] = np.sum(np.mean(np.power(Xi, de), axis=0) - np.prod(Xi, axis=0))

        loss = np.mean(cost * ws)

        print("Loss: %f, LearnRate: %f\n" % (loss, learn_rate))

        # check convergence
        if np.abs(loss - last_loss) < eps_SC:
            conv = True
            break
        else:
            last_loss = loss

    return X, epoch, conv



def embed_hypergraph(L, dim=None, indices=(1,2), sprs=False):
    if sprs:
        eigenval, eigenvec = sparse.linalg.eigs(L, k=dim+1, which='SR')
    else:
        LL = L.toarray() if isinstance(L, csr_matrix) else L
        eigenval, eigenvec = np.linalg.eig(LL)
        
    s_eigenval = np.argsort(eigenval.real)
    vecs = []

    if dim is not None:
        vecs = [s_eigenval[index] for index in range(1, dim+1)]
    else:
        vecs = [s_eigenval[index] for index in indices]

    return eigenvec[:, vecs].real
