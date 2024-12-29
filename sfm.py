import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import cv2

def compute_fundamental(x1, x2):
    """正規化8点法を使って対応点群（x1, x2:3*nの配列）から基礎行列を計算する"""

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't mutch.")
    
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i],x1[0,i]*x2[1,i],x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i],x1[1,i]*x2[1,i],x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i],x1[2,i]*x2[1,i],x1[2,i]*x2[2,i]]
        
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3,3)

    U, S, V  = linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F
    
def compute_epipole(F):
    """基礎行列Fから（右側の）エピ極を計算する（左のエピ極はF.T）"""

    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]

def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
    """エピポールとエピポーラ線を画像に描画する"""

    m, n = im.shape[:2]
    line = np.dot(F, x)

    t = np.linspace(0, n, 100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    ndx = (lt>=0) & (lt<m)
    plt.plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0]/epipole[2], epipole[1]/epipole[2], "r*")

def triangulate_point(x1, x2, P1, P2):
    """最小二乗法を用いて点の組を三角測量する"""

    M = np.zeros((6, 6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U, S, V = np.linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]

def triangulate(x1, x2, P1, P2):
    """x1, x2(3*nの同時座標)の点の2視点三角測量"""

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match!")
    
    X = [triangulate_point(x1[:,i], x2[:,i], P1, P2) for i in range(n)]

    return np.array(X).T
    
def compute_P(x, X):
    """2D-3Dの対応の組（同時座標系）からカメラ行列を推定する"""

    n = x.shape[1]
    if X.shape[1] != n:
        raise ValueError("Number of points don't match!")
    
    M = np.zeros((3*n, 12+n))

    for i in range(n):
        M[3*i,0:4] = X[:,i]
        M[3*i+1,4:8] = X[:,i]
        M[3*i+2,8:12] = X[:,i]
        M[3*i:3*i+3,i+12] = -x[:,i]

    U, S, V = linalg.svd(M)

    return V[-1,:12].reshape((3,4))

def compute_P_from_fundamental(F):
    """P1 = [I 0]と仮定して第2のカメラ行列を基礎行列から計算する"""

    e = compute_epipole(F.T)
    Te = skew(e)
    return np.vstack((np.dot(Te,F.T).T,e)).T

def skew(a):
    """任意のvについてa x v= Avになる交代行列A"""

    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def compute_P_from_essential(E):
    """基本行列から第2のカメラ行列を計算する（P1 = [I 0]）を仮定"""

    U, S, V = linalg.svd(E)
    if np.det(np.dot(U, V)) < 0:
        V = -V
    E = np.dot(U, np.dot(np.diag([1,1,0]), V))

    Z = skew([0, 0, -1])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    P2 = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:,2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:,2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:,2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:,2])).T]
    
class RansacModel(object):
    """ransac.pyを用いて基礎行列を当てはめるためのclass"""

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """8つの選択した点を使って基礎行列を推定する"""

        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]

        F = compute_fundamental_normalized(x1, x2)
        return F
    
    def get_error(self, data, F):
        """全ての対応についてx^T F xを計算し,変換された点の誤差を返す"""

        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        Fx1 = np.dot(F, x1)
        Fx2 = np.dot(F, x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = (np.diag(np.dot(x1.T, np.dot(F, x2))))**2 / denom

        return err


def compute_fundamental_normalized(x1, x2):
    """正規化8点法を使って対応点群から基礎行列を計算する"""

    n = x1.shape[1]

    if x2.shape[1] != n:
        raise ValueError("Number of points don't match!")
    
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1*mean_1[0]], [0, S1, -S1*mean_1[1]], [0,0,1]])
    x1 = np.dot(T1, x1)

    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2*mean_2[0]], [0, S2, -S2*mean_2[1]], [0,0,1]])
    x2 = np.dot(T2, x2)

    F = compute_fundamental(x1, x2)

    F = np.dot(T1.T, np.dot(F, T2))

    return F/F[2,2]

def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    """ransac.pyを使って基礎行列Fをロバストに推定"""

    import ransac

    data = np.vstack((x1, x2))

    F, randac_data = ransac.ransac(data.T, model, 8, maxiter, match_threshold, 20, return_all=True)

    return F, randac_data["inliers"]