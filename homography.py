import numpy as np
import matplotlib.pyplot as plt

def nomalize(points):
    """同時座標系の点の集合を最後のrowが1になるように正規化する"""
    
    for row in points:
        row /= points[-1]

    return points

def make_homog(points):
    """点の集合(dim*n)の配列を同時座標系に変換する"""

    return np.vstack((points, np.ones((1, points.shape[1]))))

def H_from_points(fp, tp):
    """線形なDLT法を使ってfpをtpに対応づけるホモグラフィー行列Hを求める"""
    
    if fp.shape != tp.shape:
        raise RuntimeError("number of points do not match!")
    
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = np.dot(C1, fp)

    m = np.mean(tp[:2], axis=1)
    maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = np.dot(C2, tp)

    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i]*fp[0][i], tp[0][i]*fp[1][i], tp[0][i]]
        A[2*i+1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i]*fp[0][i], tp[1][i]*fp[1][i], tp[1][i]]

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2, 2]

def Haffine_from_points(fp, tp):
    """fpをtpに変換するアフィン変換行列Hを求める"""

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = np.dot(C1, fp)

    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy()
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = np.dot(C2, tp)

    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)

    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2,2]

class RansacModel(object):
    """RANSACを用いてホモグラフィー行列を求める"""

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """4つの対応点にホモグラフィーを当てはめる"""
        data = data.T

        fp = data[:3,:4]
        tp = data[3:,:4]

        return H_from_points(fp, tp)
    
    def get_error(self, data, H):
        """全ての対応にホモグラフィーを当てはめ,各変換点との誤差を返す"""
        data = data.T
        
        fp = data[:3]
        tp = data[3:]

        fp_transformed = np.dot(H, fp)

        nz = np.nonzero(fp_transformed[2])
        for i in range(3):
            fp_transformed[i][nz] /= fp_transformed[2][nz]

        return np.sqrt(np.sum((tp-fp_transformed)**2, axis=0))
    
def H_from_ransac(fp, tp, model, maxiter=1000, match_threshold=10):
    """RANSACを用いて対応点からホモグラフィー行列Hをロバストに推定"""

    import ransac

    data = np.vstack((fp, tp))

    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_threshold, 10, return_all=True)

    return H, ransac_data["inliers"]