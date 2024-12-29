import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import cv2

class Camera(object):
    """ピンホールカメラを表すクラス"""

    def __init__(self, P):
        """カメラモデルP=K[P|t]を初期化する"""

        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None

    def project(self,X):
        """X(4*nの配列)の点を射影し,座標を正規化する"""

        x = np.dot(self.P, X)

        for i in range(3):
            x[i] /= x[2]

        return x
    
    def factor(self):
        """P=K[R|t]に従い,カメラ行列をK,R,tに分解する"""
        
        """
        K, R = linalg.rq(self.P[:,:3])

        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)
        self.t = np.dot(linalg.inv(self.K), self.P[:,3])
        """
        
        tmp = self.P[:3]  # (3x4) projection matrix
        K, R, t = cv2.decomposeProjectionMatrix(tmp)[:3]
        K /= K[2, 2]  # 3x3 intrinsics matrix
        t = np.dot(linalg.inv(K), self.P[:,3])

        self.K = K
        self.R = R
        self.t = t

        return self.K, self.R, self.t
    
    def center(self):
        """カメラ中心を計算して返す"""
        if self.c is not None:
            return self.c
        else:
            self.factor()
            self.c = -np.dot(self.R.T, self.t)
            return self.c
    
def rotation_matrix(a):
    """ベクトルaを軸に回転する3Dの回転行列を返す"""

    R = np.eye(4)
    R[:3, :3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

    return R