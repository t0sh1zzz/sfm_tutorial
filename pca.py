from PIL import Image
import numpy as np

def pca(X):
    """
    主成分分析

    入力: X 訓練データを平板化した配列を行として格納した行列
    出力: 写像行列（次元の重要度順），分散，平均
    """

    # 次元数を取得
    num_data,dim = X.shape

    # データをセンタリング
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        M = np.dot(X, X.T) # 共分散行列
        e, EV = np.linalg.eigh(M) # 固有値と固有ベクトル
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        S = np.sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:, i] //= S # / と // の違いに注意
    else:
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]

    return V,S,mean_X
