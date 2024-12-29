import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def compute_harris_response(im, sigma=3):
    """
    グレースケール画像の各ピクセルについて
    Harrisコーナー検出器の応答時間を定義する
    """

    # 微分係数
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0,1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1,0), imy)

    # Harris行列の成分を計算する
    Wxx = filters.gaussian_filter(imx*imx, sigma) 
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # 判別式と対角成分
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    epsilon = 1e-10

    return Wdet / (Wtr + epsilon)

def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """
    Harris応答画像からコーナーを返す
    min_distはコーナーや画像から分離する最小ピクセル数
    """

    # しきい値thresholdを超えるコーナー候補を見つける
    corner_threshold = harrisim.max()*threshold
    harrisim_t = (harrisim > corner_threshold)*1

    # 候補の座標を得る
    coords = np.array(harrisim_t.nonzero()).T

    # 候補の値を得る
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # 候補をソートする
    index = np.argsort(candidate_values)

    # 許容する点の座標を配列に格納する
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # 最小距離を考慮しながら最良の配列を得る
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0

    return filtered_coords

def plot_harris_points(image, filtered_coords):
    """
    画像中に見つかったコーナーを描画
    """
    print(filtered_coords)
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], "*")
    plt.axis('off')
    plt.show()

def get_descriptors(image, filtered_coords, wid=5):
    """
    各点について，点の周辺で幅 2*wid+1 の近傍ピクセル値を返す
    """
    desc = []
    for coord in filtered_coords:
        patch = image[coord[0]-wid:coord[0]+wid+1, coord[1]-wid:coord[1]+wid+1].flatten()
        desc.append(patch)

    return desc

def match(desc1, desc2, threshold=0.5):
    """
    正規相互相関を用いて第1画像の点について第2画像の対応点を選択する
    """
    n = len(desc1[0])

    # 対応点ごとの距離
    d = -np.ones((len(desc1), len(desc2)))
    epsilon = 1e-10
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i]-np.mean(desc1)) / (np.std(desc1[i])+epsilon)
            d2 = (desc2[j]-np.mean(desc2)) / (np.std(desc2[j])+epsilon)
            ncc_value = sum(d1*d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value

    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores

def match_twosided(desc1, desc2, threshold=0.5):
    """
    双方向の確認により対応の安全性を高める
    """
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = np.where(matches_12 >= 0)[0]

    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12

def appendimages(im1, im2):
    """
    2つの画像を左右に並べた画像を返す
    """
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1), im1.shape[1])), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2), im2.shape[1])), axis=0)

    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """
    対応点を結んで左右に並べた画像を返す
    入力: im1, im2 配列形式の画像
          locs1, locs2 特徴点の座標
          machscores matchの出力
          show_below 対応の下に画像を出力するならtrue
    """

    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))

    plt.imshow(im3)

    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.plot([locs1[i][1], locs2[m][1]+cols1], [locs1[i][0], locs2[m][0]], "c")

        plt.axis("off")