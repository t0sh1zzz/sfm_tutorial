import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def feature_detection(image_path):
    """
    SIFTを用いて特徴点抽出
    """
    img = cv2.imread(image_path)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    return img, keypoints, descriptors

def plot_features(image, keypoints):
    """
    特徴点の描画
    """
    plt.figure()
    img_sift = cv2.drawKeypoints(image, keypoints, None, flags=4)
    plt.imshow(img_sift)
    plt.show()

def my_match(desc1, desc2):
    """
    自分で作ったマッチング
    """
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
    
    dist_ratio = 0.5
    desc1_size = desc1.shape

    matchscores = np.zeros(desc1_size, 'int')
    desc2t = desc2.T

    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[1,:], desc2t)
        dotprods = 0.9999*dotprods

        indx = np.argsort(np.arccos(dotprods))
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores

def match(desc1, desc2):
    """出力は対応点のidx"""
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros(desc1_size[0], 'int')
    desc2t = desc2.T

    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:], desc2t)
        dotprods = 0.9999 * dotprods
        indx = np.argsort(np.arccos(dotprods))

        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores

def match_twosided(desc1, desc2):
    """
    双方向マッチング
    """
    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

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

    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange']

    for i,m in enumerate(matchscores):
        if m>0:
            color = colors[i % len(colors)]
            plt.plot([locs1[i][0], locs2[m][0]+cols1], [locs1[i][1], locs2[m][1]], color=color, linewidth=1)

        plt.axis("off")

def _process_image(imagename, resultname, params="--edge=thresh 10 --peak-thresh 5"):
    """
    画像を処理してファイルに結果を保存する
    """
    if imagename[-3:] != "pgm":
        im = Image.open(imagename).convert("L")
        im.save("tmp.pgm")
        imagename = "tmp.pgm"

    cmmd = str("sift "+imagename+" --output="+resultname+" "+params)
    os.system(cmmd)
    print("precessed", imagename, "to", resultname)

def _read_features_from_file(filename):
    """
    特徴用を用いて行列形式で返す
    """
    f = np.loadtxt(filename)
    return f[:,:4], f[:,4:]

def _write_features_to_files(filename, locs, desc):
    """
    特徴点の配置と記述子をファイルに保存する
    """
    np.savetext(filename, np.hstack((locs, desc)))

def _plot_features(im, locs, circle=False):
    """
    画像を特徴量と共に描画する
    入力:im 配列形式の画像
    出力:locs 各特徴量の座標とスケール、方向
    """

    def draw_circle(c, r):
        t = np.angle(0,1.01,.01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plt.plot(x, y, "b", linewidth=2)

    plt.imshow()
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:,0], locs[:,1],"ob")
    
    plt.axis("off")

def process_image(imagename, resultname):
    """
    画像を処理してファイルに結果を保存する
    """
    if imagename[-3:] != "pgm":
        im = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("tmp.pgm", im)
        imagename = "tmp.pgm"
    
    img = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    info = np.array([np.hstack((np.array(kp.pt), np.array(kp.size), np.array(kp.angle), np.array(desc))) for kp, desc in zip(keypoints, descriptors)])
    info = info.reshape([-1, 132])

    np.savetxt(resultname, info)

    # test
    # print("len(keys): ", len(keypoints), " len(desc): ", len(descriptors))
    # print(info[1])
    
    # print("Processed", imagename, "to", resultname)
    
def read_features_from_file(filename):
    """
    特徴量を読み込んで行列形式で返す
    """
    f = np.loadtxt(filename)
    locs = f[:, :4]
    desc = f[:, 4:]
    return locs, desc

def write_features_to_files(filename, locs, desc):
    """
    特徴点の配置と記述子をファイルに保存する
    """
    features = np.hstack((locs, desc))
    np.savetxt(filename, features)

def plot_features(im, locs, circle=False):
    """
    画像を特徴量と共に描画する
    入力: im 配列形式の画像
    出力: locs 各特徴量の座標とスケール、方向
    """

    def draw_circle(c, r):
        t = np.linspace(0, 2 * np.pi, 100)
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        plt.plot(x, y, "b", linewidth=2)

    plt.imshow(im, cmap='gray')

    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:, 0], locs[:, 1], "ob")

    plt.axis("off")

def process_image_multiview(images_path, output_path=""):
    """
    多視点画像のsift特徴量を.siftファイルに出力
    """
    
    files = os.listdir(images_path)

    for i in files:
        image_path = os.path.join(images_path, i)

        image_name = i.split(".", 1)[0]
        process_image(image_path, f"{output_path}/{image_name}.sift")

def matching_multiview(images_path, sift_path):
    """
    多視点画像をsift特徴量を用いてマッチング
    """

    files_images = os.listdir(images_path)
    files_sift = os.listdir(sift_path)
    nbr_images = len(files_images)
    matchscores = np.zeros((nbr_images, nbr_images))

    for i in range(nbr_images):
        for j in range(i, nbr_images):
            print("comparing ", files_images[i], " ", files_images[j])


            l1, d1 = read_features_from_file(os.path.join(sift_path, files_sift[i]))
            l2, d2 = read_features_from_file(os.path.join(sift_path, files_sift[j]))

            matches = match_twosided(d1, d2)

            nbr_matches = np.sum(matches > 0)

            print("number of matches = ", nbr_matches)
            matchscores[i, j] = nbr_matches

    for i in range(nbr_images):
        for j in range(i+1, nbr_images):
            matchscores[j, i] = matchscores[i, j]
    
    return matchscores