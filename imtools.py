import os
from PIL import Image
import numpy as np

def get_imlist(path):
    """path に指定されたディレクトリのすべてのjpgファイル名のリストを返す"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]

def imresize(im, sz):
    """PILを用いて画像配列のサイズを変更する"""
    pil_im = Image.fromarray(np.uint8(im))