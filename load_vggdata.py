import camera
import numpy as np
from PIL import Image

im1 = np.array(Image.open("/home/jaxa/toshiki/sfmtutorial/images/001.jpg"))
im2 = np.array(Image.open("/home/jaxa/toshiki/sfmtutorial/images/002.jpg"))

points2D = [np.loadtxt("2D/00"+str(i+1)+".corners").T for i in range(3)]

points3D = np.loadtxt("3D/p3d").T

corr = np.genfromtxt("2D/nview-corners", dtype="int", missing_values="*")

P = [camera.Camera(np.loadtxt("2D/00"+str(i+1)+".P")) for i in range(3)]