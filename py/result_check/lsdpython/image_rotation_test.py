import numpy as np
import scipy.io as io
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2


image_file = "/home/kluger/ma/data/real_world/york/all_orig_images/P1020177.jpg"

cameraParams = io.loadmat("/home/kluger/ma/data/real_world/york/cameraParameters.mat")

f = cameraParams['focal'][0, 0]
ps = cameraParams['pixelSize'][0, 0]
pp = cameraParams['pp'][0, :]

print f, ps, pp

K = np.matrix([[f / ps, 0, pp[0]], [0, f / ps, pp[1]], [0, 0, 1]])

K_inv = np.linalg.inv(K)

image = ndimage.imread(image_file)

imageWidth = image.shape[1]
imageHeight = image.shape[0]


def make_axis_rotation_matrix(direction, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general
    axis by a specified angle.

    R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)

    Parameters:

        angle : float a
        direction : array d
    """
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)

    eye = np.eye(3, dtype=np.float64)
    ddt = np.outer(d, d)
    skew = np.array([[0, d[2], -d[1]],
                     [-d[2], 0, d[0]],
                     [d[1], -d[0], 0]], dtype=np.float64)

    mtx = ddt + np.cos(angle) * (eye - ddt) + np.sin(angle) * skew
    return mtx


R0 = np.matrix(make_axis_rotation_matrix(np.array([0,0,1]), 0*np.pi*1.0/10))
R1 = np.matrix(make_axis_rotation_matrix(np.array([0,1,0]), 0*np.pi*1.0/10))
print R1
R2 = np.matrix(make_axis_rotation_matrix(np.array([1,0,0]), np.pi*1.0/10))
print R2
R = R2*R1*R0
print R
H = K*R*K.I

im_out = cv2.warpPerspective(image, H, (imageWidth,imageHeight), flags=cv2.INTER_LANCZOS4)

plt.imshow(image)
plt.figure()
plt.imshow(im_out)
plt.show()



