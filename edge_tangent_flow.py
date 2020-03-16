"""
Coherent Line Drawing - Pablo Eliseo
"""
#Edge Tangent Flow
import warnings
import math
import cv2
import numpy as np
warnings.filterwarnings("error")
#######################   AUX FUNCTIONS   ##########################
def compute_tangent(v):
    theta = math.pi/2 #90º
    #rot matrix 90º
    x = v[0] * np.cos(theta) - v[1] * np.sin(theta)
    y = v[1] * np.cos(theta) + v[0] * np.sin(theta)
    return [x, y]


####################################################################
######################    PAPER'S FUNCTIONS    #####################
####################################################################

# t_0 (0) no equation, specified at the end of the section
def initial_etf(img):
    tangent = np.zeros(img.shape + (2,), dtype=np.float32)
    img_n = cv2.normalize(img.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    #Generate gradients
    x_grad, y_grad = [], []
    x_grad = cv2.Sobel(img_n, cv2.CV_32FC1, 1, 0, ksize=5) #kernel size = 5
    y_grad = cv2.Sobel(img_n, cv2.CV_32FC1, 0, 1, ksize=5) #kernel size = 5

    #Compute gradient
    gradient_mag = cv2.sqrt(x_grad**2.0 + y_grad**2.0)
    gradient_mag = cv2.normalize(gradient_mag.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

    h, w = img.shape[0], img.shape[1]
    for i in range(h):
        for j in range(w):
            a = x_grad[i][j]
            b = y_grad[i][j]
            vect = np.array([a, b], ndmin=1)
            cv2.normalize(vect.astype("float32"), vect, 0.0, 1.0, cv2.NORM_MINMAX)
            tangent[i][j] = compute_tangent(vect)
    return tangent, gradient_mag

# t_new for every pixel, no equation
def compute_image_etf(tangent, kernel, gradient_mag):
    tangent_new = np.zeros(tangent.shape, dtype=np.float32)
    h, w = tangent.shape[0], tangent.shape[1]
    for x in range(h):
        for y in range(w):
            tangent_new[x][y] = compute_new_etf(x, y, tangent, kernel, gradient_mag)
    return tangent_new
   
# t_new (1)
def compute_new_etf(x, y, tangent, kernel, gradient_mag):
    try:
        t_new = np.zeros(2, dtype=np.float32)
        h, w = tangent.shape[0], tangent.shape[1]
        for c in range(x - kernel, x + kernel + 1):
            for r in range(y - kernel, y + kernel + 1):
                x_tan = tangent[x][y]
                if (r < 0 or r >= w or c < 0 or c >= h):
                    continue
                y_tan = tangent[c][r]
                a = np.array([x, y])
                b = np.array([c, r])
                phi = compute_phi(x_tan, y_tan)
                ws = compute_ws(a, b, kernel)
                wm = compute_wm(gradient_mag[x][y], gradient_mag[c][r])
                wd = compute_wd(x_tan, y_tan)
                t_new += phi * y_tan * ws * wm * wd
    except RuntimeWarning:
        print("error")
    return t_new

# ws (2)
def compute_ws(x_vect, y_vect, radius):
    if np.linalg.norm(x_vect-y_vect) < radius:
        return 1
    return 0

 # wm (3)
def compute_wm(x_grad_norm, y_grad_norm):
    eta = 1 #Specified in paper
    return (1/2) * (1 + np.tanh(eta*(x_grad_norm - y_grad_norm)))

# wd (4)
def compute_wd(x_tan, y_tan):
    return abs(np.dot(x_tan, y_tan))

# phi (5)
def compute_phi(x_tan, y_tan):
    if np.dot(x_tan, y_tan) > 0:
        return 1
    return -1

####################################################################
#########################  MAIN FUNCTION  ##########################
####################################################################

if __name__ == '__main__':
    KERNEL = 5
    IMG = cv2.imread("prueba1.jpg", 0) #0 for grayscale image
    TANGENT, GRADIENT_MAG = initial_etf(IMG)
    ITERATIONS = 1
    for it in range(ITERATIONS):
        TANGENT = compute_image_etf(TANGENT, KERNEL, GRADIENT_MAG)
    print(TANGENT)
    