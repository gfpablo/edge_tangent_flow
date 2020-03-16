"""
Coherent Line Drawing - Pablo Eliseo
"""
#Edge Tangent Flow
import sys
import math
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

################# Classes ###################
class ETF:
    """Main class, Edge Tangent Flow"""
    def __init__(self, file):
        self.__kernel = 5
        img = cv2.imread(str(file), 0) #0 for grayscale image
        self.shape = img.shape+(2,)
        self.flow_field = np.zeros(self.shape, dtype=np.float32)
        img_n = cv2.normalize(img.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

        #Generate partial derivatives
        x_der, y_der = [], []
        x_der = cv2.Sobel(img_n, cv2.CV_32FC1, 1, 0, ksize=self.__kernel) #kernel size = 5
        y_der = cv2.Sobel(img_n, cv2.CV_32FC1, 0, 1, ksize=self.__kernel) #kernel size = 5

        #Compute gradient
        self.gradient_mag = cv2.sqrt(x_der**2.0 + y_der**2.0)
        self.gradient_mag = cv2.normalize(self.gradient_mag.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

        h, w = img.shape[0], img.shape[1]
        for i in range(h):
            for j in range(w):
                a = x_der[i][j]
                b = y_der[i][j]
                vect = np.array([a, b], ndmin=1)
                cv2.normalize(vect.astype("float32"), vect, 0.0, 1.0, cv2.NORM_MINMAX)
                self.flow_field[i][j] = compute_tangent(vect)
    
    
    def compute_ETF(self):
        tangent_new = np.zeros(self.shape, dtype=np.float32)
        h, w = self.shape[0], self.shape[1]
        for x in range(h):
            for y in range(w):
                tangent_new[x][y] = self.__compute_new_etf(x, y)
        self.flow_field = tangent_new
    
    def save(self):
        np.save("EFT.npy", self.flow_field)
        np.save("gradient_magnitude.npy", self.gradient_mag)

    ## PRIVATE METHODS ##
    def __compute_new_etf(self, x, y):
        t_new = np.zeros(2, dtype=np.float32)
        h, w = self.flow_field.shape[0], self.flow_field.shape[1]
        for c in range(x - self.__kernel, x + self.__kernel + 1):
            for r in range(y - self.__kernel, y + self.__kernel + 1):
                x_tan = self.flow_field[x][y]
                if (r < 0 or r >= w or c < 0 or c >= h):
                    continue
                y_tan = self.flow_field[c][r]
                a = np.array([x, y])
                b = np.array([c, r])
                phi = compute_phi(x_tan, y_tan)
                ws = compute_ws(a, b, self.__kernel)
                wm = compute_wm(self.gradient_mag[x][y], self.gradient_mag[c][r])
                wd = compute_wd(x_tan, y_tan)
                t_new += phi * y_tan * ws * wm * wd
        return t_new

#######################   AUX FUNCTIONS   ##########################
def compute_tangent(v):
    # if v = (v[0],v[1]), v_perpendicular = [-v[0], v[1]] 
    theta = math.pi/2 #90º
    x = v[0] * np.cos(theta) - v[1] * np.sin(theta) #-v[1]
    y = v[1] * np.cos(theta) + v[0] * np.sin(theta) # v[0]
    return [x, y]

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
    INSTANCE = ETF(sys.argv[1])
    for i in range(int(sys.argv[2])):
        INSTANCE.compute_ETF()
    INSTANCE.save()
    print(INSTANCE.flow_field)
