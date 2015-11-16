# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

import util

cifar_imageSize = (32,32)



######################################################################
# main
######################################################################

def main() :

    X = np.zeros((100, 3072))
    for index in xrange(100):
        name = "extract/train/" + str(index+1) + ".png"
        img = mpimg.imread(name)
        X[index] = img.flatten()
        # plt.imshow(img)
        # plt.show()

    # average_pic = np.average(X, axis=0)
    # util.show_image(average_pic)

    U, mu = util.PCA(X)
    util.plot_gallery([util.vec_to_image(U[:,i]) for i in range(1000, 1010)])

    l_list = [1, 10, 50, 100, 500, 1288]
    for l in l_list:
        Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        # util.plot_gallery([X_rec[i, :] for i in xrange(12)])


if __name__ == "__main__" :
   main()