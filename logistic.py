# numpy libraries
import numpy as np
import csv

# matplotlib libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import util

from sklearn.linear_model import LogisticRegression






def main():

    classes = ['frog', 'deer', 'ship', 'airplane']
    X_train = np.zeros((100, 3072))
    y_train = np.zeros(100)
    X_test = np.zeros((30, 3072))
    y_test = np.zeros(30)
    labels = open('trainLabels_modified.csv', 'rb')
    labelsReader = csv.reader(labels)

    for index in xrange(130):
        name = "training_data/" + str(index+1) + ".png"
        img = mpimg.imread(name)
        if index < 100:
            X_train[index] = img.flatten()
        else:
            X_test[index-100] = img.flatten()

    count = -1
    for row in labelsReader:
        if count == 130:
            break
        if count >= 0:
            if count < 100:
                y_train[count] = classes.index(row[1])
            else:
                y_test[count-100] = classes.index(row[1])
        count += 1
    print y_train


    clf = LogisticRegression(fit_intercept=True, C=100)
    clf.fit(X_train, y_train)


if __name__ == "__main__" :
    main()