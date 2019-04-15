import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV



def funsvr(x, y, x_test):
    y = np.array(y)
    y = y.reshape(-1, 1)

    svr_rbf = svm.SVR(kernel='rbf', gamma=0.01, C=1e3)
    svr_rbf.fit(x, y)
    svr_linear = svm.SVR(kernel='linear', C=1e3)
    svr_linear.fit(x, y)
    svr_poly = svm.SVR(kernel='poly', degree=3, C=1e3)
    svr_poly.fit(x, y)
    model = Ridge()
    alpha_can = np.logspace(-3, 2, 10)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x, y)

    model1 = Lasso()
    alpha_can = np.logspace(-3, 2, 10)
    lasso_model1 = GridSearchCV(model1, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model1.fit(x, y)



    y_rbf = svr_rbf.predict(x_test)
    y_linear = svr_linear.predict(x_test)
    y_poly = svr_poly.predict(x_test)
    y_hat = lasso_model.predict(x_test)
    y2_hat = lasso_model1.predict(x_test)

    plt.plot(x_test, y_rbf, 'r-', linewidth=2, label='RBF Kernel')
    plt.plot(x_test, y_linear, 'g-', linewidth=2, label='Linear Kernel')
    plt.plot(x_test, y_poly, 'b--', linewidth=2, label='Polynomial Kernel')
    plt.plot(x_test, y_hat, 'b-', linewidth=2, label='Ridge')
    plt.plot(x_test, y2_hat, 'y-', linewidth=2, label='Lasso')
    plt.plot(x, y, 'mo', markersize=6, label='raw data')
    # plt.legend(loc='upper left')
    # plt.title('moniter', fontsize=8)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    x = np.arange(1, 8)
    x = x.reshape(-1, 1)

    y1 = [-2.91, -3.78, -4.14, 5.39, 5.45, 7.02, -5.41]
    y2 = [-2.56, -3.51, -3.31, -2.92, -3.72, -4.19, -4.46]
    y3 = [-1.92, -3.10, -2.31, -2.06, -1.74, -4.18, -2.02]
    y4 = [-0.32, -0.91, -0.49, 1.44, 3.11, 5.43, 4.89]
    y5 = [-0.92, -0.79, -0.59, -0.48, 0.09, -3.22, -3.93]
    y6 = [2.93, 2.61, 2.27, 2.02, 2.40, 0.74, 0.82]
    y7 = [-1.19, -1.78, -1.72, -1.62, -1.20, -4.66, -4.26]
    y8 = [-0.85, -0.28, 2.26, 1.93, 2.39, 1.97, 1.80]
    y9 = [-1.73, -1.47, -2.60, -1.93, -1.45, 2.15, 0.91]
    y10 = [-1.73, -1.47, -2.60, -1.93, -1.45, 2.15, 0.91]

    yi = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]

    y = []
    for i in yi:
        i = np.array(i)
        i = i.reshape(-1, 1)
        y.append(i)

    x_test = np.arange(1, 15).reshape(-1, 1)

    plt.figure(figsize=(18, 12))
    plt.subplot(251)
    funsvr(x, y[0], x_test)
    plt.title('Moniter 1', fontsize=8)
    plt.subplot(252)
    funsvr(x, y[1], x_test)
    plt.title('Moniter 2', fontsize=8)
    plt.subplot(253)
    funsvr(x, y[2], x_test)
    plt.title('Moniter 3', fontsize=8)
    plt.subplot(254)
    funsvr(x, y[3], x_test)
    plt.title('Moniter 4', fontsize=8)
    plt.subplot(255)
    funsvr(x, y[4], x_test)
    plt.title('Moniter 5', fontsize=8)
    plt.subplot(256)
    funsvr(x, y[5], x_test)
    plt.title('Moniter 6', fontsize=8)
    plt.legend(loc='lower left')
    plt.subplot(257)
    funsvr(x, y[6], x_test)
    plt.title('Moniter 7', fontsize=8)
    plt.subplot(258)
    funsvr(x, y[7], x_test)
    plt.title('Moniter 8', fontsize=8)
    plt.subplot(259)
    funsvr(x, y[8], x_test)
    plt.title('Moniter 9', fontsize=8)
    plt.subplot(2,5,10)
    funsvr(x, y[9], x_test)
    plt.title('Moniter 10', fontsize=8)


    # plt.title(u'Ten Moniters')
    # plt.legend(loc='lower left')
    # plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=2.5, hspace=0.2, wspace=0.3)
    plt.savefig("image3.png")
    plt.show()







