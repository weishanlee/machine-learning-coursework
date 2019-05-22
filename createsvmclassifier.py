"""
INPUT:	
xTr : dxn input vectors
yTr : nx1 input labels
alphas : nx1 vector of alphas generated from quadratic program solution of svm
bias : bias of classifier generated by recoverBias
ktype : type of kernel
kpar : parameter of kernel
compute

Output:
svmclassify : a classifier (svmclassify(xTe) defined in this function 
that returns the binary predictions on xTe)

Creates an svm classifierthat can make predictions on new test data
"""
import numpy as np
from computeK import computeK

# def createsvmclassifier(xTr, yTr, alphas, bias, ktype, kpar):
#     # classifier that returns all ones
#     def svmclassify(xTe):
#         d,m = xTe.shape
#
#         K = computeK(ktype, xTr, xTe, kpar)
#
#         preds = np.sign(K.T.dot(alphas * yTr).T + bias)  # 1 x m
#         return preds
#
#
#
#     return svmclassify
#
def createsvmclassifier(xTr, yTr, alphas, bias, ktype, kpar):
    # classifier that returns all ones
    def svmclassify(xTe):
        d, n = xTe.shape

        y = yTr*alphas

        K = computeK(ktype, xTr, xTe, kpar)

        return np.sign(np.dot(y.T, K) + bias).T

    return svmclassify