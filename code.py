import enum
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


#define data
n_samples=5000
d=2
test_size=0.2
X,y=datasets.make_blobs(n_samples=n_samples, centers=2, n_features=d,cluster_std=.5,random_state=0)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#define hyper-parameters
n_train=X_train.shape[0]
h=(1/n_train**(1/4))**(1/(2*d+10))

fig=plt.figure()
plt.scatter(X[:,0],X[:,1],c=y)
plt.title("Make_blobs data from scikit-learn")
fig.savefig("data")

def comp_xj(j,h=h):
    return j*h

def comp_j(x,h=h):
    y=x/h
    j=np.round(y,0)
    return j


def classify(x,X_tr,y_tr,alpha):
    n_train=X_tr.shape[0]
    j=comp_j(x)
    xj=comp_xj(j)
    diff=X_tr-xj
    B=(np.max(diff,axis=1)<h)
    
    eps=np.random.laplace(0,2**(d+1)/alpha,size=X_tr.shape[0])
    
    Z=np.empty_like(eps)
    for i in range(X_tr.shape[0]):
        if i<n_train/2:
            Z[i]=B[i]+eps[i]
        else:
            Z[i]=y_tr[i]*B[i]+eps[i]

    T=0
    for i in range(X_tr.shape[0]):
        if i<n_train/2:
            T-=1/2*Z[i]
        else:
            T+=Z[i]
        
    T/=n_train
    C=T>=0

    return C

def get_acc(alpha):
    pred=np.zeros_like(y_test)
    k=0
    for i in range(pred.shape[0]):
        pred[i]=classify(X_test[i,:],X_train,y_train,alpha=alpha)
        k+=pred[i]==y_test[i]

    print("Alpha: %.2f\t Accuracy: %f " %(alpha,k/pred.shape[0]))

    return k/pred.shape[0]

alphas=np.linspace(start=0.01,stop=10,num=30)
acc=np.empty_like(alphas)
for i,alpha in enumerate(alphas):
    acc[i]=get_acc(alpha=alpha)

fig=plt.figure()
plt.plot(alphas,acc)
plt.xlabel("Alpha")
plt.ylim(top=1)
plt.ylabel("Accuracy")
plt.title("Trade-off between privacy and accuracy")
plt.show()
fig.savefig("Results")




