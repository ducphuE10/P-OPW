import numpy as np
from numpy.linalg import norm


def cal_cosin(i,j):
  return np.dot(i,j)/(norm(i)*norm(j))


def norm1(i,j):
  return norm(i-j,1)

def norm2(i,j):
  return norm(i-j,2)

def get_distance(x,y,distance = 'cosin'):
    '''TODO: implement faster version'''
    D = []
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            assert x[i].shape == y[j].shape
            if distance == 'norm1':
                D.append(norm1(x[i],y[j])/x[i].shape[0])
            elif distance == 'norm2':
                D.append(norm2(x[i],y[j]))
            elif distance == 'mix':
                D.append(1-cal_cosin(x[i],y[j]) + norm2(x[i],y[j]))
            elif distance == 'cosin':
                D.append(1-cal_cosin(x[i],y[j]))
            else:
                raise NotImplementedError
            
    D = np.array(D).reshape(x.shape[0],-1)
    return D