import ot 
import numpy as np

def POT_feature_2sides(a,b,D, m=None, nb_dummies=1):
  # a = np.ones(D.shape[0])/D.shape[0]
  # b = np.ones(D.shape[1])/D.shape[1]
  if m < 0:
      raise ValueError("Problem infeasible. Parameter m should be greater"
                        " than 0.")
  elif m > np.min((np.sum(a), np.sum(b))):
      raise ValueError("Problem infeasible. Parameter m should lower or"
                        " equal than min(|a|_1, |b|_1).")
  # import ipdb; ipdb.set_trace()
  b_extended = np.append(b, [(np.sum(a) - m) / nb_dummies] * nb_dummies)
  a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
  D_extended = np.zeros((len(a_extended), len(b_extended)))
  D_extended[-nb_dummies:, -nb_dummies:] = np.max(D) * 2
  D_extended[:len(a), :len(b)] = D
  return a_extended, b_extended, D_extended

def POT_feature_1side(a,b,D, m=0.8, nb_dummies=1):
  # a = np.ones(D.shape[0])*m/D.shape[0]
  # b = np.ones(D.shape[1])/D.shape[1]
  a = a*m
  '''drop on side b --> and dummpy point on side a'''
  a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
  D_extended = np.zeros((len(a_extended), len(b)))
  D_extended[:len(a), :len(b)] = D
  return a_extended, b,D_extended

def opw_distance_2(D,lambda1=20, lambda2=0.1, delta=1):
  N =D.shape[0]
  M = D.shape[1]

  E = np.zeros((N,M))
  for i in range(N):
    for j in range(M):
      E[i,j] = (i/N - j/M)**2
  
    l = np.zeros((N,M))
  for i in range(N):
    for j in range(M):
      l[i,j] = abs(i/N - j/M)/(np.sqrt(1/N**2 + 1/M**2))
  F = l**2
  return D + lambda1*E + lambda2*(F/2 + np.log(delta*np.sqrt(2*np.pi)))

def opw_distance(D, lambda1=0, lambda2=0.1, delta=1):
  N =D.shape[0]
  M = D.shape[1]

  E = np.zeros((N,M))
  for i in range(N):
    for j in range(M):
      E[i,j] = 1/((i/N - j/M)**2 + 1)

  l = np.zeros((N,M))
  for i in range(N):
    for j in range(M):
      l[i,j] = abs(i/N - j/M)/(np.sqrt(1/N**2 + 1/M**2))
  F = l**2
  return D - lambda1*E + lambda2*(F/2 + np.log(delta*np.sqrt(2*np.pi)))


def entropic_opw_1(a, b, D, lambda1, lambda2, delta=1, m=None, numItermax=1000):
  '''
  Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). 
  Iterative Bregman projections for regularized transportation problems. 
  SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
  '''
  D = opw_distance(D, lambda1, lambda2, delta)
  T = ot.partial.entropic_partial_wasserstein(a, b, D, lambda2, m, numItermax)
  return T

def entropic_opw_2(a, b, D, lambda1, lambda2, delta=1, m=None, numItermax=1000, dropBothSides = False):
  '''
  Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in
  optimal transport and Monge-Ampere obstacle problems. Annals of
  mathematics, 673-730.
  '''
  D = opw_distance(D, lambda1, lambda2, delta)
  if dropBothSides:
    a,b,D = POT_feature_2sides(a,b,opw_distance(D),m)
  else:
    #drop side b
    a,b,D = POT_feature_1side(a,b,opw_distance(D),m)
  
  T = ot.sinkhorn(a, b, D,lambda2)
  return np.sum(T*D)

def entropic_opw_3(a, b, D, lambda1, lambda2, delta=1, m=None, numItermax=1000, dropBothSides = False):
  '''
  Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in
  optimal transport and Monge-Ampere obstacle problems. Annals of
  mathematics, 673-730.
  '''
  D = opw_distance(D, lambda1, lambda2, delta)
  if dropBothSides:
    a,b,D = POT_feature_2sides(a,b,opw_distance(D),m)
  else:
    #drop side b
    a,b,D = POT_feature_1side(a,b,opw_distance(D),m)
  
  T = ot.partial.entropic_partial_wasserstein(a, b, D, lambda2, m, numItermax)
  return np.sum(T*D)