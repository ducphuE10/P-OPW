from utils import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from OPW import opw
import numpy as np


delta,lambda1,lambda2 = 1,1,0.1 #on UCR dataset

X_train, y_train = get_data('FacesUCR/FacesUCR_TRAIN.txt')
X_test, y_test = get_data('FacesUCR/FacesUCR_TEST.txt')

X_train = np.array(X_train)
X_test = np.array(X_test)


neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)

print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# x_0 = np.array(X_train[0]).reshape(-1,1)
# y_0 = np.array(X_test[0]).reshape(-1,1)

