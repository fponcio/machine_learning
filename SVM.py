import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier          #compare SVM and KNN

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2) #testsize is the number of data to test

#print(x_train, y_train)             #0: benign, 1: malignant

classes = ['malignant', 'benign']

#clf = svm.SVC()                         #Support Vector Classification
#clf = svm.SVC(kernel = "linear", C = 2)
#clf = svm.SVC(kernel = "poly")          #compare values
clf = KNeighborsClassifier(n_neighbors = 9)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)