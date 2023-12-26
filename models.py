from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from kernels import cauchy_kernel, laplace_kernel, r_quadratic


#SVM with linear kernel function
class Svm_Model():

    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X_train, y_train):

        # train model
        if self.kernel == 'linear':
            parameters = {'C': [0.01, 0.1, 1, 10, 100]}
            svc = svm.SVC(kernel=self.kernel, probability=False)
            self.clf = GridSearchCV(svc, parameters)

        elif self.kernel == cauchy_kernel:
            parameters = {'C': [0.01, 0.1, 1, 10, 100]}
            svc = svm.SVC(kernel=self.kernel, probability=True)
            self.clf = GridSearchCV(svc, parameters)

        elif self.kernel == 'rbf' or self.kernel == 'sigmoid' or self.kernel == laplace_kernel:
            parameters = {'C': [0.01, 0.1, 1, 10, 100],'gamma':[0.01, 0.1, 1, 10, 100]}
            svc = svm.SVC(kernel=self.kernel, probability=True)
            self.clf = GridSearchCV(svc, parameters)

        elif self.kernel == r_quadratic:
            parameters = {'C': [0.01, 0.1, 1, 10, 100],'gamma':[0.01, 0.1, 1, 10, 100]}
            svc = svm.SVC(kernel=self.kernel, probability=True)
            self.clf = GridSearchCV(svc, parameters)

        else:
            raise KeyError("Choose a valid kernel")
        
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        if self.kernel == 'linear':
            y_pred = self.clf.predict(X_test)
        else:
            y_pred = self.clf.predict_proba(X_test)

        return y_pred


#Random forest
class Random_Forest():
    def __init__(self):
        self.parameters = {'n_estimators': [10, 50, 100, 150, 200]}

    def fit(self,X_train,y_train):

        rf = RandomForestClassifier(random_state=123)
        self.clf = GridSearchCV(rf, self.parameters)
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        # get y_predict and find accuracy
        y_pred = self.clf.predict(X_test)

        return y_pred

#KNN
class Knn():
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):

        self.clf = KNeighborsClassifier(self.k)
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)

        return y_pred
