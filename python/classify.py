import pickle
import random
import numpy as np

from sklearn import metrics
from sklearn.svm import SVC
from matplotlib import pyplot as plt

import config

from data import load_data


def classify():
    dttr = HandDTTR()

    dttr.fit()

    print('[*] Saving model')
    dttr.save()
    print('[*] Saved!')

    print(dttr.validate())



class HandDTTR:

    def __init__(self):
        self.svc = SVC(C=0.8)
        self.x = []
        self.y = []
        self.X_train, self.Y_train = [], []
        self.test_size = 100
        self.train_size = 1900

    def fit(self, verbose=1):
        if verbose:
            print('[*] Loading training data')

        data = load_data()
        random.shuffle(data)

        if verbose:
            print('[*] Processing training data')

        for item in data:
            self.x.append(np.array(item[0]).flatten())
            self.y.append(item[1])

        self.X_train = np.array(self.x[:self.train_size])
        self.Y_train = np.array(self.y[:self.train_size])

        if verbose:
            print('[*] Fitting SVC')

        self.svc.fit(self.X_train, self.Y_train)

        if verbose:
            print('[*] Done!')

    def validate(self) -> float:
        test_x = self.x[self.train_size:self.train_size + self.test_size]
        test_y = self.y[self.train_size:self.train_size + self.test_size]

        predictions = self.predict(test_x)
        acc = metrics.accuracy_score(test_y, predictions)

        return acc

    def predict(self, x):
        return self.svc.predict(x)

    def save(self, path='HandDTTR.model'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path='HandDTTR.model') -> object:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def test(self):
        img = self.X_train[random.randint(1, 100)].reshape(config.WIDTH, config.HEIGHT, 3)
        plt.imshow(img)
        plt.show()

        print(self.predict([img.flatten()]))




if __name__ == '__main__':
    classify()

