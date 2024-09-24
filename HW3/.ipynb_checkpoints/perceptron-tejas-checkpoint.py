#Tejas Acharya

import numpy as np

class Perceptron():
    def __init__(self, eta, i_max):
        self.eta = eta
        self.I_MAX = i_max

    def fit(self, X, y, is_version_1=True):
        self.i = 0
        self.C = len(np.unique(y))
        self.N, self.D = X.shape
        self.weights = []

        initial_weight = np.ones((self.C, self.D))

        self.weights.append(initial_weight)

        data = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

        self.version = is_version_1
        if (self.version):
            self.sgd_1(data)
        else:
            self.sgd_2(data)

        print('Fitting Done')
        return
    
    def sgd_1(self, data):
        halt_condition = False
        num_epochs = 0

        while not halt_condition:
            data = np.random.permutation(data)
            X = data[:, :-1]
            y = data[:, -1].astype('int32')
            for i in range(self.N):
                x_i, y_i = X[i], y[i]
                new_weight = self.get_new_weight(x_i, y_i)
                self.weights.append(new_weight)
                self.i += 1
            num_epochs += 1
            halt_condition = self.has_halted(X, y)
    
        return

    def get_new_weight(self, x, y):
        current_g = self.get_g(x)
        k = y
        if (np.argmax(current_g) == k):
            return self.weights[-1]
        else:
            old_weight = self.weights[-1]
            new_weight = np.empty_like(old_weight)

            new_weight[k] = old_weight[k] + (self.eta * x)
            
            g_not_k = current_g[current_g != current_g[k]]
            
            l = k

            if len(g_not_k) == 0:
                while l == k:
                    l = np.random.randint(0, self.C)
            else:
                l = np.argmax(g_not_k)
            
            new_weight[l] = old_weight[l] - (self.eta * x)
            
            for m in range(self.C):
                if (m != k) and (m != l):
                    new_weight[m] = old_weight[m]
            
            return new_weight
    

    def get_g(self, x):
        g = np.dot(self.weights[-1], x)
        print(g)
        return g


    def has_halted(self, X, y):
        if self.version:
            if self.i >= self.I_MAX:
                # last_100_weights = []
                # for i in range(self.C):
                #     last_100_weights.append(self.weights[i][-100:, :])
                # error_rate = np.zeros((100, ))

                # for i in range(100):
                #     num_incorrect = 0
                #     for j in range(self.N):
                #         g = np.zeros((self.C, ))
                #         for k in range(self.C):
                #             g[k] = np.dot(last_100_weights[k][i, :].T, X[j, :])
                #         if np.argmax(g) != y[j]:
                #             num_incorrect += 1
                #     error_rate[i] = num_incorrect
                # i0 = np.argmin(error_rate)
                # print(f'Best Iteration: {i0}')
                # print(f'Best Error rate: {error_rate[i0] / self.N}')  
                return True
            else:
                # last_epoch_weights = []
                # for i in range(self.C):
                #     last_epoch_weights.append(self.weights[i][-self.N:, :])
                # a = np.all(np.isclose(last_epoch_weights[:, : , :], last_epoch_weights[:, -1, :]))
                # print(a)
                return False
        else:
            pass

    def is_epoch_weight_updated(self):
        prev_weight = self.weights[-self.N - 1]
        current_weight = None
        for i in self.weights[self.N:]:
            current_weight = i
            if current_weight != prev_weight:
                return True
            else:
                prev_weight = current_weight
        return False
    
    def predict(self, x):
        g_hat = np.dot(self.weights[-1], x)
        return np.argmax(g_hat) + 1