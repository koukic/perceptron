import numpy as np
from matplotlib import pyplot

class NeuralNetwork:

    # コンストラクタ
    def __init__(self, n_input, n_hidden, n_output):
        self.hidden_weight = np.random.random_sample((n_hidden, n_input + 1))   #三次元
        self.output_weight = np.random.random_sample((n_output, n_hidden + 1))
        self.hidden_momentum = np.zeros((n_hidden, n_input + 1))        #weightの値を更新
        self.output_momentum = np.zeros((n_output, n_hidden + 1))
    # 学習
    def train(self, X, T, epsilon, mu, epoch):
        self.error = np.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            for i in range(N):
                x,t = X[i],T[i]
                self.update_weight(x, t, epsilon, mu)

            self.error[epo] = self.calc_error(X, T)
    # 結果の予想
    def predict(self, X):
        N = X.shape[0]
        C = np.zeros(N).astype('int')
        Y = np.zeros((N, X.shape[1]))
        for i in range(N):
            x = X[i]
            z, y = self.forward(x)
            # 大きい方を答えとして出力
            Y[i] = y
            C[i] = y.argmax()

        return (C, Y)
    # 誤差関数のグラフ
    def error_graph(self):
        pyplot.ylim(0.0, 2.0)
        pyplot.plot(np.arange(0, self.error.shape[0]), self.error)
        pyplot.show()


    # 活性化関数（シグモイド関数）
    def f(self, arr):
        return np.vectorize(lambda x: 1.0 / (1.0 + np.exp(-x)))(arr)


    def forward(self, x):
        # z: output in hidden layer, y: output in output layer
        z = self.f(self.hidden_weight.dot(np.r_[np.array([1]), x]))
        y = self.f(self.output_weight.dot(np.r_[np.array([1]), z]))

        return (z, y)

    def update_weight(self, x, t, epsilon, mu):
        z, y = self.forward(x)

        # update output_weight
        output_delta = (y - t) * y * (1.0 - y)
        _output_weight = self.output_weight
        self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * np.r_[np.array([1]), z] - mu * self.output_momentum
        self.output_momentum = self.output_weight - _output_weight

        # update hidden_weight
        hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * z * (1.0 - z)
        _hidden_weight = self.hidden_weight
        self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * np.r_[np.array([1]), x]
        self.hidden_momentum = self.hidden_weight - _hidden_weight


    def calc_error(self, X, T):  #エラーの計算
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x,t = X[i],T[i]

            z, y = self.forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0

        return err
    
def main():
      # XORで用いる入力x1,x2の組み合わせ
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # 答えが0である確立と１である確立の組み合わせ
    T = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    N = X.shape[0] # number of data  行

    input_size = X.shape[1]  #二次元　列
    hidden_size = 2 #隠れ層
    output_size = 2
    epsilon = 0.1
    mu = 0.9
    epoch = 10000  #学習回数

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X, T, epsilon, mu, epoch)  #学習開始
    nn.error_graph()   #誤差関数のグラフ

    C, Y = nn.predict(X)
    # 答えを表示
    for i in range(N):
        x , y , c = X[i] , Y[i] , C[i]
        print(x,y,c)
  
    
if __name__ == "__main__":
    main()

