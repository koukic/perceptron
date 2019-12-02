import numpy as np
import matplotlib.pylab as plt

# パーセプトロンクラス
class PerceptronClassifier:

    def __init__(self, alpha, t , x):
        self.alpha = alpha
        self.weight = np.random.uniform(-1.0, 1.0, 3)   # -1.0～1.0の乱数を３つ
        self.x = x
        self.t = t
        # 点を描画
        self.plot_pixels()
    # 点を描画する    
    def plot_pixels(self):
        # 点を画面に描画
        for p,type in zip(self.x,self.t):   #zip関数使っているよ
            print(p,":",type)
            if type == 1:
                plt.plot(p[0],p[1],"o",color="b") # 1は青い○p[0]とp[1]はx,y
            else:
                plt.plot(p[0],p[1],"x",color="r") # 0は赤い×
                
    #　学習
    def learn(self):
        updated = True      #更新が必要かどうか初期値
        n = 0
        while updated:      #updateがtrueならずっと
            updated = False         #一旦
            for category, features in zip(self.t, self.x):
                predict = self.classify(features)   # 点が上か下かを評価
                if predict != category:
                    # 線の描画
                    self.plot_line(n,False)  
                    # 重みの更新
                    t = 2 * (category - 0.5)    # category0なら-1、category1なら1
                    self.weight = self.weight + self.alpha * t * np.append(features, 1)                   #重みの調整     
                    updated = True
                    n = n + 1
        # 確定した線を描画する
        self.plot_line(n,True)
        
    # 線の表示
    def plot_line(self,n,last):
         print(n,":",self.weight)
         plt.xlim([-0.1,1.1]) # Xの範囲は-0.1から1.1
         plt.ylim([-0.1,1.1]) # yの範囲は-0.1から1.1
         if self.weight[1] != 0:
             x = np.arange(-0.1,1.1,0.5)  # xの値域(0, 1, 2, 3)
             y = -self.weight[0] / self.weight[1] * x - self.weight[2] / self.weight[1]
         elif self.weight[0] != 0:
             y = np.arrange(-0.1,1.1,0.1)
             x = self.weight[2] / self.weight[0]
         else:
             x = 0
             y = 0
         if last == True:
             plt.plot(x,y,"k-")      # 黒の直線を引く
         else:
             plt.plot(x,y,"g-",linestyle="dotted")      # 緑の直線を引く
             
    #　分類
    def classify(self, features):
        score = np.dot(self.weight, np.append(features, 1)) # 関数による評価
        # ステップ関数で分類
        return self.f(score);

    # 活性化関数（ステップ関数）
    def f(self,x):
        if x > 0:
            return 1
        else:
            return 0
        
    # 処理結果の表示
    def plot_show(self):
        plt.show()


def main():
    # 点の座標
    x = np.array([[0, 0],[0,1],[1,0],[1,1]])
    # 手の野種類（○：1 ×:0) 上のそれぞれの座標を評価
    t = np.array([0,1,1,1])    
    # サイズを２にして、αを0,1二設定
    classifier = PerceptronClassifier(0.05,t,x)
    # 学習フェーズ
    classifier.learn()    
    # 結果の描画
    classifier.plot_show()

if __name__ == "__main__":
    main()
