import numpy as np
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# classification_error関数を呼び出す
mclust = rpackages.importr('mclust')
classification_error = robjects.r('classError')


# 普通のパターン
true = np.array([0,1,2,2,1])
pred = np.array([0,1,2,2,0])
rtrue = numpy2ri.py2rpy(true)
rpred = numpy2ri.py2rpy(pred)
res = classification_error(rtrue, rpred)
print(res) # miss 5, errorRate 0.2 -> -> pred配列5個目(値1)がミス分類
print('---')

# 全部正解パターン
true = np.array([0,1,2,2,1])
pred = np.array(['a','b','c','c','b'])
rtrue = numpy2ri.py2rpy(true)
rpred = numpy2ri.py2rpy(pred)
res = classification_error(rtrue, rpred)
print(res) # miss None, errorRate 0
print('---')


# 2個目の1がcに誤判別されたパターン
true = np.array([0,1,2,2,1])
pred = np.array(['a','c','c','c','b'])
rtrue = numpy2ri.py2rpy(true)
rpred = numpy2ri.py2rpy(pred)
res = classification_error(rtrue, rpred)
print(res) # miss 2, errorRate 0.2 -> pred配列2個目('c')がミスの分類という意味, 2->cが最も多い写像なので正解として誤判定を計算しているのだろう
print('---')

# 真のラベル1と2がそれぞれaとbそれぞれに2個ずつ写像されたとき(半々なので、どちらの写像が正解なのか不明)
true = np.array([0,1,2,2,1])
pred = np.array(['a','b','b','c','c'])
rtrue = numpy2ri.py2rpy(true)
rpred = numpy2ri.py2rpy(pred)
res = classification_error(rtrue, rpred)
print(res) # miss 3 4, errorRate 0.4 -> pred配列3,4個目がミス分類 (予想)アルゴリズム的に最初の1->bを正解の写像としているから、pred3番目がミスとでている. pred4番目は1->cを正解写像としているから2->cを誤判定としてそう. つまり、predのラベル毎にtrueの写像を再定義している. miss (3 5) or (1 4) になると思ったが、missclassifiedはあんまり参考にならんかも？
print('---')


# サブクラスがあるパターン
true = np.array([0,0,0,1,1,1,1,1,1,1])
pred = np.array(['a','a','a','b','b','b','c','c','c'])
rtrue = numpy2ri.py2rpy(true)
rpred = numpy2ri.py2rpy(pred)
res = classification_error(rtrue, rpred)
print(res) # miss None, errorRate 0 -> bに分類されたのは1つのラベルのみ、cに分類されたもの1つのラベルのみ、従って誤判定率は0
print('---')
