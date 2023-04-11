import rpy2.robjects.packages as rpackages

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1) # 任意のCRANミラーを選択
utils.install_packages("mclust") # mclust パッケージをインストール
utils.install_packages("fpc") # nselectboot
