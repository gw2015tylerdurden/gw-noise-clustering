import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri


def classError(true, pred):
    # 初回実行時のみTrueにしてインストールする
    if True:
        utils = rpackages.importr('utils')
        utils.install_packages('mclust')

    # classification_error関数を呼び出す
    mclust = rpackages.importr('mclust')
    classification_error = robjects.r('classError')

    rtrue = numpy2ri.py2rpy(true)
    rpred = numpy2ri.py2rpy(pred)
    res = classification_error(rtrue, rpred)
    error_rate = res.rx2('errorRate')[0]
    print('error_rate,' + error_rate)
