#!/usr/bin/python
# -*- coding: utf-8 -*-

#一定フレーム内同士で計算する
#それ以外では計算しない(たぶん、関連が無いから)
#バートレット検定で第何位までの成分を使うか決める(未)
#ave=0, Sxx=Iを実装(2015.12.11)

#写像して(U1*Wx,U2*W2)相関をとったけど、やはり1になる。検算としては間違ってないかと思われる
#では、動いていない関節が強調されているのでは無いか？調べてみる(2015.12.14)

#ベクトルの足し算
#固有値固有ベクトルの大きい順に並べ替え(2015.12.15)

import sys
import os.path
import math
import json
import time

import numpy as np
from numpy import linalg as NLA
import scipy as sp
from scipy import linalg as SLA

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

#from pylab import *
import pylab as pl

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA

class CCA(QtGui.QWidget):

    def __init__(self):

        super(CCA, self).__init__()

        #UIの初期化
        self.initUI()

        #ROSのパブリッシャなどの初期化
        rospy.init_node('roscca', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.ppub = rospy.Publisher('joint_diff', PointStamped, queue_size=10)

        #rvizのカラー設定(未)
        self.carray = []
        clist = [[1,1,0,1],[0,1,0,1],[1,0,0,1]]
        for c in clist:
            color = ColorRGBA()
            color.r = c[0]
            color.g = c[1]
            color.b = c[2]
            color.a = c[3]
            self.carray.append(color) 

    def initUI(self):
        grid = QtGui.QGridLayout()
        form = QtGui.QFormLayout()
        
        #ファイル入力ボックス
        self.txtSepFile = QtGui.QLineEdit()
        btnSepFile = QtGui.QPushButton('...')
        btnSepFile.setMaximumWidth(40)
        btnSepFile.clicked.connect(self.chooseDbFile)
        boxSepFile = QtGui.QHBoxLayout()
        boxSepFile.addWidget(self.txtSepFile)
        boxSepFile.addWidget(btnSepFile)
        form.addRow('input file', boxSepFile)
        
        #window size
        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('40')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        #frame size
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('40')
        self.frmSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.frmSizeBox.setFixedWidth(100)
        form.addRow('frame size', self.frmSizeBox)

        #tableに表示する相関係数のしきい値
        self.ThesholdBox = QtGui.QLineEdit()
        self.ThesholdBox.setText('0.0')
        self.ThesholdBox.setAlignment(QtCore.Qt.AlignRight)
        self.ThesholdBox.setFixedWidth(100)

        btnUpDate =  QtGui.QPushButton('update')
        btnUpDate.setMaximumWidth(100)
        btnUpDate.clicked.connect(self.updateTable)
        boxUpDate = QtGui.QHBoxLayout()
        boxUpDate.addWidget(self.ThesholdBox)
        boxUpDate.addWidget(btnUpDate)
        form.addRow('corr theshold', boxUpDate)

        """
        self.AnimeDelayBox =  QtGui.QLineEdit()
        self.AnimeDelayBox.setText('0')
        self.AnimeDelayBox.setAlignment(QtCore.Qt.AlignRight)
        self.AnimeDelayBox.setFixedWidth(100)

        btnAnime =  QtGui.QPushButton('create')
        btnAnime.setMaximumWidth(100)
        btnAnime.clicked.connect(self.animetionTable)
        boxAnime = QtGui.QHBoxLayout()
        boxAnime.addWidget(self.AnimeDelayBox)
        boxAnime.addWidget(btnAnime)
        form.addRow('delay', boxAnime)
        """

        #exec
        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.doExec)
        boxCtrl.addWidget(btnExec)
        
        """
        #update
        boxUDCtrl = QtGui.QHBoxLayout()
        btnUDExec = QtGui.QPushButton('update')
        btnUDExec.clicked.connect(self.updateTable)
        boxUDCtrl.addWidget(btnUDExec)
        """

        #テーブルの初期化
        #horizonはuser2の時間
        self.table = QtGui.QTableWidget(self)
        self.table.setColumnCount(0)
        self.table.setHorizontalHeaderLabels("use_2 time") 
        jItem = QtGui.QTableWidgetItem(str(0))
        self.table.setHorizontalHeaderItem(0, jItem)

        #アイテムがクリックされたらグラフを更新
        self.table.itemClicked.connect(self.updateColorTable)
        self.table.setItem(0, 0, QtGui.QTableWidgetItem(1))

        boxTable = QtGui.QHBoxLayout()
        boxTable.addWidget(self.table)
 
        #配置
        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)
        #grid.addLayout(boxUDCtrl,3,0)
        grid.addLayout(boxTable,3,0)

        self.setLayout(grid)
        self.resize(400,100)

        self.setWindowTitle("cca window")
        self.show()
        
        #self.initColorTable()


    def chooseDbFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile.setText(f)
                return
        return self.txtSepFile.setText('')


    def jsonInput(self):
        filename = self.txtSepFile.text()
        f = open(filename, 'r')
        jsonData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()

        self.DATAS = []

        for user in jsonData:
            #angle
            data = []
            self.datasSize = len(user["datas"])
            self.dataDimen = len(user["datas"][0]["data"])
            for j in range(self.datasSize):
                data.append(user["datas"][j]["data"])

            self.DATAS.append(data)
        """
        print "DATAS[0]:"
        print self.DATAS[0]
        print "DATAS[1]:"
        print self.DATAS[1]
        """

    def selectJoints(self):
        idx = [3, 4, 5, 6, 11, 12, 23, 24, 25, 26, 27, 28]
        self.dataDimen = len(idx)
        #idx = [1, 3]
        datas = []
        for data in self.DATAS:
            #print "data:"+str(data)
            uses = []
            for d in data:
                use = []
                for i in idx:
                    use.append(d[i])
                uses.append(use)
            #print "uses:"+str(uses)
            datas.append(uses)

        self.DATAS = datas


    def cutDatas(self):
        
        th = 250
        
        if self.datasSize > th:
            datas = []
            for data in self.DATAS:
                uses = []
                for i,d in enumerate(data):
                    if i < th:
                        uses.append(d)
                    else:
                        break
                datas.append(uses)
            self.DATAS = datas
            #print "datas:"+str(datas)
            self.datasSize = len(self.DATAS[0])

    def doExec(self):

        print "exec!"

        #ファイル入力
        self.jsonInput()

        #使用する関節角度を選択
        self.selectJoints()

        #大きすぎるデータの場合カットする
        self.cutDatas()

        self.winSize = int(self.winSizeBox.text())
        self.frmSize = int(self.frmSizeBox.text())

        self.canoniExec1()
        #self.time_setting()
        self.updateTable()
        #self.updateColorTable()
        print "end"

    """
    def animetionTable(self):
        delay = int(self.AnimeDelayBox.text())
        
        print "delay:"+str(delay)
        #pl.clf()

        #fig, ax = pl.subplots()
        #pl.subplots_adjust(left=0.25, bottom=0.25)

        pl.ion()
        x = pl.arange(self.dataDimen+1)
        y = pl.arange(self.dataDimen+1)
        self.anim_X, self.anim_Y = pl.meshgrid(x, y)
        
        #axfreq = plt.axes([0, 0, self.dataMaxRange, self.dataMaxRange], axisbg=axcolor)
        t=0
        self.sliderColorTable(t, delay)

    def sliderColorTable(self, t, delay):

        pl.clf()
        
        pl.subplot(1,2,1)
        pl.pcolor(self.anim_X, self.anim_Y, self.mWx[delay+(1+self.dataMaxRange)*t])
        pl.gca().set_aspect('equal')
        pl.colorbar()
        pl.gray()
        pl.title("user 1")
        
        pl.subplot(1,2,2)
        pl.pcolor(self.anim_X, self.anim_Y, self.mWy[delay+(1+self.dataMaxRange)*t])
        pl.gca().set_aspect('equal')
        pl.colorbar()
        pl.gray()
        pl.title("user 2")
        #pl.tight_layout()
        
        pl.draw()
        #time.sleep(0.01)
        
        #pl.show()
        #pl.show(block=False) 
    """


    def updateColorTable(self, cItem):
        print "now viz!"+str(cItem.row())+","+str(cItem.column())

        row = cItem.row()
        col = cItem.column()

        pl.clf()
        #pl.ion()

        Wx = self.mWx[row*self.dataMaxRange+col]
        Wy = self.mWy[row*self.dataMaxRange+col]

        x = pl.arange(self.dataDimen+1)
        y = pl.arange(self.dataDimen+1)
        X, Y = pl.meshgrid(x, y)

        parea = 4
        
        pl.subplot2grid((parea,2),(0,0))
        pl.pcolor(X, Y, Wx)
        pl.gca().set_aspect('equal')
        pl.colorbar()
        pl.gray()
        pl.title("user 1")

        pl.subplot2grid((parea,2),(0,1))
        pl.pcolor(X, Y, Wy)
        pl.gca().set_aspect('equal')
        pl.colorbar()
        pl.gray()
        pl.title("user 2")
        #pl.tight_layout()
        

        #データの取得
        U1 = []
        U2 = []

        for w in range(self.winSize):
            U1.append(self.DATAS[0][row+w])
            U2.append(self.DATAS[1][col+w])

        nU1 = self.stdNorm(U1)
        nU2 = self.stdNorm(U2)
        x = np.linspace(0, self.winSize-1, self.winSize)


        #forで回して第三位の正準相関までとる？
        for i in range(3):
            pl.subplot2grid((parea,2),(i+1,0),colspan=2)
            fU1 = np.dot(nU1.T, Wx[:,i:i+1]).T
            fU2 = np.dot(nU2.T, Wy[:,i:i+1]).T
            fU1 = np.squeeze(np.asarray(fU1))
            fU2 = np.squeeze(np.asarray(fU2))
 
            pl.scatter(x, fU1, color='r')
            pl.scatter(x, fU2, color='b')

        #print "fU1:"
        #print fU1
        #print "fU2:"
        #print fU2

        pl.draw()
        #pl.show()
        pl.show(block=False) 

    def updateTable(self):
        
        self.threshold = float(self.ThesholdBox.text())
        
        if(len(self.ccaMat)==0):
            print "No Corr Data! Push exec button..."

        self.table.clear()
        font = QtGui.QFont()
        font.setFamily(u"DejaVu Sans")
        font.setPointSize(5)
        
        self.table.horizontalHeader().setFont(font)
        self.table.verticalHeader().setFont(font)

        self.table.setRowCount(len(self.ccaMat))
        self.table.setColumnCount(len(self.ccaMat))


        for i in range(len(self.ccaMat)):
            jItem = QtGui.QTableWidgetItem(str(i))
            self.table.setHorizontalHeaderItem(i, jItem)

        hor = True

        for i in range(len(self.ccaMat)):
            iItem = QtGui.QTableWidgetItem(str(i))
            self.table.setVerticalHeaderItem(i, iItem)
            self.table.verticalHeaderItem(i).setToolTip(str(i))
            #時間軸にデータを入れるなら↓
            #self.table.verticalHeaderItem(i).setToolTip(str(self.timedata[i]))
            for j in range(len(self.ccaMat[i])):
                
                if hor == True:
                    jItem = QtGui.QTableWidgetItem(str(j))
                    self.table.setHorizontalHeaderItem(j, jItem)
                    self.table.horizontalHeaderItem(j).setToolTip(str(j))
                    hot = False
                
                c = 0
                if self.ccaMat[i][j] > self.threshold:
                    c = self.ccaMat[i][j]*255

                self.table.setItem(i, j, QtGui.QTableWidgetItem())
                self.table.item(i, j).setBackground(QtGui.QColor(c,c,c))
                self.table.item(i, j).setToolTip(str(self.ccaMat[i][j]))
        self.table.setVisible(False)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.table.setVisible(True)


    def canoniExec1(self):

        self.dataMaxRange = self.datasSize - self.winSize + 1
        self.ccaMat = [[0 for i in range(self.dataMaxRange)] for j in range(self.dataMaxRange)]

        self.frameRange = self.datasSize - self.frmSize + 1
        self.dataRange = self.frmSize - self.winSize + 1
        
        print "datasSize:"+str(self.datasSize)
        print "dataMaxRange:"+str(self.dataMaxRange)
        print "frameRange:"+str(self.frameRange)
        print "dataRange:"+str(self.dataRange)

        #self.mWxList = np.zeros([self.dataDimen, self.dataDimen])
        #self.mWyList = np.zeros([self.dataDimen, self.dataDimen])

        self.output = 1

        self.mWx = [np.zeros([self.dataDimen, self.dataDimen]) for i in range(self.dataMaxRange*self.dataMaxRange)]
        self.mWy = [np.zeros([self.dataDimen, self.dataDimen]) for i in range(self.dataMaxRange*self.dataMaxRange)]

        for f in range(self.frameRange):
            print "f:"+str(f)+"---"
            if f == 0:
                for t1 in range(self.dataRange):
                    rho = 0
                    time1 = 0
                    time2 = 0
                    for t2 in range(self.dataRange):
                        USER1 = []
                        USER2 = []
                        for w in range(self.winSize):
                            USER1.append(self.DATAS[0][t1+f+w])
                            USER2.append(self.DATAS[1][t2+f+w])

                        tmp_rho, Wxl, Wyl = self.canoniCorr(USER1, USER2)
                        self.ccaMat[t1+f][t2+f] = tmp_rho
                        self.mWx[(t2+f)+(t1+f)*self.dataMaxRange] = Wxl
                        self.mWy[(t2+f)+(t1+f)*self.dataMaxRange] = Wyl

                        """
                        if math.fabs(tmp_rho) > math.fabs(rho):
                            rho = tmp_rho                
                            time1 = t1
                            time2 = t2
                        """
            else:
                for t1 in range(self.dataRange - 1):
                    USER1=[]
                    USER2=[]
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][f+t1+w])
                        USER2.append(self.DATAS[1][f+self.dataRange-1+w])
                    tmp_rho, Wxl, Wyl = self.canoniCorr(USER1, USER2)
                    self.ccaMat[f+t1][f+self.dataRange-1] = tmp_rho
                    self.mWx[(f+self.dataRange-1)+(f+t1)*self.dataMaxRange] = Wxl
                    self.mWy[(f+self.dataRange-1)+(f+t1)*self.dataMaxRange] = Wyl
                    #self.ccaMat[f+t1][f+self.dataRange-1] = float(tmp_rho) 
                for t2 in range(self.dataRange):
                    USER1=[]
                    USER2=[]
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][f+self.dataRange-1+w])
                        USER2.append(self.DATAS[1][f+t2+w])
                    tmp_rho, Wxl, Wyl = self.canoniCorr(USER1, USER2)
                    self.ccaMat[f+self.dataRange-1][f+t2] = tmp_rho
                    self.mWx[(f+t2)+(f+self.dataRange-1)*self.dataMaxRange] = Wxl
                    self.mWy[(f+t2)+(f+self.dataRange-1)*self.dataMaxRange] = Wyl

        #print "mwx:"
        #print self.mWxList
        #print "mwy:"
        #print self.mWyList

    #正規化
    def stdNorm(self, U):

        mat = np.matrix(U).T
        mat = mat - mat.mean(axis=1)
        mcov = np.cov(mat)
        p,l,pt = NLA.svd(mcov)
        lsi = SLA.sqrtm(SLA.inv(np.diag(l))) 
        snU =  np.dot(np.dot(lsi, p.T), mat)

        #print "cov:"
        #print np.cov(snU)

        return snU

    def maxIdx(self, U):
        pass

    """
    正準相関
    U1, U2はn*p行列, n*q行列(nは二群で同値)
    U=[[u_11,u_21,...,u_p1],[u_12,u_22,...,u_p2],...,[u_n1,u_n2,...,u_np]]
    (例)
    [(n=4)*(p=3)行列]は以下のformで代入する
    [[69, 82, 33], [39, 94, 48], [34, 90, 80], [26, 61, 49]]
    """
    def canoniCorr(self, U1, U2):

        #ave=0, Sxx=I
        nU1 = self.stdNorm(U1)
        nU2 = self.stdNorm(U2)

        data = np.r_[nU1, nU2]
        p,n = nU1.shape
        q,n = nU2.shape
        S = np.cov(data)

        Sxx = S[:p,:p]
        Sxy = S[:p,p:]
        Syy = S[p:,p:]

        A = np.dot(Sxy, Sxy.T)
        #固有値問題
        #lambs, Wx = SLA.eigh(A)
        lambs, Wx = NLA.eig(A)

        idx = lambs.argsort()[::-1]
        lambs = np.sqrt(lambs[idx])
        Wx=Wx[:,idx]
        Wy = np.dot(np.dot(Sxy.T, Wx), SLA.inv(np.diag(lambs)))
        print "lambs:"
        print lambs

        
        #wwx = np.fabs(Wx[:,0:1])
        #wwy = np.fabs(Wy[:,0:1])
        #maxWxIdx = max(enumerate(wwx), key=lambda x:x[1])[0]
        #maxWyIdx = max(enumerate(wwy), key=lambda x:x[1])[0]
        #print "maxWxIdx:"
        #print maxWxIdx
        #print "maxWyIdx:"
        #print maxWyIdx
        #print Wx[:,0:1]
        #print Wx[:,0:1].shape

        #Wxl = np.dot(np.fabs(Wx),np.diag(lambs))
        #Wyl = np.dot(np.fabs(Wy),np.diag(lambs))

        Wxl = np.dot(Wx,np.diag(lambs))
        Wyl = np.dot(Wy,np.diag(lambs))

        """
        print "wwx:"
        print wwx
        print "wwy:"
        print wwy
        """


    
        #d = self.bartlettTest(p, q, lambs)

        #fU1 = np.dot(U1.T, Wx[:,0:1]).T
        #fU2 = np.dot(U2.T, Wy[:,0:1]).T
        #fU1 = np.dot(nU1.T, Wx[:,0:1]).T
        #fU2 = np.dot(nU2.T, Wy[:,0:1]).T
        #fU1 = np.squeeze(np.asarray(fU1))
        #fU2 = np.squeeze(np.asarray(fU2))
        #print "fU1:"
        #print fU1
        #print "fU2:"
        #print fU2  
        """
        if self.output == 1:
            #fU1 = np.dot(nU1.T, Wx)
            #fU2 = np.dot(nU2.T, Wy)
            fU1 = np.dot(nU1.T, Wx[:,0:1]).T
            fU2 = np.dot(nU2.T, Wy[:,0:1]).T
            fU1 = np.squeeze(np.asarray(fU1))
            fU2 = np.squeeze(np.asarray(fU2))
            print "fU1:"
            print fU1
            print "fU2:"
            print fU2
            
            x = np.linspace(0, self.winSize-1, self.winSize)
            pl.scatter(x, fU1, color='r')
            pl.scatter(x, fU2, color='b')
            pl.show()
            self.output = 0
        """

        """
        print "fU1:"
        print fU1
        print "fU2:"
        print fU2
        data_t = np.r_[fU1, fU2]
        print "cov:"
        print np.cov(data_t)
        """
        return float(lambs[0]), Wxl, Wyl



    def bartlettTest(self, rows, cols, lambs):
        lSize = len(lambs)
        M = rows-1/2*(cols+cols+3)

        for i in range(lSize):
            #有意水準を求める
            alf = 0.05
            sig = sp.special.chdtri((cols-i+1)*(cols-i+1), alf)
            w = 1
            for j in range(i, lSize):
                w = w*(1-lambs[lSize-j-1])
                #print "j:"+str(j)+", lam:"+str(lambs[lSize-j-1])+", w:"+str(w)
            print "w:"+str(w)
            
            bart = M*math.log(w)
            #print  "bart["+str(i)+"]:"+str(bart) +" > sig("+str(alf)+"):"+str(sig)
        
            if bart > sig:
                print  "bart["+str(i)+"]:"+str(bart) +" > sig("+str(alf)+"):"+str(sig)
            else:
                break
        print "i:"+str(i)

        return i

def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()