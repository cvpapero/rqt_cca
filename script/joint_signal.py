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

#固有ベクトルの可視化(2016.1.7)

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


class GRAPH(QtGui.QWidget):
    def __init__(self):
        super(GRAPH, self).__init__()

    def rhoPlot(self, cMat, filename, winsize, framesize):

        dlen = len(ccaMat)
        pl.clf()
        pl.ion()

        #np.zeros([self.dataDimen, self.dataDimen])     
        """
        tmpMat = np.array(ccaMat[0]) 
        for i in range(dlen-1):
            tmpMat = np.r_[tmpMat,ccaMat[i+1]]
            print "tmpmat:"+str(i)
            print tmpMat
        ccaMat = tmpMat #np.matrix(ccaMat)
        """
        print "ccaMat:"
        print cMat

        #x = pl.arange(dlen)
        #y = pl.arange(dlen)
        print "dlen:"+str(dlen)
        Y,X = np.mgrid[slice(0, dlen, 1),slice(0, dlen, 1)]

        #print "len X:"+str(len(X))+", X:"+str(X)
        #X, Y = np.mgrid[0:dlen:complex(0, dlen), 0:dlen:complex(0, dlen)]

        #X, Y = pl.meshgrid(x, y)
        pl.pcolor(X, Y, cMat)
        pl.xlim(0,dlen-1)
        pl.ylim(0,dlen-1)
        #pl.pcolormesh(X, Y, ccaMat)
        pl.colorbar()
        pl.gray()
        pl.draw()
        outname = str(filename) + "_" + str(winsize) + "_" + str(framesize)+".png"
        pl.savefig(outname)


    def vecPlot(self, row, col, mWx, mWy, frmSize, winSize):
        pl.clf()
        pl.ion()

        #pWy = mWy[row][col]
        #まずはx方向のWx
        pWx = mWx[row,col:(frmSize-winSize+1)+row,0,:]
        pWx = np.sqrt(pWx * pWx)
        r,c = pWx.shape
        x = pl.arange(c+1)
        y = pl.arange(r+1)
        X, Y = pl.meshgrid(x, y)

        pl.subplot2grid((1,2),(0,0))
        pl.pcolor(X, Y, pWx)
        pl.xlim(0,c)
        pl.ylim(0,r)
        pl.colorbar()
        pl.title("user_1 (t:"+str(row)+")")
        pl.gray()


        pWy = mWy[row,col:(frmSize-winSize+1)+row,0,:]
        pWy = np.sqrt(pWy * pWy)
        r,c = pWx.shape
        x = pl.arange(c+1)
        y = pl.arange(r+1)
        X, Y = pl.meshgrid(x, y)

        pl.subplot2grid((1,2),(0,1))
        pl.pcolor(X, Y, pWy)    
        pl.xlim(0,c)
        pl.ylim(0,r)
        pl.colorbar()
        pl.title("user_2 (t:"+str(col)+")")
        pl.gray()

        pl.draw()

        print "pWx shape:",pWx.shape

        #col = pWx[]
        """
        x = pl.arange(Dimen+1)
        y = pl.arange(Range+1)
        X, Y = pl.meshgrid(x, y)

        matWx = []
        matWy = []
        for i in range(len(mWx)):


            #第一固有ベクトルだけ
            matWx.append(Wx[:,0])
            matWy.append(Wy[:,0])

        pl.pcolor(X, Y, vecWx)
        pl.colorbar()
        pl.gray()
        pl.draw()
        """
        #print Wx

       
        #pl.plot() 

    def drawPlot(self, row, col, mWx, mWy, ccaMat, DATAS, dataMaxRange, dataDimen, winSize):
        #def drawPlot(self, row, col):
        print "draw:"+str(row)+", "+str(col)

        #test
        pWx = mWx[row,col:,0,:]
        print "pWx:",pWx
        print "shape:",pWx.shape

        #データの取得
        U1 = []
        U2 = []

        for w in range(winSize):
            U1.append(DATAS[0][row+w])
            U2.append(DATAS[1][col+w])

        nU1 = CCA().stdNorm(U1)
        nU2 = CCA().stdNorm(U2)

        #pl.clf()
        pl.ion()

        Wx = mWx[row][col]
        Wy = mWy[row][col]
        #Wx = mWx[row*dataMaxRange+col]
        #Wy = mWy[row*dataMaxRange+col]

        x = pl.arange(dataDimen+1)
        y = pl.arange(dataDimen+1)
        X, Y = pl.meshgrid(x, y)
       
        
        strCorU1s = []
        strCorU2s = []
        for i in range(dataDimen):
            fU1 = np.dot(nU1.T, Wx[:,i:i+1]).T
            fU2 = np.dot(nU2.T, Wy[:,i:i+1]).T
            strU1 = np.corrcoef(fU1, nU1)
            strU2 = np.corrcoef(fU2, nU2)  
            strCorU1 = np.squeeze(np.asarray(strU1[0:1,1:]))
            strCorU2 = np.squeeze(np.asarray(strU2[0:1,1:]))
            strCorU1s.append(strCorU1)
            strCorU2s.append(strCorU2)
            
            #print np.dot(Wx[:,i:i+1].T,Wx[:,i:i+1]).mean()

        sWx = np.array(np.matrix(strCorU1s))
        sWy = np.array(np.matrix(strCorU2s))
        
        #print Wx.shape
        #print Wx
        #print sWx.shape
        #print sWx

        pl.subplot2grid((3,2),(0,0),colspan=2)
        pl.xlim(0,dataDimen)
        pl.ylim(0,1)
        pl.xticks(fontsize=10)
        pl.yticks(fontsize=10)
        pl.plot(ccaMat[row][col])
        pl.title("eigen value (user 1:"+str(row)+", user 2:"+str(col)+")",fontsize=11)

        pl.subplot2grid((3,2),(1,0))
        pl.pcolor(X, Y, sWx)
        pl.gca().set_aspect('equal')
        pl.colorbar()
        pl.gray()
        pl.xticks(fontsize=10)
        pl.yticks(fontsize=10)
        pl.title("user 1:"+str(row),fontsize=11)

        pl.subplot2grid((3,2),(1,1))
        pl.pcolor(X, Y, sWy)
        pl.gca().set_aspect('equal')
        pl.colorbar()
        pl.gray()
        pl.xticks(fontsize=10)
        pl.yticks(fontsize=10)
        pl.title("user 2:"+str(col),fontsize=11)

        x = np.linspace(0, winSize-1, winSize)

        #forで回して第三位の正準相関までとる？
        pl.subplot2grid((3,2),(2,0),colspan=2)
        ls = ["-","--","-."]
        rhos = ""
        order = 3
        for i in range(order):
            fU1 = np.dot(nU1.T, Wx[:,i:i+1]).T
            fU2 = np.dot(nU2.T, Wy[:,i:i+1]).T

            fU1 = np.squeeze(np.asarray(fU1))
            fU2 = np.squeeze(np.asarray(fU2))

            rho = round(ccaMat[row][col][i],5)

            pl.plot(x, fU1, label="user1:"+str(rho), linestyle=ls[i])
            pl.plot(x, fU2, label="user2:"+str(rho), linestyle=ls[i])
            rhos += str(rho)+", "

        leg = pl.legend(prop={'size':9})
        leg.get_frame().set_alpha(0.7)
        pl.xticks(fontsize=10)
        pl.yticks(fontsize=10)
        pl.title("canonical variate (eig val:"+rhos.rstrip(", ")+")",fontsize=11)

        #print "fU1:"
        #print fU1
        #print "fU2:"
        #print fU2
        pl.tight_layout()

        pl.draw()
        #pl.show()
        #pl.show() 

        

class CCA(QtGui.QWidget):

    def __init__(self):

        super(CCA, self).__init__()

        #UIの初期化
        self.initUI()

        #ROSのパブリッシャなどの初期化
        rospy.init_node('joint_sig', anonymous=True)
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
        self.winSizeBox.setText('90')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        #frame size
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('110')
        self.frmSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.frmSizeBox.setFixedWidth(100)
        form.addRow('frame size', self.frmSizeBox)

        #selected joints
        self.selected = QtGui.QRadioButton('selected')
        form.addRow('dimension', self.selected)


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


    def jsonInput(self, filename):
        f = open(filename, 'r')
        jsonData = json.load(f)

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

    def selectJoints(self):
        #only 0 index
        idx = [0]
        #idx = [3, 4, 5, 6, 11, 12, 23, 24, 25, 26, 27, 28]
        #idx = [0,1,2,7,8,9,10,13,14,15,16,17,18,19,20,21,22]
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
        
        th = 300
        
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
        self.winSize = int(self.winSizeBox.text())
        self.frmSize = int(self.frmSizeBox.text())


        hitbtn = self.selected.isChecked()
        print "hit:"+str(hitbtn)

        print "exec!"

        #ファイル入力
        filename = self.txtSepFile.text()
        self.jsonInput(filename)

        #使用する関節角度を選択
        if hitbtn == True: 
            self.selectJoints()

        #大きすぎるデータの場合カットする
        self.cutDatas()

        print self.DATAS[0][0]
        print len(self.DATAS[0][0])

        signal_1 = []
        signal_2 = []
        order = 0
        for i in range(len(self.DATAS[0])):
            signal_1.append(self.DATAS[0][i][order]*(180./np.pi))
            signal_2.append(self.DATAS[1][i][order]*(180./np.pi))

    
        pl.plot(signal_1, color='r')
        pl.plot(signal_2, color='b')
        pl.title("joint[0]_r:u1,b:u2")
        pl.show()
        #print datas.shape
        #GRAPH().vecPlot()

        
        #self.canoniExec1()
        self.corrExec(signal_1, signal_2)
        #self.time_setting()
        self.updateTable()
        #self.updateColorTable()
        print "end"


    def updateColorTable(self, cItem):
        print "now viz:"+str(cItem.row())+","+str(cItem.column())

        row = cItem.row()
        col = cItem.column()

        #GRAPH().drawPlot(row, col, self.mWx, self.mWy, self.ccaMat, self.DATAS, self.dataMaxRange, self.dataDimen, self.winSize)
        GRAPH().vecPlot(row, col, self.mWx, self.mWy, self.frmSize, self.winSize)
        

    def updateTable(self):
        GRAPH().rhoPlot(self.ccaMat, self.filename, self.winSize, self.frmSize)

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
                rho = round(self.ccaMat[i][j][0],5)
                rho_data = str(round(self.ccaMat[i][j][0],5))+", "+str(round(self.ccaMat[i][j][1],5))+", "+str(round(self.ccaMat[i][j][2],5))
                if rho > self.threshold:
                    c = rho*255

                self.table.setItem(i, j, QtGui.QTableWidgetItem())
                self.table.item(i, j).setBackground(QtGui.QColor(c,c,c))
                self.table.item(i, j).setToolTip(str(rho_data))
        self.table.setVisible(False)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.table.setVisible(True)

    def corrExec(sig1, sig2):
        self.dataMaxRange = self.datasSize - self.winSize + 1

        self.ccaMat = np.zeros([self.dataMaxRange, self.dataMaxRange, self.dataDimen])

        self.frameRange = self.datasSize - self.frmSize + 1
        self.dataRange = self.frmSize - self.winSize + 1

        for f in range(self.frameRange):
            print "f:"+str(f)+"---"
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
                    # tmp_rho is np.array[]
                    self.ccaMat[t1+f][t2+f] = tmp_rho
                    self.mWx[t1+f][t2+f] = Wxl
                    self.mWy[t1+f][t2+f] = Wyl

        
    def canoniExec1(self):

        self.dataMaxRange = self.datasSize - self.winSize + 1
        self.ccaMat = np.zeros([self.dataMaxRange, self.dataMaxRange, self.dataDimen])

        self.frameRange = self.datasSize - self.frmSize + 1
        self.dataRange = self.frmSize - self.winSize + 1
        
        print "datasSize:"+str(self.datasSize)
        print "dataMaxRange:"+str(self.dataMaxRange)
        print "frameRange:"+str(self.frameRange)
        print "dataRange:"+str(self.dataRange)

        self.mWx = np.zeros([self.dataMaxRange, self.dataMaxRange, self.dataDimen, self.dataDimen])
        self.mWy = np.zeros([self.dataMaxRange, self.dataMaxRange, self.dataDimen, self.dataDimen])

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
                        # tmp_rho is np.array[]
                        self.ccaMat[t1+f][t2+f] = tmp_rho
                        self.mWx[t1+f][t2+f] = Wxl
                        self.mWy[t1+f][t2+f] = Wyl

            else:
                for t1 in range(self.dataRange - 1):
                    USER1=[]
                    USER2=[]
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][f+t1+w])
                        USER2.append(self.DATAS[1][f+self.dataRange-1+w])
                    tmp_rho, Wxl, Wyl = self.canoniCorr(USER1, USER2)
                    self.ccaMat[t1+f][f+self.dataRange-1] = tmp_rho
                    self.mWx[t1+f][f+self.dataRange-1] = Wxl
                    self.mWy[t1+f][f+self.dataRange-1] = Wyl

                for t2 in range(self.dataRange):
                    USER1=[]
                    USER2=[]
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][f+self.dataRange-1+w])
                        USER2.append(self.DATAS[1][f+t2+w])
                    tmp_rho, Wxl, Wyl = self.canoniCorr(USER1, USER2)
                    self.ccaMat[f+self.dataRange-1][f+t2] = tmp_rho
                    self.mWx[f+self.dataRange-1][f+t2] = Wxl
                    self.mWy[f+self.dataRange-1][f+t2] = Wyl

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
        #print "lambs:"
        #print lambs

        Wxl = np.dot(Wx,np.diag(lambs))
        Wyl = np.dot(Wy,np.diag(lambs))
        

        #d = self.bartlettTest(p, q, lambs)

        avesU1 = []
        avesU2 = []
        strCorU1s = []
        strCorU2s = []
        for i in range(p):
            fU1 = np.dot(nU1.T, Wx[:,i:i+1]).T
            fU2 = np.dot(nU2.T, Wy[:,i:i+1]).T
            strU1 = np.corrcoef(fU1, nU1)
            strU2 = np.corrcoef(fU2, nU2)  
            strCorU1 = np.squeeze(np.asarray(strU1[0:1,1:]))
            strCorU2 = np.squeeze(np.asarray(strU2[0:1,1:]))
            strCorU1s.append(strCorU1)
            strCorU2s.append(strCorU2)

            #sqU1 = [x ** 2 for x in strCorU1]
            #sqU2 = [x ** 2 for x in strCorU2]

            #avesU1.append(np.average(sqU1))
            #avesU2.append(np.average(sqU2))

            #print "stu1:"
            #print strCorU1
            #print "sqU1"
            #print sqU1
            #print "stu2:"
            #print strCorU2
            #print sqU2
        
        #print "strMatU1:"+str(strMatU1)
        #if max(avesU1) > 0.08:
        #print "aveU1:"+str(sum(avesU1))
        
        #if max(avesU2) > 0.08:
        #print "aveU2:"+str(sum(avesU2))
            
        #Wxl = np.matrix(strCorU1s)
        #Wyl = np.matrix(strCorU2s)
        #print "wxl2 shape:"+str(Wxl.shape)
        #print "wxl2:"+str(Wxl)
        #ave = np.average(lambs)
        #lambs[0]
        #return float(lambs), Wx, Wy
        return np.array(lambs), Wx, Wy



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

#class TEST(object):

    #corr = CCA()
    #corr2 = GRAPH()


def main():
    app = QtGui.QApplication(sys.argv)
    #test = TEST()
    corr = CCA()
    graph = GRAPH()

    sys.exit(app.exec_())


if __name__=='__main__':
    main()
