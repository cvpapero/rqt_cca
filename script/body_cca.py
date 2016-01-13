#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
2016.1.11
決定版をつくる

"""

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

import pylab as pl
import rospy

import pyper
import pandas as pd

class GRAPH(QtGui.QWidget):
    def __init__(self):
        super(GRAPH, self).__init__()

    def rhoPlot(self, cMat, filename, winsize, framesize):

        dlen = len(ccaMat)
        pl.clf()
        pl.ion()

        Y,X = np.mgrid[slice(0, dlen, 1),slice(0, dlen, 1)]

        pl.pcolor(X, Y, cMat[:,:,0])
        pl.xlim(0,dlen)
        pl.ylim(0,dlen)

        pl.colorbar()
        pl.gray()
        pl.draw()
        outname = str(filename) + "_" + str(winsize) + "_" + str(framesize)+".png"
        pl.savefig(outname)


    def vecPlot(self, row, col, r_m, mWx, mWy, data1, data2, frmSize, winSize):
        pl.clf()
        pl.ion()

        #pWy = mWy[row][col]
        #まずはx方向のWx
        pWx = mWx[row,col:(frmSize-winSize+1)+row,:,0]
        #print "pWx sum:",np.sum(pWx[0,:])
        #print np.mean(pWx*pWx, axis=0)
        sqWx = np.sqrt(pWx * pWx)
        r,c = sqWx.shape
        x = pl.arange(c+1)
        y = pl.arange(r+1)
        X, Y = pl.meshgrid(x, y)

        pl.subplot2grid((2,2),(0,0))
        pl.pcolor(X, Y, sqWx)
        pl.xlim(0,c)
        pl.ylim(0,r)
        pl.colorbar()
        pl.title("user_1 (t:"+str(row)+")")
        pl.gray()

        
        pWy = mWy[row,col:(frmSize-winSize+1)+row,:,0]
        #print "pWy sum:",np.sum(pWy[0,:],axis=1)
        #print np.sum(pWy*pWy,axis=1)
        #print np.mean(pWy*pWy, axis=0)
        sqWy = np.sqrt(pWy * pWy)
        r,c = sqWx.shape
        x = pl.arange(c+1)
        y = pl.arange(r+1)
        X, Y = pl.meshgrid(x, y)

        pl.subplot2grid((2,2),(0,1))
        pl.pcolor(X, Y, sqWy)    
        pl.xlim(0,c)
        pl.ylim(0,r)
        pl.colorbar()
        pl.title("user_2 (t:"+str(col)+")")
        pl.gray()

        """
        pl.subplot2grid((2,2),(1,0),colspan=2)
        pr, pc = pWx.shape
        for i in range(pc):
            pl.plot(pWx[:,i], color="r")
        """

        xl = np.arange(c)        
        pl.subplot2grid((2,2),(1,0))
        subx1 = np.delete(sqWx,0,0)
        subx2 = np.delete(sqWx,r-1,0)
        print "sum sub:",sum(subx2-subx1)
        pl.bar(xl, np.fabs(np.mean(subx2-subx1,axis=0)))
        #pl.bar(xl, np.std(sqWx, axis=0))
        pl.xlim(0,c)
        #pl.ylim(0,1)

        pl.subplot2grid((2,2),(1,1))
        subx1 = np.delete(sqWy,0,0)
        subx2 = np.delete(sqWy,r-1,0)
        pl.bar(xl, np.fabs(np.mean(subx2-subx1,axis=0)))
        #pl.bar(xl, np.std(sqWy, axis=0))
        pl.xlim(0,c)
        #pl.ylim(0,1)

        """
        pl.subplot2grid((2,2),(1,0),colspan=2)
        U1 = []
        U2 = []
        od = 0
        for w in range(winSize):
            U1.append(data1[row+w][od])
            U2.append(data2[col+w][od])
        pl.plot(U1, color="r")
        pl.plot(U2, color="b")        
        """
        """
        #正準相関変量の表示
        U1 = []
        U2 = []
        for w in range(winSize):
            U1.append(data1[row+w])
            U2.append(data2[col+w])
        U1 = np.array(U1)
        U2 = np.array(U2)
        U1 = U1 - U1.mean(axis=0)
        U2 = U2 - U2.mean(axis=0)

        print "u1 s:",U1.shape
        print "wx s:",mWx[row,col,0,:].shape
    
        ls = ["-","--","-."]
        rhos = ""
        order = 3
        xl = np.linspace(0, winSize-1, winSize)
        for i in range(order):

            fU1 = np.dot(U1, mWx[row,col,i,:]).T
            fU2 = np.dot(U2, mWy[row,col,i,:]).T

            fU1 = np.squeeze(np.asarray(fU1))
            fU2 = np.squeeze(np.asarray(fU2))

            rho = round(r_m[row][col][i],5)

            pl.plot(xl, fU1, label="user1:"+str(rho), linestyle=ls[i])
            pl.plot(xl, fU2, label="user2:"+str(rho), linestyle=ls[i])
            rhos += str(rho)+", "

        leg = pl.legend(prop={'size':9})
        leg.get_frame().set_alpha(0.7)
        pl.title("canonical variate (eig val:"+rhos.rstrip(", ")+")",fontsize=11)
        """
        pl.xticks(fontsize=10)
        pl.yticks(fontsize=10)

        pl.tight_layout()

        pl.draw()

        #print "pWx shape:",pWx.shape


    def drawPlot(self, row, col, mWx, mWy, ccaMat, data1, data2, dataMaxRange, dataDimen, winSize):
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
            U1.append(data1[row+w])
            U2.append(data2[col+w])

        nU1 = CCA().stdn(U1)
        nU2 = CCA().stdn(U2)

        #pl.clf()
        pl.ion()

        Wx = mWx[row][col]
        Wy = mWy[row][col]

        X, Y = pl.meshgrid(pl.arange(dataDimen+1), pl.arange(dataDimen+1))
       
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

        pl.tight_layout()
        pl.draw()


class CCA(QtGui.QWidget):

    def __init__(self):
        super(CCA, self).__init__()
        #UIの初期化
        self.initUI()

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

        #exec
        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.doExec)
        boxCtrl.addWidget(btnExec)

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
        grid.addLayout(boxTable,3,0)

        self.setLayout(grid)
        self.resize(400,100)

        self.setWindowTitle("cca window")
        self.show()

    def chooseDbFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile.setText(f)
                return
        return self.txtSepFile.setText('')

    def updateColorTable(self, cItem):
        r = cItem.row()
        c = cItem.column()
        print "now viz r:",r,", c:",c
        print "cca:",self.r_m[r][c]
        #GRAPH().drawPlot(r, c, self.wx_m, self.wy_m, self.r_m, self.data1, self.data2, self.dtmr, self.dtd, self.wins)
        GRAPH().vecPlot(r, c, self.r_m, self.wx_m, self.wy_m, self.data1, self.data2, self.frms, self.wins)
        
    def updateTable(self):
        #GRAPH().rhoPlot(self.r_m, self.filename, self.wins, self.frms)
        th = float(self.ThesholdBox.text())
        if(len(self.r_m)==0):
            print "No Corr Data! Push exec button..."
        self.table.clear()
        font = QtGui.QFont()
        font.setFamily(u"DejaVu Sans")
        font.setPointSize(5)
        self.table.horizontalHeader().setFont(font)
        self.table.verticalHeader().setFont(font)
        self.table.setRowCount(len(self.r_m))
        self.table.setColumnCount(len(self.r_m))
        for i in range(len(self.r_m)):
            jItem = QtGui.QTableWidgetItem(str(i))
            self.table.setHorizontalHeaderItem(i, jItem)
        hor = True
        for i in range(len(self.r_m)):
            iItem = QtGui.QTableWidgetItem(str(i))
            self.table.setVerticalHeaderItem(i, iItem)
            self.table.verticalHeaderItem(i).setToolTip(str(i))
            #時間軸にデータを入れるなら↓
            #self.table.verticalHeaderItem(i).setToolTip(str(self.timedata[i]))
            for j in range(len(self.r_m[i])):
                if hor == True:
                    jItem = QtGui.QTableWidgetItem(str(j))
                    self.table.setHorizontalHeaderItem(j, jItem)
                    self.table.horizontalHeaderItem(j).setToolTip(str(j))
                    hot = False
                c = 0
                rho = round(self.r_m[i][j][0],5)
                rho_data = str(round(self.r_m[i][j][0],5))+", "+str(round(self.r_m[i][j][1],5))+", "+str(round(self.r_m[i][j][2],5))
                if rho > th:
                    c = rho*255
                self.table.setItem(i, j, QtGui.QTableWidgetItem())
                self.table.item(i, j).setBackground(QtGui.QColor(c,c,c))
                self.table.item(i, j).setToolTip(str(rho_data))
        self.table.setVisible(False)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.table.setVisible(True)

    def doExec(self):
        #print "exec!"

        self.out = 0

        self.data1 = []
        self.data2 = []

        #ws:window_size, fs:frame_size 
        self.wins = int(self.winSizeBox.text())
        self.frms = int(self.frmSizeBox.text())

        hitbtn = self.selected.isChecked()
        #print "setect joints:"+str(hitbtn)
        #input file
        filename = self.txtSepFile.text()
        self.data1, self.data2 = self.jsonInput(filename)
        #select joints
        if hitbtn == True: 
            self.data1, self.data2 = self.selectInput(self.data1, self.data2)
        #if data is big then...
        self.data1, self.data2 = self.cutDatas(self.data1, self.data2, 300)

        #dmr:data_max_range, frmr:frame_range, dtr:data_range
        self.dtmr = self.dts - self.wins + 1
        self.frmr = self.dts - self.frms + 1
        self.dtr = self.frms - self.wins + 1

        #print "datas_size:",self.dts
        #print "data_max_range:",self.dtmr
        #print "frame_range:",self.frmr
        #print "data_range:",self.dtr

        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        self.r_m, self.wx_m, self.wy_m = self.ccaExec(self.data1, self.data2)
        self.updateTable()
        print "end"

    def jsonInput(self, filename):
        f = open(filename, 'r')
        jsonData = json.load(f)
        f.close()
        datas = []
        for user in jsonData:
            #data is joint angle
            data = []
            #dts:data_size, dtd:data_dimension
            self.dts = len(user["datas"])
            self.dtd = len(user["datas"][0]["data"])
            for j in range(self.dts):
                data.append(user["datas"][j]["data"])
            datas.append(data)
        return datas[0], datas[1]

    def selectInput(self, data1, data2):
        idx = [3, 4, 5, 6, 11, 12, 23, 24, 25, 26, 27, 28]
        #idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,0]
        #idx = [1,2,3,0]
        self.dtd = len(idx)
        datas1 = []
        datas2 = []
        for (ds1, ds2) in zip(data1, data2):
            #print "data:"+str(data)
            use1 = []
            use2 = []
            for i in idx:                   
                use1.append(ds1[i])
                use2.append(ds2[i])
            datas1.append(use1)
            datas2.append(use2)
        return datas1, datas2

    def cutDatas(self, data1, data2, th):        
        if self.dts > th:
            datas1 = []
            datas2 = []
            for i,(ds1,ds2) in enumerate(zip(data1,data2)):
                if i < th:
                    datas1.append(ds1)
                    datas2.append(ds2)
                else:
                    break
            self.dts = len(datas1)
            return datas1, datas2
        else:
            return data1, data2


    def ccaExec(self, data1, data2):
        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors

        s = 0
        r_m = np.zeros([self.dtmr, self.dtmr, self.dtd])
        wx_m = np.zeros([self.dtmr, self.dtmr, self.dtd, self.dtd])
        wy_m = np.zeros([self.dtmr, self.dtmr, self.dtd, self.dtd])

        for f in range(self.frmr):
            print "f: ",f,"/",self.frmr-1
            if f == 0:
                for t1 in range(self.dtr):
                    for t2 in range(self.dtr):
                        u1 = []
                        u2 = []
                        for w in range(self.wins):
                            u1.append(data1[t1+f+w])
                            u2.append(data2[t2+f+w])
                        r_m[t1+f][t2+f], wx_m[t1+f][t2+f], wy_m[t1+f][t2+f] = self.ccas(u1, u2, s)
            else:
                od = f+self.dtr-1
                for t1 in range(self.dtr-1):
                    u1 = []
                    u2 = []
                    for w in range(self.wins):
                        u1.append(data1[f+t1+w])
                        u2.append(data2[od+w])
                    r_m[t1+f][od], wx_m[t1+f][od], wy_m[t1+f][od] = self.ccas(u1, u2, s)
                for t2 in range(self.dtr):
                    u1 = []
                    u2 = []
                    for w in range(self.wins):
                        u1.append(data1[od+w])
                        u2.append(data2[f+t2+w])
                    r_m[od][f+t2], wx_m[od][f+t2], wy_m[od][f+t2] = self.ccas(u1, u2, s)
        return r_m, wx_m, wy_m

    def ccas(self, u1, u2, s):
        if s == 0:
            r,x,y = self.cca(u1, u2)
            return r, x, y
        elif s == 1:
            r,x,y = self.cca1(u1, u2)
            return r, x, y
        else:
            r,x,y = self.cca2(u1, u2)
            return r, x, y


    def cca1(self, u1, u2):
        #n = len(u1)
        p = len(u1[0])
        #ave=0, Sxx=I
        #S = np.cov(np.r_[self.stdn(u1), self.stdn(u2)])
        #u1 = np.array(self.stdn(u1))
        #u2 = np.array(u2)
        
        if self.out == 0:
            #print "u1:",u1
            print "len(u1):",len(u1),",len(u1[0]):",len(u1[0])
            #print u1.shape
            stds = self.stdn(u1)
            sr,sc=stds.shape
            print "std u1:",stds
            self.out = 1

        S = np.cov(self.stdn(u1), self.stdn(u2), bias=1)
        Sxy = S[:p,p:]
        lambs, wx = NLA.eig(np.dot(Sxy, Sxy.T))
        #print "wx:",wx
        #順番の入れ替え
        #print "bf lambs:",lambs
        #print "wx:",wx

        idx = lambs.argsort()[::-1]
        lambs = np.sqrt(lambs[idx])
        wx = wx[:,idx]

        #print "idx",idx
        #print "af lambs:",lambs
        #print "wx:",wx
        #wx2 = wx*wx
        #print "wx^2.sum(0):",wx2.sum(axis=0) 
        #print "wx^2.sum(1):",wx2.sum(axis=1)

        wy = np.dot(np.dot(Sxy.T, wx), SLA.inv(np.diag(lambs)))

        #wy2 = wy*wy
        #print "wy^2.sum(0):",wy2.sum(axis=0) 
        #print "wy^2.sum(1):",wy2.sum(axis=1)

        return np.array(lambs), wx, wy
        
    """
    #正規化
    def stdn(self, u):
        mat =  np.array(u).T - np.array(u).T.mean(axis=0)
        #print mat
        #print mat.shape
        mcov = np.cov(u)
        #print mcov.shape
        p,l,pt = NLA.svd(mcov)
        lsi = SLA.sqrtm(SLA.inv(np.diag(l))) 
        print lsi.shape
        print p.T.shape
        print mat.shape

        snu =  np.dot(np.dot(lsi, p.T), mat.T)
        return snu
    """
    #正規化
    def stdn(self, U):
        mat = np.matrix(U).T
        mat = mat - mat.mean(axis=1)
        mcov = np.cov(mat)
        p,l,pt = SLA.svd(mcov)
        lsi = SLA.sqrtm(SLA.inv(np.diag(l))) 
        H = np.dot(lsi, p.T)
        snU = np.dot(H, mat)

        if self.out == 0:
            print "u:",np.matrix(U).T
            print "mat:",mat
            print "mcov:",mcov
            print "p:",p
            print "l:",l
            print "pt"
            print "lsi:",lsi
            print np.dot(lsi, p.T)
        #print "cov:"
        #print np.cov(snU)

        return snU


    def cca(self, X, Y):
        '''
        正準相関分析
        http://en.wikipedia.org/wiki/Canonical_correlation
        '''    
        X = np.array(X)
        Y = np.array(Y)
        n, p = X.shape
        n, q = Y.shape
        
        # zero mean
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        
        # covariances
        S = np.cov(X.T, Y.T, bias=1)
        
        # S = np.corrcoef(X.T, Y.T)
        SXX = S[:p,:p]
        SYY = S[p:,p:]
        SXY = S[:p,p:]
        SYX = S[p:,:p]
        
        # 
        sqx = SLA.sqrtm(SLA.inv(SXX)) # SXX^(-1/2)
        sqy = SLA.sqrtm(SLA.inv(SYY)) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        A, s, Bh = SLA.svd(M, full_matrices=False)
        B = Bh.T      
        #print np.dot(np.dot(A[:,0].T,SXX),A[:,0])
        return s, A, B

    def cca2(self, X, Y):
        #係数が1以下にならない...??
        #row:window size, col:data dimen
        X = np.array(X)
        Y = np.array(Y)
        n, p = X.shape
        n, q = Y.shape
        #print X.shape
        # std
        X = (X - X.mean(axis=0))/X.std(axis=0)
        Y = (Y - Y.mean(axis=0))/X.std(axis=0)

        R = np.corrcoef(X.T,Y.T)

        R11 = R[:p,:p]
        R12 = R[:p,p:]
        R22 = R[p:,:p]
        #print R.shape
        #print R11.shape
        # print R12.shape
        #print R22.shape

        A = np.dot(np.dot(np.dot(NLA.inv(R11),R12),NLA.inv(R22)),R12.T)
        lambs, wx = NLA.eig(A)
        #順番の入れ替え
        #print "bf lambs:",lambs
        #print "wx:",wx
        idx = lambs.argsort()[::-1]
        lambs = np.sqrt(lambs[idx])
        wx = wx[:,idx]
        wy = np.dot(np.dot(NLA.inv(R22), R12.T), NLA.inv(np.diag(lambs)))

        return lambs, wx, wy

    def ccaR(self, U1, U2):

        r = pyper.R(use_pandas='True')

        for i, u in enumerate(U1):
            out = "x"+str(i)+"<-c(" 
            for j, el in enumerate(u):
                if j != len(u)-1:
                    out += str(el)+", "
                else:
                    out += str(el)+")"
            r(out)
                    
        for i, u in enumerate(U2):
            out = "y"+str(i)+"<-c(" 
            for j, el in enumerate(u):
                if j != len(u)-1:
                    out += str(el)+", "
                else:
                    out += str(el)+")"
            r(out)

        out = "f<-rbind("            
        for i in range(len(U1)):
            if i != len(U1)-1:
                out += "x"+str(i)+","
            else:
                out += "x"+str(i)+")"
        r(out)

        out = "g<-rbind("            
        for i in range(len(U2)):
            if i != len(U2)-1:
                out += "y"+str(i)+","
            else:
                out += "y"+str(i)+")"
        r(out)

        r("can <- cancor(f,g)")
        r("lambs <- can$cor")
        r("Wx <- can$xcoef")
        r("Wy <- can$ycoef")

        lambs = r.lambs
        Wx = r.Wx
        Wy = r.Wy

        return lambs, Wx, Wy

def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    graph = GRAPH()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
