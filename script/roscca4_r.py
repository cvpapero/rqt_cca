#!/usr/bin/python
# -*- coding: utf-8 -*-

#一定フレーム内同士で計算する
#それ以外では計算しない(たぶん、関連が無いから)
#バートレット検定で第何位までの成分を使うか決める(未)
#ave=0, Sxx=Iを実装(2015.12.11)

#写像して(U1*Wx,U2*W2)相関をとったけど、やはり1になる。検算としては間違ってないかと思われる
#では、動いていない関節が強調されているのでは無いか？調べてみる(2015.12.14)

#Rの資産を使う
#ベクトルの足し算
#固有値固有ベクトルの大きい順に並べ替え(2015.12.15)

import sys
import os.path
import math
import json

import numpy as np
from numpy import linalg as NLA
import scipy as sp
from scipy import linalg as SLA

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

from pylab import *

import pyper
import pandas as pd

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
        self.frmSizeBox.setText('50')
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


        #アイテムがクリックされたらパブリッシュする
        #self.table.itemClicked.connect(self.doViz)
        #self.table.setItem(0, 0, QtGui.QTableWidgetItem(1))

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

    def doExec(self):

        print "exec!"

        #ファイル入力
        self.jsonInput()

        self.winSize = int(self.winSizeBox.text())

        self.frmSize = int(self.frmSizeBox.text())

        self.canoniExec1()
        #self.time_setting()
        self.updateTable()
        self.updateColorTable()
        print "end"

    def updateColorTable(self):
        x = arange(self.dataDimen+1)
        y = arange(self.dataDimen+1)

        X, Y = meshgrid(x, y)
        subplot(1,2,1)
        pcolor(X, Y, self.mWxList)
        colorbar()
        title("user 1")

        subplot(1,2,2)
        pcolor(X, Y, self.mWyList)
        colorbar()
        title("user 2")
        tight_layout()

        show()   

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
        
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()



    def canoniExec1(self):

        self.dataMaxRange = self.datasSize - self.winSize + 1
        self.ccaMat = [[0 for i in range(self.dataMaxRange)] for j in range(self.dataMaxRange)]

        self.frameRange = self.datasSize - self.frmSize + 1
        self.dataRange = self.frmSize - self.winSize + 1
        
        print "datasSize:"+str(self.datasSize)
        print "dataMaxRange:"+str(self.dataMaxRange)
        print "frameRange:"+str(self.frameRange)
        print "dataRange:"+str(self.dataRange)

        #最大ベクトルのカウント
        self.mWxList = np.zeros([self.dataDimen, self.dataDimen])
        self.mWyList = np.zeros([self.dataDimen, self.dataDimen])

        self.output = 1

        for f in range(self.frameRange):
            #print "f:"+str(f)+"---"
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
                        
                        tmp_rho = self.canoniCorrR(USER1, USER2)
                        self.ccaMat[t1+f][t2+f] = float(tmp_rho)
                        #print "tmp_rho"+str(tmp_rho)
                        
                        if math.fabs(tmp_rho) > math.fabs(rho):
                            rho = tmp_rho                
                            time1 = t1
                            time2 = t2
            else:
                for t1 in range(self.dataRange - 1):
                    USER1=[]
                    USER2=[]
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][f+t1+w])
                        USER2.append(self.DATAS[1][f+self.dataRange-1+w])
                    tmp_rho = self.canoniCorrR(USER1, USER2)
                    self.ccaMat[f+t1][f+self.dataRange-1] = float(tmp_rho) 
                for t2 in range(self.dataRange):
                    USER1=[]
                    USER2=[]
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][f+self.dataRange-1+w])
                        USER2.append(self.DATAS[1][f+t2+w])
                    tmp_rho = self.canoniCorrR(USER1, USER2)
                    self.ccaMat[f+self.dataRange-1][f+t2] = float(tmp_rho)


        #print "mwx:"
        #print self.mWxList
        #print "mwy:"
        #print self.mWyList



        #print "user1 t:"+str(time1)+", user2 t:"+str(time2)+", delay(t1-t2):"+str(time1-time2)+", rho:"+str(float(rho))

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
        self.mWxList += np.dot(np.fabs(Wx),np.diag(lambs))
        self.mWyList += np.dot(np.fabs(Wy),np.diag(lambs))

        """
        print "wwx:"
        print wwx
        print "wwy:"
        print wwy
        """

        if self.output == 1:
            f = open('outcca','w')
            #string = "U1:\n"+str(U1)+"\n U2:\n"+str(U2)+"\n nU1:\n"+str(nU1)+"\n nU2:\n"+str(nU2)
            #string2 = "lambs:\n"+str(lambs)
            #print "Wx:"
            #print Wx
            #print "Wy:"
            #print Wy
            for i, u in enumerate(U1):
                out = "x"+str(i)+"<-c(" 
                for j, el in enumerate(u):
                    if j != len(u)-1:
                        out += str(el)+", "
                    else:
                        out += str(el)+")"
                f.write(out+"\n")

            for i, u in enumerate(U2):
                out = "y"+str(i)+"<-c(" 
                for j, el in enumerate(u):
                    if j != len(u)-1:
                        out += str(el)+", "
                    else:
                        out += str(el)+")"
                f.write(out+"\n")
            """
            f.write("U2:"+str(U2)+"\n")
            f.write("nU1:"+str(nU1)+"\n")
            f.write("nU2:"+str(nU2)+"\n")
            f.write("lambs:"+str(lambs))
            f.write("Wx:"+str(Wx))
            f.write("Wy:"+str(Wy))
            """
            f.close()
            print "lambs:"
            print lambs
            print "Wx:"
            print Wx
            print "Wy:"
            print Wy
            self.output = 0

        """
        #d = self.bartlettTest(p, q, lambs)

        #fU1 = np.dot(U1.T, Wx[:,0:1]).T
        #fU2 = np.dot(U2.T, Wy[:,0:1]).T
        fU1 = np.dot(U1.T, Wx).T
        fU2 = np.dot(U2.T, Wy).T
        print "fU1:"
        print fU1
        print "fU2:"
        print fU2
        data_t = np.r_[fU1, fU2]
        print "cov:"
        print np.cov(data_t)
        """
        return lambs[0]

    def canoniCorrR(self, U1, U2):

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
        lambs = pd.Series(r.get("lambs"))
        #Wx = pd.Series(r.get("Wx"))
        #Wy = pd.Series(r.get("Wy"))

        print "lambs:"
        print lambs


        #self.mWxList += np.dot(np.fabs(Wx),np.diag(lambs))
        #self.mWyList += np.dot(np.fabs(Wy),np.diag(lambs))

        return lambs[0]

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