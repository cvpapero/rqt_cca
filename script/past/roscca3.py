#!/usr/bin/python
# -*- coding: utf-8 -*-

#一定フレーム内同士で計算する
#それ以外では計算しない(たぶん、関連が無いから)
#バートレット検定で第何位までの成分を使うか決める

#import sys
#import math
#import json
#import numpy

import sys
import os.path
import math
import json
import numpy as np
from numpy import linalg as NLA

#import pyper

import scipy as sp
from scipy import linalg as SLA
from scipy import stats as ST

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

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

        #ファイル入力
        #self.jsonInput()

        #ROSのパブリッシャなどの初期化
        rospy.init_node('roscca', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.ppub = rospy.Publisher('joint_diff', PointStamped, queue_size=10)


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
        self.winSizeBox.setText('5')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        #frame size
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('7')
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
        #filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', os.path.expanduser('~') + '/Desktop')
        filename = self.txtSepFile.text()

        #f = open('test1014.json', 'r')
        f = open(filename, 'r')
        jsonData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()

        self.DATAS = []

        for user in jsonData:
            #angle
            data = []
            self.datasSize = len(user["datas"])

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

        self.jsonInput()

        self.winSize = int(self.winSizeBox.text())

        self.frmSize = int(self.frmSizeBox.text())

        self.canoniExec1()
        #self.time_setting()
        self.updateTable()
        print "end"

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
                        
                        tmp_rho = self.canoniCorr(USER1, USER2)
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
                    tmp_rho = self.canoniCorr(USER1, USER2)
                    self.ccaMat[f+t1][f+self.dataRange-1] = float(tmp_rho) 
                for t2 in range(self.dataRange):
                    USER1=[]
                    USER2=[]
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][f+self.dataRange-1+w])
                        USER2.append(self.DATAS[1][f+t2+w])
                    tmp_rho = self.canoniCorr(USER1, USER2)
                    self.ccaMat[f+self.dataRange-1][f+t2] = float(tmp_rho)

            #print "---"
            #print "user1 t:"+str(time1)+", user2 t:"+str(time2)+", delay(t1-t2):"+str(time1-time2)+", rho:"+str(float(rho))

    #正規化
    #もし逆行列がとれない場合はどうするか?
    def backSortM(self, X):
        X = np.matrix(X)
        rows, cols = X.shape
        #s = np.matrix()
        s = X[:,cols-1:cols]
        #s = [[0 for i in range(rows)]]

        for i in range(cols-1):
            s = np.c_[s, X[:,cols-i-2:cols-i-1]]

        return s

    def backSort(self, X):
        cols = len(X)
        #s = np.matrix()
        s = []
        #s = [[0 for i in range(rows)]]

        for i in range(cols):
            s.append(X[cols-i-1])

        return s

    def stdNorm(self, U1, U2):
        print "U1"
        print U1
        print "U2"
        print U2

        mat1 = np.matrix(U1).T
        print mat1
        print mat1.mean(axis=1)
        mat1 = mat1 - mat1.mean(axis=1)
        print mat1
        mat1cov = np.cov(mat1)
        print mat1cov
        p1,l1,p1t = NLA.svd(mat1cov)
        print p1
        print l1
        print p1t
        l1sq = SLA.sqrtm(SLA.inv(np.diag(l1))) 
        snU1 =  np.dot(np.dot(l1sq, p1.T), mat1)

        mat2 = np.matrix(U2).T
        mat2 = mat2 - mat2.mean(axis=1)
        mat2cov = np.cov(mat2)
        p2,l2,p2t = NLA.svd(mat2cov)
        l2sq = SLA.sqrtm(SLA.inv(np.diag(l2))) 
        snU2 =  np.dot(np.dot(l2sq, p2.T), mat2)

        print "cov:"
        print np.cov(snU1)
        print np.cov(snU2)

        return snU1, snU2

    #正準相関
    def canoniCorr(self, U1, U2):
        #print "data U1 U2"
        #print U1
        #print U2
        U1, U2 = self.stdNorm(U1, U2)

        #sX = np.matrix(U1).T
        #sY = np.matrix(U2).T

        data = np.r_[U1, U2]
        #data.extend(U1)
        #data.extend(U2)

        #print "data"
        #print data
        
        p,n = U1.shape

        #print "r:"+str(r)+", c:"+str(c)

        #sX = tX - tX.mean(axis=0)
        #sY = tY - tY.mean(axis=0)

        S = np.cov(data)

        #べつにこのやり方は間違ってない
        #S2 = np.cov(sX, sY)

        Sxx = S[:p,:p]
        Sxy = S[:p,p:]
        Syy = S[p:,p:]
        
        #print "S-S2"
        #print S-S2
        #print "S2"
        #print S2

        #print "Sxx"
        #print Sxx
        slambs = NLA.eigvalsh(Sxx)
        #lambs, vecs = SLA.eig(Sxx)
        #print "sxx lam"
        #print slambs
        #Lxx = SLA.cholesky(Sxx, lower=True)
        #Lxx = NLA.cholesky(Sxx)
        #A =  np.dot(np.dot(np.dot(np.dot(NLA.inv(Lxx),Sxy),NLA.inv(Syy)),Sxy.T),NLA.inv(Lxx.T))
        #print A
        #固有値問題
        lambs, vecs = SLA.eigh(np.dot(Sxy, Sxy.T))

        #print "vecs:"
        #print vecs
        print "lambs:"
        print lambs
        
        """
        for (i, l) in enumerate(lambs):
            if l.imag == 0 and l.real < 1 and 0 < l.real:
                print "lambs["+str(i)+"]:"+str(l)                
        """

        Wx = np.dot(NLA.inv(Lxx.T),vecs)

        #lst = [[1, 2, 3] for i in range(3)]
        #print(lst)
        # => [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

        #L = [[1/np.sqrt(lambs[i].real) for i in range(len(lambs))] for j in range(len(lambs))]
        #L = [1/lambs for i in range(len(lambs))]
        #L = np.matrix(L).T
        #print "L"
        #print L
        Wy = (np.dot(np.dot(NLA.inv(Syy),Sxy.T),Wx))/np.sqrt(lambs[0].real)
        #Wy = np.dot(L, np.dot(np.dot(NLA.inv(Syy),Sxy.T),Wx))
        #print "Wx"
        #print Wx
        rho = 0
        srho = 0
        vr, vc = vecs.shape
        
        #print "vecs:"
        #print vecs
        #print "lambs:"
        #print lambs
        #print "sxx"
        #print sxx
        #Wx=vecs[:,vc-1:vc]
        #print "WtSxxWt"
        #print np.dot(np.dot(Wx.T,sxx),Wx)

        #d = self.bartlettTest(vr, vc, lambs)
        srho = 0
        
        d = 1
        
        """
        for i in range(d):
            vec1 = Wx[:,vc-i-1:vc-i]
            lamb = np.sqrt(lambs[vc-i-1])
            vec2 = Wy[:,i:i+1]

            #v1sxyv2 = np.dot(np.dot(vec1.T,sxy),vec2)
            v1sxxv1 = np.dot(np.dot(vec1.T,Sxx),vec1)
            v2syyv2 = np.dot(np.dot(vec2.T,Syy),vec2)
            #print "v1sxxv1:"+str(v1sxxv1)
            #print "v2syyv2:"+str(v2syyv2)
            rho = lambs#np.sqrt(lambs[i].real)#v1sxyv2 / np.sqrt(np.dot(v1sxxv1,v2syyv2))
            srho = srho + rho 
           # print float(rho)
        """
        return np.sqrt(lambs[vc-1])
        #return float(srho)/d



    def bartlettTest(self, rows, cols, lambs):
        
        lSize = len(lambs)
        M = rows-1/2*(cols+cols+3)
        #print lambs
        for i in range(lSize):
            #有意水準を求める
            alf = 0.05
            sig = sp.special.chdtri((cols-i+1)*(cols-i+1), alf)
            w = 1
            for j in range(i, lSize):
                w = w*(1-lambs[lSize-j-1].real)
                print "j:"+str(j)+", lam:"+str(lambs[lSize-j-1])+", w:"+str(w)
            #print "w:"+str(w)
            
            bart = M*math.log(w)
            #print  "bart["+str(i)+"]:"+str(bart) +" > sig("+str(alf)+"):"+str(sig)
        
            if bart > sig:
                print  "bart["+str(i)+"]:"+str(bart) +" > sig("+str(alf)+"):"+str(sig)
                #ru = np.fabs(self.A[:,i:i+1])
                #rv = np.fabs(self.B[:,i:i+1])
                #ru = self.A[:,i:i+1]
                #rv = self.B[:,i:i+1]
                #print "ru-max val:"+str(np.max(ru))+", arg:"+str(np.argmax(ru))
                #print "rv-max val:"+str(np.max(rv))+", arg:"+str(np.argmax(rv))
                #print "ru-max arg:"+str(np.argmax(ru))
                #print "rv-max arg:"+str(np.argmax(rv))
                #print "---"
            else:
                break
        print "i:"+str(i)
        return 1

def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
