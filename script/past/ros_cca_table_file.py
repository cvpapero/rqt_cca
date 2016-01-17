#!/usr/bin/python
# -*- coding: utf-8 -*-

#二つの異なるデータを読み込む

#import sys
#import math
#import json
#import numpy

import sys
import os.path
import math
import json
import numpy as np
import scipy as sp
from scipy import linalg as LA
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
        rospy.init_node('ros_cca_table', anonymous=True)
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
        """
        #ファイル入力ボックス
        self.txtSepFile = []
        self.txtSepFile.append(QtGui.QLineEdit())
        btnSepFile1 = QtGui.QPushButton('...')
        btnSepFile1.setMaximumWidth(40)
        self.fileIndex = 0
        btnSepFile1.clicked.connect(self.chooseDbFile)
        boxSepFile1 = QtGui.QHBoxLayout()
        boxSepFile1.addWidget(self.txtSepFile[0])
        boxSepFile1.addWidget(btnSepFile1)
        form.addRow('input file 1', boxSepFile1)
      
        #ファイル入力ボックス2
        self.txtSepFile.append(QtGui.QLineEdit())
        btnSepFile2 = QtGui.QPushButton('...')
        btnSepFile2.setMaximumWidth(40)
        self.fileIndex = 1
        btnSepFile2.clicked.connect(self.chooseDbFile)
        boxSepFile2 = QtGui.QHBoxLayout()
        boxSepFile2.addWidget(self.txtSepFile[1])
        boxSepFile2.addWidget(btnSepFile2)
        form.addRow('input file 2', boxSepFile2)  
        """
        #window size
        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('40')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

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

    """
    def chooseDbFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile[self.fileIndex].setText(f)
                return
        return self.txtSepFile[self.fileIndex].setText('')
    """

    def jsonInput(self):
        """
        self.DATAS = []
        dSize = []
        for i in range(len(self.txtSepFile)): 
            filename = self.txtSepFile[i].text()

            #f = open('test1014.json', 'r')
            f = open(filename, 'r')
            jsonData = json.load(f)
            #print json.dumps(jsonData, sort_keys = True, indent = 4)
            f.close()

            for user in jsonData:
                #angle
                data = []
                
                dSize[i] = len(user["datas"])

                for j in range(self.datasSize):
                    data.append(user["datas"][j]["data"])
                    
                self.DATAS.append(data)

            self.datasSize = dSize[0] if dSize[0] < dSize[1] else dSize[1]
        """
        
        filenames = []
        filenames.append("1test124.json")
        filenames.append("2test124.json")
        
        jsonData = []
        dSize = []

        for i in range(len(filenames)): 
            filename = filenames[i] #self.txtSepFile[i].text()
            f = open(filename, 'r')
            jsonData.append(json.load(f))
            dSize.append(len(jsonData[i][0]["datas"]))
            #print "jsondata[i]"+str(jsonData[i][0]["datas"])


        self.datasSize = dSize[0] if dSize[0] < dSize[1] else dSize[1]

        print "datasSize:"+str(self.datasSize)

        self.DATAS = []

        for i in range(len(filenames)):
            data = [] 
            for j in range(self.datasSize):
                data.append(jsonData[i][0]["datas"][j]["data"])
            self.DATAS.append(data)

        #データをdetasSizeでカットする
        #print "DATAS[0]:"
        #print self.DATAS[0]
        #print "DATAS[1]:"
        #print self.DATAS[1]
        


    def doExec(self):

        print "exec!"

        self.jsonInput()

        self.winSize = int(self.winSizeBox.text())

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
                    
                c = 255
                if self.ccaMat[i][j] > self.threshold:
                    c = self.ccaMat[i][j] * 255

                self.table.setItem(i, j, QtGui.QTableWidgetItem())
                self.table.item(i, j).setBackground(QtGui.QColor(c,c,c))
                self.table.item(i, j).setToolTip(str(self.ccaMat[i][j]))
        
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()

    def canoniExec1(self):

        self.dataRange = self.datasSize - self.winSize + 1

        self.ccaMat = [[0 for i in range(self.dataRange)] for j in range(self.dataRange)]

        for t1 in range(self.dataRange):
            rho = 0
            time1 = 0
            time2 = 0
            for t2 in range(self.dataRange):
                USER1 = []
                USER2 = []
                for w in range(self.winSize):
                    USER1.append(self.DATAS[0][t1+w])
                    USER2.append(self.DATAS[1][t2+w])
                    
                tmp_rho = self.canoniCorr(USER1, USER2)
                self.ccaMat[t1][t2] = float(tmp_rho)
                #print "tmp_rho"+str(tmp_rho)
                
                if math.fabs(tmp_rho) > math.fabs(rho):
                    rho = tmp_rho                
                    time1 = t1
                    time2 = t2
                
            print "---"
            print "user1 t:"+str(time1)+", user2 t:"+str(time2)+", delay(t1-t2):"+str(time1-time2)+", rho:"+str(float(rho))


    #正準相関
    def canoniCorr(self, U1, U2):
        #配列から行列へ(あらかじめ転置しておく)
        tX = np.matrix(U1).T
        tY = np.matrix(U2).T

        self.p, self.n = tX.shape
        self.q, self.n = tY.shape

        #正規化
        sX = sp.stats.zscore(tX, axis=1)
        sY = sp.stats.zscore(tY, axis=1)

        #共分散行列
        cov = np.cov(sX, sY)
        sxx = cov[:self.p,:self.p]
        sxy = cov[:self.p,self.p:]
        syy = cov[self.p:,self.p:]
        a = np.dot(sxy, sxy.T)

        #固有値問題
        #eighは実対称行列のみ, 固有値・ベクトルは昇順(小→大)
        lambs, vecs = LA.eigh(a)

        #最大の固有値・ベクトルをとる
        vr, vc = vecs.shape
        vec1 = vecs[:,vc-1:vc]
        lamb = np.sqrt(lambs[vc-1])
        vec2 = np.dot(sxy.T, vec1)/lamb
        
        v1sxyv2 = np.dot(np.dot(vec1.T,sxy),vec2)
        v1sxxv1 = np.dot(np.dot(vec1.T,sxx),vec1)
        v2syyv2 = np.dot(np.dot(vec2.T,syy),vec2)

        rho = v1sxyv2 / np.sqrt(np.dot(v1sxxv1,v2syyv2))
        return rho

    """
    def time_setting(self):
        count = 0
        self.timedata = []
        for dt in range(-self.maxRange, self.maxRange+1):
            if dt > 0:
                self.timedata.append(self.time[abs(dt)]-self.time[0])
            if dt <= 0:
                self.timedata.append(self.time[0]-self.time[abs(dt)])
                    
    """


def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
