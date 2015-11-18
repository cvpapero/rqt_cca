#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
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
        self.jsonInput()

        super(CCA, self).__init__()
        self.eigArray = []
        self.eigValArray = []
        self.initUI()
        rospy.init_node('ros_cca', anonymous=True)
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

        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('30')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        boxExecCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.doExec)
        boxExecCtrl.addWidget(btnExec)
        

        self.AFrameBox = QtGui.QLineEdit()
        self.AFrameBox.setText('0')
        self.AFrameBox.setAlignment(QtCore.Qt.AlignRight)
        self.AFrameBox.setFixedWidth(100)
        form.addRow('A User Frame', self.AFrameBox)

        self.BFrameBox = QtGui.QLineEdit()
        self.BFrameBox.setText('0')
        self.BFrameBox.setAlignment(QtCore.Qt.AlignRight)
        self.BFrameBox.setFixedWidth(100)
        form.addRow('B User Frame', self.BFrameBox)

        self.eigOrder = QtGui.QComboBox(self)
        self.eigOrder.addItems(self.eigArray)
        self.eigOrder.setFixedWidth(150)
        boxEigOrder = QtGui.QHBoxLayout()
        boxEigOrder.addWidget(self.eigOrder)
        form.addRow('eigen order', boxEigOrder)

        boxPubCtrl = QtGui.QHBoxLayout()
        btnPub = QtGui.QPushButton('pub')
        btnPub.clicked.connect(self.doPub)
        boxPubCtrl.addWidget(btnPub)

        grid.addLayout(form,1,0)
        grid.addLayout(boxExecCtrl,2,0)
        grid.addLayout(boxPubCtrl,2,1)

        self.setLayout(grid)
        self.resize(400,100)
        
        self.setWindowTitle("frame select window")
        self.show()

    def jsonInput(self):
        f = open('test1014.json', 'r');
        jsonData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()

        self.data = {}
        self.pdata = {}
        self.jdatas = []
        self.time = []
        self.usrSize = len(jsonData)

        u = 0;

        for user in jsonData:
            datas = []
            tsize = len(user["datas"])
            psize = len(user["datas"][0]["jdata"])            

            pobj = {}
            for t in range(tsize):
                #angle
                datas.append(user["datas"][t]["data"])
                #points
                plist = []
                for p in range(psize):
                    pl = []
                    pl.append(user["datas"][t]["jdata"][p][0])
                    pl.append(user["datas"][t]["jdata"][p][1])
                    pl.append(user["datas"][t]["jdata"][p][2])
                    plist.append(pl)
                pobj[t] = plist
            self.jdatas.append(datas)
            self.pdata[u] = pobj
            u+=1
        
        for itime in jsonData[0]["datas"]:
            self.time.append(itime["time"])

        self.dataSize = tsize

        #joint index
        f = open('joint_index.json', 'r');
        jsonIndexData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()
        self.idata = []
        for index in jsonIndexData:
            ilist = []
            for i in index:
                ilist.append(i)
            self.idata.append(ilist)

    def doExec(self):
        self.winSize = int(self.winSizeBox.text())
        self.maxRange = self.dataSize - self.winSize

        self.dataAlig()
        self.canoniCorr()
        self.bartlettTest()
        self.updateEigOrder()

    def updateEigOrder(self):
        self.eigOrder.clear()
        self.eigOrder.addItems(self.eigArray)
        self.eigOrder.setFixedWidth(150)
        #boxEigOrder = QtGui.QHBoxLayout()
        #boxEigOrder.addWidget(self.eigOrder)

    #データの整頓(winsize分だけ並べる)
    def dataAlig(self):

        self.X = []
        self.Y = []

        self.range = self.dataSize-self.winSize+1

        for i in range(self.range):
            data = []
            for w in range(self.winSize):
                data.extend(self.jdatas[0][w+i])
            self.X.append(data)
       
        for i in range(self.range):
            data = []
            for w in range(self.winSize):
                data.extend(self.jdatas[1][w+i])
            self.Y.append(data)
       
    #正準相関
    def canoniCorr(self):
        tX = np.matrix(self.X).T
        tY = np.matrix(self.Y).T
        self.n, self.p = tX.shape
        self.n, self.q = tY.shape

        sX = tX - tX.mean(axis=0) 
        sY = tY - tY.mean(axis=0)

        #print "sX:"
        #print sX

        S = np.cov(sX.T, sY.T, bias = 1)

        #print "cov:"
        #print S

        #p,q = S.shape
        #p = p/2
        SXX = S[:self.p,:self.p]
        SYY = S[self.p:,self.p:]
        SXY = S[:self.p,self.p:]
        SYX = S[self.p:,:self.p]

        sqx = LA.sqrtm(LA.inv(SXX)) # SXX^(-1/2)
        sqy = LA.sqrtm(LA.inv(SYY)) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        self.A, self.s, Bh = LA.svd(M, full_matrices=False)
        self.B = Bh.T
 
    def bartlettTest(self):
        M = -(self.n-1/2*(self.p+self.q+3))
        #print "M:"+str(M)
        for i in range(len(self.s)):
            #有意水準を求める
            alf = 0.01
            sig = sp.special.chdtri((self.p-i)*(self.q-i), alf)
            print
            test = 1
            for j in range(len(self.s)-i):
                test = test*(1-self.s[len(self.s)-j-1])
            chi = M*math.log(test)

            if chi > sig:
                print  "test["+str(i)+"]:"+str(chi) +" > sig("+str(alf)+"):"+str(sig)
                ru = np.fabs(self.A[:,i:i+1])
                rv = np.fabs(self.B[:,i:i+1])
                arg_u = np.argmax(ru)
                arg_v = np.argmax(rv)
                self.eigArray.append(str(i))
                val = [arg_u, arg_v]
                self.eigValArray.append(val)
                print "ru-max arg:"+str(np.argmax(ru))
                print "rv-max arg:"+str(np.argmax(rv))
                #print "---"
            else:
                break



    def doPub(self):
        print "pub!"

        idx = int(self.eigOrder.currentText())
        print idx

        self.winSize = int(self.winSizeBox.text())
        
        #Ast = float(self.AFrameBox.text())
        #Bst = float(self.BFrameBox.text())
        Ast = self.eigValArray[idx][0]
        Bst = self.eigValArray[idx][1]
        print Ast
        print Bst
        self.pubViz(Ast, Bst)

        print "end"

    def pubViz(self, ast, bst):

        rate = rospy.Rate(10)

        for i in range(self.winSize):

            msgs = MarkerArray()
            
            #use1について
            msg = Marker()
            #いろんなプロパティ
            msg.header.frame_id = 'camera_link'
            msg.header.stamp = rospy.Time.now()
            msg.ns = 'j1'
            msg.action = 0
            msg.id = 1
            msg.type = 8
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.color = self.carray[2]
            #ジョイントポイントを入れる処理
            for j1 in range(len(self.pdata[0][ast+i])):
                point = Point()
                point.x = self.pdata[0][ast+i][j1][0]
                point.y = self.pdata[0][ast+i][j1][1]
                point.z = self.pdata[0][ast+i][j1][2]
                msg.points.append(point) 
            msg.pose.orientation.w = 1.0
            msgs.markers.append(msg)    
            
            msg = Marker()
            msg.header.frame_id = 'camera_link'
            msg.header.stamp = rospy.Time.now()
            msg.ns = 'j2'
            msg.action = 0
            msg.id = 2
            msg.type = 8
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.color = self.carray[1]

            for j2 in range(len(self.pdata[0][bst+i])):
                point = Point()
                point.x = self.pdata[1][bst+i][j2][0]
                point.y = self.pdata[1][bst+i][j2][1]
                point.z = self.pdata[1][bst+i][j2][2]
                msg.points.append(point) 
            msg.pose.orientation.w = 1.0

            msgs.markers.append(msg)

            self.mpub.publish(msgs)
            rate.sleep()

def main():
    app = QtGui.QApplication(sys.argv)
    cca = CCA()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
