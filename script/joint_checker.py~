#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import math
import json
import numpy

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA

class Correlation(QtGui.QWidget):

    def __init__(self):
        self.jsonInput()

        super(Correlation, self).__init__()
        self.initUI()
        rospy.init_node('correlation', anonymous=True)
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
        '''
        self.txtSepFile = gui.QLineEdit()
        btnSepFile = QtGui.QPushButton('...')
        btnSepFile.setMaximumWidth(40)
        btnSepFile.clicked.connect(self.chooseDbFile)
        boxSepFile = gui.QHBoxLayout()
        boxSepFile.addWidget(self.txtSepFile)
        boxSepFile.addWidget(btnSepFile)
        form.addRow('input file', boxSepFile)
        '''
        self.ThesholdBox = QtGui.QLineEdit()
        self.ThesholdBox.setText('0.0')
        self.ThesholdBox.setAlignment(QtCore.Qt.AlignRight)
        self.ThesholdBox.setFixedWidth(100)
        form.addRow('corr theshold', self.ThesholdBox)

    
        self.VarMinBox = QtGui.QLineEdit()
        self.VarMinBox.setText('0.01')
        self.VarMinBox.setAlignment(QtCore.Qt.AlignRight)
        self.VarMinBox.setFixedWidth(100)
        form.addRow('deg threshold', self.VarMinBox)
        

        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('40')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        self.cmbSrcJoints = QtGui.QComboBox(self)
        self.cmbSrcJoints.addItems(self.jointnames)
        self.cmbSrcJoints.setFixedWidth(150)
        boxSrcJoints = QtGui.QHBoxLayout()
        boxSrcJoints.addWidget(self.cmbSrcJoints)
        form.addRow('user_1 joint', boxSrcJoints)

        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.doExec)
        boxCtrl.addWidget(btnExec)


        self.table = QtGui.QTableWidget(self)
        self.table.setColumnCount(self.jointSize)
        self.table.setHorizontalHeaderLabels("use_2 joint") 
        for i in range(self.jointSize):
            jItem = QtGui.QTableWidgetItem(str(i))
            self.table.setHorizontalHeaderItem(i, jItem)

        font = QtGui.QFont()
        font.setFamily(u"DejaVu Sans")
        font.setPointSize(8)



        self.table.horizontalHeader().setFont(font)
        self.table.verticalHeader().setFont(font)
        self.table.resizeColumnsToContents()
        self.table.itemClicked.connect(self.doViz)
        #self.table.setItem(0, 0, QtGui.QTableWidgetItem(1))


        boxTable = QtGui.QHBoxLayout()
        boxTable.addWidget(self.table)
 
        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)
        grid.addLayout(boxTable,3,0)

        self.setLayout(grid)
        self.resize(400,100)

        self.setWindowTitle("joint select window")
        self.show()

    def jsonInput(self):
        f = open('test1014.json', 'r');
        jsonData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()

        self.data = {}
        self.pdata = {}
        self.jointnames = []
        self.time = []
        self.usrSize = len(jsonData)

        u = 0;
        for user in jsonData:
            #angle
            jsize = len(user["datas"][0]["data"])            
            tsize = len(user["datas"])
            jobj = {}
            for j in range(0, jsize):
                jlist = []
                for t in range(0, tsize):
                    jlist.append(user["datas"][t]["data"][j])
                jobj[j] = jlist
                if u == 0:
                    self.jointnames.append(str(j))
            self.data[u]=jobj
            #position
            psize = len(user["datas"][0]["jdata"])            
            tsize = len(user["datas"])
            pobj = {}
            for p in range(0, psize):
                plist = []
                for t in range(0, tsize):
                    pl = []
                    #print str(p)+","+str(t)
                    #print user["datas"][t]["jdata"][p]
                    pl.append(user["datas"][t]["jdata"][p][0])
                    pl.append(user["datas"][t]["jdata"][p][1])
                    pl.append(user["datas"][t]["jdata"][p][2])
                    plist.append(pl)
                pobj[p] = plist
            self.pdata[u] = pobj
            u+=1

        for itime in jsonData[0]["datas"]:
            #print itime["time"]
            self.time.append(itime["time"])


        self.jointSize = len(self.data[0])
        self.dataSize = len(self.data[0][0])

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
         
        if len(jsonData) == 1:
            self.data[1] = self.data[0]
            self.pdata[1] = self.pdata[0]
        #print self.idata

    def doExec(self):
        print "exec! joint_"+self.cmbSrcJoints.currentText()
        j_idx_u1 = int(self.cmbSrcJoints.currentText())
        self.winSize = int(self.winSizeBox.text())
        self.maxRange = self.dataSize - self.winSize
        self.threshold = float(self.ThesholdBox.text())
        self.varMin = float(self.VarMinBox.text())*(math.pi/180)
        self.process(j_idx_u1)
        self.time_setting()
        self.updateTable()
        print "end"

    def doViz(self, cItem):
        idx_u0 = int(self.cmbSrcJoints.currentText())
        j0idx = self.idata[idx_u0]        
        i00 = j0idx[0]
        i01 = j0idx[1]
        i02 = j0idx[2]

        idx_u1 = cItem.column()
        j1idx = self.idata[idx_u1]
        i10 = j1idx[0]
        i11 = j1idx[1]
        i12 = j1idx[2]

        usr = 0
        delay = cItem.row() - self.maxRange
        rate = rospy.Rate(10)
        for didx in range(0, self.winSize):
            dt1 = self.dataSize - self.winSize
            dt2 = dt1 - abs(delay)
            p0 = []
            p1 = []
            p2 = []
            pl = []
            angle = []
            if delay >= 0:
            
                p0.append(self.pdata[0][i00][dt2+didx])
                p0.append(self.pdata[0][i01][dt2+didx])
                p0.append(self.pdata[0][i02][dt2+didx])
                p1.append(self.pdata[1][i10][dt1+didx])
                p1.append(self.pdata[1][i11][dt1+didx])
                p1.append(self.pdata[1][i12][dt1+didx])
                p2.append(self.pointStore(self.pdata[0], dt2+didx))
                p2.append(self.pointStore(self.pdata[1], dt1+didx))
                angle.append(self.data[0][idx_u0][dt2+didx])
                angle.append(self.data[1][idx_u1][dt1+didx])

            if delay < 0:
                p0.append(self.pdata[0][i00][dt1+didx])
                p0.append(self.pdata[0][i01][dt1+didx])
                p0.append(self.pdata[0][i02][dt1+didx])
                p1.append(self.pdata[1][i10][dt2+didx])
                p1.append(self.pdata[1][i11][dt2+didx])
                p1.append(self.pdata[1][i12][dt2+didx])
                p2.append(self.pointStore(self.pdata[0], dt1+didx))
                p2.append(self.pointStore(self.pdata[1], dt2+didx))
                angle.append(self.data[0][idx_u0][dt1+didx])
                angle.append(self.data[1][idx_u1][dt2+didx])
                
            pl.append(p0)
            pl.append(p1)
            #print p2
            self.pubRviz(pl, p2)
            self.pubPoint(angle)
            rate.sleep()
            #print "didx:"+str(didx)

    def pointStore(self, pl, idx):
        parray = []
        for p in range(len(pl)):
            parray.append(pl[p][idx])
        return parray

    def pubPoint(self, p):
        msg = PointStamped()
        #print p
        msg.header.stamp = rospy.Time.now()
        msg.point.x = p[0]
        msg.point.y = p[1]
        self.ppub.publish(msg)


    def pubRviz(self, pos, joints):

        msgs = MarkerArray()
        for p in range(len(pos)):
            msg = Marker()

            msg.header.frame_id = 'camera_link'
            msg.header.stamp = rospy.Time.now()
            msg.ns = 'marker'
            msg.action = 0
            msg.id = p
            msg.type = 4
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.color = self.carray[2]

            for i in range(len(pos[p])):
                point = Point()
                point.x = pos[p][i][0]
                point.y = pos[p][i][1]
                point.z = pos[p][i][2]
                msg.points.append(point) 
            msg.pose.orientation.w = 1.0
            msgs.markers.append(msg)

        for j in range(len(joints)):
            msg = Marker()

            msg.header.frame_id = 'camera_link'
            msg.header.stamp = rospy.Time.now()
            msg.ns = 'joints'
            msg.action = 0
            msg.id = j
            msg.type = 8
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.color = self.carray[j]

            #print "joints len:"+str(len(joints[j]))
            for i in range(len(joints[j])):
                point = Point()
                point.x = joints[j][i][0]
                point.y = joints[j][i][1]
                point.z = joints[j][i][2]
                msg.points.append(point) 
            msg.pose.orientation.w = 1.0
            msgs.markers.append(msg)

        self.mpub.publish(msgs)

    def updateTable(self):
        self.table.clear()
        self.table.setRowCount(self.maxRange*2+1)
        self.table.resizeRowsToContents()
        
        hor = True

        for i in range(len(self.corrMat)):
            iItem = QtGui.QTableWidgetItem(str(-self.maxRange+i))
            self.table.setVerticalHeaderItem(i, iItem)
            self.table.verticalHeaderItem(i).setToolTip(str(self.timedata[i]))
            for j in range(len(self.corrMat[i])):

                if hor == True:
                    jItem = QtGui.QTableWidgetItem(str(j))
                    self.table.setHorizontalHeaderItem(j, jItem)
                    hot = False

                c = (1+self.corrMat[i][j])*(255/2)
                self.table.setItem(i, j, QtGui.QTableWidgetItem())
                self.table.item(i, j).setBackground(QtGui.QColor(c,c,c))
                self.table.item(i, j).setToolTip(str(self.corrMat[i][j]))

    def process(self, j_idx_u1):
        #print "user1 data[]:"+str(self.data[0])
        #print "user2 data[]:"+str(self.data[1])
        print "data size:"+str(self.dataSize)
        print "joint size:"+str(self.jointSize)
        print "winsize:"+str(self.winSize)
        print "max_range:"+str(self.maxRange)

        self.corrMat = [[0 for i in range(self.jointSize)] for j in range(self.maxRange*2+1)]

        if self.maxRange < 0:
            print "max_range:"+str(self.maxRange)+" is error"
            pass

        for j_idx_u2 in range(0, self.jointSize):
            self.calc_proc(j_idx_u1, j_idx_u2)

    def calc_proc(self, idx_u1, idx_u2): 
        count = 0
        for dt in range(-self.maxRange, self.maxRange+1):
            set1 = []
            set2 = [] 
            for idx in range(0, self.winSize):
                d1_idx = 0
                d2_idx = 0
                if dt >= 0:
                    d1_idx = self.dataSize-self.winSize-abs(dt)
                    d2_idx = self.dataSize-self.winSize

                if dt < 0:
                    d1_idx = self.dataSize-self.winSize                 
                    d2_idx = self.dataSize-self.winSize-abs(dt)
                    
                set1.append(self.data[0][idx_u1][d1_idx+idx])
                set2.append(self.data[1][idx_u2][d2_idx+idx])
            

            
            #print "user1 joint_"+str(idx_u1)+" dt:"+str(dt)+", set1:"+str(set1)
            #print "user2 joint_"+str(idx_u2)+" dt:"+str(dt)+", set2:"+str(set2)

            std1 = self.var_ave(set1)
            std2 = self.var_ave(set2)
            #print "var_ave1:"+str(std1)+", var_ave2:"+str(std2)
            if std1 >= self.varMin and std2 >= self.varMin:
                corr = numpy.corrcoef(set1, set2)
                r_val=corr[0,1]

            if math.fabs(r_val) > self.threshold:
                #print "("+str(idx_u1)+", "+ str(idx_u2)+"): dt:" + str(dt)+", r:"+str(r_val)
                self.corrMat[count][idx_u2]=r_val

            count+=1

    def time_setting(self):
        count = 0
        self.timedata = []
        for dt in range(-self.maxRange, self.maxRange+1):
            if dt > 0:
                self.timedata.append(self.time[abs(dt)]-self.time[0])
            if dt <= 0:
                self.timedata.append(self.time[0]-self.time[abs(dt)])
                    

    def var_ave(self, x):
        ave = numpy.average(x)
        total = 0
        for i in range(len(x)):
            total += abs(x[i] - ave)
        return total/len(x)


def main():
    app = QtGui.QApplication(sys.argv)
    corr = Correlation()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
