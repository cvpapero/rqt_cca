#!/usr/bin/python
# -*- coding: utf-8 -*-

#そもそものデータの取りかたが間違っていたので直す
#time1=0についてtime2=0~Nで相関をとるバージョン

import sys
import math
import json
import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy import stats as ST
from scipy.spatial import distance as DIST
#import sympy as SY
from scipy import integrate

#from sklearn.cross_decomposition import CCA

class CanCorr():

    def __init__(self):
        self.winSize = 30

        self.jsonInput()
        #self.canoniExec1()
        self.canoniExec2()
        #self.canoniCorr()
        #self.bartlettTest()

    #データのインプット(from json)
    def jsonInput(self):
        f = open('test1014.json', 'r');
        jsonData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()

        self.DATAS = []
        #self.Y = []

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

    def canoniExec1(self):
        
        dataRange = self.datasSize - self.winSize + 1
        
        for t1 in range(dataRange):
            rho = 0
            time1 = 0
            time2 = 0
            for t2 in range(dataRange):
                USER1 = []
                USER2 = []
                for w in range(self.winSize):
                    USER1.append(self.DATAS[0][t1+w])
                    USER2.append(self.DATAS[1][t2+w])
                    
                tmp_rho = self.canoniCorr(USER1, USER2)
                #print "tmp_rho"+str(tmp_rho)
                
                if math.fabs(tmp_rho) > math.fabs(rho):
                    rho = tmp_rho                
                    time1 = t1
                    time2 = t2
                
            print "---"
            print "user1 t:"+str(time1)+", user2 t:"+str(time2)+", delay(t1-t2):"+str(time1-time2)+", rho:"+str(rho)
            #print "USER1:"+str(USER1)
            #print "USER2:"+str(USER2)
            

    def canoniExec2(self):
        dataRange = self.datasSize - self.winSize 
        max_rho = 0
        max_time1 = 0
        max_time2 = 0
        max_d = 0
        for d in range(-dataRange, dataRange+1):
            rho = 0
            time1 = 0
            time2 = 0

            if d < 0:
                for r in range(self.datasSize-np.abs(d)-self.winSize+1):
                    USER1 = []
                    USER2 = []
                    t1 = np.abs(d)+r
                    t2 = r
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][t1+w])
                        USER2.append(self.DATAS[1][t2+w])
                    #print "USER1:"+str(USER1)
                    #print "USER2:"+str(USER2)
                    tmp_rho = self.canoniCorr(USER1, USER2)
                    if math.fabs(tmp_rho) > math.fabs(rho):
                        rho = tmp_rho                
                        time1 = t1
                        time2 = t2
                    #print "tmp_rho:"+str(tmp_rho)
                    #print "---"
            else:
                for r in range(self.datasSize-np.abs(d)-self.winSize+1):
                    USER1 = []
                    USER2 = []   
                    t1 = r
                    t2 = np.abs(d)+r
                    for w in range(self.winSize):
                        USER1.append(self.DATAS[0][t1+w])
                        USER2.append(self.DATAS[1][t2+w])
                    #print "USER1:"+str(USER1)
                    #print "USER2:"+str(USER2)
                    tmp_rho = self.canoniCorr(USER1, USER2)
                    if math.fabs(tmp_rho) > math.fabs(rho):
                        rho = tmp_rho                
                        time1 = t1
                        time2 = t2
                    #print "tmp_rho:"+str(tmp_rho)
            print "delay["+str(d) + "], t1:"+str(time1)+", t2:"+str(time2)+", rho:"+str(rho)
            print "---"

            if math.fabs(rho) > math.fabs(max_rho):
                max_rho = rho                
                max_time1 = time1
                max_time2 = time2
                max_d = d
        #あまり意味は無いかも,,,
        print "max delay["+str(max_d) + "], t1:"+str(max_time1)+", t2:"+str(max_time2)+", rho:"+str(max_rho)

    #正準相関
    def canoniCorr(self, U1, U2):

        #配列から行列へ(あらかじめ転置しておく)
        tX = np.matrix(U1).T
        tY = np.matrix(U2).T
        #print "tX:"
        #print tX
        #print "tY:"
        #print tY
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
        #print a

        #固有値問題
        #eighは実対称行列のみ, 固有値・ベクトルは昇順(小→大)
        lambs, vecs = LA.eigh(a)
        #print "vecs:"
        #print vecs
        #print "lamb:"
        #print lambs
        
        #最大の固有値・ベクトルをとる
        vr, vc = vecs.shape
        vec1 = vecs[:,vc-1:vc]
        lamb = np.sqrt(lambs[vc-1])
        #print vec1
        #print lamb
        #print np.cov(sX[:,0:1].T,sY[:,0:1].T)
        vec2 = np.dot(sxy.T, vec1)/lamb
        #print vec2
        
        v1sxyv2 = np.dot(np.dot(vec1.T,sxy),vec2)
        v1sxxv1 = np.dot(np.dot(vec1.T,sxx),vec1)
        v2syyv2 = np.dot(np.dot(vec2.T,syy),vec2)

        rho = v1sxyv2 / np.sqrt(np.dot(v1sxxv1,v2syyv2))
        #print "rho:"
        return rho


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
                #print  "test["+str(i)+"]:"+str(chi) +" > sig("+str(alf)+"):"+str(sig)
                ru = np.fabs(self.A[:,i:i+1])
                rv = np.fabs(self.B[:,i:i+1])
                #ru = self.A[:,i:i+1]
                #rv = self.B[:,i:i+1]
                #print "ru-max val:"+str(np.max(ru))+", arg:"+str(np.argmax(ru))
                #print "rv-max val:"+str(np.max(rv))+", arg:"+str(np.argmax(rv))
                #print "ru-max arg:"+str(np.argmax(ru))
                #print "rv-max arg:"+str(np.argmax(rv))
                #print "---"
            else:
                break

def main():
    cancorr = CanCorr()
        

if __name__=='__main__':
    main()
