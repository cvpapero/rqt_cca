#!/usr/bin/python
# -*- coding: utf-8 -*-


#標準化する

import sys
import math
import json
import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy import stats as ST
from scipy.spatial import distance as DIST
import sympy as SY
from scipy import integrate

#from sklearn.cross_decomposition import CCA

class CanCorr():

    def __init__(self):
        self.winSize = 40

        self.jsonInput()
        self.dataAlig()
        #self.dataNorm()
        self.canoniCorr()
        self.bartlettTest()

    #データのインプット(from json)
    def jsonInput(self):
        f = open('test1014.json', 'r');
        jsonData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()

        self.jdatas = []
        for user in jsonData:
            #angle
            datas = []
            self.dataRow = len(user["datas"])

            for j in range(self.dataRow):
                datas.append(user["datas"][j]["data"])

            self.jdatas.append(datas)
        #print "jdatas:"
        #print(self.jdatas)

    #データの整頓(winsize分だけ並べる)
    def dataAlig(self):

        self.X = []
        self.Y = []

        self.range = self.dataRow-self.winSize+1

        for i in range(self.range):
            data = []
            for w in range(self.winSize):
                #data.extend(self.jdatas[0][w+i])
                data.extend(self.scale(self.jdatas[0][w+i]))
            self.X.append(data)
       
        for i in range(self.range):
            data = []
            for w in range(self.winSize):
                data.extend(self.jdatas[1][w+i])
                #data.extend(self.scale(self.jdatas[1][w+i]))
            self.Y.append(data)
   
    
        '''
        print "dataAlig_X:"
        for i in range(len(self.X)):
            print(self.X[i])
        print("---")

        print "dataAlig_Y:"
        for i in range(len(self.Y)):
            print(self.Y[i])
        print("---")
        '''

    #平均=0
    def scale(self, arr):
        dst = []
        m = np.mean(arr)
        dst = np.array([(arr[i] - m) for i in range(len(arr))])
        return dst

    #正準相関
    def canoniCorr(self):
        tX = np.matrix(self.X).T
        tY = np.matrix(self.Y).T
        self.n, self.p = tX.shape
        self.n, self.q = tY.shape

        sX = tX - tX.mean(axis=0) 
        sY = tY - tY.mean(axis=0)

        print "sX:"
        print sX

        S = np.cov(sX.T, sY.T, bias = 1)

        print "cov:"
        print S

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
        
        #self.U = np.dot(np.dot(self.A.T, sqx), sX.T).T
        #self.V = np.dot(np.dot(self.B.T, sqy), sY.T).T

        print "s:"
        print self.s
        print "A:"
        print self.A
        print "B:"
        print self.B
        
        '''
        #正準構造ベクトル(縦)
        f = np.dot(sX, self.A)
        fc = np.corrcoef(f.T,sX.T)
        fc = fc[:self.p,self.p:].T
        print fc

        g = np.dot(sY, self.B)
        gc = np.corrcoef(g.T,sY.T)
        gc = gc[:self.p,self.p:].T
        print gc

        ru = fc[:,0:1]
        rv = gc[:,0:1]
        print "ru-max val:"+str(np.max(ru))+", arg:"+str(np.argmax(ru))
        print "rv-max val:"+str(np.max(rv))+", arg:"+str(np.argmax(rv))
        '''

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
                #ru = self.A[:,i:i+1]
                #rv = self.B[:,i:i+1]
                #print "ru-max val:"+str(np.max(ru))+", arg:"+str(np.argmax(ru))
                #print "rv-max val:"+str(np.max(rv))+", arg:"+str(np.argmax(rv))
                print "ru-max arg:"+str(np.argmax(ru))
                print "rv-max arg:"+str(np.argmax(rv))
                #print "---"
            else:
                break
        

        #ru = np.fabs(self.A[:self.range,0:1])
        #ru = self.A[:self.range,0:1]
        #rv = np.fabs(self.B[:self.range,0:1])
        #print "ru-min val:"+str(np.min(ru))+", arg:"+str(np.argmin(ru))
        #print "ru-max val:"+str(np.max(ru))+", arg:"+str(np.argmax(ru))
        #print "rv-min val:"+str(np.min(rv))+", arg:"+str(np.argmin(rv))
        #print "rv-max val:"+str(np.max(rv))+", arg:"+str(np.argmax(rv))

        #Ucor = np.corrcoef(sX.T,U.T)
        #Vcor = np.corrcoef(sY.T,V.T)
        #print "Ucor:"
        #print Ucor
        #print "Vcor:"
        #print Vcor

        #ru = Ucor[:self.range,self.range:self.range+1]
        #rv = Vcor[:self.range,self.range:self.range+1]
        #ru = A[:self.range,0:1]
        #rv = B[:self.range,0:1]

        #正準構造ベクトル
        #print "ru:"
        #print ru
        #print "rv:"
        #print rv
        #print "ru-min val:"+str(np.min(ru))+", arg:"+str(np.argmin(ru))
        #print "ru-max val:"+str(np.max(ru))+", arg:"+str(np.argmax(ru))
        #print "rv-min val:"+str(np.min(rv))+", arg:"+str(np.argmin(rv))
        #print "rv-max val:"+str(np.max(rv))+", arg:"+str(np.argmax(rv))

        
 

def main():
    cancorr = CanCorr()
        

if __name__=='__main__':
    main()
