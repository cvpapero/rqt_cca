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
        self.winSize = 30

        self.jsonInput()
        self.dataAlig()
        #self.dataNorm()
        self.canoniCorr()

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
                #data.extend(self.jdatas[1][w+i])
                data.extend(self.scale(self.jdatas[1][w+i]))
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
        n, p = tX.shape
        n, q = tY.shape

        sX = tX - tX.mean(axis=0) 
        sY = tY - tY.mean(axis=0)

        print "sX:"
        print sX.T

        S = np.cov(sX.T, sY.T, bias = 1)

        print "cov:"
        print S

        #p,q = S.shape
        #p = p/2
        SXX = S[:p,:p]
        SYY = S[p:,p:]
        SXY = S[:p,p:]
        SYX = S[p:,:p]

        sqx = LA.sqrtm(LA.inv(SXX)) # SXX^(-1/2)
        sqy = LA.sqrtm(LA.inv(SYY)) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        A, s, Bh = LA.svd(M, full_matrices=False)
        B = Bh.T
        
        U = np.dot(np.dot(A.T, sqx), sX.T).T
        V = np.dot(np.dot(B.T, sqy), sY.T).T

        print "s:"
        print s
        print "A:"
        print A
        print "B:"
        print B
        
        print "U:"
        print U.T
        print "V:"
        print V.T
        
        xr, xc = SXX.shape
        print "xr:"+str(xr)
        print "xc:"+str(xc)

        yr, yc = SYY.shape
        print "yr:"+str(yr)
        print "yc:"+str(yc)

        print "p:"+str(p)
        print "q:"+str(q)
        print "n:"+str(n)




        M = -(n-1/2*(xc+yc+3))
        print "M:"+str(M)
        for i in range(len(s)):
            #有意水準を求める
            sig = sp.special.chdtri((xc-i)*(yc-i),0.05)
            print "sig(0.05):"+str(sig)
            test = 1
            for j in range(len(s)-i):
                #print str(s[len(s)-j-1])
                test = test*(1-s[len(s)-j-1])
            chi = M*math.log(test)
            #print "chi:"+str(chi)
            
            if chi > sig:
                print "test["+str(i)+"]:"+str(chi)
            else:
                break
        
        Ucor = np.corrcoef(sX.T,U.T)
        Vcor = np.corrcoef(sY.T,V.T)
        #print "Ucor:"
        #print Ucor
        #print "Vcor:"
        #print Vcor

        #ru = Ucor[:self.range,self.range:self.range+1]
        #rv = Vcor[:self.range,self.range:self.range+1]
        ru = A[:self.range,0:1]
        rv = B[:self.range,0:1]

        #正準構造ベクトル
        #print "ru:"
        #print ru
        #print "rv:"
        #print rv
        print "ru-min val:"+str(np.min(ru))+", arg:"+str(np.argmin(ru))
        print "ru-max val:"+str(np.max(ru))+", arg:"+str(np.argmax(ru))
        print "rv-min val:"+str(np.min(rv))+", arg:"+str(np.argmin(rv))
        print "rv-max val:"+str(np.max(rv))+", arg:"+str(np.argmax(rv))

        
        '''
        #adatasm = np.matrix(self.adatas).T
        #print adatasm
        #fdata = adatasm[:,0:colSize/2]
        #fdata = adatasm[:,:self.range]
        #gdata = adatasm[:,self.range:]

        #print fdata
        #print gdata

        f = np.dot(self.nX, mwx)
        g = np.dot(self.nY, mwy)

        #print f
        #print g

        #print np.corrcoef(fdata[:,0:1].T,f[:,0:1].T)
        #print np.corrcoef(fdata[:,1:2].T,f[:,0:1].T)

        fcorr = np.corrcoef(self.nX.T,f.T)
        gcorr = np.corrcoef(self.nY.T,g.T)

        #print fcorr

        rf = fcorr[:self.range,self.range:self.range+1]
        rg = gcorr[:self.range,self.range:self.range+1]
        
        #rf = rf.T
        #rg = rg.T
        #正準構造ベクトル
        print "rf:"
        print rf
        print "rg:"
        print rg
        
        #寄与率
        fcrat = np.dot(rf.T,rf)/self.range
        gcrat = np.dot(rg.T,rg)/self.range
        
        print "contribution ratio:"
        print fcrat
        print gcrat

        #冗長率
        fred = np.corrcoef(fdata.T,g.T)
        gred = np.corrcoef(gdata.T,f.T)

        fred = fred[:self.range,self.range:self.range+1]
        gred = gred[:self.range,self.range:self.range+1]

        frrat = np.dot(fred.T,fred)/self.range
        grrat = np.dot(gred.T,gred)/self.range
        print "radundancy ratio:"
        print frrat
        print grrat

    '''

def main():
    cancorr = CanCorr()
        

if __name__=='__main__':
    main()
