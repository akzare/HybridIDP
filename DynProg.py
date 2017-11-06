#!/usr/bin/env python

'''
DynProg.py

Description: Dynamic programing  with visibility computation
  
'''

import sys
import numpy as np


class dynProgStruct(object):
    """
    val:      value of cost
    occ:      nb pixel occlusion
    prev:     previous element of solution
    mask:     mask of visibility used  
    XSize:    length of a line
    bestPos:  -1 before optimisation
              otherwise index of the best solution for the initial
              element of the table
    """

    def __init__(self, nblabel = 16, xsize = 1, bestPos = (-1,-1)):
        self.nblabel = nblabel
        self.XSize = xsize
        self.bestPos = bestPos
        self.val = np.zeros((xsize,nblabel), np.uint32)
        self.occ = np.zeros((xsize,nblabel), np.uint32)
        self.prev = np.zeros((xsize,nblabel,2), np.int32)
        self.mask = np.zeros((xsize,nblabel), np.uint32)

    def GETTABLEVAL(self,xxx,yyy):
        return self.val[xxx,yyy]

    def YADR(self,pos):
        y,x = pos
        return x


class DynProg:
    """
    This is a general purpose implentation of a dynamic
    programing algorithm. This implentation should be thread
    safe and multiple dynamicProg object can be simultaniously used.

    Normal calling order is:

    createDynamicProg()
    OptimizeDynamicProg()
    ResultDynamicProg()
    freeDynamicProg()
    """
    def __init__(self, 
                 appSelf, 
                 nblabel, 
                 xsize):
        self.dynamicInt = dynProgStruct(nblabel = nblabel, 
                                        xsize = xsize, 
                                        bestPos = (-1,-1))
        
        self.appSelf = appSelf
        self.nblabel = nblabel
        
        self.SmoothCapacityX = np.vectorize(self.appSelf.SmoothCapacityX)
        self.SmoothCapacityY = np.vectorize(self.appSelf.SmoothCapacityY)
        


    def OptimizeDynamicProg(self,mode):
#        print("Info:DynProg:OptimizeDynamicProg:Begin Dynamic Programming Optimisation...\n")
        
        #initialisation
        idx_nblabel = np.arange(self.nblabel)
        self.dynamicInt.val[0,:] = 0
        self.dynamicInt.prev[0,:,:] = -1
        self.dynamicInt.occ[0,:] = idx_nblabel
        self.dynamicInt.mask[0,:] = 0        
        if mode == 0 :
            indexVal, mask = self.SmoothCapacityX(-1,0,0,idx_nblabel,0,0) # X 
        else:
            indexVal, mask = self.SmoothCapacityY(-1,0,0,idx_nblabel,0,0) # Y

        for y in xrange(1,self.dynamicInt.XSize):
            for x in xrange(self.nblabel):
                occ = 0
                oldmask = 0
                oldmask = self.dynamicInt.mask[y-1,0]
                occ = self.dynamicInt.occ[y-1,0] - y
                index = (y-1,0)
                if mode == 0 :
                    res, mask = self.appSelf.SmoothCapacityX(y-1,0,y,x,occ,oldmask)
                    indexVal = res + self.dynamicInt.val[y-1,0] # X 
                else:
                    res, mask = self.appSelf.SmoothCapacityY(y-1,0,y,x,occ,oldmask)
                    indexVal = res + self.dynamicInt.val[y-1,0] # Y
                    
                idx_nblabel = np.arange(1,self.nblabel)
                occ = self.dynamicInt.occ[y-1,1:self.nblabel] - y
                oldmask = self.dynamicInt.mask[y-1,1:self.nblabel]
                if mode == 0 :
                    res, tmpmask = self.SmoothCapacityX(y-1,idx_nblabel,y,x,occ,oldmask)
                    tmp = res + self.dynamicInt.val[y-1,1:self.nblabel] # X 
                else:
                    res, tmpmask = self.SmoothCapacityY(y-1,idx_nblabel,y,x,occ,oldmask)
                    tmp = res + self.dynamicInt.val[y-1,1:self.nblabel] # Y
                tmp_min = np.amin(tmp)
                if tmp_min < indexVal :
                    tmp_min_idx = np.argmin(tmp)
                    mask = tmpmask[tmp_min_idx]
                    index = (y-1,tmp_min_idx+1)
                    indexVal = tmp[tmp_min_idx]

                # fixe value and previous  
                self.dynamicInt.val[y,x] = indexVal  
                self.dynamicInt.prev[y,x,:] = index

                # we save udes mask if required to do so 
                self.dynamicInt.mask[y,x] = mask  
                # we update with the most occlusion doner 
                if self.dynamicInt.occ[index] < x+y :
                    self.dynamicInt.occ[y,x] = x+y
                else:
                    self.dynamicInt.occ[y,x] = self.dynamicInt.occ[index]
#                print("(%d %d)(%d)=%f  " % (y,x,index,indexVal))

#            print("\n")

        # localize best final configuration
        index = (self.dynamicInt.XSize-1,0);
        indexVal = self.dynamicInt.GETTABLEVAL(self.dynamicInt.XSize-1,0)
        for x in xrange(1,self.nblabel):
            tmp = self.dynamicInt.GETTABLEVAL(self.dynamicInt.XSize-1,x)
            if tmp < indexVal :
                index = (self.dynamicInt.XSize-1,x)
                indexVal = tmp
        self.dynamicInt.bestPos = index # we put the final solution for easy extractraction

#        print("Info:DynProg:OptimizeDynamicProg:End Dynamic Programming Optimisation\n")


    def ResultDynamicProg(self,size):
#        print("Info:DynProg:ResultDynamicProg...\n")
        prev = self.dynamicInt.bestPos
        result = np.zeros((size), np.uint8)
        result[size-1] = self.dynamicInt.YADR(prev)
        for i in range(size-2,-1,-1):
            y,x = prev
            prev = self.dynamicInt.prev[y,x,:]
            result[i] = self.dynamicInt.YADR(prev)
        return result


    def ResultVisibilityDynamicProg(self,size):
#        print("Info:DynProg:ResultVisibilityDynamicProg...\n")
        # extract minimum cut associate with minimum energy
        prev = self.dynamicInt.bestPos
        result = np.zeros((size), np.uint8)
        result[size-1] = self.dynamicInt.mask[prev]
        for i in range(size-2,-1,-1):
            y,x = prev
            prev = self.dynamicInt.prev[y,x,:]
            y,x = prev
            result[i] = self.dynamicInt.mask[y,x] 
        return result


    def SmoothCapacityNaka(self):
        pass
