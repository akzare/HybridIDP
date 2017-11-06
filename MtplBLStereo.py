#!/usr/bin/env python

'''
MtplBLStereo.py

Description: Compute depth map ultra fast for multiple-baseline
             configuration using IDP with hybrid (geo-coherent VS photo-cohenrent)
             visibility approach.

Usage:
    MtplBLStereo.py [<params>]
    params:
        --constparm: print constant parameters
        --help:  print this help
'''

# Python 2/3 compatibility
from __future__ import print_function

import sys

from tempfile import TemporaryFile

from multiprocessing.pool import ThreadPool
from collections import deque
from multiprocessing import Pool, TimeoutError

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import Constants as constparm
import DynProg as dp


def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v


class App(object):
    def __init__(self,
                 GAMA=19,
                 LAMBDA=20,
                 ITERATION=1,
                 ROBUST=1,
                 KEEP_VISIBILITY=1,
                 VISIBILITY_2D=1,
                 USED_NAKAMURA=0,
                 NAKAMURA_2D=0,
                 FACTOR_SAVE=16,
                 NBLABEL=16,
                 BORDERX=3,
                 BORDERY=3,
                 LABEL_FACTOR=1,
                 IMAGE_TEST='../data/razz.gif',
                 IMAGE_LEFT='../data/scene1.row3.col2.ppm',
                 IMAGE_RIGHT='../data/scene1.row3.col4.ppm',
                 IMAGE_REF='../data/scene1.row3.col3.ppm',
                 IMAGE_BOTTOM='../data/scene1.row4.col3.ppm',
                 IMAGE_TOP='../data/scene1.row2.col3.ppm',
                 GROUNDTRUE='../data/truedisp.pgm'):
        
        print("Info:mtplBLStereoApp:init")
        
        ################################################################################
        # constant such as image path and parameters for stereo:
        # GAMA: mask smoothing parameter
        # LAMBDA: smoothing parameter
        # ITERATION: maximum number of iteration before convergence is ensure
        #            1 : 1.78 % error
        #            4 : 1.62 % error
        #            8 : 1.60 % error
        #            16: 1.57 % error
        # ROBUST: using pot model for smoothing else used a linear smoothing model
        # KEEP_VISIBILITY: if you remove this there is no smoothing on the visibility mask
        # VISIBILITY_2D: visibility of previous iteration are used this is the description used in the paper
        # USED_NAKAMURA: replace our scheme by nakamura
        # NAKAMURA_2D: use 2 camera mask or allow 1 camre mask
        # GROUNDTRUE: parameter for head and lamp scene
        # FACTOR_SAVE: disparity * FACTOR_SAVE = resulting intensity
        self.ConstParams = constparm.Constants(GAMA=GAMA, 
                                               LAMBDA=LAMBDA,
                                               ITERATION=ITERATION,
                                               ROBUST=ROBUST,
                                               KEEP_VISIBILITY=KEEP_VISIBILITY,
                                               VISIBILITY_2D=VISIBILITY_2D,
                                               USED_NAKAMURA=USED_NAKAMURA,
                                               NAKAMURA_2D=NAKAMURA_2D,
                                               FACTOR_SAVE=FACTOR_SAVE,
                                               NBLABEL=NBLABEL,
                                               BORDERX=BORDERX,
                                               BORDERY=BORDERY,
                                               LABEL_FACTOR=LABEL_FACTOR,
                                               IMAGE_TEST=IMAGE_TEST,
                                               IMAGE_LEFT=IMAGE_LEFT,
                                               IMAGE_RIGHT=IMAGE_RIGHT,
                                               IMAGE_REF=IMAGE_REF,
                                               IMAGE_BOTTOM=IMAGE_BOTTOM,
                                               IMAGE_TOP=IMAGE_TOP,
                                               GROUNDTRUE=GROUNDTRUE
                                               )

        # what we call one iteration is 2 in the code here
        self.nbIteration = self.ConstParams.ITERATION * 2

        # scan from left to right or from right to left 
        # samething for top bottom and bottom top 
        # set to one to change order 
        # The algorithme does change does values
        self.reverseOrderX = 0
        self.reverseOrderY = 0
        
        # smoothing parameter
        self.smooth = LAMBDA  # disparity smoothing
        self.gama = GAMA      # visibility mask smoothing
        
        # used to keep curent colomne or line when optimizing line or colonm 
        self.Cy = 0 # place Y when optimisation in X 
        self.Cx = 0 # place X when optimisation in Y
        
        self.first = 1


    def loadImages(self):
        print("Info:mtplBLStereoApp:loadImages")
        print('\t--Loading images...')
        
        print('\t\t--Loading refernce image...')        
        self.imgRef = cv2.imread(self.ConstParams.IMAGE_REF)
        if self.imgRef is None:
            print('Error:mtplBLStereoApp:init:Failed to load reference image:', self.ConstParams.IMAGE_REF)
            sys.exit(1)

        self.IMYSIZE, self.IMXSIZE, self.IMZSIZE = self.imgRef.shape[:]
        print('Info:mtplBLStereoApp:init:Image size: IMYSIZE:%d, IMXSIZE:%d, IMZSIZE:%d' % (self.IMYSIZE,self.IMXSIZE,self.IMZSIZE))

        if self.IMYSIZE == 0 or self.IMXSIZE == 0:
            print('Error:mtplBLStereoApp:init:Image size should not be zero:IMYSIZE:%d, IMXSIZE:%d' % (self.IMYSIZE,self.IMXSIZE))
            sys.exit(1)
        
        print('\t\t--Loading left image...')        
        self.imgLeft = cv2.imread(self.ConstParams.IMAGE_LEFT)
        if self.imgLeft is None:
            print('Error:mtplBLStereoApp:init:Failed to load left image:', self.ConstParams.IMAGE_LEFT)
            sys.exit(1)
        IMYSIZETemp, IMXSIZETemp, IMZSIZETemp = self.imgLeft.shape[:]
        if self.IMYSIZE != IMYSIZETemp or self.IMXSIZE != IMXSIZETemp or self.IMZSIZE != IMZSIZETemp:
            print('Error:mtplBLStereoApp:init:Left image size should be equal to reference image:IMYSIZE:%d, IMXSIZE:%d, IMZSIZE:%d' % (IMYSIZETemp,IMXSIZETemp,IMZSIZETemp))
            sys.exit(1)
        
        print('\t\t--Loading right image...')        
        self.imgRight = cv2.imread(self.ConstParams.IMAGE_RIGHT)
        if self.imgRight is None:
            print('Error:mtplBLStereoApp:init:Failed to load right image:', self.ConstParams.IMAGE_RIGHT)
            sys.exit(1)
        IMYSIZETemp, IMXSIZETemp, IMZSIZETemp = self.imgLeft.shape[:]
        if self.IMYSIZE != IMYSIZETemp or self.IMXSIZE != IMXSIZETemp or self.IMZSIZE != IMZSIZETemp:
            print('Error:mtplBLStereoApp:init:Right image size should be equal to reference image:IMYSIZE:%d, IMXSIZE:%d, IMZSIZE:%d' % (IMYSIZETemp,IMXSIZETemp,IMZSIZETemp))
            sys.exit(1)
        
        print('\t\t--Loading bottom image...')        
        self.imgBot = cv2.imread(self.ConstParams.IMAGE_BOTTOM)
        if self.imgBot is None:
            print('Error:mtplBLStereoApp:init:Failed to load bottom image:', self.ConstParams.IMAGE_BOTTOM)
            sys.exit(1)
        IMYSIZETemp, IMXSIZETemp, IMZSIZETemp = self.imgLeft.shape[:]
        if self.IMYSIZE != IMYSIZETemp or self.IMXSIZE != IMXSIZETemp or self.IMZSIZE != IMZSIZETemp:
            print('Error:mtplBLStereoApp:init:Bottom image size should be equal to reference image:IMYSIZE:%d, IMXSIZE:%d, IMZSIZE:%d' % (IMYSIZETemp,IMXSIZETemp,IMZSIZETemp))
            sys.exit(1)
        
        print('\t\t--Loading top image...')        
        self.imgTop = cv2.imread(self.ConstParams.IMAGE_TOP)
        if self.imgTop is None:
            print('Error:mtplBLStereoApp:init:Failed to load top image:', self.ConstParams.IMAGE_TOP)
            sys.exit(1)
        IMYSIZETemp, IMXSIZETemp, IMZSIZETemp = self.imgTop.shape[:]
        if self.IMYSIZE != IMYSIZETemp or self.IMXSIZE != IMXSIZETemp or self.IMZSIZE != IMZSIZETemp:
            print('Error:mtplBLStereoApp:init:Top image size should be equal to reference image:IMYSIZE:%d, IMXSIZE:%d, IMZSIZE:%d' % (IMYSIZETemp,IMXSIZETemp,IMZSIZETemp))
            sys.exit(1)

        print('\t\t--Loading ground image...')        
        self.imgGndTrue = cv2.imread(self.ConstParams.GROUNDTRUE)
        if self.imgGndTrue is None:
            print('Error:mtplBLStereoApp:init:Failed to load ground image:', self.ConstParams.GROUNDTRUE)
            sys.exit(1)        
        
        # depth map
        self.imgDepthMap = np.zeros((self.IMYSIZE, self.IMXSIZE), np.uint8)
        
        # visibility information
        self.visibilityInfo = np.zeros((self.IMYSIZE, self.IMXSIZE), np.uint32)
        
        # this is used to keep track of visibility outside of epipolar line
        self.visibilityOutEpi = np.zeros((self.IMYSIZE, self.IMXSIZE), np.uint32)

        # matching volume DSI  section 3.2 of working document
        # indexing order is different because access locality is different between left right DSI and bottom and top DSI 
        self.dsiLeft = np.zeros((self.IMYSIZE, self.IMXSIZE, self.ConstParams.NBLABEL), np.uint32) # DSI left
        self.dsiRight = np.zeros((self.IMYSIZE, self.IMXSIZE, self.ConstParams.NBLABEL), np.uint32) # DSI right
        self.dsiBot = np.zeros((self.IMXSIZE, self.IMYSIZE, self.ConstParams.NBLABEL), np.uint32) # DSI bottom
        self.dsiTop = np.zeros((self.IMXSIZE, self.IMYSIZE, self.ConstParams.NBLABEL), np.uint32) # DSI top

        self.smoothX = np.zeros((self.IMYSIZE, self.IMXSIZE), np.uint32)
        self.smoothY = np.zeros((self.IMYSIZE, self.IMXSIZE), np.uint32)


    def showParameters(self):
        print("Info:mtplBLStereoApp:showParameters")
        print("\t--GAMA\t\t %d" % (self.ConstParams.GAMA))
        print("\t--LAMBDA\t\t %d" % (self.ConstParams.LAMBDA))
        print("\t--ITERATION\t\t %d" % (self.ConstParams.ITERATION))
        print("\t--ROBUST\t\t %d" % (self.ConstParams.ROBUST))
        print("\t--KEEP_VISIBILITY\t\t %d" % (self.ConstParams.KEEP_VISIBILITY))
        print("\t--VISIBILITY_2D\t\t %d" % (self.ConstParams.VISIBILITY_2D))
        print("\t--USED_NAKAMURA\t\t %d" % (self.ConstParams.USED_NAKAMURA))
        print("\t--NAKAMURA_2D\t\t %d" % (self.ConstParams.NAKAMURA_2D))
        print("\t--FACTOR_SAVE\t\t %d" % (self.ConstParams.FACTOR_SAVE))
        print("\t--NBLABEL\t\t %d" % (self.ConstParams.NBLABEL))
        print("\t--BORDERX\t\t %d" % (self.ConstParams.BORDERX))
        print("\t--BORDERY\t\t %d" % (self.ConstParams.BORDERY))
        print("\t--LABEL_FACTOR\t\t %d" % (self.ConstParams.LABEL_FACTOR))
        print("\t--IMAGE_LEFT\t\t %s" % (self.ConstParams.IMAGE_LEFT))
        print("\t--IMAGE_RIGHT\t\t %s" % (self.ConstParams.IMAGE_RIGHT))
        print("\t--IMAGE_REF\t\t %s" % (self.ConstParams.IMAGE_REF))
        print("\t--IMAGE_BOTTOM\t\t %s" % (self.ConstParams.IMAGE_BOTTOM))
        print("\t--IMAGE_TOP\t\t %s" % (self.ConstParams.IMAGE_TOP))
        print("\t--GROUNDTRUE\t\t %s" % (self.ConstParams.GROUNDTRUE))


    def run(self):
        print("Info:mtplBLStereoApp:run")

        time_interval = StatValue()

        self.loadImages()

        print('Info:mtplBLStereoApp:run:Interpolating image...')
        last_frame_time = clock()        
        self.imgLeftInterplt = self.computeInterpolateImage(imgIn=self.imgLeft)
        self.imgRefInterplt = self.computeInterpolateImage(imgIn=self.imgRef)
        self.imgRightInterplt = self.computeInterpolateImage(imgIn=self.imgRight)
        self.imgBotInterplt = self.computeInterpolateImage(imgIn=self.imgBot)
        self.imgTopInterplt = self.computeInterpolateImage(imgIn=self.imgTop)
        t = clock()
        time_interval.update(t - last_frame_time)
        print('Info:mtplBLStereoApp:run:Interpolating image time interval :  %.1f ms' % (time_interval.value*1000))

        print('Info:mtplBLStereoApp:run:Computing DSI...')
        last_frame_time = clock()
        self.computeCostFunction()
        t = clock()
        time_interval.update(t - last_frame_time)
        print('Info:mtplBLStereoApp:run:Computing DSI time interval :  %.1f ms' % (time_interval.value*1000))

        print('Info:mtplBLStereoApp:run:Computing cost function...')
        self.reverseOrderX = 0
        self.reverseOrderY = 1
        last_frame_time = clock()
        self.computeSmoothFunction()
        t = clock()
        time_interval.update(t - last_frame_time)
        print('Info:mtplBLStereoApp:run:Computing cost function time interval :  %.1f ms' % (time_interval.value*1000))

        print('Info:mtplBLStereoApp:run:Optimizing...')
        last_frame_time = clock()
        res = self.optimize5images()        
        t = clock()
        time_interval.update(t - last_frame_time)
        print('Info:mtplBLStereoApp:run:Optimizing time interval :  %.1f ms' % (time_interval.value*1000))

        print('Info:mtplBLStereoApp:run:Post processing...')
        last_frame_time = clock()
        self.postProcessing()
        t = clock()
        time_interval.update(t - last_frame_time)
        print('Info:mtplBLStereoApp:run:Post processing time interval :  %.1f ms' % (time_interval.value*1000))

#        imgDepthMap_uint8 = self.imgDepthMap.astype('uint8')
#        np.savetxt('imgDepthMap_uint8.txt', imgDepthMap_uint8, delimiter=',')
#        cv2.imwrite('../data/depth.png',imgDepthMap_uint8)

        np.savetxt('imgDepthMap_uint8.txt', self.imgDepthMap, delimiter=',')
        cv2.imwrite('../data/depth.png',self.imgDepthMap)

        cv2.imshow('left', self.imgLeft)
        cv2.imshow('reference', self.imgRef)
        cv2.imshow('right', self.imgRight)
        cv2.imshow('bottom', self.imgBot)
        cv2.imshow('top', self.imgTop)
        cv2.imshow('depth', self.imgDepthMap)


    def cmpPixel(self, imgFirst, imgSec, imgFirstIntpol, imgSecIntpol, imgFirstCoord, imgSecCoord):
        """
        this is the matching cost function  
        based on Birchfield and tomasi PAMI paper and Kologorov modification also PAMI describe in section 
        3.2 of working document and known as function CostBirchfield in document
        """
        x1, y1 = imgFirstCoord
        x2, y2 = imgSecCoord
        res = 0    
        for i in xrange(self.IMZSIZE):
            c1a = 2*imgFirst[y1,x1,i] # compensate for interpolation 
            c2a = 2*imgSec[y2,x2,i]
    
            # interpolation value of image 1
            c1b = imgFirstIntpol[y1,x1,i,0]
            c1d = imgFirstIntpol[y1,x1,i,1]
            c1c = imgFirstIntpol[y1,x1-1,i,0]
            c1e = imgFirstIntpol[y1-1,x1,i,1]
    
            # interpolation of image 2
            c2b = imgSecIntpol[y2,x2,i,0]      
            c2d = imgSecIntpol[y2,x2,i,1]
            c2c = imgSecIntpol[y2,x2-1,i,0]
            c2e = imgSecIntpol[y2-1,x2,i,1]
    
            tmp  = abs( c1a-c2a )
            tmp1 = abs( c1a-c2b )
            tmp2 = abs( c1a-c2c )
            tmp3 = abs( c1a-c2d )
            tmp4 = abs( c1a-c2e )
    
            if tmp2 < tmp :
                tmp = tmp2
            if tmp1 < tmp :
                tmp = tmp1
            if tmp3 < tmp :
                tmp = tmp3
            if tmp4 < tmp :
                tmp = tmp4
         
            tmp2 = abs( c2a-c1b )
            tmp3 = abs( c2a-c1c )
            tmp4 = abs( c2a-c1d )
            tmp5 = abs( c2a-c1e )
          
            if tmp3 < tmp :
                tmp = tmp3
            if tmp2 < tmp :
                tmp = tmp2
            if tmp4 < tmp :
                tmp = tmp4
            if tmp5 < tmp :
                tmp = tmp5
    
            if tmp > 2000 :
                tmp = 2000 # maximum values
                
            res = res + tmp
        return res


    def computeInterpolateImage(self, imgIn):
        """
        we fill the interpolation information to be used latter in the cost function  cmpPixel function */
        we compute the value of intensity of half pixel (i.e. (x+.5,y) (x,y+.5))
        we do not stock the halft value but 2x it to save one division 
        """
        # b = imgIn[:,:,0]
        # g = imgIn[:,:,1]
        # r = imgIn[:,:,2]
        
        # interpolate image used to compute DSI value
        # access is as folloing   acceding pixel (x+.5,y)  di[x][y][0]
        #                                        (x-.5,y)  di[x-1][y][0]
        #                                        (x,y+.5)  di[x][y+1][1]
        #                                        (x,y-.5)  di[x][y-1][1]
        imgIntpol = np.zeros((self.IMYSIZE, self.IMXSIZE, self.IMZSIZE, 2), np.uint32)
        imgIntpol[0:self.IMYSIZE,0:self.IMXSIZE-1,:,0] = imgIn[0:self.IMYSIZE,0:self.IMXSIZE-1,:]+imgIn[0:self.IMYSIZE,1:self.IMXSIZE,:]
        imgIntpol[0:self.IMYSIZE-1,0:self.IMXSIZE,:,1] = imgIn[0:self.IMYSIZE-1,0:self.IMXSIZE,:]+imgIn[1:self.IMYSIZE,0:self.IMXSIZE,:]    
        return imgIntpol


    def cmpNaka(self):
        pass


    def computeCostFunction(self):
        """
        Filling the  DSI values
        LABEL_FACTOR is a constant should be one in all case exepte want disparity is compute at step different then one
        when no value can be compute because reprojection fall outside of
        supporting image we use maximum possible cost function value to
        flag this situation
        """
        for y in xrange(self.ConstParams.BORDERY,self.IMYSIZE-self.ConstParams.BORDERY): # these a small border for with we do not compute depth
            for x in xrange(self.ConstParams.BORDERX,self.IMXSIZE-self.ConstParams.BORDERX):
                for z in xrange(self.ConstParams.NBLABEL):
                    # left DSI
                    if (x+self.ConstParams.LABEL_FACTOR*z) < self.IMXSIZE :
                        self.dsiLeft[y,x,z] = self.cmpPixel(self.imgRef,
                                                            self.imgLeft,
                                                            self.imgRefInterplt,
                                                            self.imgLeftInterplt,
                                                            (x,y),
                                                            (x+self.ConstParams.LABEL_FACTOR*z,y))
                    else:
                        self.dsiLeft[y,x,z] = sys.maxint # indicate outside of supporting image 
                    # right DSI
                    if x-self.ConstParams.LABEL_FACTOR*z >= 0 :
                        self.dsiRight[y,x,z] = self.cmpPixel(self.imgRef,
                                                             self.imgRight,
                                                             self.imgRefInterplt,
                                                             self.imgRightInterplt,
                                                             (x,y),
                                                             (x-self.ConstParams.LABEL_FACTOR*z,y))
                    else:
                        self.dsiRight[y,x,z] = sys.maxint # indicate outside of supporting image 
    
        for x in xrange(self.ConstParams.BORDERX,self.IMXSIZE-self.ConstParams.BORDERX):
            for y in xrange(self.ConstParams.BORDERY,self.IMYSIZE-self.ConstParams.BORDERY): # these a small border for with we do not compute depth
                for z in xrange(self.ConstParams.NBLABEL):
                    # bottom DSI
                    if (y+self.ConstParams.LABEL_FACTOR*z) < self.IMYSIZE :
                        self.dsiBot[x,y,z] = self.cmpPixel(self.imgRef,
                                                           self.imgBot,
                                                           self.imgRefInterplt,
                                                           self.imgBotInterplt,
                                                           (x,y),(x,y+self.ConstParams.LABEL_FACTOR*z))
                    else:
                        self.dsiBot[x,y,z] = sys.maxint # indicate outside of supporting image 
                    # top DSI
                    if (y-self.ConstParams.LABEL_FACTOR*z) >= 0 :
                        self.dsiTop[x,y,z] = self.cmpPixel(self.imgRef,
                                                           self.imgTop,
                                                           self.imgRefInterplt,
                                                           self.imgTopInterplt,
                                                           (x,y),
                                                           (x,y-self.ConstParams.LABEL_FACTOR*z))
                    else:
                        self.dsiTop[x,y,z] = sys.maxint # indicate outside of supporting image


    def computeSmoothFunction(self):
        """
        compute the smoothing function g
        """
        # smoothing g function  section 3.1 of working document
        # access is as fallow:  g(x,y  , x+1,y) is in  smoothX[x][y]   
        #                       g(x,y  , x-1,y) is in  smoothX[x-1][y]  
        #                       g(x-1,y  , x,y) is in  smoothX[x-1][y]  
        #                       g(x+1,y  , x,y) is in  smoothX[x+1][y]  
    
        # access is as fallow:  g(x,y  , x,y+1) is in  smoothY[x][y]   
        #                       g(x,y  , x,y-1) is in  smoothY[x][y-1]  
        #                       g(x,y-1  , x,y) is in  smoothY[x][y-1]  
        #                       g(x,y+1  , x,y) is in  smoothY[x][y+1]  
        for y in xrange(self.ConstParams.BORDERY,self.IMYSIZE-self.ConstParams.BORDERY-1): # we alway remove part of the reference image for with we do not compute depth
            for x in xrange(self.ConstParams.BORDERX,self.IMXSIZE-self.ConstParams.BORDERX-1):
                self.smoothX[y,x] = self.smooth # this is lambda 
                self.smoothY[y,x] = self.smooth
                # smoothing is proportinal to image gradiant this come from kolmogorov 2002 paper
                diff = abs(np.int32(self.imgRef[y,x,0])-np.int32(self.imgRef[y,x+1,0]))+abs(np.int32(self.imgRef[y,x,1])-np.int32(self.imgRef[y,x+1,1])+abs(np.int32(self.imgRef[y,x,2])-np.int32(self.imgRef[y,x+1,2])))
        
                if diff < 3*5 :
                    self.smoothX[y,x] = self.smoothX[y,x] + 2 * self.smooth # we put 3 times lambda  in the x axis
        
                diff2 = abs(np.int32(self.imgRef[y,x,0])-np.int32(self.imgRef[y+1,x,0]))+abs(np.int32(self.imgRef[y,x,1])-np.int32(self.imgRef[y+1,x,1]))+abs(np.int32(self.imgRef[y,x,2])-np.int32(self.imgRef[y+1,x,2]))
    
                if diff2 < 3*5 :
                    self.smoothY[y,x]= self.smoothY[y,x] + 2 * self.smooth # we put 3 times lambda  in the y axis


    def optimize5images(self):
        """
        this is the main loop that implement the iterative dynamic programic using my generic dp module
        """
        result = 0
        # init 2 dp module
        # IMXSIZE-2*BORDERX : length of the line
        DynamicProgY = dp.DynProg(appSelf  = self,
                                  nblabel  = self.ConstParams.NBLABEL, 
                                  xsize    = self.IMXSIZE-2*self.ConstParams.BORDERX)
 
        # IMYSIZE-2*BORDERY : length of the line
        DynamicProgX = dp.DynProg(appSelf  = self,
                                  nblabel  = self.ConstParams.NBLABEL, 
                                  xsize    = self.IMYSIZE-2*self.ConstParams.BORDERY)

        self.first = 1 # for the first depth map computation do not used idp smoothing
        for j in xrange(self.nbIteration): 
            # we reverse opt orther so we have l->r  t->b  r->l and b ->t 
            if self.reverseOrderY == 1 :
                self.reverseOrderY = 0
            else:
                self.reverseOrderY = 1
            if self.reverseOrderX == 1 :
                self.reverseOrderX = 0
            else:
                self.reverseOrderX = 1

            self.Cx = -1
            change = 0

            # we init the 2D visibility 
            self.visibilityOutEpi[self.ConstParams.BORDERY-1,self.ConstParams.BORDERX:self.IMXSIZE-self.ConstParams.BORDERX] = self.ConstParams.BORDERY-1;
    
            for y in xrange(self.ConstParams.BORDERY,self.IMYSIZE-self.ConstParams.BORDERY):
                print("Info:mtplBLStereoApp:optimize5images:iter:%d:y=%d(range:%d..%d)" % (j,y,self.ConstParams.BORDERY,self.IMYSIZE-self.ConstParams.BORDERY))
                self.Cy = y # curent colomn use in cost function
                # do the optimisation of the problem define in <d>
                DynamicProgY.OptimizeDynamicProg(mode=0)
                # extract the argument of the resulting minimum solution
                res = DynamicProgY.ResultDynamicProg(size=self.IMXSIZE-2*self.ConstParams.BORDERX)
                # we copy back in current depth map the solution for this line
                if self.reverseOrderX :
                    for x in xrange(self.IMXSIZE-2*self.ConstParams.BORDERX): 
                        tmp =  res[self.IMXSIZE-2*self.ConstParams.BORDERX-x-1];
                        if tmp != self.imgDepthMap[self.Cy,x+self.ConstParams.BORDERX] :
                            change = change + 1
                            self.imgDepthMap[self.Cy,x+self.ConstParams.BORDERX] = tmp
                else:
                    for x in xrange(self.IMXSIZE-2* self.ConstParams.BORDERX): 
                        if res[x] != self.imgDepthMap[self.Cy,x+self.ConstParams.BORDERX] :
                            change = change + 1
                            self.imgDepthMap[self.Cy,x+self.ConstParams.BORDERX] = res[x]
      
                for x in xrange(self.ConstParams.BORDERX,self.IMXSIZE-self.ConstParams.BORDERX):
                    if self.visibilityOutEpi[self.Cy-1,x] <= self.imgDepthMap[self.Cy,x]+self.Cy :
                        self.visibilityOutEpi[self.Cy,x] = self.imgDepthMap[self.Cy,x]+self.Cy
                    else:
                        self.visibilityOutEpi[self.Cy,x] = self.visibilityOutEpi[self.Cy-1,x]
    
                res2 = DynamicProgY.ResultVisibilityDynamicProg(size=self.IMXSIZE-2*self.ConstParams.BORDERX)
                # we copy back in current depth map the solution for this line
                if self.reverseOrderX :
                    for x in xrange(self.IMXSIZE-2*self.ConstParams.BORDERX): 
                        self.visibilityInfo[self.Cy,x+self.ConstParams.BORDERX]= res2[self.IMXSIZE-2*self.ConstParams.BORDERX-x-1]
                else:
                    for x in xrange(self.IMXSIZE-2* self.ConstParams.BORDERX): 
                        self.visibilityInfo[self.Cy,x+self.ConstParams.BORDERX] = res2[x]

            result = result + 1
            if change == 0 : # if convergence acheived return should never append since prouf of non-convergence exist
                break
            self.first = 0 # at this point we are sur that a previous depth map exist so full smoothing of idp 
            self.Cy = -1 
            change = 0

            # we init the 2D visibility        
            self.visibilityOutEpi[self.ConstParams.BORDERY:self.IMYSIZE-self.ConstParams.BORDERY,self.ConstParams.BORDERX-1] = self.ConstParams.BORDERX-1;

            for x in xrange(self.ConstParams.BORDERX,self.IMXSIZE-self.ConstParams.BORDERX): # for each colom we do opt 
                print("Info:mtplBLStereoApp:optimize5images:iter:%d:x=%d(range:%d..%d)" % (j,x,self.ConstParams.BORDERX,self.IMXSIZE-self.ConstParams.BORDERX))
                self.Cx = x # curent line 
                # do the optimisation of the problem define in <d>
                DynamicProgX.OptimizeDynamicProg(mode=1)

                # extract the argument of the resulting minimum solution
                res = DynamicProgX.ResultDynamicProg(size=self.IMYSIZE-2*self.ConstParams.BORDERY)

                # we copy back in current depth map the solution for this line 
                if self.reverseOrderY :
                    for y in xrange(self.IMYSIZE-2*self.ConstParams.BORDERY): 
                        tmp  = res[self.IMYSIZE-2*self.ConstParams.BORDERY-y-1]
                        if self.imgDepthMap[y+self.ConstParams.BORDERY,self.Cx] !=  tmp :
                            imgDepthMap[y+self.ConstParams.BORDERY,self.Cx] = tmp
                            change = change + 1
                        else:
                            for y in xrange(self.IMYSIZE-2* self.ConstParams.BORDERY):    
                                if res[y] != self.imgDepthMap[y+self.ConstParams.BORDERY,self.Cx] :
                                    self.imgDepthMap[y+self.ConstParams.BORDERY,self.Cx] = res[y] 
                                    change = change + 1
      
                for y in xrange(self.ConstParams.BORDERY,self.IMYSIZE-self.ConstParams.BORDERY):
                    if self.visibilityOutEpi[y,self.Cx-1] <= self.imgDepthMap[y,self.Cx]+self.Cx :
                        self.visibilityOutEpi[y,self.Cx] = self.imgDepthMap[y,self.Cx]+self.Cx 
                    else:
                        self.visibilityOutEpi[y,self.Cx] = self.visibilityOutEpi[y,self.Cx-1];           

                # extract the argument of the resulting minimum solution
                res2 = DynamicProgY.ResultVisibilityDynamicProg(size=self.IMYSIZE-2*self.ConstParams.BORDERY)

                # we copy back in current depth map the solution for this line 
                if self.reverseOrderY :
                    for y in xrange(self.IMYSIZE-2*self.ConstParams.BORDERY):
                        self.visibilityInfo[y+self.ConstParams.BORDERY,self.Cx] = res2[self.IMYSIZE-2*self.ConstParams.BORDERY-y-1]
                else:
                    for y in xrange(self.IMYSIZE-2*self.ConstParams.BORDERY):
                        self.visibilityInfo[y+self.ConstParams.BORDERY,self.Cx] = res2[y]
           
            result = result + 1
            if change == 0 :
                break

        return result


    def postProcessing(self):
        """
        we remove noise from depth map
        this is a simple filter
        """
        for y in xrange(self.ConstParams.BORDERY+1,self.IMYSIZE-self.ConstParams.BORDERY-1): 
            for x in xrange(self.ConstParams.BORDERX+1,self.IMXSIZE-self.ConstParams.BORDERX-1):
                if self.imgDepthMap[y,x-1] == self.imgDepthMap[y,x+1] and self.imgDepthMap[y,x-1] != self.imgDepthMap[y,x] :
                    self.imgDepthMap[y,x] = self.imgDepthMap[y,x-1]
                elif self.imgDepthMap[y-1,x] == self.imgDepthMap[y+1,x] and self.imgDepthMap[y-1,x] != self.imgDepthMap[y,x]:
                    self.imgDepthMap[y,x] = self.imgDepthMap[y-1,x]


    def SmoothCapacityX(self, x1, l1, x2, d, occ, oldmask):
#        print("Info:DynProg:SmoothCapacityX...\n")
        res1 = sys.maxint # left DSI
        res2 = sys.maxint # right DSI

        scost = 0 # smoothing cost oder direction   

        # this is the initialisation
        if x1 < 0 :
            if self.reverseOrderX == 1 :
                return self.dsiLeft[self.Cy,self.IMXSIZE-self.ConstParams.BORDERX-x2-1,d], 0 # left
            else:
                return self.dsiRight[self.Cy,x2+self.ConstParams.BORDERX,d], 0 # rigth

        if self.reverseOrderX == 1 :
            x1 = self.IMXSIZE-self.ConstParams.BORDERX-x1-1
            x2 = self.IMXSIZE-self.ConstParams.BORDERX-x2-1
        else:
            x1 = x1 + self.ConstParams.BORDERX
            x2 = x2 + self.ConstParams.BORDERX

        # we chnage camera
        if self.reverseOrderX :
            if occ < d : # >= 
                # we also check for the top DSI (4)
                if self.visibilityOutEpi[self.Cy-1,x2] < d+self.Cy and self.first != 1 :
                    res1 = (self.dsiRight[self.Cy,x2,d]+self.dsiBot[x2,self.Cy,d])/2 # left and top DSI 
                else:
                    res1 = self.dsiRight[self.Cy,x2,d]
            if occ >= d :
                if  self.visibilityOutEpi[self.Cy-1,x2] < d+self.Cy and self.first != 1 :
                    res2 = self.dsiBot[x2,self.Cy,d]
                elif self.dsiTop[x2,self.Cy,d] < self.dsiLeft[self.Cy,x2,d] :
                    res2 = self.dsiTop[x2,self.Cy,d]
                else:
                    res2 = self.dsiLeft[self.Cy,x2,d] # right DSI
        else: # not reverse order 
            if occ < d :
                if self.visibilityOutEpi[self.Cy-1,x2] < d+self.Cy and self.first != 1 :  
                    res1 = (self.dsiLeft[self.Cy,x2,d]+self.dsiBot[x2,self.Cy,d])/2 # left and top DSI 
                else:
                    res1 = self.dsiLeft[self.Cy,x2,d] # left DSI 

            if occ >= d :
                if self.visibilityOutEpi[self.Cy-1,x2] < d+self.Cy and self.first != 1 :
                    res2 = self.dsiBot[x2,self.Cy,d]
                elif self.dsiTop[x2,self.Cy,d] < self.dsiRight[self.Cy,x2,d] :
                    res2 = self.dsiTop[x2,self.Cy,d]
                else:
                    res2 = self.dsiRight[self.Cy,x2,d] # right DSI

        if res1 == sys.maxint and res2 == sys.maxint :
            return 8000, 0

        scost = 0
        if self.imgDepthMap[self.Cy-1,x2] != d :
            scost = self.smoothY[self.Cy-1,x2]
        if self.imgDepthMap[self.Cy+1,x2] != d :
            scost = scost + self.smoothY[self.Cy,x2]

        if self.first == 1 :
            scost = 0

        if x1 > x2 :
            x1 = x2

        if l1 != d :
            scost = scost + self.smoothX[self.Cy,x1]
 
        if res1 == sys.maxint :
            mask = 0
            if mask != oldmask :
                res2 = res2 + self.ConstParams.GAMA
            return (np.int32(res2)+np.int32(scost))*8, 0

        mask = 1
        if mask != oldmask :
            res1 = res1 + self.ConstParams.GAMA

        return (np.int32(res1)+np.int32(scost))*8, mask    


    def SmoothCapacityY(self, y1, l1, y2, d, occ, oldmask):
        """
        this is the function used for likelihood when working on colomn
        this fonction compute the likellehood term base of visibility
        this function take about 30% of computing power
        """
#        print("Info:DynProg:SmoothCapacityY...\n")
        res1 = sys.maxint # this is a flag to indicate that
                          # cost function not avalable because
                          # of occlusion or comparaison
                          # impossible du to pixel outside of
                          # image  
        res2 = sys.maxint
        scost = 0 # smoothing cost horizontal (idp) direction  

        # this is the initialisation
        if y1 < 0 :
            if self.reverseOrderY == 1 :
                return self.dsiLeft[self.IMYSIZE-self.ConstParams.BORDERY-y2-1,self.Cx,d], 0 # left */
            else:
                return self.dsiRight[y2+self.ConstParams.BORDERY,self.Cx,d], 0 # rigth */

        if self.reverseOrderY == 1 : # are we working top toward bottom or the orther way
            y1 = self.IMYSIZE-self.ConstParams.BORDERY-y1-1
            y2 = self.IMYSIZE-self.ConstParams.BORDERY-y2-1
        else:
            y1 = y1 + self.ConstParams.BORDERY
            y2 = y2 + self.ConstParams.BORDERY

        # we change camera 
        if self.reverseOrderY :
            if occ < d : # visibility test to select mask 
                # we also check for the top DSI (4)      
                if self.visibilityOutEpi[y2,self.Cx-1] < d+self.Cx and self.first != 1 :  
                    res1 = (self.dsiLeft[y2,self.Cx,d] + self.dsiTop[self.Cx,y2,d])/2  # left and botom DSI
                else:
                    res1 = self.dsiTop[self.Cx,y2,d] # bottom DSI

            # camera along the axis  we may have to gest
            if occ >= d :
    
                if self.visibilityOutEpi[y2,self.Cx-1] < d+self.Cx and self.first != 1 :
                    res2 = self.dsiLeft[y2,self.Cx,d] # top DSI
                elif self.dsiRight[y2,self.Cx,d] < self.dsiBot[self.Cx,y2,d] :
                    res2 = self.dsiRight[y2,self.Cx,d] # nakamura
                else:
                    res2 = self.dsiBot[self.Cx,y2,d] # nakamura  

        else: # no reverse order
            if occ < d : # since opt order is change visibility check is inverted
                # we also check for the top DSI (4)      
                if self.visibilityOutEpi[y2,self.Cx-1] < d+self.Cx and self.first != 1 :
                    res1 = (self.dsiLeft[y2,self.Cx,d] + self.dsiBot[self.Cx,y2,d])/2 # left and botom DSI
                else:
                    res1 = self.dsiBot[self.Cx,y2,d] # bottom DSI
            # other camera
            if occ >= d :
                if self.visibilityOutEpi[y2,self.Cx-1] < d+self.Cx and self.first !=1 :
                    res2 = self.dsiLeft[y2,self.Cx,d] # top DSI
                elif self.dsiRight[y2,self.Cx,d] < self.dsiTop[self.Cx,y2,d] :
                    res2 = self.dsiRight[y2,self.Cx,d];
                else:
                    res2 = self.dsiTop[self.Cx,y2,d] # top DSI

        if res1 == sys.maxint and res2 == sys.maxint : # impossible to select mask other than (0,0)
            return 8000, 0
        scost = 0
        if self.imgDepthMap[y2,self.Cx-1] != d :
            scost = self.smoothX[y2,self.Cx-1]
        if self.imgDepthMap[y2,self.Cx+1] != d :
            scost = scost + self.smoothX[y2,self.Cx]
        if self.first == 1 : # first computation we use ordinary dp since no previous partial solution avalable */
            scost = 0
        if y1 > y2 :
            y1 = y2
        # smoothing in this line for pot
        if l1 != d :
            scost = scost + self.smoothY[y1,self.Cx]
        # we select the proper visibility mask and return value to optimisation module
        if res1 == sys.maxint :
            mask = 0
            if mask != oldmask :
                res2 = res2 + self.ConstParams.GAMA
            return (np.int32(res2)+np.int32(scost))*8, 0

        mask = 1
        if mask != oldmask :
            res1 = res1 + self.ConstParams.GAMA
        return (np.int32(res1)+np.int32(scost))*8, mask
                    


def nothing(*arg):
    pass

def constrast_enhance():
    cv2.namedWindow('new_depth0')
    cv2.createTrackbar('phi', 'new_depth0', 10, 10, nothing)
    cv2.createTrackbar('theta', 'new_depth0', 10, 10, nothing)
    cv2.createTrackbar('coef', 'new_depth0', 3, 10, nothing)
    maxIntensity = 255.0 # depends on dtype of image data
    # Image data
    image = cv2.imread('../data/depth.png',0) # load as 1-channel 8bit grayscale
    cv2.imshow('orig_depth',image)
#    x = np.arange(maxIntensity) 
    
    while True:
        # Parameters for manipulating image data
        phi = cv2.getTrackbarPos('phi', 'new_depth0')
        theta = cv2.getTrackbarPos('theta', 'new_depth0')
        coef = cv2.getTrackbarPos('coef', 'new_depth0')
        
        phi = np.float(phi / 10.)
        if phi == 0 :
            phi = 0.001
            
        theta = np.float(theta / 10.)
        if theta == 0 :
            theta = 0.001
            
        coef = np.float(coef / 10.)
        if coef == 0 :
            coef = 0.001

        # Increase intensity such that
        # dark pixels become much brighter, 
        # bright pixels become slightly bright
        newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**coef
        newImage0 = np.array(newImage0,dtype='uint8')

        cv2.imshow('new_depth1',newImage0)
        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:
            break
        
    cv2.imwrite('../data/new_depth0.jpg',newImage0)


if __name__ == '__main__':
    print(__doc__)

    try:
        param = sys.argv[1]
    except IndexError:
        param = ""

    mtplBLStereoApp = App()

    if "--constparm" == param:
        mtplBLStereoApp.showParameters()
    elif "--help" == param:
        print("\t--constparm\n\t\tprint constant parameters")
        print("\t--help\n\t\tprint this help")
    else:
#        threadn = cv2.getNumberOfCPUs()
#        pool = ThreadPool(processes = threadn)
#        pending = deque()
        
#        if len(pending) < threadn:
#            task = pool.apply_async(mtplBLStereoApp.run, ())
#            pending.append(task)
        
#        while len(pending) > 0 and pending[0].ready():
#            res = pending.popleft().get()

        mtplBLStereoApp.run()
        constrast_enhance()

        cv2.waitKey()
        cv2.destroyAllWindows()
