#-----------------------------------------------------------------------
#Copyright 2014 Daniel M. Pelt
#
#Contact: D.M.Pelt@cwi.nl
#Website: http://www.dmpelt.com
#
#
#This file is part of the PyMR-FBP, a Python implementation of the
#MR-FBP tomographic reconstruction method.
#
#PyMR-FBP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#PyMR-FBP is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with PyMR-FBP. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------

import Reductors
import numpy as np
import numpy.linalg as na
import scipy.ndimage.filters as snf

class Reconstructor:
    '''The standard MR-FBP reconstructor.
    
    :param projector: Projector object that implements reconstructWithFilter
    :param reductor: Reductor to use, if ``None``, ``Reductors.LogSymReductor`` is used
    :param projectorFP: Optional different projector to use for final forward projection.
    '''
    def __init__(self,projector,reductor=None,projectorFP=None):
        self.projector = projector
        if projectorFP==None:
            self.projectorFP = self.projector
        else:
            self.projectorFP = projectorFP
        if reductor==None:
            reductor = Reductors.LogSymReductor(self.projector.filterSize,self.projectorFP.filterSize)
        self.reductor = reductor
        self.__setOutCircle()
    
    def __setOutCircle(self):
        '''Creates a :class:`numpy.ndarray` mask of a circle'''
        xx, yy = np.mgrid[:self.projector.recSize, :self.projector.recSize]
        mid = (self.projector.recSize-1.)/2.
        circle = (xx - mid) ** 2 + (yy - mid) ** 2
        bnd = self.projector.recSize**2/4.
        self.outCircle=circle>bnd
    
    def _calca(self,sinogram):
        '''Returns the MR-FBP system matrix A
        
        :param sinogram: Sinogram to calculate A with
        :type sinogram: :class:`numpy.ndarray`
        '''
        a = np.zeros((self.projectorFP.nDet*self.projectorFP.nProj,int(self.reductor.outSize)),dtype=np.float32)
        for i in xrange(self.reductor.outSize):
            img = self.projector.reconstructWithFilter(sinogram,self.reductor.filters[:,i])
            img[self.outCircle]=0
            a[:,i] = self.projectorFP.forwProject(img).flatten()
        return a        
    
    def reconstruct(self,sinogram,cSinogram=None):
        '''Returns MR-FBP reconstruction
        
        :param sinogram: Sinogram to reconstruct
        :type sinogram: :class:`numpy.ndarray`
        :param cSinogram: Optional other sinogram to use as right-hand side.
        :type cSinogram: :class:`numpy.ndarray`
        '''
        if cSinogram==None: cSinogram=sinogram
        a = self._calca(sinogram)
        out = na.lstsq(a,cSinogram.flatten())
        f = self.reductor.getFilter(out[0])
        self.f = f
        return self.projector.reconstructWithFilter(sinogram,f)

class ReconstructorGradient:
    '''The MR-FBP_GM reconstructor.
    
    :param projector: Projector object that implements reconstructWithFilter
    :param reductor: Reductor to use, if ``None``, ``Reductors.LogSymReductor`` is used
    :param projectorFP: Optional different projector to use for final forward projection.
    '''
    def __init__(self,projector,reductor=None,projectorFP=None):
        self.projector = projector
        if projectorFP==None:
            self.projectorFP = self.projector
        else:
            self.projectorFP = projectorFP
        if reductor==None:
            reductor = Reductors.LogSymReductor(self.projector.filterSize,self.projectorFP.filterSize)
        self.reductor = reductor
        self.__setOutCircle()
    
    def __setOutCircle(self):
        '''Creates a :class:`numpy.ndarray` mask of a circle'''
        xx, yy = np.mgrid[:self.projector.recSize, :self.projector.recSize]
        mid = (self.projector.recSize-1.)/2.
        circle = (xx - mid) ** 2 + (yy - mid) ** 2
        bnd = self.projector.recSize**2/4.
        self.outCircle=circle>bnd
    
    def _calca(self,sinogram,lam):
        '''Returns the MR-FBP_GM system matrix A
        
        :param sinogram: Sinogram to calculate A with
        :type sinogram: :class:`numpy.ndarray`
        :param lam: Lambda to relative weight to give to gradient error
        :type lam: :class:`float`
        '''
        nfp = self.projectorFP.nDet*self.projectorFP.nProj
        ngr = self.projector.nDet*self.projector.nDet
        a = np.zeros((nfp+2*ngr,int(self.reductor.outSize)),dtype=np.float32)
        for i in xrange(self.reductor.outSize):
            img = self.projector.reconstructWithFilter(sinogram,self.reductor.filters[:,i])
            xs = snf.sobel(img,0)
            ys = snf.sobel(img,1)
            img[self.outCircle]=0
            xs[self.outCircle]=0
            ys[self.outCircle]=0
            a[0:nfp,i] = self.projectorFP.forwProject(img).flatten()
            a[nfp:nfp+ngr,i] = lam*xs.flatten()
            a[nfp+ngr:nfp+2*ngr,i] = lam*ys.flatten()
        return a        
    
    def reconstruct(self,sinogram,lam,cSinogram=None):
        '''Returns MR-FBP_GM reconstruction
        
        :param sinogram: Sinogram to reconstruct
        :type sinogram: :class:`numpy.ndarray`
        :param lam: Lambda to relative weight to give to gradient error
        :type lam: :class:`float`
        :param cSinogram: Optional other sinogram to use as right-hand side.
        :type cSinogram: :class:`numpy.ndarray`
        '''
        
        if cSinogram==None: cSinogram=sinogram
        a = self._calca(sinogram,lam)
        out = na.lstsq(a,np.hstack((cSinogram.flatten(),np.zeros(2*self.projector.nDet*self.projector.nDet))))
        f = self.reductor.getFilter(out[0])
        self.f = f
        return self.projector.reconstructWithFilter(sinogram,f)

