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

import math
import numpy as np

class Reductor(object):
    '''Base object of a ``Reductor``, that takes input data and reduces it.
    
    Implementing objects should define `outSize`, the number of elements after
    reduction, and a ``filters`` :class:`numpy.ndarray` of size ``(inSize,outSize)``, where
    each row is a basis vector in Fourier space.
    
    :param inSize: Input size of vectors.
    :type inSize: :class:`int`
    '''
    def __init__(self,inSize):
        self.size = inSize
        self.inSize = self.size
    def getFilter(self,weights):
        '''Returns actual FBP filters, given the resulting weights of a trained neural network.'''
        return np.dot(self.filters,weights)
        
class LogSymReductor(Reductor):
    '''An implementation of a ``Reductor`` with exponentially growing bin widths, and symmetric bins.
    
    :param nLinear: Number of bins of width 1 before starting exponential growth.'
    :type nLinear: :class:`int`
    '''
    def __init__(self,size,fpSize,nLinear=2):
        Reductor.__init__(self,size)
        self.name="LogSym"
        cW=0
        nW=0
        width=1
        nL = nLinear
        while cW<(fpSize-1)/2:
            if nL>0:
                nL-=1
                cW += 1
            else:
                cW += width
                width*=2
            nW+=1
        
        self.filters = np.zeros((size,nW))
        x = np.linspace(0,2*np.pi,size,False)
        cW=0
        nW=0
        width=1
        nL = nLinear
        while cW<(fpSize-1)/2:
            if nL>0:
                nL-=1
                eW = cW+1
            else:
                eW = cW+width
                width*=2
            self.filters[:,nW] += np.cos(np.outer(np.arange(cW,eW),x)).sum(0)
            cW=eW
            nW+=1
        self.outSize = nW
