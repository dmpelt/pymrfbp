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

import astra
import numpy as np

# Register MR-FBP plugin with ASTRA
import mrfbp
astra.plugin.register(mrfbp.plugin)

# Create ASTRA geometries
vol_geom = astra.create_vol_geom(256,256)
proj_geom = astra.create_proj_geom('parallel',1.0,256,np.linspace(0,np.pi,32,False))

# Create the ASTRA projector (change 'linear' to 'cuda' to use GPU)
pid = astra.create_projector('linear', proj_geom, vol_geom)
p = astra.OpTomo(pid)

# Load the phantom from disk
testPhantom = np.load('phantom.npy')

# Calculate the forward projection of the phantom
testSino = (p*testPhantom).reshape(p.sshape)

# Add some noise to the sinogram
testSino = astra.add_noise_to_sino(testSino,10**4)

# Reconstruct the image using MR-FBP, FBP, and SIRT.
mrRec = p.reconstruct('MR-FBP',testSino)
if astra.projector.is_cuda(pid):
    fbpRec = p.reconstruct('FBP_CUDA',testSino)
    sirtRec = p.reconstruct('SIRT_CUDA',testSino,200)
else:
    fbpRec = p.reconstruct('FBP',testSino)
    sirtRec = p.reconstruct('SIRT',testSino,200)

# Show the different reconstructions on screen
import pylab
pylab.gray()
pylab.subplot(221)
pylab.axis('off')
pylab.title('Phantom')
pylab.imshow(testPhantom,vmin=0,vmax=1)
pylab.subplot(222)
pylab.axis('off')
pylab.title('MR-FBP')
pylab.imshow(mrRec,vmin=0,vmax=1)
pylab.subplot(223)
pylab.axis('off')
pylab.title('FBP')
pylab.imshow(fbpRec,vmin=0,vmax=1)
pylab.subplot(224)
pylab.axis('off')
pylab.title('SIRT-200')
pylab.imshow(sirtRec,vmin=0,vmax=1)
pylab.tight_layout()
pylab.show()

