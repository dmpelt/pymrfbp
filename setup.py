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

from distutils.core import setup

setup(name='PyMR-FBP',
      version='1.0',
      description='Python implementation of MR-FBP Algorithm',
      author='D.M. Pelt',
      author_email='D.M.Pelt@cwi.nl',
      url='http://dmpelt.github.io/pymrfbp/',
      packages=['mrfbp'],
     )
