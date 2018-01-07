#!/usr/bin/env python
"""
A barebones script for generating density slices with meshoid

Usage:
    ./simple_density_slice.py <files> ... [options]

Options:
    -h --help           Show this screen.
    --rmax=<kpc>        Radius of plot window [default: 1.0]
    --plane=<x,y,z>     Slice/projection plane [default: z]
    --c=<cx,cy,cz>      Coordinates of plot window center [default: 0.0,0.0,0.0]
    --ptype=<N>         Type of particle to plot [default: 0]
    --cmap=<name>       Name of colormap to use [default: viridis]
    --limits=<min,max>  Dynamic range of the density map
    --res=<N>           Resolution [default: 400]
"""


from docopt import docopt
import numpy as np
import re
from meshoid import meshoid, FromSnapshot
from matplotlib import pyplot as plt

arguments = docopt(__doc__)
filenames = arguments["<files>"]
if not filenames:
    filenames=glob.glob('snapshot_*.hdf5')

def MakePlot(f):
    rmax = float(arguments["--rmax"])
    plane = arguments["--plane"]
    center = np.array([float(c) for c in re.split(',', arguments["--c"])])
    limits = arguments["--limits"]
    if limits:
        limits = np.array([np.log10(float(c)) for c in re.split(',', limits)])
    ptype = int(float(arguments["--ptype"]))
    res = int(float(arguments["--res"]))
    cmap = arguments["--cmap"]
    
    M = FromSnapshot(f, ptype)
    rho = M.Density()
    sigma = M.Slice(rho, size=2*rmax, center=center, plane=plane, res=res)
    print(sigma.shape)
    if not limits:
        limits = np.log10(np.percentile(sigma.flatten(),[0.1,99.9]))
        
    plt.imsave(f.replace(".hdf5",".png"), np.log10(sigma), vmin=limits[0], vmax=limits[1], cmap=cmap)

for f in filenames: MakePlot(f)