"""Generates a selection of glass files"""

from meshoid import particle_glass
import numpy as np
import h5py

for N in 16,32,64,128,256:
    x = np.float32(particle_glass(N**3,tol=1e-4))%1.
    with h5py.File(f"glass_{N}.hdf5","w") as F:
        F.create_dataset("Coordinates",data=x)
