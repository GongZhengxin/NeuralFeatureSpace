import os, sys, six # six provides python 2/3 compatibility

# Import our numerical/scientific libraries, scipy and numpy:
import numpy as np
import scipy as sp
import nibabel as nib
from os.path import join as pjoin
import h5py

import scipy.io as sio
# The neuropythy library is a swiss-army-knife for handling MRI data, especially
# anatomical/structural data such as that produced by FreeSurfer or the HCP.
# https://github.com/noahbenson/neuropythy
import neuropythy as ny

# Import graphics libraries:
# Matplotlib/Pyplot is our 2D graphing library:
import matplotlib as mpl
import matplotlib.pyplot as plt

sub = ny.hcp_subject('/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/bold/derivatives/ciftify/sub-01')
cortex = sub.hemis['lh_LR32k_FS']
sphere = cortex.registrations['fs_LR']
# v1_weight = sphere.prop('V1_weight')
coords = sphere.coordinates
res = sub.hemis['lh_LR32k_MSMSulc'].prop('curvature')