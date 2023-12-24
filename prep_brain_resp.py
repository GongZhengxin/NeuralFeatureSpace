import glob
import os
import time

import nibabel as nib
import numpy as np

from scipy.stats import zscore

t_begin = time.time()
# ===============================================================
# settings

# cifti dir
cifti_dir = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD/derivatives/ciftify'

session_info = {'imagenet': ['imagenet01','imagenet02','imagenet03','imagenet04'],
                'coco' : ['coco']
}
session_name = 'coco'
# experiment info
subs = ['sub-03', 'sub-04', 'sub-08']#'sub-01', 'sub-02', 'sub-05', 'sub-06', 'sub-07', 'sub-09'
for sub in subs: 
    sessions = session_info[session_name]

    # output paths
    resp_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodretinotopy/mfm_locwise_fullpipeline/prep/brain_response'

    # collect run folder
    folders = []
    for session in sessions:
        for run_folder in glob.glob(os.path.join(cifti_dir, sub, "results/", f"*{session}*")):
            folders.append(run_folder)
    folders = sorted(folders)

    # collect resp files
    resp_files = []
    for folder in folders:
        for dscalar_file in glob.glob(os.path.join(folder, "*dscalar.nii")): 
            resp_files.append(dscalar_file) # 顺序为 1 10 2 3 ...

    # collect resp
    resp = []
    for resp_file in resp_files:
        resp.append(nib.load(resp_file).get_fdata().astype(np.float32))

    brain_resp = np.concatenate(resp)

    np.save(os.path.join(resp_path, f'{sub}_{session_name}_beta.npy'), brain_resp)

    print(f'at prep brian resp {sub}_{session_name}_beta.npy : {time.time() - t_begin} s')

