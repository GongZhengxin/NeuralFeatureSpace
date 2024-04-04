import numpy as np
import pandas as pd
import nibabel as nib
import os
import glob
from os.path import join as pjoin 
from scipy.stats import zscore
from utils import save2cifti, save_ciftifile

csv_dir = '/nfs/z1/userhome/zzl-xsk/NSD/nsddata_stimuli/stimuli/nsd/nsd_stim_info_merged.csv'
cifti_dir = '/nfs/z1/userhome/zzl-xsk/NSD/derivatives/ciftify'
subjs = ['subj01','subj02','subj03','subj04','subj05','subj06','subj07','subj08'] #
subj_index = {'subj01':'subject1' , 'subj02':'subject2', 'subj03': 'subject3', 'subj04': 'subject4',
                    'subj05':'subject5', 'subj06':'subject6','subj07':'subject7', 'subj08' : 'subject8'}
n_sessions = {'subj01':40, 'subj02':40, 'subj03':32, 'subj04':30, 'subj05':40, 'subj06':32, 'subj07':40, 'subj08':30 }
betas_path = '/nfs/z1/userhome/zzl-xsk/NSD/derivatives/ciftify/subj01/MNINonLinear/Results'
df = pd.read_csv(csv_dir)

for subj in subjs:
    print(f'{subj} start!')
    # 确定当前被试有多少个trial
    subj_sessions = n_sessions[subj]
    n_trials =subj_sessions * 750
    
    # 找到当前被试实验时，各图片第一次呈现所在的trial
    column_index = subj_index[subj]
    rep0 = df[f'{column_index}_rep0'].values # 此时rep0是按照73000的nsdid排序的
    
    # 按照被试实验时trial先后顺序，重新整理图片顺序
    # 此时变成被试内部 10000 张图片排序，
    # 但是保存出来的是 图片 的 nsdid 
    image_seq = []
    for itrial in range(1, n_trials + 1):
        temp0 = np.where(rep0 == itrial)[0]
        if len(temp0) > 0:
            image_seq.append(temp0[0])

    print(len(image_seq))
    print(image_seq[0])
    # 拿出 当前被试 rep0 rep1 rep2 数据（出现在第几个trial）
    df_subj = df.loc[(df[f'{column_index}'] != 0), [f'{column_index}_rep0', f'{column_index}_rep1', f'{column_index}_rep2']]
    # 按照被试内部 10000 张图片排序
    df_subj_reindex = df_subj.reindex(image_seq)
    print(df_subj_reindex)
    
    # 初始化图片激活
    beta_data = np.nan*np.zeros((*df_subj_reindex.shape, 91282))
    # 循环每个 run 
    for session in range(1, n_sessions[subj] + 1):
        session_beta = nib.load(os.path.join(cifti_dir, subj, 'MNINonLinear', 'Results', f'betas_session{session:02d}',f'betas_session{session:02d}_Atlas.dtseries.nii' )).get_fdata()
        session_beta = (session_beta - np.nanmean(session_beta)) / np.nanstd(session_beta)
        mask = df_subj_reindex.isin(np.arange((session - 1)*750 + 1, session*750 + 1))
        beta_pos = np.where(mask)
        beta_data[beta_pos] = session_beta
        print(f'Session{session:02d} prepared! Has NaN : {np.isnan(session_beta[:, 0:59412]).sum()}')

    beta_mean = np.nanmean(beta_data, axis=1)
    print(f'beta_mean : {beta_mean.shape}')

    save_dir = pjoin(cifti_dir, subj, f'MNINonLinear/Results/average_betas')
    os.makedirs(save_dir, exist_ok=True)
    save_file_name = f'{subj}_average-session-zscored-betas'
    save_ciftifile(beta_mean, pjoin(save_dir, save_file_name))
    print(f'saved! {pjoin(save_dir, save_file_name)}')