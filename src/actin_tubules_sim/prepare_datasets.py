import mrcfile
from tifffile import imsave
import numpy as np
import os
import shutil


def create_folders(root_dir):
    gt_dir = os.path.join(root_dir, 'Train', 'SR', 'GT')
    raw_dir = os.path.join(root_dir, 'Train', 'SR', 'Raw')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            gt_files = [f for f in os.listdir(subdir_path) if f.endswith('SIM_gt.mrc')]
            for idx, gt_file in enumerate(gt_files):
                shutil.copy(os.path.join(subdir_path, gt_file), gt_dir)
                new_gt_name = os.path.join(gt_dir, f'{idx + 1}.mrc')
                os.rename(os.path.join(gt_dir, gt_file), new_gt_name)
                
                raw_gt_file = os.path.join(subdir_path, f'RawSIMData_gt.mrc')
                if os.path.exists(raw_gt_file):
                    raw_gt_new_name = os.path.join(raw_dir, f'{idx + 1}.mrc')
                    shutil.copy(raw_gt_file, raw_gt_new_name)
                else:
                    print(f"Warning: RawSIMData_gt file does not exist for {gt_file}. Skipping...")
            

def convert_mrc_to_tiff(input_path, output_path):
    with mrcfile.open(input_path) as mrc:
        data = mrc.data.squeeze() 
        imsave(output_path, data.astype(np.float32))