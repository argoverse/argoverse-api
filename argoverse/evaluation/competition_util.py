"""Utilities for submitting to Argoverse tracking and forecasting competitions"""

import os
from typing import Dict
import h5py
import tempfile
import shutil
import zipfile
import numpy as np

NUM_TEST_SEQUENCE = 79391
def generate_forecasting_h5(data: Dict[int, np.ndarray], output_path: str, filename: str ='argoverse_forecasting_baseline'):
    """
    Helper function to generate the result h5 file for argoverse forecasting challenge
    
    Args:
        data: a dictionary of trajectory, with the key being the sequence ID. For each sequence, the trajectory should be stored in a (9,30,2) np.ndarray
        output_path: path to the output directory to store the output h5 file
        filename: to be used as the name of the file
        
    Returns:
        
    """
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf = h5py.File(os.path.join(output_path,filename+'.h5'), 'w')
    
    d_all = []
    counter = 0
    for key,value in data.items():
        print('\r'+str(counter+1)+'/'+str(len(data)),end="")
        assert type(key) == int, f'ERROR: The dict key should be of type int representing the sequence ID, currently getting {type(key)}'
        assert value.shape == (9,30,2), f'ERROR: the data should be of shape (9,30,2), currently getting {value.shape}'
        
        value = value.reshape(270,2)
        
        d = np.array([[key,np.float32(x),np.float32(y)] for x,y in value])
        
        if len(d_all) == 0:
            d_all = d
        else:
            d_all = np.concatenate([d_all,d],0)
        counter += 1
    hf.create_dataset('argoverse_forecasting', data=d_all, compression="gzip", compression_opts=9)
    hf.close()
    
def generate_tracking_zip(input_path: str, output_path: str, filename: str ='argoverse_tracking'):
    """
    Helper function to generate the result zip file for argoverse tracking challenge
    
    Args:
        input path: path to the input directory which contain per_sweep_annotations_amodal/
        output_path: path to the output directory to store the output zip file
        filename: to be used as the name of the file
        
    Returns:
        
    """
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dirpath = tempfile.mkdtemp()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for log_id in os.listdir(input_path):
        if log_id.startswith('.'):
            continue
        shutil.copytree(os.path.join(input_path,log_id,'per_sweep_annotations_amodal'),os.path.join(dirpath,log_id,'per_sweep_annotations_amodal'))

    shutil.make_archive(os.path.join(output_path,'argoverse_tracking'), 'zip',dirpath)
    shutil.rmtree(dirpath)