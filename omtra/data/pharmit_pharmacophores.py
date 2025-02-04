import numpy as np
import json
import subprocess
import os
from typing import Tuple


def get_lig_only_pharmacophore(sdf_path, tmp_path, ph_type_to_idx) -> Tuple[np.ndarray, np.ndarray]:
    #get pharmacophores from pharmit
    phfile = tmp_path / 'pharmacophore.json'
    cmd = f'./pharmit pharma -in {sdf_path} -out {phfile}'  # Using executable pharmit file for testing (remove ./)
    subprocess.check_call(cmd,shell=True)

    #some files have another json object in them - only take first
    #in actuality, it is a bug with how pharmit/openbabel is dealing
    #with gzipped sdf files that causes only one molecule to be read
    decoder = json.JSONDecoder()
    ph = decoder.raw_decode(open(phfile).read())[0]

    if ph['points']:
        feature_coords = np.array([(p['x'],p['y'],p['z']) for p in ph['points'] if p['enabled']])
        feature_kind = np.array([ph_type_to_idx[p['name']] for p in ph['points'] if p['enabled']])
    else:
        return None, None
    return feature_coords, feature_kind

