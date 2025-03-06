import pytest
import numpy as np
import json
import subprocess
from rdkit import Chem
from omtra.data.pharmacophores import get_pharmacophores
from omtra.constants import ph_type_to_idx, ph_idx_to_type

@pytest.mark.parametrize("sdf_path, pdb_path", [
    ("test_lig.sdf", None),
    ("test_lig.sdf", "test_rec.pdb")
])
def test_pharmacophores(tmp_path, sdf_path, pdb_path):
    """Test that get_pharmacophores produces the same output as Pharmit."""
    
    ph_out_file = tmp_path / "pharmit_output.json"
    lig = Chem.SDMolSupplier(sdf_path)[0]
    
    if pdb_path:
        rec = Chem.MolFromPDBFile(pdb_path)
        X, P, _, I = get_pharmacophores(ph_type_to_idx, lig, rec)
        our_features = [(ph_idx_to_type[idx], P[i], I[i]) for i, idx in enumerate(X) if I[i]]
        cmd = f'pharmit pharma -receptor {pdb_path} -in {sdf_path} -out {ph_out_file}'
        
    else:
        X, P, _, _ = get_pharmacophores(ph_type_to_idx, lig)
        our_features = [(ph_idx_to_type[idx], P[i]) for i, idx in enumerate(X)]
        cmd = f'pharmit pharma -in {sdf_path} -out {ph_out_file}'
        
    subprocess.check_call(cmd, shell=True)
    decoder = json.JSONDecoder()
    ph = decoder.raw_decode(open(ph_out_file).read())[0]
    pharmit_features = [
        (point['name'], np.array([point['x'], point['y'], point['z']])) for point in ph['points'] if point['enabled']
    ]

    unmatched_features = [
        {"name": name, "coordinate": coord.tolist()}
        for name, coord in pharmit_features
        if not any(name == expected_name and np.allclose(coord, expected_coord, atol=0.01)
                for expected_name, expected_coord, *_ in our_features)
    ]

    assert not unmatched_features