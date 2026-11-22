"""Source linkage retained alongside new Pharmit tensors; no database or Zarr I/O."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from rdkit import Chem, rdBase


def source_record(source_id, smiles, sdf, record_index, mol, xace, atom_map):
    if not isinstance(smiles, str) or not smiles.strip() or not str(sdf).strip():
        raise ValueError('Original source SMILES and SDF locator must be retained')
    reference = Chem.RemoveHs(Chem.Mol(mol))
    if reference.GetNumAtoms() != len(xace.x):
        raise ValueError('Tensorization changed source heavy-atom count')
    if ([a.GetSymbol() for a in reference.GetAtoms()] != [atom_map[int(i)] for i in xace.a]
            or [a.GetFormalCharge() for a in reference.GetAtoms()] != np.asarray(xace.c).tolist()
            or not np.allclose(reference.GetConformer().GetPositions(), xace.x, atol=1e-6, rtol=0)):
        raise ValueError('Tensorization changed source heavy-atom correspondence')
    return {'version': 1, 'source_id': int(source_id), 'source_smiles': str(smiles),
            'source_sdf': str(sdf), 'source_record_index': int(record_index),
            'source_heavy_conformer_rdkit_molblock_sha256': hashlib.sha256(Chem.MolToMolBlock(reference).encode()).hexdigest(),
            'rdkit_version': rdBase.rdkitVersion,
            'graph_to_source_heavy_atom': list(range(reference.GetNumAtoms())),
            'identity_status': 'original_database_record_retained_not_stereo_certified'}


def encode_records(records, node_lookup):
    lookup = np.asarray(node_lookup)
    if (lookup.ndim != 2 or lookup.shape[1] != 2 or lookup.dtype.kind not in 'iu'
            or (lookup[:, 1] <= lookup[:, 0]).any()
            or (len(lookup) and (lookup[0, 0] != 0 or not np.array_equal(lookup[1:, 0], lookup[:-1, 1])))):
        raise ValueError('Invalid relative graph atom lookup')
    if len(records) != len(node_lookup):
        raise ValueError('Source records and graph count differ')
    encoded = []
    for record, (start, end) in zip(records, node_lookup):
        if record.get('version') != 1 or not record.get('source_smiles') or not record.get('source_sdf'):
            raise ValueError('Missing original Pharmit source identity')
        if (type(record.get('source_id')) is not int or record['source_id'] < 0
                or not isinstance(record.get('source_record_index'), int) or record['source_record_index'] < 0):
            raise ValueError('Missing source database ID or conformer record index')
        mapping = record['graph_to_source_heavy_atom']
        if sorted(mapping) != list(range(int(end-start))):
            raise ValueError('Invalid source heavy-atom mapping')
        encoded.append(json.dumps(record, sort_keys=True, separators=(',', ':')))
    return np.asarray(encoded, dtype=str)


def decode_records(tensors):
    if 'source_records_json' not in tensors:
        raise ValueError('Legacy Pharmit chunk has no source identity; recover its source join before rebuilding')
    records = [json.loads(str(value)) for value in tensors['source_records_json']]
    encode_records(records, tensors['node_lookup'])
    return records


def publish_records(store_path, records, graph_start, atom_offset, node_lookup, chunk_path):
    """Atomically publish a graph-indexed sidecar; identical retries are allowed."""
    encode_records(records, node_lookup)
    digest = hashlib.sha256(Path(chunk_path).read_bytes()).hexdigest()
    rows = []
    for offset, (record, (start, end)) in enumerate(zip(records, node_lookup)):
        rows.append(dict(record, graph_index=int(graph_start+offset),
                         atom_start=int(atom_offset+start), atom_end=int(atom_offset+end),
                         phase1_chunk_sha256=digest))
    payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n' for row in rows).encode()
    directory = Path(store_path)/'identity';directory.mkdir(exist_ok=True)
    path = directory/f'{graph_start:012d}_{graph_start+len(records):012d}.jsonl'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name);stream.write(payload);stream.flush();os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError('Existing Pharmit source sidecar differs; refusing overwrite')
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return path
