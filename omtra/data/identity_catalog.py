"""Immutable, worker-safe source identity lookup bound to an exact dataset view."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import numpy as np
from rdkit import Chem


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):result.update(block)
    return result.hexdigest()


def view_fingerprint(dataset):
    """Bind indices to store schema and ordered, filtered membership; not all raw chunks."""
    root = Path(dataset.store_path)
    digest = hashlib.sha256()
    metadata = [root/'zarr.json']
    if hasattr(dataset, 'system_lookup'):
        fields = ['system_id', 'ligand_id'] if 'system_id' in dataset.system_lookup else ['lig_sdf', 'rec_pdb']
        fields += ['lig_atom_start', 'lig_atom_end', 'lig_bond_start', 'lig_bond_end']
        for row in dataset.system_lookup[fields].itertuples(index=False, name=None):
            digest.update(json.dumps(list(row), separators=(',', ':')).encode()+b'\n')
    else:
        metadata += [root/p/'zarr.json' for p in ('lig/node/graph_lookup','lig/edge/graph_lookup')]
    digest.update(str(len(dataset)).encode())
    for path in metadata:
        digest.update(str(path.relative_to(root)).encode());digest.update(file_hash(path).encode())
    return digest.hexdigest()


def build_catalog(output, manifest, records, seal=False):
    """Publish a new catalog. An incomplete review catalog cannot serve training."""
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    if manifest.get('version') != 1:raise ValueError('Unknown identity catalog version')
    views = manifest['views'];evidence = manifest['evidence']
    for value in evidence.values():
        if file_hash(value['path']) != value['sha256']:raise ValueError('Source evidence hash differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=output.parent, prefix='.identity_catalog_')
    os.close(handle);connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE identities (view TEXT NOT NULL, idx INTEGER NOT NULL,
                status TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(view,idx));
            CREATE INDEX identity_status ON identities(view,status,idx);
        ''')
        for record in records:
            view = record['view'];index = record['index'];status = record['status']
            if view not in views or type(index) is not int or not 0 <= index < views[view]['count']:
                raise ValueError('Record index is outside its bound dataset view')
            if status not in ('approved','quarantine','review'):raise ValueError('Unknown identity disposition')
            if record.get('evidence') not in evidence:raise ValueError('Record lacks source evidence')
            if status == 'approved':
                if record.get('source_kind') not in ('source_database','ccd_definition') or record.get('independent') is not True:
                    raise ValueError('Approved identity needs an independently reviewed chemical source')
                mol = Chem.MolFromSmiles(record['smiles'])
                if mol is None:raise ValueError('Invalid source SMILES')
                if any(atom.GetIsotope() for atom in mol.GetAtoms()):raise ValueError('Unsupported isotope identity')
                mol = Chem.RemoveHs(mol);mapping = record['graph_to_identity']
                if any(type(i) is not int for i in mapping) or sorted(mapping) != list(range(mol.GetNumAtoms())):raise ValueError('Invalid identity atom mapping')
                from omtra.data.requested_stereo import labels_from_identity
                atoms,bonds=labels_from_identity(mol)
                if (atoms==6).any() or 4 in bonds.values():raise ValueError('Unsupported stereo identity')
            elif not record.get('reason'):raise ValueError('Review/quarantine requires an explicit reason')
            connection.execute('INSERT INTO identities VALUES (?,?,?,?)',
                               (view,index,status,json.dumps(record,sort_keys=True)))
        coverage = {}
        for view, spec in views.items():
            counts = dict(connection.execute('SELECT status,COUNT(*) FROM identities WHERE view=? GROUP BY status',(view,)))
            coverage[view] = {'expected':spec['count'],'missing':spec['count']-sum(counts.values()),**counts}
            if seal and (sum(counts.values()) != spec['count'] or counts.get('review',0) or not counts.get('approved',0)):
                raise ValueError('Cannot seal incomplete, unresolved or entirely quarantined identity coverage')
        metadata = dict(manifest, sealed=bool(seal), coverage=coverage, quarantine_scope='identity_fixed_tasks_only')
        connection.execute('INSERT INTO metadata VALUES (?,?)',('manifest',json.dumps(metadata,sort_keys=True)))
        connection.commit();connection.close();connection=None
        os.link(temporary,output)
    finally:
        if connection is not None:connection.close()
        Path(temporary).unlink(missing_ok=True)
    return metadata


class IdentityCatalog:
    def __init__(self, path, require_sealed=True):
        self.path = str(Path(path).resolve());self._connection=None;self._pid=None
        row = self.connection.execute('SELECT value FROM metadata WHERE key=?',('manifest',)).fetchone()
        self.manifest = json.loads(row[0]);self._verified_evidence=set()
        if self.manifest.get('version') != 1:
            self.close();raise ValueError('Unknown identity catalog version')
        if require_sealed and not self.manifest.get('sealed'):
            self.close();raise ValueError('Identity catalog is not sealed for training')

    def close(self):
        if self._connection is not None:self._connection.close()
        self._connection=None;self._pid=None;self._verified_evidence=set()

    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    @property
    def connection(self):
        if self._connection is None or self._pid != os.getpid():
            if self._connection is not None:self._connection.close()
            uri = Path(self.path).as_uri()+'?mode=ro&immutable=1'
            self._connection=sqlite3.connect(uri,uri=True);self._pid=os.getpid()
            self._verified_evidence=set()
        return self._connection

    def __getstate__(self):
        state=dict(self.__dict__);state['_connection']=None;state['_pid']=None;state['_verified_evidence']=set()
        return state

    def bind(self, view, dataset):
        if not self.manifest.get('sealed'):raise ValueError('Unsealed identity coverage cannot bind a training dataset')
        spec=self.manifest['views'].get(view)
        if spec is None or spec['count'] != len(dataset) or spec['sha256'] != view_fingerprint(dataset):
            raise ValueError(f'Identity catalog does not match dataset membership/schema: {view}')
        return BoundIdentityView(self,view)

    def record(self, view, index):
        row=self.connection.execute('SELECT status,payload FROM identities WHERE view=? AND idx=?',(view,int(index))).fetchone()
        if row is None:raise ValueError(f'Missing source identity: {view}|{index}')
        if row[0] != 'approved':raise ValueError(f'Identity is {row[0]}: {view}|{index}')
        record=json.loads(row[1]);key=record['evidence']
        if key not in self._verified_evidence:
            source=self.manifest['evidence'][key]
            if file_hash(source['path']) != source['sha256']:raise ValueError('Identity source evidence changed')
            self._verified_evidence.add(key)
        return record


class BoundIdentityView:
    def __init__(self,catalog,view):self.catalog=catalog;self.view=view

    def allowed_mask(self,start,end):
        mask=np.ones(end-start,dtype=bool)
        rows=self.catalog.connection.execute('SELECT idx FROM identities WHERE view=? AND status=? AND idx>=? AND idx<?',
                                             (self.view,'quarantine',int(start),int(end)))
        for (index,) in rows:mask[index-start]=False
        return mask

    def attach(self,graph,task,index):
        from omtra.data.requested_stereo import eligible,clear_stereo,attach_identity
        from omtra.data.identity_audit import assess_stereo_compatibility
        if not eligible(task):clear_stereo(graph);return graph
        record=self.catalog.record(self.view,index)
        mol=Chem.RemoveHs(Chem.MolFromSmiles(record['smiles']))
        mol=Chem.RenumberAtoms(mol,record['graph_to_identity'])
        # Chemistry and mapping are validated before any coordinate compatibility check.
        attach_identity(graph,task,mol,source='supplied')
        result=assess_stereo_compatibility(mol,graph.nodes['lig'].data['x_1_true'].detach().cpu().numpy())
        if not result['compatible']:
            clear_stereo(graph)
            raise ValueError(f'Source identity conflicts with training target: {self.view}|{index}: {result}')
        return graph
