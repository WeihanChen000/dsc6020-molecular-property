"""Read-only source-integrity, environment and partition descriptive audit."""
import importlib.metadata,json,platform,sys
import numpy as np,pandas as pd,psutil,torch
from rdkit import Chem
from core import ROOT,sha,dump

def main():
    manifest=json.loads((ROOT/'inherited_manifest.json').read_text())
    for x in manifest['files']:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','rdkit','torch','chemprop','transformers','matplotlib','lightning','psutil']}
    raw_checks=[]
    for item in json.loads((ROOT/'provenance/raw_data_audit.json').read_text()):
        name='delaney-processed.csv' if item['dataset']=='ESOL' else 'Lipophilicity.csv'
        path=ROOT/'provenance/raw_data'/name;assert sha(path)==item['raw_sha256']
        raw=pd.read_csv(path);samples=pd.read_csv(ROOT/'inputs'/item['dataset']/'samples.csv');selected=raw.iloc[samples.source_row.to_numpy()]
        molecules=[Chem.MolFromSmiles(s) for s in selected.smiles]
        assert [Chem.MolToSmiles(m,isomericSmiles=True) for m in molecules]==list(samples.smiles)
        assert [Chem.MolToSmiles(m,isomericSmiles=False) for m in molecules]==list(samples.connectivity)
        np.testing.assert_allclose(selected[item['target']],samples.y_true)
        raw_checks.append({'dataset':item['dataset'],'sha256':sha(path),'labels_and_smiles_checked':len(samples),'source_url':item['source_url']})
    dump(ROOT/'environment.json',{'python':sys.version,'platform':platform.platform(),'packages':packages,'torch_cuda_available':torch.cuda.is_available(),'logical_cpus':psutil.cpu_count(),'memory_GiB':psutil.virtual_memory().total/1024**3,'provenance_files_hash_verified':len(manifest['files']),'raw_data_checks':raw_checks,'runtime_executable':sys.executable})
    rows=[];jobs=json.loads((ROOT/'jobs.json').read_text())
    for path in sorted(set(j['partition_file'] for j in jobs)):
        f=pd.read_csv(ROOT/path)
        for partition,g in f.groupby('partition'):
            rows.append({'file':path,'partition':partition,'n':len(g),'label_mean':g.y_true.mean(),'label_sd':g.y_true.std(),'scaffolds':g.scaffold.nunique(),'acyclic_n':g.scaffold.eq('__ACYCLIC__').sum()})
    pd.DataFrame(rows).to_csv(ROOT/'partition_audit.csv',index=False)
    print('Inherited files verified:',len(manifest['files']),'packages:',packages,flush=True)
if __name__=='__main__':main()
