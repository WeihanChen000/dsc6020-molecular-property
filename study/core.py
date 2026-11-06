"""Shared v3 models and train-only preprocessing; candidates never score test data."""
from pathlib import Path
import copy,hashlib,json,random,time
import joblib,numpy as np,pandas as pd,torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
torch.set_num_threads(4);threadpool_limits(4)
NEURAL={'mlp','descriptor_mlp','dmpnn','dmpnn_budget'}
GRAPH={'dmpnn','dmpnn_budget'}
RIDGES={'ridge','descriptor_ridge','fusion_ridge'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def signature():return hashlib.sha256(b''.join((ROOT/n).read_bytes() for n in ['core.py','worker.py','protocol.json','jobs.json'])).hexdigest()
def score(y,p):
    assert np.isfinite(p).all()
    return {'rmse':float(np.sqrt(np.mean((np.asarray(y)-p)**2))),'mae':float(np.mean(np.abs(np.asarray(y)-p)))}

class DMPNN(nn.Module):
    def __init__(self):
        super().__init__()
        from chemprop import nn as cpnn
        c=PROTOCOL['dmpnn']
        self.mp=cpnn.BondMessagePassing(d_h=c['hidden_dim'],depth=c['depth'],dropout=c['dropout'])
        self.agg=cpnn.MeanAggregation()
        self.head=cpnn.RegressionFFN(input_dim=c['hidden_dim'],hidden_dim=c['hidden_dim'],n_layers=1,dropout=c['dropout'])
    def forward(self,batch):return self.head(self.agg(self.mp(batch),batch.batch)).squeeze(1)

class MLP(nn.Module):
    def __init__(self,input_dim):
        super().__init__();c=PROTOCOL['mlp']
        self.net=nn.Sequential(nn.Linear(input_dim,c['hidden_dim']),nn.ReLU(),nn.Dropout(c['dropout']),nn.Linear(c['hidden_dim'],1))
    def forward(self,x):return self.net(x).squeeze(1)

def make_graphs(smiles):
    from chemprop import featurizers
    from rdkit import Chem
    f=featurizers.SimpleMoleculeMolGraphFeaturizer()
    return [f(Chem.MolFromSmiles(s)) for s in smiles]
def batch_graph(graphs,idx):
    from chemprop.data import BatchMolGraph
    return BatchMolGraph([graphs[int(i)] for i in idx])

def features(dataset,route,cache):
    key=(dataset,'graph' if route in GRAPH else route)
    if key in cache:return cache[key]
    folder=ROOT/'inputs'/dataset;samples=pd.read_csv(folder/'samples.csv')
    if route in GRAPH:X=make_graphs(samples.connectivity)
    else:
        names=['features.npz','embeddings.npz'] if route=='fusion_ridge' else ['ecfp_count.npz'] if route=='ecfp_rf' else ['embeddings.npz'] if route in ['ridge','mlp'] else ['features.npz']
        xs=[]
        for name in names:
            z=np.load(folder/name);assert list(z['sample_id'])==list(samples.sample_id);xs.append(z['X'])
        X=np.concatenate(xs,axis=1) if len(xs)>1 else xs[0]
    cache[key]=X;return X

@torch.inference_mode()
def neural_predict(model,X,indices,route,mean,scale):
    model.eval();p=[]
    for start in range(0,len(indices),128):
        idx=indices[start:start+128]
        p.append(model(batch_graph(X,idx) if route in GRAPH else X[idx]).numpy())
    return np.concatenate(p)*scale+mean

def train_neural(route,X,y,train,valid,param,seed,folder):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    cfg=PROTOCOL['dmpnn' if route in GRAPH else 'mlp'];mean=float(y[train].mean());scale=max(float(y[train].std()),1e-8)
    if route in GRAPH:model=DMPNN();input_dim=None
    else:
        prep=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()).fit(X[train])
        X=torch.tensor(prep.transform(X),dtype=torch.float32);assert torch.isfinite(X).all()
        input_dim=X.shape[1];model=MLP(input_dim);joblib.dump(prep,folder/'preprocessor.joblib')
    optimizer=torch.optim.Adam(model.parameters(),lr=param['lr'],weight_decay=cfg['weight_decay'])
    target=torch.tensor((y-mean)/scale,dtype=torch.float32)
    best=float('inf');best_epoch=0;best_state=None;stale=0;history=[];epoch=0;stop=cfg['max_epochs'];extended=False;extended450=False
    started=time.perf_counter()
    def save(name):
        meta={'route':route,'seed':seed,'lr':param['lr'],'best_epoch':best_epoch,'epochs_run':epoch,
              'extended':extended,'extended450':extended450,'near_epoch_limit':bool(epoch==stop and best_epoch>=.9*stop),
              'valid_rmse':best,'y_mean':mean,'y_scale':scale,'fit_seconds':time.perf_counter()-started,
              'parameters':sum(p.numel() for p in model.parameters()),'input_dim':input_dim}
        torch.save({'state_dict':best_state,'metadata':meta},folder/name)
        return meta
    while epoch<stop:
        epoch+=1;model.train();order=rng.permutation(train);total=0.
        for start in range(0,len(order),cfg['batch_size']):
            idx=order[start:start+cfg['batch_size']]
            batch=batch_graph(X,idx) if route in GRAPH else X[idx]
            optimizer.zero_grad();loss=nn.functional.mse_loss(model(batch),target[idx]);assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(idx)
        current=score(y[valid],neural_predict(model,X,valid,route,mean,scale))['rmse']
        history.append({'epoch':epoch,'train_mse_standardized':total/len(train),'valid_rmse':current})
        if current<best:best=current;best_epoch=epoch;stale=0;best_state=copy.deepcopy(model.state_dict())
        else:stale+=1
        if epoch%50==0:print(f'{route} seed={seed} lr={param["lr"]} epoch={epoch} best_valid={best:.5f}',flush=True)
        if route=='dmpnn_budget' and epoch==300:save('budget300.pt')
        if stale>=cfg['patience']:break
        if route in GRAPH and epoch==cfg['max_epochs'] and best_epoch>=130:stop=300;extended=True
        if route=='dmpnn_budget' and epoch==300 and best_epoch>=270:stop=PROTOCOL['budget_extension_max'];extended450=True
    meta=save('model.pt')
    if route=='dmpnn_budget' and not (folder/'budget300.pt').exists():save('budget300.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False);dump(folder/'fit.json',meta)
    return meta

def train_tabular(route,X,y,train,valid,param,seed,folder):
    if route in ['rf','ecfp_rf']:
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestRegressor(n_estimators=200,n_jobs=4,random_state=seed,**param))
    else:model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=param['alpha'],solver='svd'))
    start=time.perf_counter();model.fit(X[train],y[train]);elapsed=time.perf_counter()-start
    meta={'route':route,'seed':seed,'params':param,'valid_rmse':score(y[valid],model.predict(X[valid]))['rmse'],'fit_seconds':elapsed,'input_dim':X.shape[1]}
    joblib.dump(model,folder/'model.joblib',compress=3);dump(folder/'fit.json',meta);return meta

def restore(route,folder,X,idx,checkpoint='model.pt'):
    if route not in NEURAL:return joblib.load(folder/'model.joblib').predict(X[idx])
    c=torch.load(folder/checkpoint,map_location='cpu',weights_only=True)
    model=DMPNN() if route in GRAPH else MLP(X.shape[1]);model.load_state_dict(c['state_dict'])
    if route not in GRAPH:
        p=folder/'preprocessor.joblib'
        prep=joblib.load(p if p.exists() else folder/'input_scaler.joblib')
        X=torch.tensor(prep.transform(X),dtype=torch.float32)
    return neural_predict(model,X,idx,route,c['metadata']['y_mean'],c['metadata']['y_scale'])
