"""Leakage-safe classical and non-learned baselines for OpenCity."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def flatten_loader(loader,cap=None,seed=0):
    xs,ys,tids=[],[],[]
    for b in loader:
        batch,node,_length=b["history"].shape; cal=b["calendar"][:,None,:].expand(-1,node,-1).numpy()
        static=b["static"].numpy(); history=b["history"].numpy(); mask=b["history_mask"].numpy()
        x=np.concatenate((history,mask,static,np.broadcast_to(cal,(batch,node,4))),-1).reshape(batch*node,-1)
        y=b["target"].numpy().reshape(-1); time=np.repeat(b["time_id"].numpy(),node); valid=y>=0
        xs.append(x[valid]); ys.append(y[valid]); tids.append(time[valid])
    x,y,t=np.concatenate(xs),np.concatenate(ys),np.concatenate(tids)
    if cap and len(y)>cap:
        idx=np.random.default_rng(seed).choice(len(y),cap,replace=False); x,y,t=x[idx],y[idx],t[idx]
    return x,y,t


def build_baseline(name,seed):
    if name=="logistic":
        return make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight="balanced",C=1.,random_state=seed))
    if name=="hist_gbdt":
        return HistGradientBoostingClassifier(max_iter=250,learning_rate=.08,max_leaf_nodes=31,
                                               l2_regularization=1.,class_weight="balanced",random_state=seed)
    if name=="lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="multiclass",n_estimators=500,num_leaves=63,learning_rate=.05,
                              class_weight="balanced",random_state=seed,n_jobs=-1,verbosity=-1)
    raise ValueError(name)


class SourcePrior:
    def fit(self,y): self.p=(np.bincount(y,minlength=3)+1)/(len(y)+3); return self
    def predict_proba(self,x): return np.broadcast_to(self.p,(len(x),3)).copy()


class Persistence:
    """Classifies the latest observed speed with source-only thresholds."""
    def __init__(self,history_steps,stats): self.history_steps,self.stats=history_steps,stats
    def predict_proba(self,x):
        normalized=x[:,self.history_steps-1]; speed=normalized*self.stats["iqr"]+self.stats["median"]
        pred=np.zeros(len(x),int); pred[speed<=self.stats["warning"]]=1; pred[speed<=self.stats["severe"]]=2
        p=np.full((len(x),3),.01); p[np.arange(len(x)),pred]=.98; return p/p.sum(1,keepdims=True)
