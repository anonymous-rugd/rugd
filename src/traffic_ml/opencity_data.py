"""OpenCity multi-city speed data and leakage-safe forecasting windows."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class OpenCityStore:
    name: str
    speed: np.ndarray
    adjacency: np.ndarray
    static: np.ndarray
    timestamps: np.ndarray

    @classmethod
    def open(cls, root: str | Path, name: str, start: str, interval: int) -> OpenCityStore:
        p=Path(root)/name; speed=np.load(p/f"{name}.npz",allow_pickle=False)["data"].astype(np.float32)
        adjacency=np.load(p/f"{name}_rn_adj.npy",allow_pickle=False).astype(np.float32)
        if speed.ndim!=2 or adjacency.shape!=(speed.shape[1],speed.shape[1]):
            raise ValueError(f"Invalid OpenCity shapes for {name}: {speed.shape}, {adjacency.shape}")
        np.fill_diagonal(adjacency,1.); outd=adjacency.sum(1); ind=adjacency.sum(0)

        static=np.column_stack((np.log1p(outd),np.log1p(ind),outd/(outd.mean()+1e-6),
                                ind/(ind.mean()+1e-6))).astype(np.float32)
        static=(static-static.mean(0))/(static.std(0)+1e-6)
        timestamps=np.datetime64(start)+np.arange(len(speed))*np.timedelta64(interval,"m")
        return cls(name,speed,adjacency,static,timestamps)

    def normalized_adjacency(self) -> torch.Tensor:
        a=torch.from_numpy(self.adjacency); degree=a.sum(1).clamp_min(1)
        return degree.rsqrt()[:,None]*a*degree.rsqrt()[None,:]


@dataclass(frozen=True)
class IndexSplit:
    train: np.ndarray; calibration: np.ndarray; validation: np.ndarray; test: np.ndarray


def temporal_indices(length: int, history: int, horizon: int, fractions, embargo: int) -> IndexSplit:
    """Forecast-index split; history never crosses backward from a later partition."""
    valid=np.arange(history,length-horizon+1); n=len(valid)
    i,j,k=[int(n*x) for x in np.cumsum(fractions)[:-1]]
    groups=(valid[:i],valid[i+embargo:j],valid[j+embargo:k],valid[k+embargo:])
    if any(len(x)==0 for x in groups): raise ValueError("Empty temporal partition")
    return IndexSplit(*groups)


def fit_source_statistics(stores: list[OpenCityStore], splits: list[IndexSplit], cfg: dict) -> dict:
    """Fit normalization and class cutoffs exclusively on source training targets."""
    values=[]
    for store,split in zip(stores,splits):
        x=store.speed[split.train]; values.append(x[x>0])
    pooled=np.concatenate(values); median=float(np.median(pooled)); q25,q75=np.quantile(pooled,[.25,.75])
    severe=float(np.quantile(pooled,cfg["target"]["severe_quantile"]))
    warning=float(np.quantile(pooled,cfg["target"]["warning_quantile"]))
    if not severe<warning: raise ValueError("Degenerate source congestion thresholds")
    return {"median":median,"iqr":float(max(q75-q25,1e-6)),"severe":severe,"warning":warning,
            "fit_scope":"source_train_only","n_values":len(pooled)}


def speed_to_class(speed: np.ndarray, stats: dict) -> np.ndarray:
    """0 green, 1 warning, 2 severe; missing values map to -1."""
    y=np.full(speed.shape,-1,dtype=np.int64); valid=speed>0
    y[valid]=0; y[valid & (speed<=stats["warning"])]=1; y[valid & (speed<=stats["severe"])]=2
    return y


class CityWindowDataset(Dataset):
    def __init__(self, store: OpenCityStore, indices: np.ndarray, history: int, horizon: int,
                 stats: dict, city_index: int):
        self.store,self.indices,self.history,self.horizon,self.stats,self.city_index=store,indices,history,horizon,stats,city_index
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        t=int(self.indices[i]); raw=self.store.speed[t-self.history:t].T.copy(); mask=raw>0
        x=(raw-self.stats["median"])/self.stats["iqr"]; x[~mask]=0
        target_speed=self.store.speed[t+self.horizon-1]; target=speed_to_class(target_speed,self.stats)
        stamp=self.store.timestamps[t]; minute=int(stamp.astype("datetime64[m]").astype(int)%(24*60)); day=int(stamp.astype("datetime64[D]").astype(int)+3)%7
        calendar=np.array([np.sin(2*np.pi*minute/1440),np.cos(2*np.pi*minute/1440),
                           np.sin(2*np.pi*day/7),np.cos(2*np.pi*day/7)],np.float32)
        history_stamps=self.store.timestamps[t-self.history:t]
        history_minutes=(history_stamps.astype("datetime64[m]").astype(np.int64)%(24*60))
        history_tod=(history_minutes//self.store_interval_minutes()).astype(np.int64)
        history_days=((history_stamps.astype("datetime64[D]").astype(np.int64)+3)%7).astype(np.int64)
        return {"history":torch.from_numpy(x),"history_mask":torch.from_numpy(mask),
            "target":torch.from_numpy(target),"target_speed":torch.from_numpy(target_speed.copy()),
            "static":torch.from_numpy(self.store.static.copy()),"calendar":torch.from_numpy(calendar),
            "history_tod":torch.from_numpy(history_tod),
            "history_dow":torch.from_numpy(history_days),
            "time_id":torch.tensor(t),"city":torch.tensor(self.city_index)}

    def store_interval_minutes(self) -> int:
        """Infer the regular sampling interval used for categorical TOD bins."""
        if len(self.store.timestamps) < 2:
            return 30
        delta = self.store.timestamps[1] - self.store.timestamps[0]
        return int(delta / np.timedelta64(1, "m"))


def write_data_manifest(root: str | Path, output: str | Path, cities: list[str]) -> None:
    records=[]
    for city in cities:
        for p in sorted((Path(root)/city).glob("*")):
            h=hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
            records.append({"path":str(p),"bytes":p.stat().st_size,"sha256":h.hexdigest()})
    Path(output).write_text(json.dumps(records,indent=2),encoding="utf-8")
