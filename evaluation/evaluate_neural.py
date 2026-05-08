#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sEMG Baseline Waveform Model Inference (v2.0)

Compatible with generate_test_data.py v6.8.0:
  All (snr, k) combinations are now valid — k=5 appears at ALL SNR values.
  No fair_snr_set adjustment needed; k=X rows aggregate over all SNR conditions
  equally, so the noise_type table comparison is inherently fair.

Synced metric set:
  SNRimp, RMSE, PRD, LSD, RMSE_ARV, RMSE_ZCR, RMSE_MNF, RMSE_MDF, RMSE_Kurtosis
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from scipy import signal

SEMG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SEMG_ROOT)

from baseline_models import BASELINE_MODEL_REGISTRY


# ============================================================================
# Noise type helpers
# ============================================================================
def _infer_color_subtype_from_paths(noise_paths_str: str) -> Optional[str]:
    if not noise_paths_str or noise_paths_str == "nan":
        return None
    subtypes = set()
    for p in noise_paths_str.split("|"):
        bn = os.path.basename(p.strip()).lower()
        if "color" not in bn: continue
        if "_pink_"  in bn: subtypes.add("Pink")
        elif "_brown_" in bn: subtypes.add("Brown")
    return subtypes.pop() if len(subtypes) == 1 else None


def _noise_type_labels(noise_types_str: str, noise_paths_str: str) -> List[str]:
    if not noise_types_str or str(noise_types_str) == "nan":
        return []
    types  = [t.strip() for t in str(noise_types_str).split("+") if t.strip()]
    labels = list(types)
    if "Color" in types:
        sub = _infer_color_subtype_from_paths(str(noise_paths_str))
        if sub: labels.append(sub)
    labels.append(f"k={len(types)}")
    return labels


# ============================================================================
# Metric functions
# ============================================================================
def cal_SNR(clean: np.ndarray, test: np.ndarray) -> float:
    clean = clean.reshape(-1).astype(np.float64); test = test.reshape(-1).astype(np.float64)
    n_pw = np.dot(test-clean, test-clean); s_pw = np.dot(clean, clean)
    return 100.0 if n_pw < 1e-12 else float(10.0*np.log10(s_pw/(n_pw+1e-12)))

def cal_SNRimp(clean, denoised, noisy) -> float:
    return float(cal_SNR(clean, denoised) - cal_SNR(clean, noisy))

def cal_RMSE(clean, enhanced) -> float:
    c=clean.reshape(-1).astype(np.float64); e=enhanced.reshape(-1).astype(np.float64)
    return float(np.sqrt(np.mean((e-c)**2)))

def cal_PRD(clean, enhanced) -> float:
    c=clean.reshape(-1).astype(np.float64); e=enhanced.reshape(-1).astype(np.float64)
    return float(np.sqrt(np.sum((e-c)**2)/(np.sum(c**2)+1e-12))*100)

def cal_ARV(emg: np.ndarray, window_size: int = 200) -> np.ndarray:
    emg=np.abs(emg.reshape(-1).astype(np.float64))
    return np.array([emg[i:i+window_size].mean() for i in range(0,emg.shape[0],window_size)
                     if len(emg[i:i+window_size])>0], dtype=np.float64)

def cal_RMSE_ARV(clean, enhanced, window_size: int = 200) -> float:
    a,b=cal_ARV(clean,window_size),cal_ARV(enhanced,window_size); n=min(len(a),len(b))
    return float(np.sqrt(np.mean((a[:n]-b[:n])**2))) if n>0 else 0.0

def cal_LSD(clean: np.ndarray, enhanced: np.ndarray, sr: int = 1000,
            n_fft: int = 512, hop: int = 128, f_min: float = 20.0,
            f_max: float = 500.0, eps: float = 1e-10) -> float:
    clean=clean.reshape(-1).astype(np.float64); enhanced=enhanced.reshape(-1).astype(np.float64)
    if len(clean)<3*n_fft or len(enhanced)<3*n_fft: return np.nan
    win=np.hanning(n_fft)
    freq,_,S_c=signal.stft(clean,fs=sr,window=win,nperseg=n_fft,noverlap=n_fft-hop)
    _,_,S_e=signal.stft(enhanced,fs=sr,window=win,nperseg=n_fft,noverlap=n_fft-hop)
    mask=(freq>=f_min)&(freq<=f_max)
    if not mask.any(): mask=np.ones(len(freq),dtype=bool)
    log_c=10.0*np.log10(np.abs(S_c[mask,:])**2+eps)
    log_e=10.0*np.log10(np.abs(S_e[mask,:])**2+eps)
    return float(np.mean(np.sqrt(np.mean((log_c-log_e)**2,axis=0))))

def _zcr_per_window(x: np.ndarray, window_size: int = 200, sr: int = 1000) -> np.ndarray:
    x=x.reshape(-1).astype(np.float64)
    return np.array([float(np.sum(np.diff(np.sign(x[i:i+window_size]))!=0))/(window_size/sr)
                     for i in range(0,len(x)-window_size+1,window_size)], dtype=np.float64)

def cal_RMSE_ZCR(clean, enhanced, window_size: int = 200, sr: int = 1000) -> float:
    z_c=_zcr_per_window(clean,window_size,sr); z_e=_zcr_per_window(enhanced,window_size,sr)
    n=min(len(z_c),len(z_e))
    return float(np.sqrt(np.mean((z_c[:n]-z_e[:n])**2))) if n>0 else 0.0

def _mnf_per_window(emg: np.ndarray, sr: int = 1000,
                    f_min: float = 20.0, f_max: float = 500.0) -> np.ndarray:
    emg=emg.reshape(-1).astype(np.float64)
    if len(emg)<200: return np.array([100.0],dtype=np.float64)
    freq,_,spec=signal.stft(emg,fs=sr,window='boxcar',nperseg=200,noverlap=0,nfft=1024,boundary='constant')
    spec=np.abs(spec); rec_win=signal.get_window('boxcar',200)
    spec=spec/np.sqrt(1.0/rec_win.sum()**2)
    si=max(0,min(np.searchsorted(freq,f_min),len(freq)-1))
    ei=max(si+1,min(np.searchsorted(freq,f_max),len(freq)))
    freq_r,spec_r=freq[si:ei],spec[si:ei,:]
    wf=np.sum(freq_r[:,np.newaxis]*spec_r,axis=0); sp=np.sum(spec_r,axis=0)
    valid=sp>1e-12; MNF=np.zeros_like(sp,dtype=np.float64)
    if np.any(valid): MNF[valid]=wf[valid]/sp[valid]; MNF[~valid]=np.median(MNF[valid])
    else: MNF[:]=100.0
    return MNF.astype(np.float64)

def cal_RMSE_MNF(clean, enhanced, sr: int = 1000,
                 f_min: float = 20.0, f_max: float = 500.0) -> float:
    try:
        mc=_mnf_per_window(clean,sr,f_min,f_max); me=_mnf_per_window(enhanced,sr,f_min,f_max)
        n=min(len(mc),len(me))
        return float(np.sqrt(np.mean((mc[:n]-me[:n])**2))) if n>0 else 0.0
    except Exception as e: print(f"[WARN] RMSE_MNF: {e}"); return 0.0

def _mdf_per_window(x: np.ndarray, window_size: int = 200, sr: int = 1000,
                    f_min: float = 20.0, f_max: float = 500.0, nfft: int = 1024) -> np.ndarray:
    x=x.reshape(-1).astype(np.float64); freq=np.fft.rfftfreq(nfft,1.0/sr)
    mask=(freq>=f_min)&(freq<=f_max); freq_r=freq[mask]; mdf=[]
    for i in range(0,len(x)-window_size+1,window_size):
        seg=x[i:i+window_size]; pad=np.zeros(nfft); pad[:len(seg)]=seg
        psd_r=np.abs(np.fft.rfft(pad*np.hanning(nfft)))**2; psd_r=psd_r[mask]
        total=psd_r.sum()
        if total<1e-12: mdf.append(np.nan); continue
        cum=np.cumsum(psd_r); idx=min(int(np.searchsorted(cum,total/2.0)),len(freq_r)-1)
        mdf.append(float(freq_r[idx]))
    arr=np.array(mdf,dtype=np.float64); valid=np.isfinite(arr)
    if valid.any(): arr[~valid]=np.nanmedian(arr[valid])
    else: arr[:]=(f_min+f_max)/2.0
    return arr

def cal_RMSE_MDF(clean, enhanced, window_size: int = 200, sr: int = 1000,
                 f_min: float = 20.0, f_max: float = 500.0) -> float:
    try:
        mc=_mdf_per_window(clean,window_size,sr,f_min,f_max)
        me=_mdf_per_window(enhanced,window_size,sr,f_min,f_max)
        n=min(len(mc),len(me))
        return float(np.sqrt(np.mean((mc[:n]-me[:n])**2))) if n>0 else 0.0
    except Exception as e: print(f"[WARN] RMSE_MDF: {e}"); return 0.0

def _kurtosis_per_window(x: np.ndarray, window_size: int = 200) -> np.ndarray:
    x=x.reshape(-1).astype(np.float64); out=[]
    for i in range(0,len(x)-window_size+1,window_size):
        seg=x[i:i+window_size]; std=seg.std()
        out.append(float(((seg-seg.mean())**4).mean()/std**4-3.0) if std>1e-12 else 0.0)
    return np.array(out,dtype=np.float64)

def cal_RMSE_Kurtosis(clean, enhanced, window_size: int = 200) -> float:
    kc=_kurtosis_per_window(clean,window_size); ke=_kurtosis_per_window(enhanced,window_size)
    n=min(len(kc),len(ke))
    return float(np.sqrt(np.mean((kc[:n]-ke[:n])**2))) if n>0 else 0.0

DEFAULT_METRICS = ["SNRimp","RMSE","PRD","LSD","RMSE_ARV","RMSE_ZCR","RMSE_MNF","RMSE_MDF","RMSE_Kurtosis"]

def calculate_all_metrics(clean, denoised, noisy, sr=1000, arv_window=200,
                          f_min=20.0, f_max=500.0, lsd_n_fft=512, lsd_hop=128) -> dict:
    c=np.asarray(clean,dtype=np.float64).reshape(-1)
    d=np.asarray(denoised,dtype=np.float64).reshape(-1)
    n=np.asarray(noisy,dtype=np.float64).reshape(-1)
    fns=[("SNRimp",lambda:cal_SNRimp(c,d,n)),("RMSE",lambda:cal_RMSE(c,d)),
         ("PRD",lambda:cal_PRD(c,d)),
         ("LSD",lambda:cal_LSD(c,d,sr=sr,n_fft=lsd_n_fft,hop=lsd_hop,f_min=f_min,f_max=f_max)),
         ("RMSE_ARV",lambda:cal_RMSE_ARV(c,d,arv_window)),
         ("RMSE_ZCR",lambda:cal_RMSE_ZCR(c,d,arv_window,sr)),
         ("RMSE_MNF",lambda:cal_RMSE_MNF(c,d,sr,f_min,f_max)),
         ("RMSE_MDF",lambda:cal_RMSE_MDF(c,d,arv_window,sr,f_min,f_max)),
         ("RMSE_Kurtosis",lambda:cal_RMSE_Kurtosis(c,d,arv_window))]
    out={}
    for name,fn in fns:
        try:
            val=fn(); out[name]=float(val) if np.isfinite(val) else np.nan
        except Exception as e:
            print(f"[WARN] {name}: {e}"); out[name]=np.nan
    return out


# ============================================================================
# Results collector + table builders
# ============================================================================
_SINGLE_TYPE_ORDER = ["PLI","ECG","MOA","WGN","Color","Pink","Brown"]
_KCOUNT_ORDER      = ["k=1","k=2","k=3","k=4","k=5"]


class ResultsCollector:
    """
    Simple results aggregator.
    No fair_snr_set needed: with v6.8.0 test data, all k values appear
    at all SNR conditions, so k=X aggregation is inherently fair.
    """
    def __init__(self):
        self.results          = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
        self.by_noise_type    = defaultdict(lambda:defaultdict(list))
        self.by_snr_noisetype = defaultdict(lambda:defaultdict(list))
        self.n_total = self.n_ok = 0

    def add(self, db, snr, k, metrics, noise_type_labels=None):
        self.n_total += 1
        for name,val in metrics.items():
            if val is None or not np.isfinite(val): continue
            fval=float(val)
            self.results[db][snr][k][name].append(fval)
            if noise_type_labels:
                for lbl in noise_type_labels:
                    self.by_noise_type[lbl][name].append(fval)
                    self.by_snr_noisetype[(snr,lbl)][name].append(fval)
        self.n_ok += 1

    def _collect(self, db=None, snr=None, k=None):
        vals=defaultdict(list)
        for d in (list(self.results) if db is None else [db]):
            if d not in self.results: continue
            for s in (list(self.results[d]) if snr is None else [snr]):
                if s not in self.results[d]: continue
                for kk in (list(self.results[d][s]) if k is None else [k]):
                    if kk not in self.results[d][s]: continue
                    for m,arr in self.results[d][s][kk].items(): vals[m].extend(arr)
        return {m:{"mean":float(np.mean(a)),"std":float(np.std(a,ddof=1)) if len(a)>1 else 0.0,"n":int(len(a))}
                for m,a in vals.items() if a}

    def summary(self, db=None, snr=None, k=None): return self._collect(db,snr,k)

    def noise_type_summary(self, label):
        return {m:{"mean":float(np.mean(a)),"std":float(np.std(a,ddof=1)) if len(a)>1 else 0.0,"n":int(len(a))}
                for m,a in self.by_noise_type[label].items() if a}

    def snr_noisetype_summary(self, snr, label):
        return {m:{"mean":float(np.mean(a)),"std":float(np.std(a,ddof=1)) if len(a)>1 else 0.0,"n":int(len(a))}
                for m,a in self.by_snr_noisetype[(snr,label)].items() if a}

    def all_noise_type_labels(self): return sorted(self.by_noise_type.keys())
    def all_snr_inputs(self): return sorted({s for s,_ in self.by_snr_noisetype},key=int)


def make_snr_k_table(col: ResultsCollector, metric: str,
                     db: Optional[str] = None) -> pd.DataFrame:
    snrs,ks=set(),set()
    for d in col.results:
        if db and d!=db: continue
        for s in col.results[d]: snrs.add(s); ks.update(col.results[d][s].keys())
    rows=[]
    for s in sorted(snrs,key=int):
        row={"SNR":s}
        for k in sorted(ks):
            sm=col.summary(db=db,snr=s,k=k)
            row[f"k={k}"]=round(sm[metric]["mean"],7) if metric in sm else np.nan
        sm_all=col.summary(db=db,snr=s)
        row["Avg"]=round(sm_all[metric]["mean"],7) if metric in sm_all else np.nan
        rows.append(row)
    row_avg={"SNR":"Avg"}
    for k in sorted(ks):
        sm=col.summary(db=db,k=k)
        row_avg[f"k={k}"]=round(sm[metric]["mean"],7) if metric in sm else np.nan
    sm_all=col.summary(db=db)
    row_avg["Avg"]=round(sm_all[metric]["mean"],7) if metric in sm_all else np.nan
    rows.append(row_avg)
    return pd.DataFrame(rows)


def make_noise_type_table(col: ResultsCollector,
                          metrics_list: List[str]) -> pd.DataFrame:
    all_lbl=set(col.all_noise_type_labels())
    ordered=[l for l in _SINGLE_TYPE_ORDER+_KCOUNT_ORDER if l in all_lbl]
    ordered+=[l for l in sorted(all_lbl) if l not in ordered]
    rows=[]
    for lbl in ordered:
        sm=col.noise_type_summary(lbl)
        row={"noise_type":lbl}
        for m in metrics_list:
            row[f"{m}_mean"]=round(sm[m]["mean"],7) if m in sm else np.nan
            row[f"{m}_std"] =round(sm[m]["std"],7)  if m in sm else np.nan
            row[f"{m}_n"]   =sm[m]["n"]              if m in sm else 0
        rows.append(row)
    return pd.DataFrame(rows)


def make_snr_noisetype_table(col: ResultsCollector, metric: str) -> pd.DataFrame:
    all_lbl=set(col.all_noise_type_labels()); snr_inputs=col.all_snr_inputs()
    ordered=[l for l in _SINGLE_TYPE_ORDER+_KCOUNT_ORDER if l in all_lbl]
    ordered+=[l for l in sorted(all_lbl) if l not in ordered]
    rows=[]
    for snr in snr_inputs:
        row={"SNR_input":snr}
        for lbl in ordered:
            sm=col.snr_noisetype_summary(snr,lbl)
            row[lbl]=round(sm[metric]["mean"],7) if metric in sm else np.nan
        overall=col.summary(snr=snr)
        row["Avg"]=round(overall[metric]["mean"],7) if metric in overall else np.nan
        rows.append(row)
    row_avg={"SNR_input":"Avg"}
    for lbl in ordered:
        sm=col.noise_type_summary(lbl)
        row_avg[lbl]=round(sm[metric]["mean"],7) if metric in sm else np.nan
    sm_all=col.summary()
    row_avg["Avg"]=round(sm_all[metric]["mean"],7) if metric in sm_all else np.nan
    rows.append(row_avg)
    return pd.DataFrame(rows)


# ============================================================================
# Inference
# ============================================================================
def run_inference(config, model_name, model_path, test_data_path, output_dir,
                  batch_size=32, metrics_list=None, sampling_rate=1000):
    if metrics_list is None: metrics_list=DEFAULT_METRICS
    metrics_cfg=config.get("metrics",{}) or {}
    f_min=float(metrics_cfg.get("f_min",20.0)); f_max=float(metrics_cfg.get("f_max",500.0))
    lsd_n_fft=int(metrics_cfg.get("lsd_n_fft",512)); lsd_hop=int(metrics_cfg.get("lsd_hop",128))

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if model_name not in BASELINE_MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")
    model=BASELINE_MODEL_REGISTRY[model_name]().to(device)
    state=torch.load(model_path,map_location=device)
    if isinstance(state,dict) and "model" in state: state=state["model"]
    model.load_state_dict(state); model.eval()
    print(f"[Model] {model_name} <- {model_path}")

    print(f"\n[Test Data] {test_data_path}")
    data=np.load(test_data_path,allow_pickle=True)["data"]
    print(f"  Samples: {len(data)}")
    has_noise_paths=(len(data)>0 and isinstance(data[0],dict) and "noise_paths" in data[0])

    collector=ResultsCollector()
    n_batches=(len(data)+batch_size-1)//batch_size

    with torch.inference_mode():
        for bi in tqdm(range(n_batches),desc=f"Inference [{model_name}]"):
            s,e=bi*batch_size,min((bi+1)*batch_size,len(data))
            batch=data[s:e]
            cleans=[item["clean"] for item in batch]
            noisys=[item["noisy"] for item in batch]
            scales=[item["scale"] for item in batch]
            metas=[{"dataset":item["dataset"],"snr":item["snr"],"k":item["k"],
                    "noise_types":str(item.get("noise_types","")),"noise_paths":str(item.get("noise_paths",""))}
                   for item in batch]

            noisy_t=torch.from_numpy(np.stack(noisys)).float().to(device)
            pred_t=model(noisy_t)
            if pred_t.ndim==1: pred_t=pred_t.unsqueeze(0)
            L=noisy_t.size(-1)
            if pred_t.size(-1)!=L: pred_t=pred_t[:,:L]

            pred_np=pred_t.cpu().numpy(); noisy_np=noisy_t.cpu().numpy()
            clean_np=np.stack(cleans)

            for i,meta in enumerate(metas):
                scale=float(scales[i])
                m=calculate_all_metrics(clean_np[i]*scale, pred_np[i]*scale, noisy_np[i]*scale,
                                        sr=sampling_rate, arv_window=200, f_min=f_min, f_max=f_max,
                                        lsd_n_fft=lsd_n_fft, lsd_hop=lsd_hop)
                m={k:v for k,v in m.items() if k in metrics_list}
                nt=_noise_type_labels(meta["noise_types"],meta["noise_paths"] if has_noise_paths else "")
                collector.add(db=meta["dataset"],snr=meta["snr"],k=meta["k"],metrics=m,noise_type_labels=nt)

    print(f"\n[Done] {collector.n_ok}/{collector.n_total} samples")
    os.makedirs(output_dir,exist_ok=True)
    overall=collector.summary()

    pd.DataFrame([{"metric":m,**st} for m,st in overall.items()]).to_csv(
        os.path.join(output_dir,"overall_summary.csv"),index=False,float_format="%.10g")

    for metric in metrics_list:
        if metric not in overall: continue
        make_snr_k_table(collector,metric=metric).to_csv(
            os.path.join(output_dir,f"table_all_{metric}.csv"),index=False,float_format="%.10g")
        for db in sorted(collector.results):
            make_snr_k_table(collector,metric=metric,db=db).to_csv(
                os.path.join(output_dir,f"table_{db}_{metric}.csv"),index=False,float_format="%.10g")

    nt_df=make_noise_type_table(collector,metrics_list)
    nt_df.to_csv(os.path.join(output_dir,"table_noise_type_all_metrics.csv"),index=False,float_format="%.10g")
    for m in metrics_list:
        mcol=f"{m}_mean"
        if mcol in nt_df.columns:
            nt_df[["noise_type",mcol,f"{m}_std",f"{m}_n"]].rename(
                columns={mcol:"mean",f"{m}_std":"std",f"{m}_n":"n"}
            ).to_csv(os.path.join(output_dir,f"table_noisetype_{m}.csv"),index=False,float_format="%.10g")
        if m not in overall: continue
        make_snr_noisetype_table(collector,m).to_csv(
            os.path.join(output_dir,f"table_snr_x_noisetype_{m}.csv"),index=False,float_format="%.10g")

    W=70
    print(f"\n{'='*W}\nOVERALL METRICS — {model_name}\n{'='*W}")
    for grp,mlist in [("Signal Quality",["SNRimp","RMSE","PRD","LSD"]),
                       ("Feature – Time",["RMSE_ARV","RMSE_ZCR"]),
                       ("Feature – Freq",["RMSE_MNF","RMSE_MDF"]),
                       ("Feature – Stat",["RMSE_Kurtosis"])]:
        print(f"\n  -- {grp} --")
        for m in mlist:
            if m not in metrics_list or m not in overall: continue
            st=overall[m]
            unit=(" dB" if m in("SNRimp","LSD") else " %" if m=="PRD" else
                  " Hz" if m in("RMSE_MNF","RMSE_MDF") else " cross/s" if m=="RMSE_ZCR" else "")
            print(f"    {m:<16s}: {st['mean']:9.4f} +/- {st['std']:7.4f}{unit}  (n={st['n']})")
    print(f"\n✓ Results saved to: {output_dir}\n{'='*W}\n")


def main():
    ap=argparse.ArgumentParser(description="sEMG Baseline Inference v2.0")
    ap.add_argument("--config",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--ckpt",required=True); ap.add_argument("--test-data",required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--metrics",default=",".join(DEFAULT_METRICS)); ap.add_argument("--sr",type=int,default=1000)
    args=ap.parse_args()
    with open(args.config) as f: config=yaml.safe_load(f)
    metrics=[m.strip() for m in args.metrics.split(",") if m.strip()]
    run_inference(config=config,model_name=args.model,model_path=args.ckpt,
                  test_data_path=args.test_data,output_dir=args.output,
                  batch_size=args.batch,metrics_list=metrics,sampling_rate=args.sr)

if __name__=="__main__":
    main()