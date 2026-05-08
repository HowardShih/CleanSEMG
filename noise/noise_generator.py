#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sEMG Noise Generation and Online Mixing Module  v6.8.0

WGN power allocation (same rule as generate_test_data.py v6.8.0):

  Normal case (per-component SNR >= floor):
      All k types receive equal power.

  WGN capping (k>=2, per-component SNR < floor):
      WGN  → fixed at WGN_COMPONENT_SNR_MIN_DB (-5 dB) component power
      Others → share remaining power equally
      Total noise preserved → SNR label correct

  k=1 WGN below floor:
      WGN excluded from pool; another type chosen instead.

Training SNR sampling:
  Total SNR is now sampled from the full range [snr_train_min, snr_train_max]
  regardless of whether WGN is present. The WGN power cap is applied
  during mixing, not at sampling time. This gives the model exposure to
  WGN across all SNR levels (always at >= -5 dB component strength).
"""

import os
import random
import argparse
import json
import csv
from typing import List, Tuple, Dict, Optional
from glob import glob
from fractions import Fraction
from collections import Counter, defaultdict

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, iirnotch
import scipy.io
import wfdb

WGN_COMPONENT_SNR_MIN_DB = -5.0   # must match generate_test_data.py v6.8.0


# ============================================================================
# Reproducibility
# ============================================================================
def seed_everything(seed: int) -> None:
    seed = int(seed) & 0xffffffff
    random.seed(seed); np.random.seed(seed)


# ============================================================================
# Signal processing utilities
# ============================================================================
def apply_bandpass_filter(x, fs, low, high, order=4):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < max(64, 3*order+1): return x
    nyq = fs / 2.0; high = min(high, nyq*0.99)
    if high <= low: return x
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def apply_lowpass_filter(x, fs, cutoff=200.0, order=4):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < max(64, 3*order+1): return x
    nyq = fs / 2.0
    b, a = butter(order, min(cutoff, nyq*0.99)/nyq, btype="low")
    return filtfilt(b, a, x)

def apply_notch_filter(x, fs, freq=50.0, q=30.0):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 128: return x
    b, a = iirnotch(freq, q, fs=fs)
    return filtfilt(b, a, x)

def resample_signal_poly(x, from_fs, to_fs):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0 or from_fs == to_fs: return x
    frac = Fraction(to_fs, from_fs).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float64)

def ensure_length(x, L):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0: return np.zeros(L, dtype=np.float64)
    if x.size >= L: return x[:L]
    return np.tile(x, (L//x.size)+2)[:L]


# ============================================================================
# Config helpers
# ============================================================================
def get_output_base(config):
    root = config["paths"]["root"]; base = config["paths"]["output"]["base"]
    return base if os.path.isabs(base) else os.path.join(root, base)

def _get_bp_params(config):
    bp = config.get("preprocessing",{}).get("bandpass",{})
    return (bool(bp.get("enabled",True)), float(bp.get("low_cutoff",20.0)),
            float(bp.get("high_cutoff",500.0)), int(bp.get("order",4)))

def _get_notch_params(config):
    nc = config.get("preprocessing",{}).get("notch",{})
    return (bool(nc.get("enabled",True)), float(nc.get("freq_hz",50.0)), float(nc.get("q",30.0)))


# ============================================================================
# ECG split
# ============================================================================
def _ecg_split_file(config):
    d = os.path.join(get_output_base(config),"noise"); os.makedirs(d,exist_ok=True)
    return os.path.join(d,"ecg_split.json")

def get_or_create_ecg_test_ids(config, ecg_ids):
    sp=_ecg_split_file(config); cfg=config.get("noise",{}).get("generation",{}).get("ecg",{})
    cnt=int(cfg.get("test_record_count",4)); seed=int(config.get("project",{}).get("random_seed",12345))
    if os.path.exists(sp):
        try:
            obj=json.load(open(sp))
            if obj.get("seed")==seed and obj.get("test_record_count")==cnt:
                test_ids=[int(x) for x in obj["test_ids"]]
                return [r for r in ecg_ids if r not in set(test_ids)], test_ids
        except: pass
    rng=random.Random(seed); test_ids=rng.sample(list(ecg_ids),cnt)
    train_ids=[r for r in ecg_ids if r not in set(test_ids)]
    json.dump({"seed":seed,"test_record_count":cnt,"test_ids":test_ids,"train_ids":train_ids},
              open(sp,"w"),indent=2)
    return train_ids, test_ids


# ============================================================================
# Noise Generators
# ============================================================================
class NoiseGenerator:
    def __init__(self, target_fs, time_length, config):
        self.target_fs=int(target_fs); self.time_length=int(time_length)
        self.L=self.target_fs*self.time_length
        self.bp_enabled,self.bp_low,self.bp_high,self.bp_order=_get_bp_params(config)
    def _bp(self, x):
        if self.bp_enabled:
            return apply_bandpass_filter(x,self.target_fs,self.bp_low,self.bp_high,self.bp_order)
        return np.asarray(x,dtype=np.float64).reshape(-1)

class PLIGenerator(NoiseGenerator):
    def generate(self, count, config):
        c=config.get("noise",{}).get("generation",{}).get("pli",{})
        t=np.arange(self.L,dtype=np.float64)/self.target_fs; out=[]
        for _ in range(count):
            f0=float(random.choice(c.get("fundamental_hz",[50])))
            if c.get("drift_enabled",True): f0+=random.uniform(-float(c.get("drift_range",0.3)),float(c.get("drift_range",0.3)))
            H=int(c.get("harmonics",5)); alpha=float(c.get("amplitude_decay",1.0))
            hs=list(range(1,H+1)) if c.get("all_harmonics",True) else list(range(1,2*H,2))[:H]
            y=sum((1/k**alpha)*np.sin(2*np.pi*k*f0*t+2*np.pi*random.random()) for k in hs)
            if y.std()>0: y/=y.std()
            out.append(ensure_length(self._bp(y),self.L))
        return out

class WGNGenerator(NoiseGenerator):
    def generate(self, count):
        return [ensure_length(self._bp(np.random.normal(0,1,self.L)),self.L) for _ in range(count)]

class ColorNoiseGenerator(NoiseGenerator):
    def generate(self, count, config):
        c=config.get("noise",{}).get("generation",{}).get("color",{})
        types=c.get("types",["pink","brown"]); mode=str(c.get("sample_mode","random")).lower()
        pa=float(c.get("pink_alpha",1.0)); ba=float(c.get("brown_alpha",2.0))
        freqs=np.fft.rfftfreq(self.L,1.0/self.target_fs); freqs[0]=1e-10; out=[]
        for i in range(count):
            nt=types[i%len(types)] if mode=="alternating" else random.choice(types)
            alpha=pa if nt=="pink" else ba
            w=np.random.randn(self.L//2+1)+1j*np.random.randn(self.L//2+1)
            x=np.fft.irfft(w/(freqs**(alpha/2)),n=self.L).real.astype(np.float64)
            if x.std()>0: x/=x.std()
            out.append((ensure_length(self._bp(x),self.L),nt))
        return out

class ECGGenerator(NoiseGenerator):
    ECG_IDS=[16265,16272,16273,16420,16483,16539,16773,16786,16795,
             17052,17453,18177,18184,19088,19090,19093,19140,19830]
    def __init__(self,ecg_root,target_fs,time_length,config):
        super().__init__(target_fs,time_length,config)
        self.ecg_root=ecg_root; self.fs_src=128; self.seg_src_len=int(time_length*128)
        self.notch_en,self.notch_freq,self.notch_q=_get_notch_params(config)
    def _read_128(self,rid,start):
        try:
            rec=wfdb.rdrecord(os.path.join(self.ecg_root,str(rid)),sampfrom=int(start),sampto=int(start+self.seg_src_len))
            p=getattr(rec,"p_signal",None)
            if p is None: return None
            x=p[:,0].astype(np.float64)
            return ensure_length(x,self.seg_src_len) if x.size<self.seg_src_len else x
        except: return None
    def _rlen(self,rid):
        try: return int(wfdb.rdheader(os.path.join(self.ecg_root,str(rid))).sig_len)
        except: return None
    def _proc(self,x128):
        x=resample_signal_poly(x128,self.fs_src,self.target_fs)
        if self.notch_en: x=apply_notch_filter(x,self.target_fs,self.notch_freq,self.notch_q)
        return ensure_length(self._bp(apply_lowpass_filter(x,self.target_fs,200.0)),self.L)
    @staticmethod
    def _ok(x): s=x.std(); return s>1e-6 and np.abs(x).max()>1e-6 and np.isfinite(s)
    def generate_cross_hour(self,ids,total,sph=1,hours=24,seed=None):
        rng=random.Random(int(seed)&0xffffffff) if seed else random.Random()
        plan=[h for h in range(hours) for _ in range(sph)]
        if not plan: plan=[0]
        while len(plan)<total: plan+=plan
        plan=plan[:total]; noises,manifest=[],[]
        for _ in range(total*10):
            if len(noises)>=total: break
            h=plan[len(noises)%len(plan)]; rid=rng.choice(ids); rl=self._rlen(rid)
            if not rl: continue
            hs=h*int(3600*self.fs_src); he=min(rl,(h+1)*int(3600*self.fs_src))
            if he-hs<=self.seg_src_len+1: continue
            s=rng.randint(hs,he-self.seg_src_len); x128=self._read_128(rid,s)
            if x128 is None: continue
            x=self._proc(x128); ok=self._ok(x)
            manifest.append({"index":len(manifest),"record_id":rid,"hour":h,
                              "start_128":s,"flagged":int(not ok),"status":"accepted" if ok else "rejected"})
            if ok: noises.append(x)
        return noises,manifest
    def generate_random(self,ids,total,seed=None):
        rng=random.Random(int(seed)&0xffffffff) if seed else random.Random()
        noises,manifest=[],[]
        for _ in range(total*10):
            if len(noises)>=total: break
            rid=rng.choice(ids); rl=self._rlen(rid)
            if not rl or rl<=self.seg_src_len+1: continue
            s=rng.randint(0,rl-self.seg_src_len); x128=self._read_128(rid,s)
            if x128 is None: continue
            x=self._proc(x128); ok=self._ok(x)
            manifest.append({"index":len(manifest),"record_id":rid,"hour":-1,
                              "start_128":s,"flagged":int(not ok),"status":"accepted" if ok else "rejected"})
            if ok: noises.append(x)
        return noises,manifest
    
def _safe_token(s: str) -> str:
    s = str(s)
    out = []
    for c in s:
        if c.isalnum() or c in ("-", "_"):
            out.append(c)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _infer_moa_source_from_path(path: str) -> str:
    p = str(path).lower()
    if "machado" in p or "balbinot" in p:
        return "Machado"
    if "nstdb" in p or "noise_stress" in p or "mit-bih" in p or "mitbih" in p:
        return "NSTDB"
    return "Unknown"


def _get_moa_specs(src: Dict, mode: str):
    specs = []

    machado_key = f"moa_machado_{mode}"
    nstdb_key = f"moa_nstdb_{mode}"

    if src.get(machado_key):
        specs.append(("Machado", src[machado_key]))

    if src.get(nstdb_key):
        specs.append(("NSTDB", src[nstdb_key]))

    # Backward-compatible fallback
    if not specs:
        old_key = f"moa_{mode}"
        if src.get(old_key):
            specs.append((None, src[old_key]))

    return specs

class MOAGenerator(NoiseGenerator):
    def __init__(self, moa_path, target_fs, time_length, config, source_hint=None):
        super().__init__(target_fs, time_length, config)
        self.moa_path = moa_path
        self.source_hint = source_hint
        self.notch_en, self.notch_freq, self.notch_q = _get_notch_params(config)

    def generate(self):
        out = []

        mat_files = sorted(glob(os.path.join(self.moa_path, "**", "*.mat"), recursive=True))

        for mp in mat_files:
            try:
                m = scipy.io.loadmat(mp)

                x = np.asarray(m["a"]).squeeze() if "a" in m else next(
                    (
                        np.asarray(v).squeeze()
                        for k, v in m.items()
                        if not k.startswith("__")
                        and np.issubdtype(np.asarray(v).dtype, np.number)
                    ),
                    None,
                )

                if x is None or x.size < 256:
                    continue

                x = x.astype(np.float64).reshape(-1)

                if self.notch_en:
                    x = apply_notch_filter(x, 2000, self.notch_freq, self.notch_q)

                x = np.convolve(x, np.ones(51) / 51, mode="valid")[::2]

                if self.target_fs != 1000:
                    x = resample_signal_poly(x, 1000, self.target_fs)

                x = self._bp(x)

                seg = ensure_length(x, self.L) if x.size < self.L else x[random.randint(0, x.size - self.L):][:self.L]

                source = self.source_hint or _infer_moa_source_from_path(mp)
                stem = _safe_token(os.path.splitext(os.path.basename(mp))[0])
                rel_path = os.path.relpath(mp, self.moa_path)

                out.append({
                    "signal": seg.copy(),
                    "source": source,
                    "stem": stem,
                    "source_file": rel_path,
                })

            except Exception as e:
                print(f"[WARN] MOA {mp}: {e}")

        return out


# ============================================================================
# Build noise library
# ============================================================================
def _save_csv(path,rows):
    if not rows: return
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

def build_noise_library(config,mode):
    is_train=(mode=="train"); out_base=get_output_base(config)
    noise_dir=os.path.join(out_base,config["paths"]["output"][f"noise_{mode}"])
    os.makedirs(noise_dir,exist_ok=True)
    seed=int(config.get("project",{}).get("random_seed",12345))+(0 if is_train else 999)
    seed_everything(seed)
    src=config["paths"]["noise_sources"]
    fs=int(config["noise"]["generation"].get("target_fs",config["preprocessing"]["segmentation"]["target_fs"]))
    tlen=int(config["noise"]["generation"]["time_length"])
    counts=config["noise"]["generation"][f"{mode}_count"]
    bp_en,bp_lo,bp_hi,_=_get_bp_params(config)
    nt_en,nt_fr,nt_q=_get_notch_params(config)

    print(f"\n{'='*80}\nBuild Noise Library ({mode}) — v6.8.0\n{'='*80}")
    print(f"WGN floor: {WGN_COMPONENT_SNR_MIN_DB} dB component SNR (capping applied in mix)")

    def savedir(n):
        d=os.path.join(noise_dir,n); os.makedirs(d,exist_ok=True); return d

    print("\n[1/5] PLI")
    g=PLIGenerator(fs,tlen,config); lst=g.generate(int(counts.get("PLI",0)),config)
    d=savedir("PLI")
    for i,x in enumerate(lst): np.save(os.path.join(d,f"PLI_{i}.npy"),x.astype(np.float32))
    print(f"  Saved {len(lst)}")

    print("\n[2/5] WGN")
    g=WGNGenerator(fs,tlen,config); lst=g.generate(int(counts.get("WGN",0)))
    d=savedir("WGN")
    for i,x in enumerate(lst): np.save(os.path.join(d,f"WGN_{i}.npy"),x.astype(np.float32))
    print(f"  Saved {len(lst)}")

    print("\n[3/5] Color (pink/brown)")
    g=ColorNoiseGenerator(fs,tlen,config); results=g.generate(int(counts.get("Color",0)),config)
    d=savedir("Color"); pc=bc=0
    for i,(x,ct) in enumerate(results):
        np.save(os.path.join(d,f"Color_{ct}_{i}.npy"),x.astype(np.float32))
        if ct=="pink": pc+=1
        else: bc+=1
    print(f"  Saved {len(results)} (pink={pc}, brown={bc})")

    print("\n[4/5] ECG")
    d=savedir("ECG")
    if os.path.exists(src["ecg"]):
        ecg=ECGGenerator(src["ecg"],fs,tlen,config)
        train_ids,test_ids=get_or_create_ecg_test_ids(config,ecg.ECG_IDS)
        ids=train_ids if is_train else test_ids
        ec=config.get("noise",{}).get("generation",{}).get("ecg",{})
        hourly=bool(ec.get("hourly_sampling",True)) and is_train
        segs=int(ec.get("train_segments" if is_train else "test_segments",24 if is_train else 4))
        if hourly:
            lst,man=ecg.generate_cross_hour(ids,segs,int(ec.get("segments_per_hour",1)),int(ec.get("total_hours",24)),seed)
        else:
            lst,man=ecg.generate_random(ids,segs,seed)
        for i,x in enumerate(lst): np.save(os.path.join(d,f"ECG_{i}.npy"),x.astype(np.float32))
        _save_csv(os.path.join(noise_dir,f"ECG_manifest_{mode}.csv"),man)
        print(f"  Saved {len(lst)} (hourly={hourly})")
    else:
        print(f"  [SKIP] {src['ecg']}")

    print("\n[5/5] MOA")
    d = savedir("MOA")
    moa_specs = _get_moa_specs(src, mode)

    all_moa_items = []
    moa_manifest = []

    if moa_specs:
        for source_hint, mp in moa_specs:
            if not mp or not os.path.exists(mp):
                print(f"  [SKIP] {source_hint or 'MOA'}: {mp}")
                continue

            items = MOAGenerator(
                mp,
                fs,
                tlen,
                config,
                source_hint=source_hint,
            ).generate()

            all_moa_items.extend(items)

        for i, item in enumerate(all_moa_items):
            source = item.get("source", "Unknown")
            stem = item.get("stem", f"{i}")
            signal = item["signal"]

            fname = f"MOA_{_safe_token(source)}_{i:03d}_{_safe_token(stem)}.npy"
            out_path = os.path.join(d, fname)

            np.save(out_path, signal.astype(np.float32))

            moa_manifest.append({
                "index": i,
                "file": fname,
                "source": source,
                "source_file": item.get("source_file", ""),
            })

        _save_csv(os.path.join(noise_dir, f"MOA_manifest_{mode}.csv"), moa_manifest)
        print(f"  Saved {len(all_moa_items)}")
    else:
        print("  [SKIP] no MOA source configured")

    print(f"\n{'='*80}\nSummary ({mode})\n{'='*80}")
    for nt in ["PLI","WGN","Color","ECG","MOA"]:
        nd=os.path.join(noise_dir,nt)
        if os.path.isdir(nd):
            nf=len(glob(os.path.join(nd,"*.npy")))
            if nt=="Color":
                np_=len(glob(os.path.join(nd,"*_pink_*.npy"))); nb=len(glob(os.path.join(nd,"*_brown_*.npy")))
                print(f"  {nt}: {nf} (pink={np_}, brown={nb})")
            else: print(f"  {nt}: {nf}")


# ============================================================================
# WGN power allocation (shared logic with generate_test_data.py)
# ============================================================================
def _compute_noise_targets(clean_pow: float, total_snr: float,
                            selected: List[str]) -> Dict[str, float]:
    """
    Compute target noise power for each selected noise type.

    WGN rule:
      - Equal split if WGN's equal-share component SNR is >= floor.
      - Cap WGN only if equal-share component SNR would be below floor.
      - k=1 WGN below floor is handled upstream by excluding WGN.
    """
    k = len(selected)
    total_noise = clean_pow / (10.0 ** (total_snr / 10.0))

    if k <= 0:
        return {}

    if "WGN" not in selected or k <= 1:
        return {nt: total_noise / k for nt in selected}

    # WGN present, k >= 2.
    # This is the maximum allowed WGN power corresponding to the -5 dB floor.
    wgn_equal_pow = total_noise / k
    wgn_floor_pow = clean_pow / (10.0 ** (WGN_COMPONENT_SNR_MIN_DB / 10.0))

    # If equal split already keeps WGN at or above the floor, do not cap.
    # In power terms: component SNR >= floor  <=>  WGN power <= floor power.
    if wgn_equal_pow <= wgn_floor_pow:
        return {nt: total_noise / k for nt in selected}

    # Otherwise, WGN would be too strong, so cap WGN and redistribute the rest.
    remaining = total_noise - wgn_floor_pow

    # Numerical guard. This should not be negative after the condition above.
    if remaining < 0:
        remaining = 0.0

    per_other = remaining / (k - 1)

    return {
        nt: (wgn_floor_pow if nt == "WGN" else per_other)
        for nt in selected
    }


# ============================================================================
# Online Noise Mixer
# ============================================================================
class OnlineNoiseMixer:
    """
    Online noise mixer for training (v6.8.0).

    Training SNR is sampled from the full range [snr_train_min, snr_train_max].
    WGN power capping is applied in mix() — not at sampling time.
    This ensures WGN's component SNR is always >= WGN_COMPONENT_SNR_MIN_DB,
    consistent with the test data policy in generate_test_data.py v6.8.0.
    """

    NOISE_TYPES = ["PLI", "ECG", "MOA", "WGN", "Color"]

    def __init__(self, noise_root, config=None, noise_types=None,
                 cache_noise=True, seed=None):
        self.noise_root  = noise_root
        self.cfg         = config or {}
        self.noise_types = noise_types or self.cfg.get("noise",{}).get("types",self.NOISE_TYPES)
        self.cache_noise = cache_noise
        self.noise_paths: Dict[str,List[str]] = {}
        self.noise_cache: Dict[str,np.ndarray] = {}

        ncfg=self.cfg.get("noise",{})
        self.k_min         = int(ncfg.get("k_types",{}).get("min",1))
        self.k_max         = int(ncfg.get("k_types",{}).get("max",5))
        self.snr_train_min = float(ncfg.get("snr_train",{}).get("min",-15.0))
        self.snr_train_max = float(ncfg.get("snr_train",{}).get("max",15.0))
        self.snr_dist      = str(ncfg.get("snr_train",{}).get("distribution","uniform")).lower()
        # snr_test_grid: unified grid (no separate wgn grid needed)
        self.snr_test_grid = list(ncfg.get("snr_test",{}).get("grid",[-15,-10,-5,0,5,10,15]))

        self.stats = {"total_mixed":0,"noise_type_counts":Counter(),
                      "k_distribution":Counter(),"snr_values":defaultdict(list)}
        self.base_seed = int(seed)&0xffffffff if seed is not None else None
        if seed is not None:
            self.rng=random.Random(seed); self.rng_np=np.random.default_rng(seed)
        else:
            self.rng=random.Random(); self.rng_np=np.random.default_rng()

        self._load_paths()
        if cache_noise: self._cache_all()

    def _load_paths(self):
        for nt in self.noise_types:
            d=os.path.join(self.noise_root,nt)
            ps=sorted(glob(os.path.join(d,"*.npy"))) if os.path.isdir(d) else []
            if ps: self.noise_paths[nt]=ps
        if not self.noise_paths: raise ValueError(f"No noise under {self.noise_root}")

    def _cache_all(self):
        print("[OnlineNoiseMixer] Caching …")
        for ps in self.noise_paths.values():
            for p in ps:
                if p not in self.noise_cache: self.noise_cache[p]=np.load(p).astype(np.float64)
        print(f"[OnlineNoiseMixer] {len(self.noise_cache)} files")
        for nt,ps in self.noise_paths.items():
            if nt=="Color":
                np_=sum(1 for p in ps if "_pink_" in os.path.basename(p))
                nb =sum(1 for p in ps if "_brown_" in os.path.basename(p))
                print(f"  {nt}: {len(ps)} (pink={np_}, brown={nb})")
            else: print(f"  {nt}: {len(ps)}")
        print(f"  WGN floor: {WGN_COMPONENT_SNR_MIN_DB} dB component (capped, not excluded)")

    def _get(self, path):
        return self.noise_cache.get(path) or np.load(path).astype(np.float64)

    @staticmethod
    def _seg(x, L, rng):
        x=np.asarray(x,dtype=np.float64).reshape(-1)
        if x.size<L: return ensure_length(x,L).copy()
        s=rng.randint(0,x.size-L); return x[s:s+L].copy()

    def mix(self, clean, k=None, snr=None, noise_types=None,
            mode="train", seed=None):
        if seed is not None:
            su=int(seed)&0xffffffff; rng=random.Random(su); rnp=np.random.default_rng(su)
        else:
            rng,rnp,su=self.rng,self.rng_np,self.base_seed

        clean=np.asarray(clean,dtype=np.float64).reshape(-1); L=clean.size
        if L==0:
            return clean, {"snr":None,"k":0,"noise_types":[],"scalars":[],"noise_paths":[],"has_wgn":False,"wgn_capped":False}

        available=list(self.noise_paths.keys())
        if not available:
            return clean, {"snr":None,"k":0,"noise_types":[],"scalars":[],"noise_paths":[],"has_wgn":False,"wgn_capped":False}

        max_k=min(self.k_max,len(available)); min_k=max(1,min(self.k_min,max_k))
        k = rng.randint(min_k,max_k) if k is None else max(min_k,min(int(k),max_k))

        # Sample SNR from full range (no WGN-specific floor — capping handles it)
        if mode=="train":
            snr_val=(float(rnp.uniform(self.snr_train_min,self.snr_train_max))
                     if snr is None else float(snr))
            snr_val=float(np.clip(snr_val,self.snr_train_min,self.snr_train_max))
        else:
            snr_val=(float(rng.choice(self.snr_test_grid))
                     if snr is None else float(snr))

        # For k=1, remove WGN from pool if it would be below floor
        avail_k = list(available)
        if k == 1 and "WGN" in avail_k and snr_val < WGN_COMPONENT_SNR_MIN_DB:
            avail_k = [t for t in avail_k if t != "WGN"]

        k = min(k, len(avail_k))

        if noise_types is not None:
            selected=[t for t in noise_types if t in avail_k]
            rest=[t for t in avail_k if t not in selected]
            if len(selected)<k: selected+=rng.sample(rest,min(k-len(selected),len(rest)))
            selected=selected[:k]
        else:
            selected=rng.sample(avail_k, k)

        has_wgn = "WGN" in selected
        clean_pow = float(np.dot(clean, clean))

        # Compute per-type power targets with WGN capping
        targets = _compute_noise_targets(clean_pow, snr_val, selected)
        wgn_capped = (
            has_wgn and k >= 2 and
            (snr_val + 10.0 * np.log10(k)) < WGN_COMPONENT_SNR_MIN_DB
        )

        if clean_pow < 1e-12:
            return clean.copy(), {
                "snr":snr_val,"k":k,"noise_types":selected,
                "scalars":[0.0]*k,"noise_paths":[],"has_wgn":has_wgn,"wgn_capped":wgn_capped
            }

        combined=np.zeros(L,dtype=np.float64); scalars=[]; used_paths=[]
        for nt in selected:
            p=rng.choice(self.noise_paths[nt]); used_paths.append(p)
            nseg=self._seg(self._get(p),L,rng)
            npow=float(np.dot(nseg,nseg))
            s=float(np.sqrt(targets[nt]/npow)) if npow>1e-12 else 0.0
            scalars.append(s); combined+=s*nseg

        self.stats["total_mixed"]+=1
        self.stats["k_distribution"][k]+=1
        self.stats["snr_values"][mode].append(snr_val)
        for nt in selected: self.stats["noise_type_counts"][nt]+=1

        return clean+combined, {
            "snr":snr_val,"k":k,"noise_types":selected,"scalars":scalars,
            "noise_paths":used_paths,"mode":mode,"has_wgn":has_wgn,
            "wgn_capped":wgn_capped,"seed":su,
        }

    def get_statistics(self):
        out={"total_mixed":self.stats["total_mixed"],
             "noise_type_distribution":dict(self.stats["noise_type_counts"]),
             "k_distribution":dict(self.stats["k_distribution"]),"snr_statistics":{}}
        for mode,lst in self.stats["snr_values"].items():
            if lst:
                out["snr_statistics"][mode]={
                    "count":len(lst),"min":float(np.min(lst)),"max":float(np.max(lst)),
                    "mean":float(np.mean(lst)),"std":float(np.std(lst))}
        return out

    def reset_statistics(self):
        self.stats={"total_mixed":0,"noise_type_counts":Counter(),
                    "k_distribution":Counter(),"snr_values":defaultdict(list)}


class OnlineMixingDataset:
    def __init__(self, clean_data, noise_root, config=None, mode="train", transform=None, seed=None):
        self.clean_data=clean_data; self.mode=mode; self.transform=transform
        self.mixer=OnlineNoiseMixer(noise_root,config,cache_noise=True,seed=seed)
    def __len__(self): return len(self.clean_data)
    def __getitem__(self, idx):
        clean=self.clean_data[idx]
        cn=clean.numpy() if hasattr(clean,"numpy") else np.asarray(clean)
        nn,_=self.mixer.mix(cn,mode=self.mode)
        if self.transform: return self.transform(nn),self.transform(cn)
        try:
            import torch
            return torch.as_tensor(nn,dtype=torch.float32),torch.as_tensor(cn,dtype=torch.float32)
        except ImportError: return nn,cn


# ============================================================================
# CLI
# ============================================================================
def main():
    parser=argparse.ArgumentParser(description="sEMG Noise v6.8.0")
    parser.add_argument("--config",default="config.yaml")
    parser.add_argument("--mode",choices=["train","test","both"],default="both")
    parser.add_argument("--demo",action="store_true")
    args=parser.parse_args()

    import yaml
    with open(args.config) as f: config=yaml.safe_load(f)

    if args.demo:
        noise_root=os.path.join(get_output_base(config),config["paths"]["output"]["noise_train"])
        if not os.path.isdir(noise_root):
            print("[ERROR] Run: python noise.py --mode both"); return
        seed=int(config.get("project",{}).get("random_seed",12345))
        mixer=OnlineNoiseMixer(noise_root,config,cache_noise=True,seed=seed)
        clean=np.sin(2*np.pi*100*np.linspace(0,2,2000,endpoint=False))

        print("\n[Training demo — SNR sampled from full range, WGN capped if needed]")
        for i in range(4):
            _,info=mixer.mix(clean,mode="train",seed=seed+i)
            comp=(info["snr"]+10*np.log10(info["k"])) if info["has_wgn"] else None
            cap="*CAPPED*" if info.get("wgn_capped") else ""
            wgn_note=f"WGN comp={comp:+.1f}dB {cap}" if comp is not None else "no WGN"
            print(f"  SNR={info['snr']:+.1f}dB  k={info['k']}  {wgn_note}  types={info['noise_types']}")

        print("\n[k=5 at low SNR (capping demo)]")
        for snr in [-15,-10,-5]:
            _,info=mixer.mix(clean,mode="test",snr=snr,k=5,seed=seed+snr)
            comp=snr+10*np.log10(5)
            cap="→ WGN capped at -5dB" if info.get("wgn_capped") else "→ equal split"
            print(f"  total={snr:+d}dB  WGN component={comp:+.1f}dB  {cap}")

        print("\n[Stats]"); print(json.dumps(mixer.get_statistics(),indent=2))
        return

    if args.mode in ("train","both"): build_noise_library(config,"train")
    if args.mode in ("test","both"):  build_noise_library(config,"test")
    print("\n✓ Done")


if __name__ == "__main__":
    main()