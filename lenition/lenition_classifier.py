"""
lenition_classifier.py
======================
Can acoustics alone predict a hand-coded "full lenition" label for medial retroflex
syllables (zh/ch/sh), across speakers?

Train on one hand-annotated speaker, test on the other (both directions), using only
acoustic features measured at the syllable onset: zero-crossing rate, high-frequency energy
share (5-8 kHz), spectral centre of gravity, spectral tilt, intensity, and harmonics-to-noise
ratio (precomputed with Praat). Two feature sets are compared: boundaries from the alignment
pipeline (with the boundary corrector) vs. hand boundaries (an upper bound). Also prints a
learning curve: AUC vs. number of hand-labelled training tokens.

Usage:  python lenition_classifier.py ROOT     (ROOT defaults to the current directory)

Expects, under ROOT (not included in this repo; they come from participant recordings):
  token_tables/corrector_predictions_O4O5.csv   one row per token with pipeline and hand boundaries
  token_tables/speaker89_sheet.csv              hand-coded labels and measures
  recording and scripts/recording/S{4,5}.WAV    the two recordings
  context_mfa/praat_S{4,5}.npz                  Praat pitch/HNR frames (array H = HNR at 100 frames/s)
"""
import sys
import numpy as np, pandas as pd, librosa, scipy.signal as ss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
R=sys.argv[1] if len(sys.argv)>1 else '.'
P=pd.read_csv(f'{R}/token_tables/corrector_predictions_O4O5.csv')
X=pd.read_csv(f'{R}/token_tables/speaker89_sheet.csv')
P=P[P.position=='medial'].reset_index(drop=True)
rec2n={'O4':4,'O5':5}
audio={}; praat={}
for rec,n in rec2n.items():
    y,_=librosa.load(f'{R}/recording and scripts/recording/S{n}.WAV',sr=16000); audio[rec]=y.astype('float32'); z=np.load(f'{R}/context_mfa/praat_S{n}.npz'); praat[rec]=z
def feats(rec,s,e,tag):
    y=audio[rec]; z=praat[rec]; d={}
    for name,(a,b) in {'on':(s,s+0.04),'mid':(s+0.04,s+0.09),'nuc':(e-0.06,e-0.02)}.items():
        seg=y[int(a*16000):int(b*16000)]
        if len(seg)<200: 
            for k in ('zcr','hf','cog','tilt','db','hnr'): d[f'{tag}{name}_{k}']=np.nan
            continue
        seg=seg*np.hanning(len(seg)); f,pxx=ss.periodogram(seg,16000,nfft=1024); pxx=pxx+1e-12
        d[f'{tag}{name}_zcr']=float(np.mean(np.abs(np.diff(np.sign(seg)))>0)); d[f'{tag}{name}_hf']=float(pxx[(f>=5000)&(f<8000)].sum()/pxx.sum())
        d[f'{tag}{name}_cog']=float((f*pxx).sum()/pxx.sum()); d[f'{tag}{name}_tilt']=float(10*np.log10(pxx[(f>=3000)&(f<8000)].sum()/pxx[(f>=100)&(f<1000)].sum()))
        d[f'{tag}{name}_db']=float(10*np.log10((seg**2).mean()+1e-12)); i0,i1=int(a*100),int(b*100); d[f'{tag}{name}_hnr']=float(np.nanmean(z['H'][i0:i1])) if i1>i0 else np.nan
    return d
rows=[]
for i,r in P.iterrows():
    sc=r.s_p+r.pdelta_s/1000; ec=r.e_p+r.pdelta_e/1000; sh=r.s_p+r.delta_s/1000; eh=r.e_p+r.delta_e/1000
    row=dict(rec=r.rec,lenited=r.lenited,retroflex=r.retroflex,cond=r.cond,dur_corr=1000*(ec-sc))
    row.update(feats(r.rec,sc,ec,'c_')); row.update(feats(r.rec,sh,eh,'h_')); rows.append(row)
D=pd.DataFrame(rows); D['onset_minus_nuc_db_c']=D.c_on_db-D.c_nuc_db; D['onset_minus_nuc_db_h']=D.h_on_db-D.h_nuc_db
D.to_csv(f'{R}/token_tables/lenition_features_medial_O4O5.csv',index=False)
C=[c for c in D.columns if c.startswith('c_')]+['onset_minus_nuc_db_c']; H=[c for c in D.columns if c.startswith('h_')]+['onset_minus_nuc_db_h']
D['ret_ch']=(D.retroflex=='ch').astype(int); D['ret_sh']=(D.retroflex=='sh').astype(int)
ctx=['ret_ch','ret_sh','cond']
def model(kind): return make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000)) if kind=='lr' else HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=0.05,random_state=0)
def fit_eval(feats,kind,ntrain=None,seed=0):
    out=[]
    for tr,te in (('O4','O5'),('O5','O4')):
        a=D[D.rec==tr]; b=D[D.rec==te]
        if ntrain: a=a.sample(ntrain,random_state=seed)
        m=model(kind); Xa=a[feats].fillna(a[feats].median()); m.fit(Xa,a.lenited); p=m.predict_proba(b[feats].fillna(a[feats].median()))[:,1]
        out.append((roc_auc_score(b.lenited,p),balanced_accuracy_score(b.lenited,p>0.5)))
    return np.mean(out,axis=0)
print(len(D),'medial tokens; lenited share',round(D.lenited.mean(),2))
print('\nPredict the hand-coded "full lenition" label from acoustics only, trained on one speaker, tested on the other (medial tokens):')
for nm,f in (('acoustics at the PIPELINE-CORRECTED target boundaries',C+ctx),('acoustics at HAND boundaries (upper bound)',H+ctx)):
    for kind in ('lr','gb'):
        a,b=fit_eval(f,kind); print(f'  {nm:55s} {kind}:  AUC {a:.2f}   balanced accuracy {b:.2f}')
print('\nLearning curve (logistic, corrected-boundary features): AUC on the other speaker vs number of hand-labelled training tokens (mean of 20 draws):')
for n in (20,40,80,160,320,460):
    v=[fit_eval(C+ctx,'lr',ntrain=n,seed=s)[0] for s in range(20)]; print(f'  n={n:3d}  AUC {np.mean(v):.3f}  (sd {np.std(v):.3f})')
