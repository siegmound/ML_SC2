from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss

def load_df(path: Path):
    with zipfile.ZipFile(path) as zf:
        name=[n for n in zf.namelist() if n.endswith('.csv')][0]
        with zf.open(name) as fh:
            return pd.read_csv(fh)

def split(df, seed):
    X=df.drop(columns=['p1_wins','replay_id','time_sec'])
    y=df['p1_wins'].astype(int)
    g=df['replay_id'].astype(str)
    outer=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=seed)
    tv_idx,test_idx=next(outer.split(X,y,groups=g))
    X_tv,y_tv,g_tv=X.iloc[tv_idx],y.iloc[tv_idx],g.iloc[tv_idx]
    X_test,y_test,g_test=X.iloc[test_idx],y.iloc[test_idx],g.iloc[test_idx]
    inner=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=seed)
    tr_idx,val_idx=next(inner.split(X_tv,y_tv,groups=g_tv))
    return X_tv.iloc[tr_idx],y_tv.iloc[tr_idx],X_tv.iloc[val_idx],y_tv.iloc[val_idx],X_test,y_test,pd.concat([g_tv.iloc[tr_idx],g_tv.iloc[val_idx]]).nunique(),g_test.nunique()

def metrics(y_true, prob):
    pred=(np.asarray(prob)>=0.5).astype(int)
    return dict(accuracy=float(accuracy_score(y_true,pred)), balanced_accuracy=float(balanced_accuracy_score(y_true,pred)), roc_auc=float(roc_auc_score(y_true,prob)), log_loss=float(log_loss(y_true,prob,labels=[0,1])))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', type=Path, required=True)
    ap.add_argument('--dataset-name', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--model', choices=['logreg','rf','xgb'], required=True)
    ap.add_argument('--output-dir', type=Path, default=Path('/mnt/data/ml_sc2_remake/results/block9_multiseed'))
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df=load_df(args.dataset_zip)
    rows=[]
    for seed in args.seeds:
        X_train,y_train,X_val,y_val,X_test,y_test,n_train_rep,n_test_rep=split(df,seed)
        X_train_full=pd.concat([X_train,X_val])
        y_train_full=pd.concat([y_train,y_val])
        if args.model=='logreg':
            model=Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(max_iter=500,class_weight='balanced',random_state=seed))])
            model.fit(X_train_full,y_train_full)
            prob=model.predict_proba(X_test)[:,1]
            extra={}
        elif args.model=='rf':
            model=RandomForestClassifier(n_estimators=20,max_depth=8,min_samples_split=5,min_samples_leaf=2,max_features='sqrt',random_state=seed,n_jobs=-1)
            model.fit(X_train_full,y_train_full)
            prob=model.predict_proba(X_test)[:,1]
            extra={}
        else:
            from xgboost import XGBClassifier
            model=XGBClassifier(objective='binary:logistic',eval_metric='logloss',tree_method='hist',device='cpu',random_state=seed,n_estimators=120,learning_rate=0.05,max_depth=5,min_child_weight=3,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,early_stopping_rounds=15,n_jobs=4)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
            prob=model.predict_proba(X_test)[:,1]
            extra={'best_iteration': int(getattr(model,'best_iteration',-1))}
        row=dict(model_name=args.model, dataset_name=args.dataset_name, seed=seed, n_rows=int(len(df)), n_replays=int(df['replay_id'].astype(str).nunique()), n_train_rows=int(len(X_train_full)), n_test_rows=int(len(X_test)), n_train_replays=int(n_train_rep), n_test_replays=int(n_test_rep), **extra, **metrics(y_test,prob))
        rows.append(row)
        print(json.dumps(row))
    det=pd.DataFrame(rows)
    out=args.output_dir/f'{args.dataset_name}_{args.model}_multiseed_detailed.csv'
    det.to_csv(out,index=False)
    agg=det.agg({'accuracy':['mean','std'],'balanced_accuracy':['mean','std'],'roc_auc':['mean','std'],'log_loss':['mean','std']}).to_dict()
    (args.output_dir/f'{args.dataset_name}_{args.model}_multiseed_summary.json').write_text(json.dumps({'dataset_name':args.dataset_name,'model_name':args.model,'seeds':args.seeds,'aggregate':agg,'detailed_csv':str(out)}, indent=2), encoding='utf-8')
if __name__=='__main__':
    main()
