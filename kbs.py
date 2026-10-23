"""Reproducible knowledge-constrained innovation decision support prototype.

All default data are SYNTHETIC for software verification, not scientific evidence.
CSV schema: firm_id, year, analytics, knowledge, digital, budget, risk,
initiative, cost, min_digital, future_success. One row per firm-year-initiative.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ['analytics', 'knowledge', 'digital', 'budget', 'risk', 'cost', 'min_digital']
CAT = ['initiative']
REQUIRED = {'firm_id', 'year', 'future_success', *NUM, *CAT}


def synthetic(n_firms=300, seed=31):
    rng = np.random.default_rng(seed)
    rows = []
    initiatives = [('customer_ai', 0.22, 0.15), ('process_automation', 0.42, 0.35), ('digital_platform', 0.68, 0.55)]
    for firm in range(n_firms):
        analytics, knowledge, risk = rng.uniform(0, 1, 3)
        budget = rng.uniform(.25, 1)
        for year in range(2021, 2026):
            digital = np.clip(.2 + .3*analytics + .25*knowledge + .035*(year-2021) + rng.normal(0,.09), 0, 1)
            for kind, cost, minimum in initiatives:
                logit = -1.8 + 1.0*analytics + .9*knowledge + 1.2*digital - 1.1*risk - .4*cost + .2*(year-2021)
                prob = 1/(1+np.exp(-logit))
                rows.append(dict(firm_id=f'F{firm:04d}', year=year, analytics=analytics, knowledge=knowledge,
                    digital=digital, budget=budget, risk=risk, initiative=kind, cost=cost,
                    min_digital=minimum, future_success=int(rng.random()<prob)))
    return pd.DataFrame(rows)


def validate(df):
    missing = REQUIRED - set(df)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df[list(NUM)].isna().all(axis=None): raise ValueError('All numeric values missing')
    if not set(df.future_success.dropna().unique()).issubset({0,1}): raise ValueError('Target must be binary')
    if not df[['firm_id','year','initiative']].duplicated().sum()==0: raise ValueError('Duplicate firm/year/initiative')
    if (df['year'] < 1900).any(): raise ValueError('Invalid year')
    return df.copy()


def model():
    prep = ColumnTransformer([
        ('numeric',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),
        ('category',OneHotEncoder(handle_unknown='ignore'),CAT)])
    return Pipeline([('prep',prep),('classifier',LogisticRegression(max_iter=1500,random_state=31))])


def evaluate(df):
    df = validate(df)
    years = sorted(df.year.unique())
    if len(years)<3: raise ValueError('Need at least three distinct years for temporal split')
    train = df[df.year.isin(years[:-2])]; test = df[df.year==years[-1]]
    if train.future_success.nunique()<2: raise ValueError('Training split needs two classes')
    m = model(); m.fit(train[NUM+CAT],train.future_success)
    p = m.predict_proba(test[NUM+CAT])[:,1]
    def stats(y, score):
        return {'n':len(y),'roc_auc':round(float(roc_auc_score(y,score)),4) if len(np.unique(y))==2 else None,
            'average_precision':round(float(average_precision_score(y,score)),4),
            'brier':round(float(brier_score_loss(y,score)),4)}
    base = np.full(len(test),float(train.future_success.mean()))
    result={'split': {'train_years':[int(y) for y in years[:-2]], 'validation_year':int(years[-2]), 'test_year':int(years[-1])},
        'learned_model':stats(test.future_success,p), 'prevalence_baseline':stats(test.future_success,base)}
    choices=test[['firm_id','year','initiative','budget','cost','digital','min_digital','risk']].copy()
    choices['estimated_probability']=p
    choices['feasible']=(choices.cost<=choices.budget)&(choices.digital>=choices.min_digital)
    # NOT a causal ROI estimate. Score is a stated decision heuristic, not a fitted payoff model.
    choices['score']=.7*choices.estimated_probability-.2*choices.cost-.1*choices.risk
    options=choices[choices.feasible].sort_values(['firm_id','score','initiative'],ascending=[True,False,True])
    selections=options.drop_duplicates('firm_id')
    result['recommendations']={'eligible_firms':int(choices.firm_id.nunique()),'firms_with_feasible_choice':int(selections.firm_id.nunique()),
        'mean_selected_score':round(float(selections.score.mean()),4) if len(selections) else None}
    return result, choices, selections


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', help='Real research CSV; omit for synthetic smoke-test')
    parser.add_argument('--output',default='outputs')
    args=parser.parse_args()
    df=pd.read_csv(args.csv) if args.csv else synthetic()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    results,choices,recommendations=evaluate(df)
    results['data_provenance']='USER-SUPPLIED / NOT VERIFIED' if args.csv else 'SYNTHETIC DEMONSTRATION — NOT RESEARCH RESULTS'
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    recommendations.to_csv(out/'recommendations.csv',index=False)
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
