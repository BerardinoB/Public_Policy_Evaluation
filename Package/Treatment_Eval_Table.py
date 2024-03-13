import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_res = pd.read_csv('./Results/Result_preds_Final.csv')
model_list = ['LGB','XGB','CatBoost','RF','SVM','RF_Survival','Cox','AAF','Weibull_AFT','Lasso','Ridge','GBSurvival','SVM_Survival']

dict_res = {}
for model in model_list:
    dict_res[model] = []
    if model in ['AAF','Weibull_AFT','GBSurvival']:
        for perc in np.percentile(df_res['yhat_'+model],q=list(range(0,110,10))[::-1]):
            dict_res[model].append(df_res.loc[-df_res['yhat_'+model]<=-perc,'Treatment'].mean())
    else:
        for perc in np.percentile(df_res['yhat_'+model],q=range(0,110,10)):
            dict_res[model].append(df_res.loc[df_res['yhat_'+model]<=perc,'Treatment'].mean())
            
def _BNAUT_(return_name=False):
    if return_name:
        return 'Best Normalizer Area Under Treatment'
    j = 500
    list_best_treat = [0]
    sort_treat = np.array(sorted(df_res['Treatment'],reverse=True))
    for q in range(500,df_res['Treatment'].shape[0],500):
        list_best_treat.append(sort_treat[:j].mean())
        j=q
    list_best_treat.append(df_res['Treatment'].mean())
    return list_best_treat

def _RNAUT_(x,y,return_name=False):
    assert len(x)==len(y)==2
    if return_name:
        return 'Random Normalizer Area Under Treatment'
    slope = (y[1]-y[0])/(x[1]-x[0])
    return [slope*x for x in np.linspace(x[0],x[1],100)]

def _PNAUT_(x,return_name=False):
    if return_name:
        return 'Proportional Normalizer Area Under Treatment'
    return [0]+[x]*100

def normalizer(name,kargs):
    if kargs==None:
        kargs = {'return_name':False}
    dict_norm = {'BNAUT':_BNAUT_,'RNAUT':_RNAUT_,'PNAUT':_PNAUT_}
    assert name in dict_norm.keys(), 'name should be one of the following: {}'.format(dict_norm.keys())
    return dict_norm[name](**kargs)

def trapezoid_role(f,f_opt,kargs):
    dict_norm = {'BNAUT':normalizer,'RNAUT':normalizer,'PNAUT':normalizer}
    if isinstance(f_opt,str): assert f_opt in dict_norm.keys(), 'name should be one of the following: {}'.format(dict_norm.keys())
    f_base = dict_norm[f_opt](f_opt,kargs)
    
    a = 0
    b = 100
    n = len(range(0,110,10))
    h = (b - a) / (n - 1)
    I_trap_model= (h/2)*(f[0] + 2 * sum(f[1:n-1]) + f[n-1])

    I_trap_best = (h/2)*(f_base[0] + 2 * sum(f_base[1:n-1]) + f_base[n-1])
    return I_trap_model/I_trap_best


kargs = {'BNAUT':None,'RNAUT':{'x':[0,100],'y':[0,0.177]},'PNAUT':{'x':0.177}}
dict_tab = {'Model':[],'Lower 20%':[],'Upper 20%':[],'BNAUT':[],'RNAUT':[],'PNAUT':[]}
for model in model_list:
    dict_tab['Model'].append(model)
    if model in ['AAF','Weibull_AFT','GBSurvival']:
        perc = np.percentile(df_res['yhat_'+model],q=20)
        dict_tab['Upper 20%'].append(df_res.loc[df_res['yhat_'+model]<=perc,'Treatment'].mean())
        perc = np.percentile(df_res['yhat_'+model],q=80)
        dict_tab['Lower 20%'].append(df_res.loc[df_res['yhat_'+model]>=perc,'Treatment'].mean())
    else:
        perc = np.percentile(df_res['yhat_'+model],q=20)
        dict_tab['Lower 20%'].append(df_res.loc[df_res['yhat_'+model]<=perc,'Treatment'].mean())
        perc = np.percentile(df_res['yhat_'+model],q=80)
        dict_tab['Upper 20%'].append(df_res.loc[df_res['yhat_'+model]>=perc,'Treatment'].mean())
    for score in ['BNAUT','RNAUT','PNAUT']:
        dict_tab[score].append(trapezoid_role(dict_res[model],score,kargs[score]))

for model in model_list:
    dict_res[model][0] = 0
    plt.plot(range(0,110,10),dict_res[model],label=model)
plt.plot([0,100],[0,df_res['Treatment'].mean()], color='red', label='Random')
plt.title("Treatment Gain Chart")
plt.ylabel("Treatment (%)")
plt.xlabel("Percentile")
plt.legend(loc="lower right")
plt.margins(x=0,y=0.01)
plt.show()

print(pd.DataFrame.from_dict(dict_tab).round(3))


