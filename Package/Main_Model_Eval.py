import pandas as pd
from ML_Survival_Model import SurvivalModelEvaluation


if __name__ == "__main__":
    df_res = pd.read_csv('./Results/Result_preds.csv')
    df_res['yhat_GBSurvival'] = -df_res['yhat_GBSurvival'].values
    df_res['yhat_AAF'] = -df_res['yhat_AAF'].values
    df_res['yhat_Weibull_AFT'] = -df_res['yhat_Weibull_AFT'].values
    sme = SurvivalModelEvaluation('./','XGB')
    dict_res_eval = {}
    for model in [col.replace('yhat_','') for col in df_res.columns if 'yhat_' in col]:
        print('Processing model: {}'.format(model))
        sme.model_name = model
        dict_res_eval[model] = sme.bootstrap_eval(df_res,'all',boot_iter=100)
    pd.DataFrame.from_dict(dict_res_eval).to_excel('./Results/Results_Table.xlsx')
    
    print('The End!!!')
    