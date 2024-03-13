import os
import pandas as pd
from ML_Survival_Model import SurvivalModelPipeline

if __name__ == "__main__":
    dict_basepath = {'OS':'./','Windows':None}
    datapath = os.path.join(dict_basepath['OS'],'Survival_Analysis/Data/Data_Step2.xlsx')
    data_org = pd.read_excel(datapath).drop(columns='Unnamed: 0')

    use_cols = 'Total'
    dict_cols_drop = {'Total':['_2016-2019','AvgAll','SumAll'],
                    'Before':['_2016-2019','AvgAll','SumAll','Treatment','days_activity','2014','2015','2016'],
                    'After':['_2016-2019','AvgAll','SumAll','2011','2012','2013']}
    col_drop = [col for col in data_org.columns if any(val for val in dict_cols_drop[use_cols] if val in col)]
    col_drop += ['IMPORTO_LIQUIDATO','IMPORTI_TOTALI_LIQUIDATI','INTERVENTO_TECNICO']
    list_cols_for_label_transform = ['A10_GruppiAteco2007']
    cols_do_not_std = ['INTERVENTO_TECNICO','TimeFail','Fail','I_longitudine','I_latitudine','Capoluogo','Flag_Invest','Treatment']
    list_cols_to_std = [col for col in data_org.columns if col not in cols_do_not_std]
    list_cols_to_std = [col for col in list_cols_to_std if col not in col_drop]

    dict_cols_preproc = {'label_transform':list_cols_for_label_transform,'std':list_cols_to_std,'col_drop':col_drop}

    pipe = SurvivalModelPipeline(basepath=dict_basepath['OS'],
                                model_name=['Lasso','Ridge','LGB','XGB','RF','SVM',
                                            'CatBoost','GBSurvival','RF_Survival',
                                            'SVM_Survival','Cox','AAF','Weibull_AFT'],
                                dict_preproc=dict_cols_preproc)    
    data = pipe.kfold_cross_val(data_org,save=True)

    print('The End!!!')
