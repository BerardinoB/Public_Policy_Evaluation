import os
os.chdir('/Users/berardinobarile/Documents/Angelo/Inail/Survival_Analysis/Package/')
import pandas as pd
import numpy as np
from ML_Survival_Model import UpliftModel

if __name__=='__main__':
    basepath = './'
    uplift_model = UpliftModel(basepath,model_name=['LGB','XGB','CatBoost','RF','SVM','Lasso','Ridge','GBSurvival', 'RF_Survival', 'SVM_Survival', 'Cox', 'AAF', 'Weibull_AFT'],dist_metric=['manhattan','euclidean','cosine'],
                        n_neigh=list(range(1,11)),cols_match='Lasso')
    df_res = pd.read_csv(os.path.join(basepath,'Results/Result_preds_Final.csv')).drop(columns=['Unnamed: 0'])
    uplift = uplift_model.calculate_uplift(data=df_res,uplift_type='ATE',calibrate_proba=False)
    
    uplift_model.df_uplift_final.to_csv('./Results/df_uplift_matched_Lasso_plus_Survival.csv')

    print('The End!!!')
