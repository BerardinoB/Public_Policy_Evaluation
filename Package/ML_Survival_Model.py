import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,jaccard_score,log_loss, pairwise_distances,roc_curve,confusion_matrix
from tqdm import tqdm
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest as sksurv_RF_survival
from sksurv.svm import FastKernelSurvivalSVM
from sklearn.impute import IterativeImputer
from lifelines import CoxPHFitter
from lifelines.fitters.aalen_additive_fitter import AalenAdditiveFitter
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from lifelines.utils import concordance_index as c_index
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
import os, deepsurv
import pickle5 as pickle
from sklearn.calibration import CalibratedClassifierCV



class DataParser():
    def __init__(self,basepath,worning='on') -> None:
        self.worning = worning
        self.basepath = basepath
        self.dict_label_transform = {}
        self.dict_standard_scaler = {}
        self.dict_fitter_assertion = {}
        self.missing_imputer = None
        self.col_to_drop_for_cox = None
        
    def _presence_of_cols_in_df_assertion_(self,data,list_col,method_name):
        cols_not_in_df = [col for col in list_col if col not in data.columns]
        if len(cols_not_in_df)>0:
            assert False, 'Some of the columns are not present in DataFrame. Method Name failure:{}'.format(method_name)
    
    def _cols_drop_(self,data,list_cols_drop=[]) -> pd.DataFrame:
        if len(list_cols_drop)!=0:
            cols_not_in_df = [col for col in list_cols_drop if col not in data.columns]
            cols_in_df = [col for col in list_cols_drop if col in data.columns]
            if len(cols_not_in_df)!=0 and self.worning=='on':
                print('The following columns are not in the DataFrame:',cols_not_in_df)
            if len(cols_in_df)==0: 
                print('None of the columns are in the DataFrame. You should solve the problem before continue!!! Method Failure: _cols_drop_')
                assert False
            data.drop(columns=cols_in_df,inplace=True)
        return data
            
    def label_transform(self,data,label,list_cols_transform,method,col_label) -> pd.DataFrame:
        if list_cols_transform==None: 
            return data
        self._presence_of_cols_in_df_assertion_(data,list_cols_transform,method_name='label_transform')
        if method=='fit_transform':
            data = pd.concat([data,label],axis=1)
            for col_to_label_transform in list_cols_transform:
                val_unique = [val for val in data[col_to_label_transform].unique()]
                dict_map_risc = {val:data.loc[data[col_to_label_transform]==val,col_label].mean() for val in val_unique}
                data.loc[:,col_to_label_transform] = data[col_to_label_transform].map(dict_map_risc)
                self.dict_label_transform[col_to_label_transform] = dict_map_risc
                self.assertion_label_transform = True
            return data.drop(columns=label.columns)
        elif method=='transform':
            for col_to_label_transform in list_cols_transform:
                assert self.dict_label_transform!=None and col_to_label_transform in self.dict_label_transform.keys(),'Some columns are not fitted. Call label transform fitter before calling transfrom method!!!'
                data.loc[:,col_to_label_transform] = data[col_to_label_transform].map(self.dict_label_transform[col_to_label_transform])
            return data
                
    def _missing_imputer_method_(self,data,method) -> pd.DataFrame:
        n_missing = data.isnull().sum().values.sum()
        missing_imputer_obj_path = os.path.join(self.basepath,'Survival_Analysis/Utils')
        if n_missing==0:
            print('Data have no missing. No imputation is applied')
            return None
        list_columns = data.columns
        if method=='fit_transform':
            imp = IterativeImputer()
            data = pd.DataFrame(imp.fit_transform(data),columns=list_columns)
            self.missing_imputer = imp
            self.dict_fitter_assertion['_missing_imputer_method_'] = True
        elif method=='transform':
            assert self.dict_fitter_assertion['_missing_imputer_method_'], 'Missing Imputer requires you to call the fit_transform method before calling transform method!!!'
            data = self.missing_imputer.transform(data)
            data = pd.DataFrame(data,columns=list_columns)
        elif method=='load_transform':
            with open(os.path.join(missing_imputer_obj_path,'missing_imputer.pickle'), 'rb') as handle:
                self.missing_imputer = pickle.load(handle)
            data = pd.DataFrame(self.missing_imputer.transform(data),columns=list_columns)
            self.dict_fitter_assertion['_missing_imputer_method_'] = True
        elif method=='fit_and_save':
            imp = IterativeImputer().fit(data)
            pickle.dump(imp, open(os.path.join(missing_imputer_obj_path,'missing_imputer.pickle'), 'wb'))
            return None
        return data
    
    def standard_scaler_method(self,data,list_cols_to_std,method):
        if list_cols_to_std==None:
            return data
        self._presence_of_cols_in_df_assertion_(data,list_cols_to_std,method_name='standard_scaler_method')
        if method=='fit_transform':
            for col in list_cols_to_std:
                ss = StandardScaler()
                self.dict_standard_scaler[col] = ss.fit(data[col].values.reshape(-1,1))
                data[col] = ss.transform(data[col].values.reshape(-1,1))
            self.dict_fitter_assertion['standard_scaler_method'] = True
        elif method=='transform':
            assert self.dict_fitter_assertion['standard_scaler_method'], 'Standard Scaler requires you to call the fit_transform method before calling transform method!!!'
            for col in list_cols_to_std:
                data[col] = self.dict_standard_scaler[col].transform(data[col].values.reshape(-1,1))
        return data
    
    def _drop_corr_cols_for_cox(self,data,method,corr_threshold=0.8):
        if method=='fit_transform':
            corr_matrix = data.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            self.col_to_drop_for_cox = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        elif method=='transform':
            assert self.col_to_drop_for_cox!=None, 'Dropping correlated columns requires you to call the fit_transform method before calling transform method!!!'
        return data.drop(columns=self.col_to_drop_for_cox)


class GeneralizedSurvivalModel():
    def __init__(self, model_name=None, dict_iperparams=None) -> None:
        self.model_name = model_name
        dict_default_params = {'LGB':{'class_weight':'balanced'},
                                'XGB':{'scale_pos_weight':0.08},
                                'RF':{'class_weight':'balanced','n_jobs':-1},
                                'SVM':{'class_weight':'balanced','probability':True,'kernel':'rbf'},
                                'CatBoost':{'auto_class_weights':'Balanced'},
                                'Lasso':{'penalty':'l1','class_weight':'balanced','solver':'liblinear'},
                                'Ridge':{'penalty':'l2','class_weight':'balanced','solver':'liblinear'},
                                'GBSurvival':{'loss':'squared'},
                                'RF_Survival':{'n_estimators':100,'n_jobs':-1,
                                               'min_samples_split':10,
                                                'min_samples_leaf':15,
                                                'max_features':"sqrt"},
                                'SVM_Survival':{'kernel':'rbf'},
                                'Cox':{'penalizer':0.9},
                                'AAF':{'coef_penalizer':0.9},
                                'Weibull_AFT':{'penalizer':0.9},
                                'DeepSurv':{'n_in':10,
                                            'L1_reg': 10.0,
                                            'batch_norm': False,
                                            'dropout': 0.1,
                                            'hidden_layers_sizes': [100, 50,25],
                                            'learning_rate': 1e-05,
                                            'lr_decay': 0.001,
                                            'momentum': 0.9,
                                            'standardize': False}}
        if (self.model_name in dict_default_params.keys()) and dict_iperparams==None:
            dict_iperparams = dict_default_params[self.model_name]
        self.dict_iperparams = dict_iperparams
        self.assertion_training = False
        self.flag_unbalance_data = False
        self.dict_model_class = {'LGB':LGBMClassifier,
                                'XGB':xgb.XGBClassifier,
                                'RF':RandomForestClassifier,
                                'SVM':SVC,
                                'CatBoost':CatBoostClassifier,
                                'Lasso':LogisticRegression,
                                'Ridge':LogisticRegression,
                                'Cox':CoxPHFitter,
                                'AAF':AalenAdditiveFitter,
                                'Weibull_AFT':WeibullAFTFitter,
                                'GBSurvival':GradientBoostingSurvivalAnalysis,
                                'SVM_Survival':FastKernelSurvivalSVM,
                                'RF_Survival':sksurv_RF_survival,
                                'DeepSurv':deepsurv.DeepSurv}
        
    def model_instance(self) -> dict:
        if self.dict_iperparams==None:
            return {self.model_name:self.dict_model_class[self.model_name]()}
        else:
            return {self.model_name:self.dict_model_class[self.model_name](**self.dict_iperparams)}
        
    def _check_data_(self,X_train,y_train,duration_col,event_col,method):
        if not isinstance(X_train,pd.DataFrame):
            assert False, 'X_train is not a DataFrame object'
            
        if self.model_name in ['LGB','XGB','CatBoost','RF','SVM','GBSurvival','RF_Survival','SVM_Survival'] or (self.model_name in ['Cox','AAF','Weibull_AFT'] and method=='testing'):
            if len([col for col in [duration_col,event_col] if col in X_train.columns])!=0:
                assert False, 'Columns {0} and {1} detected in X_train'.format(duration_col,event_col)
            
        if self.model_name in ['LGB','XGB','CatBoost','RF','SVM'] and method=='training':
            if not isinstance(y_train,pd.Series):
                assert False, 'y_train is not instance of pd.Series'
            
        elif self.model_name in ['GBSurvival','RF_Survival','SVM_Survival'] and method=='training':
            if not isinstance(y_train,pd.DataFrame):
                assert False, 'y_train should be istance of pd.DataFrame with columns {0} and {1}'.format(event_col,duration_col)
            if len([col for col in [event_col,duration_col] if col not in y_train.columns])!=0:
                assert False, 'Columns {0} and {1} are not in y_train'
                
    def _y_for_survival_(self,y_event,y_time):
        return np.array([(fail,time) for fail,time in zip(y_event,y_time)],dtype=[('event', bool), ('time', np.float64)])
    
    def _data_for_DeepSuv_(self,X_train,y_train,duration_col,event_col):
        return {'x':X_train.values.astype('float32'),
                't':y_train[duration_col].values.astype('float32'),
                'e':y_train[event_col].values.astype('int32')}
            
    def _training_task_(self,X_train,y_train,duration_col,event_col):
        self._check_data_(X_train,y_train,duration_col,event_col,method='training')
        if self.model_name in ['GBSurvival','RF_Survival','SVM_Survival']:
            y_train = self._y_for_survival_(y_train[event_col],y_train[duration_col])
            self.model.fit(X_train,y_train)
        elif self.model_name in ['Cox','AAF','Weibull_AFT']:
            X_train = pd.concat([X_train.reset_index(drop=True),y_train.reset_index(drop=True)],axis=1)
            #Notice: due to collinearity we drop columns with corr value>0.8
            self.model.fit(X_train,duration_col='TimeFail', event_col='Fail')
            # print(k_fold_cross_validation(self.model, data_process, duration_col='TimeFail', event_col='Fail', k=3, scoring_method="concordance_index"))
        elif self.model_name in ['LGB','XGB','Lasso','Ridge','RF','SVM']:
            # self.model.fit(X_train,y_train,sample_weight=compute_sample_weight("balanced", y_train)) #XGB
            self.model.fit(X_train,y_train)
        elif self.model_name in ['CatBoost']:
            self.model.fit(X_train,y_train,verbose=False)
        elif self.model_name in ['DeepSurv']:
            data_process = self._data_for_DeepSuv_(X_train,y_train,duration_col,event_col)
            self.dict_iperparams['n_in'] = data_process['x'].shape[1]
            self.model = self.model_instance()[self.model_name]
            self.model.train(data_process,n_epochs=500)
        
    def fit(self,X,y=None,duration_col='TimeFail', event_col='Fail'):
        self.assertion_training = True
        if self.model_name in ['LGB','XGB','CatBoost','Lasso','Ridge','RF','SVM'] and isinstance(y,pd.DataFrame):
            y = y[event_col].copy()
        self.model = self.model_instance()[self.model_name]
        self._training_task_(X_train=X,y_train=y,duration_col=duration_col,event_col=event_col)
    
    def _prediction_task_(self,X_test,duration_col='TimeFail', event_col='Fail'):
        assert self.assertion_training, 'you should call train on the model before predict can be used'
        self._check_data_(X_test,None,duration_col,event_col,method='testing')
        if self.model_name in ['GBSurvival','RF_Survival','SVM_Survival']:
            return self.model.predict(X_test)
        elif self.model_name=='Cox':
            return self.model.predict_partial_hazard(X_test).values
        elif self.model_name in ['AAF','Weibull_AFT']:
            return self.model.predict_expectation(X_test).values
        elif self.model_name in ['LGB','XGB','CatBoost','Lasso','Ridge','RF','SVM']:
            return self.model.predict_proba(X_test)[:,1]
        elif self.model_name=='DeepSurv':
            return self.model.predict_risk(X_test.values.astype('float32'))
        
    def predict(self,X):
        assert self.assertion_training, 'Fit method not called yet!!!'
        return self._prediction_task_(X_test=X)
    

class SurvivalModelPipeline(DataParser,GeneralizedSurvivalModel):
    def __init__(self,basepath,model_name,kfold_path_save=None,dict_iperparams=None,dict_opt=None,dict_preproc=None) -> None:
        DataParser.__init__(self,basepath)
        if isinstance(model_name,list):
            self.model_name_list = list(model_name)
            if dict_iperparams==None:
                self.dict_iperparams_multimodel = {model:None for model in self.model_name_list}
            else:
                self.dict_iperparams_multimodel = dict_iperparams
        else:
            assert isinstance(model_name,str), 'model_name should be either instance of str or list!!!'
            self.model_name_list = None
            GeneralizedSurvivalModel.__init__(self,model_name,dict_iperparams)
        self.dict_opt = dict_opt
        if kfold_path_save==None:
            self.kfold_path_save = 'Survival_Analysis/Results/Result_preds.csv'
        else:
            self.kfold_path_save = kfold_path_save
        if dict_preproc==None:
            self.list_cols_for_label_transform = None
            self.list_cols_to_std = None
            self.col_drop = []
        else:
            if 'label_transform' in dict_preproc.keys():
                self.list_cols_for_label_transform = dict_preproc['label_transform']
            else:
                self.list_cols_for_label_transform = None
            if 'std' in dict_preproc.keys():
                self.list_cols_to_std = dict_preproc['std']
            else:
                self.list_cols_to_std = None
            if 'col_drop' in dict_preproc.keys():
                self.col_drop = dict_preproc['col_drop']
            else:
                self.col_drop = []
                
    def _generalized_survival_model_wrapper_(self,iter):
        if self.model_name_list!=None:
            GeneralizedSurvivalModel.__init__(self,self.model_name_list[iter],self.dict_iperparams_multimodel[self.model_name_list[iter]])
        
    def _train_test_task_(self,data_train,data_test,duration_col, event_col):
        X_train,y_train = data_train.drop(columns=[duration_col,event_col]), data_train.loc[:,[duration_col,event_col]]
        X_test = data_test.drop(columns=[duration_col,event_col])
        X_train = self.label_transform(data=X_train,label=data_train.loc[:,[event_col]], list_cols_transform=self.list_cols_for_label_transform,method='fit_transform',col_label=event_col)
        if self.model_name not in ['LGB','XGB','CatBoost']:
            X_train = self._missing_imputer_method_(data=X_train,method='load_transform')
        X_train = self.standard_scaler_method(data=X_train,list_cols_to_std=self.list_cols_to_std,method='fit_transform')
        self.fit(X_train,y_train)
        X_test = self.label_transform(data=X_test,label=None,list_cols_transform=self.list_cols_for_label_transform,method='transform',col_label=event_col)
        if self.model_name not in ['LGB','XGB','CatBoost']:
            X_test = self._missing_imputer_method_(data=X_test,method='transform')
        X_test = self.standard_scaler_method(data=X_test,list_cols_to_std=self.list_cols_to_std,method='transform')
        return self.predict(X_test)
    
    def kfold_cross_val(self,data,duration_col='TimeFail', event_col='Fail',n_splits=10,shuffle=True,save=False):
        data = self._cols_drop_(data=data.copy(deep=True),list_cols_drop=self.col_drop)
        skf = StratifiedKFold(n_splits=n_splits,shuffle=shuffle)
        df_final = data.copy(deep=True)
        for train_index, test_index in skf.split(X=data, y=data[event_col]):
            data_train,data_test = data.iloc[train_index,:],data.iloc[test_index,:]
            if self.model_name_list!=None:
                for iter in range(len(self.model_name_list)):
                    self._generalized_survival_model_wrapper_(iter)
                    yhat = self._train_test_task_(data_train,data_test,duration_col,event_col)
                    df_final.loc[test_index,'yhat_'+self.model_name] = yhat
            else:
                yhat = self._train_test_task_(data_train,data_test)
                df_final.loc[test_index,'yhat_'+self.model_name] = yhat
            if save:
                df_final.to_csv(os.path.join(self.basepath,self.kfold_path_save))
        return df_final
   
    
class SurvivalModelEvaluation():
    def __init__(self,basepath,model_name) -> None:
        self.basepath = basepath
        self.model_name = model_name
        self.dict_metric = {'C-index':self._concordance_idx_,'AUC':self._AUC_score_,'neg_log_likelihood':self._negative_log_likelihood_,
                            'F1':f1_score,'Precision':precision_score,'Recall':recall_score,'jaccard_score':jaccard_score,
                            'sensitivity':self._sensitivity_,'specificity':self._specificity_}
    
    def _calculate_opt_threshold_(self,y_true,yhat):
        false_pos_rate, true_pos_rate, proba = roc_curve(y_true,yhat)
        return sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
        
    def _probs_bin_(self,yhat_test,y_true_train,yhat_train,min_max_std):
        if min_max_std:
            minmax = MinMaxScaler()
            yhat_train = minmax.fit_transform(yhat_train.values.reshape(-1,1))
            yhat_test = minmax.transform(yhat_test.values.reshape(-1,1))
        optimal_proba_cutoff = self._calculate_opt_threshold_(y_true_train,yhat_train)
        return np.array([1 if i >= optimal_proba_cutoff else 0 for i in yhat_test])
    
    def _adjust_metric_(self,metric_res,metric):
        if self.model_name in ['AAF','GBSurvival','Weibull_AFT']:
            if metric in ['C-index']:
                return 1-metric_res
            return metric_res
        elif self.model_name in ['CatBoost','XGB','LGB','Lasso','Ridge','RF','SVM']:
            if metric in ['C-index']:
                return 1-metric_res
            return metric_res
        elif self.model_name in ['RF_Survival','SVM_Survival','Cox']:
            if metric in ['C-index']:
                return 1-metric_res
            return metric_res
        else:
            return metric_res
            
    def _sensitivity_(self,y_true,y_pred,average):
        cm = confusion_matrix(y_true,y_pred)
        return cm[0,0]/(cm[0,0]+cm[0,1])

    def _specificity_(self,y_true,y_pred,average):
        cm = confusion_matrix(y_true,y_pred)
        return cm[1,1]/(cm[1,0]+cm[1,1])
    
    def _AUC_score_(self,df_res,duration_col,event_col):
        auc_val = roc_auc_score(df_res[event_col],df_res['yhat_'+self.model_name])
        return self._adjust_metric_(auc_val,metric='AUC')
    
    def _concordance_idx_(self,df_res,duration_col,event_col):
        c_idx_val =  c_index(df_res[duration_col],df_res['yhat_'+self.model_name],event_observed=df_res[event_col])
        return self._adjust_metric_(c_idx_val,metric='C-index')
    
    def _negative_log_likelihood_(self,df_res,duration_col,event_col):
        yhat_std = MinMaxScaler().fit_transform(df_res['yhat_'+self.model_name].values.reshape(-1,1))
        if self.model_name in ['Weibull_AFT','AAF','SVM_Survival']:
            return log_loss(1-df_res[event_col].values,yhat_std)
        else:
            return log_loss(df_res[event_col].values,yhat_std)
    
    def _binary_metric_(self,df_res,df_for_opt_threshold,event_col,average,metric,min_max_std):
        yhat_bin = self._probs_bin_(yhat_test=df_res['yhat_'+self.model_name],
                                    y_true_train=df_for_opt_threshold[event_col],
                                    yhat_train=df_for_opt_threshold['yhat_'+self.model_name],
                                    min_max_std=min_max_std)
        val =  metric(df_res[event_col],yhat_bin,average=average)
        return self._adjust_metric_(val,metric=metric)
    
    def _metric_assertion_(self,metric):
        assert isinstance(metric,str) and metric in self.dict_metric.keys(), 'metric should be one of the following: {}'.format([m for m in self.dict_metric.keys()])
        
    def calculate_performance_metric(self,df_res,metric,df_threshold=None,duration_col='TimeFail',event_col='Fail',average='weighted',min_max_std=False):
        self._metric_assertion_(metric)
        if metric in ['F1','Precision','Recall','sensitivity','specificity','jaccard_score']:
            if not isinstance(df_threshold,pd.DataFrame):
                return self._binary_metric_(df_res,df_res,event_col,average,self.dict_metric[metric],min_max_std=min_max_std)
            else:
                return self._binary_metric_(df_res,df_threshold,event_col,average,self.dict_metric[metric],min_max_std=min_max_std)
        else:
            return self.dict_metric[metric](df_res,duration_col,event_col)
        
    def bootstrap_eval(self,df_res,metrics,boot_iter,duration_col='TimeFail',event_col='Fail',average='weighted',min_max_std=False):
        if isinstance(metrics,str):
            metric_list = [metrics]
        else:
            metric_list = metrics
        if len(metric_list)==1 and metric_list[0]=='all':
            metric_list = list(self.dict_metric.keys())
        dict_res_iter = {m:[] for m in metric_list}
        for _ in tqdm(range(boot_iter)):
            idx_ones = df_res[df_res[event_col]==1].sample(frac=0.8).index
            idx_zeros = df_res[df_res[event_col]==0].sample(frac=0.8).index
            idx = np.array(list(idx_ones)+list(idx_zeros))
            not_idx_ones = [i for i in df_res[df_res[event_col]==1].index if i not in idx_ones]
            not_idx_zeros = [i for i in df_res[df_res[event_col]==0].index if i not in idx_zeros]
            not_idx = np.array(not_idx_ones+not_idx_zeros)
            df_res_boot = df_res.iloc[not_idx,:].sample(n=2*len(not_idx),replace=True)
            df_threshold = df_res.iloc[idx,:].copy()
            for metric in metric_list:
                dict_res_iter[metric].append(self.calculate_performance_metric(df_res_boot,
                                                                               metric,
                                                                               df_threshold,
                                                                               min_max_std=min_max_std,
                                                                               duration_col=duration_col,
                                                                               event_col=event_col,
                                                                               average=average))
        return {metric:[round(np.mean(dict_res_iter[metric]),3),round(np.std(dict_res_iter[metric]),3)] for metric in dict_res_iter.keys()}

class SKLCalibrator():
    def __init__ (self):
        self.classes_ = [0,1]
        
    def fit(self,probs):
        return probs
    
    def predict_proba(self,X):
        return np.array([[np.abs(1-x),x] for x in X])

class UpliftModel(SKLCalibrator):
    def __init__(self,basepath=None,model_name=None,dist_metric='manhattan',n_neigh=1,cols_match=None,years_for_match=None) -> None:
        super().__init__()
        self.basepath = basepath
        self.model_name = model_name
        self.dist_metric = dist_metric
        self.n_neigh = n_neigh
        if cols_match==None:
            self.cols_match = ['ROA','ROE','ROS','EBITDA_eur','Attivo','Patrimonio','Debiti']
        else:
            self.cols_match = cols_match
        if years_for_match==None:
            self.years_for_match = ['2011','2012']
        else:
            self.years_for_match = years_for_match
        if cols_match!='Lasso':
            self.cols_match = [col+anno for col in self.cols_match for anno in self.years_for_match]
        self.str_params = ['model_name','n_neigh','dist_metric']
        self.params = [self.model_name,self.n_neigh,self.dist_metric]
        instances = [(str,tuple,list),(int,tuple,list),(str,tuple,list)]
        self._check_params_(self.params,instances)
        
    def _check_params_(self,params,instances):
        for i,(param,instance) in enumerate(zip(params,instances)):
            if isinstance(param,instance[0]):
                self.params[i] = [param]
            else:
                assert isinstance(param,instance),'params should be instance of: str, tuple, list. Got:{1}'.format(instance)

    def calibrate_proba(self,df_res,mode,model_name,method='sigmoid',event_col='Fail',return_calibrator=False):
        calibrated_clf = CalibratedClassifierCV(self, cv="prefit",method=method,n_jobs=-1)
        if mode=='fit':
            if return_calibrator:
                return calibrated_clf.fit(df_res['yhat_'+model_name],df_res[event_col])
            else:
                self.calibrated_clf = calibrated_clf.fit(df_res['yhat_'+model_name],df_res[event_col])
        elif mode=='predict_proba':
            return self.calibrated_clf.predict_proba(df_res['yhat_'+model_name])
    
    def _col_match_lasso_(self,data,treat_col,alpha=1):
        anno_out = [anno for anno in ['2011','2012','2013','2014','2015'] if anno not in self.years_for_match]
        cols_drop = [col for col in data.columns if 'yhat_' in col or any(x for x in anno_out if x in col)]
        cols_drop += ['TimeFail','Fail','A10_GruppiAteco2007','RISCH_AZI_ACCORPATO', 'IMPORTO_PROGETTO_RICHIESTO',
                      'IMPORTO_PROGETTO_RICHIESTO_AMM', 'IMPORTO_ANTICIPO_RICHIESTO','I_longitudine', 'I_latitudine',
                      'days_activity','Population','Surface', 'Density', 'Altitude']
        X = data.drop(columns=cols_drop).copy()
        y = data[treat_col]
        df_imp_treat = pd.DataFrame(SimpleImputer().fit_transform(X.loc[X[treat_col]==1,:]),columns=X.columns)
        df_imp_control = pd.DataFrame(SimpleImputer().fit_transform(X.loc[X[treat_col]==0,:]),columns=X.columns)
        df_imp_treat.drop(columns=treat_col,inplace=True)
        df_imp_control.drop(columns=treat_col,inplace=True)
        df_imp = pd.concat([df_imp_treat,df_imp_control])
        model = Lasso(alpha=alpha,max_iter=100000).fit(df_imp,y)

        idx = np.where(model.coef_!=0)
        self.cols_match = df_imp.columns[idx]
    
    def _pairwise_kernel_(self,data,treat_col,dict_param):
        data_std = pd.DataFrame(dict_param['scaler'].fit_transform(data.loc[:,self.cols_match]),columns=self.cols_match)
        data_std[treat_col] = data[treat_col].values
        df_imp_treat = pd.DataFrame(SimpleImputer().fit_transform(data_std.loc[data_std[treat_col]==1,self.cols_match]),columns=self.cols_match)
        df_imp_control = pd.DataFrame(SimpleImputer().fit_transform(data_std.loc[data_std[treat_col]==0,self.cols_match]),columns=self.cols_match)
        return pairwise_distances(df_imp_treat,df_imp_control,metric=dict_param['dist_metric'])
    
    def _uplift_(self,data,treat_col,dict_param,uplift_type) -> np.array:
        dict_param['scaler'] = MinMaxScaler()
        X_dist = self._pairwise_kernel_(data,treat_col,dict_param)
        df_treat = data[data[treat_col]==1].copy()
        df_control = data[data[treat_col]==0].copy()
        idx_dist_matrix = np.argsort(X_dist,axis=1)
        uplift_ATT = df_treat['yhat_'+dict_param['model_name']].values.reshape(-1,1) - df_control['yhat_'+dict_param['model_name']].values[idx_dist_matrix[:,:dict_param['n_neigh']]]
        if uplift_type=='ATT':
            return uplift_ATT
        elif uplift_type=='ATE':
            idx_dist_matrix = np.argsort(X_dist,axis=0)
            uplift_ATC = df_treat['yhat_'+dict_param['model_name']].values[idx_dist_matrix.T[:,:dict_param['n_neigh']]] - df_control['yhat_'+dict_param['model_name']].values.reshape(-1,1)
        return np.array(list(uplift_ATT)+list(uplift_ATC))
            
    def _gen_dict_params_combinations_(self):
        len_params = len([len(p) for p in self.params])
        prod_list = list(product(self.params[0],self.params[1]))
        if len_params==2:
            return prod_list
        for i in range(2,len_params):
            prod_list = list(product(prod_list,self.params[i]))
        return [(list(tup[0])+[x for x in tup[1:]]) for tup in prod_list]
    
    def _df_calibration_(self,data,event_col,n_splits):
        df_calibrated = pd.DataFrame([],columns=data.columns)
        skf = StratifiedKFold(n_splits=n_splits,shuffle=True)
        dict_calib_prob = {}
        for train_index, test_index in skf.split(X=data, y=data[event_col]):
            for model in self.model_name:
                self.calibrate_proba(data.iloc[train_index,:],model_name=model,mode='fit',event_col='Fail')
                dict_calib_prob[model] = self.calibrate_proba(data.iloc[test_index,:],model_name=model,mode='predict_proba',event_col='Fail')[:,1]
            data.loc[test_index,['yhat_'+m for m in self.model_name]] = np.array(list(dict_calib_prob.values())).T
            df_calibrated = pd.concat([df_calibrated,data.iloc[test_index,:]])
        return df_calibrated
        
    def calculate_uplift(self,data=None,event_col='Fail',treat_col='Treatment',uplift_type='ATT',calibrate_proba=False):
        if not isinstance(data,pd.DataFrame):
            data = pd.read_csv(os.path.join(self.basepath,'Results/Result_preds.csv')).drop(columns=['Unnamed: 0'])
        if self.cols_match=='Lasso':
            self._col_match_lasso_(data,treat_col)
        if calibrate_proba:
            data = self._df_calibration_(data.copy(),event_col,n_splits=10)
        dict_res = {}
        if uplift_type=='ATT':
            self.df_uplift_final = data[data[treat_col]==1].copy(deep=True)
        elif uplift_type=='ATE':
            self.df_uplift_final = pd.concat([data[data[treat_col]==1].copy(),data[data[treat_col]==0].copy()])
        for i,params in enumerate(self._gen_dict_params_combinations_()):
            dict_param_temp = {str_param:p for str_param,p in zip(self.str_params,params)}
            uplift = self._uplift_(data,treat_col,dict_param_temp,uplift_type)
            dict_res[tuple(params)] = uplift
            for j in range(len(self.str_params)):
                self.df_uplift_final.loc[:,'Uplift_'+str(i)+'_param_'+str(j)] = np.array([params[j]]*self.df_uplift_final.shape[0])
            self.df_uplift_final.loc[:,'Uplift_'+str(i)] = uplift.mean(axis=1)
        return dict_res







