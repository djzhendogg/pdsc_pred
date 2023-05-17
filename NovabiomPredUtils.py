import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import shap




class NovaPredictTools:

    def Lazyregressor_vae(features, target, 
                      size_of_test=0.2,
                      scaler = MinMaxScaler(),
                      random_st = 0
                       ):
        X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size = size_of_test, random_state = random_st)
        
        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)

        clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        train, test = clf.fit(x_train, x_test, Y_train, Y_test)
        test_mod = test.iloc[:-1, :]
        return test_mod



    def LazyClass_vae(features, target, 
                      size_of_test=0.2,
                      scaler = MinMaxScaler(),
                      use_scaler = True,
                      random_st = 0
                       ):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = size_of_test, random_state = random_st)
        if use_scaler:
            scaler.fit(X_train)
            sc_x_train = scaler.transform(X_train)
            scaled_test_x_ = scaler.transform(X_test)

            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(sc_x_train, scaled_test_x_, y_train, y_test)
            return models
        
        else:
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            
            return models
        
        
    def LGBMClassifier_grid_cv(features, target,
                                scaler = MinMaxScaler(),
                                use_scaler=True,
                                random_st = 0,
                                n_cv = 5,
                                shuffle_on = True,
                                objective_st='binary',
                                learning_rat=0.1,
                                n_boost_round=500
                                ):
        
        param_gr = {
            'num_leaves': [31, 20],
            'reg_alpha': [0.1, 0.5],
            'min_data_in_leaf': [50, 100],
            'lambda_l1': [0, 1],
            'lambda_l2': [0, 1]
            }
        

        if use_scaler:
            scaler.fit(features)
            sc_feature = scaler.transform(features)
            gkf = KFold(n_splits=n_cv, shuffle=shuffle_on, random_state=random_st).split(features, target)
            
            
            lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective=objective_st, num_boost_round=n_boost_round, learning_rate=learning_rat)

            gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_gr, cv=gkf)
            lgb_model = gsearch.fit(sc_feature, target)

            return lgb_model.best_params_, lgb_model.best_score_
        
        else:
            gkf = KFold(n_splits=n_cv, shuffle=shuffle_on, random_state=random_st).split(features, target)
            
            
            lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective=objective_st, num_boost_round=n_boost_round, learning_rate=learning_rat)

            gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_gr, cv=gkf)
            lgb_model = gsearch.fit(features, target)

            return lgb_model.best_params_, lgb_model.best_score_
        
            

    def LGBMClass_Learning(features, target, target_name, best_grid, 
                            scaler = MinMaxScaler(),
                            use_scaler=True,
                            random_st = 0,
                            size_of_test=0.2
                            ):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = size_of_test, random_state = random_st)
        
        if use_scaler:
            scaler.fit(X_train)
            x_train = scaler.transform(X_train)
            x_test = scaler.transform(X_test)
            
            lgb_model = lgb.LGBMClassifier(**best_grid)
            lgb_model.fit(x_train, y_train)
            print("Train Accuracy:",lgb_model.score(x_train, y_train))
            print("TEST ACCURACY: ",lgb_model.score(x_test, y_test))


            y_pred = lgb_model.predict(x_test)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, y_pred, digits=3))
            y_test_proba = lgb_model.predict_proba(x_test)
            # print(y_test_proba)
            varience_df = pd.DataFrame(y_test_proba, columns = ['healthy', target_name])
            yyy = y_test.to_numpy()
            varience_df['real_target'] = yyy
            print("VARIANCE DATA")
            print(varience_df)

            x_train_names = pd.DataFrame(x_train, columns=X_train.columns)
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(x_train_names)
            shap.summary_plot(shap_values, x_train_names)
            return
        
        else:
            lgb_model = lgb.LGBMClassifier(**best_grid)
            lgb_model.fit(X_train, y_train)
            print("Train Accuracy:",lgb_model.score(X_train, y_train))
            print("TEST ACCURACY: ",lgb_model.score(X_test, y_test))


            y_pred = lgb_model.predict(X_test)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, y_pred, digits=3))
            y_test_proba = lgb_model.predict_proba(X_test)
            # print(y_test_proba)
            varience_df = pd.DataFrame(y_test_proba, columns = ['healthy', target_name])
            yyy = y_test.to_numpy()
            varience_df['real_target'] = yyy
            print("VARIANCE DATA")
            print(varience_df)

            x_train_names = pd.DataFrame(X_train, columns=X_train.columns)
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(x_train_names)
            shap.summary_plot(shap_values, x_train_names)
            return
        
            

    def Proverca(feat_train, targ_train, features, target, natural_target_name, best_grid, 
                            scaler = MinMaxScaler(),
                            ):
        scaler.fit(feat_train)
        x_train = scaler.transform(feat_train)
        x_test = scaler.transform(features)

        lgb_model = lgb.LGBMClassifier(**best_grid)
        lgb_model.fit(x_train, targ_train)
        y_pred = lgb_model.predict(x_test)
        y_test_proba = lgb_model.predict_proba(x_test)
        varience_df = pd.DataFrame(y_test_proba, columns = ['healthy', natural_target_name])

        varience_df['predicted_target'] = y_pred
        pupu = target.to_numpy()
        varience_df['real_target'] = pupu



        x_train_names = pd.DataFrame(features, columns=features.columns)
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(x_train_names)
        
        return varience_df, x_train_names, shap_values
       