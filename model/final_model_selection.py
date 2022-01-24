from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


def final_model_selection(x_train, y_train, x_test, y_test):
        # lgbm
        lgbm = LGBMClassifier(random_state=1)
        lgbm.fit(x_train, y_train)
        lgbm_pred = lgbm.predict(x_test)
        lgbm_predictions = [round(value) for value in lgbm_pred]
        accuracy_lgbm = accuracy_score(y_test, lgbm_predictions)
        # xgb
        xgb = XGBClassifier(random_state=1)
        xgb.fit(x_train, y_train)
        xgb_pred = xgb.predict(x_test)
        xgb_predictions = [round(value) for value in xgb_pred]
        accuracy_xgb = accuracy_score(y_test, xgb_predictions)
        # RF
        rfr = RandomForestClassifier(random_state=1)
        rfr.fit(x_train, y_train)
        rfr_pred = rfr.predict(x_test)
        rfr_predictions = [round(value) for value in rfr_pred]
        accuracy_rf = accuracy_score(y_test, rfr_predictions)

        # accuracy score를 기준으로 모델간 비교하여 best 모델선정
        if max(accuracy_lgbm, accuracy_xgb , accuracy_rf) == accuracy_lgbm:
                params= params_lgbm
                model= LGBMClassifier()
        elif max(accuracy_lgbm, accuracy_xgb , accuracy_rf) == accuracy_xgb:
                params= params_xgb
                model= XGBClassifier()
        elif max(accuracy_lgbm, accuracy_xgb , accuracy_rf) == accuracy_rf:
                params= params_rf
                model= RandomForestClassifier()

        # best 모델에서 random search 로 하이퍼 파리미터 최적화
        best_model= RandomizedSearchCV(model, params, random_state=1, cv=5, n_iter=100, scoring='neg_accuracy_score')
        best_model.fit(x_train, y_train)
        return best_model
