import pandas as pd
import numpy as np
import warnings
from sklearn.externals.joblib import Parallel
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

warnings.simplefilter("ignore")

print("Importing Data....")

adults = pd.read_csv("adult.csv",  index_col=False)
adults_int = adults.filter(["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"])
adults_cat = adults.filter(["workclass", "education", "martial-status", "occupation", "relationship", "race", "sex", "native-country", "outcome"])

# Label encode
for column in adults_cat.columns:
    le = LabelEncoder()
    le.fit(adults_cat[column])
    temp_df = {column: le.transform(adults_cat[column])}
    adults.update(temp_df)

# train test split
train = adults.sample(frac=0.7,random_state=200)
test = adults.drop(train.index)

# take y from data
y_train = train.outcome.tolist()
X_train = train.drop(["outcome"],  axis=1)
y_test = test.outcome.tolist()
X_test = test.drop(["outcome"],  axis=1)

# Grid Search
# RF GS
"""params = {"n_estimators" : list(range(5,35)), "max_depth" : list(range(2,10))}
grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=1), params, n_jobs=-1, verbose=1, scoring="roc_auc")
grid_search_cv.fit(X_train, y_train)
print("Random Forest Grid Search:")
print(grid_search_cv.best_estimator_)"""
"""RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=34, n_jobs=1,
            oob_score=False, random_state=1, verbose=0, warm_start=False)"""
# DT GS
"""params = {"max_depth" : list(range(5, 50)), "max_features" : list(range(1, 14))}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=1), params, n_jobs=-1, verbose=1, scoring="roc_auc")
grid_search_cv.fit(X_train, y_train)
print("Decision Tree Grid Search:")
print(grid_search_cv.best_estimator_)"""
"""DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=9,
            max_features=9, max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='best')"""

# Random Forest Model
"""rf_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=9, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=34, n_jobs=1,
            oob_score=False, random_state=1, verbose=0, warm_start=False)
rf_clf.fit(X_train, y_train)
joblib.dump(rf_clf, 'rf_clf.joblib')"""
rf_clf = joblib.load('rf_clf.joblib')

# cross validate
print("Random Forest Cross Validation:")
print(cross_val_score(rf_clf, X_train, y_train, cv=10, n_jobs=-1, scoring="roc_auc"))
"""Random Forest Cross Validation:
[0.91614251 0.91296251 0.91007151 0.92002825 0.91715557 0.90767106
 0.91122192 0.91610671 0.91242212 0.90893643]"""
# mean of 0.9132718590000002

# Decision Tree Model
"""dt_clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=9,
            max_features=9, max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='best')
dt_clf.fit(X_train, y_train)
joblib.dump(dt_clf, 'dt_clf.joblib')"""
dt_clf = joblib.load('dt_clf.joblib')

# cross validate
print("Decision Tree Cross Validation:")
print(cross_val_score(dt_clf, X_train, y_train, cv=10, n_jobs=-1, scoring="roc_auc"))
"""Decision Tree Cross Validation:
[0.89125274 0.88454197 0.88131537 0.90301728 0.89824362 0.89524424
 0.90240259 0.89448146 0.89696007 0.89254759]"""
# mean of 0.894000693

# SVC Model
# svc_clf = SVC(random_state=1, probability=True)
# svc_clf.fit(X_train, y_train)
# joblib.dump(svc_clf, 'svc_clf.joblib')
svc_clf = joblib.load('svc_clf.joblib')

# cross validate
print("SVC Cross Validation:")
print(cross_val_score(svc_clf, X_train, y_train, cv=10, n_jobs=-1, scoring="roc_auc"))
"""SVC Cross Validation:
[0.60114645 0.56773497 0.57797914 0.6043275  0.58838928 0.59996115
 0.57724241 0.57752752 0.57794946 0.57352858]"""

# hard voting classifer hard
# hard_voting_clf = VotingClassifier(estimators=[('rf', rf_clf), ('svc', svc_clf), ('dt', dt_clf)], voting='hard', n_jobs=-1)
# hard_voting_clf.fit(X_train, y_train)
# joblib.dump(hard_voting_clf, 'hard_voting_clf.joblib')
hard_voting_clf = joblib.load('hard_voting_clf.joblib')

# roc_auc score
# ERROR: predict_proba is not available when voting='hard'
# hard coded cross val scores
narr_X_train = X_train.as_matrix()
narr_y_train = np.array(y_train)
scores = []
k_folds = KFold(n_splits=10, shuffle=True)

for train_index, test_index in k_folds.split(narr_X_train):
    X = narr_X_train[train_index]
    y = narr_y_train[train_index]
    pred = hard_voting_clf.predict(X)
    roc_auc = roc_auc_score(pred, y)
    scores.append(roc_auc)

print("Hard Voting Roc_Auc Scores:")
print(scores)
"""Hard Voting Roc_Auc Scores:
[0.8597667907648499, 0.8606720390598028, 0.8603856899067265, 0.8595739331237385, 0.8600077938750486, 0.8605638395615038,
 0.860272704894949, 0.8610906878641633, 0.860450661416379, 0.8593689740924194]"""
# mean of 0.8602153114559581

# soft voting classifer hard
# soft_voting_clf = VotingClassifier(estimators=[('rf', rf_clf), ('svc', svc_clf), ('dt', dt_clf)], voting='soft', n_jobs=-1)
# soft_voting_clf.fit(X_train, y_train)
# joblib.dump(soft_voting_clf, 'soft_voting_clf.joblib')
soft_voting_clf = joblib.load('soft_voting_clf.joblib')

# cross validate
print("Soft Voting Cross Validation:")
print(cross_val_score(soft_voting_clf, X_train, y_train, cv=10, n_jobs=-1, scoring="roc_auc"))
"""Soft Voting Cross Validation:
[0.9132824  0.91209343 0.90748991 0.91915446 0.91615403 0.90774069
 0.91344524 0.91418438 0.91128331 0.90590521]"""
# mean of 0.9120733059999999 

# Bagging Model
# bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=500,  n_jobs=-1)
# bagging.fit(X_train, y_train)
# joblib.dump(bagging, 'bagging.joblib')
bagging = joblib.load('bagging.joblib')

# cross validate
print("Non-Limited Bagging Cross Validation:")
print(cross_val_score(bagging, X_train, y_train, cv=10, n_jobs=-1, scoring="roc_auc"))
"""Non-Limited Bagging Cross Validation:
[0.90428845 0.90212674 0.89858183 0.91252483 0.91043956 0.90004932
 0.90280645 0.91006999 0.90853863 0.90065935]"""
# mean of 0.9050085150000001

# Limited Bagging Model
# bagging_limit = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=500, n_jobs=-1, max_features=5)
# bagging_limit.fit(X_train, y_train)
# joblib.dump(bagging_limit, 'bagging_limit.joblib')
bagging_limit = joblib.load('bagging_limit.joblib')

# cross validate
print("Limited Bagging Cross Validation:")
print(cross_val_score(bagging_limit, X_train, y_train, cv=10, n_jobs=-1, scoring="roc_auc"))
"""Limited Bagging Cross Validation:
[0.91114006 0.90600515 0.90447692 0.91289654 0.91337873 0.89626096
 0.90850977 0.91262679 0.90485772 0.90659112]"""
# mean of 0.9076743760000001

# Gradint Boosted
"""params = {"n_estimators" : list(range(5, 25)), "learning_rate" : [0.1, 0.01, 0.2, 0.3, 0.4, 0.5, 0.9]}
grid_search_cv = GridSearchCV(GradientBoostingClassifier(random_state=1), params, n_jobs=-1, verbose=1, scoring="roc_auc")
grid_search_cv.fit(X_train, y_train)
print("GB Grid Search:")
print(grid_search_cv.best_estimator_)"""
"""GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.4, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=24,
              presort='auto', random_state=1, subsample=1.0, verbose=0,
              warm_start=False)"""

"""gb_clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.4, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=24,
              presort='auto', random_state=1, subsample=1.0, verbose=0,
              warm_start=False)
gb_clf.fit(X_train, y_train)
joblib.dump(gb_clf, 'gb_clf.joblib')"""
gb_clf = joblib.load('gb_clf.joblib')

# cross validate
print("GB Cross Validation:")
print(cross_val_score(gb_clf, X_train, y_train, cv=10, n_jobs=-1, scoring="roc_auc"))
"""GB Cross Validation:
[0.92385902 0.91691264 0.9179079  0.93034624 0.92324595 0.91476821
 0.9188218  0.9187097  0.91788472 0.91778658]"""
# mean of 0.920024276
