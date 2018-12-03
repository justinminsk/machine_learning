import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# logging.basicConfig(filename='grid_search.txt', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(filename='tree_models.txt', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info(" ")
logging.info("-----------------------")
logging.info("New Model")
logging.info("-----------------------")
logging.info(" ")

warnings.simplefilter("ignore")

data = pd.read_csv("train.csv", na_values=["NULL","nan"])

logging.info("Data Imported")

# drop null rows
data.dropna()

logging.info("Dropped NULL")

# create int cats
cat_columns = ["Weekday", "Upc", "DepartmentDescription", "FinelineNumber", "TripType"]
for cat in cat_columns:
    data[cat] = data[cat].astype('category')

cat_columns = data.select_dtypes(['category']).columns

data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

logging.info("Data Removed of Strings")

# train test split
train=data.sample(frac=0.9)
test=data.drop(train.index)

# take y from data
y_train = train.TripType.tolist()
X_train = train.drop(["TripType"],  axis=1)
y_test = test.TripType.tolist()
X_test = test.drop(["TripType"],  axis=1)

# print(X_train.head())

logging.info("Train Test Split")

 # create np arrays
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = np.array(y_train)
y_test = np.array(y_test)


def main_process():

    warnings.simplefilter("ignore")
    
    # Grid Search
    # RF GS
    """
    params = {"n_estimators" : list(range(5,15)), "max_depth" : list(range(2,5))}
    grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=1), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(X_train, y_train)
    logging.info("Random Forest Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    """

    # DT GS
    """
    params = {"max_depth" : list(range(5, 10)), "max_features" : list(range(1, 6))}
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=1), params, n_jobs=-1, verbose=1)
    grid_search_cv.fit(X_train, y_train)
    logging.info("Decision Tree Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    """

    # RF
    """
    rf_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=4, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=8, n_jobs=1,
                oob_score=False, random_state=1, verbose=0, warm_start=False)
    rf_clf.fit(X_train, y_train)
    logging.info("Trained Random Forest Model")
    joblib.dump(rf_clf, 'rf_clf.joblib')
    """
    rf_clf = joblib.load('rf_clf.joblib')

    # cross validate
    print("Random Forest Cross Validation:")
    cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=10, n_jobs=-1)
    print(cv_scores)
    print(np.mean(cv_scores))

    # DT
    """
    dt_clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=9,
                max_features=5, max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=1, splitter='best')
    dt_clf.fit(X_train, y_train)
    logging.info("Trained Decision Tree Model")
    joblib.dump(dt_clf, 'dt_clf.joblib')
    """
    dt_clf = joblib.load('dt_clf.joblib')

    # cross validate
    print("Decision Tree Cross Validation:")
    cv_scores = cross_val_score(dt_clf, X_train, y_train, cv=10, n_jobs=-1)
    print(cv_scores)
    print(np.mean(cv_scores))

    # GBF
    gbf_clf = GradientBoostingClassifier(n_estimators=8, learning_rate=0.1)
    gbf_clf.fit(X_train, y_train)
    print("Trained Gradient Boosted Model")
    joblib.dump(gbf_clf, 'gbf_clf.joblib')
    # gbf_clf = joblib.load('gbf_clf.joblib')

    # cross validate
    print("Gradient Boosted Cross Validation:")
    cv_scores = cross_val_score(gbf_clf, X_train, y_train, cv=5, n_jobs=-1)
    print(cv_scores)
    print(np.mean(cv_scores))


if __name__ == "__main__":
    main_process()
