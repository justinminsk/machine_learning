import pandas as pd 
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import Imputer, LabelEncoder
import tensorflow as tf

warnings.simplefilter("ignore")

# read data in and add column header
data = pd.read_csv("thyroid/thyroid.csv", header=None, na_values="?", index_col=False, names=["age","sex","on_thyroxine",
    "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
    "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
    "tumor", "hypopituitary", "psych", "TSH_measured", "TSH", "T3_measured", "T3",
    "TT4_measured", "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured",
    "TBG", "referral_source", "diagnosis"])

temp_list = []

# get only the code and remove the index
for entry in data.diagnosis:
    value = entry.split("[",1)[0]
    temp_list.append(value)

temp_df = pd.DataFrame({"diagnosis" : temp_list})
data.update(temp_df)

# Replace cat (false/true) and (male/female) varibles with 0 and 1
data = data.replace("f",0)
data = data.replace("t",1)
data = data.replace("M",0)
data = data.replace("F",1)

# Turn referral_source into int cats
# TODO: Figure out if this can be turned into a one hot
le = LabelEncoder()
le.fit(data.referral_source)
temp_df = {"referral_source": le.transform(data.referral_source)}
data.update(temp_df)

# train test split
train=data.sample(frac=0.7,random_state=200)
test=data.drop(train.index)

# take y from data
y_train = train.diagnosis.tolist()
X_train = train.drop(["diagnosis"],  axis=1)
y_test = test.diagnosis.tolist()
X_test = test.drop(["diagnosis"],  axis=1)

# replace Nan's with most freq
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)
imp = imp.fit(X_test)
X_test_imp = imp.transform(X_test)

# Label encode outcomes
le = LabelEncoder()
le.fit(y_train)
y_train_le = le.transform(y_train)
le.fit(y_test)
y_test_le = le.transform(y_test)

# create model
rf_clf = RandomForestClassifier(max_depth=5, random_state=0)
rf_clf.fit(X_train_imp, y_train_le)

# cross validate
print("Random Forest CV:")
print(cross_val_score(rf_clf, X_train_imp, y_train_le, cv=5))

# create model
ada_clf =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train_imp, y_train_le)

# cross validate
print("Ada Boost CV:")
print(cross_val_score(ada_clf, X_train_imp, y_train_le, cv=5))

# pred test and get a score
pred = ada_clf.predict(X_test_imp)
print("Ada Boost Test F1 Score:")
# use a beta value of 0.4 for precision, also weight each catagory
# from sklearn: Calculate metrics for each label, and find their average weighted by support
print(fbeta_score(pred, y_test_le, 0.4, average='weighted'))
print("Ada Boost Test Accuracy:")
print(ada_clf.score(X_test_imp, y_test_le))

# create model
svm_clf = SVC(probability=True)
svm_clf.fit(X_train_imp, y_train_le)

# cross validate
print("SVM CV:")
print(cross_val_score(svm_clf, X_train_imp, y_train_le, cv=5))

# create model
dt_clf = DecisionTreeClassifier(max_depth=8)
dt_clf.fit(X_train_imp, y_train_le)

# cross validate
print("Decision Tree CV:")
print(cross_val_score(dt_clf, X_train_imp, y_train_le, cv=5))

# pred test and get a score
pred = dt_clf.predict(X_test_imp)
print("Decision Tree Test F1 Score:")
# use a beta value of 0.4 for precision, also weight each catagory
# from sklearn: Calculate metrics for each label, and find their average weighted by support
print(fbeta_score(pred, y_test_le, 0.4, average='weighted'))
print("Decision Tree Test Accuracy:")
print(dt_clf.score(X_test_imp, y_test_le))

# create decision tree
export_graphviz(dt_clf, out_file="DTthyroid.dot", feature_names=X_train.columns, filled=True)

# voting classifer
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('ada', ada_clf), ('svc', svm_clf), ('dt', dt_clf)],
    voting='soft')

voting_clf.fit(X_train_imp, y_train_le)

print("Voting CV:")
print(cross_val_score(voting_clf, X_train_imp, y_train_le, cv=5))

# pred test and get a score
pred = voting_clf.predict(X_test_imp)
print("Voting Test F1 Score:")
# use a beta value of 0.4 for precision, also weight each catagory
# from sklearn: Calculate metrics for each label, and find their average weighted by support
print(fbeta_score(pred, y_test_le, 0.4, average='weighted'))
print("Voting Test Accuracy:")
print(voting_clf.score(X_test_imp, y_test_le))
