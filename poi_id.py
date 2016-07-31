import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester

features_list = ['poi',
                'salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Transform data from dictionary to the Pandas DataFrame
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
#Order columns in DataFrame, exclude email column
df = df[features_list]
df = df.replace('NaN', np.nan)

# Replace 'NaN' in financial features with 0
df.ix[:,:15] = df.ix[:,:15].fillna(0)

# impute missing values of email features 
email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
df.loc[df[df.poi == 1].index,email_features] = imp.fit_transform(df[email_features][df.poi == 1])
df.loc[df[df.poi == 0].index,email_features] = imp.fit_transform(df[email_features][df.poi == 0])

# correct typos in a data set
df.ix['BELFER ROBERT','total_payments'] = 3285
df.ix['BELFER ROBERT','deferral_payments'] = 0
df.ix['BELFER ROBERT','restricted_stock'] = 44093
df.ix['BELFER ROBERT','restricted_stock_deferred'] = -44093
df.ix['BELFER ROBERT','total_stock_value'] = 0
df.ix['BELFER ROBERT','director_fees'] = 102500
df.ix['BELFER ROBERT','deferred_income'] = -102500
df.ix['BELFER ROBERT','exercised_stock_options'] = 0
df.ix['BELFER ROBERT','expenses'] = 3285
df.ix['BELFER ROBERT',]
df.ix['BHATNAGAR SANJAY','expenses'] = 137864
df.ix['BHATNAGAR SANJAY','total_payments'] = 137864
df.ix['BHATNAGAR SANJAY','exercised_stock_options'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY','restricted_stock'] = 2.60449e+06
df.ix['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2.60449e+06
df.ix['BHATNAGAR SANJAY','other'] = 0
df.ix['BHATNAGAR SANJAY','director_fees'] = 0
df.ix['BHATNAGAR SANJAY','total_stock_value'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY']

# exclude outliers
df = df.drop(['TOTAL', 'LAVORATO JOHN J', 'MCMAHON JEFFREY'],0)

#create additional feature: fraction of person's email to POI to all sent messages
df['fraction_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
#clean all 'inf' values which we got if the person's from_messages = 0
df = df.replace('inf', 0)

#feature selection with DecisionTreeClassifier
#Decision tree using features with non-null importance
clf = DecisionTreeClassifier(random_state = 75)
clf.fit(df.ix[:,1:], df.ix[:,:1])
# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
print "Feature selection with DecisionTreeClassifier:"
for f_i in features_importance:
    print f_i
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

#Decision Tree Classifier with standard parametres 
clf = DecisionTreeClassifier(random_state = 75)
my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

#Random Forest with standard parameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 75)
clf.fit(df.ix[:,1:], np.ravel(df.ix[:,:1]))
# selecting the features with non null importance, sorting and creating features_list for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
print "Feature selection with Random Forest:"
for f_i in features_importance[:11]:
    print f_i
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')
# number of features for best result was found iteratively
features_list2 = features_list[:11]
my_dataset = df[features_list2].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()

# GaussianNB with feature standartization, selection, PCA
clf = GaussianNB()
# data set standartization
scaler = StandardScaler()
df_norm = df[features_list]
df_norm = scaler.fit_transform(df_norm.ix[:,1:])
# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
features_list2 = ['poi']+range(3)
my_dataset = pd.DataFrame(SelectKBest(f_classif, k=8).fit_transform(df_norm, df.poi), index = df.index)
#PCA
pca = PCA(n_components=3)
my_dataset2 = pd.DataFrame(pca.fit_transform(my_dataset),  index=df.index)
my_dataset2.insert(0, "poi", df.poi)
my_dataset2 = my_dataset2.to_dict(orient = 'index')  
tester.dump_classifier_and_data(clf, my_dataset2, features_list2)
tester.main()

# final model
clf = DecisionTreeClassifier(criterion = 'entropy', 
                             min_samples_split = 19,
                             random_state = 75,
                             min_samples_leaf=6, 
                             max_depth = 3)
clf.fit(df.ix[:,1:], df.poi)
# feature selection
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')
my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
print "Final model:"
tester.main() 