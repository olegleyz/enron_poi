
<img src="http://www.londoncommunications.co.uk/wp-content/uploads/2013/04/Enron.jpg">
# Identifing Fraud from Enron Email and financial data
25.07.2016  
Oleg Leyzerov

## Citation
This dataset consists from ENRON emails and financial data publicly available for research. The ENRON Email dataset was collected and prepared by the CALO Project (A Cognitive Assistant that Learns and Organizes), details are described in https://www.cs.cmu.edu/~./enron/. The financial data was published in Payments to Insiders report by FindLaw and available at www.findlaw.com

## Abstract
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I'm putting my new skills by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## Understanding the Dataset and Question
The goal of the project is to go through the thought process of data exploration (learning, cleaning and preparing the data), feature selecting/engineering (selecting the features which influence mostly on the target, create new features (which explains the target the better than existing) and, probably, reducing the dimensionality of the data using principal component analysis (PCA)), picking/tuning one of the supervised machine learning algorithm and validating it to get the accurate person of interest identifier model.

### Data Exploration
The features in the data fall into three major types, namely financial features, email features and POI labels.  
* There are 146 samples with 20 features and a binary classification ("poi"), 2774 data points.
* Among 146 samples, there are 18 POI and 128 non-POI.
* Among 2774, there are 1358 (48.96%) data points with NaN values.


```python
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

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Transform data from dictionary to the Pandas DataFrame
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
#Order columns in DataFrame, exclude email column
df = df[features_list]
df = df.replace('NaN', np.nan)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
    Data columns (total 19 columns):
    poi                          146 non-null bool
    salary                       95 non-null float64
    bonus                        82 non-null float64
    long_term_incentive          66 non-null float64
    deferred_income              49 non-null float64
    deferral_payments            39 non-null float64
    loan_advances                4 non-null float64
    other                        93 non-null float64
    expenses                     95 non-null float64
    director_fees                17 non-null float64
    total_payments               125 non-null float64
    exercised_stock_options      102 non-null float64
    restricted_stock             110 non-null float64
    restricted_stock_deferred    18 non-null float64
    total_stock_value            126 non-null float64
    to_messages                  86 non-null float64
    from_messages                86 non-null float64
    from_this_person_to_poi      86 non-null float64
    from_poi_to_this_person      86 non-null float64
    dtypes: bool(1), float64(18)
    memory usage: 21.8+ KB



```python
#split of POI and non-POI in the dataset
poi_non_poi = df.poi.value_counts()
poi_non_poi.index=['non-POI', 'POI']
print "POI / non-POI split"
poi_non_poi
```

    POI / non-POI split





    non-POI    128
    POI         18
    Name: poi, dtype: int64



### Data Cleansing


```python
print "Amount of NaN values in the dataset: ", df.isnull().sum().sum()
```

    Amount of NaN values in the dataset:  1263


According to the financial data from FindLaw, NaN values represent values of 0 but not the missing value. That's why I replace all NaNs with 0.


```python
# Replacing 'NaN' in financial features with 0
df.ix[:,:15] = df.ix[:,:15].fillna(0)
```

NaN values in email features means the information is missing. I'm going to split the data into 2 classes: POI/non-POI  and impute the missing values with median of each class.


```python
email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

#impute missing values of email features 
df.loc[df[df.poi == 1].index,email_features] = imp.fit_transform(df[email_features][df.poi == 1])
df.loc[df[df.poi == 0].index,email_features] = imp.fit_transform(df[email_features][df.poi == 0])

```

I'm going to check the accuracy of the financial data by summing up the payment features and comparing it with the total_payment feature and stock features and comparing with the total_stock_value.


```python
#check data: summing payments features and compare with total_payments
payments = ['salary',
            'bonus', 
            'long_term_incentive', 
            'deferred_income', 
            'deferral_payments',
            'loan_advances', 
            'other',
            'expenses', 
            'director_fees']
df[df[payments].sum(axis='columns') != df.total_payments]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>to_messages</th>
      <th>from_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-102500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>102500.0</td>
      <td>3285.0</td>
      <td>0.0</td>
      <td>44093.0</td>
      <td>-44093.0</td>
      <td>944.0</td>
      <td>41.0</td>
      <td>6.0</td>
      <td>26.5</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>15456290.0</td>
      <td>2604490.0</td>
      <td>-2604490.0</td>
      <td>15456290.0</td>
      <td>0.0</td>
      <td>523.0</td>
      <td>29.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock_value = ['exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred']
df[df[stock_value].sum(axis='columns') != df.total_stock_value]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>to_messages</th>
      <th>from_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELFER ROBERT</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-102500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3285.0</td>
      <td>102500.0</td>
      <td>3285.0</td>
      <td>0.0</td>
      <td>44093.0</td>
      <td>-44093.0</td>
      <td>944.0</td>
      <td>41.0</td>
      <td>6.0</td>
      <td>26.5</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>0.0</td>
      <td>137864.0</td>
      <td>15456290.0</td>
      <td>2604490.0</td>
      <td>-2604490.0</td>
      <td>15456290.0</td>
      <td>0.0</td>
      <td>523.0</td>
      <td>29.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Two samples have the mistakes in the data entry. So I'm going to correct them and check that everything is correct now (empty DataFrames mean no samples with mistakes in the data set).


```python
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
df[df[payments].sum(axis='columns') != df.total_payments]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>to_messages</th>
      <th>from_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df[df[stock_value].sum(axis='columns') != df.total_stock_value]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>poi</th>
      <th>salary</th>
      <th>bonus</th>
      <th>long_term_incentive</th>
      <th>deferred_income</th>
      <th>deferral_payments</th>
      <th>loan_advances</th>
      <th>other</th>
      <th>expenses</th>
      <th>director_fees</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>restricted_stock</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>to_messages</th>
      <th>from_messages</th>
      <th>from_this_person_to_poi</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### Outlier Investigation
On the previous step I've cleaned the data from missing values and typos. Now I would like to discover the outliers. Descriptive statistics determins outliers of the distibution as the values which are higher than Q2 + 1.5*IQR or less than Q2 - 1.5*IQR, where Q2 median of the distribution, IQR - interquartile range. 
I'm going to calculate the sum of outlier variables for each person and sort them descending.


```python
outliers = df.quantile(.5) + 1.5 * (df.quantile(.75)-df.quantile(.25))
pd.DataFrame((df[1:] > outliers[1:]).sum(axis = 1), columns = ['# of outliers']).\
    sort_values('# of outliers',  ascending = [0]).head(7)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># of outliers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOTAL</th>
      <td>12</td>
    </tr>
    <tr>
      <th>LAY KENNETH L</th>
      <td>12</td>
    </tr>
    <tr>
      <th>FREVERT MARK A</th>
      <td>12</td>
    </tr>
    <tr>
      <th>WHALLEY LAWRENCE G</th>
      <td>11</td>
    </tr>
    <tr>
      <th>SKILLING JEFFREY K</th>
      <td>11</td>
    </tr>
    <tr>
      <th>LAVORATO JOHN J</th>
      <td>9</td>
    </tr>
    <tr>
      <th>MCMAHON JEFFREY</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



Our data set is really small, so I'm going to consider just 5% of the samples with most number of outlier variables:  
1. The first value is 'TOTAL' which is the total value of financial payments from the FindLaw data. As it's doesn't make any sence for our solution, I'm going to exclude it from the data set.  
2. Kenneth Lay and Jeffrey Skilling are very well known persons from ENRON - I should keep them as they represent anomalies but not the outliers.  
3. Mark Frevert and Lawrence Whalley are not so very well known but top managers of the Enron who also represent valuable examples for the model - I'm also going to keep them in the data set.  
4. John Lavorato is not very well known person as far as I've searched in the internet. I don't think he represents a valid point and exclude him.   
5. Jeffrey Mcmahon is the former treasurer who worked before guilty Ben Glisan. I would exclude him from the data set as he worked before the guilty treasurer and might add some confusion to the model.  

From considered 7 persons I've ended up with excluding 3 of them (1 typo 'TOTAL' and 2 persons).



```python
scaler = StandardScaler()
df_norm = df[features_list]
df_norm = scaler.fit_transform(df_norm.ix[:,1:])

clf = GaussianNB()

features_list2 = ['poi']+range(8)

my_dataset = pd.DataFrame(SelectKBest(f_classif, k=8).fit_transform(df_norm, df.poi), index = df.index)
my_dataset.insert(0, "poi", df.poi)
my_dataset = my_dataset.to_dict(orient = 'index')  

dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()
```

    GaussianNB()
    	Accuracy: 0.33760	Precision: 0.14848	Recall: 0.83800	F1: 0.25226	F2: 0.43447
    	Total predictions: 15000	True positives: 1676	False positives: 9612	False negatives:  324	True negatives: 3388
    



```python
# exclude 3 outliers from the data set
df = df.drop(['TOTAL', 'LAVORATO JOHN J', 'MCMAHON JEFFREY'],0)
```

## Optimize Feature Selection/Engineering
During the work on the project, I've played with the different features and models. One strategy was to standardize features, apply principal component analysis and GaussianNB classifier, another strategy was to use decision tree classifier, incl. choosing the features with features importance attribute and tuning the model.  

### Create new features
For both strategies I've tried to create new features as a fraction of almost all financial variables (f.ex. fractional bonus as fraction of bonus to total_payments, etc.). Logic behind email feature creation was to check the fraction of emails, sent to POI, to all sent emails; emails, received from POI, to all received emails.  
I've end up with using one new feature fraction_to_POI:


```python
#create additional feature: fraction of person's email to POI to all sent messages
df['fraction_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
#clean all 'inf' values which we got if the person's from_messages = 0
df = df.replace('inf', 0)
```

Second strategy showed significantly better results so the rest of the project I'm going to concentrate on it.
Decision tree doesn't require me to do any feature scaling so I've skipped this step.  

### Intelligently select features
On the feature selection step I've fitted my DecisionTreeClassifier with all features and as a result received number of features with non-null feature importance, sorted by importance.


```python
#Decision tree using features with non-null importance
clf = DecisionTreeClassifier(random_state = 75)
clf.fit(df.ix[:,1:], df.ix[:,:1])

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
for f_i in features_importance:
    print f_i
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')
```

    ['fraction_to_poi', 0.35824390243902443]
    ['expenses', 0.26431889023871075]
    ['to_messages', 0.16306330961503368]
    ['other', 0.084740740740740714]
    ['deferred_income', 0.070617283950617254]
    ['from_poi_to_this_person', 0.059015873015873015]


According to feature__importances_ attribute of the classifier, just created fraction_to_poi feature has the highest importance for the model. The number of features used for the model may cause different results. Later on the algorithm tuning step I'm going to re-iterate the step of choosing features with non-null importance so the number of them will be changed.  
I'm using random state equal to 75 in decision tree and random forest to be able to represent the results. The exact value was manually chosen for better performance of decision tree classifier.

## Pick and Tune an Algorithm
I've played with 3 machine learning algorithms:
* Decision Tree Classifier
* Random Forest
* GaussianNB

For decision tree and random forest I've selected just features with non-null importance based on clf.features_importances__. On the next step I've iteratively changed the number of features from 1 to all in order to achieve the best performance.  
For the GaussianNB classifier I've applied a number of steps to achieve the result:
* standardized features;  
* applied SelectKBest function from sklearn to find k best features for the algorithm (I've ended up with k = 8 which gave me better result for k in a range from 1 to all);  
* applied PCA to decrease the dimensionality of the data (I've ended up with n_components = 3).  
  
I ended up using Decision Tree Classifier. Decision tree showed the best result and was significantly faster than RandomForest so I could easily tune it.  
I've gotten the following results from the algorithms before tuning (using tester.py, provided in advance):


```python
pd.DataFrame([[0.90880, 0.66255, 0.64400, 0.65314],
              [0.89780, 0.70322, 0.40400, 0.51318],
              [0.86447, 0.49065, 0.43300, 0.46003]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['Decision Tree Classifier', 'Random Forest', 'Gaussian Naive Bayes'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Decision Tree Classifier</th>
      <td>0.90880</td>
      <td>0.66255</td>
      <td>0.644</td>
      <td>0.65314</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.89780</td>
      <td>0.70322</td>
      <td>0.404</td>
      <td>0.51318</td>
    </tr>
    <tr>
      <th>Gaussian Naive Bayes</th>
      <td>0.86447</td>
      <td>0.49065</td>
      <td>0.433</td>
      <td>0.46003</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Decision Tree Classifier with standard parametres 
clf = DecisionTreeClassifier(random_state = 75)
my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main() 
```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=75, splitter='best')
    	Accuracy: 0.90880	Precision: 0.66255	Recall: 0.64400	F1: 0.65314	F2: 0.64763
    	Total predictions: 15000	True positives: 1288	False positives:  656	False negatives:  712	True negatives: 12344
    



```python
#Random Forest with standard parameters
clf = RandomForestClassifier(random_state = 75)
clf.fit(df.ix[:,1:], np.ravel(df.ix[:,:1]))

# selecting the features with non null importance, sorting and creating features_list for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

# number of features for best result was found iteratively
features_list2 = features_list[:11]
my_dataset = df[features_list2].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()
```

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=75, verbose=0, warm_start=False)
    	Accuracy: 0.89780	Precision: 0.70322	Recall: 0.40400	F1: 0.51318	F2: 0.44158
    	Total predictions: 15000	True positives:  808	False positives:  341	False negatives: 1192	True negatives: 12659
    



```python
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

dump_classifier_and_data(clf, my_dataset2, features_list2)
tester.main()
```

    GaussianNB()
    	Accuracy: 0.86447	Precision: 0.49065	Recall: 0.43300	F1: 0.46003	F2: 0.44342
    	Total predictions: 15000	True positives:  866	False positives:  899	False negatives: 1134	True negatives: 12101
    


### Tune the algorithm
Bias-variance tradeoff is one of the key dilema in machine learning. High bias algorithms has no capacity to learn, high variance algorithms react poorly in case they didn't see such data before. Predictive model should be tuned to achieve compromise. The process of changing the parameteres of algorithms is algorithm tuning and it lets us find the golden mean and best result. If I don't tune the algorithm well, I don't get the best result I could.  
Algorithm might be tuned manually by iteratively changing the parameteres and tracking the results. Or GridSearchCV might be used which makes this automatically.  
I've tuned the parameteres of my decision tree classifier by sequentially tuning parameter by parameter and got the best F1 using these parameters:


```python
clf = DecisionTreeClassifier(criterion = 'entropy', 
                             min_samples_split = 19,
                             random_state = 75,
                             min_samples_leaf=6, 
                             max_depth = 3)
```


```python
clf.fit(df.ix[:,1:], df.poi)

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)

features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main() 
```

    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=6,
                min_samples_split=19, min_weight_fraction_leaf=0.0,
                presort=False, random_state=75, splitter='best')
    	Accuracy: 0.93673	Precision: 0.83238	Recall: 0.65800	F1: 0.73499	F2: 0.68678
    	Total predictions: 15000	True positives: 1316	False positives:  265	False negatives:  684	True negatives: 12735
    


## Validate and Evaluate
### Usage of Evaluation Metrics 
In the project I've used F1 score as key measure of algorithms' accuracy. It considers both the precision and the recall of the test to compute the score.  
Precision is the ability of the classifier not label as positive sample that is negative.  
Recall is the ability of the classifier to find all positive samples.  
The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.

My tuned decision tree classifier showed precision 0.82238 and recall 0.65800 with the resulting F1 score 0.73499.
I can explain it as 82.24% of the called POI are POI and 65.80% of POI are identified.

### Validation Strategy
The validation is a process of model performance evaluation. Classic mistace is to use small data set for the model training or validate model on the same data set as train it.  
There are a number of strategies to validate the model. One of them is to split the available data into train and test data another one is to perform a cross validation: process of splitting the data on k beans equal size; run learning experiments; repeat this operation number of times and take the average test result.  

### Algorithm Performance
For validation I'm using provided tester function which performs stratified shuffle split cross validation approach using StratifiedShuffleSplit function from sklearn.cross_validation library. The results are:


```python
pd.DataFrame([[0.93673, 0.83238, 0.65800, 0.73499]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['Decision Tree Classifier'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Decision Tree Classifier</th>
      <td>0.93673</td>
      <td>0.83238</td>
      <td>0.658</td>
      <td>0.73499</td>
    </tr>
  </tbody>
</table>
</div>



## Reflection (Conclusions)
Before the start of this project I was completely sure that building the machine learning is about choosing the right algorithm from the black box and some magic.
Working on the person of interest identifier I've been recursively going through the process of data exploration, outlier detection and algorithm tuning and spend most of the time on a data preparation. The model performance raised significantly after missing values imputation, extra feature creation and feature selection and less after algorithm tuning which shows me once again how important to fit the model with the good data.  
This experience might be applied to other fraud detection tasks. I think there is way of the model improvement by using and tuning alternative algorithms like Random Forest.

### Limitations of the study
It’s important to identify and acknowledge the limitation of the study. My conclusions are based just on the provided data set which represent just 145 persons. To get the real causation, I should gather all financial and email information about all enron persons which is most probably not possible. Missing email values were imputed with median so the modes of the distributions of email features are switched to the medians. Algorithms were tuned sequentially (I've changed one parameter to achieve better performance and then swithched to another parameter. There is a chance that othere parameters in combination might give better model's accuracy).


## References:
Enron data set: https://www.cs.cmu.edu/~./enron/  
FindLaw financial data: http://www.findlaw.com  
Visualization of POI: http://www.nytimes.com/packages/html/national/20061023_ENRON_TABLE/index.html  
Enron on Wikipedia: https://en.wikipedia.org/wiki/Enron
F1 score on Wikipedia: https://en.wikipedia.org/wiki/F1_score


```python

```
