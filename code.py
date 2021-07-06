#!/usr/bin/env python
# coding: utf-8

# In[214]:


#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


# In[215]:


#IMPORTING THE DATASET
a_rec = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/application.csv")
print(a_rec.shape)
c_rec = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/credit_record.csv")
#print(c_rec.head(5))


# In[216]:


#CHECKING FOR NULL VALUES
a_rec.isnull().sum()


# In[217]:


#DROPPING THE NULL VALUES
a_rec.drop('OCCUPATION_TYPE',axis = 1,inplace = True)


# In[218]:


#DIMENSION CHECK
print(a_rec.shape)


# In[219]:


#DROPPING DUPLICATE ROWS
a_rec = a_rec.drop_duplicates('ID', keep='last')
a_rec


# In[220]:


#PRINTING COLUMNS WITH CATEGORICAL DATA
cat_col = [i for i in a_rec.columns if a_rec.dtypes[i]=='object']
print(cat_col)


# In[221]:


#PRINTING COLUMNS WITH NUMERIAL DATA
num_col = [i for i in a_rec.columns if a_rec.dtypes[i]!='object']
print(num_col)


# In[222]:


#CONVERTING AGE FROM DAYS TO YEARS
a_rec['DAYS_BIRTH'] = round(a_rec['DAYS_BIRTH']/-365,0)
a_rec.rename(columns={'DAYS_BIRTH':'AGE_YEARS'}, inplace=True)


# In[223]:


#CHECKING FOR UNEMPLOYED PEOPLE
a_rec[a_rec['DAYS_EMPLOYED']>0]['DAYS_EMPLOYED'].unique()


# In[224]:


#RELACING UNEMLYED WITH 0 IN DAYS EMPLOYED COLUMN 
a_rec['DAYS_EMPLOYED'].replace(365243, 0, inplace=True)


# In[225]:


#CONVERTING DAYS EMPLYED IN YEARS EMPLOYED
a_rec['DAYS_EMPLOYED'] = abs(round(a_rec['DAYS_EMPLOYED']/-365,0))
a_rec.rename(columns={'DAYS_EMPLOYED':'YEARS_EMPLOYED'}, inplace=True) 


# In[226]:


#CHECKING COUNT OF UNIQUE VALUES IN COLUMNS
a_rec['FLAG_MOBIL'].value_counts()


# In[227]:


# As all the values in column are 1, hence dropping column
a_rec.drop('FLAG_MOBIL', axis=1, inplace=True)


# In[228]:


#CHECKING COUNT OF UNIQUE VALUES IN COLUMN
a_rec['FLAG_WORK_PHONE'].value_counts()


# In[229]:


#CHECKING COUNT OF UNIQUE VALUES IN COLUMN
a_rec['FLAG_PHONE'].value_counts()


# In[230]:


#CHECKING COUNT OF UNIQUE VALUES IN COLUMN
a_rec['FLAG_EMAIL'].value_counts()


# In[231]:


#CHECKING COUNT OF UNIQUE VALUES IN COLUMN
a_rec['CNT_FAM_MEMBERS'].value_counts()


# In[232]:


#PRINTING THE  RECORD COLUMN
c_rec.head(5)


# In[233]:


#CHECKING COUNT OF  UNIQUE VALUES IN COLUMN
c_rec['STATUS'].value_counts()


# In[234]:


# categorizing 'STATUS' column to binary classification   0 : Good Client and 1 : bad client
c_rec['STATUS'].replace(['C', 'X'],0, inplace=True)


# In[235]:


c_rec['STATUS'].replace(['2','3','4','5'],1, inplace=True)


# In[236]:


c_rec['STATUS'].value_counts()


# In[237]:


c_rec['STATUS'] = c_rec['STATUS'].astype('int')


# In[238]:


c_rec['STATUS'].value_counts()


# In[239]:


c_rec['STATUS'].value_counts(normalize=True)*100


# In[240]:


c_rec_new = c_rec.groupby('ID').agg(max).reset_index()


# In[241]:


c_rec_new


# In[242]:


#DROPPING THE MONTHS BALANCE COLUMN FROM C_RECORD
c_rec_new.drop('MONTHS_BALANCE', axis=1, inplace=True)
c_rec_new.head()


# In[243]:


#VISUALIZATION OF APPLICATION RECORD


# In[244]:


sns.boxplot(a_rec['CNT_CHILDREN'])


# In[245]:


a_rec['CNT_CHILDREN'].describe()


# In[246]:


sns.boxplot(a_rec['AMT_INCOME_TOTAL'])


# In[247]:


a_rec['AMT_INCOME_TOTAL'].describe()


# In[248]:


sns.boxplot(a_rec['AGE_YEARS'])


# In[ ]:





# In[249]:


sns.boxplot(a_rec['YEARS_EMPLOYED'])


# In[250]:


a_rec['YEARS_EMPLOYED'].describe()


# In[251]:


sns.boxplot(a_rec['CNT_FAM_MEMBERS'])


# In[252]:



a_rec['FLAG_EMAIL'].describe()


# In[253]:


#REMOVING OURLIERS


# In[254]:


a_rec.shape


# In[255]:


high_bound = a_rec['CNT_CHILDREN'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = a_rec['CNT_CHILDREN'].quantile(0.001)
print('low_bound :', low_bound)


# In[256]:


a_rec = a_rec[(a_rec['CNT_CHILDREN']>=low_bound) & (a_rec['CNT_CHILDREN']<=high_bound)]


# In[257]:


high_bound = a_rec['AMT_INCOME_TOTAL'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = a_rec['AMT_INCOME_TOTAL'].quantile(0.001)
print('low_bound :', low_bound)


# In[258]:


a_rec = a_rec[(a_rec['AMT_INCOME_TOTAL']>=low_bound) & (a_rec['AMT_INCOME_TOTAL']<=high_bound)]


# In[259]:


a_rec.shape


# In[260]:


high_bound = a_rec['YEARS_EMPLOYED'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = a_rec['YEARS_EMPLOYED'].quantile(0.001)
print('low_bound :', low_bound)


# In[261]:


a_rec = a_rec[(a_rec['YEARS_EMPLOYED']>=low_bound) & (a_rec['YEARS_EMPLOYED']<=high_bound)]


# In[262]:


high_bound = a_rec['CNT_FAM_MEMBERS'].quantile(0.999)
print('high_bound :', high_bound)
low_bound = a_rec['CNT_FAM_MEMBERS'].quantile(0.001)
print('low_bound :', low_bound)


# In[263]:


a_rec = a_rec[(a_rec['CNT_FAM_MEMBERS']>=low_bound) & (a_rec['CNT_FAM_MEMBERS']<=high_bound)]


# In[264]:


a_rec


# In[265]:


#MERGING DATA FRAMES


# In[266]:


final_data = pd.merge(a_rec, c_rec_new, on='ID', how='inner')
final_data


# In[267]:


final_data.drop('ID', axis=1, inplace=True)


# In[268]:


#DROPPING DUPLICATE RECORDS FROM FINAL DATASET


# In[269]:



final_data = final_data.drop_duplicates()
final_data.reset_index(drop=True ,inplace=True)


# In[270]:


final_data.shape


# In[271]:


#CHECKING FOR NULL VALUES IN FINAL DATA
final_data.isnull().sum()


# In[272]:


#VISUALIZING THE FINAL CLEANED DATA


# In[273]:


#DRAWING THE CORRELATION MATRIX
sns.heatmap(final_data.corr(), annot=True)


# In[274]:


#FEATURE
final_data


# In[275]:


#FEATURE SCALING
cat_columns = final_data.columns[(final_data.dtypes =='object').values].tolist()
cat_columns


# In[276]:


#CONVERTING ALL NON NUMERICAL DATA TO NUMERICAL DATA
from sklearn.preprocessing import LabelEncoder

for col in cat_columns:
        globals()['LE_{}'.format(col)] = LabelEncoder()
        final_data[col] = globals()['LE_{}'.format(col)].fit_transform(final_data[col])
final_data.head() 


# In[277]:


#CORRELATION MATRIX OF FINAL DATA
final_data.corr()


# In[278]:


#SEPARATING RESPONSE AND INDEPENDENT VARIABLES
x = final_data.drop(['STATUS'], axis=1)
y = final_data['STATUS']


# In[279]:


print(x.head())
y.head()


# In[280]:


#FITTING MACHINE LEARNING MODELS


# In[281]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state = 0)


# In[282]:


# fitting Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

log_model = LogisticRegression()
log_model.fit(x_train, y_train)

print('Logistic Model Accuracy : ', log_model.score(x_test, y_test)*100, '%')

prediction = log_model.predict(x_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test, prediction))


# In[283]:


# Decision Tree classification

from sklearn.tree import DecisionTreeClassifier

decision_model = DecisionTreeClassifier(max_depth=12,min_samples_split=8)

decision_model.fit(x_train, y_train)

print('Decision Tree Model Accuracy : ', decision_model.score(x_test, y_test)*100, '%')

prediction = decision_model.predict(x_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test, prediction))


# In[284]:


# Random Forest classification

from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier(n_estimators=250,
                                            max_depth=12,
                                            min_samples_leaf=16)

RandomForest_model.fit(x_train, y_train)

print('Random Forest Model Accuracy : ', RandomForest_model.score(x_test, y_test)*100, '%')

prediction = RandomForest_model.predict(x_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test, prediction))


# In[285]:


# Support Vector Machine classification

from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(x_train, y_train)

print('Support Vector Classifier Accuracy : ', svc_model.score(x_test, y_test)*100, '%')

prediction = svc_model.predict(x_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test, prediction))


# In[286]:


# K Nearest Neighbor classification

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 7)

knn_model.fit(x_train, y_train)

print('KNN Model Accuracy : ', knn_model.score(x_test, y_test)*100, '%')

prediction = knn_model.predict(x_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test, prediction))


# In[287]:


# XGBoost  classification


from xgboost import XGBClassifier

XGB_model = XGBClassifier()

XGB_model.fit(x_train, y_train)

print('XGBoost Model Accuracy : ', XGB_model.score(x_test, y_test)*100, '%')

prediction = XGB_model.predict(x_test)
print('\nConfusion matrix :')
print(confusion_matrix(y_test, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test, prediction))


# In[288]:


#BALANCING DATASET


# In[289]:


# scaling all features
from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
x_train_scaled = pd.DataFrame(MMS.fit_transform(x_train), columns=x_train.columns)
x_test_scaled = pd.DataFrame(MMS.transform(x_test), columns=x_test.columns)


# In[290]:


# adding samples to minority class using SMOTE
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

x_train_oversam, y_train_oversam = oversample.fit_resample(x_train_scaled, y_train)
x_test_oversam, y_test_oversam = oversample.fit_resample(x_test_scaled, y_test)


# In[291]:


# Original majority and minority class
y_train.value_counts(normalize=True)*100


# In[292]:


#converting array to series
y_train_oversam=pd.Series(y_train_oversam)


# In[293]:


# after using SMOTE 
y_train_oversam.value_counts(normalize=True)*100


# In[294]:


#MACHINE LEARNING MODEL AFTER BALANCING


# In[295]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

log_model = LogisticRegression()
log_model.fit(x_train_oversam, y_train_oversam)

print('Logistic Model Accuracy : ', log_model.score(x_test_oversam, y_test_oversam)*100, '%')

prediction = log_model.predict(x_test_oversam)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_oversam, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test_oversam, prediction))


# In[296]:


# Decision Tree classification

from sklearn.tree import DecisionTreeClassifier

decision_model = DecisionTreeClassifier(max_depth=12,min_samples_split=8)

decision_model.fit(x_train_oversam, y_train_oversam)

print('Decision Tree Model Accuracy : ', decision_model.score(x_test_oversam, y_test_oversam)*100, '%')

prediction = decision_model.predict(x_test_oversam)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_oversam, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test_oversam, prediction))


# In[297]:


# Random Forest classification

from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier(n_estimators=250,
                                            max_depth=12,
                                            min_samples_leaf=16)

RandomForest_model.fit(x_train_oversam, y_train_oversam)

print('Random Forest Model Accuracy : ', RandomForest_model.score(x_test_oversam, y_test_oversam)*100, '%')

prediction = RandomForest_model.predict(x_test_oversam)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_oversam, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test_oversam, prediction))


# In[298]:


# Support Vector Machine classification

from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(x_train_oversam, y_train_oversam)

print('Support Vector Classifier Accuracy : ', svc_model.score(x_test_oversam, y_test_oversam)*100, '%')

prediction = svc_model.predict(x_test_oversam)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_oversam, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test_oversam, prediction))


# In[299]:


# K Nearest Neighbor classification

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 7)

knn_model.fit(x_train_oversam, y_train_oversam)

print('KNN Model Accuracy : ', knn_model.score(x_test_oversam, y_test_oversam)*100, '%')

prediction = knn_model.predict(x_test_oversam)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_oversam, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test_oversam, prediction))


# In[300]:


# XGBoost  classification

from xgboost import XGBClassifier

XGB_model = XGBClassifier()

XGB_model.fit(x_train_oversam, y_train_oversam)

print('XGBoost Model Accuracy : ', XGB_model.score(x_test_oversam, y_test_oversam)*100, '%')

prediction = XGB_model.predict(x_test_oversam)
print('\nConfusion matrix :')
print(confusion_matrix(y_test_oversam, prediction))
      
print('\nClassification report:')      
print(classification_report(y_test_oversam, prediction))


# In[301]:


#VALIDATION


# In[302]:


#K-Fold Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(5)


# In[303]:


# Logistic Regression

results=cross_val_score(log_model,x,y,cv=kfold)
print(results*100,'\n')

print(np.mean(results)*100)


# In[304]:


# Decision Tree classification

results=cross_val_score(decision_model,x,y,cv=kfold)
print(results*100,'\n')

print(np.mean(results)*100)


# In[305]:


# Random Forest classification

results=cross_val_score(RandomForest_model,x,y,cv=kfold)
print(results*100,'\n')

print(np.mean(results)*100)


# In[306]:


# Support Vector Machine classification

results=cross_val_score(svc_model,x,y,cv=kfold)
print(results*100,'\n')

print(np.mean(results)*100)


# In[307]:


# K Nearest Neighbor classification

results=cross_val_score(knn_model,x,y,cv=kfold)
print(results*100,'\n')

print(np.mean(results)*100)


# In[308]:


# XGBoost classification

results=cross_val_score(XGB_model,x,y,cv=kfold)
print(results*100,'\n')

print(np.mean(results)*100)


# In[309]:


#STRATIFIED SHUFFLE SPLIT


# In[310]:


from sklearn.model_selection import StratifiedShuffleSplit
ssplit=StratifiedShuffleSplit(n_splits=5,test_size=0.30)


# In[311]:


# Logistic Regression

results=cross_val_score(log_model,x,y,cv=ssplit)
print(results*100,'\n')

print(np.mean(results)*100)


# In[312]:


# Decision Tree classification

results=cross_val_score(decision_model,x,y,cv=ssplit)
print(results*100,'\n')

print(np.mean(results)*100)


# In[313]:


# Random Forest classification

results=cross_val_score(RandomForest_model,x,y,cv=ssplit)
print(results*100,'\n')

print(np.mean(results)*100)


# In[314]:


# Support Vector Machine classification

results=cross_val_score(svc_model,x,y,cv=ssplit)
print(results*100,'\n')

print(np.mean(results)*100)


# In[315]:


# K Nearest Neighbor classification

results=cross_val_score(knn_model,x,y,cv=ssplit)
print(results*100,'\n')

print(np.mean(results)*100)


# In[316]:


# XGBoost classification

results=cross_val_score(XGB_model,x,y,cv=ssplit)
print(results*100,'\n')

print(np.mean(results)*100)


# In[317]:


#CONCLUSION
#We predict that XG boost model is best for the credit card approval prediction with accuracy of 83.73219373219372 %

