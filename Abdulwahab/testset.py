# Logarthmic regretion:
'''
this code will create a model to predect the class of a song based on ->
some variables

'''

#%%
# First we import the libs we will use
from wsgiref.headers import tspecials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
##Loading the CSV file
test = pd.read_csv("../data/test_set/music_test.csv")


#%%
## removing the Unnamed colume made when saving dataset after cleaning it.
# orgDS = orgDS.drop('Unnamed: 0',1)
test.info()
#%%

test.head(5)

#%%
## making results and features for training and testing
# y = orgDS.iloc[:,-1].values
# X = orgDS[["duration_in min/ms","acousticness","Popularity","instrumentalness","speechiness","loudness"]].values
# X = orgDS[["duration_in min/ms",
    # "acousticness","Popularity","instrumentalness"]].values

#
#%%
test.isnull().sum()

#%%
test["Popularity"].fillna(test.Popularity.mean(), inplace=True)
from sklearn.impute import KNNImputer

kn = KNNImputer(n_neighbors=6)
test['instrumentalness']=kn.fit_transform(test[['instrumentalness']])
test['key']=kn.fit_transform(test[['key']])
#%%
## all number features
X = test.iloc[:,2:-1].values
y = test.iloc[:,-1].values

# %%
# here we will split the data set into trining and testing
from sklearn.model_selection import train_test_split
# class_col = test.Class
X_train, X_test, y_train, y_test = train_test_split(X,
     y,test_size = 0.25,train_size=0.75, random_state = 123,stratify=class_col)
# %%
# starting the training
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
# %%
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
# %%
predictions = logistic_regression.predict(X_test)
# %%
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#%%
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = logistic_regression.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
# %%

# %%
test.info()
# %%
