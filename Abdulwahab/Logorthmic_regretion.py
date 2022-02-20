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
orgDS = pd.read_csv("../data/music_dataset_cleaned.csv")


#%%
## removing the Unnamed colume made when saving dataset after cleaning it.
orgDS = orgDS.drop('Unnamed: 0',1)

#%%
orgDS.info()

#%%

orgDS.head(5)

#%%
# read the cleaned data after filling the missing 
# music_df = pd.read_csv('./data/music_dataset_cleaned.csv')

# Change the class type to string 
orgDS['Class'] = orgDS.Class.astype(str)

# replacing classes with its actuall labels 
class_map={'0':'Acoustic/Folk','1':'Alt_Music','2':'Blues','3':'Bollywood','4':'Country','5':'HipHop', '6':'IndieAlt','7':'Instrumental','8':'Metal','9':'Pop','10':'Rock'}
orgDS = orgDS
orgDS['genre']= orgDS['Class'].map(class_map)

#
sns.set_theme(style="ticks")
fig_dims = (8, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.histplot(
    orgDS,
    x="genre",hue="genre", 
    multiple="stack",
    ax=ax,
)
plt.xticks(rotation=45)
plt.show()

#%%
## making results and features for training and testing
# y = orgDS.iloc[:,-1].values
# X = orgDS[["duration_in min/ms","acousticness","Popularity","instrumentalness","speechiness","loudness"]].values
# X = orgDS[["duration_in min/ms",
#     "acousticness","Popularity","instrumentalness"]].values
#%%
## all number features
X = orgDS.iloc[:,2:-2].values
y = orgDS.iloc[:,-2].values

# %%
# here we will split the data set into trining and testing
from sklearn.model_selection import train_test_split
class_col = orgDS.Class
X_train, X_test, y_train, y_test = train_test_split(X,
     y,test_size = 0.25,train_size=0.75, random_state = 123,stratify=class_col)

#%%

# Change the class type to string 
# y_train= y_train.astype(str)



#
sns.set_theme(style="ticks")
fig_dims = (8, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.histplot(
    y_train,
    multiple="stack",
    ax=ax,
)

# plt.xticks(rotation=45)

plt.show()

#%%

# Change the class type to string 
y_test=  y_test.astype(str)
#
sns.set_theme(style="ticks")
fig_dims = (8, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.histplot(
     y_test,
    multiple="stack",
    ax=ax,
)
plt.xticks(rotation=45)
plt.show()
# %%

# starting the training
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
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
#%%
# First we import the libs we will use
# from wsgiref.headers import tspecials
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
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
test.info()
#%%
# test.iloc[:,2:]
#%%
## all number features
X_testt = test.iloc[:,2:].values
# y_testt = test.iloc[:,-1].values
#%%
X_testt = sc.transform(X_testt)
#%%
predictions_2 = logistic_regression.predict(X_testt)

# %%
import numpy
(unique, counts) = numpy.unique(predictions_2, return_counts=True)

frequencies = numpy.asarray((unique, counts)).T# %%

# %%
frequencies
# %%
numpy.unique(predictions_2, return_counts=True)
# %%
ynew = logistic_regression.predict_proba(X_testt)
# %%
ynew
# %%
