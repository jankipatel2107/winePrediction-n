import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pickle

wine = pd.read_csv('TrainingDataset.csv', delimiter = ";")
wine.head()
wine.info()

bins = [0, 5.5, 7.5, 10]
labels = [0, 1, 2]
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)
wine.head()

for key in wine:
    print(key, wine[key].min(), wine[key].max())

X_train = wine.drop('quality', axis = 1)
y_train = wine['quality']
 
test_data = pd.read_csv('ValidationDataset.csv', delimiter = ";")
X_test = test_data.drop('quality', axis = 1)
test_data['quality'] = pd.cut(test_data['quality'], bins=bins, labels=labels)
y_test = test_data['quality']

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

rfc_model = RandomForestClassifier(n_estimators=200)
print(type(rfc_model))
rfc_model.fit(X_train, y_train)
pred_rfc = rfc_model.predict(X_test)

cross_val = cross_val_score(estimator = rfc_model, X = X_train, y = y_train, cv = 10 )
print(classification_report(y_test, pred_rfc))
print(cross_val.mean())

filename = 'winePredictionModel.sav'
pickle.dump(rfc_model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
print(type(loaded_model))


d = {'fixed acidity': [8.9], 
     'volatile acidity': [0.22],  
     'citric acid':[0.48],  
     'residual sugar':[1.8], 
     'chlorides':[0.077], 
     'free sulfur dioxide': [29.0],
     'total sulfur dioxide': [60.0],
     'density': [0.9968],
     'pH':[3.39],
     'sulphates':[0.53],
     'alcohol':[9.4],
      }
print(type(d))
df = pd.DataFrame(data=d)
print(df.head())
print(type(df['pH']))

loaded_model.predict(df)