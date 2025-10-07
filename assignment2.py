# Our import statements for this problem
import pandas as pd
import numpy as np
import patsy as pt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


xgb = XGBClassifier(
    n_estimators=50, 
    max_depth=3,
    learning_rate=0.5, 
    objective='multi:softmax')

# The code to implement a decision tree
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
new_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

y = data['meal']
x = data.drop(['meal','id','DateTime'],axis=1)


yt = new_data['meal']
xt = new_data.drop(['meal','id','DateTime'],axis=1)


# If we have imported data and created x, y already:
kf = KFold(n_splits=10) # 10 "Folds"

models = [] # We will store our models here
num_classes = len(np.unique(y))
model = XGBClassifier(
            n_estimators=150, 
            max_depth=15,
            learning_rate=0.5, 
            objective='multi:softmax',
            num_class=num_classes)

#model = RF(n_estimators=150, n_jobs=-1,  max_depth=10, random_state=42)

for train, test in kf.split(x): # Iterate over folds

    modelFit = model.fit(x.values[train], y.values[train]) # Fit model
    accuracy = accuracy_score(y.values[test],    # Store accuracy
    modelFit.predict(x.values[test]))
    print("Accuracy: ", accuracy)            # Print results
    models.append([modelFit, accuracy])      # Store it all

print("Mean Model Accuracy: ", np.mean([model[1] for model in models[1:]]))
print("Model Accuracy Standar Deviation: ", np.std([model[1] for model in models[1:]]))


# Make predictions based on the testing x values
pred = modelFit.predict(xt)