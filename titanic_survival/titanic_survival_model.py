import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'passengers.csv')

# Load the passenger data
passengers = pd.read_csv(filename)

# Update sex column to numerical
passengers_age_mean = passengers["Age"].mean()
passengers["Sex"] = passengers["Sex"].map({"female": 1,'male': 0})

# Fill the nan values in the age column
passengers["Age"].fillna(value=passengers_age_mean, inplace=True)

# Create a first class column
passengers["FirstClass"] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers["SecondClass"] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# Select the desired features
features = passengers[["Sex", "Age", "FirstClass", "SecondClass"]]
survival = passengers["Survived"]

# Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival, test_size = 0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and train the model
titanic_model = LogisticRegression()
titanic_model.fit(x_train, y_train)

# Score the model on the train data
train_score = titanic_model.score(x_train, y_train)
print("Training score: ", train_score)

# Score the model on the test data
test_score = titanic_model.score(x_test, y_test)
print("Test score: ", test_score)

# Analyze the coefficients
print("Titanic features coefficients: ", titanic_model.coef_)


# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0,19.0,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
predictions = titanic_model.predict(sample_passengers)
predictions_proba = titanic_model.predict_proba(sample_passengers)
print("Surived or died: ", predictions)
print("Probability of survival, first flaot is the death probability: \n", predictions_proba.tolist())
