import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn import preprocessing

data = load_breast_cancer()
X= data.data # you could also store a subset of the data to variable X, to train your model with
y= data.target
label_names = data.target_names # associated label names
feature_names = data.feature_names # associated data names
print(feature_names)

x_train, x_test, y_train, y_test= train_test_split(X, y, train_size = 0.8, test_size = 0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

# show the importance of each coefficient
coefficients = model.coef_
coefficients = coefficients.tolist()[0]
plt.subplot(2, 1, 1)
plt.bar(range(len(coefficients)), coefficients)

# calculate the cancer probability for every test data
predicted_proba = model.predict_proba(x_test)
predicted_proba = np.reshape(predicted_proba, 228).tolist()
#print(predicted_proba[:10])
#plt.subplot(2, 1, 2)
#plt.plot(range(len(predicted_proba[:20])), predicted_proba[:20])

# classify the test data: cancer or not
predicted = model.predict(x_test)
predicted = predicted.tolist()

# recall are the true positives (actual cancer) divided by the prositive predictions
# the model got right and wrong -> model says negative but is positive + model says positive and its positive
recall = recall_score(predicted, y_test)
print(recall)

plt.subplot(2, 1, 2)
plt.plot(range(len(predicted[:30])), predicted[:30])
plt.plot(range(len(y_test[:30])), y_test[:30])
plt.show()
