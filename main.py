import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from utilitys import Utilitys
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')

utility = Utilitys()
df_processed = utility.preprocess(df)

#Splitting  data into training, test and validation sets
X = df_processed.loc[:, df_processed.columns != 'Outcome']
y = df_processed.loc[:, 'Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

model = Sequential()

#Add hidden Layers
model.add(Dense(32, activation = 'relu', input_dim = 8))
model.add(Dense(16, activation = 'relu'))

#Add output Layers
model.add(Dense(1, activation = 'sigmoid'))

#Model compilation
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Model training
model.fit(X_train, y_train, epochs = 200)

#Testing accuracy
scores = model.evaluate(X_train, y_train)
print("Training Accuracy : %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test)
print("Testing Accuracy : %.2f%%\n" % (scores[1]*100))

#Confusion matrix
y_test_pred_prob = model.predict(X_test)
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

c_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(c_matrix, annot=True, yticklabels=['No Diabetes', 'Diabetes'], xticklabels=['No Diabetes', 'Diabetes'], cbar=False, cmap='Blues', fmt='d')
ax.set_ylabel('Actual')
ax.set_xlabel('Prediction')
plt.title('Confusion Matrix')
plt.show()

#ROC curve
y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR,TPR)
plt.plot([0,1],[0,1],'--', color = 'black')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

