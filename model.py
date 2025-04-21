import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# dataset from here: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

# mitbih
# MIT-BIH Arrhythmia Database
# 109,446 samples
# 5 categories of heartbeats
# 'N': Normal Beats, 'S': Supraventricular Ectopy Beats, 'V': Ventricular Ectopy Beats, 'F': Fusion Beats, 'Q': Unclassifiable Beats

train_mitbih = pd.read_csv('mitbih_train.csv', header =None)
test_mitbih = pd.read_csv('mitbih_test.csv', header = None)

# the last column holds the actual classification so that goes in y
X_train_mitbih = train_mitbih.iloc[:, :-1].values
y_train_mitbih = train_mitbih.iloc[:, -1].values
X_test_mitbih = test_mitbih.iloc[:, :-1].values
y_test_mitbih = test_mitbih.iloc[:, -1].values

class_labels = train_mitbih[train_mitbih.shape[1]-1].unique() # there are 4 possible labels. Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
# 'Normal Beats','Supraventricular Ectopy Beats','Ventricular Ectopy Beats','Fusion Beats','Unclassifiable Beats'
class_labels

class_names = {0: 'Normal Beats', 1: 'Supraventricular Ectopy Beats', 2: 'Ventricular Ectopy Beats', 3: 'Fusion Beats', 4: 'Unclassifiable Beats'}

fig, axes = plt.subplots(5, 1, figsize=(20, 25), sharex=True)

for idx, label in enumerate(class_names):
  signal = train_mitbih[train_mitbih.iloc[:, -1] == label].iloc[0, :-1].values
  ax = axes[idx]
  ax.plot(signal)
  ax.set_title(class_names[label])
  ax.set_ylabel("Amplitude")

axes[-1].set_xlabel("Time")
plt.show()


rf_mitbih = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_mitbih.fit(X_train_mitbih, y_train_mitbih)
y_pred_mitbih = rf_mitbih.predict(X_test_mitbih)

print('Accuracy: ', accuracy_score(y_test_mitbih, y_pred_mitbih)) # Accuracy of 0.97469
print('Classification report: ', classification_report(y_test_mitbih, y_pred_mitbih))


# ptbdb 
# PTB Diagnostic ECG Database
# 14,552 samples
# 2 categories: Normal and Abnormal
# very last column holds classification (0 for normal, 1 for abnormal)

abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)
normal = pd.read_csv('ptbdb_normal.csv', header=None)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

for i in range(3):
  ax1.plot(normal.iloc[i, :-1].values, label = f'Sample {i+1}')
ax1.set_title('Normal ECG Signals')
ax1.set_ylabel('Amplitude')
ax1.legend()
  
for i in range(3):
  ax2.plot(abnormal.iloc[i, :-1].values, label = f'Sample {i+1}')
ax2.set_title('Abnormal ECG Signals')
ax2.set_ylabel('Amplitude')
ax2.legend()

ax2.set_xlabel('Time')
plt.show()


ptb = pd.concat([abnormal, normal], axis = 0)
ptb = ptb.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle the dataset
X = ptb.iloc[:, :-1].values
y = ptb.iloc[:, -1].values

X_train_ptb, X_test_ptb, y_train_ptb, y_test_ptb = train_test_split(X, y, test_size=0.2, random_state=42)

rf_ptb = RandomForestClassifier(n_estimators=100, random_state=42)
rf_ptb.fit(X_train_ptb, y_train_ptb)
y_pred_ptb = rf_ptb.predict(X_test_ptb)
print('Accuracy: ', accuracy_score(y_test_ptb, y_pred_ptb)) # Accuracy of 0.97698
print('Classification report: ', classification_report(y_test_ptb, y_pred_ptb))


