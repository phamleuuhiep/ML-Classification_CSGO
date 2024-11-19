# Report: 
# I prefer using recall metric as my primary goal is capturing as many as actual win as possible. When rendering 
# the ProfileReport using ydata_profiling, it appears that the library does not render the any correlation matrices. By googling, 
# it is highly the reason that the correlations are below a threshold like 0.1 by default, so they might not be shown in the report.
# Therefore, I conclude the dataset involves classification problem does not have any linear relations. That's the reason why SVC is 
# quite good for training.
#             precision    recall  f1-score   support

#         Lost       0.70      0.80      0.75       110
#          Tie       0.00      0.00      0.00        14
#          Win       0.75      0.74      0.75       103

#     accuracy                           0.72       227
#    macro avg       0.48      0.51      0.50       227
# weighted avg       0.68      0.72      0.70       227

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("csgo.csv")
# profile = ProfileReport(data, title="CSGO Report", explorative=True)
# profile.to_file("Report CSGO.html")

# Split data
data = data.drop(["day","month","year","date","wait_time_s","match_time_s","team_a_rounds","team_b_rounds"],axis=1)

# One-hot encode
data = pd.get_dummies(data, columns=["map"])

# Split
target = "result"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=27)

# Preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
model = SVC()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))
cm = np.array(confusion_matrix(y_test, y_predict))
confusion = pd.DataFrame(cm, index=["Win","Lost","Tie"], columns=["Win","Lost","Tie"])
sns.heatmap(confusion, annot=True)
plt.show()