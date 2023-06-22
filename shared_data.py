import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split , learning_curve , GridSearchCV
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
data = pd.read_excel('dataset.xlsx')
data =data.drop('Patient ID', axis=1)
columns = data.columns.tolist()
float_columns = data.select_dtypes(include=['float']).columns.tolist()

missing_rate = data.isna().sum()/data.shape[0]
blood_columns = list(data.columns[(missing_rate < 0.9) & (missing_rate >0.88)])
viral_columns = list(data.columns[(missing_rate < 0.80) & (missing_rate > 0.75)])
key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']
data_prep = data[key_columns + blood_columns + viral_columns]


obj_code = {'negative':0,
            'positive':1,
            'not_detected':0,
            'detected':1}

for col in data_prep.select_dtypes('object').columns:
       data_prep.loc[:, col] = data_prep[col].map(obj_code).astype(float)

columns_prep = data.columns.tolist()
float_columns_prep = data.select_dtypes(include=['float']).columns.tolist()

data_prep = data_prep.dropna(axis=0)
 # Split the data into features (X) and target (y)
y = data_prep["SARS-Cov-2 exam result"]
X = data_prep.drop("SARS-Cov-2 exam result", axis=1)  
   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.astype(np.int64)
X_test = X_test.astype(np.int64)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
# Train the models
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
ypred_tree = tree.predict(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
ypred_knn = knn.predict(X_test)

randomF = RandomForestClassifier(random_state=0)
randomF.fit(X_train, y_train)
ypred_randomF = randomF.predict(X_test)

# Generate classification reports
tree_report = classification_report(y_test, ypred_tree)
knn_report = classification_report(y_test, ypred_knn)
randomF_report = classification_report(y_test, ypred_randomF)

# Create the classification report plots
tree_fig = go.Figure(data=[go.Table(header=dict(values=["Classification Report - Decision Tree"]), cells=dict(values=[tree_report.split("\n")]))])
knn_fig = go.Figure(data=[go.Table(header=dict(values=["Classification Report - KNN"]), cells=dict(values=[knn_report.split("\n")]))])
randomF_fig = go.Figure(data=[go.Table(header=dict(values=["Classification Report - Random Forest"]), cells=dict(values=[randomF_report.split("\n")]))])




feature_importances = randomF.feature_importances_

importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot the feature importances
fig_feature_importance = go.Figure(data=[go.Bar(x=importance_df['Feature'], y=importance_df['Importance'])])
fig_feature_importance.update_layout(title="Feature Importance: Random Forest")
threshold = 0.05

# Filter the important feature names based on the threshold
important_features = importance_df.loc[importance_df['Importance'] >= threshold, 'Feature']

# Filter the columns of X_train and X_test based on important features
X_train_filtered = X_train[important_features]
X_test_filtered = X_test[important_features]


randomF_best = RandomForestClassifier(random_state=0)
randomF_best.fit(X_train_filtered, y_train)
ypred_randomF_best = randomF_best.predict(X_train_filtered)

randomF_report_best = classification_report(y_train, ypred_randomF_best)
randomF_fig_best = go.Figure(data=[go.Table(header=dict(values=["Classification Report - Random Forest"]), cells=dict(values=[randomF_report_best.split("\n")]))])

N, train_score, val_score = learning_curve(tree, X_train, y_train,cv=4, scoring='f1',train_sizes=np.linspace(0.1, 1, 10))

fig = go.Figure()
fig.add_trace(go.Scatter(x=N, y=train_score.mean(axis=1), mode='lines', name='train score'))
fig.add_trace(go.Scatter(x=N, y=val_score.mean(axis=1), mode='lines', name='validation score'))
fig.update_layout(title="Learning Curve: Decision Tree",title_x=0.5)
N, train_score, val_score = learning_curve(knn, X_train, y_train,cv=4, scoring='f1',train_sizes=np.linspace(0.1, 1, 10))

fig_knn= go.Figure()
fig_knn.add_trace(go.Scatter(x=N, y=train_score.mean(axis=1), mode='lines', name='train score'))
fig_knn.add_trace(go.Scatter(x=N, y=val_score.mean(axis=1), mode='lines', name='validation score'))
fig_knn.update_layout(title="Learning Curve: knn Tree",title_x=0.5)
N, train_score, val_score = learning_curve(randomF_best, X_train, y_train,cv=4, scoring='f1',train_sizes=np.linspace(0.1, 1, 10))

fig_randomF= go.Figure()
fig_randomF.add_trace(go.Scatter(x=N, y=train_score.mean(axis=1), mode='lines', name='train score'))
fig_randomF.add_trace(go.Scatter(x=N, y=val_score.mean(axis=1), mode='lines', name='validation score'))
fig_randomF.update_layout(title="Learning Curve: Random Forest",title_x=0.5)