import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle


df=pd.read_csv(r"arthritis_dataset.csv")

label_encoder = LabelEncoder()
df["Diagnosis"] = label_encoder.fit_transform(df["Diagnosis"])

print(df)

categorical_features = ['Sex', 'Swelling']

encoder = OneHotEncoder(drop='if_binary',sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_features])

one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_features))

df = pd.concat([df.drop(categorical_features,axis=1),one_hot_df],axis=1)


X = df.drop(columns='Diagnosis')
y = df['Diagnosis']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


numeric_features = [
    'Age', 'RF', 'Anti_CCP', 'ESR', 'CRP', 'Joint_Pain_Score',
    'Morning_Stiffness_Min'
]
categorical_features = ['Sex_M', 'Swelling_Yes']

X_train_num = X_train[numeric_features].copy()
X_test_num = X_test[numeric_features].copy()

X_train_cat = X_train[categorical_features].copy()
X_test_cat = X_test[categorical_features].copy()

num_imputer = SimpleImputer(strategy='median')
X_train_num_imputed = num_imputer.fit_transform(X_train_num)
X_test_num_imputed = num_imputer.transform(X_test_num)

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num_imputed)
X_test_num_scaled = scaler.transform(X_test_num_imputed)


cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat_imputed = cat_imputer.fit_transform(X_train_cat)
X_test_cat_imputed = cat_imputer.transform(X_test_cat)


X_train_final = np.hstack([X_train_num_scaled, X_train_cat_imputed])
X_test_final = np.hstack([X_test_num_scaled, X_test_cat_imputed])


X_train_final.shape,X_test_final.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_final, y_train)
y_pred = rf_model.predict(X_test_final)


cv_scores = cross_val_score(rf_model, X_train_final, y_train, cv=kfold, scoring='accuracy')
cv_scores

print(np.mean(cv_scores))


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))


param_grid_rf = {
    'n_estimators': [50,100, 200],
    'max_depth': [None, 8,10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [0.2,0.5,0.7,1],
}


from sklearn.model_selection import GridSearchCV

rf_grid = GridSearchCV(rf_model,param_grid_rf,cv=5,verbose=2,n_jobs=-1)
rf_grid.fit(X_train_final,y_train)
best_model=rf_grid.best_estimator_
best_model
print(rf_grid.best_score_)

y_pred=best_model.predict(X_test_final)
print(accuracy_score(y_test,y_pred))

rf_grid.best_params_



# with open('model.pkl', 'wb') as f:
#     pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

