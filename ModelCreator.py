import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
from scipy.stats import randint


print("Laster data")

train_data = pd.read_csv('training_data.csv', parse_dates=['vdate'])
metadata = pd.read_csv('metadata.csv')



print("Forhåndsprosesserer")

train_data.fillna(method='ffill', inplace=True)

# kollonner
categorical_columns = ['gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 
                       'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother',
                       'fibrosisandother', 'malnutrition']
numerical_columns = ['rcount', 'hemo', 'hematocrit', 'neutrophils', 'sodium',
                     'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']


train_data = pd.get_dummies(train_data, columns=categorical_columns)


scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])


selected_features = train_data.columns.difference(['id', 'vdate', 'discharged', 'facid', 'lengthofstay'])
X = train_data[selected_features]
y = train_data['lengthofstay']

print("splitter datasett")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Trener modell")
# Initialize and Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_val)
print("R2 Score:", r2_score(y_val, y_pred))

# Define Parameter Distributions
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [None] + list(randint(10, 30).rvs(size=3)),
    'max_features': ['sqrt', 'log2'],
}

print("Randomized search starter")

random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=3, scoring='r2', n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)

print("Modellagring")
# Best Model
best_model = random_search.best_estimator_

# Save the best model 
joblib.dump(best_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')




print("Finished")