import pandas as pd
import joblib

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Definer kollonner
categorical_columns = ['gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 
                       'substancedependence', 'psychologicaldisordermajor', 'depress', 'psychother',
                       'fibrosisandother', 'malnutrition']

numerical_columns = ['rcount', 'hemo', 'hematocrit', 'neutrophils', 'sodium',
                     'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']

test_data = pd.read_csv('test_data.csv', dtype={'gender': str})

# test-data brukte double som 'gender' verdi, så endret dette for å passe modell
test_data.fillna(method='ffill', inplace=True)

if '0.0' in test_data['gender'].unique():
    test_data['gender'] = test_data['gender'].replace({'0.0': 'F'})
if '1.0' in test_data['gender'].unique():
    test_data['gender'] = test_data['gender'].replace({'1.0': 'M'})

# modell ble generert uten facid, så dette måtte fjernes for å kunne kjøre test-data
if 'facid' in test_data.columns:
    test_data.drop('facid', axis=1, inplace=True)

# One-hot encode categorical columns
test_data = pd.get_dummies(test_data, columns=categorical_columns)

# skalering
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

# Save the original 'id' column
original_id = test_data['id'].copy()

# Load the model's feature names
try:
    trained_feature_names = model.feature_names_in_
except AttributeError:
    print("")


# Endre rekkefølge
test_data = test_data[['id'] + list(trained_feature_names)]

# Bare bruk features fra modell
length_of_stay_predictions = model.predict(test_data[trained_feature_names])

result_df = pd.DataFrame({
    'id': original_id,
    'lengthofstay': length_of_stay_predictions
})

# lagre forventet resultat
result_df.to_csv('length_of_stay_predictions3.csv', index=False)

print("Finished")