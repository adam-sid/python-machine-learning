import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib as jl
import m2cgen as m2c
from sklearn.metrics import mean_absolute_error, mean_squared_error

VALIDATION_PERCENT = 0.2
NODE_ID = 1
TARGET = 'temperature'

NODE_PREFIX = 'node_' + f'{NODE_ID}'
TARGET_COL = NODE_PREFIX + '_' + TARGET

# Read the csv and make it into pandas dataframe
data_frame = pd.DataFrame(pd.read_csv(NODE_PREFIX + '_training_data.csv'))

# Remove sensor data other than the target
cols_to_remove = ['ts']
for col in data_frame.columns:
    col_name = str(col)
    if col_name != TARGET_COL and col_name.startswith(NODE_PREFIX):
        cols_to_remove.append(col_name)

data_frame = data_frame.drop(columns = cols_to_remove)

# Set the split for training and validation data
cut_off_index = int(round((1- VALIDATION_PERCENT) * len(data_frame), 0))

# Define the data 
training_data = data_frame.iloc[:cut_off_index]
validation_data = data_frame.iloc[cut_off_index:]

# Define the features used for prediction
features = ['day_sin', 'day_cos', 'year_sin', 'year_cos', 'temp', 'pressure', 
'humidity', 'uvi', 'clouds', 'wind_speed', 'wind_gust', 'rain_1h', 'snow_1h']

# Initialise LGBM with following parameters
training_model = lgb.LGBMRegressor(
    n_estimators=250, 
    learning_rate=0.05
)

# Fit the model using features and target data 
training_model.fit(
    training_data[features],
    training_data[TARGET_COL],
    eval_metric='rmse',
    eval_set=[(validation_data[features], validation_data[TARGET_COL])],
    # Stop running training after 50 iterations of no improvement
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)]
)

# If it ends from best iteration get the date from this
best_iteration = getattr(training_model, "best_iteration_", None)
if best_iteration:
    prediction = training_model.predict(validation_data[features], num_iteration=best_iteration) 
else:
    training_model.predict(validation_data[features])

#Console log useful stats
print("Mean absolute error:", mean_absolute_error(validation_data[TARGET_COL], prediction))
print("Root mean squared error:", np.sqrt(mean_squared_error(validation_data[TARGET_COL], prediction)))
print("Features used:", training_model.booster_.feature_name())

#Export final model to JS
javascript_final_model = m2c.export_to_javascript(training_model)

with open(f'{TARGET}{NODE_ID}.js', 'w') as f:
    f.write(javascript_final_model)