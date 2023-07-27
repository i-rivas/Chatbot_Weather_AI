import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from dotenv import load_dotenv

# Loads .env file
load_dotenv()

# Assigns .env variable "local_path_url" to path
# You can also just use the actual path that the weather data is in by
# replacing path in df = pd.read_csv(path) with ex. df = pd.read_csv(F:/Users/user1/Documents/weather_data.csv)
path = os.getenv("local_path_url")

# Assume df is your DataFrame with the weather data
df = pd.read_csv(path)

# Convert date to a numerical feature
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(dt.datetime.toordinal)

# Define your features and target variable
X = df[['Date']]
y = df['Average']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predicts tomorrow's average temperature
date_to_predict = dt.datetime.today() + dt.timedelta(days=1)

# Convert the date to the same numerical format as our training data
date_to_predict_ordinal = date_to_predict.toordinal()

# Reshape the date to match the shape of our training data
date_to_predict_ordinal = np.array(date_to_predict_ordinal).reshape(-1, 1)

# Convert numpy array to DataFrame and provide column name
date_to_predict_df = pd.DataFrame(date_to_predict_ordinal, columns=X_train.columns)

# Make the prediction
predicted_temperature = model.predict(date_to_predict_df)

# Convert the ordinal date back to a regular date for printing
date_to_predict = dt.datetime.fromordinal(date_to_predict_ordinal[0][0])

print(f"The predicted average temperature for {date_to_predict.date()} is {round(predicted_temperature[0], 2)} F")


