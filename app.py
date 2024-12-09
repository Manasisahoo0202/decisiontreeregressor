from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression  # Replace with your model type
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__) 

# Load the trained model
model = joblib.load('model.pkl')

# Define the features used in the model (including 'Time' but excluding 'Room_Occupancy_Count' and 'Date')
features = [
    'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
    'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
    'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
    'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR', 'Time'  # Add 'Time' here as a feature
]

# Load the dataset
data = pd.read_csv('Occupancy_Estimation.csv')

# Define the feature columns
features = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 
            'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']
target_column = 'Room_Occupancy_Count'  

#all features are present in the dataset
missing_features = [col for col in features if col not in data.columns]
if missing_features:
    print(f"Missing columns in dataset: {missing_features}")
else:
    # Select features and target
    X = data[features]
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train)  # Fit the model with training data

    # Make predictions on the test dataset
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Output metrics
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R² Score:", r2)

@app.route('/')
def index():
    return render_template('index.html', mse=mse, mae=mae, r2=r2)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input features from the form
    input_features = []
    
    for feature in features:
        if feature == 'Time':  # Special case for 'Time' input
            time_value = request.form.get('Time')
            try:
                # Convert Time input (HH:MM:SS) into seconds since midnight
                hours, minutes, seconds = map(int, time_value.split(":"))
                time_in_seconds = hours * 3600 + minutes * 60 + seconds
                input_features.append(time_in_seconds)
            except ValueError:
                return "Invalid time format. Please use HH:MM:SS format."
        else:
            feature_value = request.form.get(feature)
            try:
                input_features.append(float(feature_value))  # Convert all other features to float
            except ValueError:
                return f"Invalid input for {feature}. Please enter numeric values."

    # Ensure the input features list matches the model's expected feature count
    print(f"Input features: {input_features}")

    # Make a prediction using the trained model
    prediction = model.predict([input_features])[0]

    # Return the result to the client
    return render_template('index.html', prediction=prediction, mse=mse, mae=mae, r2=r2)

if __name__ == "__main__":
    app.run(debug=True)
