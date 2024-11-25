import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
# Load sample sensor data (Replace with actual sensor data from smart glasses)
# Data format: Columns - ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'activity']
data = pd.read_csv('sensor_data.csv')

# Separate features and labels
X = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
y = data['activity']

# Encode labels if necessary
y = pd.factorize(y)[0]  # Converts activity names to integers

# Standardize features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Train SVM Classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate both models
svm_preds = svm_model.predict(X_test)
dt_preds = dt_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_preds))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_preds))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_preds))
import joblib

# Save the SVM model for deployment
joblib.dump(svm_model, 'svm_model.pkl')

# Save the Decision Tree model for deployment
joblib.dump(dt_model, 'dt_model.pkl')
# Load a model (e.g., SVM for deployment)
deployed_model = joblib.load('svm_model.pkl')

# Simulated real-time sensor data input
real_time_data = np.array([[0.12, 0.34, 0.56, -0.67, 0.23, -0.45]])  # Replace with actual input
real_time_data_scaled = scaler.transform(real_time_data)

# Predict activity
predicted_activity = deployed_model.predict(real_time_data_scaled)
print("Predicted Activity:", predicted_activity)
