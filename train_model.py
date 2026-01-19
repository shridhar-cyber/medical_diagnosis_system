import pandas as pd

# Load dataset
data = pd.read_csv("data/diabetes.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Show dataset info
print("\nDataset Info:")
print(data.info())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Show basic statistics
print("\nBasic Statistics:")
print(data.describe())

from sklearn.model_selection import train_test_split




X = data.drop("Outcome", axis=1)  
y = data["Outcome"]             


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nNumber of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy on Test Data: {:.2f}%".format(accuracy * 100))

joblib.dump(model, "model/trained_model.pkl")
print("\nTrained model saved as 'model/trained_model.pkl'")
