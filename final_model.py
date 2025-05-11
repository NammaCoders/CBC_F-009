import joblib  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

dataSet = pd.read_csv("diabetes.csv")

# Replace 0 with median for required columns
columns_to_replace = ["Pregnancies", "Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
for col in columns_to_replace:
    dataSet[col] = dataSet[col].replace(0, dataSet[col].median())

# Only select the 5 columns you're using in HTML form
X = dataSet[["Pregnancies", "Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]]
y = dataSet["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale only these selected columns
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(XGBClassifier(), param_grid, scoring='accuracy', cv=5)
grid.fit(X_train_scaled, y_train)
print("Best XGBoost Accuracy:", grid.best_score_)
print("Best Params:", grid.best_params_)




# Save model and scaler
best_model = grid.best_estimator_
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(sc, "scaler.pkl")

print("Model and Scaler saved successfully 🚀🔥")





