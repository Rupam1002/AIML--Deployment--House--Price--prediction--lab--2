from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

from preprocessing import load_and_split

# load and split the dataset
X_train, X_test, y_train, y_test = load_and_split()

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"model MSE: {mse}")
joblib.dump(model, "bostom_model.pkl")
print("model saved successfully")


