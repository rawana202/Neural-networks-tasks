import preprocessing
import Adaline
from sklearn.metrics import accuracy_score


data = preprocessing.load_data("penguins.csv")
X_train, X_test, y_train, y_test = preprocessing.preprocess_data(data, selected_classes=["Adelie", "Chinstrap"])


adaline = Adaline.Adaline(learningRate=0.001, iterationsNumber=100)
adaline.fit(X_train, y_train, mse_threshold=0.01)
predictions = adaline.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy * 100:.2f}%")