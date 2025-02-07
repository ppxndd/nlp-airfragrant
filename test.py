import joblib

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
encoder = joblib.load('encoder.pkl')

text = ["สดใส"]
X = vectorizer.transform(text)
prediction = model.predict(X)
predicted_label = encoder.inverse_transform(prediction)

print(predicted_label)