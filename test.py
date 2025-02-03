import joblib

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
encoder = joblib.load('encoder.pkl')

text = ["ง่วงนอน"]
X = vectorizer.transform(text)
prediction = model.predict(X)
predicted_label = encoder.inverse_transform(prediction)

print(predicted_label)