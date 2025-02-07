from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score


df = pd.read_csv("sentence.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["sentence"])  # แปลงข้อความเป็นตัวเลข
encoder = LabelEncoder()
Y = encoder.fit_transform(df["categorize"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# บันทึกโมเดล
joblib.dump(model, 'rf_model.pkl')

# บันทึก vectorizer และ encoder เพื่อใช้งานในอนาคต
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(encoder, 'encoder.pkl')

# y_pred = model.predict(X_test)

# # คำนวณความแม่นยำ
# accuracy = accuracy_score(y_test, y_pred)

# # แสดงผลลัพธ์
# print(f"Accuracy: {accuracy * 100:.2f}%")