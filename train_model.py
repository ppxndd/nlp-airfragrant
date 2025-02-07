from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("sentence.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["sentence"])  # แปลงข้อความเป็นตัวเลข
encoder = LabelEncoder()
Y = encoder.fit_transform(df["categorize"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)


# ทำนายผลลัพธ์จากชุดข้อมูลทดสอบ
y_pred = model.predict(X_test)

# # การประเมินผลการทำนาย
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# คำนวณความแม่นยำของโมเดล
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# ระบุ labels ให้ตรงกับคลาสที่มีใน y_test
labels = np.unique(y_test)

# รายงานผลการทดสอบที่ละเอียดขึ้น
print(classification_report(y_test, y_pred, labels=labels, target_names=encoder.classes_))

# # บันทึกโมเดล
# joblib.dump(model, 'rf_model.pkl')

# # บันทึก vectorizer และ encoder เพื่อใช้งานในอนาคต
# joblib.dump(vectorizer, 'vectorizer.pkl')
# joblib.dump(encoder, 'encoder.pkl')


# คำนวณ confusion matrix
cm = confusion_matrix(y_test, y_pred)

# แสดงผลเป็น heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
