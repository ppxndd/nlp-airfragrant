from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd
from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from scipy.sparse import hstack


class Sentence(BaseModel):
    sentence: str


# โหลดโมเดลที่บันทึกไว้
# model = joblib.load('rf_model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')
# encoder = joblib.load('encoder.pkl')
thai_stopwords = list(thai_stopwords())
# โหลดโมเดล
model = joblib.load('model.pkl')
# โหลด vectorizer
vectorizer = joblib.load('vectorizer.pkl')
def my_tokenizer(text):
    return text.split(' ')
cvec = joblib.load('cvec.pkl')

# โหลด scaler
scaler = joblib.load('scaler.pkl')
# โหลด encoder_sentiment
encoder_sentiment = joblib.load('encoder_sentiment.pkl')
lr = joblib.load('lr.pkl')
# โหลด encoder
encoder = joblib.load('encoder.pkl')

def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                     if word.lower not in thai_stopwords)
    return final
def getSentiment(my_text):
  my_tokens = text_process(my_text)
  my_bow = cvec.transform(pd.Series([my_tokens]))
  my_predictions = lr.predict(my_bow)
  return my_predictions[0]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือใส่ URL ที่อนุญาตได้
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก HTTP method (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # อนุญาตทุก headers
)

@app.get('/test')
def test():
    return {"result_message":"test"}

# @app.post("/predict_feeling")
# async def predict_feeling(sentence:Sentence):
#     text = [sentence.sentence]
#     X = vectorizer.transform(text)
#     prediction = model.predict(X)
#     predicted_label = encoder.inverse_transform(prediction)
#     feeling = None
#     if (predicted_label[0] == 'c'):
#         feeling = 'วาตะ'
#     elif (predicted_label[0] == 's'):
#         feeling = 'เสมหะ'
#     else:
#         feeling = 'ปิตตะ'
#     return {"result": feeling, "result_message": f"Prediction for '{text[0]}': {predicted_label[0]}"}

@app.post("/predict_feeling")
async def predict_feeling(sentence:Sentence):
    text = [sentence.sentence]
    new_sentence = "เครียด"
    # 🔹 คำนวณความยาวประโยค
    new_sentence_length = len(text)
    # 🔹 แปลงข้อความเป็นตัวเลขด้วย TF-IDF
    new_sentence_tfidf = vectorizer.transform([text])
    # 🔹 แปลง `sentence_length` ให้เป็น Standardized Feature
    new_sentence_length_scaled = scaler.transform([[new_sentence_length]])
    #  🔹 แปลง `sentiment` ของประโยค (ในที่นี้ให้สมมติว่าเป็น 'positive')
    new_sentence_sentiment = encoder_sentiment.transform([getSentiment(new_sentence)]).reshape(-1, 1)
    # 🔹 รวมฟีเจอร์ทั้ง 3 (TF-IDF + sentence_length + sentiment)
    new_sentence_features = hstack((new_sentence_tfidf, new_sentence_length_scaled, new_sentence_sentiment))
    # 🔹 ทำนายผล
    predicted_label = model.predict(new_sentence_features)

    # 🔹 แสดงผลการทำนาย
    predicted_category = encoder.inverse_transform(predicted_label)
    return predicted_category[0]