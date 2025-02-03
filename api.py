from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel


class Sentence(BaseModel):
    sentence: str


# โหลดโมเดลที่บันทึกไว้
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
encoder = joblib.load('encoder.pkl')

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

@app.post("/predict_feeling")
async def predict_feeling(sentence:Sentence):
    text = [sentence.sentence]
    X = vectorizer.transform(text)
    prediction = model.predict(X)
    predicted_label = encoder.inverse_transform(prediction)
    feeling = None
    if (predicted_label[0] == 'c'):
        feeling = 'วาตะ'
    elif (predicted_label[0] == 's'):
        feeling = 'เสมหะ'
    else:
        feeling = 'ปิตตะ'
    return {"result": feeling, "result_message": f"Prediction for '{text[0]}': {predicted_label[0]}"}