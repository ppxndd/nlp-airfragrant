from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel


class Sentence(BaseModel):
    sentence: str


# โหลดโมเดลที่บันทึกไว้
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
encoder = joblib.load('encoder.pkl')
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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