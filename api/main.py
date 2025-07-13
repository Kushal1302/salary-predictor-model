from fastapi import FastAPI,Request
import numpy as np
import joblib

app = FastAPI()

# Load model and encoders
model = joblib.load("model/model.pkl")
le_country = joblib.load("model/le_country.pkl")
le_education = joblib.load("model/le_education.pkl")

@app.get("/")
def home():
    return {"message":"Welcome to the Salary Predictor"}


@app.post("/predict-salary")
async def predict(request:Request):
    body = await request.json()
    try:
        country = body["country"]
        education = body["education"]
        experience = float(body["experience"])

        x = np.array([[le_country.transform([country])[0],le_education.transform([education])[0],experience]])
        
        prediction = model.predict(x)
        return {"predicted_salary": round(prediction[0], 2)}

    except Exception as e:
        return {"error": str(e)}