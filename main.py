from fastapi import staticfiles
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException,Request,Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from fastapi.templating import Jinja2Templates

app=FastAPI()
model=joblib.load('spamemail.joblib')
convertor=joblib.load('converter.joblib')
app.mount("/static", StaticFiles(directory="static"), name="static")
templates=Jinja2Templates(directory="templates")

class input_mail(BaseModel):
  mail:str=Field(description="Enter the mail you got")

@app.get("/")
def home(request: Request):
  return templates.TemplateResponse(
    request=request,
    name="index.html",
  )

@app.post("/predict_spam")
def predict_spam(request: Request,
                 input_mail:str=Form(...)):
  try:
    email_vec=convertor.transform([input_mail])
    print("vector created")
    prediction=model.predict(email_vec)
    print(prediction)
    if prediction[0]==1:
        result="Alert!it is a spam"
    else:
        result="it is not spam"
    print("result")
    return templates.TemplateResponse(
             request=request,
             name='index.html',
             context={
                "prediction":result
             }
           
          )        
  except Exception as e:

        return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "prediction": f"Error: {str(e)}"
    }
)


