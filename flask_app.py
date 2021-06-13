from flask import Flask, request, jsonify, render_template, session, url_for, redirect
from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm
import numpy as np
from tensorflow.keras.models import load_model
import  numpy as np
import joblib
from model_prediction import get_predictions

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

@app.route("/", methods=['GET', 'POST'])
def index():
  form = FlowerForm()

  if form.validate_on_submit():
    session["sep_len"] = form.sep_len.data
    session["sep_wid"] = form.sep_wid.data
    session["pet_len"] = form.pet_len.data
    session["pet_wid"] = form.pet_wid.data
    
    return redirect(url_for('prediction'))
  
  return render_template('home.html', form = form)

  # return ("<h1>Flask is running</h1>")

model = load_model("final_iris_model.h5")
scaler = joblib.load("iris_scaler.pkl")

@app.route('/prediction')
def prediction():
  content = {}
  
  content["sepal_length"] = float(session["sep_len"])
  content["sepal_width"] = float(session["sep_wid"])
  content["petal_length"] = float(session["pet_len"])
  content["petal_width"] = float(session["pet_wid"])

  results = get_predictions(model, scaler, content)
  return render_template('prediction.html', results = results)
  


@app.route('/api/flower', methods=['POST'])
def flower_prediction():
  content = request.json
  print('Content:::::;', content)
  result =  get_predictions(model, scaler, content)
  return jsonify(result)
  

class FlowerForm(FlaskForm):
  sep_len = TextField("Sepal Length")
  sep_wid = TextField("Sepal Width")
  pet_len = TextField("Petal Length")
  pet_wid = TextField("Petal Width")
  submit = SubmitField("PREDICT")







if __name__ == "__main__":
    app.run()
  


