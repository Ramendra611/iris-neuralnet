import joblib
from tensorflow.keras.models import load_model
import  numpy as np

def get_predictions(model, scaler, input_json):
    sepal_length = input_json["sepal_length"]
    sepal_width = input_json["sepal_width"]
    petal_length = input_json["petal_length"]
    petal_width = input_json["petal_width"]
    
    flower  = [[sepal_length,sepal_width,petal_length,petal_width]] 
    flower = scaler.transform(flower)
    classes = np.array(["setosa", "versicolor", "virginica"])
    class_ind = model.predict_classes(flower)[0]
    return classes[class_ind]

# model = load_model("final_iris_model.h5")
# scaler = joblib.load("iris_scaler.pkl")
# flower = {"sepal_length":5,
# "sepal_width":3,
# "petal_length":1,
# "petal_width":0}
# pred = get_predictions(model, scaler, flower )     
# print('pred:: \n', pred) 