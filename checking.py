import requests



flower = {"sepal_length":15,
"sepal_width":13,
"petal_length":11,
"petal_width":10}

result = requests.post('http://127.0.0.1:5000/api/flower', json = flower)
print('result::: ', result.text)