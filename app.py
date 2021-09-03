from flask import Flask, render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/prediction', methods=['POST','GET'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        output = 'You Have Heart Disease!!!'

    else:
        output = "You Don't Have Heart Disease!!!"

    return render_template('index.html', prediction_result='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
