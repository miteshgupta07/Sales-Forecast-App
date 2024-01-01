import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
path="E:\Data Science\Project\Sales Forecast App\Model"
model = pickle.load(open(path+'\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)


    return render_template('index.html', prediction_text='Forecasted Sales: $ {}'.format(output))
    # print(output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)