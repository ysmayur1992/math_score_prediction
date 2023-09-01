from flask import Flask,render_template,request
from src.pipelines.predict_pipeline import Prediction_Pipeline
from src.pipelines.predict_pipeline import Custom_Object

app = Flask(__name__)

@app.route('/')
def Homepage():
    return render_template('index.html')

@app.route('/predict_the_score',methods=['GET','POST'])
def predict_score():
    if(request.method == 'GET'):
        return render_template('home.html')
    else:
        obj = Custom_Object(
            gender=request.form.get('gender'),
            ethnicity=request.form.get('ethnicity'),
            education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            courses=request.form.get('test_preparation_course'),
            reading=int(request.form.get('reading_score')),
            writing=int(request.form.get('writing_score'))
        )

        df = obj.get_DataFrame()
        predict = Prediction_Pipeline()
        score = predict.prediction(df)

        return render_template('home.html',results=score)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
