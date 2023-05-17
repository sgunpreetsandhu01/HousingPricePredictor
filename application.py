from flask import Flask,request,render_template,flash,redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import random



from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)


app.config['SECRET_KEY'] = 'nMhI_ptwhZ3GS56qSyaBoA'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/images'

ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

df = pd.read_csv("Models/final_data.csv")

cities = list(pd.unique(df['citi']))
cities.sort()


image_id = 0


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/uploadimage',methods=['GET','POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('uploadimage.html')
    else:
        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            global image_id
            image_id = random.randint(0,100)
            filename = str(image_id) + '.jpg'
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image was uploaded sucessfully','success')
            return render_template('home.html',cities=cities)
        
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

@app.route('/uploadimage/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html',cities=cities)
    else:
        print(image_id)
        print(request.form.get('city'))
        print(request.form.get('street'))
        print(request.form.get('bed'))

        filename = '/images/'+str(image_id)+'.jpg'
        
        user_data = {
                "street":request.form.get('street'),
                "city":request.form.get('city'),
                "bed":request.form.get('bed'),
                "bath":request.form.get('bath'),
                "sqft":request.form.get('sqft')
            }

        data = CustomData(
            image_id = image_id,
            street = request.form.get('street'),
            city = request.form.get('city'),
            bed = request.form.get('bed'),
            bath = request.form.get('bath'),
            sqft = float(request.form.get('sqft'))
        )

        

        pred_df = data.get_data_frame_final()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('predict.html',user_data=user_data,house_img=filename,results=results)

if __name__ == '__main__':
    app.run()
