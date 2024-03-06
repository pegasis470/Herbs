import numpy as np
import cv2
from search import Predict
import pandas as pd
import keras.utils as image
from flask import *
import os
import shutil
app=Flask(__name__)




@app.route('/')
def home():
    return render_template("index.ejs")
@app.route('/camera')
def Cam():
    return render_template('camera.ejs')

@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        file=request.files['file']
        file.save(file.filename)
        result,q1,q2,q3,q4=Predict(file.filename)
        shutil.copy(f'{file.filename}','static/input.jpg')
        os.remove(file.filename)
        return render_template("result.ejs",result=result,q1=q1,q2=q2,q3=q3,q4=q4)

if __name__=='__main__':
    app.run(debug=True)
    app.config['TEMPLATiES_AUTO_RELOAD']=True
