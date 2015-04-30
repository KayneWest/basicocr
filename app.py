#!/usr/bin/env python
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, send_file
from werkzeug import secure_filename
import ocrolib
import util
import numpy as np
from PIL import Image
import glob


app = Flask(__name__)
app_home = os.getcwd()

model = "/Users/mkrzus/basicocropy/models/my-model3-00160000.pyrnn.gz"

UPLOAD_FOLDER = '/Users/mkrzus/basicocropy/uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMAGES = '/Users/mkrzus/basicocropy/uploads/images/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGES'] = IMAGES


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def convert_file(filename):
    return filename.rsplit('.', 1)[0]+".bin.png"


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    files1 = glob.glob(app.config['UPLOAD_FOLDER']+"/*.png")
    files2 = glob.glob(app.config['IMAGES']+"/*.png")
    for f in files1+files2:
        os.remove(f)

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # save as a binary file
            filename = convert_file(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            p = util.Prediction(app.config['UPLOAD_FOLDER']+filename, model)
            p.image_to_string()
            # more efficient process to save during the prediction phase
            texts = []
            for item in p.repository:
                #for file recovery indisplay function
                name =  item[1][0]
                image = item[1][1]
                pred = item[0]
                im = Image.fromarray(image)
                im.save(app.config['IMAGES']+name)
                texts.append((name, pred))
            return render_template('images.html', results=texts)

            
    return '''
    <!doctype html>
    <title>Super Basic Ocr Demo</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/images/<filename>')
def send_file(filename):
    return send_from_directory(IMAGES, filename)


if __name__ == '__main__':
    app.debug = True
    app.run()


