from flask import Flask, render_template, url_for, redirect
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import subprocess

UPLOAD_FOLDER = '/home/alex/diplom_app/data/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

input_file = ''
result = ''
width_bar = ''

@app.route('/')
def index():
    global input_file
    return render_template('index.html', input_file=input_file, result=result, width_bar=width_bar)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global input_file
    if request.method == 'POST':
        f = request.files['file']
        if f:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            input_file = f.filename
            return redirect(url_for('index'))
        else:
            return redirect(url_for('index'))

@app.route('/data')
def download():
    return send_from_directory('data/', 'arm_compiler.BUILD')

@app.route('/screen_result')
def screen_result():
    global result, width_bar
    result = subprocess.run(['python3', '/home/alex/diplom/diploma/web_inference.py', '/home/alex/diplom/diploma/42.csv'], stdout=subprocess.PIPE, text=True)
    result = int(float(result.stdout.splitlines()[1])*100)
    width_bar = str(result*10)+'%'
    return redirect(url_for('index'))

