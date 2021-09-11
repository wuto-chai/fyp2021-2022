import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import base64
from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
from io import BytesIO
import server_main
import cv2
import numpy as np

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'mp4', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_result():
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'output.txt'), 'r') as f:
        x_vals = ["0:00:00"]
        y_vals = [0]
        for line in f.readlines():
            line = line.split(',')
            if int(line[0]) % 25 == 0:
                x_vals.append(line[1])
                y_vals.append(line[3])
        return x_vals, y_vals


def draw_pic(x_vals, y_vals):
    x = x_vals
    tick_spacing = x
    tick_spacing = int(len(x_vals) / 10)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(x_vals, y_vals, 'o-')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xlabel("Time")
    plt.ylabel("Counted people")
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_url = base64.b64encode(buffer.getvalue()).decode()
    pic = "data:image/png;base64," + plot_url
    return pic


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('base.html')

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

          #  with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as image:
          #      result_str = base64.b64encode(image.read())

        #str = result_str.decode()
        server_main.run(source=file, save_img=True, output_dir=UPLOAD_FOLDER)
        result = read_result()
    return render_template('base.html', data=result[1][-1], pic=draw_pic(result[0], result[1]))
    # return redirect(url_for('test'))
