from warnings import filterwarnings
filterwarnings("ignore")
from flask import Flask
from flask import render_template, request, redirect, send_file
import os

# Imports for model
from model_script.model import model
from matplotlib import cm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = 'C:\\Users\\ACER\\Documents\\workspace\\repositories\\Final_Year_Project\\static'

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/segment', methods = ['GET','POST'])
def segment():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            filename = str(image.filename)
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))
            print("Saved!")
            convert(filename)
            # os.remove(r'C:\Users\ACER\Documents\workspace\skin lesion\Final_Year_Project\static\*.bmp')
            return redirect(request.url)
    return render_template('segment.html')

def enhance(img):
    sub = (model.predict(img.reshape(1,192,256,3))).flatten()

    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

def convert(filename):
    filename = os.path.join(r"C:\Users\ACER\Documents\workspace\repositories\Final_Year_Project\static",filename)
    inp_image = np.array(Image.open(filename))
    print('Segmenting...\n')
    # img_pred = model.predict(inp_image.reshape(1,192,256,3))
    img_pred = enhance(inp_image).reshape(192,256)
    print('Segmented!\n')
    # img_pred = img_pred.reshape(192,256)
    plt.imshow(img_pred)
    im = Image.fromarray(np.uint8(cm.gist_earth(img_pred)*255))
    im = im.convert("L")
    im.save(r"C:\Users\ACER\Documents\workspace\repositories\Final_Year_Project\static\segmented.bmp")
    print('Saved!\n')


@app.route('/download', methods = ['GET','POST'])
def download():
    return send_file(r"C:\Users\ACER\Documents\workspace\repositories\Final_Year_Project\static\segmented.bmp")

@app.route('/contact')
def contact():
    return render_template("contact.html")

# @app.route('/return_File')
# def return_File():
#     return send_file(r"C:\Users\ACER\Documents\workspace\skin lesion\Final_Year_Project\static\segmented.bmp")

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)
