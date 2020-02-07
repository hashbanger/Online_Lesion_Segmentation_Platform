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
import cv2 

app = Flask(__name__)
#app.config['IMAGE_UPLOADS'] = 'C:\\Users\\ACER\\Documents\\workspace\\repositories\\Online_Lesion_Segmentation_Platform\\static'
app.config['IMAGE_UPLOADS'] = '/home/prashant/git_repos_hashbanger/Online_Lesion_Segmentation_Platform/static'

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
            # filename = str(image.filename)
            filename = "input_image.bmp"
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], filename))
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

def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def convert(filename):
    # filename = os.path.join(r"C:\Users\ACER\Documents\workspace\repositories\Online_Lesion_Segmentation_Platform\static",filename)
    filename = os.path.join(r"/home/prashant/git_repos_hashbanger/Online_Lesion_Segmentation_Platform/static",filename)
    
    
    inp_image = np.array(Image.open(filename))
    print('Segmenting...\n')

    img_pred = enhance(inp_image).reshape(192,256)
    img_crop = get_segment_crop(img=inp_image, mask= img_pred)
    print('Segmented!\n')
    print("Cropped!\n")

    im_1 = Image.fromarray(np.uint8(cm.gist_earth(img_pred)*255))
    im_1 = im_1.convert("L")
    # im_1.save(r"C:\Users\ACER\Documents\workspace\repositories\Online_Lesion_Segmentation_Platform\static\segmented.bmp")
    im_1.save(r"/home/prashant/git_repos_hashbanger/Online_Lesion_Segmentation_Platform/static/segmented.bmp")

    dim = (256, 192)
    im_2 = cv2.resize(img_crop, dim, interpolation = cv2.INTER_AREA)
    im_2 = Image.fromarray(im_2)
    im_2 = im_2.convert("RGB")
    # im_2.save(r"C:\Users\ACER\Documents\workspace\repositories\Online_Lesion_Segmentation_Platform\static\cropped.bmp")
    im_2.save(r"/home/prashant/git_repos_hashbanger/Online_Lesion_Segmentation_Platform/static/cropped.bmp")




@app.route('/download', methods = ['GET','POST'])
def download():
    # return send_file(r"C:\Users\ACER\Documents\workspace\repositories\Online_Lesion_Segmentation_Platform\static\segmented.bmp")
    return send_file(r"/home/prashant/git_repos_hashbanger/Online_Lesion_Segmentation_Platform/static/segmented.bmp")



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
