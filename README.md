# Online Lesion Segmentation Platform
Segmentation Platform for Skin Lesion Segmentation Online using Flask nad Python

## Intructions  

### For Windows
- Create a virtual environment
	- virtualenv venv
	- cd /venv/Scripts/activate
	- cd ../..
- Install the requirements using `pip install -r requirements.txt`
- Change the image saving paths according to the local system
- Download trained .h5 model from [here](https://drive.google.com/open?id=1BAG2F6BjKRK4zePTkSy15Rbi5au1u698)  
- place the trained model in directory named **model**
- run the app using `python run.py`

### For Ubuntu
- Create a virtual environment
	- virtualvenv --python=python3 venv
	- source venv/bin/activate
- Install the requirements using `pip3 install -r requirements.txt`
- Change the image saving paths according to the local system
- Download trained .h5 model from [here](https://drive.google.com/open?id=1BAG2F6BjKRK4zePTkSy15Rbi5au1u698)  
- place the trained model in directory named **model**
- run the app using `python run.py`

