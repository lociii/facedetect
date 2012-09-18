facedetect
==========

Webservice to detect faces in images based on openCV


Requirements
-------------

Werkzeug >= 0.8.3  
opencv >= 2.1

(may work with previous versions, but not tested)

HowTo (Debian)
---------------

git clone git://github.com/lociii/facedetect.git  
cd facedetect  
pip install -r requirements.txt  
aptitude install libcv2.1 python-opencv  
python facedetect.py --host=0.0.0.0 --port=4000

Usage
------

Send HTTP request with url of the image as argument