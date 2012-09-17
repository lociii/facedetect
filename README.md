facedetect
==========

Webservice to detect faces in images based on openCV


Requirements
-------------

Werkzeug >= 0.8.3  
opencv >= 2.4.2

(may work with previous versions, but not tested)

Startup
--------

python facedetect.py --localhost --port=4000 --opencv=/home/user/OpenCV-2.4.2

Usage
------

Send HTTP request with url of the image as argument