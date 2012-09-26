#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv
import json
import urllib2
import os
from argparse import ArgumentParser
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
from werkzeug.exceptions import HTTPException, MethodNotAllowed, BadRequest, \
    NotFound, InternalServerError
from tempfile import NamedTemporaryFile

min_size = (20, 20)
max_size = 300
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0
cascade_xml = "haarcascade_frontalface_alt2.xml"


@Request.application
def application(request):
    tempfile = NamedTemporaryFile()

    try:
        if request.method != 'GET':
            raise MethodNotAllowed(['GET'])
        if request.args.get('url') is None:
            raise BadRequest('Missing parameter url')

        try:
            response = urllib2.urlopen(request.args.get('url'))
            data = response.read()
        except urllib2.HTTPError:
            raise NotFound('url does not exist')

        try:
            tempfile.file.write(data)
            image = cv.LoadImage(tempfile.name,
                                 iscolor=cv.CV_LOAD_IMAGE_GRAYSCALE)
        except Exception:
            raise BadRequest('No picture found on url')

        faces = detect(image)

        response = Response(json.dumps(faces))
    except HTTPException, e:
        return e
    else:
        return response
    finally:
        if tempfile is not None:
            tempfile.close()


def detect(img):
    angle = 0

    # scale input image for faster processing
    width = img.width
    height = img.height
    ratio = 1.0
    if width > max_size or height > max_size:
        if width > height:
            width = max_size
            ratio = width / float(img.width)
            height = int(ratio * height)
        else:
            height = max_size
            ratio = height / float(img.height)
            width = int(ratio * width)
    img_small = cv.CreateImage((width, height), cv.IPL_DEPTH_8U, 1)
    cv.Resize(img, img_small, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(img_small, img_small)

    faces = []
    try:
        path = os.path.normpath(os.path.dirname(
            os.path.realpath(__file__))) + os.sep
        cascade = cv.Load(path + cascade_xml)
    except TypeError:
        raise InternalServerError('XML definition not found')

    while len(faces) == 0 and angle <= 360:
        # rotate image
        if angle > 0:
            img_small = rotateImage(img_small)

        faces = cv.HaarDetectObjects(
            img_small, cascade, cv.CreateMemStorage(), haar_scale,
            min_neighbors, haar_flags, min_size)
        if faces:
            for index, face in enumerate(faces):
                ((x, y, w, h), n) = face
                faces[index] = calculatePosition(x, y, w, h, img_small, angle)
        else:
            angle += 90

    return faces


def rotateImage(image):
    # transposed image
    timg = cv.CreateImage(
        (image.height, image.width), image.depth, image.channels)

    # rotate clockwise
    cv.Transpose(image, timg)
    cv.Flip(timg, timg, flipMode=1)
    return timg


def calculatePosition(x, y, w, h, img_small, angle):
    img_width = float(img_small.width)
    img_height = float(img_small.height)
    if angle == 90 or angle == 270:
        img_width = float(img_small.height)
        img_height = float(img_small.width)

    x_new = None
    y_new = None
    w_new = None
    h_new = None

    if angle == 90:
        x = x + w

        x_new = y
        y_new = img_height - x
        w_new = h
        h_new = w
    elif angle == 180:
        x = x + w
        y = y + h

        x_new = img_width - x
        y_new = img_height - y
        w_new = w
        h_new = h
    elif angle == 270:
        y = y + h

        x_new = img_width - y
        y_new = x
        w_new = h
        h_new = w

    if angle:
        x = x_new
        y = y_new
        w = w_new
        h = h_new

    x = int(x / img_width * 100)
    y = int(y / img_height * 100)
    w = int(w / img_width * 100)
    h = int(h / img_height * 100)

    return x, y, w, h

if __name__ == '__main__':
    parser = ArgumentParser(description="Face detection web service")
    parser.add_argument("--hostname", dest="hostname", type=str,
                        help="Hostname to bind service to", required=False,
                        default="localhost")
    parser.add_argument("--port", dest="port", type=int,
                        help="Port to bind service to", required=False,
                        default=4000)
    args = parser.parse_args()

    run_simple(args.hostname, args.port, application, use_reloader=True)
