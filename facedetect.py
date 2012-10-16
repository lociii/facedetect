#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import cv
import json
import urllib2
from argparse import ArgumentParser
from werkzeug.serving import run_simple
from werkzeug.routing import Map, Rule
from werkzeug.wrappers import Request, Response
from werkzeug.exceptions import HTTPException, BadRequest, NotFound
import StringIO
import Image

min_size = (20, 20)
max_size = 300
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0
cascade_xml = 'haarcascade_frontalface_alt2.xml'


class Facedetect(object):

    stats = {
        'processed_images': 0,
        'no_image_found': 0,
        'failed_urls': 0,
        'detected_faces': 0,
    }

    cascade = None

    def __init__(self):
        self.image = None
        self.url_map = Map([
            Rule('/', endpoint='detect'),
            Rule('/favicon.ico', endpoint='favicon'),
            Rule('/status', endpoint='status'),
        ])

    # Serve empty favicon.ico
    def on_favicon(self, request):
        return Response()

    def on_status(self, request):
        response = self.stats
        response['status'] = 'Working for you.'
        return Response(json.dumps(response))

    def on_detect(self, request):
        self.load_image(request.args.get('url'))
        faces = self.detect_face()
        self.stats['processed_images'] += 1
        self.stats['detected_faces'] += len(faces)
        return Response(json.dumps(faces))

    def load_image(self, url):
        try:
            response = urllib2.urlopen(url)
            img = StringIO.StringIO(response.read())
            img = Image.open(img).convert('L')
        except (urllib2.HTTPError, urllib2.URLError):
            self.stats['failed_urls'] += 1
            raise NotFound('url does not exist')

        try:
            self.image = cv.CreateImageHeader(img.size, cv.IPL_DEPTH_8U, 1)
            cv.SetData(self.image, img.tostring())
        except Exception:
            self.stats['no_image_found'] += 1
            raise BadRequest('No picture found on url')

    def detect_face(self):
        angle = 0

        # scale input image for faster processing
        width = self.image.width
        height = self.image.height
        ratio = 1.0
        if width > max_size or height > max_size:
            if width > height:
                width = max_size
                ratio = width / float(self.image.width)
                height = int(ratio * height)
            else:
                height = max_size
                ratio = height / float(self.image.height)
                width = int(ratio * width)
        img_small = cv.CreateImage((width, height), cv.IPL_DEPTH_8U, 1)
        cv.Resize(self.image, img_small, cv.CV_INTER_LINEAR)

        cv.EqualizeHist(img_small, img_small)

        faces = []

        while len(faces) == 0 and angle <= 360:
            # rotate image
            if angle > 0:
                img_small = self.rotateImage(img_small)

            faces = cv.HaarDetectObjects(
                img_small, self.cascade, cv.CreateMemStorage(), haar_scale,
                min_neighbors, haar_flags, min_size)
            if faces:
                for index, face in enumerate(faces):
                    ((x, y, w, h), n) = face
                    faces[index] = self.calculatePosition(x, y, w, h, img_small, angle)
            else:
                angle += 90

        return faces

    def calculatePosition(self, x, y, w, h, img_small, angle):
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

    def rotateImage(self, image):
        # transposed image
        timg = cv.CreateImage(
            (image.height, image.width), image.depth, image.channels)

        # rotate clockwise
        cv.Transpose(image, timg)
        cv.Flip(timg, timg, flipMode=1)
        return timg

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            return getattr(self, 'on_' + endpoint)(request, **values)
        except NotFound, e:
            return NotFound('url does not exist')
        except HTTPException, e:
            return e

    def wsgi_app(self, environ, start_response):
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)


def parse_args():
    parser = ArgumentParser(description='Face detection web service')
    parser.add_argument('--hostname', '-H', type=str, default='localhost',
                        help='Hostname to bind service to')
    parser.add_argument('--port', '-p', type=int, default=4000,
                        help='Port to bind service to')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debugging')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app = Facedetect()

    # Load cascade file only once
    try:
        app.cascade = cv.Load(cascade_xml)
    except TypeError:
        print "Cascade '%s' not found. Exiting!" % cascade_xml
        sys.exit(1)

    run_simple(args.hostname, args.port, app, use_debugger=args.debug,
               use_reloader=args.debug)
