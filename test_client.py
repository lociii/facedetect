#!/usr/bin/env python

import urllib2
import urllib
import sys

facedetect_url = 'http://localhost:4000'

if len(sys.argv) != 2:
    print 'USAGE: %s <image url>' % sys.argv[0]
    sys.exit()

data = {'url': sys.argv[1]}
data = urllib.urlencode(data)

response = None
try:
    response = urllib2.urlopen('%s?%s' % (facedetect_url, data))
except urllib2.HTTPError, e:
    print e.read()

if response:
    print response.read()
