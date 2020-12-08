from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import os
from os import listdir
from os.path import isfile, join

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(1 / float(tokens[1]))
    return images, np.asarray(times, dtype=np.float32)

jpg_files = [f for f in listdir('/home/winter/Pictures/g2/') if isfile(join('/home/winter/Pictures/g2/', f)) and f.endswith(".jpg")]
for file in jpg_files:
    data = cv.imread(join('/home/winter/Pictures/g2/', file))
    crop_data = data[532:2952, 1184:3604, :]
    cv.imwrite(join('/home/winter/Pictures/g2/', file.replace(".jpg", "_crop.jpg")), crop_data)


images, times = loadExposureSeq("/home/winter/Pictures/g2/")
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)
merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)
tonemap = cv.createTonemap(2.2)
ldr = tonemap.process(hdr)
merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)
cv.imwrite('/home/winter/Pictures/g2/fusion.png', fusion * 255)
cv.imwrite('/home/winter/Pictures/g2/ldr.png', ldr * 255)
cv.imwrite('/home/winter/Pictures/g2/hdr.hdr', hdr)