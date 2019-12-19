import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math
import os 

def vertical_slice(img, out_name, outdir, h_slice_size,w_slice_size):
    width, height = img.size
    upper = 0
    left = 0
    slices = int(math.ceil(height/h_slice_size))
    count = 1
    spray = 0
    for slice in range(slices):
        if count == slices:
            lower = height
        else:
            lower = int(count * h_slice_size)
        spray += 1
        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += h_slice_size
        horizontal_slice(working_slice, out_name, outdir, w_slice_size,spray)
        #working_slice.save(os.path.join(outdir, "slice_" + out_name + "_" + str(count)+".png"))
        count += 1

def horizontal_slice(img, out_name, outdir, slice_size,spray):
    width, height = img.size
    upper = 0
    left = 0
    slices = int(math.ceil(width/slice_size))
    count = 1
    for slice in range(slices):
        if count == slices:
            right = width
        else:
            right = int(count * slice_size)  
        bbox = (left, upper, right, height)
        working_slice = img.crop(bbox)
        left += slice_size
        working_slice.save(os.path.join(outdir, "slice_image_" + str(i) + str(spray) +'_' + out_name + "_" + str(count)+".png"))
        count += 1


if __name__ == '__main__':
    i = 0
    filename = input("Input file name:")
    IMG = Image.open('./img/'+filename)
    width, height = IMG.size
    vertical_slice(IMG, filename, "./sliced/",(height/3),(width/3))