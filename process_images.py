#!/usr/bin/env python
'''Crop an image to just the portions containing text.
Usage:
    ./crop_morphology.py path/to/image.jpg
This will place the cropped image in path/to/image.crop.png.
For details on the methodology, see
http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html
'''
############
import glob
import os
import random
import sys
import random
import math
import json
from collections import defaultdict
import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter
from pylab import *
import argparse,glob,os,os.path
import traceback
from scipy.ndimage import measurements
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter
from multiprocessing import Pool
import ocrolib
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *


def DSAVE(title,image):
    if not args.debug: return
    if type(image)==list:
        assert len(image)==3
        image = transpose(array(image),[1,2,0])
    fname = "_"+title+".png"
    print "debug",fname
    imsave(fname,image)

def compute_line_seeds(binary,bottom,top,colseps,scale,threshold=0.2,vscale=1.0):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = threshold
    vrange = int(vscale*scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
    bmarked *= (bottom>t*amax(bottom)*t)*(1-colseps)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked *= (top>t*amax(top)*t/2)*(1-colseps)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = zeros(binary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = maximum_filter(seeds,(1,int(1+scale)))
    seeds *= (1-colseps)
    #DSAVE("lineseeds",[seeds,0.3*tmarked+0.7*bmarked,binary])
    seeds,_ = morph.label(seeds)
    return seeds

def compute_gradmaps(binary,scale,vscale=1.0,hscale=1.0,usegauss=False):
    # use gradient filtering to find baselines
    boxmap = psegutils.compute_boxmap(binary,scale)
    cleaned = boxmap*binary
    #DSAVE("cleaned",cleaned)
    if usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned,(vscale*0.3*scale,
                                            hscale*6*scale),order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned,(max(4,vscale*0.3*scale),
                                        hscale*scale),order=(1,0))
        grad = uniform_filter(grad,(vscale,hscale*6*scale))
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    return bottom,top,boxmap



def compute_colseps_conv(binary,scale=1.0,csminheight=10,maxcolseps=2):
    """Find column separators by convolution and
    thresholding."""
    h,w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0*binary,(scale,scale*0.5))
    smoothed = uniform_filter(smoothed,(5.0*scale,1))
    thresh = (smoothed<amax(smoothed)*0.1)
    #DSAVE("1thresh",thresh)
    # find column edges by filtering
    grad = gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
    grad = uniform_filter(grad,(10.0*scale,1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad>0.5*amax(grad))
    #DSAVE("2grad",grad)
    # combine edges and whitespace
    seps = minimum(thresh,maximum_filter(grad,(int(scale),int(5*scale))))
    seps = maximum_filter(seps,(int(2*scale),1))
    #DSAVE("3seps",seps)
    # select only the biggest column separators
    seps = morph.select_regions(seps,sl.dim0,min=csminheight*scale,nbest=maxcolseps+1)
    #DSAVE("4seps",seps)
    return seps


def compute_colseps(binary,scale,blackseps=True):
    """Computes column separators either from vertical black lines or whitespace."""
    colseps = compute_colseps_conv(binary,scale)
    #DSAVE("colwsseps",0.7*colseps+0.3*binary)
    if blackseps:
        seps = compute_separators_morph(binary,scale)
        #DSAVE("colseps",0.7*seps+0.3*binary)
        #colseps = compute_colseps_morph(binary,scale)
        colseps = maximum(colseps,seps)
        binary = minimum(binary,1-seps)
    return colseps,binary

def remove_hlines(binary,scale,maxsize=10):
    labels,_ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if sl.width(b)>maxsize*scale:
            labels[b][labels[b]==i+1] = 0
    return array(labels!=0,'B')

def compute_segmentation(binary,scale):
    """Given a binary image, compute a complete segmentation into
    lines, computing both columns and text lines."""
    binary = array(binary,'B')

    # start by removing horizontal black lines, which only
    # interfere with the rest of the page segmentation
    binary = remove_hlines(binary,scale)

    # do the column finding
    colseps,binary = compute_colseps(binary,scale)

    # now compute the text line seeds
    bottom,top,boxmap = compute_gradmaps(binary,scale)
    seeds = compute_line_seeds(binary,bottom,top,colseps,scale)
    #DSAVE("seeds",[bottom,top,boxmap])

    # spread the text line seeds to all the remaining
    # components
    llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
    spread = morph.spread_labels(seeds,maxdist=scale)
    llabels = where(llabels>0,llabels,spread*binary)
    segmentation = llabels*binary
    return segmentation

def compute_separators_morph(binary,scale,sepwiden=10,maxseps=2):
    """Finds vertical black lines corresponding to column separators."""
    d0 = int(max(5,scale/4))
    d1 = int(max(5,scale))+sepwiden
    thick = morph.r_dilation(binary,(d0,d1))
    vert = morph.rb_opening(thick,(10*scale,1))
    vert = morph.r_erosion(vert,(d0//2,sepwiden))
    vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2*maxseps)
    vert = morph.select_regions(vert,sl.dim0,min=20*scale,nbest=maxseps)
    return vert

def compute_lines(segmentation,scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i,o in enumerate(lobjects):
        if o is None: continue
        if sl.dim1(o)<2*scale or sl.dim0(o)<scale: continue
        mask = (segmentation[o]==i+1)
        if amax(mask)==0: continue
        result = record()
        result.label = i+1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines



def extract(image):

    try:
        binary = ocrolib.read_image_binary(image)
        binary = 1-binary

        scale = psegutils.estimate_scale(binary)
        segmentation = compute_segmentation(binary,scale)

        # ...lines = compute_lines(segmentation,scale)

        # compute the reading order
        lines = psegutils.compute_lines(segmentation,scale)
        order = psegutils.reading_order([l.bounds for l in lines])
        lsort = psegutils.topsort(order)

        # renumber the labels so that they conform to the specs
        nlabels = amax(compute_segmentation)+1
        renumber = zeros(nlabels,'i')
        for i,v in enumerate(lsort): renumber[lines[v].label] = 0x010000+(i+1)
        segmentation = renumber[segmentation]

        outputdir = "http://127.0.0.1:5000/uploads/"
        
        lines = [lines[i] for i in lsort]
        ocrolib.write_page_segmentation("%s.pseg.png"%outputdir,segmentation)


        cleaned = ocrolib.remove_noise(binary,args.noise)
        for i,l in enumerate(lines):
            binline = psegutils.extract_masked(1-cleaned,l,pad=args.pad,expand=args.expand)
            ocrolib.write_image_binary("%s/01%04x.bin.png"%(outputdir,i+1),binline)
        #print "%6d"%i,fname,"%4.1f"%scale,len(lines)
    except:
        print ('error')




def extract2(image):
    binary = ocrolib.read_image_binary(image)
	binary = 1-binary
	return binary




