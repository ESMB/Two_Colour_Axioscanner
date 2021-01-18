#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:00:15 2021

@author: Mathew
"""

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters,measure
from PIL import Image
import pandas as pd
from scipy.spatial import distance
import tifffile
import czifile

# Function to load images:

def load_image(toload):
    
    image=imread(toload)
    
    return image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_otsu(input_image):
    threshold_value=filters.threshold_otsu(input_image)    
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image
    
# Function to show the particular image:
def show(input_image):
    plt.imshow(input_image,cmap="Reds")
    plt.show()

# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# This is to look at coincidence purely in terms of pixels

def coincidence_analysis_pixels(binary_image1,binary_image2):
    pixel_overlap_image=binary_image1&binary_image2         
    pixel_overlap_count=pixel_overlap_image.sum()
    pixel_fraction=pixel_overlap_image.sum()/binary_image1.sum()
    
    return pixel_overlap_image,pixel_overlap_count,pixel_fraction

# Look at coincidence in terms of features. Needs binary image input 

def feature_coincidence(binary_image1,binary_image2):
    number_of_features,labelled_image1=label_image(binary_image1)          # Labelled image is required for this analysis
    coincident_image=binary_image1 & binary_image2        # Find pixel overlap between the two images
    coincident_labels=labelled_image1*coincident_image   # This gives a coincident image with the pixels being equal to label
    coinc_list, coinc_pixels = np.unique(coincident_labels, return_counts=True)     # This counts number of unique occureences in the image
    # Now for some statistics
    total_labels=labelled_image1.max()
    total_labels_coinc=len(coinc_list)
    fraction_coinc=total_labels_coinc/total_labels
    
    # Now look at the fraction of overlap in each feature
    # First of all, count the number of unique occurances in original image
    label_list, label_pixels = np.unique(labelled_image1, return_counts=True)
    fract_pixels_overlap=[]
    for i in range(len(coinc_list)):
        overlap_pixels=coinc_pixels[i]
        label=coinc_list[i]
        total_pixels=label_pixels[label]
        fract=1.0*overlap_pixels/total_pixels
        fract_pixels_overlap.append(fract)
    
    
    # Generate the images
    coinc_list[0]=1000000   # First value is zero- don't want to count these. 
    coincident_features_image=np.isin(labelled_image1,coinc_list)   # Generates binary image only from labels in coinc list
    coinc_list[0]=0
    non_coincident_features_image=~np.isin(labelled_image1,coinc_list)  # Generates image only from numbers not in coinc list.
    
    return coinc_list,coinc_pixels,fraction_coinc,coincident_features_image,non_coincident_features_image,fract_pixels_overlap

# Function to measure minimum distances between two sets of data
def minimum_distance(measurements1,measurements2):
    s1 = measurements1[["centroid-0","centroid-1"]].to_numpy()
    s2 = measurements2[["centroid-0","centroid-1"]].to_numpy()
    minimum_lengths=distance.cdist(s1,s2).min(axis=1)
    return minimum_lengths


pathList=[]

pathList.append(r"/Users/Mathew/Desktop/Axioscan/")

for i in range(len(pathList)):
    
    directory=pathList[i]
    
    # Run functions for aptamer
    filename="apt.tif"
    apt_image=load_image(directory+filename)
    apt_threshold,apt_binary=threshold_image_otsu(apt_image)
    apt_number,apt_labelled=label_image(apt_binary)
    print("%d feautres were detected in the aptamer image."%apt_number)
    apt_measurements=analyse_labelled_image(apt_labelled,apt_image)
    apt_measurements.to_csv(directory + '/' + 'all_aptamer_metrics.csv', sep = '\t')
    
    # Run functions for antibody
    filename="ab.tif"
    ab_image=load_image(directory+filename)
    ab_threshold,ab_binary=threshold_image_otsu(ab_image)
    ab_number,ab_labelled=label_image(ab_binary)
    print("%d feautres were detected in the antibody image."%ab_number)
    ab_measurements=analyse_labelled_image(ab_labelled,ab_image)
    ab_measurements.to_csv(directory + '/' + 'all_ab_metrics.csv', sep = '\t')
    
    # Run all functions for nucleus
    filename="nuc.tif"
    nuc_image=load_image(directory+filename)
    nuc_threshold,nuc_binary=threshold_image_otsu(nuc_image)
    nuc_number,nuc_labelled=label_image(nuc_binary)
    print("%d nuclei were detected in the image."%nuc_number)
    nuc_measurements=analyse_labelled_image(nuc_labelled,nuc_image)
    nuc_measurements.to_csv(directory + '/' + 'all_nuc_metrics.csv', sep = '\t')
    
    # Coincidence functions
    
    apt_pixel_coincident_image,apt_pixel_overal_count,apt_pixel_fraction=coincidence_analysis_pixels(apt_binary,ab_binary)
    print("%.2f of aptamer pixels had coincidence with the antibody image."%apt_pixel_fraction)
    
    ab_pixel_coincident_image,ab_pixel_overal_count,ab_pixel_fraction=coincidence_analysis_pixels(ab_binary,apt_binary)
    print("%.2f of antibody pixels had coincidence with the antibody image."%ab_pixel_fraction)

    apt_coinc_list,apt_coinc_pixels,apt_fraction_coinc,apt_coincident_features_image,apt_noncoincident_features_image,apt_fraction_pixels_overlap=feature_coincidence(apt_binary,ab_binary)
    print("%.2f of aptamer features had coincidence with features in antibody image. Average overlap was %2f."%(apt_fraction_coinc,sum(apt_fraction_pixels_overlap)/len(apt_fraction_pixels_overlap)))
    
    aptamer_coincident_tosave=apt_coincident_features_image*apt_image
    im = Image.fromarray(aptamer_coincident_tosave)
    im.save(directory+'Aptamer_features_coincident.tif')
    
    aptamer_noncoincident_tosave=apt_noncoincident_features_image*apt_image
    im = Image.fromarray(aptamer_noncoincident_tosave)
    im.save(directory+'Aptamer_features_noncoincident.tif')
    
    
    ab_coinc_list,ab_coinc_pixels,ab_fraction_coinc,ab_coincident_features_image,ab_noncoincident_features_image,ab_fraction_pixels_overlap=feature_coincidence(ab_binary,apt_binary)
    print("%.2f of antibody features had coincidence with features in aptamer image. Average overlap was %2f."%(ab_fraction_coinc,sum(ab_fraction_pixels_overlap)/len(ab_fraction_pixels_overlap)))
    
    antibody_coincident_tosave=ab_coincident_features_image*ab_image
    im = Image.fromarray(antibody_coincident_tosave)
    im.save(directory+'Antibody_features_coincident.tif')
    
    antibody_noncoincident_tosave=ab_noncoincident_features_image*ab_image
    im = Image.fromarray(antibody_noncoincident_tosave)
    im.save(directory+'Antibody_features_noncoincident.tif')

    
    aptamer_distance_to_nuc=minimum_distance(apt_measurements,nuc_measurements)
    antibody_distance_to_nuc=minimum_distance(ab_measurements,nuc_measurements)  
    nuc_distance_to_nuc=minimum_distance(nuc_measurements,nuc_measurements)
   
    

    
    
    
    
    