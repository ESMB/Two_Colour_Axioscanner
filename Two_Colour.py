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
import czifile

# Where to save overall results
root_directory=r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/"  

# Paths to analyse below


pathList=[]


pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD041_19_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD036_17_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD008_15_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD026_18_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD036_17_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD041_19_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD030_18_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD008_15_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD015_16_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD027_18_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD015_16_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD025_17_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD030_18_2_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD024_18_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD016_16_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD053_16_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD016_14_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD016_14_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD022_17_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD053_16_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD024_18_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD050_19_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD050_19_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD008_15_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD030_18_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD026_18_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD036_17_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD026_18_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD030_18_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD041_19_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD025_17_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD027_18_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD015_16_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD027_18_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD025_17_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD016_16_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD030_18_2_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD016_14_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD022_17_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD053_16_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD022_17_1")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD030_18_2_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD024_18_2")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD016_16_0")
pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/OneDrive - University of Edinburgh/2021_01_19/SD050_19_2")
filename="Image.czi"

pixel_size=325 # Pixel size in nm


# Function to load images:

def load_image(toload):
    
    image=imread(toload)
    
    return image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_otsu(input_image):
    threshold_value=filters.threshold_otsu(input_image)    
    binary_image=input_image>threshold_value

    return threshold_value,binary_image


# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image
    
# Function to show the particular image:
def show(input_image,color=''):
    if(color=='Red'):
        plt.imshow(input_image,cmap="Reds")
        plt.show()
    elif(color=='Blue'):
        plt.imshow(input_image,cmap="Blues")
        plt.show()
    elif(color=='Green'):
        plt.imshow(input_image,cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
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


Output_overall = pd.DataFrame(columns=['Image', 'Nuc_threshold', 'Ab_threshold','Apt_threshold','Number of nuclei','ab number of features','apt number of features',
                                       'ab area','ab area (SD)','ab length','ab length (SD)','ab perimeter','ab perimeter (SD)','ab intensity','ab intensity (SD)',
                                       'ab max intensity','ab max intensity (SD)','ab closest nuc','ab closest nuc (SD)','ab pixel coincidence','ab feature coincidence',
                                       'ab feature overlap','apt area','apt area (SD)','apt length','apt length (SD)','apt perimeter','apt perimeter (SD)','apt intensity','apt intensity (SD)',
                                       'apt max intensity','apt max intensity (SD)','apt closest nuc','apt closest nuc (SD)','apt pixel coincidence','apt feature coincidence',
                                       'apt feature overlap'])

for i in range(len(pathList)):
    
    directory=pathList[i]+"/"
    
    # For fixed thresholds:
    # aptamer_threshold=14506
    # nucleus_threshold=2172
    # antibody_threshold=9684
    
    # Load .czi images
    
    img = czifile.imread(directory+filename)
    nuc_image=img[0,0,0,0,:,:,0]
    ab_image=img[0,0,0,1,:,:,0]
    apt_image=img[0,0,0,2,:,:,0]

    im = Image.fromarray(nuc_image)
    im.save(directory+'DAPI_Py.tif')

    im = Image.fromarray(ab_image)
    im.save(directory+'AB_Py.tif')


    im = Image.fromarray(apt_image)
    im.save(directory+'Apt_Py.tif')

    # Run functions for aptamer
    
    apt_threshold,apt_binary=threshold_image_otsu(apt_image)
    # apt_threshold,apt_binary=threshold_image_fixed(apt_image,aptamer_threshold)
    im = Image.fromarray(apt_binary)
    im.save(directory+'Apt_Binary.tif')
    apt_number,apt_labelled=label_image(apt_binary)
    print("%d feautres were detected in the aptamer image."%apt_number)
    apt_measurements=analyse_labelled_image(apt_labelled,apt_image)
    # apt_measurements.to_csv(directory + '/' + 'all_aptamer_metrics.csv', sep = '\t')
    
    # Run functions for antibody
    
    ab_threshold,ab_binary=threshold_image_otsu(ab_image)
    # ab_threshold,ab_binary=threshold_image_fixed(ab_image,antibody_threshold)
    ab_number,ab_labelled=label_image(ab_binary)
    im = Image.fromarray(ab_binary)
    im.save(directory+'AB_Binary.tif')
    print("%d feautres were detected in the antibody image."%ab_number)
    ab_measurements=analyse_labelled_image(ab_labelled,ab_image)
    # ab_measurements.to_csv(directory + '/' + 'all_ab_metrics.csv', sep = '\t')
    
    # Run all functions for nucleus
    
    nuc_threshold,nuc_binary=threshold_image_otsu(nuc_image)
    # nuc_threshold,nuc_binary=threshold_image_fixed(nuc_image,nucleus_threshold)
    nuc_number,nuc_labelled=label_image(nuc_binary)
    im = Image.fromarray(nuc_binary)
    im.save(directory+'Nuc_Binary.tif')
    print("%d nuclei were detected in the image."%nuc_number)
    nuc_measurements=analyse_labelled_image(nuc_labelled,nuc_image)
    # nuc_measurements.to_csv(directory + '/' + 'all_nuc_metrics.csv', sep = '\t')
    
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

    
    aptamer_distance_to_nuc=minimum_distance(apt_measurements,nuc_measurements)*pixel_size/1000.0
    antibody_distance_to_nuc=minimum_distance(ab_measurements,nuc_measurements)*pixel_size/1000.0
    nuc_distance_to_nuc=minimum_distance(nuc_measurements,nuc_measurements)*pixel_size/1000.0
   
  
    
    
   # Output data
    ab_measurements['Distance to nucleus']=minimum_distance(ab_measurements,nuc_measurements)
    ab_measurements.to_csv(directory + '/' + 'all_ab_metrics.csv', sep = '\t')
    
    apt_measurements['Distance to nucleus']=minimum_distance(apt_measurements,nuc_measurements)
    apt_measurements.to_csv(directory + '/' + 'all_apt_metrics.csv', sep = '\t')
    
    Out_nc=open(directory+'/'+'Thresholds.txt','w')   # Open file for writing to.
    Out_nc.write("Nucleus threshold = %.2f \n" %np.float64(nuc_threshold))
    Out_nc.write("Aptamer threshold = %.2f \n" %np.float64(apt_threshold))
    Out_nc.write("Antibody threshold = %.2f \n" %np.float64(ab_threshold))
    Out_nc.close() # Close the file.

    imRGB = np.zeros((nuc_binary.shape[0],nuc_binary.shape[1],3))
    imRGB[:,:,0] = apt_binary
    imRGB[:,:,1] = ab_binary
    imRGB[:,:,2] = nuc_binary
    
    fig, ax = plt.subplots(1,1,figsize=(40, 40))
    ax.imshow(imRGB)
    plt.savefig(directory+"binary.png")
    
 
    # Plot histograms
    
    plt.hist(aptamer_distance_to_nuc, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Distance to nearest nucleus (\u03bcm)')
    plt.ylabel('Number of Features')
    plt.title('Aptamer distance to nearest nucleus')
    plt.savefig(directory+'/'+'Aptamer_to_nucleus_distances.pdf')
    
    plt.hist(antibody_distance_to_nuc, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Distance to nearest nucleus (\u03bcm)')
    plt.ylabel('Number of Features')
    plt.title('Antibody distance to nearest nucleus')
    plt.savefig(directory+'/'+'Antibody_to_nucleus_distances.pdf')
    
    apt_areas=apt_measurements['area']*(pixel_size/1000.0)**2
    apt_lengths=apt_measurements['major_axis_length']*pixel_size/1000.0
    apt_perimeter=apt_measurements['perimeter']*pixel_size/1000.0
    apt_mean_intensities=apt_measurements['mean_intensity']
    apt_max_intensities=apt_measurements['max_intensity']
    
    
    mean_apt_areas=apt_areas.mean()
    std_apt_areas=apt_areas.std()
    mean_apt_lengths=apt_lengths.mean()
    std_apt_lengths=apt_lengths.std()
    mean_apt_perimeter=apt_perimeter.mean()
    std_apt_perimeter=apt_perimeter.std()
    mean_apt_intensities=apt_mean_intensities.mean()
    std_apt_intensities=apt_mean_intensities.std()
    mean_apt_max_intensities=apt_max_intensities.mean()
    std_apt_max_intensities=apt_max_intensities.std()
    mean_apt_max_intensities=apt_max_intensities.mean()
    std_apt_max_intensities=apt_max_intensities.std()
    mean_apt_nuc=aptamer_distance_to_nuc.mean()
    std_apt_nuc=aptamer_distance_to_nuc.std()
    
    Out_nc=open(directory+'/'+'Aptamer_overall_stats.txt','w')   # Open file for writing to.
    Out_nc.write("Number of detected features = %.2f \n" %apt_number)
    Out_nc.write("Area = %.2f +/- %.2f \n" %(mean_apt_areas,std_apt_areas))
    Out_nc.write("Length = %.2f +/- %.2f \n" %(mean_apt_lengths,std_apt_lengths))
    Out_nc.write("Perimeter = %.2f +/- %.2f \n" %(mean_apt_perimeter,std_apt_perimeter))
    Out_nc.write("Mean intensity = %.2f +/- %.2f \n" %(mean_apt_intensities,std_apt_intensities))
    Out_nc.write("Max intensity = %.2f +/- %.2f \n" %(mean_apt_max_intensities,std_apt_max_intensities))
    Out_nc.write("Closest nucleus = %.2f +/- %.2f \n" %(mean_apt_nuc,std_apt_nuc))
    Out_nc.write("Pixel coincidence = %.2f \n" %apt_pixel_fraction)
    Out_nc.write("Feature coincidence = %.2f \n" %apt_fraction_coinc)
    Out_nc.write("Feature overlap with antibody = %.2f \n" %(sum(apt_fraction_pixels_overlap)/len(apt_fraction_pixels_overlap)))
    
    
    Out_nc.close() # Close the file.


    ab_areas=ab_measurements['area']*(pixel_size/1000.0)**2
    ab_lengths=ab_measurements['major_axis_length']*pixel_size/1000.0
    ab_perimeter=ab_measurements['perimeter']*pixel_size/1000.0
    ab_mean_intensities=ab_measurements['mean_intensity']
    ab_max_intensities=ab_measurements['max_intensity']
    

    mean_ab_areas=ab_areas.mean()
    std_ab_areas=ab_areas.std()
    mean_ab_lengths=ab_lengths.mean()
    std_ab_lengths=ab_lengths.std()
    mean_ab_perimeter=ab_perimeter.mean()
    std_ab_perimeter=ab_perimeter.std()
    mean_ab_intensities=ab_mean_intensities.mean()
    std_ab_intensities=ab_mean_intensities.std()
    mean_ab_max_intensities=ab_max_intensities.mean()
    std_ab_max_intensities=ab_max_intensities.std()
    mean_ab_max_intensities=ab_max_intensities.mean()
    std_ab_max_intensities=ab_max_intensities.std()
    mean_ab_nuc=antibody_distance_to_nuc.mean()
    std_ab_nuc=antibody_distance_to_nuc.std()
    
    Out_nc=open(directory+'/'+'antibody_overall_stats.txt','w')   # Open file for writing to.
    Out_nc.write("Number of detected features = %.2f \n" %ab_number)
    Out_nc.write("Area = %.2f +/- %.2f \n" %(mean_ab_areas,std_ab_areas))
    Out_nc.write("Length = %.2f +/- %.2f \n" %(mean_ab_lengths,std_ab_lengths))
    Out_nc.write("Perimeter = %.2f +/- %.2f \n" %(mean_ab_perimeter,std_ab_perimeter))
    Out_nc.write("Mean intensity = %.2f +/- %.2f \n" %(mean_ab_intensities,std_ab_intensities))
    Out_nc.write("Max intensity = %.2f +/- %.2f \n" %(mean_ab_max_intensities,std_ab_max_intensities))
    Out_nc.write("Closest nucleus = %.2f +/- %.2f \n" %(mean_ab_nuc,std_ab_nuc))
    Out_nc.write("Pixel coincidence = %.2f \n" %ab_pixel_fraction)
    Out_nc.write("Feature coincidence = %.2f \n" %ab_fraction_coinc)
    Out_nc.write("Feature overlap with aptamer= %.2f \n" %(sum(ab_fraction_pixels_overlap)/len(ab_fraction_pixels_overlap)))
    
    Out_nc.close() # Close the file.
    
    plt.hist(apt_areas, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Area of feature (\u03bcm$^2$)')
    plt.ylabel('Number of Features')
    plt.title('Area of aptamer feature')
    plt.savefig(directory+'/'+'Aptamer_areas.pdf')
    plt.show()
    
    plt.hist(apt_lengths, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Length (major axis) (\u03bcm)')
    plt.ylabel('Number of Features')
    plt.title('Length of aptamer features')
    plt.savefig(directory+'/'+'Aptamer_lengths.pdf')
    plt.show()
    
    plt.hist(apt_perimeter, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Perimeter of feature (\u03bcm)')
    plt.ylabel('Number of Features')
    plt.title('Perimeter of aptamer features')
    plt.savefig(directory+'/'+'Aptamer_perimeters.pdf')
    plt.show()
    
    plt.hist(apt_mean_intensities, bins = 50,range=[10000,100000], rwidth=0.9,color='#607c8e')
    plt.xlabel('Mean intensity (AU)')
    plt.ylabel('Number of Features')
    plt.title('Mean intensities')
    plt.savefig(directory+'/'+'Aptamer_intensities_mean.pdf')
    plt.show()
    
    plt.hist(apt_max_intensities, bins = 50,range=[10000,100000], rwidth=0.9,color='#607c8e')
    plt.xlabel('Maximum intensity (AU)')
    plt.ylabel('Number of Features')
    plt.title('Maximum intensities')
    plt.savefig(directory+'/'+'Aptamer_intensities_max.pdf')
    plt.show()
    
    plt.hist(ab_areas, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Area of feature (\u03bcm$^2$)')
    plt.ylabel('Number of Features')
    plt.title('Area of antibody feature')
    plt.savefig(directory+'/'+'antibody_areas.pdf')
    plt.show()
    
    plt.hist(ab_lengths, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Length (major axis) (\u03bcm)')
    plt.ylabel('Number of Features')
    plt.title('Length of antibody features')
    plt.savefig(directory+'/'+'antibody_lengths.pdf')
    plt.show()
    
    plt.hist(ab_perimeter, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
    plt.xlabel('Perimeter of feature (\u03bcm)')
    plt.ylabel('Number of Features')
    plt.title('Perimeter of antibody features')
    plt.savefig(directory+'/'+'antibody_perimeters.pdf')
    plt.show()
    
    plt.hist(ab_mean_intensities, bins = 50,range=[8000,30000], rwidth=0.9,color='#607c8e')
    plt.xlabel('Mean intensity (AU)')
    plt.ylabel('Number of Features')
    plt.title('Mean intensities')
    plt.savefig(directory+'/'+'antibody_intensities_mean.pdf')
    plt.show()
    
    plt.hist(ab_max_intensities, bins = 50,range=[8000,30000], rwidth=0.9,color='#607c8e')
    plt.xlabel('Maximum intensity (AU)')
    plt.ylabel('Number of Features')
    plt.title('Maximum intensities')
    plt.savefig(directory+'/'+'antibody_intensities_max.pdf')
    plt.show()



    Output_overall = Output_overall.append({'Image': directory, 'Nuc_threshold':nuc_threshold,'Ab_threshold':ab_threshold,
                                            'Apt_threshold':apt_threshold,'Number of nuclei':nuc_number,'ab number of features':ab_number,'apt number of features':apt_number,'ab area':mean_ab_areas,
                                            'ab area (SD)':std_ab_areas,'ab length':mean_ab_lengths,'ab length (SD)':std_ab_lengths,'ab perimeter':mean_ab_perimeter,'ab perimeter (SD)':std_ab_perimeter,
                                            'ab intensity':mean_ab_intensities,'ab intensity (SD)':std_ab_intensities,'ab max intensity':mean_ab_max_intensities,'ab max intensity (SD)':std_ab_max_intensities,
                                            'ab closest nuc':mean_ab_nuc,'ab closest nuc (SD)':std_ab_nuc,'ab pixel coincidence':ab_pixel_fraction,'ab feature coincidence':ab_fraction_coinc,
                                           'ab feature overlap':(sum(ab_fraction_pixels_overlap)/len(ab_fraction_pixels_overlap)),'apt area':mean_apt_areas,
                                            'apt area (SD)':std_apt_areas,'apt length':mean_apt_lengths,'apt length (SD)':std_apt_lengths,'apt perimeter':mean_apt_perimeter,'apt perimeter (SD)':std_apt_perimeter,
                                            'apt intensity':mean_apt_intensities,'apt intensity (SD)':std_apt_intensities,'apt max intensity':mean_apt_max_intensities,'apt max intensity (SD)':std_apt_max_intensities,
                                            'apt closest nuc':mean_apt_nuc,'apt closest nuc (SD)':std_apt_nuc,'apt pixel coincidence':apt_pixel_fraction,'apt feature coincidence':apt_fraction_coinc,
                                           'apt feature overlap':(sum(apt_fraction_pixels_overlap)/len(apt_fraction_pixels_overlap)),
                                          }, ignore_index=True)

    Output_overall.to_csv(root_directory + '/' + 'all.csv', sep = '\t')    


   
    