#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:53:40 2020

@author: Mathew
"""
# These are the packages we are using. 
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters,measure
from PIL import Image


filename="ALS_1.5uM_APT_washes.tif"
directory="/Users/Mathew/Desktop/Axioscan/"

image=imread(directory+filename)


# This is to extract the frames that contain the antibody, aptamer and nucleus
imagenuc=image[:,:,0]
imageab=image[:,:,1]
imageapt=image[:,:,2]

# imagenuc=image[500:1000,500:1000,0]
# imageab=image[500:1000,500:1000,1]
# imageapt=image[500:1000,500:1000,2]



# Determine a threshold using the Otsu method - Note, if I was comparing cases etc., I'd keep the threhsold constant, and not use Otsu for all of them. 
threshold_ab=filters.threshold_otsu(imageab)    
filteredab=imageab>threshold_ab

threshold_apt=filters.threshold_otsu(imageapt)    
filteredapt=imageapt>threshold_apt   

# Show the binary images. 

# plt.imshow(filteredab,cmap="Blues")
# plt.show()

# plt.imshow(filteredapt,cmap="Reds")
# plt.show()

# Coincidence- this is to just look at coincidence on a pixel-by-pixel basis.

Coinc_image_pixels=filteredapt & filteredab

# plt.imshow(Coinc_image_pixels)
# plt.show()

# Now to get some statistics:

aptamer_coinc_pixels=Coinc_image_pixels.sum()/filteredapt.sum()

print("The fraction of aptamer pixels coincident is %.2f" %aptamer_coinc_pixels)

antibody_coinc_pixels=Coinc_image_pixels.sum()/filteredab.sum()

print("The fraction of antibody pixels coincident is %.2f" %antibody_coinc_pixels)





# Now find the different features in the thresholded image. The measure.label function selects the different features and 
# labels them.
label_ab=measure.label(filteredab)
label_apt=measure.label(filteredapt)


# Need to find which labels are also coincident- i.e. if aptamer, also antibody?

# This multiplies the coincident binary image with the labelled image
coincident_label_apt=label_apt*Coinc_image_pixels

# This detects/counts the number of unqiue numbers in the image- i.e. labels
coinc_aptamer_list, coinc_apt_pixels = np.unique(coincident_label_apt, return_counts=True)

# Repeat for AB
coincident_label_ab=label_ab*Coinc_image_pixels
coinc_antibody_list, coinc_antibody_pixels = np.unique(coincident_label_ab, return_counts=True)

# Now get some statistics
total_apt_clusters=label_apt.max()
total_ab_clusters=label_ab.max()

total_apt_coinc_clusters=len(coinc_aptamer_list)
total_ab_coinc_clusters=len(coinc_antibody_list)

fraction_apt_clust_coinc=total_apt_coinc_clusters/total_apt_clusters
fraction_ab_clust_coinc=total_ab_coinc_clusters/total_ab_clusters

print("The fraction of aptamer clusters coincident is %.2f" %fraction_apt_clust_coinc)

print("The fraction of antibody clusters coincident is %.2f" %fraction_ab_clust_coinc)

# Now need to generate coincident and non-coincident images

coinc_aptamer_list[0]=1000000   # First value is zero- don't want to count these. 

aptamer_coincident=np.isin(label_apt,coinc_aptamer_list)
# plt.imshow(aptamer_coincident,cmap="Reds")
# plt.show()
coinc_aptamer_list[0]=0

aptamer_noncoincident=~np.isin(label_apt,coinc_aptamer_list)
# plt.imshow(aptamer_noncoincident,cmap="Reds")
# plt.show()

aptamer_coincident_tosave=aptamer_coincident*imageapt
im = Image.fromarray(aptamer_coincident_tosave)
im.save(directory+'aptamer_coinc.tif')

aptamer_noncoincident_tosave=aptamer_noncoincident*imageapt
im = Image.fromarray(aptamer_noncoincident_tosave)
im.save(directory+'aptamer_noncoinc.tif')


# Now for the antibody channels:
    
coinc_antibody_list[0]=1000000   # First value is zero- don't want to count these. 

antibody_coincident=np.isin(label_ab,coinc_antibody_list)
# plt.imshow(antibody_coincident,cmap="Reds")
# plt.show()
coinc_antibody_list[0]=0

antibody_noncoincident=~np.isin(label_ab,coinc_antibody_list)
# plt.imshow(antibody_noncoincident,cmap="Reds")
# plt.show()

antibody_coincident_tosave=antibody_coincident*imageab
im = Image.fromarray(antibody_coincident_tosave)
im.save(directory+'antibody_coinc.tif')

antibody_noncoincident_tosave=antibody_noncoincident*imageab
im = Image.fromarray(antibody_noncoincident_tosave)
im.save(directory+'antibody_noncoinc.tif')



# Determine percentage overlap in each cluster- i.e. proportion of aptamer coincident with AB and viceversa.


label_ab=measure.label(filteredab)
label_apt=measure.label(filteredapt)


# Need to find which labels are also coincident- i.e. if aptamer, also antibody?

# This detects/counts the number of unqiue numbers in the image- i.e. labels
aptamer_list, apt_pixels = np.unique(label_apt, return_counts=True)


apt_fract_overlap=[]
for i in range(len(coinc_aptamer_list)):
    overlap_pixels=coinc_apt_pixels[i]
    num=coinc_aptamer_list[i]
    total_pixels=apt_pixels[num]
    apt_fract=1.0*overlap_pixels/total_pixels
    apt_fract_overlap.append(apt_fract)

antibody_list, antibody_pixels = np.unique(label_ab, return_counts=True)


ab_fract_overlap=[]
for i in range(len(coinc_antibody_list)):
    overlap_pixels=coinc_antibody_pixels[i]
    num=coinc_antibody_list[i]
    total_pixels=antibody_pixels[num]
    ab_fract=1.0*overlap_pixels/total_pixels
    ab_fract_overlap.append(ab_fract)




# Measure parameters of labelled regions. 

table=measure.regionprops_table(label_apt,properties=('area','centroid','orientation','major_axis_length','minor_axis_length'))


# Get the area and length data. 
areas=table['area']
lengths=table['major_axis_length']

number=len(areas)  # Count the number of features detected. 

print(number)

# Plot some histograms. 

plt.hist(areas, bins = 50,range=[0,100], rwidth=0.9,color='#607c8e')
plt.xlabel('Area (pixels)')
plt.ylabel('Number of Features')
plt.title('Area of features')
plt.show()

plt.hist(lengths, bins = 50,range=[0,200], rwidth=0.9,color='#607c8e')
plt.xlabel('Length (pixels)')
plt.ylabel('Number of Features')
plt.title('Length')
plt.show()



