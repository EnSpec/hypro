#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:38:51 2019

@author: nanfeng
"""

from ENVI import read_envi_header, write_envi_header
import numpy as np

raw_sca_file = '/media/nanfeng/My Passport/Hyspex/Output/CEDAR-CREEK-EW_20180724_01/merge/CEDAR-CREEK-EW_20180724_01_SCA'
new_sca_file = '/media/nanfeng/My Passport/Hyspex/Output/CEDAR-CREEK-EW_20180724_01/merge/CEDAR-CREEK-EW_20180724_01_NewSCA'
background_mask_file = '/media/nanfeng/My Passport/Hyspex/Output/CEDAR-CREEK-EW_20180724_01/merge/CEDAR-CREEK-EW_20180724_01_BackgroundMask'

raw_sca_header = read_envi_header(raw_sca_file+'.hdr')
raw_sca_image = np.memmap(raw_sca_file, mode='r', dtype='float32', shape=(raw_sca_header['bands'],raw_sca_header['lines'],raw_sca_header['samples']))

bg_mask_header = read_envi_header(background_mask_file+'.hdr')
bg_mask_image = np.memmap(background_mask_file, mode='r', dtype='bool', shape=(bg_mask_header['lines'],bg_mask_header['samples']))


view_zenith = np.copy(raw_sca_image[0,:,:])
view_azimuth = np.copy(raw_sca_image[1,:,:])

raw_sca_image.flush()
sun_azimuth = float(raw_sca_header['sun zenith'])
relative_azimuth = sun_azimuth-view_azimuth

index = relative_azimuth>0.0
view_zenith[index] = -view_zenith[index]
view_zenith = view_zenith*100
view_zenith[bg_mask_image] = 9100
view_zenith = np.int16(view_zenith)

relative_azimuth[relative_azimuth<0] += 360.0
relative_azimuth[relative_azimuth>180] = 360.0-relative_azimuth[relative_azimuth>180]
relative_azimuth = relative_azimuth*10
relative_azimuth[bg_mask_image] = -1
relative_azimuth = np.int16(relative_azimuth)

abg_altitude = np.ones(relative_azimuth.shape, dtype='int16')*674
abg_altitude[bg_mask_image] = -1

import matplotlib.pyplot as plt
plt.imshow(view_zenith)
plt.colorbar()
plt.show()

plt.imshow(relative_azimuth)
plt.colorbar()
plt.show()

fid = open(new_sca_file, 'wb')
fid.write(view_zenith.tostring())
fid.write(relative_azimuth.tostring())
fid.write(abg_altitude.tostring())
fid.close()

raw_sca_header['bands'] = 3
raw_sca_header['data type'] = 2
raw_sca_header['GPS long-lat-alt'] = [-93.16, 45.40, 676.65]
raw_sca_header['heading[deg]'] = 273.5
raw_sca_header['DEM height[m]'] = 255.12
write_envi_header(new_sca_file+'.hdr', raw_sca_header)
