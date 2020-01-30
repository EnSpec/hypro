#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to do image classifications.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)

solar_flux_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','solar_flux.dat')

def pre_classification(pre_class_image_file, rdn_image_file, sun_zenith, distance, background_mask_file=None):
    """ Pre-classify the image.
    Notes:
        (1) The classification algorithm used here is from ATCOR.
    Arguments:
        pre_class_image_file: str
            Pre-classification image filename.
        rdn_image_file: str
            Radiance image filename, either BIL or BSQ.
        sun_zenith: float
            Sun zenith angle in degrees.
        distance: float
            Sun-Earth distance.
        background_mask_file: str
            Background mask filename.
    """

    if os.path.exists(pre_class_image_file):
        logger.info('Write the pre-classification map to %s.' %(pre_class_image_file))
        return

    from ENVI     import empty_envi_header, read_envi_header, write_envi_header
    from Spectra import get_closest_wave, resample_solar_flux

    # Read radiance image data.
    rdn_header = read_envi_header(os.path.splitext(rdn_image_file)[0]+'.hdr')
    if rdn_header['interleave'].lower() == 'bil':
        rdn_image = np.memmap(rdn_image_file,
                              dtype='float32',
                              mode='r',
                              shape=(rdn_header['lines'],
                                     rdn_header['bands'],
                                     rdn_header['samples']))
    elif rdn_header['interleave'].lower() == 'bsq':
        rdn_image = np.memmap(rdn_image_file,
                              dtype='float32',
                              mode='r',
                              shape=(rdn_header['bands'],
                                     rdn_header['lines'],
                                     rdn_header['samples']))
    else:
        logger.error('Cannot read radiance data from %s format file.' %rdn_header['interleave'])

    # Read solar flux.
    solar_flux = resample_solar_flux(solar_flux_file, rdn_header['wavelength'], rdn_header['fwhm'])
    cos_sun_zenith = np.cos(np.deg2rad(sun_zenith))
    d2 = distance**2

    # Initialize classification.
    pre_class_image = np.memmap(pre_class_image_file,
                                dtype='uint8',
                                mode='w+',
                                shape=(rdn_header['lines'],
                                       rdn_header['samples']))
    pre_class_image[:,:] = 0

    # Define VNIR sensor wavelengths.
    blue_wave, blue_band = get_closest_wave(rdn_header['wavelength'], 470)
    green_wave, green_band = get_closest_wave(rdn_header['wavelength'], 550)
    red_wave, red_band = get_closest_wave(rdn_header['wavelength'], 660)
    nir_wave, nir_band = get_closest_wave(rdn_header['wavelength'], 850)

    # Define SWIR sensor wavelengths.
    cirrus_wave, cirrus_band = get_closest_wave(rdn_header['wavelength'], 1380)
    swir1_wave, swir1_band = get_closest_wave(rdn_header['wavelength'], 1650)
    swir2_wave, swir2_band = get_closest_wave(rdn_header['wavelength'], 2200)

    # Determine the sensor type.
    if_vnir =  abs(blue_wave-470)<20 and abs(green_wave-550)<20 and abs(red_wave-660)<20 and abs(nir_wave-850)<20
    if_swir =  abs(cirrus_wave-1380)<20 and abs(swir1_wave-1650)<20 and abs(swir2_wave-2200)<20
    if_whole = if_vnir and if_swir

    # Do classification.
    if if_whole:
        # Calculate the reflectance at different bands.
        if rdn_header['interleave'].lower() == 'bil':
            blue_refl = rdn_image[:,blue_band,:]*np.pi*d2/(solar_flux[blue_band]*cos_sun_zenith)
            green_refl = rdn_image[:,green_band,:]*np.pi*d2/(solar_flux[green_band]*cos_sun_zenith)
            red_refl = rdn_image[:,red_band,:]*np.pi*d2/(solar_flux[red_band]*cos_sun_zenith)
            nir_refl = rdn_image[:,nir_band,:]*np.pi*d2/(solar_flux[nir_band]*cos_sun_zenith)

            cirrus_refl = rdn_image[:,cirrus_band,:]*np.pi*d2/(solar_flux[cirrus_band]*cos_sun_zenith)
            swir1_refl = rdn_image[:,swir1_band,:]*np.pi*d2/(solar_flux[swir1_band]*cos_sun_zenith)
            swir2_refl = rdn_image[:,swir2_band,:]*np.pi*d2/(solar_flux[swir2_band]*cos_sun_zenith)
        else:
            blue_refl = rdn_image[blue_band,:,:]*np.pi*d2/(solar_flux[blue_band]*cos_sun_zenith)
            green_refl = rdn_image[green_band,:,:]*np.pi*d2/(solar_flux[green_band]*cos_sun_zenith)
            red_refl = rdn_image[red_band,:,:]*np.pi*d2/(solar_flux[red_band]*cos_sun_zenith)
            nir_refl = rdn_image[nir_band,:,:]*np.pi*d2/(solar_flux[nir_band]*cos_sun_zenith)

            cirrus_refl = rdn_image[cirrus_band,:,:]*np.pi*d2/(solar_flux[cirrus_band]*cos_sun_zenith)
            swir1_refl = rdn_image[swir1_band,:,:]*np.pi*d2/(solar_flux[swir1_band]*cos_sun_zenith)
            swir2_refl = rdn_image[swir2_band,:,:]*np.pi*d2/(solar_flux[swir2_band]*cos_sun_zenith)
        rdn_image.flush()
        del rdn_image
        
        # Calculate NDSI.
        ndsi = (green_refl-swir1_refl)/(green_refl+swir1_refl+1e-10)

        # water
        water = (nir_refl<0.05)&(swir1_refl<0.03)

        # land
        land = ~water

        # shadow
        shadow = (water|land)&(green_refl<0.01)
        land = land&(~shadow)
        water = water&(~shadow)

        # cloud over land
        Tc = 0.20
        cloud_over_land = land&(blue_refl>Tc)&(red_refl>0.15)&(nir_refl<red_refl*2.0)&(nir_refl>red_refl*0.8)&(nir_refl>swir1_refl)&(ndsi<0.7)
        land = land&(~cloud_over_land)

        # cloud over water
        cloud_over_water = water&(blue_refl<0.40)&(blue_refl>0.20)&(green_refl<blue_refl)&(nir_refl<green_refl)&(swir1_refl<0.15)&(ndsi<0.20)
        water = water&(~cloud_over_water)

        # cloud shadow
        cloud_shadow = (water|land)&(nir_refl<0.12)&(nir_refl>0.04)&(swir1_refl<0.20)
        land = land&(~cloud_shadow)
        water = water&(~cloud_shadow)

        # snow/ice
        snow_ice = land&(((blue_refl>0.22)&(ndsi>0.60))|((green_refl>0.22)&(ndsi>0.25)&(swir2_refl<0.5*green_refl)))
        land = land&(~snow_ice)

        # cirrus over land and water
        thin_cirrus = (cirrus_refl<0.015)&(cirrus_refl>0.010)
        medium_cirrus = (cirrus_refl<0.025)&(cirrus_refl>=0.015)
        thick_cirrus = (cirrus_refl<0.040)&(cirrus_refl>=0.025)

        thin_cirrus_over_land = land&thin_cirrus
        land = land&(~thin_cirrus_over_land)

        medium_cirrus_over_land = land&medium_cirrus
        land = land&(~medium_cirrus_over_land)

        thick_cirrus_over_land = land&thick_cirrus
        land = land&(~thick_cirrus_over_land)

        thin_cirrus_over_water = water&thin_cirrus
        water = water&(~thin_cirrus_over_water)

        medium_cirrus_over_water = water&thin_cirrus
        water = water&(~medium_cirrus_over_water)

        thick_cirrus_over_water = water&thin_cirrus
        water = water&(~thick_cirrus_over_water)

        cirrus_cloud = (water|land)&(cirrus_refl<0.050)&(cirrus_refl>0.040)
        land = land&(~cirrus_cloud)
        water = water&(~cirrus_cloud)

        thick_cirrus_cloud = (water|land)&(cirrus_refl>0.050)
        land = land&(~thick_cirrus_cloud)
        water = water&(~thick_cirrus_cloud)

        # haze over water
        T2 = 0.04
        T1 = 0.12
        thin_haze_over_water = water&(nir_refl>=T1)&(nir_refl<=0.06)
        water = water&(~thin_haze_over_water)
        medium_haze_over_water = water&(nir_refl>=0.06)&(nir_refl<=T2)
        water = water&(~medium_haze_over_water)

        # Assign class values.
        pre_class_image[shadow] = 1
        pre_class_image[thin_cirrus_over_water] = 2
        pre_class_image[medium_cirrus_over_water] = 3
        pre_class_image[thick_cirrus_over_water] = 4
        pre_class_image[land] = 5
        pre_class_image[snow_ice] = 7
        pre_class_image[thin_cirrus_over_land] = 8
        pre_class_image[medium_cirrus_over_land] = 9
        pre_class_image[thick_cirrus_over_land] = 10
        pre_class_image[thin_haze_over_water] = 13
        pre_class_image[medium_haze_over_water] = 14
        pre_class_image[cloud_over_land] = 15
        pre_class_image[cloud_over_water] = 16
        pre_class_image[water] = 17
        pre_class_image[cirrus_cloud] = 18
        pre_class_image[thick_cirrus_cloud] = 19
        pre_class_image[cloud_shadow] = 22

        # Clear data.
        del shadow, thin_cirrus_over_water, medium_cirrus_over_water
        del thick_cirrus_over_water, land, snow_ice
        del thin_cirrus_over_land, medium_cirrus_over_land, thick_cirrus_over_land
        del thin_haze_over_water, medium_haze_over_water, cloud_over_land
        del cloud_over_water, water, cirrus_cloud
        del thick_cirrus_cloud, cloud_shadow

    elif if_vnir:
        # Calculate the reflectance at different bands.
        if rdn_header['interleave'].lower() == 'bil':
            blue_refl = rdn_image[:,blue_band,:]*np.pi*d2/(solar_flux[blue_band]*cos_sun_zenith)
            green_refl = rdn_image[:,green_band,:]*np.pi*d2/(solar_flux[green_band]*cos_sun_zenith)
            red_refl = rdn_image[:,red_band,:]*np.pi*d2/(solar_flux[red_band]*cos_sun_zenith)
            nir_refl = rdn_image[:,nir_band,:]*np.pi*d2/(solar_flux[nir_band]*cos_sun_zenith)
        else:
            blue_refl = rdn_image[blue_band,:,:]*np.pi*d2/(solar_flux[blue_band]*cos_sun_zenith)
            green_refl = rdn_image[green_band,:,:]*np.pi*d2/(solar_flux[green_band]*cos_sun_zenith)
            red_refl = rdn_image[red_band,:,:]*np.pi*d2/(solar_flux[red_band]*cos_sun_zenith)
            nir_refl = rdn_image[nir_band,:,:]*np.pi*d2/(solar_flux[nir_band]*cos_sun_zenith)
        rdn_image.flush()
        del rdn_image
        
        # water
        water = nir_refl<0.05

        # land
        land = ~water

        # shadow
        shadow = (water|land)&(green_refl<0.01)
        land = land&(~shadow)
        water = water&(~shadow)

        # cloud over land
        Tc = 0.20
        cloud_over_land = land&(blue_refl>Tc)&(red_refl>0.15)&(nir_refl<red_refl*2.0)&(nir_refl>red_refl*0.8)
        land = land&(~cloud_over_land)

        # cloud over water
        cloud_over_water = water&(blue_refl<0.40)&(blue_refl>0.20)&(green_refl<blue_refl)&(nir_refl<green_refl)
        water = water&(~cloud_over_water)

        # cloud shadow
        cloud_shadow = (water|land)&(nir_refl<0.12)&(nir_refl>0.04)
        land = land&(~cloud_shadow)
        water = water&(~cloud_shadow)

        # haze over water
        T2 = 0.04
        T1 = 0.12
        thin_haze_over_water = water&(nir_refl>=T1)&(nir_refl<=0.06)
        water = water&(~thin_haze_over_water)
        medium_haze_over_water = water&(nir_refl>=0.06)&(nir_refl<=T2)
        water = water&(~medium_haze_over_water)

        # Assign class values.
        pre_class_image[shadow] = 1
        pre_class_image[land] = 5
        pre_class_image[thin_haze_over_water] = 13
        pre_class_image[medium_haze_over_water] = 14
        pre_class_image[cloud_over_land] = 15
        pre_class_image[cloud_over_water] = 16
        pre_class_image[water] = 17
        pre_class_image[cloud_shadow] = 22

        # Clear data.
        del shadow, land, thin_haze_over_water
        del medium_haze_over_water, cloud_over_land, cloud_over_water
        del water, cloud_shadow

    elif if_swir:
        # Calculate the reflectance at different bands.
        if rdn_header['interleave'] == 'bil':
            cirrus_refl = rdn_image[:,cirrus_band,:]*np.pi*d2/(solar_flux[cirrus_band]*cos_sun_zenith)
            swir1_refl = rdn_image[:,swir1_band,:]*np.pi*d2/(solar_flux[swir1_band]*cos_sun_zenith)
            swir2_refl = rdn_image[:,swir2_band,:]*np.pi*d2/(solar_flux[swir2_band]*cos_sun_zenith)
        else:
            cirrus_refl = rdn_image[cirrus_band,:,:]*np.pi*d2/(solar_flux[cirrus_band]*cos_sun_zenith)
            swir1_refl = rdn_image[swir1_band,:,:]*np.pi*d2/(solar_flux[swir1_band]*cos_sun_zenith)
            swir2_refl = rdn_image[swir2_band,:,:]*np.pi*d2/(solar_flux[swir2_band]*cos_sun_zenith)
        rdn_image.flush()
        del rdn_image
        
        # water
        water = swir1_refl<0.03

        # land
        land = ~water

        # cirrus over land and water
        thin_cirrus = (cirrus_refl<0.015)&(cirrus_refl>0.010)
        medium_cirrus = (cirrus_refl<0.025)&(cirrus_refl>=0.015)
        thick_cirrus = (cirrus_refl<0.040)&(cirrus_refl>=0.025)

        thin_cirrus_over_land = land&thin_cirrus
        land = land&(~thin_cirrus_over_land)

        medium_cirrus_over_land = land&medium_cirrus
        land = land&(~medium_cirrus_over_land)

        thick_cirrus_over_land = land&thick_cirrus
        land = land&(~thick_cirrus_over_land)

        thin_cirrus_over_water = water&thin_cirrus
        water = water&(~thin_cirrus_over_water)

        medium_cirrus_over_water = water&thin_cirrus
        water = water&(~medium_cirrus_over_water)

        thick_cirrus_over_water = water&thin_cirrus
        water = water&(~thick_cirrus_over_water)

        cirrus_cloud = (water|land)&(cirrus_refl<0.050)&(cirrus_refl>0.040)
        land = land&(~cirrus_cloud)
        water = water&(~cirrus_cloud)

        thick_cirrus_cloud = (water|land)&(cirrus_refl>0.050)
        land = land&(~thick_cirrus_cloud)
        water = water&(~thick_cirrus_cloud)

        # Assign class values.
        pre_class_image[thin_cirrus_over_water] = 2
        pre_class_image[medium_cirrus_over_water] = 3
        pre_class_image[thick_cirrus_over_water] = 4
        pre_class_image[land] = 5
        pre_class_image[thin_cirrus_over_land] = 8
        pre_class_image[medium_cirrus_over_land] = 9
        pre_class_image[thick_cirrus_over_land] = 10
        pre_class_image[water] = 17
        pre_class_image[cirrus_cloud] = 18
        pre_class_image[thick_cirrus_cloud] = 19

        # Clear data.
        del thin_cirrus_over_water, medium_cirrus_over_water, thick_cirrus_over_water,
        del land, thin_cirrus_over_land, medium_cirrus_over_land, thick_cirrus_over_land
        del water, cirrus_cloud, thick_cirrus_cloud

    else:
        logger.error('Cannot find appropriate wavelengths to do pre-classification.')

    # Apply background mask.
    if background_mask_file is not None:
        bg_mask_header = read_envi_header(os.path.splitext(background_mask_file)[0]+'.hdr')
        bg_mask_image = np.memmap(background_mask_file,
                                    mode='r',
                                    dtype='bool',
                                    shape=(bg_mask_header['lines'],
                                           bg_mask_header['samples']))
        pre_class_image[bg_mask_image] = 0
        bg_mask_image.flush()
        del bg_mask_image
        
    # Clear data.
    pre_class_image.flush()
    del pre_class_image
    
    # Write header.
    pre_class_header = empty_envi_header()
    pre_class_header['description'] = 'Pre-classification'
    pre_class_header['file type'] = 'ENVI Standard'
    pre_class_header['samples'] = rdn_header['samples']
    pre_class_header['lines'] = rdn_header['lines']
    pre_class_header['classes'] = 23
    pre_class_header['bands'] = 1
    pre_class_header['byte order'] = 0
    pre_class_header['header offset'] = 0
    pre_class_header['interleave'] = 'bsq'
    pre_class_header['data type'] = 1
    pre_class_header['class names'] = [
                    '0: geocoded background',
                    '1: shadow',
                    '2: thin cirrus (water)',
                    '3: medium cirrus (water)',
                    '4: thick cirrus (water)',
                    '5: land (clear)',
                    '6: saturated',
                    '7: snow/ice',
                    '8: thin cirrus (land)',
                    '9: medium cirrus (land)',
                    '10: thick cirrus (land)',
                    '11: thin haze (land)',
                    '12: medium haze (land)',
                    '13: thin haze/glint (water)',
                    '14: med. haze/glint (water)',
                    '15: cloud (land)',
                    '16: cloud (water)',
                    '17: water',
                    '18: cirrus cloud',
                    '19: cirrus cloud (thick)',
                    '20: bright',
                    '21: topogr. shadow',
                    '22: cloud shadow']
    pre_class_header['class lookup'] = [
                    80, 80, 80,
                    0,  0,  0,
                    0, 160, 250,
                    0, 200, 250,
                    0, 240, 250,
                    180, 100, 40,
                    200,  0,  0,
                    250, 250, 250,
                    240, 240, 170,
                    225, 225, 150,
                    210, 210, 120,
                    250, 250, 100,
                    230, 230, 80,
                    80, 100, 250,
                    130, 130, 250,
                    180, 180, 180,
                    160, 160, 240,
                    0,  0, 250,
                    180, 180, 100,
                    160, 160, 100,
                    200, 200, 200,
                    40, 40, 40,
                    50, 50, 50]

    pre_class_header['map info'] = rdn_header['map info']
    pre_class_header['coordinate system string'] = rdn_header['coordinate system string']
    write_envi_header(os.path.splitext(pre_class_image_file)[0]+'.hdr', pre_class_header)

    logger.info('Write the pre-classification map to %s.' %(pre_class_image_file))
