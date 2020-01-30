#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to estimate visibility.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)

# Define typical water vapor column values of the atmosphere database. Do not make any change to them.
atm_db_wvc_lut = {'subarctic_winter': 4.2,
                  'midlatitude_winter': 8.5,
                  'USstandard': 14.2,
                  'subarctic_summer': 20.8,
                  'midlatitude_summer': 29.2,
                  'tropical': 41.1}

def estimate_vis(vis_file, ddv_file, atm_lut_file, rdn_file, sca_file, background_mask_file):
    """ Estimate visibility.
    Arguments:
        vis_file: str
            Visibility map filename.
        ddv_file: str
            Dark dense vegetation map filename.
        atm_lut_file: str
            Atmospheric lookup table filename.
        rdn_file: str
            Radiance filename.
        sca_file: str
            Scan angle filename.
        background_mask_file:
            Background mask filename.
    """

    if os.path.exists(ddv_file) and os.path.exists(vis_file):
        logger.info('Write the visibility map to %s.' %vis_file)
        logger.info('Write the DDV map to %s.' %ddv_file)
        return

    from ENVI    import read_envi_header, empty_envi_header, write_envi_header
    from AtmLUT  import read_binary_metadata
    from Spectra import get_closest_wave
    from AtmCorr import atm_corr_band

    # Read radiance header.
    rdn_header = read_envi_header(os.path.splitext(rdn_file)[0]+'.hdr')

    # Find VNIR and SWIR sensor wavelengths.
    red_wave, red_band = get_closest_wave(rdn_header['wavelength'], 660)
    nir_wave, nir_band = get_closest_wave(rdn_header['wavelength'], 850)
    swir1_wave, swir1_band = get_closest_wave(rdn_header['wavelength'], 1650)
    swir2_wave, swir2_band = get_closest_wave(rdn_header['wavelength'], 2130)

    # Determine the sensor type.
    if_vnir =  abs(red_wave-660)<20 and abs(nir_wave-850)<20
    if_swir =  abs(swir1_wave-1650)<20 or abs(swir2_wave-2130)<20

    if if_vnir and if_swir:
        logger.info('Both VNIR and SWIR bands are used for estimating visibility.')
    elif if_vnir:
        logger.info('Only VNIR bands are used for estimating visibility.')
    elif if_swir:
        logger.info('Only SWIR bands are available, a constant visibility (23 km) is used.')
    else:
        logger.error('Cannot find appropriate bands for estimating visibility.')

    # Read atmospheric lookup table.
    atm_lut_metadata = read_binary_metadata(atm_lut_file+'.meta')
    atm_lut_metadata['shape'] = tuple([int(v) for v in atm_lut_metadata['shape']])
    atm_lut_RHO = np.array([float(v) for v in atm_lut_metadata['RHO']])
    atm_lut_WVC = np.array([float(v) for v in atm_lut_metadata['WVC']])
    atm_lut_VIS = np.array([float(v) for v in atm_lut_metadata['VIS']])
    atm_lut_VZA = np.array([float(v) for v in atm_lut_metadata['VZA']])
    atm_lut_RAA = np.array([float(v) for v in atm_lut_metadata['RAA']])

    atm_lut = np.memmap(atm_lut_file,
                        dtype=atm_lut_metadata['dtype'],
                        mode='r',
                        shape=atm_lut_metadata['shape'])# shape = (RHO, WVC, VIS, VZA, RAA, WAVE)

    # Read radiance image.
    rdn_image = np.memmap(rdn_file,
                          dtype='float32',
                          mode='r',
                          shape=(rdn_header['bands'],
                                 rdn_header['lines'],
                                 rdn_header['samples']))

    # Read VZA and RAA image.
    sca_header = read_envi_header(os.path.splitext(sca_file)[0]+'.hdr')
    saa = float(sca_header['sun azimuth'])
    sca_image = np.memmap(sca_file,
                          dtype='float32',
                          offset=0,
                          shape=(sca_header['bands'],
                                 sca_header['lines'],
                                 sca_header['samples']))
    # vza
    vza_image = np.copy(sca_image[0,:,:])
    # raa
    raa_image = saa-sca_image[1,:,:]
    raa_image[raa_image<0] += 360.0
    raa_image[raa_image>180] = 360.0-raa_image[raa_image>180]
    # clear data
    sca_image.flush()
    del sca_header, saa, sca_image

    # Set visibility and water vapor column values.
    metadata = read_binary_metadata(atm_lut_file+'.meta')
    tmp_wvc_image = np.ones(vza_image.shape)*atm_db_wvc_lut[metadata['atm_mode']]
    tmp_vis_image = np.ones(vza_image.shape)*23
    del metadata

    # Read background mask.
    bg_header = read_envi_header(os.path.splitext(background_mask_file)[0]+'.hdr')
    bg_mask = np.memmap(background_mask_file,
                        dtype='bool',
                        mode='r',
                        shape=(bg_header['lines'],
                               bg_header['samples']))

    # Calculate NDVI.
    red_refl = atm_corr_band(atm_lut_WVC, atm_lut_VIS, atm_lut_VZA, atm_lut_RAA, atm_lut[...,red_band],
                             tmp_wvc_image, tmp_vis_image, vza_image, raa_image, rdn_image[red_band,:,:],
                             bg_mask)
    
    nir_refl = atm_corr_band(atm_lut_WVC, atm_lut_VIS, atm_lut_VZA, atm_lut_RAA, atm_lut[...,nir_band],
                             tmp_wvc_image, tmp_vis_image, vza_image, raa_image, rdn_image[nir_band,:,:],
                             bg_mask)
    ndvi = (nir_refl-red_refl)/(nir_refl+red_refl+1e-10)
    vis_image = np.zeros((rdn_header['lines'], rdn_header['samples']))

    if if_vnir and if_swir:
        # Decide which SWIR band to use.
        if abs(swir2_wave-2130)<20:
            swir_wave = swir2_wave
            swir_band = swir2_band
            swir_refl_upper_bounds = [0.05, 0.10, 0.12]
            red_swir_ratio = 0.50
        else:
            swir_wave = swir1_wave
            swir_band = swir1_band
            swir_refl_upper_bounds = [0.10, 0.15, 0.18]
            red_swir_ratio = 0.25

        # Calculate swir refletance.
        swir_refl = atm_corr_band(atm_lut_WVC, atm_lut_VIS, atm_lut_VZA, atm_lut_RAA, atm_lut[...,swir_band],
                                  tmp_wvc_image, tmp_vis_image, vza_image, raa_image, rdn_image[swir_band,:,:],
                                  bg_mask)

        # Get DDV mask.
        for swir_refl_upper_bound in swir_refl_upper_bounds:
            ddv_mask = (ndvi>0.10)&(swir_refl<swir_refl_upper_bound)&(swir_refl>0.01)
            percentage = np.sum(ddv_mask[~bg_mask])/np.sum(~bg_mask)
            if percentage > 0.02:
                logger.info('The SWIR wavelength %.2f is used for detecting dark dense vegetation.' %swir_wave)
                logger.info('The SWIR reflectance upper boundary is %.2f.' %swir_refl_upper_bound)
                logger.info('The number of DDV pixels is %.2f%%.' %(percentage*100))
                break
        # Estimate Visibility.
        rows, columns = np.where(ddv_mask)

        # Estimate red reflectance.
        red_refl[rows, columns] = red_swir_ratio*swir_refl[rows, columns]
        del swir_refl
        for row, column in zip(rows, columns):
            interp_rdn = interp_atm_lut(atm_lut_RHO, atm_lut_WVC, atm_lut_VZA, atm_lut_RAA, atm_lut[...,red_band],
                                        red_refl[row,column], tmp_wvc_image[row, column], vza_image[row, column], raa_image[row, column])
            vis_image[row, column] = np.interp(rdn_image[red_band,row,column], interp_rdn[::-1], atm_lut_VIS[::-1])
        rdn_image.flush()
        del rdn_image
        logger.info('Visibility [km] statistics: min=%.2f, max=%.2f, avg=%.2f, sd=%.2f.' %(vis_image[ddv_mask].min(), vis_image[ddv_mask].max(), vis_image[ddv_mask].mean(), vis_image[ddv_mask].std()))

        # Fill gaps with average values.
        vis_image[~ddv_mask] = vis_image[ddv_mask].mean()
        vis_image[bg_mask] = -1000.0

        # Write the visibility data.
        fid = open(vis_file, 'wb')
        fid.write(vis_image.astype('float32').tostring())
        fid.close()
        del vis_image

        vis_header = empty_envi_header()
        vis_header['description'] = 'Visibility [km]'
        vis_header['samples'] = rdn_header['samples']
        vis_header['lines'] = rdn_header['lines']
        vis_header['bands'] = 1
        vis_header['byte order'] = 0
        vis_header['header offset'] = 0
        vis_header['interleave'] = 'bsq'
        vis_header['data ignore value'] = -1000.0
        vis_header['data type'] = 4
        vis_header['map info'] = rdn_header['map info']
        vis_header['coordinate system string'] = rdn_header['coordinate system string']
        write_envi_header(os.path.splitext(vis_file)[0]+'.hdr', vis_header)
        logger.info('Write the visibility image to %s.' %vis_file)

        # Write the DDV data.
        fid = open(ddv_file, 'wb')
        fid.write(ddv_mask.tostring())
        fid.close()
        del ddv_mask

        ddv_header = empty_envi_header()
        ddv_header['description'] = 'DDV mask (1: Dark dense vegetaion; 0: non-dark dense vegetation)'
        ddv_header['samples'] = rdn_header['samples']
        ddv_header['lines'] = rdn_header['lines']
        ddv_header['bands'] = 1
        ddv_header['byte order'] = 0
        ddv_header['header offset'] = 0
        ddv_header['interleave'] = 'bsq'
        ddv_header['data type'] = 1
        ddv_header['map info'] = rdn_header['map info']
        ddv_header['coordinate system string'] = rdn_header['coordinate system string']
        write_envi_header(os.path.splitext(ddv_file)[0]+'.hdr', ddv_header)

        logger.info('Write the DDV mask to %s.' %ddv_file)
    elif if_vnir:
        pass
    elif if_swir:
        pass

    # Clear data
    del ndvi, red_refl, nir_refl, rdn_header

def interp_atm_lut(atm_lut_RHO, atm_lut_WVC, atm_lut_VZA, atm_lut_RAA, atm_lut, rho, wvc, vza, raa):
    """ Interpolate the atmospheric lookup table for visibility estimation.
    Arguments:
        atm_lut_RHO, atm_lut_WVC, atm_lut_VZA, atm_lut_RAA: list of floats
            Atmospheric lookup table grids.
        atm_lut: ndarray
            Atmospheric lookup table, shape=(RHO, WVC, VIS, VZA, RAA).
    Returns:
        interp_rdn: 1D array
            Interpolated radiance.
    """

    from AtmLUT import get_interp_range, combos

    # Get water vapor column intepolation range.
    rho_dict = get_interp_range(atm_lut_RHO, rho)
    wvc_dict = get_interp_range(atm_lut_WVC, wvc)
    vza_dict = get_interp_range(atm_lut_VZA, vza)
    raa_dict = get_interp_range(atm_lut_RAA, raa)

    # Update interpolated radiance.
    interp_rdn = np.zeros(atm_lut.shape[2])
    index_combos = combos([list(rho_dict.keys()), list(wvc_dict.keys()), list(vza_dict.keys()), list(raa_dict.keys())])
    for index_combo in index_combos:
        rho_index, wvc_index, vza_index, raa_index = index_combo
        interp_rdn += atm_lut[rho_index, wvc_index,:,vza_index,raa_index]*rho_dict[rho_index]*wvc_dict[wvc_index]*vza_dict[vza_index]*raa_dict[raa_index]

    return interp_rdn
