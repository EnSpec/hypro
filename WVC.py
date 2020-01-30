#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to estimate water vapor column.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)
solar_flux_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','solar_flux.dat')

def build_wvc_model(wvc_model_file, atm_lut_file, rdn_header_file, vis=40):
    """ Build water vapor models.
    Arguments:
        wvc_model_file: str
            Water vapor column model filename.
        atm_lut_file: str
            Atmosphere lookup table file.
        rdn_header_file: str
            Radiance header filename.
        vis: float
            Visibility in [km].
    """

    if os.path.exists(wvc_model_file):
        logger.info('Write WVC models to %s.' %wvc_model_file)
        return

    from AtmLUT  import read_binary_metadata, get_interp_range
    from ENVI    import read_envi_header
    from Spectra import get_closest_wave, resample_spectra
    import json

    # Get sensor wavelengths and fwhms.
    header = read_envi_header(rdn_header_file)
    sensor_waves = np.array(header['wavelength'])
    sensor_fwhms = np.array(header['fwhm'])

    # Read atmosphere look-up-table metadata.
    metadata = read_binary_metadata(atm_lut_file+'.meta')
    metadata['shape'] = tuple([int(v) for v in metadata['shape']])
    atm_lut_WVC = np.array([float(v) for v in metadata['WVC']])
    atm_lut_VIS = np.array([float(v) for v in metadata['VIS']])
    atm_lut_VZA = np.array([float(v) for v in metadata['VZA']])
    atm_lut_RAA = np.array([float(v) for v in metadata['RAA']])
    atm_lut_WAVE = np.array([float(v) for v in metadata['WAVE']])
    # vza index
    vza_index = np.where(atm_lut_VZA==min(atm_lut_VZA))[0][-1]
    # raa index
    raa_index = np.where(atm_lut_RAA==min(atm_lut_RAA))[0][-1]

    # Load atmosphere look-up-table.
    atm_lut = np.memmap(atm_lut_file,
                        dtype='float32',
                        mode='r',
                        shape=metadata['shape']) # dimension=[RHO, WVC, VIS, VZA, RAA, WAVE]

    # Subtract path radiance.
    atm_lut_rdn = atm_lut[1,:,vza_index,raa_index,:] - atm_lut[0,:,vza_index,raa_index,:] # dimesion=[WVC, VIS, WAVE]
    atm_lut.flush()
    del atm_lut
    
    # vis intepolation range
    vis_dict = get_interp_range(atm_lut_VIS, vis)

    # Do interpolation.
    interp_rdn = np.zeros((len(atm_lut_WVC), len(atm_lut_WAVE)))
    for vis_index, vis_delta in vis_dict.items():
        interp_rdn += atm_lut_rdn[:,vis_index,:]*vis_delta
    del atm_lut_rdn, vis_index, vis_delta

    # Build WVC models.
    model_waves = [(890, 940, 1000),
                   (1070, 1130, 1200)]
    wvc_model = dict()
    for waves in model_waves:
        # Get model wavelengths.
        wave_1, wave_2, wave_3 = waves
        left_wave, left_band = get_closest_wave(sensor_waves, wave_1)
        middle_wave, middle_band = get_closest_wave(sensor_waves, wave_2)
        right_wave, right_band = get_closest_wave(sensor_waves, wave_3)

        # Build the model.
        if np.abs(left_wave-wave_1)<20 and np.abs(middle_wave-wave_2)<20 and np.abs(right_wave-wave_3)<20:
            model = dict()

            # bands
            model['band'] = [int(left_band), int(middle_band), int(right_band)]

            # wavelengths
            model['wavelength'] = [left_wave, middle_wave, right_wave]

            # weights
            left_weight = (right_wave-middle_wave)/(right_wave-left_wave)
            right_weight = (middle_wave-left_wave)/(right_wave-left_wave)
            model['weight'] = [left_weight, right_weight]

            # ratios and wvcs
            resampled_rdn = resample_spectra(interp_rdn, atm_lut_WAVE,
                                             sensor_waves[model['band']],
                                             sensor_fwhms[model['band']])
            ratio = resampled_rdn[:,1]/(left_weight*resampled_rdn[:,0]+right_weight*resampled_rdn[:,2])
            index = np.argsort(ratio)
            model['ratio'] = list(ratio[index])
            model['wvc'] = list(atm_lut_WVC[index])

            # Save model parameters.
            wvc_model['WVC_Model_%d' %wave_2] = model

            # Clear data.
            del left_weight, right_weight, resampled_rdn, ratio, index, model
    del interp_rdn

    # Save WVC models to a json file.
    with open(wvc_model_file, 'w') as fid:
        json.dump(wvc_model, fid, indent=4)
    logger.info('Write WVC models to %s.' %wvc_model_file)

def estimate_wvc(wvc_file, rdn_file, sensors, sun_zenith, distance, background_mask_file):
    """ Estimate water vapor column.
    Arguments:
        wvc_file: str
            Water vapor column image filename.
        rdn_file: str
            Radiance image filename.
        sensors: dict
             Sensors.
        mask_file: str
            Mask image filename.
        sun_zenith: float
            Sun zenith angle.
        distance: float
            Earth-to-sun distance.
    """

    if os.path.exists(wvc_file):
        logger.info('Save the WVC image to %s.' %(wvc_file))
        return

    from ENVI    import read_envi_header, empty_envi_header, write_envi_header
    from Spectra import get_closest_wave, resample_solar_flux
    import json

    # Read radiance image.
    rdn_header = read_envi_header(rdn_file+'.hdr')
    rdn_image = np.memmap(rdn_file,
                          dtype='float32',
                          mode='r',
                          shape=(rdn_header['bands'],
                                 rdn_header['lines'],
                                 rdn_header['samples']))
    # Read boundary mask.
    bg_mask_header = read_envi_header(background_mask_file+'.hdr')
    bg_mask_image = np.memmap(background_mask_file,
                              mode='r',
                              dtype='bool',
                              shape=(bg_mask_header['lines'],
                                     bg_mask_header['samples']))

    # Mask out dark pixels.
    good_mask_image = np.full((rdn_header['lines'], rdn_header['samples']), True)
    solar_flux = resample_solar_flux(solar_flux_file, rdn_header['wavelength'], rdn_header['fwhm'])
    cos_sun_zenith = np.cos(np.deg2rad(sun_zenith))
    d2 = distance**2
    # If the reflectance at 850 nm is less than 0.01, then mask out these pixels.
    wave, band = get_closest_wave(rdn_header['wavelength'], 470)
    if abs(wave-470)<20:
        refl = rdn_image[band,:,:]*np.pi*d2/(solar_flux[band]*cos_sun_zenith)
        good_mask_image &= (refl>0.01)
        del refl
    # If the reflectance at 850 nm is less than 0.10, then mask out these pixels.
    wave, band = get_closest_wave(rdn_header['wavelength'], 850)
    if abs(wave-850)<20:
        refl = rdn_image[band,:,:]*np.pi*d2/(solar_flux[band]*cos_sun_zenith)
        good_mask_image &= (refl>0.10)
        del refl
    # If the reflectance at 1600 nm is less than 0.10, then mask out these pixels.
    wave, band = get_closest_wave(rdn_header['wavelength'], 1600)
    if abs(wave-1600)<20:
        refl = rdn_image[band,:,:]*np.pi*d2/(solar_flux[band]*cos_sun_zenith)
        good_mask_image &= (refl>0.10)
        del refl
    del wave, band, solar_flux, cos_sun_zenith, d2

    # Estimate water vapor columns.
    wvc_image = np.zeros((rdn_header['lines'],
                          rdn_header['samples']))
    for sensor_index, sensor_dict in sensors.items():
        # Read water vapor column models.
        wvc_model_file = sensor_dict['wvc_model_file']
        wvc_models = json.load(open(wvc_model_file, 'r'))

        # Estimate wvc.
        for wvc_model in wvc_models.values():
            # Find band indices.
            bands = []
            for wave in wvc_model['wavelength']:
                _, band = get_closest_wave(rdn_header['wavelength'], wave)
                bands.append(band)
                del band

            # Calculate ratios.
            ratio_image = rdn_image[bands[1],:,:]/(1e-10+rdn_image[bands[0],:,:]*wvc_model['weight'][0]+rdn_image[bands[2],:,:]*wvc_model['weight'][1])

            # Calculate water vapor columns.
            wvc_image[good_mask_image] += np.interp(ratio_image[good_mask_image], wvc_model['ratio'], wvc_model['wvc']).astype('float32')

        # Clear data
        del ratio_image, bands, wvc_model
    rdn_image.flush()
    del rdn_image
    
    # Average.
    wvc_image /= len(sensors)

    # Read solar flux.
    wvc_image[~good_mask_image] = wvc_image[good_mask_image].mean()
    logger.info('WVC [mm] statistics: min=%.2f, max=%.2f, avg=%.2f, sd=%.2f.' %(wvc_image[good_mask_image].min(), wvc_image[good_mask_image].max(), wvc_image[good_mask_image].mean(), wvc_image[good_mask_image].std()))
    wvc_image[bg_mask_image] = -1000.0
    del bg_mask_image, good_mask_image

    # Save water vapor column.
    fid = open(wvc_file, 'wb')
    fid.write(wvc_image.astype('float32').tostring())
    fid.close()

    # Write header.
    wvc_header = empty_envi_header()
    wvc_header['description'] = 'Water vapor column [mm]'
    wvc_header['file type'] = 'ENVI Standard'
    wvc_header['bands'] = 1
    wvc_header['lines'] = rdn_header['lines']
    wvc_header['samples'] = rdn_header['samples']
    wvc_header['file type'] = 'ENVI Standard'
    wvc_header['interleave'] = 'bsq'
    wvc_header['byte order'] = 0
    wvc_header['data type'] = 4
    wvc_header['data ignore value'] = -1000.0
    wvc_header['map info'] = rdn_header['map info']
    wvc_header['coordinate system string'] = rdn_header['coordinate system string']
    write_envi_header(wvc_file+'.hdr', wvc_header)
    del wvc_header, rdn_header

    logger.info('Save the WVC image to %s.' %(wvc_file))
