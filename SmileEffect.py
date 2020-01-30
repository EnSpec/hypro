#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to detect smile effects.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)

# Define atmosphere absorption features.
absorption_features = {429:  [424, 437],
                       486:  [482, 498],
                       517:  [510, 523],
                       586:  [584, 597],
                       686:  [683, 697],
                       762:  [753, 769],
                       820:  [807, 827],
                       940:  [922, 970],
                       1130: [1100, 1160],
                       1268: [1258, 1280],
                       1470: [1450, 1490],
                       1572: [1563, 1585],
                       2004: [1994, 2030],
                       2055: [2040, 2080],
                       2317: [2300, 2330],
                       2420: [2400, 2435]
                       }

def detect_smile_effect(sensor_dict, atm_lut_file):
    """ Detect smile effect.
    Arguments:
        sensor_dict: dict
            Sensor configurations.
        atm_lut_file: str
            Raw atmosphere lookup table filename.
    """

    if os.path.exists(sensor_dict['smile_effect_at_atm_features_file']) and os.path.exists(sensor_dict['smile_effect_file']):
        logger.info('Write smile effect at atmosphere aborption features to %s.' %sensor_dict['smile_effect_at_atm_features_file'])
        logger.info('Write smile effect to %s.' %sensor_dict['smile_effect_file'])
        return

    from ENVI    import empty_envi_header, read_envi_header, write_envi_header
    from Spectra import get_closest_wave
    from scipy   import optimize, interpolate
    import json

    # Read averaged radiance.
    header = read_envi_header(sensor_dict['avg_rdn_file']+'.hdr')
    sensor_rdn = np.memmap(sensor_dict['avg_rdn_file'],
                           mode='r',
                           dtype='float32',
                           shape=(header['lines'],
                                  header['samples'])) # shape=(bands, samples)
    samples = header['samples']
    sensor_wave = np.array([float(v) for v in header['waves'].split(',')])
    sensor_fwhm = np.array([float(v) for v in header['fwhms'].split(',')])

    tmp_vza = header['VZA'].split(',')
    tmp_raa = header['RAA'].split(',')
    vza = []
    raa = []
    for i in range(samples):
        try:
            vza.append(float(tmp_vza[i]))
            raa.append(float(tmp_raa[i]))
        except:
            vza.append(np.nan)
            raa.append(np.nan)
            logger.info('No spectrum for column: %s.' %(i+1))
    del tmp_vza, tmp_raa
    del header
    vza = np.array(vza)
    raa = np.array(raa)
    if np.all(np.isnan(vza)) or np.all(np.isnan(raa)):
        raise IOError("Cannot detect smile effects since all columns do not have spectra.")

    nonnan_index = np.where((~np.isnan(vza))&(~np.isnan(vza)))[0]
    vza = np.interp(np.arange(samples), nonnan_index, vza[nonnan_index])
    raa = np.interp(np.arange(samples), nonnan_index, raa[nonnan_index])

    #import matplotlib.pyplot as plt
    #plt.plot(vza, raa, '.-')
    #plt.show()
    #raise IOError('xxx')

    # Assign visibility.
    vis = [40]*samples

    # Estimate water vapor column.
    logger.info('Estimate WVC from averaged radiance spectra.')
    wvc_models = json.load(open(sensor_dict['wvc_model_file'], 'r'))
    wvc = np.zeros(samples)
    for model in wvc_models.values():
        ratio = sensor_rdn[model['band'][1], :]/(sensor_rdn[model['band'][0], :]*model['weight'][0]+sensor_rdn[model['band'][2], :]*model['weight'][1])
        wvc += np.interp(ratio, model['ratio'], model['wvc'])
        del ratio
    wvc /= len(wvc_models)
    wvc_isnan = np.isnan(wvc) # Look for NaN values
    if np.any(wvc_isnan):
        logger.info('WVC could not be estimated for some columns. '
                    'Missing values will be interpolated.')
        interpolate_values(wvc, wvc_isnan) # Replace NaNs with interpolated values
    logger.info('WVC [mm] statistics: min=%.2f, max=%.2f, avg=%.2f, sd=%.2f.' %(wvc.min(), wvc.max(), wvc.mean(), wvc.std()))
    del wvc_models, model
    
    # Interpolate atmospheric look-up table.
    logger.info('Interpolate atmospheric look-up table.')
    atm_lut_wave, atm_lut_rdn = interp_atm_lut(atm_lut_file, wvc, vis, vza, raa) # shape=(samples, bands)
    del wvc, vis, vza, raa

    # Detect smile effects at atmosphere absorption features.
    logger.info('Detect smile effects at atmosphere absorption features.')
    shifts = []
    band_indices = []
    n_features = 0
    for wave, wave_range in absorption_features.items():
        # Get sensor band range.
        sensor_wave0, sensor_band0 = get_closest_wave(sensor_wave, wave_range[0])
        sensor_wave1, sensor_band1 = get_closest_wave(sensor_wave, wave_range[1])
        center_wave, band_index = get_closest_wave(sensor_wave, wave)

        # Check if continue.
        if abs(sensor_wave0-wave_range[0])>20 or abs(sensor_wave1-wave_range[1])>20:
            continue

        # Get LUT band range.
        _, atm_lut_band0 = get_closest_wave(atm_lut_wave, sensor_wave0-20)
        _, atm_lut_band1 = get_closest_wave(atm_lut_wave, sensor_wave1+20)

        # Optimize.
        logger.info('Absorption feature center wavelength and range [nm] = %d: %d-%d.' %(wave, wave_range[0], wave_range[1]))
        x = []
        for sample in range(samples):
            p = optimize.minimize(cost_fun, [0, 0], method='BFGS',
                                  args=(sensor_wave[sensor_band0:sensor_band1+1],
                                        sensor_fwhm[sensor_band0:sensor_band1+1],
                                        sensor_rdn[sensor_band0:sensor_band1+1, sample],
                                        atm_lut_wave[atm_lut_band0:atm_lut_band1],
                                        atm_lut_rdn[sample, atm_lut_band0:atm_lut_band1]))
            x.append(p.x)
        x = np.array(x)

        # Do linear interpolation for invalid values.
        tmp_index  = ~(np.abs(x[:,0])>10.0)
        x[:,0] = np.interp(np.arange(samples), np.arange(samples)[tmp_index], x[tmp_index,0])
        x[:,1] = np.interp(np.arange(samples), np.arange(samples)[tmp_index], x[tmp_index,1])
        del tmp_index

        # Append shifts.
        shifts.append(x)
        band_indices.append(band_index)
        n_features += 1

    # Clear data.
    del p, x, atm_lut_rdn
    sensor_rdn.flush()
    del sensor_rdn
    
    # Reshape data.
    shifts = np.dstack(shifts).astype('float32').swapaxes(0,1).swapaxes(1,2)

    # Write shifts to a binary file.
    fid = open(sensor_dict['smile_effect_at_atm_features_file'], 'wb')
    fid.write(shifts[0,:,:].tostring()) # shift in wavelength
    fid.write(shifts[1,:,:].tostring()) # shift in fwhm
    fid.close()

    # Write header.
    header = empty_envi_header()
    header['description'] = 'Smile effects detected at atmosphere absorption features'
    header['file type'] = 'ENVI Standard'
    header['bands'] = 2
    header['lines'] = n_features
    header['samples'] = samples
    header['interleave'] = 'bsq'
    header['byte order'] = 0
    header['data type'] = 4
    header['spectral center wavelengths'] = list(sensor_wave[band_indices])
    header['spectral bandwiths'] = list(sensor_fwhm[band_indices])
    header['band indices'] = band_indices
    write_envi_header(sensor_dict['smile_effect_at_atm_features_file']+'.hdr', header)
    del header
    logger.info('Write smile effect at atmosphere absorption features to %s.' %sensor_dict['smile_effect_at_atm_features_file'])

    # Do interpolations.
    fid = open(sensor_dict['smile_effect_file'], 'wb')
    x = np.arange(samples)
    y = sensor_wave[band_indices]
    x_new = x
    y_new = sensor_wave

    # center wavelength
    z = np.zeros((shifts.shape[1], shifts.shape[2]), dtype='float64')
    for feature in range(shifts.shape[1]):
        p = np.poly1d(np.polyfit(x, shifts[0,feature,:], 4))
        z[feature, :] = p(x)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    z_new = f(x_new, y_new)+np.expand_dims(sensor_wave, axis=1)
    fid.write(z_new.astype('float32').tostring())

    # bandwidth
    z = np.zeros((shifts.shape[1], shifts.shape[2]), dtype='float64')
    for feature in range(shifts.shape[1]):
        p = np.poly1d(np.polyfit(x, shifts[1,feature,:], 5))
        z[feature, :] = p(x)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    z_new = f(x_new, y_new)+np.expand_dims(sensor_fwhm, axis=1)
    fid.write(z_new.astype('float32').tostring())
    fid.close()
    del f, x, y, z, z_new

    # Write header.
    header = empty_envi_header()
    header['description'] = 'Spectral Center Wavelengths and Spectral Bandwidths'
    header['file type'] = 'ENVI Standard'
    header['bands'] = 2
    header['lines'] = len(y_new)
    header['samples'] = len(x_new)
    header['interleave'] = 'bsq'
    header['byte order'] = 0
    header['data type'] = 4
    write_envi_header(sensor_dict['smile_effect_file']+'.hdr', header)
    del header, x_new, y_new

    logger.info('Write smile effect to %s.' %sensor_dict['smile_effect_file'])

def interp_atm_lut(atm_lut_file, WVC, VIS, VZA, RAA):
    """ Interpolate atmosphere look-up-table to different water vapor columns (WVC),
        visibilities (VIS), view zenith angles (VZA) and relative azimuth angles (RAA).
    Arguments:
        atm_lut_file: str
            Atmosphere look-up-table filename.
        WVC, VIS, VZA, RAA: list of floats
            Water vapor column, visibility, view zenith angles and relative azimuth angles.
    Returns:
        WAVE: array
            Wavelengths of the atmosphere look-up-table radiance.
        lut_rdn: 2D array
            Interpolated path radiance (albedo=0.0, 0.5, 1.0).
    """

    from AtmLUT import read_binary_metadata, get_interp_range, combos

    # Read atmosphere look-up table grids.
    atm_lut_metadata = read_binary_metadata(atm_lut_file+'.meta')
    atm_lut_metadata['shape'] = tuple([int(v) for v in atm_lut_metadata['shape']])
    atm_lut_WVC = np.array([float(v) for v in atm_lut_metadata['WVC']])
    atm_lut_VIS = np.array([float(v) for v in atm_lut_metadata['VIS']])
    atm_lut_VZA = np.array([float(v) for v in atm_lut_metadata['VZA']])
    atm_lut_RAA = np.array([float(v) for v in atm_lut_metadata['RAA']])
    atm_lut_WAVE = np.array([float(v) for v in atm_lut_metadata['WAVE']])

    # Read atmosphere look-up table data.
    atm_lut = np.memmap(atm_lut_file,
                        dtype=atm_lut_metadata['dtype'],
                        mode='r',
                        shape=atm_lut_metadata['shape'])# shape = (RHO, WVC, VIS, VZA, RAA, WAVE)

    # Initialize interpolated radiance.
    interp_rdn = np.zeros((len(WVC), len(atm_lut_WAVE)), dtype='float32')

    # Do interpolations.
    for i in range(len(WVC)):
        wvc, vis, vza, raa = WVC[i], VIS[i], VZA[i], RAA[i]

        # Get interpolation ranges.
        wvc_dict = get_interp_range(atm_lut_WVC, wvc)
        vis_dict = get_interp_range(atm_lut_VIS, vis)
        vza_dict = get_interp_range(atm_lut_VZA, vza)
        raa_dict = get_interp_range(atm_lut_RAA, raa)

        # Get combos.
        index_combos = combos([list(wvc_dict.keys()), list(vis_dict.keys()), list(vza_dict.keys()), list(raa_dict.keys())])

        # Update interpolated radiance.
        for index_combo in index_combos:
            wvc_index, vis_index, vza_index, raa_index = index_combo
            interp_rdn[i,:] += atm_lut[1,wvc_index,vis_index,vza_index,raa_index,:]*wvc_dict[wvc_index]*vis_dict[vis_index]*vza_dict[vza_index]*raa_dict[raa_index]
        del index_combo, index_combos

    # Clear atmosphere look-up table.
    atm_lut.flush()
    del atm_lut
    
    return atm_lut_WAVE, interp_rdn

def average_rdn(avg_rdn_file, rdn_image_file, sca_image_file, pre_class_image_file):
    """ Average radiance along each column.
    Arguments:
        avg_rdn_file: str
            Average radiance data filename.
        rdn_image_file: 3D array
            Radiance image filename, in BIL format.
        sca_image_file: 3D array
            Scan angle image filename, in BSQ format.
        pre_class_image_file: str
            Pre-classification image filename.
    """

    if os.path.exists(avg_rdn_file):
        logger.info('Write the averaged radiance data to %s.' %avg_rdn_file)
        return

    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    # Read radiance image data.
    rdn_header = read_envi_header(os.path.splitext(rdn_image_file)[0]+'.hdr')
    rdn_image = np.memmap(rdn_image_file,
                          dtype='float32',
                          mode='r',
                          shape=(rdn_header['lines'],
                                 rdn_header['bands'],
                                 rdn_header['samples']))

    # Read classification map.
    pre_class_header = read_envi_header(os.path.splitext(pre_class_image_file)[0]+'.hdr')
    pre_class_image = np.memmap(pre_class_image_file,
                                dtype='uint8',
                                mode='r',
                                shape=(pre_class_header['lines'],
                                       pre_class_header['samples']))
    mask = pre_class_image==5 # 5: land (clear)
    pre_class_image.flush()
    del pre_class_header, pre_class_image

    # Average radiance along each column.
    fid = open(avg_rdn_file, 'wb')
    info = 'Band (max=%d): ' %rdn_header['bands']
    for band in range(rdn_header['bands']):
        if band%20==0:
            info += '%d, ' %(band+1)

        # Build a temporary mask.
        tmp_mask = mask&(rdn_image[:,band,:]>0.0)

        # Average radiance.
        rdn = np.ma.array(rdn_image[:,band,:], mask=~tmp_mask).mean(axis=0) # shape = (samples,1)

        # Interpolate bad radiance values.
        for bad_sample in np.where(rdn.mask)[0]:
            if bad_sample==0:
                rdn[bad_sample] = rdn[bad_sample+1]
            elif bad_sample==rdn_header['samples']-1:
                rdn[bad_sample] = rdn[bad_sample-1]
            else:
                rdn[bad_sample] = (rdn[bad_sample-1]+rdn[bad_sample+1])/2.0

        # Write average radiance data into the file.
        fid.write(rdn.data.astype('float32'))

        # Clear data.
        del tmp_mask, rdn
    fid.close()
    info += str(rdn_header['bands'])
    logger.info(info)
    rdn_image.flush()
    del rdn_image
    
    # Read scan angles.
    sca_header = read_envi_header(os.path.splitext(sca_image_file)[0]+'.hdr')
    sca_image = np.memmap(sca_image_file,
                          dtype='float32',
                          mode='r',
                          shape=(sca_header['bands'],
                                 sca_header['lines'],
                                 sca_header['samples']))
    saa = float(sca_header['sun azimuth'])

    # Average scan angles.
    avg_vza = np.ma.array(sca_image[0,:,:], mask=~mask).mean(axis=0)
    avg_vaa = np.ma.array(sca_image[1,:,:], mask=~mask).mean(axis=0)
    avg_raa = saa-avg_vaa
    avg_raa[avg_raa<0] += 360.0
    avg_raa[avg_raa>180] = 360.0-avg_raa[avg_raa>180]
    sca_image.flush()
    del sca_image
    
    # Write header.
    avg_rdn_header = empty_envi_header()
    avg_rdn_header['description'] = 'Averaged radiance in [mW/(cm2*um*sr)]'
    avg_rdn_header['samples'] = rdn_header['samples']
    avg_rdn_header['lines'] = rdn_header['bands']
    avg_rdn_header['bands'] = 1
    avg_rdn_header['byte order'] = 0
    avg_rdn_header['header offset'] = 0
    avg_rdn_header['interleave'] = 'bsq'
    avg_rdn_header['data type'] = 4
    avg_rdn_header['waves'] = rdn_header['wavelength']
    avg_rdn_header['fwhms'] = rdn_header['fwhm']
    avg_rdn_header['VZA'] = list(avg_vza)
    avg_rdn_header['RAA'] = list(avg_raa)
    write_envi_header(avg_rdn_file+'.hdr', avg_rdn_header)

    logger.info('Write the averaged radiance data to %s.' %avg_rdn_file)

def interpolate_values(A, map):
    """ Replace array elements with interpolated values. Input array is modified in-place.
    Arguments:
        A: 1D array
		    Input array to be modified.
		map: 1D array
		    Boolean map indicating which elements should be replaced
    """
    
    def indices(x): return x.nonzero()[0]
    A[map] = np.interp(indices(map), indices(~map), A[~map])

def cost_fun(shifts, sensor_wave, sensor_fwhm, sensor_rdn, lut_wave, lut_rdn):
    """ Cost function.
    Arguments:
        shifts: list of float
            Shift in wavelength and FWHM.
        sensor_wave: 1D array
            Sensor wavelengths.
        sensor_fwhm: 1D array
            Sensor FWHMs.
        sensor_rdn: 1D array
            Sensor radiance.
        lut_wave: 1D array
            LUT wavelengths.
        lut_rdn: 1D array
            LUT at-sensor radiance.
    Returns:
        cost: float
            Squared error.
    """

    from Spectra import continuum_removal, resample_spectra

    # Apply shifts.
    sensor_wave = sensor_wave+shifts[0]
    sensor_fwhm = sensor_fwhm+shifts[1]

    # Resample LUT radiance to sensor wavelengths.
    lut_rdn = resample_spectra(lut_rdn, lut_wave, sensor_wave, sensor_fwhm)

    # Do continuum removal.
    sensor_rdn = continuum_removal(sensor_rdn, sensor_wave)
    lut_rdn = continuum_removal(lut_rdn, sensor_wave)

    # Calculate cost
    cost = np.sum((sensor_rdn-lut_rdn)**2)

    return cost
