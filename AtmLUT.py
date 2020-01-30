#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to do atmospheric corrections.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)

# Define atmosphere database parameters. Do not make any change to them.
atm_db_VZA_grid_size = 5 # view zenith angle grid size, in [deg]
atm_db_RAA_grid_size = 15 # relative azimuth angle grid size, in [deg]
atm_db_RHO =  np.array([0, 0.5, 1.0]) # surface albedo, unitless
atm_db_WVC =  np.array([4, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # water vapor column, in [mm]
atm_db_VIS =  np.array([5, 10, 20, 40, 80, 120]) # aerosol visibility, in [km]
atm_db_WAVE =  np.arange(4000, 25001)/10 # wavelength, in [nm]
atm_db_SZA = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]) # sun zenith angle, in [deg]
atm_db_VZA = np.array([0, 5, 10, 15, 20, 25, 30, 40]) # view zenith angle, in [deg]
atm_db_RAA = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]) # relative azimuth angle, in [deg]

def build_atm_lut(flight_dict):
    """ Make an atmospheric lookup table.
    Arguments:
        flight_dict: dict
            Flight configurations.
    """

    from DEM  import get_avg_elev
    import glob

    if os.path.exists(flight_dict['raw_atm_lut_file']):
        logger.info('Write the raw ALT to %s.' %flight_dict['raw_atm_lut_file'])
        return

    # Get sun angles (zenith and azimuth), above-sea-level elevation, above-ground flight altitude.
    sensor_dict = flight_dict['sensors'][list(flight_dict['sensors'].keys())[0]]
    atm_lut_sza, atm_lut_saa = tuple(flight_dict['sun_angles'])
    atm_lut_elev = get_avg_elev(sensor_dict['dem_image_file'])/1000.0 # 1000.0 converts [m] to [km].
    imugps = np.loadtxt(sensor_dict['processed_imugps_file'])
    flight_altitude = imugps[:,3].mean()/1000.0 # 1000.0 converts [m] to [km]
    atm_lut_zout = flight_altitude - atm_lut_elev
    del imugps, flight_altitude, sensor_dict

    # Get the ELEV and ZOUT grids of the atmosphere database.
    atm_db_folders = glob.glob(os.path.join(flight_dict['atm_database_dir'], flight_dict['atm_mode'], 'ELEV_*ZOUT_*'))
    if atm_db_folders == []:
        raise IOError('Cannot find any atmospheric database file under %s.' %(os.path.join(flight_dict['atm_database_dir'], flight_dict['atm_mode'])))
    atm_db_ELEV = []
    atm_db_ZOUT = []
    for folder in atm_db_folders:
        tmp = os.path.basename(folder).split('_')
        atm_db_ELEV.append(int(tmp[1])/1000.0)
        atm_db_ZOUT.append(int(tmp[3])/1000.0)
    del atm_db_folders, folder
    atm_db_ELEV = sorted(list(set(atm_db_ELEV))) # Remove duplicates, and sort values;
    atm_db_ZOUT = sorted(list(set(atm_db_ZOUT))) # Remove duplicates, and sort values;
    
    # Get the elevation interpolation range.
    if np.all(np.array(atm_db_ELEV)<atm_lut_elev) or np.all(np.array(atm_db_ELEV)>atm_lut_elev):
        raise IOError('The above-sea-level elevation (%.2f) is out of range (%.2f - %.2f).' %(atm_lut_elev, min(atm_db_ELEV), max(atm_db_ELEV)))
    elev_dict = get_interp_range(atm_db_ELEV, atm_lut_elev)

    # Get the above-ground flight altitude interpolation range.
    if np.all(np.array(atm_db_ZOUT)<atm_lut_zout) or np.all(np.array(atm_db_ZOUT)>atm_lut_zout):
        raise IOError('The above-ground flight altitude (%.2f) is out of range (%.2f - %.2f).' %(atm_lut_zout, min(atm_db_ZOUT), max(atm_db_ZOUT)))
    zout_dict = get_interp_range(atm_db_ZOUT, atm_lut_zout)

    # Get the sun zenith angle interpolation range.
    if np.all(np.array(atm_db_SZA)<atm_lut_sza) or np.all(np.array(atm_db_SZA)>atm_lut_sza):
        raise IOError('The sun zenith angle (%.2f) is out of range (%.2f - %.2f).' %(atm_lut_sza, min(atm_db_SZA), max(atm_db_SZA)))
    sza_dict = get_interp_range(atm_db_SZA, atm_lut_sza)
    
    # Print out interpolation ranges.
    logger.info('Atmospheric database interpolation point and range:')
    index = list(elev_dict.keys())
    logger.info('ELEV [km] = %.2f, %.2f - %.2f' %(atm_lut_elev, atm_db_ELEV[index[0]], atm_db_ELEV[index[1]]))
    index = list(zout_dict.keys())
    logger.info('ZOUT [km] = %.2f, %.2f - %.2f' %(atm_lut_zout, atm_db_ZOUT[index[0]], atm_db_ZOUT[index[1]]))
    index = list(sza_dict.keys())
    logger.info('SZA [deg] = %.2f, %.2f - %.2f' %(atm_lut_sza, atm_db_SZA[index[0]], atm_db_SZA[index[1]]))
    
    # Initialize the atmosphere lookup table.
    atm_lut = np.memmap(flight_dict['raw_atm_lut_file'],
                        dtype='float32',
                        mode='w+',
                        shape=(len(atm_db_RHO),
                               len(atm_db_WVC),
                               len(atm_db_VIS),
                               len(atm_db_VZA),
                               len(atm_db_RAA),
                               len(atm_db_WAVE)))
    atm_lut[...] = 0.0

    # Do interpolations.
    index_combos = combos([list(elev_dict.keys()), list(zout_dict.keys()), list(sza_dict.keys())])
    for index_combo in index_combos:
        elev_index, zout_index, sza_index = index_combo
        # Get atmosphere database filename: atm_database/ELEV_xxxx_ZOUT_xxxx/ELEV_xxxx_ZOUT_xxxx_SZA_xxx.
        basename = 'ELEV_%04d_ZOUT_%04d' %(atm_db_ELEV[elev_index]*1000, atm_db_ZOUT[zout_index]*1000)
        atm_db_file = os.path.join(flight_dict['atm_database_dir'],
                                   flight_dict['atm_mode'],
                                   basename,
                                   '%s_SZA_%03d' %(basename, atm_db_SZA[sza_index]))

        # Read atmosphere database data.
        atm_db = np.memmap(atm_db_file,
                           dtype='float32',
                           mode='r',
                           shape=(len(atm_db_RHO),
                                  len(atm_db_WVC),
                                  len(atm_db_VIS),
                                  len(atm_db_VZA),
                                  len(atm_db_RAA),
                                  len(atm_db_WAVE)))

        # Update the atmosphere lookup table.
        atm_lut += atm_db*elev_dict[elev_index]*zout_dict[zout_index]*sza_dict[sza_index]

        # Clear database.
        atm_db.flush()
        del basename, atm_db_file, atm_db
    del index_combos

    # Divide radiance by the sun-earth-distance adjusting factor and 10 (to convert radiance to mW/(cm2*um*sr)).
    atm_lut /= ((flight_dict['sun_earth_distance']**2)*10.0)
    atm_lut.flush()
    del atm_lut
    
    # Write the metadata of the atmosphere lookup table.
    atm_lut_metadata = dict()
    atm_lut_metadata['description'] = 'Raw atmospheric lookup table radiance [mW/(cm2*um*sr)]'
    atm_lut_metadata['dtype'] = 'float32'
    atm_lut_metadata['shape'] = [len(atm_db_RHO), len(atm_db_WVC), len(atm_db_VIS), len(atm_db_VZA), len(atm_db_RAA), len(atm_db_WAVE)]
    atm_lut_metadata['dimension'] = ['RHO', 'WVC', 'VIS', 'VZA', 'RAA', 'WAVE']
    atm_lut_metadata['atm_mode'] = flight_dict['atm_mode']
    atm_lut_metadata['ELEV'] = atm_lut_elev
    atm_lut_metadata['ZOUT'] = atm_lut_zout
    atm_lut_metadata['SZA'] = atm_lut_sza
    atm_lut_metadata['SAA'] = atm_lut_saa
    atm_lut_metadata['RHO'] = list(atm_db_RHO)
    atm_lut_metadata['WVC'] = list(atm_db_WVC)
    atm_lut_metadata['VIS'] = list(atm_db_VIS)
    atm_lut_metadata['VZA'] = list(atm_db_VZA)
    atm_lut_metadata['RAA'] = list(atm_db_RAA)
    atm_lut_metadata['WAVE'] = list(atm_db_WAVE)
    write_binary_metadata(flight_dict['raw_atm_lut_file']+'.meta', atm_lut_metadata)

    logger.info('Write the raw ALT to %s.' %flight_dict['raw_atm_lut_file'])

def resample_atm_lut(resampled_atm_lut_file, raw_atm_lut_file, rdn_header_file):
    """ Resample atmosphere lookup table radiance to sensor wavelengths.
    Arguments:
        resampled_atm_lut_file: str
            Resampled atmosphere lookup table filename.
        raw_atm_lut_file: str
            Raw atmosphere lookup table filename.
        rdn_header_file: str
            Radiance header filename.
    """

    if os.path.exists(resampled_atm_lut_file):
        logger.info('Write the resampled ALT to %s.' %resampled_atm_lut_file)
        return

    from ENVI    import read_envi_header
    from Spectra import resample_spectra

    # Read atmosphere look-up table grids.
    atm_lut_metadata = read_binary_metadata(raw_atm_lut_file+'.meta')
    atm_lut_metadata['shape'] = tuple([int(v) for v in atm_lut_metadata['shape']])
    atm_lut_RHO = np.array([float(v) for v in atm_lut_metadata['RHO']])
    atm_lut_WVC = np.array([float(v) for v in atm_lut_metadata['WVC']])
    atm_lut_VIS = np.array([float(v) for v in atm_lut_metadata['VIS']])
    atm_lut_VZA = np.array([float(v) for v in atm_lut_metadata['VZA']])
    atm_lut_RAA = np.array([float(v) for v in atm_lut_metadata['RAA']])
    atm_lut_WAVE = np.array([float(v) for v in atm_lut_metadata['WAVE']])

    # Read atmosphere look-up table data.
    atm_lut = np.memmap(raw_atm_lut_file,
                        dtype=atm_lut_metadata['dtype'],
                        mode='r',
                        shape=atm_lut_metadata['shape'])# shape = (RHO, WVC, VIS, VZA, RAA, WAVE)

    # Reshape.
    atm_lut = atm_lut.reshape((len(atm_lut_RHO)*len(atm_lut_WVC)*len(atm_lut_VIS)*len(atm_lut_VZA)*len(atm_lut_RAA), len(atm_lut_WAVE)))

    # Read radiance header.
    rdn_header = read_envi_header(rdn_header_file)

    # Do spectral resampling.
    atm_lut = resample_spectra(atm_lut, atm_lut_WAVE, rdn_header['wavelength'], rdn_header['fwhm'])
    atm_lut = atm_lut.reshape((len(atm_lut_RHO),
                               len(atm_lut_WVC),
                               len(atm_lut_VIS),
                               len(atm_lut_VZA),
                               len(atm_lut_RAA),
                               len(rdn_header['wavelength'])))

    # Write data into the file.
    fid = open(resampled_atm_lut_file, 'wb')
    fid.write(atm_lut.astype('float32').tostring())
    fid.close()
    del atm_lut

    # Write metadata.
    atm_lut_metadata['shape'] = [len(atm_lut_RHO), len(atm_lut_WVC), len(atm_lut_VIS), len(atm_lut_VZA), len(atm_lut_RAA), len(rdn_header['wavelength'])]
    atm_lut_metadata['WAVE'] = rdn_header['wavelength']
    write_binary_metadata(resampled_atm_lut_file+'.meta', atm_lut_metadata)

    logger.info('Write the resampled ALT to %s.' %resampled_atm_lut_file)

def write_binary_metadata(metadata_file, metadata):
    """ Write the metadata of a binary file.
    Arguments:
        metadata_file: str
            Metadata filename.
        metadata: dict
            Metadata.
    """

    fid = open(metadata_file, 'w')
    for key in metadata.keys():
        if metadata[key] is None:
            continue
        if type(metadata[key]) is list:
            value = []
            for i, v in enumerate(metadata[key]):
                if (i+1)%5==0:
                    value.append(str(v)+'\n')
                else:
                    value.append(str(v))
            value = '{%s}' %(', '.join(value))
        else:
            value = str(metadata[key])
        fid.write('%s = %s\n' %(key, value))
    fid.close()

def read_binary_metadata(metadata_file):
    """ Read the metadata of a binary file.
    Arguments:
        metadata_file: str
            Metadata filename.
    Returns:
        metadata: dict
            Metadata.
    """

    fid = open(metadata_file, 'r')
    trans_tab = str.maketrans(dict.fromkeys('\n{}'))
    metadata = dict()
    while 1:
        line = fid.readline()
        if '=' in line:
            key, value = line.split('=', 1)
            if ('{' in value) and ('}' not in value):
                while '}' not in line:
                    line = fid.readline()
                    if line.strip()[0] == ';':
                        continue
                    value += line
            key = key.strip()
            if ('{' in value) and ('}' in value):
                value = value.translate(trans_tab).strip()
                value = list(map(str.strip, value.split(',')))
            else:
                value = value.translate(trans_tab).strip()
            metadata[key] = value
        if line == '':
            break
    fid.close()

    return metadata

def get_interp_range(xs, x):
    """ Get the interpolation range.
    """

    x_index0 = np.where(xs<=x)[0][-1]
    x_index1 = np.where(xs>x)[0][0]
    x_delta0 = (xs[x_index1]-x)/(xs[x_index1]-xs[x_index0])
    x_delta1 = (x-xs[x_index0])/(xs[x_index1]-xs[x_index0])

    return {x_index0: x_delta0, x_index1: x_delta1}

def combos(indices):
    """ Return all combinations of indices in a list of index sublists.
    Arguments:
        indices: list of int lists
            List of index lists.
    """

    import itertools

    return list(itertools.product(*indices))
