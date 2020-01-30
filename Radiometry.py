#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to do radiometric calibrations.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)

def make_radio_cali_file_Hyspex(radio_cali_file, dn_image_file, setting_file):
    """ Make a Hyspex radiometric calibration file.
    Arguments:
        radio_cali_file: str
            Hyspex radiometric calibration coefficiets filename.
        dn_image_file: str
            Hyspex digital number (DN) image filename.
        setting_file: str
            Hyspex radiometric calibration setting filename.
    """

    if os.path.exists(radio_cali_file):
        logger.info('Write the radiometric calibration coefficients to %s.' %radio_cali_file)
        return

    from ENVI     import empty_envi_header, write_envi_header
    from Spectra import estimate_fwhms_from_waves
    from scipy    import constants

    # Read meatadata from the raw Hyspex image.
    header = dict()
    try:
        fid = open(dn_image_file, 'rb')
    except:
        logger.error('Cannot read calibration data from %s.' %dn_image_file)
    header['word'] = np.fromfile(fid, dtype=np.int8, count=8)
    header['hdrSize'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['serialNumber'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['configFile'] = np.fromfile(fid, dtype=np.int8, count=200)
    header['settingFile'] = np.fromfile(fid, dtype=np.int8, count=120)

    header['scalingFactor'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    header['electronics'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['comsettingsElectronics'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['comportElectronics'] = np.fromfile(fid, dtype=np.int8, count=44)
    header['fanSpeed'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['backTemperature'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['Pback'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['Iback'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['Dback'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['comport'] = np.fromfile(fid, dtype=np.int8, count=64)

    header['detectstring'] = np.fromfile(fid, dtype=np.int8, count=200)
    header['sensor'] = np.fromfile(fid, dtype=np.int8, count=176)
    header['temperature_end'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    header['temperature_start'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    header['temperature_calibration'] = np.fromfile(fid, dtype=np.float64, count=1)[0]

    header['framegrabber'] = np.fromfile(fid, dtype=np.int8, count=200)
    header['ID'] = np.fromfile(fid, dtype=np.int8, count=200)
    header['supplier'] = np.fromfile(fid, dtype=np.int8, count=200)
    header['leftGain'] = np.fromfile(fid, dtype=np.int8, count=32)
    header['rightGain'] = np.fromfile(fid, dtype=np.int8, count=32)

    header['comment'] = np.fromfile(fid, dtype=np.int8, count=200)
    header['backgroundFile'] = np.fromfile(fid, dtype=np.int8, count=200)
    header['recordHD'] = np.fromfile(fid, dtype=np.int8, count=1)
    header['unknownPOINTER'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['serverIndex'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['comsettings'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['numberOfBackground'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['spectralSize'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['spatialSize'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['binning'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['detected'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['integrationTime'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['frameperiod'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['defaultR'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['defaultG'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['defaultB'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['bitshift'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['temperatureOffset'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['shutter'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['backgroundPresent'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['power'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['current'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['bias'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['bandwidth'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['vin'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['vref'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['sensorVin'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['sensorVref'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['coolingTemperature'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['windowStart'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['windowStop'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['readoutTime'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['p'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['i'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['d'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['numberOfFrames'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['nobp'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['dw'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['EQ'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['lens'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    header['FOVexp'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['scanningMode'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['calibAvailable'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['numberOfAvg'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['SF'] = np.fromfile(fid, dtype=np.float64, count=1)[0]

    header['apertureSize'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    header['pixelSizeX'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    header['pixelSizeY'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    header['temperature'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    header['maxFramerate'] = np.fromfile(fid, dtype=np.float64, count=1)[0]

    header['spectralCalibPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['REPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['QEPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['backgroundPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    header['badPixelsPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]

    header['imageFormat'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    header['spectralVector'] = np.fromfile(fid, dtype=np.float64, count=header['spectralSize'])
    header['QE'] = np.fromfile(fid, dtype=np.float64, count=header['spectralSize'])
    header['RE'] = np.fromfile(fid, dtype=np.float64, count=header['spectralSize']*header['spatialSize'])
    header['RE'].shape = (header['spectralSize'], header['spatialSize'])
    header['background'] = np.fromfile(fid, dtype=np.float64, count=header['spectralSize']*header['spatialSize'])
    header['background'].shape = (header['spectralSize'], header['spatialSize'])

    header['badPixels'] = np.fromfile(fid, dtype=np.uint32, count=header['nobp'])
    if header['serialNumber']>=3000 and header['serialNumber']<=5000:
        header['backgroundLast'] = np.fromfile(fid, dtype=np.float64, count=header['spectralSize']*header['spatialSize'])
        header['backgroundLast'].shape = (header['spectralSize'], header['spatialSize'])

    fid.close()

    # Convert int8array to string.
    def from_int8array_to_string(int8_array):
        string = ''
        for int8_value in int8_array:
            if int8_value == 0:
                break
            string += chr(int8_value)
        return string
    for key in header:
        if header[key].dtype.name == 'int8':
            header[key] = from_int8array_to_string(header[key])

    # Replace QE, wavelength and RE values if the setting file exists.
    if setting_file is not None:
        setting = get_hyspex_setting(setting_file)
        header['QE'] = np.array(setting['QE'])
        header['spectralVector'] = np.array(setting['spectral_calib'])
        header['RE'] = np.array(setting['RE']).reshape((setting['spectral_size'], setting['spatial_size']))

    # Calculate other values.
    header['spectralSampling'] = np.diff(header['spectralVector'])
    header['spectralSampling'] = np.append(header['spectralSampling'], header['spectralSampling'][-1])
    header['solidAngle'] = header['pixelSizeX']*header['pixelSizeY']
    header['satValue'] = np.power(2, 16.0-header['bitshift']) - 1
    header['fwhms'] = estimate_fwhms_from_waves(header['spectralVector'])

    # Calculate gain and offset coefficents.
    fid = open(radio_cali_file, 'wb')

    # gain
    h = constants.Planck # Planck constant
    c = constants.c*1e+9 # Light speed in nm/s
    SF = header['SF'] # float
    RE = header['RE'] # shape=(bands, samples)
    QE = np.expand_dims(header['QE'], axis=1) # shape=(bands, 1)
    center_wavelength = np.expand_dims(header['spectralVector'], axis=1) # shape=(bands, 1)
    wavelength_interval = np.expand_dims(header['spectralSampling'], axis=1) # shape=(bands, 1)
    integration_time = header['integrationTime'] # float
    aperture_area = np.pi*header['apertureSize']*header['apertureSize'] # float
    solid_angle = header['solidAngle']# float
    gain = h*c*1e6/(RE*QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)*100.0 # shape=(bands, samples); 100.0 to convert radiance to mW/(cm2*um*sr).
    fid.write(gain.astype('float64').tostring())
    del h, c, SF, RE, QE, center_wavelength, wavelength_interval, integration_time, aperture_area, solid_angle, gain

    # offset
    fid.write(header['background'].tostring())
    if header['serialNumber']>=3000 and header['serialNumber']<=5000:
        fid.write(header['backgroundLast'].tostring())
        bands = 3
    else:
        bands = 2
    fid.close()

    # Write header.
    radio_cali_header = empty_envi_header()
    radio_cali_header = empty_envi_header()
    radio_cali_header['description'] = 'Hyspex radiometric calibration coefficients.'
    radio_cali_header['file type'] = 'ENVI Standard'
    radio_cali_header['bands'] = bands
    radio_cali_header['lines'] = header['spectralSize']
    radio_cali_header['samples'] = header['spatialSize']
    radio_cali_header['interleave'] = 'bsq'
    radio_cali_header['byte order'] = 0
    radio_cali_header['data type'] = 5
    radio_cali_header['band names'] = ['gain', 'background'] if bands==2 else ['gain', 'background', 'backgroundLast']
    radio_cali_header['waves'] = list(header['spectralVector'])
    radio_cali_header['fwhms'] = list(header['fwhms'])
    write_envi_header(os.path.splitext(radio_cali_file)[0]+'.hdr', radio_cali_header)
    del radio_cali_header, header

    logger.info('Write the radiometric calibration coefficients to %s.' %radio_cali_file)

def dn2rdn_Hyspex(rdn_image_file, dn_image_file, radio_cali_file, acquisition_time):
    """ Do Hyspex radiometric calibration.
    Arguments:
        rdn_image_file: str
            Radiance image filename.
        dn_image_file: str
            Hyspex DN image filename.
        radio_cali_file: str
            Hyspex radiometric calibration coefficients filename.
        acquisition_time: datetime object
            Acquisition time.
    """
    
    if os.path.exists(rdn_image_file):
        logger.info('Write the radiance image to %s.' %rdn_image_file)
        return

    from ENVI  import empty_envi_header, read_envi_header, write_envi_header

    # Read calibration coefficients.
    radio_cali_header = read_envi_header(os.path.splitext(radio_cali_file)[0]+'.hdr')
    radio_cali_coeff = np.memmap(radio_cali_file,
                                dtype='float64',
                                mode='r',
                                shape=(radio_cali_header['bands'],
                                       radio_cali_header['lines'],
                                       radio_cali_header['samples']))
    wavelengths = np.array([float(v) for v in radio_cali_header['waves'].split(',')])
    fwhms = np.array([float(v) for v in radio_cali_header['fwhms'].split(',')])

    # Read DN image.
    dn_header = read_envi_header(os.path.splitext(dn_image_file)[0]+'.hdr')
    dn_image = np.memmap(dn_image_file,
                         dtype='uint16',
                         mode='r',
                         offset=dn_header['header offset'],
                         shape=(dn_header['lines'],
                                dn_header['bands'],
                                dn_header['samples']))

    # Get gain coefficients.
    gain = radio_cali_coeff[0,:,:] # shape=(bands, samples)

    # Do radiometric calibration.
    info = 'Line (max=%d): ' %dn_header['lines']
    fid = open(rdn_image_file, 'wb')
    for from_line in range(0, dn_header['lines'], 500):
        info += '%d, ' %(from_line+1)

        # Determine chunck size.
        to_line = min(from_line+500, dn_header['lines'])

        # Get offset coefficients.
        if radio_cali_header['bands']==2:
            offset = radio_cali_coeff[1,:,:] # shape=(bands, samples)
        else:
            background = np.stack([radio_cali_coeff[1,:,:]]*(to_line-from_line))
            backgroundLast = np.stack([radio_cali_coeff[2,:,:]]*(to_line-from_line))
            factor = np.arange(from_line, to_line)/dn_header['lines']
            offset = background+(backgroundLast-background)*factor[:,np.newaxis, np.newaxis] # shape=(to_line-from_line, bands, samples)
            del background, backgroundLast, factor

        # Convert DN to radiance.
        rdn = (dn_image[from_line:to_line,:,:].astype('float32')-offset)*gain # shape=(to_line-from_line, bands, samples)

        # Write radiance to the file.
        fid.write(rdn.astype('float32').tostring())

        # Clear temporary data.
        del rdn, to_line, offset
    fid.close()
    info += '%d, Done!' %dn_header['lines']
    logger.info(info)

    # Clear data.
    dn_image.flush()
    radio_cali_coeff.flush()
    del gain, from_line
    del dn_image, radio_cali_coeff
    
    # Write header.
    rdn_header = empty_envi_header()
    rdn_header['description'] = 'Hyspex radiance in mW/(cm2*um*sr)'
    rdn_header['file type'] = 'ENVI Standard'
    rdn_header['samples'] = dn_header['samples']
    rdn_header['lines'] = dn_header['lines']
    rdn_header['bands'] = dn_header['bands']
    rdn_header['byte order'] = 0
    rdn_header['header offset'] = 0
    rdn_header['interleave'] = 'bil'
    rdn_header['data type'] = 4
    rdn_header['wavelength'] = list(wavelengths)
    rdn_header['fwhm'] = list(fwhms)
    rdn_header['wavelength units'] = 'nm'
    rdn_header['default bands'] = dn_header['default bands']
    rdn_header['acquisition time'] = acquisition_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
    write_envi_header(os.path.splitext(rdn_image_file)[0]+'.hdr', rdn_header)
    del radio_cali_header, dn_header, rdn_header

    logger.info('Write the radiance image to %s.' %rdn_image_file)

def resample_rdn(resampled_rdn_image_file, raw_rdn_image_file, smile_effect_file):
    """ Resample radiance spectra.
    Arguments:
        resampled_rdn_image_file: str
            Resampled radiance image filename.
        raw_rdn_image_file: str
            Raw radiance image filename.
        smile_effect_file: str
            Smile effect filename.
    """

    if os.path.exists(resampled_rdn_image_file):
        logger.info('Write the spectrally resampled radiance image to %s.' %resampled_rdn_image_file)
        return

    from ENVI  import read_envi_header, write_envi_header
    from scipy import interpolate

    # Read the old radiance image.
    raw_rdn_header = read_envi_header(os.path.splitext(raw_rdn_image_file)[0]+'.hdr')
    raw_rdn_image = np.memmap(raw_rdn_image_file,
                              dtype='float32',
                              mode='r',
                              shape=(raw_rdn_header['lines'],
                                     raw_rdn_header['bands'],
                                     raw_rdn_header['samples']))

    # Read spectral center wavelengths and bandwidths.
    smile_effect_header = read_envi_header(os.path.splitext(smile_effect_file)[0]+'.hdr')
    smile_effect_data = np.memmap(smile_effect_file,
                                  dtype='float32',
                                  mode='r',
                                  shape=(smile_effect_header['bands'],
                                         smile_effect_header['lines'],
                                         smile_effect_header['samples']))

    # Do spectral interpolation.
    info = 'Line (max=%d): ' %smile_effect_header['lines']
    fid = open(resampled_rdn_image_file, 'wb')
    for from_line in range(0, raw_rdn_header['lines'], 500):
        info += '%d, ' %(from_line+1)
        # Determine chunck size.
        to_line = min(from_line+500, raw_rdn_header['lines'])

        # Initialize.
        rdn = np.zeros((to_line-from_line,raw_rdn_header['bands'],raw_rdn_header['samples'])) # shape=(from_line:to_line, bands, samples)

        # Do spectral interpolation.
        for sample in range(raw_rdn_header['samples']):
            f = interpolate.interp1d(smile_effect_data[0,:,sample],
                                     raw_rdn_image[from_line:to_line,:,sample],
                                     kind='cubic',
                                     fill_value='extrapolate',
                                     axis=1)
            rdn[:,:,sample] = f(raw_rdn_header['wavelength'])
            del f

        # Write interpolated radiance into the file.
        fid.write(rdn.astype('float32').tostring())
        del rdn, to_line
    fid.close()
    info += '%d, Done!' %smile_effect_header['lines']
    logger.info(info)

    # Clear data.
    del smile_effect_data
    raw_rdn_image.flush()
    del raw_rdn_image
    
    # Write the header.
    write_envi_header(os.path.splitext(resampled_rdn_image_file)[0]+'.hdr', raw_rdn_header)
    del raw_rdn_header

    logger.info('Write the spectrally resampled radiance image to %s.' %resampled_rdn_image_file)

def get_hyspex_setting(setting_file):
    """ Read Hyspex setting data.
    Arguments:
        setting_file: str
            Hyspex setting filename.
    Returns:
        setting: dict
            Hyspex setting.
    """

    setting_value_type = {"serialnumber": "int",
                          "configfile": "str",
                          "serverindex": "int",
                          "RecordHD": "str",
                          "framegrabber": "str",
                          "comsettings_electronics": "int",
                          "comsettings": "int",
                          "electronics": "int",
                          "readout_time": "float",
                          "EQ": "float",
                          "ScanningMode": "int",
                          "CalibAvailable": "int",
                          "lens": "int",
                          "FOVexp": "int",
                          "number_of_background": "int",
                          "detectstring": "str",
                          "sensor": "str",
                          "ID": "str",
                          "supplier": "str",
                          "spectral_size": "int",
                          "spatial_size": "int",
                          "default_R": "int",
                          "default_G": "int",
                          "default_B": "int",
                          "bitshift": "int",
                          "binning": "int",
                          "window_start": "int",
                          "window_stop": "int",
                          "integrationtime": "int",
                          "frameperiod": "int",
                          "nobp": "int",
                          "dw": "int",
                          "shutter": "int",
                          "SF": "float",
                          "max_framerate": "float",
                          "aperture_size":"float",
                          "pixelsize_x": "float",
                          "pixelsize_y": "float",
                          "Temperature_Calibration": "float",
                          "AIM_gain": "int",
                          "AIM_midlevel": "int",
                          "vref": "int",
                          "vin": "int",
                          "sensor_vref": "int",
                          "sensor_vin": "int",
                          "spectral_calib": "list_float",
                          "RE": "list_float",
                          "QE": "list_float",
                          "bad_pixels": "list_int"}
    trans_table = str.maketrans("\n"," ")
    setting = dict()
    fid = open(setting_file, 'r')
    line = fid.readline()
    flag = True
    while line:
        if "=" in line:
            key, value = line.split("=", 1)
            # Add field if not in the default list.
            if key.strip() not in setting_value_type.keys():
                setting_value_type[key.strip()] = 'str'
            # Keep reading if value is ''.
            if value.strip() == '':
                line = fid.readline()
                while (not "=" in line) and (not line.strip() is ''):
                    value += line
                    line = fid.readline()
                flag = False
            # Extract values.
            val_type = setting_value_type[key.strip()]
            if val_type == "list_float":
                tmp = value.translate(trans_table).strip().split(" ")
                value= np.array([float(x) for x in tmp])
            elif val_type == "list_int":
                tmp = value.translate(trans_table).strip()
                if tmp == '':
                    value = None
                else:
                    value= np.array([int(x) for x in tmp.split(" ")])
            elif val_type == "list_str":
                value= [x.strip() for x in value.translate(trans_table).strip().split(" ")]
            elif val_type == "int":
                value = int(value.translate(trans_table))
            elif val_type == "float":
                value = float(value.translate(trans_table))
            elif val_type == "str":
                value = value.translate(trans_table).strip().lower()
            setting[key.strip()] = value
        if flag is False:
            flag = True
        else:
            line = fid.readline()
    fid.close()
    return setting
