#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""              _   _                                 ____
                | | | | _   _  ___  _ __    ___ __  __|  _ \  _ __   ___
                | |_| || | | |/ __|| '_ \  / _ \\ \/ /| |_) || '__| / _ \
                |  _  || |_| |\__ \| |_) ||  __/ >  < |  __/ | |   | (_) |
                |_| |_| \__, ||___/| .__/  \___|/_/\_\|_|    |_|    \___/
                        |___/      |_|
(1) HyspexPro aims to do geometric and radiometric corrections on Hyspex images.
    Our imaging system consists of two Hyspex cameras (VNIR-1800 and SWIR-384)and one
    GPS-Inertial Measurement Unit (IMU) sensor (iTraceRT-F400). The VNIR-1800 sensor has 182
    spectral bands within 400-1000 nm (spectral sampling = 3.26 nm). The SWIR-384 sensor has 288 bands
    within 960-2500 nm (spectral sampling = 5.45 nm). For more details about these cameras, please refer to
    https://www.hyspex.no/products/. The iTraceRT-F400 sensor records the attitudes (roll, pitch and heading angles)
    and positions (longitude, latitude and altitude) of airplanes.
    
(2) Geometric correction focuses on removing image geometric distortions mainly caused by
    platform motions, which can be characterized by roll, pitch and heading angles.
    We adapted the ray-tracing method proposed by Meyer P. (1994, Remote Sensing of Environment, vol: 49, 118-130)
    to calculate the map coordinates (easting, northing and elevation) of each raw image pixel (a process called geo-referencing).
    Then, the `warp` tool provided by GDAL (Geospatial Data Abstraction Library) was used to resample the raw image
    to regular map grids (a process called geo-rectification).
    
(3) Radiometric correction focuses on reducing radiometric distortions mainly caused by
    sensor smile effects (shifts in sensor center wavelengths and Full-Width-at-Half-Maximum (FWHM))
    and atmospheric effects (absorptions and scatterings). The spectral matching method proposed by
    Gao B. et al. (2004, Remote Sensing of Environment, vol: 90, 424-433) was adapted to detect
    sensor smile effects. Then, raw radiance spectra were resampled to common center wavelengths
    using a cubic interpolation. In the atmospheric correction step, two atmosphere parameters (water vapor column and visibility)
    were retrieved from image spectra. The water vapor column was estimated via the Continuum Interpolated Band Ratio (CIBR)
    technique proposed by Kaufman Y. et al. (1992, IEEE Transactions on Geoscience and Remote Sensing, vol 30, 871–884).
    The estimation of visibility was based on the Dense Dark Vegetation (DDV) method proposed by Kaufman Y. et al.
    (1997, IEEE Transactions on Geoscience and Remote Sensing, vol 35, 1286-1298). All radiometric correction steps
    require an atmosphere lookup table which consists of the at-senor radiance simulated under different atmospheric
    and sun-target-view geometric conditions. An open-source radiative transfer model, LibRadTran, was used
    for this purpose. For more details about LibRadTran, please refer to http://www.libradtran.org/doku.php.

(4) The input dataset of HyspexPro includes:
    1) Hyspex raw digital number (DN) images;
    2) Hyspex IMU&GPS data extracted from HyspexNAV;
    3) Hyspex sensor model files provided by NEO;
    4) DEM;
    5) Hyspex lab radiometric re-calibration data provided by NEO (optional);
    6) Ground control points data (optional).
    
    The output dataset of HyspexPro includes:
    1) Hyspex ground surface reflectance images;
    2) View angles (zenith and azimuth);
    3) DEM;

    Below is the standard workflow of HyspexPro:
                            Input dataset
                                  |
                                 \|/
                                  V
                            Prepare IMU&GPS
                                  |
                                 \|/
                                  V
                            Prepare DEM
                                  |
                                 \|/
                                  V               Yes
            If GCPs (Ground Control Points) exist ----> Do boresighting
                                  | No                         |
                                 \|/  /________________________|
                                  V   \
                      Calculate input geometry
                                  |
                                 \|/
                                  V
                        Calculate scan angle
                                  |
                                 \|/
                                  V
                     Build geographic lookup table
                                  |
                                 \|/
                                  V
                    Build atmospheric lookup table
                                  |
                                 \|/
                                  V
                     Do radiometric calibration
                                  |
                                 \|/
                                  V
                        Detect smile effects
                                  |
                                 \|/
                                  V
                      Resample radiance spectra
                                  |
                                 \|/
                                  V
                    Orthorectify radiance image
                                  |
                                 \|/
                                  V            Yes
                     If multiple sensors exist ----> Merge images
                                  | No                    |
                                 \|/  /___________________|
                                  V   \
                         Estimate visibility
                                  |
                                 \|/
                                  V
                     Estimate water vapor column
                                  |
                                 \|/
                                  V
                      Do atmospheric correction
                                  |
                                 \|/
                                  V
                           Output dataset

"""

import argparse, glob, json, logging, os, re

from Geography        import get_map_crs, get_sun_angles
from IMUGPS           import prepare_imugps_Hyspex
from SensorModel      import determine_if_rotated, make_sensor_model
from DEM              import prepare_dem
from Boresight        import boresight_calibration
from GeoReferencing   import calculate_igm, calculate_sca, build_glt
from Figure           import plot_image_area, plot_angle_geometry, make_quicklook, plot_avg_rdn, plot_wvc_model, plot_smile_effect
from AtmLUT           import build_atm_lut, resample_atm_lut
from Radiometry       import make_radio_cali_file_Hyspex, dn2rdn_Hyspex, resample_rdn
from Classification   import pre_classification
from SmileEffect      import average_rdn, detect_smile_effect
from WVC              import build_wvc_model, estimate_wvc
from GeoRectification import orthorectify_rdn, orthorectify_dem, orthorectify_sca
from ImageMerging     import merge_dem_sca, merge_rdn
from VIS              import estimate_vis
from AtmCorr          import atm_corr_image

def get_flight_indices(config):
    """ Get Hyspex flight indices.
    Arguments:
        config: dict
            Configurations.
    Returns:
        flight_indicies: list of strs
            Flight indices.
    """

    sensor_name = config['Sensors'][list(config['Sensors'].keys())[0]]['id']
    flight_indices = []
    dn_image_files = glob.glob(os.path.join(config['Data']['input_dir'], '*%s*.hyspex' %sensor_name))
    for dn_image_file in dn_image_files:
        basename = os.path.basename(dn_image_file)
        span = re.search('%s' %sensor_name, basename).span()
        flight_indices.append(basename[:span[0]-1])
        del basename, span
    del dn_image_files, sensor_name

    return flight_indices

def create_flight_log(output_dir, log_file_basename):
    """ Create a Hyspex flight processing log.
    Arguments:
        output_dir: str
            Output directory.
        log_file_basename: str
            Log file basename.
    Returns:
        flight_log: logging object
            Flight log.
    """

    log_file = os.path.join(output_dir, '%s.log' %log_file_basename)
    logging.basicConfig(filename=log_file,
                        level=logging.DEBUG,
                        format="%(asctime)s %(funcName)25s: %(message)s",
                        datefmt='%Y-%m-%dT%H:%M:%S',
                        filemode='w')
    flight_log = logging.getLogger()

    return flight_log

def initialize_flight_dict(config, flight_index):
    """ Initialize a Hyspex flight dictionary.
    Arguments:
        config: dict
            User-defined configurations.
        flight_index: str
            Flight index.
    Returns:
        flight_dict: dict
            Flight dictionary.
    """

    # flight dictionary
    flight_dict = dict()

    # flight output directory
    flight_dict['output_dir'] = os.path.join(config['Data']['output_dir'], flight_index)
    if not os.path.exists(flight_dict['output_dir']):
        os.mkdir(flight_dict['output_dir'])

    # flight atmospheric lookup table directory
    flight_dict['atm_dir'] = os.path.join(flight_dict['output_dir'], 'atm')
    if not os.path.exists(flight_dict['atm_dir']):
        os.mkdir(flight_dict['atm_dir'])

    # flight merged image directory.
    flight_dict['merge_dir'] = os.path.join(flight_dict['output_dir'], 'merge')
    if not os.path.exists(flight_dict['merge_dir']):
        os.mkdir(flight_dict['merge_dir'])

    # atmospheric correction parameters
    flight_dict['atm_database_dir'] = config['Atmospheric_Correction']['atm_database_dir']
    flight_dict['atm_mode'] = config['Atmospheric_Correction']['atm_mode']
    flight_dict['vis_retrieval'] = config['Atmospheric_Correction']['vis_retrieval']
    flight_dict['wvc_retrieval'] = config['Atmospheric_Correction']['wvc_retrieval']

    # raw dem
    flight_dict['dem'] = config['DEM']

    # boresight offsets
    flight_dict['boresight_options'] = config['Geometric_Correction']['boresight']['options']

    # sensor dictionary
    flight_dict['sensors'] = dict()
    for sensor_index in config['Sensors'].keys():
        # sensor parameters
        sensor_dict = config['Sensors'][sensor_index].copy()

        # digital number (DN) image
        dn_image_file = search_file(config['Data']['input_dir'], '%s_%s*.hyspex' %(flight_index, sensor_dict['id']))
        sensor_dict['dn_image_file'] = dn_image_file

        # radiometric calibration parameters
        sensor_dict['setting_file'] = config['Radiometric_Calibration']['setting_file'][sensor_index]

        # geometric correction parameters
        sensor_dict['pixel_size'] = config['Geometric_Correction']['pixel_size'][sensor_index]
        if config['Geometric_Correction']['boresight']['gcp_file'] is None or config['Geometric_Correction']['boresight']['gcp_file'][sensor_index] is None:
            sensor_dict['gcp_file'] = None
        else:
            if dn_image_file in config['Geometric_Correction']['boresight']['gcp_file'][sensor_index].keys():
                sensor_dict['gcp_file'] = config['Geometric_Correction']['boresight']['gcp_file'][sensor_index][dn_image_file]
            else:
                sensor_dict['gcp_file'] = None

        # boresight offsets
        sensor_dict['boresight_offsets'] = config['Geometric_Correction']['boresight']['offsets'][sensor_index]

        # sensor model
        sensor_dict['sensor_model_file'] = config['Geometric_Correction']['sensor_model_file'][sensor_index]

        # imugps
        raw_imugps_file = search_file(config['Data']['input_dir'], '%s_%s*.txt' %(flight_index, sensor_dict['id']))
        sensor_dict['raw_imugps_file'] = raw_imugps_file

        # output directory
        sensor_dict['output_dir'] = os.path.join(flight_dict['output_dir'], sensor_index)
        if not os.path.exists(sensor_dict['output_dir']):
            os.mkdir(sensor_dict['output_dir'])

        flight_dict['sensors'][sensor_index] = sensor_dict
        del sensor_dict, dn_image_file, raw_imugps_file

    return flight_dict

def search_file(in_dir, keyword):
    """ Search a specific file with the keyword.
    Arguments:
        in_dir: str
            Hyspex data input directory.
        keyword: str
            Searching keyword.
    Returns:
        file: str
            Found filename.
    """

    file = glob.glob(os.path.join(in_dir, keyword))

    if len(file) == 0:
        raise IOError('Cannot find any file in %s with the keyword: %s.' %(in_dir, keyword))
    elif len(file) > 1:
        raise IOError('Multiple files are found in %s with the keyword: %s.' %(in_dir, keyword))
    else:
        return file[0]

def get_center_lon_lat(raw_imugps_file):
    """ Get Hyspex image center longitude and latitude.
    Arguments:
        raw_imugps_file: str
            Hyspex raw imugps filename.
    Returns:
        [lon, lat]: list of floats
            Image center longitude and latitude.
    """

    import numpy as np

    imugps = np.loadtxt(raw_imugps_file)
    lon, lat = imugps[:,1].mean(), imugps[:,2].mean()
    del imugps

    return [lon, lat]

def get_acquisition_time(dn_header_file, raw_imugps_file):
    """ Get Hyspex image acquistion time.
    Notes:
        (1) This code is adapted from Brendan Heberlein (bheberlein@wisc.edu).
    Arguments:
        header_file: str
            Hyspex DN image header filename.
        imugps_file: str
            Hyspex raw imugps filename.
    Returns:
        when: datetime object
            Image acquisition time.
    """

    from datetime import datetime, timedelta
    from ENVI     import read_envi_header

    import numpy  as np

    header = read_envi_header(dn_header_file)
    week_start = datetime.strptime(f"{header['acquisition date']} 00:00:00", "%Y-%m-%d %H:%M:%S")
    week_seconds = np.loadtxt(raw_imugps_file)[:,7].mean()
    epoch = datetime(1980, 1, 6, 0, 0)
    gps_week = (week_start-epoch).days//7
    time_elapsed = timedelta(days=gps_week*7, seconds=week_seconds)
    when = epoch+time_elapsed

    return when

def get_sun_earth_distance(when):
    """ Get sun-earth distance of a day.
    Arguments:
        when: datetime object
            Date and time.
    Returns:
        d: float
            Sun-Earth distance.
    """

    import numpy as np
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    d = np.loadtxt(os.path.join(cur_dir, 'data/sun-earth-distance.dat'))
    doy = when.timetuple().tm_yday
    d = d[doy-1]
    
    return d

def HyspexPro(config_file):
    # Load configurations.
    config = json.load(open(config_file, 'r'))
    
    # Make an output directory.
    if not os.path.exists(config['Data']['output_dir']):
        os.mkdir(config['Data']['output_dir'])

    # Get flight indices.
    flight_indices = get_flight_indices(config)

    # Create a processing log file.
    flight_log = create_flight_log(config['Data']['output_dir'], os.path.basename(config['Data']['output_dir']))

    # Process each flight.
    for flight_index in flight_indices:
        print(flight_index)
        #----------------------------------------Part 0----------------------------------------#
        flight_log.info('%sFlight: %s%s' %('='*20, flight_index, '='*20))

        # Initialize the flight dictionary.
        flight_dict = initialize_flight_dict(config, flight_index)

        #----------------------------------------Part 1----------------------------------------#
        flight_log.info('%sPART 1: Extract flight information.' %('-'*10))

        # center longitude and latitude
        sensor_dict = flight_dict['sensors'][list(flight_dict['sensors'].keys())[0]]
        flight_dict['center_lon_lat'] = get_center_lon_lat(sensor_dict['raw_imugps_file'])
        flight_log.info('Image center longitude and latitude [deg]: %.6f, %.6f' %(flight_dict['center_lon_lat'][0], flight_dict['center_lon_lat'][1]))

        # image acquisition time
        flight_dict['acquisition_time'] = get_acquisition_time(os.path.splitext(sensor_dict['dn_image_file'])[0]+'.hdr', sensor_dict['raw_imugps_file'])
        flight_log.info('Image acquisition time: %s' %flight_dict['acquisition_time'])

        # sun-earth distance
        flight_dict['sun_earth_distance'] = get_sun_earth_distance(flight_dict['acquisition_time'])

        # map coordinate system
        flight_dict['map_crs'] = get_map_crs(flight_dict['dem'], flight_dict['center_lon_lat'][0], flight_dict['center_lon_lat'][1])
        flight_log.info('Map coordinate system: %s' %flight_dict['map_crs'].GetAttrValue('projcs'))

        # sun zenith and azimuth angles
        flight_dict['sun_angles'] = get_sun_angles(flight_dict['center_lon_lat'][0], flight_dict['center_lon_lat'][1], flight_dict['acquisition_time'])
        flight_log.info('Sun zenith and azimuth angle [deg]: %.2f, %.2f' %(flight_dict['sun_angles'][0], flight_dict['sun_angles'][1]))
        del sensor_dict

        #----------------------------------------Part 2----------------------------------------#
        flight_log.info('%sPart 2: Do geo-referencings.' %('-'*10))
        for sensor_index, sensor_dict in flight_dict['sensors'].items():
            flight_log.info('Sensor: %s' %sensor_index)

            # Initialize.
            basename = os.path.basename(sensor_dict['dn_image_file'][:-len('_raw.hyspex')])

            # Process IMUGPS.
            flight_log.info('Prepare the IMU (Inertial Measurement Unit) and GPS (Global Positioning System) data.')
            sensor_dict['processed_imugps_file'] = os.path.join(sensor_dict['output_dir'], basename+'_ProcessedIMUGPS.txt')
            prepare_imugps_Hyspex(sensor_dict['processed_imugps_file'],
                                  sensor_dict['raw_imugps_file'],
                                  sensor_dict['boresight_offsets'],
                                  flight_dict['map_crs'],
                                  flight_dict['boresight_options'])

            # Generate sensor model.
            if sensor_dict['sensor_model_file'] is None:
                flight_log.info('Generate sensor model.')
                sensor_dict['sensor_model_file'] = os.path.join(sensor_dict['output_dir'], basename+'_SensorModel.txt')
                if_rotated = determine_if_rotated(sensor_dict['processed_imugps_file'])
                if if_rotated:
                    flight_log.info('The sensor is 180 degree rotated.')
                make_sensor_model(sensor_dict['sensor_model_file'],
                                  sensor_dict['fov'],
                                  sensor_dict['ifov'][1],
                                  sensor_dict['samples'],
                                  if_rotated)

            # Process DEM.
            flight_log.info('Prepare the DEM (Digital Elevation Model) data.')
            sensor_dict['dem_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_DEM')
            prepare_dem(sensor_dict['dem_image_file'],
                        flight_dict['dem'],
                        sensor_dict['processed_imugps_file'],
                        sensor_dict['fov'],
                        flight_dict['map_crs'],
                        sensor_dict['pixel_size'])

            # Do boresighting if the gcp file is available.
            if sensor_dict['gcp_file'] is not None:
                flight_log.info('Do boresighting.')
                sensor_dict['boresight_file'] = os.path.join(sensor_dict['output_dir'], basename+'_Boresight')
                boresight_calibration(sensor_dict['boresight_file'],
                                      sensor_dict['gcp_file'],
                                      sensor_dict['processed_imugps_file'],
                                      sensor_dict['sensor_model_file'],
                                      sensor_dict['dem_image_file'],
                                      flight_dict['boresight_options'])

            # Build IGM.
            flight_log.info('Calculate the IGM (Input Geometry).')
            sensor_dict['igm_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_IGM')
            calculate_igm(sensor_dict['igm_image_file'],
                          sensor_dict['processed_imugps_file'],
                          sensor_dict['sensor_model_file'],
                          sensor_dict['dem_image_file'],
                          flight_dict['boresight_options'])

            # Create SCA.
            flight_log.info('Calculate the SCA (Scan Angle).')
            sensor_dict['raw_sca_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_RawSCA')
            calculate_sca(sensor_dict['raw_sca_image_file'],
                          sensor_dict['processed_imugps_file'],
                          sensor_dict['igm_image_file'],
                          flight_dict['sun_angles'])

            # Build GLT.
            flight_log.info('Build the GLT (Geographic Lookup Table).')
            sensor_dict['glt_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_GLT')
            build_glt(sensor_dict['glt_image_file'],
                      sensor_dict['igm_image_file'],
                      sensor_dict['pixel_size']/2.0,
                      flight_dict['map_crs'])

            # Plot image areas.
            flight_log.info('Plot the image area.')
            sensor_dict['image_area_figure_file'] = os.path.join(sensor_dict['output_dir'], basename+'_ImageArea.png')
            plot_image_area(sensor_dict['image_area_figure_file'],
                            sensor_dict['dem_image_file'],
                            sensor_dict['igm_image_file'],
                            sensor_dict['processed_imugps_file'])

            # Build angle geometries.
            flight_log.info('Plot the angle geometry.')
            sensor_dict['angle_geometry_figure_file'] = os.path.join(sensor_dict['output_dir'], basename+'_AngleGeometry.png')
            plot_angle_geometry(sensor_dict['angle_geometry_figure_file'],
                                sensor_dict['raw_sca_image_file'])

            del basename
        del sensor_index, sensor_dict

        #----------------------------------------Part 3----------------------------------------#
        flight_log.info('%sPart 3: Build an ALT (Atmospheric Lookup Table).' %('-'*10))
        flight_dict['raw_atm_lut_file'] = os.path.join(flight_dict['atm_dir'], '%s_RawALT' %flight_index)
        build_atm_lut(flight_dict)

        #----------------------------------------Part 4----------------------------------------#
        flight_log.info('%sPart 4: Do radiometric calibrations.' %('-'*10))
        for sensor_index, sensor_dict in flight_dict['sensors'].items():
            flight_log.info('Sensor: %s' %sensor_index)

            # Initialize.
            basename = os.path.basename(sensor_dict['dn_image_file'][:-len('_raw.hyspex')])

            # Get radiometric calibration coefficients.
            flight_log.info('Make a radiometric calibration coefficients file.')
            sensor_dict['radio_cali_coeff_file'] = os.path.join(sensor_dict['output_dir'], basename+'_RadioCaliCoeff')
            make_radio_cali_file_Hyspex(sensor_dict['radio_cali_coeff_file'],
                                        sensor_dict['dn_image_file'],
                                        sensor_dict['setting_file'])

            # Do radiometric calibration.
            flight_log.info('Convert DN (Digital Number) to radiance.')
            sensor_dict['raw_rdn_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_RawRdn')
            dn2rdn_Hyspex(sensor_dict['raw_rdn_image_file'],
                          sensor_dict['dn_image_file'],
                          sensor_dict['radio_cali_coeff_file'],
                          flight_dict['acquisition_time'])

            # Make a quicklook.
            flight_log.info('Make a quicklook.')
            sensor_dict['quicklook_figure_file'] = os.path.join(sensor_dict['output_dir'], basename+'_Quicklook.tif')
            make_quicklook(sensor_dict['quicklook_figure_file'],
                           sensor_dict['raw_rdn_image_file'],
                           sensor_dict['glt_image_file'])

            del basename
        del sensor_index, sensor_dict

        #----------------------------------------Part 5----------------------------------------#
        flight_log.info('%sPart 5: Detect smile effects.' %('-'*10))
        for sensor_index, sensor_dict in flight_dict['sensors'].items():
            flight_log.info('Sensor: %s' %sensor_index)

            # Initialize.
            basename = os.path.basename(sensor_dict['dn_image_file'][:-len('_raw.hyspex')])

            # Pre-classify the image.
            flight_log.info('Pre-classify the radiance image.')
            sensor_dict['pre_class_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_PreClass')
            pre_classification(sensor_dict['pre_class_image_file'],
                               sensor_dict['raw_rdn_image_file'],
                               flight_dict['sun_angles'][0],
                               flight_dict['sun_earth_distance'])

            # Average radiance along each column.
            flight_log.info('Average the radiance along image columns.')
            sensor_dict['avg_rdn_file'] = os.path.join(sensor_dict['output_dir'], basename+'_AvgRdn')
            average_rdn(sensor_dict['avg_rdn_file'],
                        sensor_dict['raw_rdn_image_file'],
                        sensor_dict['raw_sca_image_file'],
                        sensor_dict['pre_class_image_file'])

            # Plot average radiance spectra.
            flight_log.info('Plot the averaged radiance spectra.')
            sensor_dict['avg_rdn_figure_file'] = os.path.join(sensor_dict['output_dir'], basename+'_AvgRdn.png')
            plot_avg_rdn(sensor_dict['avg_rdn_figure_file'],
                         sensor_dict['avg_rdn_file'])

            # Build WVC model.
            flight_log.info('Build the WVC (Water Vapor Column) estimation model.')
            sensor_dict['wvc_model_file'] = os.path.join(sensor_dict['output_dir'], basename+'_WVCModel.json')
            build_wvc_model(sensor_dict['wvc_model_file'],
                            flight_dict['raw_atm_lut_file'],
                            os.path.splitext(sensor_dict['raw_rdn_image_file'])[0]+'.hdr')

            # Plot WVC model.
            flight_log.info('Plot the WVC model.')
            sensor_dict['wvc_model_figure_file'] = os.path.join(sensor_dict['output_dir'], basename+'_WVCModel.png')
            plot_wvc_model(sensor_dict['wvc_model_figure_file'],
                           sensor_dict['wvc_model_file'])

            # Detect smile effects.
            flight_log.info('Detect smile effects.')
            sensor_dict['smile_effect_at_atm_features_file'] = os.path.join(sensor_dict['output_dir'], basename+'_SmileEffectAtAtmFeatures')
            sensor_dict['smile_effect_file'] = os.path.join(sensor_dict['output_dir'], basename+'_SmileEffect')
            detect_smile_effect(sensor_dict,
                                flight_dict['raw_atm_lut_file'])

            # Plot smile effects at atmospheric absorption features.
            flight_log.info('Plot the smile effects at atmosphere abosorption features.')
            sensor_dict['smile_effect_figure_file'] = os.path.join(sensor_dict['output_dir'], basename+'_SmileEffectAtAtmFeatures.png')
            plot_smile_effect(sensor_dict['smile_effect_figure_file'],
                              sensor_dict['smile_effect_at_atm_features_file'])

            # Resample radiance spectra.
            flight_log.info('Resample the radiance spectra.')
            sensor_dict['resampled_rdn_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_ResampledRdn')
            resample_rdn(sensor_dict['resampled_rdn_image_file'],
                         sensor_dict['raw_rdn_image_file'],
                         sensor_dict['smile_effect_file'])

            del basename
        del sensor_index, sensor_dict

        #----------------------------------------Part 6----------------------------------------#
        flight_log.info('%sPart 6: Do geo-rectifications.' %('-'*10))
        for sensor_index, sensor_dict in flight_dict['sensors'].items():
            flight_log.info('Sensor: %s' %sensor_index)

            # Initialize.
            basename = os.path.basename(sensor_dict['dn_image_file'][:-len('_raw.hyspex')])

            # Orthorectify radiance images.
            flight_log.info('Geo-rectify the radiance image.')
            sensor_dict['ortho_rdn_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_OrthoRdn')
            orthorectify_rdn(sensor_dict['ortho_rdn_image_file'],
                             sensor_dict['resampled_rdn_image_file'],
                             sensor_dict['glt_image_file'])

            # Orthorectify DEM images.
            flight_log.info('Geo-rectify the DEM image.')
            sensor_dict['ortho_dem_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_OrthoDEM')
            orthorectify_dem(sensor_dict['ortho_dem_image_file'],
                             sensor_dict['igm_image_file'],
                             sensor_dict['glt_image_file'])

            # Orthorectify SCA images.
            flight_log.info('Geo-rectify the SCA image.')
            sensor_dict['ortho_sca_image_file'] = os.path.join(sensor_dict['output_dir'], basename+'_OrthoSCA')
            orthorectify_sca(sensor_dict['ortho_sca_image_file'],
                             sensor_dict['raw_sca_image_file'],
                             sensor_dict['glt_image_file'])

        #----------------------------------------Part 7----------------------------------------#
        flight_log.info('%sPart 7: Merge images from different sensors.' %('-'*10))

        # Merge DEM and SCA.
        flight_log.info('Merge orthorectified DEM and SCA images.')
        flight_dict['background_mask_file'] = os.path.join(flight_dict['merge_dir'], '%s_BackgroundMask' %flight_index)
        flight_dict['merged_dem_file'] = os.path.join(flight_dict['merge_dir'], '%s_DEM' %flight_index)
        flight_dict['merged_sca_file'] = os.path.join(flight_dict['merge_dir'], '%s_SCA' %flight_index)
        merge_dem_sca(flight_dict['background_mask_file'],
                      flight_dict['merged_dem_file'],
                      flight_dict['merged_sca_file'],
                      flight_dict['sensors'])

        # Merge radiance images.
        flight_log.info('Merge orthorectified radiance images.')
        flight_dict['merged_rdn_file'] = os.path.join(flight_dict['merge_dir'], '%s_Rdn' %flight_index)
        merge_rdn(flight_dict['merged_rdn_file'],
                  flight_dict['background_mask_file'],
                  flight_dict['sensors'])

        #----------------------------------------Part 8----------------------------------------#
        flight_log.info('%sPart 8: Do atmospheric corrections.' %('-'*10))

        # Resample atmosphere look-up table to sensor wavelengths.
        flight_log.info('Resample the raw ALT to sensor wavelengths.')
        flight_dict['resampled_atm_lut_file'] = os.path.join(flight_dict['atm_dir'], '%s_ResampledALT' %flight_index)
        resample_atm_lut(flight_dict['resampled_atm_lut_file'],
                         flight_dict['raw_atm_lut_file'],
                         os.path.splitext(flight_dict['merged_rdn_file'])[0]+'.hdr')

        # Estimate visibility.
        flight_dict['vis_file'] = os.path.join(flight_dict['merge_dir'], '%s_VIS' %flight_index)
        flight_dict['ddv_file'] = os.path.join(flight_dict['merge_dir'], '%s_DDV' %flight_index)
        estimate_vis(flight_dict['vis_file'],
                     flight_dict['ddv_file'],
                     flight_dict['resampled_atm_lut_file'],
                     flight_dict['merged_rdn_file'],
                     flight_dict['merged_sca_file'],
                     flight_dict['background_mask_file'])

        # Estimate water vapor column.
        flight_log.info('Estimate WVC.')
        flight_dict['wvc_file'] = os.path.join(flight_dict['merge_dir'], '%s_WVC' %flight_index)
        estimate_wvc(flight_dict['wvc_file'],
                     flight_dict['merged_rdn_file'],
                     flight_dict['sensors'],
                     flight_dict['sun_angles'][0],
                     flight_dict['sun_earth_distance'],
                     flight_dict['background_mask_file'])

        # Do atmospheric corrections.
        flight_log.info('Do atmospheric corrections.')
        flight_dict['refl_file'] = os.path.join(flight_dict['merge_dir'], '%s_Refl' %flight_index)
        atm_corr_image(flight_dict)

    logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    flight_dict = HyspexPro(args.config_file)
