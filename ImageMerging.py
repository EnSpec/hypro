#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to merge images from different sensors.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)
def merge_dem_sca(background_mask_file, merged_dem_file, merged_sca_file, sensors):
    """ Merge DEM and SCA images.
    Arguments:
        background_mask_file: str
            Background mask filename.
        merged_dem_file: str
            Merged DEM file.
        merged_sca_file: str
            Merged SCA file.
        sensors: dict
            Sensor dictionaries.
    """
    
    if os.path.exists(background_mask_file) and os.path.exists(merged_dem_file) and os.path.exists(merged_sca_file):
        logger.info('Write the background mask to %s.' %background_mask_file)
        logger.info('Write the merged DEM image to %s.' %merged_dem_file)
        logger.info('Write the merged SCA image to %s.' %merged_sca_file)
        return

    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    """
        Get spatial extent and pixel size.
    """
    ulx, uly, lrx, lry, pixel_size = -np.inf, np.inf, np.inf, -np.inf, np.inf
    for sensor_index, sensor_dict in sensors.items():
        tmp_header = read_envi_header(os.path.splitext(sensor_dict['ortho_dem_image_file'])[0]+'.hdr')
        tmp_ulx, tmp_uly = float(tmp_header['map info'][3]), float(tmp_header['map info'][4])
        tmp_pixel_size = float(tmp_header['map info'][5])
        tmp_lrx, tmp_lry = tmp_ulx+tmp_header['samples']*tmp_pixel_size, tmp_uly-tmp_header['lines']*tmp_pixel_size
        if tmp_ulx > ulx:
            ulx = tmp_ulx
        if tmp_uly < uly:
            uly = tmp_uly
        if tmp_lrx < lrx:
            lrx = tmp_lrx
        if tmp_lry > lry:
            lry = tmp_lry
        if tmp_pixel_size < pixel_size:
            pixel_size = tmp_pixel_size
        del tmp_ulx, tmp_uly, tmp_pixel_size, tmp_lrx, tmp_lry
    pixel_size = 2*pixel_size
    ulx, uly = np.ceil(ulx/pixel_size+1)*pixel_size, np.floor(uly/pixel_size-1)*pixel_size
    lrx, lry = np.ceil(lrx/pixel_size-1)*pixel_size, np.floor(lry/pixel_size+1)*pixel_size
    map_info = [tmp_header['map info'][0], 1, 1, ulx, uly, pixel_size, pixel_size]+tmp_header['map info'][7:]
    crs = tmp_header['coordinate system string']
    del tmp_header
    logger.info('The spatial range and pixel size of merged images:')
    logger.info('Map x = %.2f - %.2f' %(ulx, lrx))
    logger.info('Map y = %.2f - %.2f' %(lry, uly))
    logger.info('Pixel size = %.2f' %pixel_size)

    """
        Determine regular map grids.
    """
    x, y = np.meshgrid(np.arange(ulx, lrx, pixel_size), np.arange(uly, lry, -pixel_size))
    mask = np.full(x.shape, True, dtype='bool')
    
    """
        Build a background mask.
    """
    # Use DEM and SCA images to build this mask.
    for sensor_index, sensor_dict in sensors.items():
        # Use dem.
        tmp_header = read_envi_header(os.path.splitext(sensor_dict['ortho_dem_image_file'])[0]+'.hdr')
        tmp_image = np.memmap(sensor_dict['ortho_dem_image_file'],
                              dtype='float32',
                              mode='r',
                              shape=(tmp_header['lines'],
                                     tmp_header['samples']))
        tmp_ulx, tmp_uly = float(tmp_header['map info'][3]), float(tmp_header['map info'][4])
        tmp_pixel_size = float(tmp_header['map info'][5])
        resampled_image = resample_ortho_dem(np.copy(tmp_image), tmp_ulx, tmp_uly, tmp_pixel_size, x, y)
        mask = mask&(resampled_image>0.0)
        
        # Use sca.
        tmp_header = read_envi_header(os.path.splitext(sensor_dict['ortho_sca_image_file'])[0]+'.hdr')
        tmp_image = np.memmap(sensor_dict['ortho_sca_image_file'],
                              dtype='float32',
                              mode='r',
                              shape=(tmp_header['bands'],
                                     tmp_header['lines'],
                                     tmp_header['samples']))
        tmp_ulx, tmp_uly = float(tmp_header['map info'][3]), float(tmp_header['map info'][4])
        tmp_pixel_size = float(tmp_header['map info'][5])
        resampled_image = resample_ortho_sca(np.copy(tmp_image[0,:,:]), tmp_ulx, tmp_uly, tmp_pixel_size, x, y) 
        mask = mask&(resampled_image>0.0)

        # Clear data.
        del tmp_header, tmp_ulx, tmp_uly, tmp_pixel_size, resampled_image
        tmp_image.flush()
        del tmp_image
        
    mask = ~mask # 1: background pixels; 0: non-background pixels.

    # Write the mask to a file.
    fid = open(background_mask_file, 'wb')
    fid.write(mask.tostring())
    fid.close()

    # Write the mask header to a file.
    header = empty_envi_header()
    header['description'] = 'Background mask (0: non-background; 1: background)'
    header['file type'] = 'ENVI Standard'
    header['samples'] = mask.shape[1]
    header['lines'] = mask.shape[0]
    header['bands'] = 1
    header['byte order'] = 0
    header['header offset'] = 0
    header['interleave'] = 'bsq'
    header['data type'] = 1
    header['map info'] = map_info
    header['coordinate system string'] = crs
    write_envi_header(os.path.splitext(background_mask_file)[0]+'.hdr', header)
    del header

    logger.info('Write the background mask to %s.' %background_mask_file)

    """
        Merge DEM.
    """
    # Read the first DEM.
    sensor_index = list(sensors.keys())[0]
    sensor_dict = sensors[sensor_index]
    raw_header = read_envi_header(os.path.splitext(sensor_dict['ortho_dem_image_file'])[0]+'.hdr')
    raw_image = np.memmap(sensor_dict['ortho_dem_image_file'],
                          dtype='float32',
                          mode='r',
                          shape=(raw_header['lines'],
                                 raw_header['samples']))
    resampled_image = resample_ortho_dem(np.copy(raw_image),
                                         float(raw_header['map info'][3]),
                                         float(raw_header['map info'][4]),
                                         float(raw_header['map info'][5]),
                                         x, y)
    resampled_image[mask] = -1000.0

    # Write the merged DEM to a file.
    fid = open(merged_dem_file, 'wb')
    fid.write(resampled_image.astype('float32').tostring())
    fid.close()
    del raw_header, sensor_index, sensor_dict, resampled_image
    raw_image.flush()
    del raw_image
    
    # Write the merged DEM header to a file.
    header = empty_envi_header()
    header['description'] = 'Merged DEM, in [m]'
    header['file type'] = 'ENVI Standard'
    header['samples'] = mask.shape[1]
    header['lines'] = mask.shape[0]
    header['bands'] = 1
    header['byte order'] = 0
    header['header offset'] = 0
    header['interleave'] = 'bsq'
    header['data type'] = 4
    header['data ignore value'] = -1000.0
    header['map info'] = map_info
    header['coordinate system string'] = crs
    write_envi_header(os.path.splitext(merged_dem_file)[0]+'.hdr', header)
    del header

    logger.info('Write the merged DEM image to %s.' %merged_dem_file)


    """
        Merge SCA.
    """
    # Read the first SCA.
    sensor_index = list(sensors.keys())[0]
    sensor_dict = sensors[sensor_index]
    raw_header = read_envi_header(os.path.splitext(sensor_dict['ortho_sca_image_file'])[0]+'.hdr')
    raw_image = np.memmap(sensor_dict['ortho_sca_image_file'],
                          dtype='float32',
                          mode='r',
                          shape=(raw_header['bands'],
                                 raw_header['lines'],
                                 raw_header['samples']))

    # Write data.
    fid = open(merged_sca_file, 'wb')
    for band in range(raw_header['bands']):
        resampled_image = resample_ortho_sca(np.copy(raw_image[band,:,:]),
                                             float(raw_header['map info'][3]),
                                             float(raw_header['map info'][4]),
                                             float(raw_header['map info'][5]),
                                             x, y)
        resampled_image[mask] = -1000.0
        fid.write(resampled_image.astype('float32').tostring())
        del resampled_image
    fid.close()
    del sensor_index, sensor_dict
    raw_image.flush()
    del raw_image
    
    # Write header.
    header = empty_envi_header()
    header['description'] = 'Merged SCA, in [deg]'
    header['file type'] = 'ENVI Standard'
    header['samples'] = mask.shape[1]
    header['lines'] = mask.shape[0]
    header['bands'] = 2
    header['byte order'] = 0
    header['header offset'] = 0
    header['interleave'] = 'bsq'
    header['data type'] = 4
    header['band names'] = ['Sensor Zenith [deg]', 'Sensor Azimuth [deg]']
    header['sun azimuth'] = raw_header['sun azimuth']
    header['sun zenith'] = raw_header['sun zenith']
    header['data ignore value'] = -1000.0
    header['map info'] = map_info
    header['coordinate system string'] = crs
    write_envi_header(os.path.splitext(merged_sca_file)[0]+'.hdr', header)
    del raw_header, header
    logger.info('Write the merged SCA image to %s.' %merged_sca_file)

def merge_rdn(merged_image_file, mask_file, sensors):
    """ Merge radiance images.
    Arguments:
        merged_image_file: str
            Merged radiance image filename.
        mask_file: str
            Background mask filename.
        sensors: dict
            Sensor dictionaries.
    """

    if os.path.exists(merged_image_file):
        logger.info('Write the merged refletance image to %s.' %merged_image_file)
        return

    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    # Read mask.
    mask_header = read_envi_header(os.path.splitext(mask_file)[0]+'.hdr')
    mask_image = np.memmap(mask_file,
                           mode='r',
                           dtype='bool',
                           shape=(mask_header['lines'],
                                  mask_header['samples']))

    # Get the map upper-left coordinates and pixel sizes of VNIR and SWIR images.
    ulx, uly, pixel_size = float(mask_header['map info'][3]), float(mask_header['map info'][4]), float(mask_header['map info'][5])

    # Determine regular map grids.
    x, y = np.meshgrid(ulx+np.arange(mask_header['samples'])*pixel_size,
                       uly-np.arange(mask_header['lines'])*pixel_size)
    del ulx, uly, pixel_size

    # Read radiance header and image.
    header_dict = dict()
    image_file_dict = dict()
    bands_waves_fwhms = []
    for sensor_index, sensor_dict in sensors.items():
        tmp_header = read_envi_header(os.path.splitext(sensor_dict['ortho_rdn_image_file'])[0]+'.hdr')
        for band in range(tmp_header['bands']):
            bands_waves_fwhms.append(['%s_%d' %(sensor_index,band), tmp_header['wavelength'][band], tmp_header['fwhm'][band]])
        header_dict[sensor_index] = tmp_header
        image_file_dict[sensor_index] = sensor_dict['ortho_rdn_image_file']
    bands_waves_fwhms.sort(key = lambda x: x[1])
    
    # Merge images.
    wavelengths = []
    fwhms = []
    fid  = open(merged_image_file, 'wb')
    for v in bands_waves_fwhms:
        # Determine which sensor, band to read.
        sensor_index, band = v[0].split('_')
        band = int(band)
        wavelengths.append(v[1])
        fwhms.append(v[2])
        header = header_dict[sensor_index]
        image_file = image_file_dict[sensor_index]
        
        # Write image.
        if ((v[1]>=1339.0)&(v[1]<=1438.0))|((v[1]>=1808.0)&(v[1]<=1978.0))|(v[1]>=2467.0):
            resampled_image = np.zeros(x.shape)
        else:
            offset = header['header offset']+4*band*header['lines']*header['samples']# in bytes      
            rdn_image = np.memmap(image_file,
                                  dtype='float32',
                                  mode='r',
                                  offset=offset,
                                  shape=(header['lines'], header['samples']))
            resampled_image = resample_ortho_rdn(np.copy(rdn_image),
                                                 float(header_dict[sensor_index]['map info'][3]),
                                                 float(header_dict[sensor_index]['map info'][4]),
                                                 float(header_dict[sensor_index]['map info'][5]),
                                                 x, y)
            resampled_image[mask_image] = 0.0
            rdn_image.flush()
            del rdn_image
            
        fid.write(resampled_image.astype('float32').tostring())
        del resampled_image
    fid.close()
    del header_dict, image_file_dict, x, y
    mask_image.flush()
    del mask_image
    
    # Write header.
    header = empty_envi_header()
    header['description'] = 'Merged radiance, in [mW/(cm2*um*sr)]'
    header['file type'] = 'ENVI Standard'
    header['samples'] = mask_header['samples']
    header['lines'] = mask_header['lines']
    header['bands'] = len(wavelengths)
    header['byte order'] = 0
    header['header offset'] = 0
    header['interleave'] = 'bsq'
    header['data type'] = 4
    header['wavelength'] = wavelengths
    header['fwhm'] = fwhms
    header['wavelength units'] = 'nm'
    header['acquisition time'] = tmp_header['acquisition time']
    header['map info'] = mask_header['map info']
    header['coordinate system string'] = mask_header['coordinate system string']
    write_envi_header(os.path.splitext(merged_image_file)[0]+'.hdr', header)
    del header, tmp_header

    logger.info('Write the merged refletance image to %s.' %merged_image_file)
    
def resample_ortho_sca(raw_image, raw_ulx, raw_uly, raw_pixel_size, x, y):
    """ Resample geosca image to new map grids.
    Arguments:
        raw_image: 2D array
            Raw geosca image data.
        raw_ulx, raw_uly: float
            Map coordinates of the upper-left corner of the raw image.
        raw_pixel_size: float
            Pixel size of the raw image.
        x, y: 2D array
            New map grids.
    Returns:
        resampled_image: 2D array
            Resampled geosca image.
    """

    from scipy import ndimage

    # Average reflectance
    weights = np.ones((2,2))
    count = raw_image>=0.0
    raw_image[~count] = 0.0
    avg_image = ndimage.convolve(raw_image, weights)
    count = ndimage.convolve(count.astype('int8'), weights)
    avg_image[count>0] /= count[count>0]
    avg_image[count==0] = -1000.0
    del count

    # Resample
    samples = ((x-raw_ulx)/raw_pixel_size).astype('int32')
    lines = ((raw_uly-y)/raw_pixel_size).astype('int32')
    resampled_image = avg_image[lines, samples]
    del avg_image, samples, lines

    return resampled_image

def resample_ortho_dem(raw_image, raw_ulx, raw_uly, raw_pixel_size, x, y):
    """ Resample dem image to new map grids.
    Arguments:
        raw_image: 2D array
            Raw ortho dem image data.
        raw_ulx, raw_uly: float
            Map coordinates of the upper-left corner of the raw image.
        raw_pixel_size: float
            Pixel size of the raw image.
        x, y: 2D array
            New map grids.
    Returns:
        resampled_image: 2D array
            Resampled geodem image.
    """

    from scipy import ndimage

    # Average reflectance
    weights = np.ones((2,2))
    count = raw_image>0.0
    raw_image[~count]=0.0
    avg_image = ndimage.convolve(raw_image, weights)
    count = ndimage.convolve(count.astype('int8'), weights)
    avg_image[count>0] /= count[count>0]
    avg_image[count==0] = -1000.0
    del count

    # Resample
    samples = ((x-raw_ulx)/raw_pixel_size).astype('int32')
    lines = ((raw_uly-y)/raw_pixel_size).astype('int32')
    resampled_image = avg_image[lines, samples]
    del avg_image, samples, lines

    return resampled_image

def resample_ortho_rdn(raw_image, raw_ulx, raw_uly, raw_pixel_size, x, y):
    """ Resample radiance image to new map grids.
    Arguments:
        raw_image: 2D array
            Radiance image data.
        raw_ulx, raw_uly: float
            Map coordinates of the upper-left corner of the raw image.
        raw_pixel_size: float
            Pixel size of the raw image.
        x, y: 2D array
            New map grids.
    Returns:
        resampled_image: 2D array
            Resampled refletance image.
    """

    from scipy import ndimage

    # Average radiance.
    weights = np.ones((2,2))
    count = raw_image>0.0
    raw_image[~count]=0.0
    avg_image = ndimage.convolve(raw_image, weights)
    count = ndimage.convolve(count.astype('int8'), weights)
    avg_image[count>0] /= count[count>0]
    del count

    # Resample.
    samples = ((x-raw_ulx)/raw_pixel_size).astype('int32')
    lines = ((raw_uly-y)/raw_pixel_size).astype('int32')
    resampled_image = avg_image[lines, samples]
    del avg_image, samples, lines, raw_image
    
    return resampled_image
