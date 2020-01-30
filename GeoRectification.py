#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to do geo-rectification.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, numpy as np
logger = logging.getLogger(__name__)

def orthorectify_sca(ortho_sca_image_file, sca_image_file, glt_image_file):
    """ Do orthorectifications on SCA.
    Arguments:
        ortho_sca_image_file: str
            Orthorectified SCA image filename.
        sca_image_file: str
            SCA image filename.
        glt_image_file: str
            Geographic look-up table image filename.
    """

    if os.path.exists(ortho_sca_image_file):
        logger.info('Write the geo-rectified SCA image to %s.' %ortho_sca_image_file)
        return

    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    # Read SCA image.
    sca_header = read_envi_header(os.path.splitext(sca_image_file)[0]+'.hdr')

    # Read GLT image.
    glt_header = read_envi_header(os.path.splitext(glt_image_file)[0]+'.hdr')
    glt_image = np.memmap(glt_image_file,
                          dtype=np.int32,
                          mode='r',
                          shape=(glt_header['bands'],
                                 glt_header['lines'],
                                 glt_header['samples']))

    # Get spatial indices.
    I, J = np.where((glt_image[0,:,:]>=0)&(glt_image[1,:,:]>=0))
    ortho_sca_image = np.zeros((glt_header['lines'], glt_header['samples']), dtype='float32')

    # Orthorectify SCA.
    fid = open(ortho_sca_image_file, 'wb')
    for band in range(sca_header['bands']):
        ortho_sca_image[:,:] = -1000.0
        offset = sca_header['header offset']++4*band*sca_header['lines']*sca_header['samples']
        sca_image = np.memmap(sca_image_file,
                              dtype='float32',
                              mode='r',
                              offset=offset,
                              shape=(sca_header['lines'], sca_header['samples']))
        ortho_sca_image[I,J] = sca_image[glt_image[0,I,J], glt_image[1,I,J]]
        fid.write(ortho_sca_image.tostring())
        sca_image.flush()
        del sca_image
    fid.close()
    del ortho_sca_image
    glt_image.flush()
    del glt_image
    
    # Write header
    ortho_sca_header = empty_envi_header()
    ortho_sca_header['description'] = 'Geo-rectified SCA, in [deg]'
    ortho_sca_header['file type'] = 'ENVI Standard'
    ortho_sca_header['samples'] = glt_header['samples']
    ortho_sca_header['lines'] = glt_header['lines']
    ortho_sca_header['bands'] = sca_header['bands']
    ortho_sca_header['byte order'] = 0
    ortho_sca_header['header offset'] = 0
    ortho_sca_header['data ignore value'] = -1000.0
    ortho_sca_header['interleave'] = 'bsq'
    ortho_sca_header['data type'] = 4
    ortho_sca_header['band names'] = sca_header['band names']
    ortho_sca_header['sun zenith'] = sca_header['sun zenith']
    ortho_sca_header['sun azimuth'] = sca_header['sun azimuth']
    ortho_sca_header['map info'] = glt_header['map info']
    ortho_sca_header['coordinate system string'] = glt_header['coordinate system string']
    write_envi_header(ortho_sca_image_file+'.hdr', ortho_sca_header)
    del glt_header, sca_header, ortho_sca_header

    logger.info('Write the geo-rectified SCA image to %s.' %ortho_sca_image_file)

def orthorectify_dem(ortho_dem_image_file, igm_image_file, glt_image_file):
    """ Do orthorectifications on DEM.
    Arguments:
        ortho_dem_image_file: str
            Orthorectified DEM image filename.
        igm_image_file: str
            IGM image filename.
        glt_image_file: str
            Geographic look-up table image filename.
    """

    if os.path.exists(ortho_dem_image_file):
        logger.info('Write the geo-rectified DEM image to %s.' %ortho_dem_image_file)
        return

    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    # Read IGM image (the third band is DEM).
    igm_header = read_envi_header(igm_image_file+'.hdr')
    igm_image = np.memmap(igm_image_file,
                          dtype='float64',
                          mode='r',
                          shape=(igm_header['bands'],
                                 igm_header['lines'],
                                 igm_header['samples']))
    del igm_header

    # Read GLT image.
    glt_header = read_envi_header(glt_image_file+'.hdr')
    glt_image = np.memmap(glt_image_file,
                          dtype=np.int32,
                          mode='r',
                          shape=(glt_header['bands'],
                                 glt_header['lines'],
                                 glt_header['samples']))

    # Get spatial indices.
    I, J = np.where((glt_image[0,:,:]>=0)&(glt_image[1,:,:]>=0))
    ortho_dem_image = np.zeros((glt_header['lines'], glt_header['samples']), dtype='float32')

    # Orthorectify DEM.
    fid = open(ortho_dem_image_file, 'wb')
    ortho_dem_image[:,:] = -1000.0
    ortho_dem_image[I,J] = igm_image[2, glt_image[0,I,J], glt_image[1,I,J]]
    fid.write(ortho_dem_image.tostring())
    fid.close()
    del ortho_dem_image
    igm_image.flush()
    glt_image.flush()
    del igm_image, glt_image
    
    # Write header.
    ortho_dem_header = empty_envi_header()
    ortho_dem_header['description'] = 'Geo-rectified DEM, in [m]'
    ortho_dem_header['file type'] = 'ENVI Standard'
    ortho_dem_header['samples'] = glt_header['samples']
    ortho_dem_header['lines'] = glt_header['lines']
    ortho_dem_header['bands'] = 1
    ortho_dem_header['byte order'] = 0
    ortho_dem_header['header offset'] = 0
    ortho_dem_header['data ignore value'] = -1000.0
    ortho_dem_header['interleave'] = 'bsq'
    ortho_dem_header['data type'] = 4
    ortho_dem_header['map info'] = glt_header['map info']
    ortho_dem_header['coordinate system string'] = glt_header['coordinate system string']
    write_envi_header(ortho_dem_image_file+'.hdr', ortho_dem_header)
    del glt_header, ortho_dem_header

    logger.info('Write the geo-rectified DEM image to %s.' %ortho_dem_image_file)

def orthorectify_rdn(ortho_rdn_image_file, rdn_image_file, glt_image_file):
    """ Do orthorectifications on radiance.
    Arguments:
        ortho_rdn_image_file: str
            Orthorectified radiance filename.
        rdn_image_file: str
            Radiance image filename.
        glt_image_file: str
            Geographic look-up table image filename.
    """

    if os.path.exists(ortho_rdn_image_file):
        logger.info('Write the geo-rectified radiance image to %s.' %ortho_rdn_image_file)
        return

    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    # Read radiance image.
    rdn_header = read_envi_header(rdn_image_file+'.hdr')
    rdn_image = np.memmap(rdn_image_file,
                          dtype='float32',
                          mode='r',
                          shape=(rdn_header['lines'],
                                 rdn_header['bands'],
                                 rdn_header['samples']))

    # Read GLT image.
    glt_header = read_envi_header(glt_image_file+'.hdr')
    glt_image = np.memmap(glt_image_file,
                          dtype=np.int32,
                          mode='r',
                          offset=0,
                          shape=(glt_header['bands'],
                                 glt_header['lines'],
                                 glt_header['samples']))

    # Get spatial indices.
    I, J = np.where((glt_image[0,:,:]>=0)&(glt_image[1,:,:]>=0))
    ortho_image = np.zeros((glt_header['lines'], glt_header['samples']), dtype='float32')

    # Orthorectify radiance.
    fid = open(ortho_rdn_image_file, 'wb')
    info = 'Band (max=%d): ' %rdn_header['bands']
    for band in range(rdn_header['bands']):
        if band%20==0:
            info += '%d, ' %(band+1)
        ortho_image[:,:] = 0.0
        ortho_image[I,J] = rdn_image[glt_image[0,I,J], band, glt_image[1,I,J]]
        fid.write(ortho_image.tostring())
    fid.close()
    info += 'Done! %s' %rdn_header['bands']
    logger.info(info)
    del ortho_image
    rdn_image.flush()
    glt_image.flush()
    del rdn_image, glt_image
    
    # Write header.
    ortho_rdn_header = empty_envi_header()
    ortho_rdn_header['description'] = 'Geo-rectified radiance, in [mW/(cm2*um*sr)]'
    ortho_rdn_header['file type'] = 'ENVI Standard'
    ortho_rdn_header['samples'] = glt_header['samples']
    ortho_rdn_header['lines'] = glt_header['lines']
    ortho_rdn_header['bands'] = rdn_header['bands']
    ortho_rdn_header['byte order'] = 0
    ortho_rdn_header['header offset'] = 0
    ortho_rdn_header['interleave'] = 'bsq'
    ortho_rdn_header['data type'] = 4
    ortho_rdn_header['wavelength'] = rdn_header['wavelength']
    ortho_rdn_header['fwhm'] = rdn_header['fwhm']
    ortho_rdn_header['wavelength units'] = 'nm'
    ortho_rdn_header['default bands'] = rdn_header['default bands']
    ortho_rdn_header['acquisition time'] = rdn_header['acquisition time']
    ortho_rdn_header['map info'] = glt_header['map info']
    ortho_rdn_header['coordinate system string'] = glt_header['coordinate system string']
    write_envi_header(ortho_rdn_image_file+'.hdr', ortho_rdn_header)
    del glt_header, rdn_header, ortho_rdn_header

    logger.info('Write the geo-rectified radiance image to %s.' %ortho_rdn_image_file)
