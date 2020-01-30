#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to do geo-referencing.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import gdal, logging, os, numpy as np
logger = logging.getLogger(__name__)
from numba import guvectorize, jit
import warnings
warnings.filterwarnings("ignore")

def calculate_igm(igm_image_file, imugps_file, sensor_model_file, dem_image_file, boresight_options):
    """ Create an input geometry (IGM) image.
    Arguments:
        igm_image_file: str
            The IGM image filename.
        imugps_file: str
            The IMUGPS filename.
        sensor_model_file: str
            The sensor model filename.
        dem_image_file: str
            The DEM image filename.
        boresight_options: list of boolean
            Boresight offset options, true or false.
    """

    if os.path.exists(igm_image_file):
        logger.info('Write the IGM to %s.' %igm_image_file)
        return

    from ENVI import empty_envi_header, write_envi_header
    from scipy import interpolate
    
    # Read IMU and GPS data.
    imugps = np.loadtxt(imugps_file) # ID, X, Y, Z, R, P, H, R_Offset, P_Offset, H_Offset, Grid_Convergence

    # Read sensor model data.
    sensor_model = np.loadtxt(sensor_model_file, skiprows=1)[:,1:]

    # Read DEM data.
    ds = gdal.Open(dem_image_file, gdal.GA_ReadOnly)
    dem_image = ds.GetRasterBand(1).ReadAsArray()
    dem_geotransform = ds.GetGeoTransform()
    dem_prj = ds.GetProjection()
    ds = None

    # Apply boresight offsets.
    if boresight_options[0]:
        imugps[:,4] += imugps[:,7]
    if boresight_options[1]:
        imugps[:,5] += imugps[:,8]
    if boresight_options[2]:
        imugps[:,6] += imugps[:,9]
    if boresight_options[3]:
        imugps[:,3] += imugps[:,10]
    imugps[:,6] -= imugps[:,11] # Heading - grid convergence

    # Get scan vectors.
    L0 = get_scan_vectors(imugps[:,4:7], sensor_model)
    del sensor_model

    # Get start and end points of ray tracing.
    index = dem_image>0.0
    dem_min = dem_image[index].min()
    dem_max = dem_image[index].max()
    del index
    xyz0, xyz1 = get_xyz0_xyz1(imugps[:,1:4], L0, dem_min, dem_max)
    
    # Ray-tracing.
    lines = imugps.shape[0]
    samples = L0.shape[1]
    logger.info('Beginning ray-tracing ({} scanlines)...'.format(lines))
    igm_image = ray_tracer_ufunc(xyz0, xyz1, L0, dem_image, dem_geotransform)
    del dem_image, xyz0, xyz1, L0, imugps
    logger.info('Ray-tracing complete.')

    # Interpolate IGM.
    nan_lines = []
    nonnan_lines = []
    for line in np.arange(lines):
        # Find Nan values.
        nan_flag = np.isnan(igm_image[0, line, :])

        # If all columns are Nan values;
        if np.all(nan_flag):
            del nan_flag
            nan_lines.append(line)
            continue

        # If some columns are Nan values;
        if np.any(nan_flag):
            nan_samples = np.arange(samples)[nan_flag]
            nonnan_samples = np.arange(samples)[~nan_flag]
            f = interpolate.interp1d(nonnan_samples, igm_image[:, line, nonnan_samples], axis=1, fill_value="extrapolate")
            igm_image[:, line, nan_samples] = f(nan_samples)
            del nan_samples, nonnan_samples, f, nan_flag

        nonnan_lines.append(line)
        
    for nan_line in nan_lines:
        index = np.argmin(np.abs(nonnan_lines-nan_line))
        nonnan_line = nonnan_lines[index]
        igm_image[:,nan_line,:] = igm_image[:,nonnan_line,:]
        del index, nonnan_line
    logger.info('IGM interpolation complete.')

    # Write IGM image.
    fid = open(igm_image_file, 'wb')
    fid.write(igm_image.tostring())
    fid.close()
    del igm_image
    
    # Write IGM header file.
    igm_header = empty_envi_header()
    igm_header['description'] = 'IGM (map coordinate system=%s)' %(dem_prj)
    igm_header['file type'] = 'ENVI Standard'
    igm_header['samples'] = samples
    igm_header['lines'] = lines
    igm_header['bands'] = 3
    igm_header['byte order'] = 0
    igm_header['header offset'] = 0
    igm_header['interleave'] = 'bsq'
    igm_header['data type'] = 5
    igm_header['band names'] = ['IGM Map X', 'IGM Map Y', 'IGM Map Z']
    igm_header_file = os.path.splitext(igm_image_file)[0]+'.hdr'
    write_envi_header(igm_header_file, igm_header)
    del igm_header, dem_prj

    logger.info('Write the IGM to %s.' %igm_image_file)
    
def calculate_sca(sca_image_file, imugps_file, igm_image_file, sun_angles):
    """ Create a scan angle (SCA) image.
    Arguments:
        sca_image_file: str
            Scan angle filename.
        imu_gps_file: str
            IMU/GPS filename.
        igm_image_file: str
            IGM image filename.
        sun_angles: list
            Sun angles [sun zenith, sun azimuth], in degrees.
    """

    if os.path.exists(sca_image_file):
        logger.info('Write the SCA to %s.' %sca_image_file)
        return
    
    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    # Read IGM data.
    igm_header = read_envi_header(os.path.splitext(igm_image_file)[0]+'.hdr')
    igm_image = np.memmap(igm_image_file,
                          dtype='float64',
                          mode='r',
                          offset=0,
                          shape=(igm_header['bands'],
                                 igm_header['lines'],
                                 igm_header['samples']))

    # Read GPS data.
    imugps = np.loadtxt(imugps_file) # ID, X, Y, Z, R, P, H, ...
    
    # Calculate sensor angles.
    DX = igm_image[0,:,:]-np.expand_dims(imugps[:,1], axis=1)
    DY = igm_image[1,:,:]-np.expand_dims(imugps[:,2], axis=1)
    DZ = igm_image[2,:,:]-np.expand_dims(imugps[:,3], axis=1)
    
    igm_image.flush()
    del imugps, igm_image
    
    view_zenith = np.abs(np.arctan(np.sqrt(DX**2+DY**2)/DZ))
    
    index = view_zenith>=np.deg2rad(40.0)
    if np.any(index):
        view_zenith[index] = np.deg2rad(39.0)
    del index
    
    view_azimuth = np.arcsin(np.abs(DX)/np.sqrt(DX**2+DY**2))
    
    index = (DX>0)&(DY<0)
    view_azimuth[index]=np.pi-view_azimuth[index]
    index = (DX<0)&(DY<0)
    view_azimuth[index]=np.pi+view_azimuth[index]
    index = (DX<0)&(DY>0)
    view_azimuth[index]=2*np.pi-view_azimuth[index]
    del DX, DY, DZ, index

    # Save scan angles.
    fid = open(sca_image_file, 'wb')
    fid.write(np.rad2deg(view_zenith).astype('float32').tostring())
    fid.write(np.rad2deg(view_azimuth).astype('float32').tostring())
    fid.close()
    del view_zenith, view_azimuth

    # Write scan angle header file.
    sca_header = empty_envi_header()
    sca_header['description'] = 'SCA [deg]'
    sca_header['file type'] = 'ENVI Standard'
    sca_header['samples'] = igm_header['samples']
    sca_header['lines'] = igm_header['lines']
    sca_header['bands'] = 2
    sca_header['byte order'] = 0
    sca_header['header offset'] = 0
    sca_header['interleave'] = 'bsq'
    sca_header['data type'] = 4
    sca_header['band names'] = ['Sensor Zenith [deg]', 'Sensor Azimuth [deg]']
    sca_header['sun zenith'] = sun_angles[0]
    sca_header['sun azimuth'] = sun_angles[1]
    sca_header_file = os.path.splitext(sca_image_file)[0]+'.hdr'
    write_envi_header(sca_header_file, sca_header)
    del sca_header

    logger.info('Write the SCA to %s.' %sca_image_file)

def build_glt(glt_image_file, igm_image_file, pixel_size, map_crs):
    """ Create a geographic lookup table (GLT) image.
    Notes:
        (1) This code is adapted from Adam Chlus's (chlus@wisc.edu) script.
        (2) The GLT image consists of two bands:
                Band 0: Sample Lookup:
                    Pixel values indicate the column number of the pixel
                    in the input geometry file that belongs at the given Y
                    location in the output image.
                Band 1: Line Lookup:
                    Pixel values indicate the row number of the pixel
                    in the input geometry file that belongs at the given X
                    location in the output image.
        (3) For more details about GLT, refer to https://www.harrisgeospatial.com/
            docs/GeoreferenceFromInputGeometry.html.
    Arguments:
        glt_image_file: str
            Geographic look-up table filename.
        igm_image_file: str
            Input geometry filename.
        pixel_size: float
            Output image pixel size.
        map_crs: osr object
            GLT image map coordinate system.
    """

    if os.path.exists(glt_image_file):
        logger.info('Write the GLT to %s.' %glt_image_file)
        return

    from ENVI import empty_envi_header, read_envi_header, write_envi_header

    # Read IGM.
    igm_header = read_envi_header(os.path.splitext(igm_image_file)[0]+'.hdr')
    igm_image = np.memmap(igm_image_file,
                          dtype='float64',
                          mode='r',
                          offset=0,
                          shape=(igm_header['bands'], igm_header['lines'], igm_header['samples']))

    # Estimate output spatial extent.
    X_Min = igm_image[0,:,:].min()
    X_Max = igm_image[0,:,:].max()
    Y_Min = igm_image[1,:,:].min()
    Y_Max = igm_image[1,:,:].max()
    X_Min = np.floor(X_Min/pixel_size)*pixel_size-pixel_size
    X_Max = np.ceil(X_Max/pixel_size)*pixel_size+pixel_size
    Y_Min = np.floor(Y_Min/pixel_size)*pixel_size-pixel_size
    Y_Max = np.ceil(Y_Max/pixel_size)*pixel_size+pixel_size
    igm_image.flush()
    del igm_image
    
    # Build VRT for IGM.
    igm_vrt_file = os.path.splitext(igm_image_file)[0]+'.vrt'
    igm_vrt = open(igm_vrt_file,'w')
    igm_vrt.write('<VRTDataset rasterxsize="%s" rasterysize="%s">\n' %(igm_header['samples'], igm_header['lines']))
    for band in range(igm_header['bands']):
        igm_vrt.write('\t<VRTRasterBand dataType="%s" band="%s">\n' % ("Float64", band+1))
        igm_vrt.write('\t\t<SimpleSource>\n')
        igm_vrt.write('\t\t\t<SourceFilename relativeToVRT="1">%s</SourceFilename>\n' %igm_image_file.replace('&', '&amp;'))
        igm_vrt.write('\t\t\t<SourceBand>%s</SourceBand>\n' %(band+1))
        igm_vrt.write('\t\t\t<SourceProperties RasterXSize="%s" RasterYSize="%s" DataType="%s" BlockXSize="%s" BlockYSize="%s" />\n' %(igm_header['samples'],
                                                                                                                                       igm_header['lines'],
                                                                                                                                       "Float64",
                                                                                                                                       igm_header['samples'], 1))
        igm_vrt.write('\t\t\t<SrcRect xOff="0" yOff="0" xSize="%s" ySize="%s" />\n' %(igm_header['samples'], igm_header['lines']))
        igm_vrt.write('\t\t\t<DstRect xOff="0" yOff="0" xSize="%s" ySize="%s" />\n' %(igm_header['samples'], igm_header['lines']))
        igm_vrt.write("\t\t</SimpleSource>\n")
        igm_vrt.write("\t</VRTRasterBand>\n")
    igm_vrt.write("</VRTDataset>\n")
    igm_vrt.close()

    # Make IGM index image.
    index_image_file = igm_image_file+'_Index'
    index_image = np.memmap(index_image_file,
                            dtype='int32',
                            mode='w+',
                            offset=0,
                            shape=(2,
                                   igm_header['lines'],
                                   igm_header['samples']))
    index_image[0,:,:], index_image[1,:,:] = np.mgrid[0:igm_header['lines'], 0:igm_header['samples']]
    index_image.flush()
    del index_image
    
    index_header = empty_envi_header()
    index_header['description'] = 'IGM Image Index'
    index_header['samples'] = igm_header['samples']
    index_header['lines'] = igm_header['lines']
    index_header['bands'] = 2
    index_header['byte order'] = 0
    index_header['header offset'] = 0
    index_header['interleave'] = 'bsq'
    index_header['data type'] = 3
    index_header['band names'] = ['Image Row', 'Image Column']
    index_header_file = os.path.splitext(index_image_file)[0]+'.hdr'
    write_envi_header(index_header_file, index_header)

    # Build VRT for IGM Index.
    index_vrt_file = os.path.splitext(index_image_file)[0]+'.vrt'
    index_vrt = open(index_vrt_file,'w')
    index_vrt.write('<VRTDataset rasterxsize="%s" rasterysize="%s">\n' %(index_header['samples'], index_header['lines']))
    index_vrt.write('\t<Metadata Domain = "GEOLOCATION">\n')
    index_vrt.write('\t\t<mdi key="X_DATASET">%s</mdi>\n' %igm_image_file.replace('&', '&amp;'))
    index_vrt.write('\t\t<mdi key="X_BAND">1</mdi>\n')
    index_vrt.write('\t\t<mdi key="Y_DATASET">%s</mdi>\n' %igm_image_file.replace('&', '&amp;'))
    index_vrt.write('\t\t<mdi key="Y_BAND">2</mdi>\n')
    index_vrt.write('\t\t<mdi key="PIXEL_OFFSET">0</mdi>\n')
    index_vrt.write('\t\t<mdi key="LINE_OFFSET">0</mdi>\n')
    index_vrt.write('\t\t<mdi key="PIXEL_STEP">1</mdi>\n')
    index_vrt.write('\t\t<mdi key="LINE_STEP">1</mdi>\n')
    index_vrt.write('\t</Metadata>\n')
    for band in range(index_header['bands']):
        index_vrt.write('\t<VRTRasterBand dataType="%s" band="%s">\n' % ("Int16", band+1))
        index_vrt.write('\t\t<SimpleSource>\n')
        index_vrt.write('\t\t\t<SourceFilename relativeToVRT="1">%s</SourceFilename>\n' %index_image_file.replace('&', '&amp;'))
        index_vrt.write('\t\t\t<SourceBand>%s</SourceBand>\n' %(band+1))
        index_vrt.write('\t\t\t<SourceProperties RasterXSize="%s" RasterYSize="%s" DataType="%s" BlockXSize="%s" BlockYSize="%s" />\n' %(index_header['samples'],
                                                                                                                                       index_header['lines'],
                                                                                                                                       "Int32",
                                                                                                                                       index_header['samples'], 1))
        index_vrt.write('\t\t\t<SrcRect xOff="0" yOff="0" xSize="%s" ySize="%s" />\n' %(index_header['samples'], index_header['lines']))
        index_vrt.write('\t\t\t<DstRect xOff="0" yOff="0" xSize="%s" ySize="%s" />\n' %(index_header['samples'], index_header['lines']))
        index_vrt.write("\t\t</SimpleSource>\n")
        index_vrt.write("\t</VRTRasterBand>\n")
    index_vrt.write("</VRTDataset>\n")
    index_vrt.close()
    
    # Build GLT.
    tmp_glt_image_file = glt_image_file+'.tif'
    tmp_glt_image = gdal.Warp(tmp_glt_image_file, index_vrt_file,
                              outputBounds=(X_Min, Y_Min, X_Max, Y_Max),
                              xRes=pixel_size, yRes=pixel_size,
                              resampleAlg='near',
                              dstSRS=map_crs.ExportToProj4(),
                              dstNodata=-1,
                              multithread=True,
                              geoloc=True)
    del tmp_glt_image

    # Convert the .tif file to ENVI format.
    ds = gdal.Open(tmp_glt_image_file, gdal.GA_ReadOnly)
    lines = ds.RasterYSize
    samples = ds.RasterXSize
    glt_image = np.memmap(glt_image_file,
                          dtype='int32',
                          mode='w+',
                          offset=0,
                          shape=(2, lines, samples))
    for band in range(ds.RasterCount):
        glt_image[band,:,:] = ds.GetRasterBand(band+1).ReadAsArray()
    glt_image.flush()
    del glt_image
    ds = None

    # Write GLT header.
    glt_header = empty_envi_header()
    glt_header['description'] = 'GLT'
    glt_header['file type'] = 'ENVI Standard'
    glt_header['samples'] = samples
    glt_header['lines'] = lines
    glt_header['bands'] = 2
    glt_header['byte order'] = 0
    glt_header['header offset'] = 0
    glt_header['interleave'] = 'bsq'
    glt_header['data type'] = 3
    glt_header['data ignore value'] = -1
    glt_header['band names'] = ['Image Row', 'Image Column']
    glt_header['coordinate system string'] = map_crs.ExportToWkt()
    glt_header['map info'] = [map_crs.GetAttrValue('projcs').replace(',',''),
              1, 1, X_Min, Y_Max, pixel_size, pixel_size, ' ', ' ',
              map_crs.GetAttrValue('datum'), map_crs.GetAttrValue('unit')]
    if map_crs.GetAttrValue('PROJECTION').lower() == 'transverse_mercator':
        glt_header['map info'][7] = map_crs.GetUTMZone()
        if Y_Max>0.0:
            glt_header['map info'][8] = 'North'
        else:
            glt_header['map info'][8] = 'South'
    glt_header_file = os.path.splitext(glt_image_file)[0]+'.hdr'
    write_envi_header(glt_header_file, glt_header)

    # Remove temporary files.
    os.remove(index_image_file)
    os.remove(index_vrt_file)
    os.remove(index_header_file)
    os.remove(igm_vrt_file)
    os.remove(tmp_glt_image_file)

    logger.info('Write the GLT to %s.' %glt_image_file)

def get_scan_vectors(imu, sensor_model):
    """ Get scan vectors.
    References:
        (1) Meyer P. (1994). A parametric approach for the geocoding of airborne
            visible/infrared imaging spectrometer (AVIRIS) data in rugged terrain.
            Remote Sensing of Environment, 49, 118-130.
    Arguments:
        imu: 2D array
            Flight IMU data, dimension: [n_lines, 3].
            Notes:
                Heading, Roll and Pitch are defined according to navigational standards.
                Column 0: Heading
                    Range: -180~180 or 0~360
                        North: 0; East: 90; West: -90 or 270
                Column 1: Roll
                    Range: -90~90
                        Right wing up: positive
                Column 2: Pitch
                    Range: -90~90
                        Aircraft nose up: positive
        sensor_model: 2D array
            Sensor model data, dimension: [n_detectors, 2].
                Notes:
                    Column 0: across-track angle component.
                    Column 1: along-track angle component.
    Returns:
        L0: 3D array
            Sensor scan vectors, dimension: [3, n_detectors, n_lines].
    """

    roll, pitch, heading = imu[:,0], imu[:,1], imu[:,2]
    n_lines = imu.shape[0]
    n_detectors = sensor_model.shape[0]

    # Navigational standard angles -> Euler angles
    heading[heading<0] = heading[heading<0]+360 # heading: -180~180 -> 0~360
    heading = 90-heading # heading angle -> euler angle
    pitch = -pitch # pitch angle -> euler angle

    # [degree] to [radian]
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    heading = np.deg2rad(heading)

    # Initialize scan vectors
    L0 = -np.ones((3, n_detectors), dtype='float32')
    L0[0,:] = np.tan(sensor_model[:,1]) # Along-track vector component
    L0[1,:] = np.tan(sensor_model[:,0])

    # Initialize rotation matrices
    R = np.zeros((3, 3, n_lines), dtype='float32')
    R[1, 1, :] = np.cos(roll)
    R[2, 1, :] = np.sin(roll)
    R[1, 2, :] = -np.sin(roll)
    R[2, 2, :] = np.cos(roll)
    R[0, 0, :] = 1

    P = np.zeros((3, 3, n_lines), dtype='float32')
    P[0, 0, :] = np.cos(pitch)
    P[2, 0, :] = -np.sin(pitch)
    P[0, 2, :] = np.sin(pitch)
    P[2, 2, :] = np.cos(pitch)
    P[1, 1, :] = 1

    H = np.zeros((3, 3, n_lines), dtype='float32')
    H[0, 0, :] = np.cos(heading)
    H[1, 0, :] = np.sin(heading)
    H[0, 1, :] = -np.sin(heading)
    H[1, 1, :] = np.cos(heading)
    H[2, 2, :] = 1

    # L0 = H*P*R*L0
    M = np.einsum('ijk,jlk->ilk', H, P)
    M = np.einsum('ijk,jlk->ilk', M, R)
    L0 = np.einsum('ijk,jl->ilk', M, L0)

    del roll, pitch, heading

    return L0

def get_xyz0_xyz1(xyz, L0, h_min, h_max):
    """ Get the starting and ending locations of ray tracing.
    References:
        (1) Schlapfer D. (2016). PARGE User Manual, Version 3.3.
    Arguments:
        xyz: 2D array
            Flight map x, y, z map coordinates, dimension: [N_lines, 3].
        L0: 3D array
            Scan vectors, dimension: [3, N_Detectors, N_Lines].
    Returns:
        xyz0, xyz1: 3D array
            Starting and ending points, dimension: [3, N_Detectors, N_Lines].
    """

    n_lines = xyz.shape[0]
    n_detectors = L0.shape[1]

    x = np.tile(xyz[:,0], (n_detectors, 1))
    y = np.tile(xyz[:,1], (n_detectors, 1))
    z = np.tile(xyz[:,2], (n_detectors, 1))

    xyz0 = np.ones((3, n_detectors, n_lines))
    xyz0[0,:,:] = (h_max-z)*L0[0,:,:]/L0[2,:,:]+x
    xyz0[1,:,:] = (h_max-z)*L0[1,:,:]/L0[2,:,:]+y
    xyz0[2,:,:] = h_max

    xyz1 = np.ones((3, n_detectors, n_lines))
    xyz1[0,:,:] = (h_min-z)*L0[0,:,:]/L0[2,:,:]+x
    xyz1[1,:,:] = (h_min-z)*L0[1,:,:]/L0[2,:,:]+y
    xyz1[2,:,:] = h_min

    del x, y, z

    return xyz0, xyz1

@guvectorize(['void(f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:], f8[:], f8[:,:,:])'],
             '(b,n,m), (b,n,m), (b,n,m), (u,v), (c) -> (b,m,n)', cache=True)
def ray_tracer_ufunc(xyz0, xyz1, L0, dem, dem_gt, output):
    """ Vectorized ray-tracing operator (numpy-style universal function).
    Arguments:
        xyz0: 3D array, shape=(3, scanlines, detectors)
            Ray-tracing starting positions for each map grid cell.
        xyz1: 3D array, shape=(3, scanlines, detectors)
            Ray-tracing ending positions for each map grid cell.
        L0: 3D array, shape=(3, scanlines, detectors)
            Scan vectors.
        dem: 2D array, shape=(scanlines, detectors)
            Digital elevation model.
        dem_gt: tuple, 6 elements
            Geotransform array containing DEM geographic parameters
             in GDAL format, i.e. as (ulx, x_res, 0, uly, 0, y_res)
        output: optional, 3D array, shape=(3, detectors, scanlines)
            Array to which outputs are written. If not passed, a new
             array is created and returned. Otherwise, the array is
             modified in place and the ufunc returns `None`.
    Returns:
        3D array if `output` is not specified, otherwise `None`.
    Notes:
        (1) Argument data types are constrained by numba signatures
             supplied to `guvectorize.` If supplied types cannot be
             coerced to required types by safe casting rules, the
             function will return an error.
    """
    
    # Geotransform: (ulx, x_res, 0, uly, 0, y_res)
    gt = np.array(dem_gt)
    dem_origin = gt[[0,3]]
    resolution = gt[[1,5]]
    
    for i in range(xyz0.shape[1]): # Iterate over detectors
        for j in range(xyz0.shape[2]): # Iterate over scanlines
            output[:,j,i] = ray_tracing(xyz0[:,i,j], xyz1[:,i,j], L0[:,i,j], dem, dem_origin, resolution)

@jit
def ray_tracing(XYZ0, XYZ1, V, DEM, DEM_X0Y0, DEM_Resolution):
    """ Implement ray-tracing to get the pixel's geo-location and elevation.
    References:
        (1) Meyer P. (1994). A parametric approach for the geocoding of airborne
            visible/infrared imaging spectrometer (AVIRIS) data in rugged terrain.
            Remote Sensing of Environment, 49, 118-130.
        (2) Amanatides J. and Woo A. (1987). A fast voxel traversal algorithm
            for ray tracing. Eurographics, 3-10.
        (3) An online example from https://www.scratchapixel.com/lessons/
            advanced-rendering/introduction-acceleration-structure/grid.
    Arguments:
        XYZ0: float list, 3 elements
            Ray-tracing starting point, [MapX0, MapY0, MapZ0].
        XYZ1: float list, 3 elements
            Ray-tracing ending point, [MapX1, MapY1, MapZ1].
        V: float list, 3 elements
            Scan vector.
        DEM: 2D array, float
            DEM image data.
        DEM_X0Y0: float list
            The upper-left corner map coordinates of the DEM.
        DEM_Resolution: float
            DEM resolution.
    Returns:
        A 3-element vector, [MapX, MapY, MapZ]: the pixel's geo-location and elevation.
    """

    if np.abs(XYZ0[0]-XYZ1[0])<1e-2 and np.abs(XYZ0[1]-XYZ1[1])<1e-2:
        return np.array([XYZ0[0], XYZ0[1], XYZ0[2]])

    cellsize_x, cellsize_y = DEM_Resolution
    y_dim, x_dim = DEM.shape
    x1, y1 = (XYZ0[0]-DEM_X0Y0[0])/cellsize_x, (XYZ0[1]-DEM_X0Y0[1])/cellsize_y
    x2, y2 = (XYZ1[0]-DEM_X0Y0[0])/cellsize_x, (XYZ1[1]-DEM_X0Y0[1])/cellsize_y

    # Get the integer and fraction parts of x1, y1, x2, y2
    x1_integer, y1_integer = int(np.floor(x1)), int(np.floor(y1))
    x1_fraction, y1_fraction = x1-x1_integer, y1-y1_integer

    # Normalized ray direction vector
    ray_direction = np.array([x2-x1, y2-y1])
    ray_direction = ray_direction/np.linalg.norm(ray_direction)

    # Initialize tx and ty
    if ray_direction[0] > 0.0:
        tx = (1.0-x1_fraction)/ray_direction[0]
        delta_tx = 1/ray_direction[0]
        x_step = 1
    elif ray_direction[0] < 0.0:
        tx = -x1_fraction/ray_direction[0]
        delta_tx = -1/ray_direction[0]
        x_step = -1
    else:
        tx = np.inf
        delta_tx = np.inf
        x_step = 0
    if ray_direction[1] > 0.0:
        ty = (1.0-y1_fraction)/ray_direction[1]
        delta_ty = 1/ray_direction[1]
        y_step = 1
    elif ray_direction[1] < 0.0:
        ty = -y1_fraction/ray_direction[1]
        delta_ty = -1/ray_direction[1]
        y_step = -1
    else:
        ty = np.inf
        delta_ty = np.inf
        y_step = 0
    the_x, the_y = x1_integer, y1_integer

    while 1:
        est_X = (DEM[the_y, the_x]-XYZ0[2])*V[0]/V[2]+XYZ0[0]
        est_Y = (DEM[the_y, the_x]-XYZ0[2])*V[1]/V[2]+XYZ0[1]
        diff_x = (est_X-DEM_X0Y0[0])/cellsize_x-the_x
        diff_y = (est_Y-DEM_X0Y0[1])/cellsize_y-the_y
        if diff_x>=-0.01 and diff_x<1.01 and diff_y>=-0.01 and diff_y<1.01:
            break
        if tx < ty:
            the_x = the_x+x_step
            tx = tx+delta_tx
        else:
            the_y = the_y+y_step
            ty = ty+delta_ty
        if (the_x<0) or (the_y <0) or (the_x>x_dim-1) or (the_y>y_dim-1):
            return np.array([np.nan, np.nan, np.nan])

    return np.array([est_X, est_Y, DEM[the_y, the_x]])
