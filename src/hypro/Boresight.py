#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020
# Nanfeng Liu <nliu58@wisc.edu>
# Philip Townsend <ptownsend@wisc.edu>
#
# Environmental Spectroscopy Laboratory
# Department of Forest & Wildlife Ecology
# University of Wisconsin – Madison
#
# Licensed under GNU GPLv3
# See `./LICENSE.txt` for complete terms

"""Functions for sensor boresighting."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def boresight_calibration(boresight_file, gcp_file, imugps_file, sensor_model_file, dem_image_file, boresight_options):
    """Estimate sensor boresight offsets.
    
    Parameters
    ----------
    boresight_file : str
        Boresight filename.
    gcp_file : str
        Ground control points filename.
    imugps_file : str
        The IMU & GPS filename.
    sensor_model_file : str
        The sensor model filename.
    dem_image_file : str
        The DEM image filename.
    boresight_options : list of bool
        Boresight offset options, true or false.
    """
    
    if os.path.exists(boresight_file):
        logger.info('Write boresight data to %s.' % boresight_file)
        return
    
    from osgeo import gdal, osr
    from scipy import optimize
    
    from hypro.Geography import define_wgs84_crs
    
    # Read IMU and GPS data.
    imugps = np.loadtxt(imugps_file)  # ID, X, Y, Z, R, P, H, Timestamp, Longitude, Latitude, Grid_Convergence, Roll...
    
    # Read sensor model data.
    sensor_model = np.loadtxt(sensor_model_file, skiprows=1)[:, 1:]
    
    # Read DEM data.
    ds = gdal.Open(dem_image_file, gdal.GA_ReadOnly)
    dem_image = ds.GetRasterBand(1).ReadAsArray()
    dem_geotransform = ds.GetGeoTransform()
    dem_prj = ds.GetProjection()
    ds = None
    
    # Read GCP data.
    gcp_data = np.loadtxt(gcp_file, comments=';')  # Longitude, Latitude, Image Column, Image Row
    
    # Convert GCP longitudes and latitudes to map x and y.
    wgs84_crs = define_wgs84_crs()
    map_crs = osr.SpatialReference()
    map_crs.ImportFromWkt(dem_prj)
    transform = osr.CoordinateTransformation(wgs84_crs, map_crs)
    gcp_xyz = np.array(transform.TransformPoints(gcp_data[:, :2]))
    del wgs84_crs, transform
    
    # Interpolate flight IMU and (x, y, z).
    lines = np.arange(imugps.shape[0])
    flight_xyz = np.zeros((gcp_data.shape[0], 3))
    flight_xyz[:, 0] = np.interp(gcp_data[:, 3] - 1, lines, imugps[:, 1])
    flight_xyz[:, 1] = np.interp(gcp_data[:, 3] - 1, lines, imugps[:, 2])
    flight_xyz[:, 2] = np.interp(gcp_data[:, 3] - 1, lines, imugps[:, 3])
    
    flight_imu = np.zeros((gcp_data.shape[0], 3))
    flight_imu[:, 0] = np.interp(gcp_data[:, 3] - 1, lines, imugps[:, 4])
    flight_imu[:, 1] = np.interp(gcp_data[:, 3] - 1, lines, imugps[:, 5])
    flight_imu[:, 2] = np.interp(gcp_data[:, 3] - 1, lines, imugps[:, 6])
    
    # Apply grid convergence.
    flight_imu[:, 2] = flight_imu[:, 2] - np.interp(gcp_data[:, 3] - 1, lines, imugps[:, 11])
    del lines
    
    # Interpolate sensor model.
    samples = np.arange(sensor_model.shape[0])
    flight_sensor_model = np.zeros((gcp_data.shape[0], 2))
    flight_sensor_model[:, 0] = np.interp(gcp_data[:, 2] - 1, samples, sensor_model[:, 0])
    flight_sensor_model[:, 1] = np.interp(gcp_data[:, 2] - 1, samples, sensor_model[:, 1])
    del samples
    
    # Optimize.
    p = optimize.minimize(cost_fun,
                          [0, 0, 0, 0],
                          method='L-BFGS-B',
                          args=(flight_xyz, flight_imu, flight_sensor_model,
                                dem_image,
                                [dem_geotransform[0], dem_geotransform[3]],
                                [dem_geotransform[1], dem_geotransform[5]],
                                gcp_xyz,
                                boresight_options))
    logger.info('Roll, pitch, heading and altitude offsets: %s, %s, %s, %s' % (p.x[0], p.x[1], p.x[2], p.x[3]))
    
    # Save offsets.
    imugps[:, 7] = p.x[0]
    imugps[:, 8] = p.x[1]
    imugps[:, 9] = p.x[2]
    imugps[:, 10] = p.x[3]
    header = ['Map coordinate system = %s' % (map_crs.ExportToWkt()),
              'Index    Map_X    Map_Y    Map_Z    Roll    Pitch    Heading    ' +
              'Roll_Offset    Pitch_Offset    Heading_Offset    Altitude_Offset    Grid_Convergence    ' +
              'Longitude    Latitude    Timestamp']
    np.savetxt(imugps_file,
               imugps,
               header='\n'.join(header),
               fmt='%d    %.3f    %.3f    %.3f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.5f')
    
    # Estimate geometric correction accuracy.
    est_gcp_xyz = estimate_gcp_xyz(p.x, flight_xyz, flight_imu, flight_sensor_model, dem_image, [dem_geotransform[0], dem_geotransform[3]], [dem_geotransform[1], dem_geotransform[5]], boresight_options)
    boresight_data = np.zeros((gcp_data.shape[0], 10))
    boresight_data[:, 0] = np.arange(gcp_data.shape[0])
    boresight_data[:, 1] = gcp_data[:, 2]
    boresight_data[:, 2] = gcp_data[:, 3]
    boresight_data[:, 3] = gcp_xyz[:, 0]
    boresight_data[:, 4] = gcp_xyz[:, 1]
    boresight_data[:, 5] = est_gcp_xyz[:, 0]
    boresight_data[:, 6] = est_gcp_xyz[:, 1]
    boresight_data[:, 7] = gcp_xyz[:, 0] - est_gcp_xyz[:, 0]
    boresight_data[:, 8] = gcp_xyz[:, 1] - est_gcp_xyz[:, 1]
    boresight_data[:, 9] = np.sqrt(boresight_data[:, 7]**2 + boresight_data[:, 8]**2)
    header = ['Map coordinate system = %s' % (map_crs.ExportToWkt()),
              'Roll offset = %s' % (p.x[0]),
              'Pitch offset = %s' % (p.x[1]),
              'Heading offset = %s' % (p.x[2]),
              'Altitude offset = %s' % (p.x[3]),
              'Min RMS = %.4f' % (boresight_data[:, 9].min()),
              'Mean RMS = %.4f' % (boresight_data[:, 9].mean()),
              'Max RMS = %.4f' % (boresight_data[:, 9].max()),
              'Index    Image_X    Image_Y    Map_X    Map_Y    Predict_X    Predict_Y    Error_X    Error_Y    RMS']
    np.savetxt(boresight_file,
               boresight_data,
               header='\n'.join(header),
               fmt='%d    %.3f    %.3f    %.3f    %.3f    %.3f    %.3f    %.3f    %.3f    %.3f')
    logger.info('Boresight accuracy (min, mean, max): %.2f, %.2f, %2.f.' % (boresight_data[:, 9].min(), boresight_data[:, 9].mean(), boresight_data[:, 9].max()))
    del boresight_data
    
    logger.info('Write boresight data to %s.' % boresight_file)


def cost_fun(boresight_offsets, flight_xyz, flight_imu, flight_sensor_model, dem_image, dem_ulxy, dem_pixel_size, gcp_xyz, boresight_options):
    """Cost function for sensor boresighting.
    
    Parameters
    ----------
    boresight_offsets : list of float
        Boresight offsets.
    flight_xyz : ndarray, 2D
        Imaging platform positions in map coordinates (x, y, & z), array with shape
        ``(n_gcps, 3)``.
    flight_imu : ndarray, 2D
        Imaging platform navigational attitudes (roll, pitch & heading), array with
        shape ``(n_gcps, 3)``.
    flight_sensor_model : ndarray, 2D
        Flight sensor model data, array with shape ``(n_gcps, 2)``.
    dem_image : ndarray, 2D
        DEM data.
    dem_ulxy : list of float
        Map coordinates of DEM upper-left corner.
    dem_pixel_size : list of float
        DEM pixel size.
    gcp_xyz : ndarray, 2D
        GPC map coordinates, array with shape ``(n_gcps, 3)``.
    boresight_options : list of bool
        Boresight offset options, true or false.
    
    Notes
    -----
    Returns the sum of spatial errors in the (x, y) plane (i.e. absolute horizontal
    distances between known & predicted control point locations) over all GCPs.
    """
    
    # Estimate GCP map coordinates.
    est_gcp_xyz = estimate_gcp_xyz(boresight_offsets, flight_xyz, flight_imu, flight_sensor_model, dem_image, dem_ulxy, dem_pixel_size, boresight_options)
    
    # Calculate cost.
    cost = np.sum((est_gcp_xyz[:, :2] - gcp_xyz[:, :2])**2)
    
    return cost


def estimate_gcp_xyz(boresight_offsets, flight_xyz, flight_imu, flight_sensor_model, dem_image, dem_ulxy, dem_pixel_size, boresight_options):
    """Estimate GCP map coordinates.
    
    Parameters
    ----------
    boresight_offsets : list of float
        Boresight offsets.
    flight_xyz : ndarray, 2D
        Imaging platform positions in map coordinates (x, y & z), array with shape
        ``(n_gcps, 3)``.
    flight_imu : ndarray, 2D
        Imaging platform navigational attitudes (roll, pitch & heading), array with
        shape ``(n_gcps, 3)``.
    flight_sensor_model : ndarray, 2D
        Flight sensor model data, array with shape ``(n_gcps, 2)``.
    dem_image : ndarray, 2D
        DEM data.
    dem_ulxy : list of float
        Map coordinates of DEM upper-left corner.
    dem_pixel_size : list of float
        DEM pixel size.
    boresight_options : list of bool
        Boresight offset options, true or false.
    
    Returns
    -------
    gcp_xyz : ndarray, 2D
        Estimated GCP coordinates in map space (x, y & z), array with shape
        ``(n_gcps, 3)``.
    """
    
    from hypro.GeoReferencing import ray_tracing
    
    # Initialize.
    n_gcps = flight_xyz.shape[0]
    roll = flight_imu[:, 0] + boresight_offsets[0] if boresight_options[0] else flight_imu[:, 0]
    pitch = flight_imu[:, 1] + boresight_offsets[1] if boresight_options[1] else flight_imu[:, 1]
    heading = flight_imu[:, 2] + boresight_offsets[2] if boresight_options[2] else flight_imu[:, 2]
    x = flight_xyz[:, 0]
    y = flight_xyz[:, 1]
    z = flight_xyz[:, 2] + boresight_offsets[3] if boresight_options[3] else flight_xyz[:, 2]
    
    # Get scan vectors.
    heading[heading < 0] = heading[heading < 0] + 360  # heading: -180~180 -> 0~360
    heading = 90 - heading  # heading angle -> euler angle
    pitch = -pitch  # pitch angle -> euler angle
    
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    heading = np.deg2rad(heading)
    
    L0 = -np.ones((3, 1, n_gcps))
    L0[0, 0, :] = np.tan(flight_sensor_model[:, 1])  # Along-track vector component
    L0[1, 0, :] = np.tan(flight_sensor_model[:, 0])
    
    R = np.zeros((3, 3, n_gcps))
    R[1, 1, :] = np.cos(roll)
    R[2, 1, :] = np.sin(roll)
    R[1, 2, :] = -np.sin(roll)
    R[2, 2, :] = np.cos(roll)
    R[0, 0, :] = 1
    
    P = np.zeros((3, 3, n_gcps))
    P[0, 0, :] = np.cos(pitch)
    P[2, 0, :] = -np.sin(pitch)
    P[0, 2, :] = np.sin(pitch)
    P[2, 2, :] = np.cos(pitch)
    P[1, 1, :] = 1
    
    H = np.zeros((3, 3, n_gcps))
    H[0, 0, :] = np.cos(heading)
    H[1, 0, :] = np.sin(heading)
    H[0, 1, :] = -np.sin(heading)
    H[1, 1, :] = np.cos(heading)
    H[2, 2, :] = 1
    
    M = np.einsum('ijk,jlk->ilk', H, P)
    M = np.einsum('ijk,jlk->ilk', M, R)
    L0 = np.einsum('ijk,jlk->ilk', M, L0)[:, 0, :]  # shape=(3, n_gcps)
    
    del roll, pitch, heading, R, P, H, M
    
    # Get start and end points of ray tracing.
    h_min = dem_image.min()
    h_max = dem_image.max()
    xyz0 = np.ones(flight_xyz.shape)  # shape=(3, n_gcps)
    xyz0[:, 0] = (h_max - z)*L0[0, :]/L0[2, :] + x
    xyz0[:, 1] = (h_max - z)*L0[1, :]/L0[2, :] + y
    xyz0[:, 2] = h_max
    xyz1 = np.ones(flight_xyz.shape)  # shape=(3, n_gcps)
    xyz1[:, 0] = (h_min - z)*L0[0, :]/L0[2, :] + x
    xyz1[:, 1] = (h_min - z)*L0[1, :]/L0[2, :] + y
    xyz1[:, 2] = h_min
    del h_min, h_max
    
    # Ray tracing.
    gcp_xyz = np.zeros((n_gcps, 3))
    for i in range(n_gcps):
        gcp_xyz[i, 0], gcp_xyz[i, 1], gcp_xyz[i, 2] = ray_tracing(xyz0[i, :], xyz1[i, :], L0[:, i], dem_image, dem_ulxy, dem_pixel_size)
    
    return gcp_xyz
