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

"""Functions for generating pushbroom sensor models."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def make_sensor_model(sensor_model_file, fov, ifov, samples, if_rotated):
    """Generate a sensor model.
    
    Parameters
    ----------
    sensor_model_file : str
        The sensor model filename.
    fov : float
        Total cross-track angular field of view (FoV) of the sensor, units=[deg].
    ifov : float
        Along-track instantaneous FoV of the sensor, units=[mrad].
    samples : int
        Image columns.
    if_rotated : bool
        If the sensor is 180 degree rotated.
    """
    
    if os.path.exists(sensor_model_file):
        logger.info('Write the sensor model to %s.' % sensor_model_file)
        return
    
    sensor_model = np.zeros((samples, 3))
    sensor_model[:, 0] = np.arange(samples)
    
    fov = np.deg2rad(fov)
    if if_rotated:
        xs = np.linspace(np.tan(fov/2), -np.tan(fov/2), num=samples)
    else:
        xs = np.linspace(-np.tan(fov/2), np.tan(fov/2), num=samples)
    
    sensor_model[:, 1] = np.arctan(xs)
    sensor_model[:, 2] = ifov/1000
    np.savetxt(sensor_model_file,
               sensor_model,
               header='pixel    vinkelx    vinkely',
               fmt='%d    %.10f    %.10f')
    del sensor_model, xs
    
    logger.info('Write the sensor model to %s.' % sensor_model_file)


def determine_if_rotated(imu_gps_file):
    """Determine if the sensor is 180 degree rotated.
    
    Parameters
    ----------
    imu_gps_file : str
        IMU & GPS file
    
    Returns
    -------
    bool
        Whether the sensor is 180 degree rotated.
    """
    
    imugps = np.loadtxt(imu_gps_file)
    flag = np.sign((imugps[-1, 3] - imugps[0, 3])*imugps[0, 6])
    
    if flag == -1:
        logger.info('The sensor is 180 degree rotated.')
        return True
    else:
        logger.info('The sensor is not 180 degree rotated.')
        return False
