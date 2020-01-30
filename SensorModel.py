#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create a sensor model.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""
import logging, os, numpy as np
logger = logging.getLogger(__name__)

def make_sensor_model(sensor_model_file, fov, ifov, samples, if_rotated):
    """ Generate a sensor model.
    Arguments:
        sensor_model_file: str
            The sensor model filename.
        fov: float
            Sensor fov [deg].
        ifov: float
            Sensor instaneous fov [mrad].
        samples: int
            Image columns.
        if_rotated: bool
            If the sensor is 180 degree rotated.
    """

    if os.path.exists(sensor_model_file):
        logger.info('Write the sensor model to %s.' %sensor_model_file)
        return

    sensor_model = np.zeros((samples, 3))
    sensor_model[:,0] = np.arange(samples)

    fov = np.deg2rad(fov)
    if if_rotated:
        xs = np.linspace(np.tan(fov/2), -np.tan(fov/2), num=samples)
    else:
        xs = np.linspace(-np.tan(fov/2), np.tan(fov/2), num=samples)

    sensor_model[:,1] = np.arctan(xs)
    sensor_model[:,2] = ifov/1000
    np.savetxt(sensor_model_file,
               sensor_model,
               header='pixel    vinkelx    vinkely',
               fmt='%d    %.10f    %.10f')
    del sensor_model, xs

    logger.info('Write the sensor model to %s.' %sensor_model_file)

def determine_if_rotated(imu_gps_file):
    """ Determine if the sensor is 180 degree rotated.
    Arguments:
        imu_gps_file: str
            IMUGPS file
    Returns:
        True or False: bool
            Whether the sensor is 180 degree rotated.
    """

    imugps = np.loadtxt(imu_gps_file)
    flag = np.sign((imugps[-1,3] - imugps[0,3])*imugps[0,6])

    if flag == -1:
        logger.info('The sensor is 180 degree rotated.')
        return True
    else:
        logger.info('The sensor is not 180 degree rotated.')
        return False