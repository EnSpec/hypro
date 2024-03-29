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

"""Functions for processing HySpex IMU & GPS data."""

import logging
import os

import numpy as np

from osgeo import osr

logger = logging.getLogger(__name__)


def prepare_imugps_Hyspex(processed_imugps_file, raw_imugps_file, boresight_offsets, map_crs, boresight_options):
    """Prepare HySpex IMU and GPS data.
    
    Parameters
    ----------
    processed_imugps_file : str
        Processed IMU & GPS filename.
    raw_imugps_file : str
        Raw IMU & GPS filename.
    boresight_offsets : list of float
        Boresight offsets, given as ``[roll, pitch, heading, altitude]``.
    boresight_options : list of bool
        Boresight offset options, true or false.
    map_crs : osr.SpatialReference
        Map coordinate system.
    """
    
    if os.path.exists(processed_imugps_file):
        logger.info('Write the IMU and GPS data to %s.' % processed_imugps_file)
        return
    
    from hypro.Geography import define_wgs84_crs, get_grid_convergence
    
    # Load raw IMU/GPS data.
    raw_imugps = np.loadtxt(raw_imugps_file)
    
    # Apply boresight offsets.
    processed_imugps = np.zeros((raw_imugps.shape[0], 15))
    
    # Scan line index.
    processed_imugps[:, 0] = raw_imugps[:, 0]
    
    # GPS and IMU.
    wgs84_crs = define_wgs84_crs()
    transform = osr.CoordinateTransformation(wgs84_crs, map_crs)
    xyz = np.array(transform.TransformPoints(raw_imugps[:, 1:3]))
    processed_imugps[:, 1] = xyz[:, 0]  # Flight easting
    processed_imugps[:, 2] = xyz[:, 1]  # Flight northing
    processed_imugps[:, 3] = raw_imugps[:, 3]  # Flight altitude
    processed_imugps[:, 4] = raw_imugps[:, 4]  # Roll
    processed_imugps[:, 5] = raw_imugps[:, 5]  # Pitch
    processed_imugps[:, 6] = raw_imugps[:, 6]  # Heading
    del transform, xyz
    
    # Boresight offsets.
    for i in range(len(boresight_options)):
        if boresight_options[i]:
            processed_imugps[:, 7+i] = boresight_offsets[i]
    
    # Grid convergence.
    grid_convergence = get_grid_convergence(raw_imugps[:, 1], raw_imugps[:, 2], map_crs)
    processed_imugps[:, 11] = grid_convergence
    
    # Longitude and latitude.
    processed_imugps[:, 12] = raw_imugps[:, 1]  # Longitude
    processed_imugps[:, 13] = raw_imugps[:, 2]  # Latitude
    
    # Timestamp.
    processed_imugps[:, 14] = raw_imugps[:, 7]
    
    # Save the new IMU/GPS data.
    header = ['Map coordinate system = %s' % (map_crs.ExportToWkt()),
              'Index    ' +
              'Map_X    Map_Y    Map_Z    Roll    Pitch    Heading    ' +
              'Roll_Offset    Pitch_Offset    Heading_Offset    Altitude_Offset    Grid_Convergence    ' +
              'Longitude    Latitude    ' +
              'Timestamp']
    np.savetxt(processed_imugps_file,
               processed_imugps,
               header='\n'.join(header),
               fmt='%d    %.3f    %.3f    %.3f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.10f    %.5f')
    logger.info('Write the IMU and GPS data to %s.' % processed_imugps_file)
