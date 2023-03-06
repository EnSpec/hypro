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

"""Functions for working with spectral data.

Notes
-----
The code here is adapted from HyTools; see [#ht-resampling]_.

References
----------
.. [#ht-resampling] https://github.com/EnSpec/HyTools-sandbox/blob/master/hytools/preprocess/resampling.py
"""

import pkgutil
import warnings

from io import BytesIO

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


def estimate_fwhms_from_waves(waves):
    """Estimate FWHM from wavelengths.
    
    Parameters
    ----------
    waves : ndarray
        Wavelengths, in nm.
    
    Returns
    -------
    fwhms : ndarray
        Full widths at half maximum, in nm.
    """
    
    gap = 0.5*np.diff(waves)
    gap = gap[1:] + gap[:-1]
    fwhms = np.append(np.append(gap[0], gap), gap[-1])
    
    return fwhms


def gaussian(x, mu, fwhm):
    """Return a Gaussian distribution.
    
    Parameters
    ----------
    x : ndarray
        Wavelengths along which to generate Gaussian.
    mu : float
        Center wavelength.
    fwhm : float
        Full width at half maximum.
    
    Returns
    -------
    array
        Numpy array of Gaussian along input range.
    """
    
    sigma = fwhm/(2*np.sqrt(2*np.log(2))) + 1e-10
    
    return np.exp(-1*((x - mu)**2/(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))


def resample_spectra(spectra, src_waves, dst_waves, dst_fwhms, src_fwhms=None):
    """Resample spectral data to new band central wavelengths & bandwidths.
    
    Parameters
    ----------
    spectra : ndarray
        Spectra to be resampled.
    src_waves : ndarray
        List of source wavelength centers.
    dst_waves : ndarray
        List of destination wavelength centers.
    dst_fwhms : ndarray
        List of destination full widths at half maximum.
    src_fwhms : ndarray
        List of source full widths at half maximum.
    
    Returns
    -------
    resampled_spectra : ndarray
        Resampled spectral data.
    
    Notes
    -----
    Given a set of source wavelengths, destination wavelengths and FWHMs, this function
    calculates the relative contribution or each input wavelength to the output wavelength.
    It assumes that output response functions follow a Gaussian distribution.
    """
    
    if src_fwhms is None:
        dst_matrix = []
        for dst_wave, dst_fwhm in zip(dst_waves, dst_fwhms):
            a = gaussian(src_waves - 0.5, dst_wave, dst_fwhm)
            b = gaussian(src_waves + 0.5, dst_wave, dst_fwhm)
            area = (a + b)/2
            dst_matrix.append(np.divide(area, np.sum(area) + 1e-10))
        coef = np.array(dst_matrix).T
    else:
        one_NM = np.arange(300, 2600)
        dst_matrix = []
        for dst_wave, dst_fwhm in zip(dst_waves, dst_fwhms):
            a = gaussian(one_NM - .5, dst_wave, dst_fwhm)
            b = gaussian(one_NM + .5, dst_wave, dst_fwhm)
            areas = (a + b)/2
            dst_matrix.append(np.divide(areas, np.sum(areas) + 1e-10))
        dst_matrix = np.array(dst_matrix)
        
        src_matrix = []
        for src_wave, src_fwhm in zip(src_waves, src_fwhms):
            a = gaussian(one_NM - .5, src_wave, src_fwhm)
            b = gaussian(one_NM + .5, src_wave, src_fwhm)
            areas = (a + b)/2
            src_matrix.append(np.divide(areas, np.sum(areas) + 1e-10))
        src_matrix = np.array(src_matrix)
        
        pseudo = np.linalg.pinv(src_matrix)
        coef = np.dot(dst_matrix, pseudo).T
    
    resampled_spectra = np.dot(spectra, coef)
    
    return resampled_spectra


def get_closest_wave(waves, center_wave):
    """Get the band index whose wavelength is closest to ``center_wave``.
    
    Parameters
    ----------
    waves : ndarray
        Wavelength array.
    center_wave : float
        Center wavelength.
    
    Returns
    -------
    tuple
        Closest wavelength and its band index.
    """
    
    band_index = np.argmin(np.abs(np.array(waves) - center_wave))
    
    return waves[band_index], band_index


def continuum_removal(spectra, waves):
    """Continuum remove spectra.
    
    Parameters
    ----------
    spectra : ndarray, 1D or 2D
        Raw spectral data, dimension: [Bands] or [Bands, Columns].
    waves : list
        Spectral wavelengths.
    
    Returns
    -------
    cont_rmd_spectra : ndarray, 1D or 2D
        Continuum-removed spectra, dimension: [Bands] or [Bands, Columns].
    """
    
    waves = np.array(waves)
    interp_spectra = (waves - waves[0])*(spectra[-1] - spectra[0])/(waves[-1] - waves[0]) + spectra[0]
    cont_rmd_spectra = spectra/(interp_spectra + 1e-10)
    
    return cont_rmd_spectra


def resample_solar_flux(sensor_waves, sensor_fwhms, file=None):
    """Resample solar flux to sensor wavelengths.
    
    Parameters
    ----------
    sensor_waves : ndarray
        Sensor wavelengths.
    sensor_fwhms : ndarray
        Sensor FWHMs.
    file : str
        Solar flux filename.
    
    Returns
    -------
    solar_flux : ndarray
        Resampled solar flux.
    """
    
    solar_flux_file = file or BytesIO(pkgutil.get_data(__package__, 'data/solar/irradiance_kurucz1992.dat'))
    solar_flux = np.loadtxt(solar_flux_file)
    solar_flux = resample_spectra(solar_flux[:, 1], solar_flux[:, 0], sensor_waves, sensor_fwhms)/10.0  # 10.0: mW/(m2 nm) -> mW/(cm2 um)
    
    return solar_flux
