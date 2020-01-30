#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to do spectral resampling and smoothing.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

"""Notes:
    (1) The code here is adapted from the Hytools:
        https://github.com/EnSpec/HyTools-sandbox/blob/master/hytools/preprocess/resampling.py
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def estimate_fwhms_from_waves(waves):
    """ Estimate fwhm from wavelengths.
    Arguments:
        waves: array
            Wavelengths, in nm.
    Returns:
        fwhms: array
            Full width at half maximum, in nm.
    """

    gap = 0.5*np.diff(waves)
    gap = gap[1:] + gap[:-1]
    fwhms = np.append(np.append(gap[0], gap), gap[-1])

    return fwhms

def gaussian(x, mu, fwhm):
    """ Return a gaussian distribution.
    Arguments:
        x: array
            Wavelengths along which to generate gaussian.
        mu: float
            Centre wavelength.
        fwhm: float
            Full width half maximum.
    Returns:
        Numpy array of gaussian along input range.
    """

    sigma = fwhm/(2*np.sqrt(2*np.log(2)))+1e-10

    return np.exp(-1*((x-mu)**2/(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))

def resample_spectra(spectra, src_waves, dst_waves, dst_fwhms, src_fwhms=None):
    """ Return a set of coeffiencients for spectrum resampling.
    Notes:
        (1) Given a set of source wavelengths, destination wavelengths and FWHMs, this
            function caculates the relative contribution or each input wavelength to
            the output wavelength. It assumes that output response functions follow
            a gaussian distribution.
    Arguments:
        spectra: array
            Spectra to be resampled.
        src_waves: array
            List of source wavelength centers.
        dst_waves: array
            List of destination wavelength centers.
        dst_fwhms: array
            List of destination full width half maxes.
        src_fwhms: array
            List of source full width half maxes.
    Returns:
        m x n matrix of coeffiecients, where m is the number of source wavelengths
        and n is the number of destination wavelengths.
    """

    if src_fwhms is None:
        dst_matrix = []
        for dst_wave, dst_fwhm in zip(dst_waves, dst_fwhms):
            a = gaussian(src_waves-0.5, dst_wave, dst_fwhm)
            b = gaussian(src_waves+0.5, dst_wave, dst_fwhm)
            area = (a+b)/2
            dst_matrix.append(np.divide(area,np.sum(area)+1e-10))
        coef = np.array(dst_matrix).T
    else:
        one_NM = np.arange(300, 2600)
        dst_matrix = []
        for dst_wave, dst_fwhm in zip(dst_waves, dst_fwhms):
            a = gaussian(one_NM-.5, dst_wave, dst_fwhm)
            b = gaussian(one_NM+.5, dst_wave, dst_fwhm)
            areas = (a+b)/2
            dst_matrix.append(np.divide(areas,np.sum(areas)+1e-10))
        dst_matrix = np.array(dst_matrix)

        src_matrix = []
        for src_wave,src_fwhm in zip(src_waves,src_fwhms):
            a = gaussian(one_NM-.5, src_wave, src_fwhm)
            b = gaussian(one_NM+.5, src_wave, src_fwhm)
            areas = (a+b)/2
            src_matrix.append(np.divide(areas,np.sum(areas)+1e-10))
        src_matrix = np.array(src_matrix)

        pseudo = np.linalg.pinv(src_matrix)
        coef = np.dot(dst_matrix, pseudo).T

    resampled_spectra = np.dot(spectra, coef)

    return resampled_spectra

def get_closest_wave(waves, center_wave):
    """ Get the band index whose wavelength is closest to `center_wav`.
    Arguments:
        waves: array
            Wavelength array.
        center_wave: float
            Center wavelength.
    Returns:
        Closest wavelength and its band index.
    """

    band_index = np.argmin(np.abs(np.array(waves)-center_wave))

    return waves[band_index], band_index

def continuum_removal(spectra, waves):
    """Continuum remove spectra.
    Arguments:
        spectra: 1D or 2D array
            Raw spectral data, dimension: [Bands] or [Bands, Columns].
        waves: list
            Spectral wavelengths.
    Returns:
        cont_rm_spectra: 1D or 2D array
            Continuum removed spectra, dimension: [Bands] or [Bands, Columns].
    """

    waves = np.array(waves)
    interp_spectra = (waves-waves[0])*(spectra[-1]-spectra[0])/(waves[-1]-waves[0])+spectra[0]
    cont_rmd_spectra = spectra/(interp_spectra+1e-10)

    return cont_rmd_spectra

def resample_solar_flux(solar_flux_file, sensor_waves, sensor_fwhms):
    """ Resample solar flux to sensor wavelengths.
    Arguments:
        solar_flux_file: str
            Solar flux filename.
        sensor_waves: array
            Sensor wavelengths.
        sensor_fwhms: array
            Sensor FWHMs.
    Returns:
        solar_flux: array
            Resampled solar flux.
    """

    solar_flux = np.loadtxt(solar_flux_file)
    solar_flux = resample_spectra(solar_flux[:,1], solar_flux[:,0], sensor_waves, sensor_fwhms)/10.0 # 10.0: mW/(m2 nm) -> mW/(cm2 um)

    return solar_flux
