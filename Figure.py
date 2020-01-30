#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to make figures.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, matplotlib.pyplot as plt, numpy as np
try:
    from osgeo import gdal
except:
    import gdal

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def plot_angle_geometry(angle_geometry_figure_file, sca_image_file):
    """ Plot the sun-target-view geometry in a polar coordinate system.
    Arguments:
        angle_geometry_figure_file: str
            Angle geometry figure filename.
        sca_image_file: str
            Scan angle image filename.
    """

    if os.path.exists(angle_geometry_figure_file):
        logger.info('Save the angle geometry figure to %s.' %angle_geometry_figure_file)
        return

    from ENVI import read_envi_header

    # Read sca image data.
    sca_header = read_envi_header(os.path.splitext(sca_image_file)[0]+'.hdr')
    sca_image = np.memmap(sca_image_file,
                          dtype='float32',
                          mode='r',
                          offset=0,
                          shape=(sca_header['bands'], sca_header['lines'], sca_header['samples']))

    # Scatter-plot view geometry.
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(np.deg2rad(sca_image[1,::10,::10].flatten()), sca_image[0,::10,::10].flatten(), color='green', marker='.',s=10)
    sca_image.flush()
    del sca_image
    
    # Scatter-plot sun geometry.
    ax.scatter(np.deg2rad(float(sca_header['sun azimuth'])), float(sca_header['sun zenith']), color='red', marker='*', s=500)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    _,_ = ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=('0 N', '45', '90 E', '135', '180 S', '225', '270 W', '315'))
    ax.tick_params(labelsize=20)
    plt.savefig(angle_geometry_figure_file, dpi=1000)
    plt.close()
    del sca_header, ax

    logger.info('Save the angle geometry figure to %s.' %angle_geometry_figure_file)

def plot_image_area(image_area_figure_file, dem_image_file, igm_image_file, imugps_file):
    """ Plot image area (DEM is used as the background).
    Arguments:
        image_area_figure_file: str
            Image area figure filename.
        dem_image_file: str
            DEM image filename.
        igm_image_file: str
            IGM image filename.
        imugps_file: str
            IMUGPS filename.
    """

    if os.path.exists(image_area_figure_file):
        logger.info('Save the image-area figure to %s.' %image_area_figure_file)
        return

    from ENVI import read_envi_header

    # Read DEM.
    ds = gdal.Open(dem_image_file, gdal.GA_ReadOnly)
    dem_image = ds.GetRasterBand(1).ReadAsArray()
    dem_geotransform = ds.GetGeoTransform()
    ds = None

    # Read IGM.
    igm_header = read_envi_header(os.path.splitext(igm_image_file)[0]+'.hdr')
    igm_image = np.memmap(igm_image_file,
                          dtype='float64',
                          mode='r',
                          offset=0,
                          shape=(2, igm_header['lines'], igm_header['samples']))
    cols = (igm_image[0,:,:]-dem_geotransform[0])/dem_geotransform[1]
    rows = (igm_image[1,:,:]-dem_geotransform[3])/dem_geotransform[5]
    igm_image.flush()
    del igm_image
    
    # Read IMUGPS
    imugps = np.loadtxt(imugps_file)

    # Make a plot
    plt.figure(figsize=(10, 10.0*dem_image.shape[0]/dem_image.shape[1]))
    plt.imshow(dem_image, cmap='gray', vmin=dem_image.min(), vmax=dem_image.max())
    del dem_image
    plt.plot(cols[:,0],  rows[:,0],  '-', color='lime', lw=2, label='Image Area')
    plt.plot(cols[:,-1], rows[:,-1], '-', color='lime', lw=2)
    plt.plot(cols[0,:],  rows[0,:],  '-', color='lime', lw=2)
    plt.plot(cols[-1,:], rows[-1,:], '-', color='lime', lw=2)
    cols = (imugps[:,1]-dem_geotransform[0])/dem_geotransform[1]
    rows = (imugps[:,2]-dem_geotransform[3])/dem_geotransform[5]
    plt.plot(cols, rows, '--', color='red', lw=2, label='Flight')
    plt.scatter(cols[0], rows[0], c='navy', s=20, label='Start')
    plt.scatter(cols[-1], rows[-1], c='orange', s=20, label='End')
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=5)
    plt.savefig(image_area_figure_file, bbox_inches='tight')
    plt.close()
    del cols, rows, imugps

    logger.info('Save the image-area figure to %s.' %image_area_figure_file)

def linear_percent_stretch(raw_image):
    """ Do linear percent stretch.
    References:
        (1) https://www.harrisgeospatial.com/docs/BackgroundStretchTypes.html
    Arguments:
        raw_image: 2D array
            Raw image data.
    Returns:
        stretched_image: 2D array
            Percent_stretched image.
    """

    stretched_image = np.zeros(raw_image.shape, dtype='uint8')
    low = np.percentile(raw_image, 2)
    high = np.percentile(raw_image, 98)
    index1 = raw_image<low
    index2 = raw_image>high
    stretched_image = np.floor((raw_image-low)/(high-low)*255).astype('uint8')
    stretched_image[index1] = 0
    stretched_image[index2] = 255

    return stretched_image

def make_quicklook(quicklook_figure_file, rdn_image_file, glt_image_file):
    """ Make a RGB quicklook image.
    Arguments:
        quicklook_figure_file: str
            Geo-rectified quicklook figure filename.
        rdn_image_file: str
            Radiance image filename, in BIL format.
        glt_image_file: str
            GLT image filename.
    """

    if os.path.exists(quicklook_figure_file):
        logger.info('Save the quicklook figure to %s.' %quicklook_figure_file)
        return

    from ENVI    import read_envi_header
    from Spectra import get_closest_wave

    # Read radiance image data.
    rdn_header = read_envi_header(os.path.splitext(rdn_image_file)[0]+'.hdr')
    rdn_image = np.memmap(rdn_image_file,
                          dtype='float32',
                          mode='r',
                          shape=(rdn_header['lines'],
                                 rdn_header['bands'],
                                 rdn_header['samples']))

    # Get RGB bands.
    if rdn_header['default bands'] is not None:
        rgb_bands = rdn_header['default bands']
    else:
        rgb_bands = []
        wave, _ = get_closest_wave(rdn_header['wavelength'], 450)
        if abs(wave-450)<10:
            for rgb_wave in [680, 550, 450]:
                _, band = get_closest_wave(rdn_header['wavelength'], rgb_wave)
                rgb_bands.append(band)
        else:
            for rgb_wave in [1220, 1656, 2146]:
                _, band = get_closest_wave(rdn_header['wavelength'], rgb_wave)
                rgb_bands.append(band)
        del band, wave

    # Read GLT image.
    glt_header = read_envi_header(os.path.splitext(glt_image_file)[0]+'.hdr')
    glt_image = np.memmap(glt_image_file,
                          dtype=np.int32,
                          mode='r',
                          shape=(2,
                                 glt_header['lines'],
                                 glt_header['samples']))

    # Write RGB image.
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(quicklook_figure_file,
                       glt_header['samples'],
                       glt_header['lines'],
                       3,
                       gdal.GDT_Byte)
    ds.SetGeoTransform((float(glt_header['map info'][3]),
                        float(glt_header['map info'][5]),
                        0, float(glt_header['map info'][4]),
                        0, -float(glt_header['map info'][6])))
    ds.SetProjection(glt_header['coordinate system string'])
    image = np.zeros((glt_header['lines'], glt_header['samples']), dtype='uint8')
    I,J = np.where((glt_image[0,:,:]>=0)&(glt_image[1,:,:]>=0))
    for band_index, rgb_band in enumerate(rgb_bands):
        image[:,:] = 0
        tmp_image = linear_percent_stretch(rdn_image[:,rgb_band,:])
        image[I,J] = tmp_image[glt_image[0,I,J], glt_image[1,I,J]]
        ds.GetRasterBand(band_index+1).WriteArray(image)
        del tmp_image
    glt_image.flush()
    rdn_image.flush()
    del glt_image, rdn_image
    ds = None
    del I, J, glt_header, rdn_header, image

    logger.info('Save the quicklook figure to %s.' %quicklook_figure_file)

def plot_avg_rdn(avg_rdn_figure_file, avg_rdn_file):
    """ Plot average radiance to a figure.
    Arguments:
        avg_rdn_figure_file: str
            Average radiance figure filename.
        avg_rdn_file: str
            Average radiance filename.
    """

    if os.path.exists(avg_rdn_figure_file):
        logger.info('Save the average radiance spectra figure to %s.' %avg_rdn_figure_file)
        return

    from ENVI import read_envi_header
    header = read_envi_header(os.path.splitext(avg_rdn_file)[0]+'.hdr')
    avg_rdn = np.memmap(avg_rdn_file,
                        mode='r',
                        dtype='float32',
                        shape=(header['lines'],
                               header['samples'])) # shape=(bands, samples)
    wavelength = np.array([float(v) for v in header['waves'].split(',')])

    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, avg_rdn, lw=1)
    plt.xlim(np.floor(wavelength.min()), np.ceil(wavelength.max()))
    plt.xlabel('Wavelength (nm)', fontsize=16)
    plt.ylabel(r'Radiance $(mW{\cdot}cm^{-2}{\cdot}{\mu}m^{-1}{\cdot}sr^{-1})$', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(avg_rdn_figure_file, dpi=1000)
    plt.close()
    avg_rdn.flush()
    del avg_rdn
    
    logger.info('Save the average radiance spectra figure to %s.' %avg_rdn_figure_file)

def plot_wvc_model(wvc_model_figure_file, wvc_model_file):
    """ Plot the WVC model to a figure.
    Arguments:
        wvc_model_figure_file: str
            Water vapor column model figure filename.
        wvc_model_file: str
            Water vapor column model filename.
    """

    if os.path.exists(wvc_model_figure_file):
        logger.info('Save the WVC model figure to %s.' %wvc_model_file)
        return

    import json

    # Read the WVC model.
    wvc_models = json.load(open(wvc_model_file, 'r'))
    colors = ['red', 'green', 'blue']

    # Plot the model.
    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(wvc_models.keys()):
        model = wvc_models[model_name]
        plt.plot(np.array(model['ratio'])*100, model['wvc'], '.-', ms=20, lw=2, color=colors[i], label=model_name)

    plt.xlim(0, 100)
    plt.ylim(0, 55)
    plt.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100], fontsize=20)
    plt.yticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50], fontsize=20)
    plt.xlabel('Absorption Ratio (%)', fontsize=20)
    plt.ylabel('WVC (mm)', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(wvc_model_figure_file, dpi=1000)
    plt.close()

    logger.info('Save the WVC model figure to %s.' %wvc_model_file)

def plot_smile_effect(smile_effect_at_atm_features_figure_file, smile_effect_at_atm_features_file):
    """ Plot smile effects at different atmosphere absorption features.
    Arguments:
        smile_effect_figure_file: str
            Smile effect figure filename.
        smile_effect_at_atm_features_file: str
            Smile effect at atm features filename.
    """

    if os.path.exists(smile_effect_at_atm_features_figure_file):
        logger.info('Save the smile effect at atmosphere absorption features figure to %s.' %smile_effect_at_atm_features_figure_file)
        return

    from ENVI import read_envi_header

    header = read_envi_header(os.path.splitext(smile_effect_at_atm_features_file)[0]+'.hdr')
    center_waves = [float(v) for v in header['spectral center wavelengths'].split(',')]
    fwhms = [float(v) for v in header['spectral bandwiths'].split(',')]
    shifts = np.memmap(smile_effect_at_atm_features_file,
                       dtype='float32',
                       mode='r',
                       shape=(header['bands'],
                              header['lines'],
                              header['samples']))
    sample_indices = np.arange(header['samples'])
    x_lim = [1, header['samples']]
    if x_lim[1]-x_lim[0]>1000:
        x_ticks = np.arange(x_lim[0], x_lim[1]+200, 200)
    else:
        x_ticks = np.arange(x_lim[0], x_lim[1]+50, 50)
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    for index, ax in enumerate(axs.flatten()):
        if index>=header['lines']:
            ax.axis('off')
            continue
        # Title
        ax.set_title(r'$\lambda$=%.2fnm, FWHM=%.2fnm' %(center_waves[index], fwhms[index]), fontsize=10)
        # Left y-axis
        y = np.copy(shifts[0,index,:])
        i = np.abs(y-y.mean())>3*y.std()
        y[i] = np.nan
        ax.plot(sample_indices, y, '-', color='blue', alpha=0.3, lw=1, label=r'$\Delta\lambda$')
        p = np.poly1d(np.polyfit(sample_indices[~i], y[~i], 6))
        ax.plot(sample_indices, p(sample_indices), '-', color='blue', lw=4)
        y_lim = ax.get_ylim()
        y_lim = [np.floor(y_lim[0]/0.5)*0.5, np.ceil(y_lim[1]/0.5)*0.5]
        if y_lim[1]-y_lim[0]<=1.0:
            y_ticks = np.arange(y_lim[0], y_lim[1]+0.25, 0.25)
        elif y_lim[1]-y_lim[0]<=3.0:
            y_ticks = np.arange(y_lim[0], y_lim[1]+0.50, 0.50)
        else:
            y_ticks = np.arange(y_lim[0], y_lim[1]+1.00, 1.00)
        y_lim = [y_ticks[0].min(), y_ticks[-1].max()]
        ax.set_yticks(y_ticks)
        y_ticklabels = ['%.2f' %v for v in y_ticks]
        ax.set_yticklabels(y_ticklabels, fontsize=10, color='blue')
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        # Right y-axis
        twin_ax = ax.twinx()
        y = np.copy(shifts[1,index,:])
        i = np.abs(y-y.mean())>3*y.std()
        y[i] = np.nan
        twin_ax.plot(sample_indices, y, '-', color='red', alpha=0.3, lw=1, label=r'$\Delta$FWHM')
        p = np.poly1d(np.polyfit(sample_indices[~i], y[~i], 6))
        twin_ax.plot(sample_indices, p(sample_indices), '-', color='red', lw=4)
        y_lim = twin_ax.get_ylim()
        y_lim = [np.floor(y_lim[0]/0.5)*0.5, np.ceil(y_lim[1]/0.5)*0.5]
        if y_lim[1]-y_lim[0]<=1.0:
            y_ticks = np.arange(y_lim[0], y_lim[1]+0.01, 0.25)
        elif y_lim[1]-y_lim[0]<=3.0:
            y_ticks = np.arange(y_lim[0], y_lim[1]+0.50, 0.50)
        else:
            y_ticks = np.arange(y_lim[0], y_lim[1]+1.00, 1.00)
        y_lim = [y_ticks[0].min(), y_ticks[-1].max()]
        twin_ax.set_yticks(y_ticks)
        y_ticklabels = ['%.2f' %v for v in y_ticks]
        twin_ax.set_yticklabels(y_ticklabels, fontsize=10, color='red')
        twin_ax.set_xlim(x_lim)
        twin_ax.set_ylim(y_lim)
        twin_ax.set_xticks(x_ticks)
        twin_ax.set_xticklabels(x_ticks, fontsize=10)
        # X-axis
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=10)
        ax.set_xlabel('Across-track Pixel', fontsize=10)
        if index==0:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = twin_ax.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2)
    fig.savefig(smile_effect_at_atm_features_figure_file, dpi=600, bbox_inches='tight')
    plt.close()
    del fig, ax, axs, header
    shifts.flush()
    del shifts
    
    logger.info('Save smile effect at atmosphere absorption features figure to %s.' %smile_effect_at_atm_features_figure_file)

def plot_wvc_histogram(wvc_histogram_figure_file, water_vapor_column_image_file):
    """ Plot water vapor column histogram.
    wvc_histogram_figure_file: str
        Water vapor column histogram figure filename.
    water_vapor_column_image_file: str
        Water vapor column image filename.
    """

    if os.path.exists(wvc_histogram_figure_file):
        logger.info('Save water vapor column histogram figure to %s.' %wvc_histogram_figure_file)
        return

    from ENVI import read_envi_header

    # Read water vapor column image
    wvc_header = read_envi_header(os.path.splitext(water_vapor_column_image_file)[0]+'.hdr')
    wvc_image = np.memmap(water_vapor_column_image_file,
                          dtype='float32',
                          mode='r',
                          shape=(wvc_header['lines'], wvc_header['samples']))

    # Plot water vapor column histogram
    wvc_bins = np.arange(0, 51, 1)
    freq = []
    for bin_index in range(len(wvc_bins)-1):
        index = (wvc_image>=wvc_bins[bin_index])&(wvc_image<wvc_bins[bin_index+1])
        freq.append(np.sum(index)/wvc_image.size*100)
    freq = np.array(freq)
    freq_max = 100
    plt.figure(figsize=(10,6))
    plt.bar(wvc_bins[:-1], freq, width=1, color='darkgreen', edgecolor='black', linewidth=1)
    plt.vlines([wvc_image.mean()], 0, freq_max, color='darkred', lw=2, linestyles='solid', label=r'WVC$_{Mean}$')
    plt.vlines([wvc_image.mean()-2*wvc_image.std()], 0, freq_max, color='darkred', lw=2, linestyles='dotted', label=r'WVC$_{Mean}$-2WVC$_{SD}$')
    plt.vlines([wvc_image.mean()+2*wvc_image.std()], 0, freq_max, color='darkred', lw=2, linestyles='dashed', label=r'WVC$_{Mean}$+2WVC$_{SD}$')
    plt.xticks(ticks=np.arange(0, 51, 5), labels=np.arange(0, 51, 5), fontsize=20)
    plt.yticks(ticks=np.arange(0,101,10), labels=np.arange(0,101,10), fontsize=20)
    plt.xlabel('Water Vapor Column (mm)', fontsize=20)
    plt.ylabel('Relative Frequency (%)', fontsize=20)
    plt.xlim(0, 50)
    plt.ylim(0, freq_max)
    plt.legend(fontsize=20)
    plt.savefig(wvc_histogram_figure_file, dpi=1000)
    plt.close()
    
    wvc_image.flush()
    del wvc_image
    
    logger.info('Save water vapor column histogram figure to %s.' %wvc_histogram_figure_file)
