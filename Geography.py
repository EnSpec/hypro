#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to process map coordinate systems.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import gdal, os, osr, numpy as np

def get_utm_zone(lon):
    """ Calculate UTM zone.
    Arguments:
        lon: float
            Longitude, in degrees. West: negative, East: positive.
    Returns:
        zone: int
            UTM zone number.
    """

    zone = int(1+(lon+180.0)/6.0)

    return zone

def is_northern(lat):
    """ Determine if it is northern hemisphere.
    Arguments:
        lat: float
            Latitude, in degrees. Northern: positive, Southern: negative.
    Returns:
        1: northern, 0: southern.
    """

    if lat < 0.0:
        return 0
    else:
        return 1

def define_utm_crs(lon, lat):
    """ Define a UTM map coordinate system.
    Arguments:
        lon: float
            Longitude, in degrees. West: negative, East: positive.
        lat: float
            Latitude, in degrees. Northern: positive, Southern: negative.
    Returns:
        crs: osr.SpatialReference object
            UTM map coordinate system.
    """

    crs = osr.SpatialReference()
    crs.SetWellKnownGeogCS("WGS84")
    zone = get_utm_zone(lon)
    crs.SetUTM(int(zone), int(is_northern(lat)))

    return crs

def define_wgs84_crs():
    """ Define a WGS84 map coordinate system.
    Returns:
        crs: osr.SpatialReference object
            WGS84 map coordinate system.
    """

    crs = osr.SpatialReference()
    crs.SetWellKnownGeogCS("WGS84")

    return crs

def get_raster_crs(file):
    """ Get the map coordinate system of a raster image.
    Arguments:
        file: str
            Georeferenced image filename.
    Returns:
        crs: osr object
            Map coordinate system.
    """

    ds = gdal.Open(file, gdal.GA_ReadOnly)
    prj = ds.GetProjection()
    ds = None

    crs = osr.SpatialReference()
    crs.ImportFromWkt(prj)

    return crs

def get_grid_convergence(lon, lat, map_crs):
    """ Get grid convergence angles.
    Arguments:
        lon: list of floats
            Longitude. West: negative; East: positive.
        lat: list of floats
            Latitde. North: positive; South: negative.
    Reuturns:
        grid_convergence: array of floats
            Grid convergence in degrees.
    """

    lon, lat = np.array(lon), np.array(lat)
    if map_crs.GetAttrValue('PROJECTION').lower() == 'transverse_mercator':
        lon0 = map_crs.GetProjParm('central_meridian')
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)
        lon0 = np.deg2rad(lon0)
        grid_convergence = np.arctan(np.tan(lon-lon0)*np.sin(lat))
        grid_convergence = np.rad2deg(grid_convergence)
    else:
        delta_lat = 1e-4
        lon_lat_0 = np.array([lon, lat]).T
        lon_lat_1 = np.array([lon, lat+delta_lat]).T
        wgs84_crs = define_wgs84_crs()
        transform = osr.CoordinateTransformation(wgs84_crs, map_crs)
        xy0 = np.array(transform.TransformPoints(lon_lat_0))
        xy1 = np.array(transform.TransformPoints(lon_lat_1))
        dx = xy1[:,0]-xy0[:,0]
        dy = xy1[:,1]-xy0[:,1]
        grid_convergence = np.abs(np.rad2deg(np.arcsin(dx/np.sqrt(dx**2+dy**2))))
        index = dx*dy>0
        grid_convergence[index] = -grid_convergence[index]

    return grid_convergence

def get_map_crs(dem, longitude, latitude):
    """ Get map coordinate system.
    Notes:
        If `dem` is a file, the map coordinate system should be
        the same as that of the dem file; otherwise define a UTM coordinate system
        based on the longitude and latitude.
    Arguments:
        dem: str or float
            DEM image filename, or user-specified DEM value.
        longitude, latitude: float
            Longitude and latitude.
    Returns:
        map_crs: osr object
            Map coordinate system.
    """

    if os.path.isfile(dem):
        map_crs = get_raster_crs(dem)
    else:
        map_crs = define_utm_crs(longitude, latitude)

    return map_crs

def get_sun_angles(longitude, latitude, utc_time):
    """ Calculate the Sun's position.
    References:
        (1) Manuel Blanco-Muriel, et al. (2001). Computing the solar vector. Solar Energy, 70(5), 431-441.
        (2) The C code is available from: http://www.psa.es/sdg/sunpos.htm
    Arguments:
        longitude: float
            Longitude, in degrees. West: negative, East: positive.
        latitude: float
            Latitude, in degrees. Northern: positive, Southern: negative.
        utc_time: datetime object
            UTC time.
    Returns:
        ZenithAngle: float
            Sun zenith angle, in degrees.
        AzimuthAngle: float
            Sun azimuth angle, in degrees.
    """

    rad = np.pi/180
    EarthMeanRadius = 6371.01 # in km
    AstronomicalUnit = 149597890 #in km

    DecimalHours = utc_time.hour+(utc_time.minute+utc_time.second/60.0)/60.0
    Aux1 = int((utc_time.month-14)/12)
    Aux2 = int(1461*(utc_time.year+4800+Aux1)/4 +
                 367*(utc_time.month-2-12*Aux1)/12 -
                 3*(utc_time.year + 4900+ Aux1)/100/4 +
                 utc_time.day-32075)
    JulianDate = Aux2-0.5+DecimalHours/24.0
    ElapsedJulianDays = JulianDate-2451545.0

    Omega = 2.1429-0.0010394594*ElapsedJulianDays
    MeanLongitude = 4.8950630+ 0.017202791698*ElapsedJulianDays
    MeanAnomaly = 6.2400600+ 0.0172019699*ElapsedJulianDays
    EclipticLongitude = MeanLongitude + 0.03341607*np.sin(MeanAnomaly)+ 0.00034894*np.sin(2*MeanAnomaly)-0.0001134-0.0000203*np.sin(Omega)
    EclipticObliquity = 0.4090928-6.2140e-9*ElapsedJulianDays+0.0000396*np.cos(Omega)
    Sin_EclipticLongitude = np.sin(EclipticLongitude)
    Y = np.cos(EclipticObliquity)*Sin_EclipticLongitude
    X = np.cos(EclipticLongitude)
    RightAscension = np.arctan2(Y, X)
    if RightAscension < 0.0:
        RightAscension = RightAscension + np.pi*2
    Declination = np.arcsin(np.sin(EclipticObliquity)*Sin_EclipticLongitude)

    GreenwichMeanSiderealTime = 6.6974243242+0.0657098283*ElapsedJulianDays+DecimalHours
    LocalMeanSiderealTime = (GreenwichMeanSiderealTime*15+longitude)*rad
    HourAngle = LocalMeanSiderealTime-RightAscension
    LatitudeInRadians = latitude*rad
    Cos_Latitude = np.cos(LatitudeInRadians)
    Sin_Latitude = np.sin(LatitudeInRadians)
    Cos_HourAngle= np.cos(HourAngle)
    ZenithAngle = np.arccos(Cos_Latitude*Cos_HourAngle*np.cos(Declination)+np.sin(Declination)*Sin_Latitude)
    Y = -np.sin(HourAngle)
    X = np.tan(Declination)*Cos_Latitude-Sin_Latitude*Cos_HourAngle
    Azimuth = np.arctan2(Y,X)
    if Azimuth < 0.0:
        Azimuth = Azimuth + np.pi*2
    Parallax = (EarthMeanRadius/AstronomicalUnit)*np.sin(ZenithAngle)
    ZenithAngle = (ZenithAngle+Parallax)/rad
    AzimuthAngle = Azimuth/rad

    return [ZenithAngle, AzimuthAngle]