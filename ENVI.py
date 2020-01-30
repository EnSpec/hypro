#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions to process ENVI format data.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

"""
Notes:
    (1) The code is adapted from the HyTools:
        https://github.com/EnSpec/HyTools-sandbox/blob/master/hytools/file_io/envi_read.py
        https://github.com/EnSpec/HyTools-sandbox/blob/master/hytools/file_io/envi_write.py
"""

from copy import deepcopy

envi_value_types = {'acquisition time': 'str',
                    'band names': 'str_list',
                    'bands': 'int', # required
                    'bbl': 'float_list',
                    'byte order': 'int', # required
                    'class lookup': 'int_list',
                    'class names': 'str_list',
                    'classes': 'int',
                    'cloud cover': 'float',
                    'complex function': 'str',
                    'coordinate system string': 'str',
                    'data gain values': 'float_list',
                    'data ignore value': 'float',
                    'data offset values': 'float_list',
                    'data reflectance gain values': 'float_list',
                    'data reflectance offset values': 'float_list',
                    'data type': 'int', # required
                    'default bands': 'int_list',
                    'default stretch': 'str',
                    'dem band': 'int',
                    'dem file': 'str',
                    'description': 'str',
                    'file type': 'str', # required
                    'fwhm': 'float_list',
                    'geo points': 'float_list',
                    'header offset': 'int',
                    'interleave': 'str', # required
                    'lines': 'int', # required
                    'map info': 'str_list',
                    'pixel size': 'float_list',
                    'projection info': 'str',
                    'read procedures': 'str_list',
                    'reflectance scale factor': 'float',
                    'rpc info': 'str',
                    'samples': 'int', # required
                    'security tag': 'str',
                    'sensor type': 'str',
                    'solar irradiance': 'float',
                    'spectra names': 'str_list',
                    'sun azimuth': 'float',
                    'sun elevation': 'float',
                    'wavelength': 'float_list',
                    'wavelength units': 'str',
                    'x start': 'float',
                    'y start': 'float',
                    'z plot average': 'int_list',
                    'z plot range': 'float_list',
                    'z plot titles': 'str'}

envi_fields = ['description',
               'file type',
               'header offset',
               'read procedures',
               'sensor type',
               'acquisition time',
               'lines',
               'samples',
               'x start',
               'y start',
               'bands',
               'default bands',
               'interleave',
               'data type',
               'byte order',
               'coordinate system string',
               'map info',
               'pixel size',
               'projection info',
               'band names',
               'bbl',
               'wavelength units',
               'wavelength',
               'fwhm',
               'dem file',
               'sun azimuth',
               'sun elevation',
               'default stretch',
               'data gain values',
               'data ignore value',
               'data offset values',
               'data reflectance gain values',
               'data reflectance offset values',
               'classes',
               'class names',
               'class lookup',
               'cloud cover',
               'geo points',
               'rpc info',
               'security tag',
               'solar irradiance',
               'spectra names',
               'z plot average',
               'z plot range',
               'z plot titles']

def read_envi_header(file):
    """ Read ENVI header.
    Arguments:
        file: str
            ENVI header filename.
    Returns:
        header: dict
            ENVI header.
    """

    header = empty_envi_header()
    fid = open(file, 'r')
    trans_tab = str.maketrans(dict.fromkeys('\n{}'))
    while 1:
        line = fid.readline()
        if '=' in line:
            key, value = line.split('=', 1)
            if ('{' in value) and ('}' not in value):
                while '}' not in line:
                    line = fid.readline()
                    if line.strip()[0] == ';':
                        continue
                    value += line
            key = key.strip()
            value = value.translate(trans_tab).strip()
            if key in envi_value_types.keys():
                if envi_value_types[key]   == 'int':
                    value = int(value)
                elif envi_value_types[key] == 'float':
                    value = float(value)
                elif envi_value_types[key] == 'str_list':
                    value = list(map(str.strip, value.split(',')))
                elif envi_value_types[key] == 'int_list':
                    value = list(map(int, value.split(',')))
                elif envi_value_types[key] == 'float_list':
                    value = list(map(float, value.split(',')))
            header[key] = value
        if line == '':
            break
    fid.close()
    header['interleave'] = header['interleave'].lower()
    if header['header offset'] is None:
        header['header offset'] = 0
    check_envi_required_fields(header)

    return header

def check_envi_required_fields(header):
    """ Check ENVI required fields.
    Arguments:
        header: dict
            ENVI header.
    """
    required_fields = ['byte order',
                       'data type',
                       'interleave',
                       'bands',
                       'lines',
                       'samples']
    for field in required_fields:
        if header[field] is None:
            raise ValueError('No value for %s!' %field)

def empty_envi_header():
    """ Generate an empty ENVI header.
    Returns:
        header: dict
            Empty ENVI header.
    """

    header = dict()
    for key in envi_fields:
        header[key] = None

    return header

def write_envi_header(file, header):
    """ Write ENVI header.
    Arguments:
        file: str
            ENVI header filename.
        header: dict
            ENVI header.
    """

    header = deepcopy(header)
    fid = open(file, 'w')
    fid.write('ENVI\n')
    for field in envi_fields:
        if header[field] is None:
            header.pop(field)
            continue
        if type(header[field]) is list:
            value = []
            if field == 'map info':
                N = 12
            elif field == 'class lookup':
                N = 3
            elif field == 'geo points':
                N = 4
            else:
                N = 5
            for i, v in enumerate(header[field]):
                if (i+1)%N==0:
                    value.append(str(v)+'\n')
                else:
                    value.append(str(v))
            value = '{%s}' %(', '.join(value))
        elif field in ['description',
                       'coordinate system string']:
            value = '{%s}' %(header[field])
        else:
            value = str(header[field])
        fid.write('%s = %s\n' %(field, value))
        header.pop(field)

    for field in header.keys():
        if header[field] is None:
            continue
        if type(header[field]) is list:
            value = []
            for i, v in enumerate(header[field]):
                if (i+1)%5==0:
                    value.append(str(v)+'\n')
                else:
                    value.append(str(v))
            value = '{%s}' %(', '.join(value))
        else:
            value = str(header[field])
        fid.write('%s = %s\n' %(field, value))

    fid.close()