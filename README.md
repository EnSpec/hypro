<p id="readme-logo-heading" align="center" width="100%">
  <img width="250px" alt="HyPro" src="https://user-images.githubusercontent.com/18175461/213893316-74d5d4c3-ce81-4023-88e5-fa4bbe728702.png"/>
</p>
<p id="readme-box-heading" align="center" width="100%">
  <b>© Copyright 2020</b><br/>
  Nanfeng Liu<br/>
  Philip Townsend<br/>
  <br/>
  Open-source software licensed under GNU GPLv3<br/>
</p>

# Package Overview

## Description

HyPro is a Python package that provides tools for processing raw hyperspectral image data from pushbroom sensors.

The main module, `HyspexPro`, is intended to be run as a script. Its functionality is summarized here.

1) `HyspexPro` aims to do geometric and radiometric corrections on HySpex images. Our imaging system consists of two HySpex cameras (VNIR-1800 and SWIR-384) and one GPS-Inertial Measurement Unit (IMU) sensor (iMAR iTraceRT-F400). The VNIR-1800 sensor has 182 spectral bands within 400-1000 nm (spectral sampling = 3.26 nm). The SWIR-384 sensor has 288 bands within 960-2500 nm (spectral sampling = 5.45 nm). For more details about these cameras, please refer to the [NEO website](https://www.hyspex.com/hyspex-products/). The iTraceRT-F400 sensor records the attitudes (roll, pitch and heading angles) and positions (longitude, latitude and altitude) of the imaging platform.

2) Geometric correction focuses on removing image geometric distortions mainly caused by platform motions, which can be characterized by roll, pitch and heading angles. We adapted the ray tracing method proposed by Meyer<sup>[[1]](#Meyer1994)</sup> to calculate the map coordinates (easting, northing and elevation) of each raw image pixel (a process called georeferencing). Then, the `warp` tool provided by [GDAL](https://gdal.org) (Geospatial Data Abstraction Library) was used to resample the raw image to regular map grids (a process called georectification).

3) Radiometric correction focuses on reducing radiometric distortions mainly caused by sensor smile effects (shifts in sensor band center wavelengths and full-widths-at-half-maximum, or FWHMs) and atmospheric effects (absorption and scattering). The spectral matching method proposed by Gao et al.<sup>[[2]](#Gao2004)</sup> was adapted to detect sensor smile effects. Then, raw radiance spectra were resampled to common center wavelengths using a cubic interpolation. In the atmospheric correction step, two atmospheric parameters (water vapor column and aerosol visibility) were retrieved from image spectra. The water vapor column was estimated via the continuum-interpolated band ratio (CIBR) technique proposed by Kaufman et al.<sup>[[3]](#Kaufman1992)</sup>. The estimation of visibility was based on the dense dark vegetation (DDV) method proposed by Kaufman et al.<sup>[[4]](#Kaufman1997)</sup>. All radiometric correction steps require an atmospheric lookup table which consists of the at-sensor radiance simulated under different atmospheric conditions and sun-target-sensor view geometries. An open-source radiative transfer model, libRadtran, was used for this purpose. For more details about libRadtran, please refer to the [website](http://www.libradtran.org/doku.php).

4) The input dataset of `HyspexPro` includes:
    * HySpex raw digital number (DN) images;
    * HySpex IMU & GPS data processed in HySpex NAV;
    * HySpex sensor model files provided by NEO;
    * DEM;
    * HySpex lab radiometric re-calibration data provided by NEO (optional);
    * Ground control points data (optional).

5) The output dataset of `HyspexPro` includes:
    * HySpex ground surface reflectance images;
    * View angles (zenith and azimuth);
    * DEM.


## Processing Workflow

The standard workflow of `HyspexPro` is shown diagrammatically below:

<p id="readme-workflow-diagram" align="center" width="100%">
  <img width="80%" alt="DN-to-Reflectance Processing Pipeline" src="https://user-images.githubusercontent.com/18175461/215301395-d581e475-ce88-4ca5-b004-1abbb3dbcb84.png">
</p>


## Credits

HyPro is authored by [Nanfeng Liu](mailto:nliu58@wisc.edu), with contributions from [Adam Chlus](mailto:chlus@wisc.edu) & [Brendan Heberlein](mailto:bheberlein@wisc.edu).

HyPro is a product of the UW Environmental Spectroscopy Laboratory (EnSpec) and is developed & maintained under [Philip Townsend](mailto:ptownsend@wisc.edu) with the support of the University of Wisconsin.


## License

HyPro is licensed for public use under [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html). For details of the licensing agreement, refer to [`./LICENSE.txt`](https://github.com/EnSpec/hypro/blob/main/LICENSE.txt). In brief:

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3,
    as published by the Free Software Foundation.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License Version 3
    along with this program. If not, see <https://www.gnu.org/licenses/>.


## Ancillary Datasets

### Solar Irradiance

HyPro is distributed with solar spectral irradiance data derived from [LBLRTM](https://github.com/AER-RC/LBLRTM), a radiative transfer model from [Verisk Atmospheric & Environmental Research](https://www.aer.com) (AER). This is a synthetic spectrum, simulated with a solar source function and database of atomic and molecular lines developed by Dr. Robert Kurucz<sup>[[5]](#Kurucz1992)</sup>. The data are derived from LBLRTM 5.21 and are distributed here with permission from AER. This same dataset is used internally by libRadtran.

The spectrum represents top-of-atmosphere spectral irradiance at a Sun-Earth distance of 1 AU, sampled at 0.1-nm intervals and expressed in units of mW/m<sup>2</sup>/nm.

Note that these are legacy data. Newer, more accurate versions of the [LBLRTM solar source function](https://github.com/AER-RC/solar-source-function) are available. The source code and derived products are subject to licensing terms dictated by AER. 


### Sun-Earth distance

Also provided are approximate Sun-Earth distances by day of year, expressed in AU. These are public-domain data provided by [USGS/NASA Landsat Missions](https://www.usgs.gov/media/files/earth-sun-distance-astronomical-units-days-year). The eccentricity of Earth's orbit produces variations in total irradiant flux on the order of ±3%. Note that the given distances are only approximate; ephemeris-based calculations may be used for more accurate results.


## References

<a name="Meyer1994"><sup><b>[1]</b></sup></a> Meyer P (1994). A parametric approach for the geocoding of airborne visible/infrared imaging spectrometer (AVIRIS) data in rugged terrain. Remote Sens Environ 49(2): 118–30. _doi:[10.1016/0034-4257(94)90048-5](https://doi.org/10.1016/0034-4257(94)90048-5)_

<a name="Gao2004"><sup><b>[2]</b></sup></a> Gao B-C, Montes MJ & Davis CO (2004). Refinement of wavelength calibrations of hyperspectral imaging data using a spectrum-matching technique. Remote Sens Environ 90(4): 424–33. _doi:[10.1016/j.rse.2003.09.002](https://doi.org/10.1016/j.rse.2003.09.002)_

<a name="Kaufman1992"><sup><b>[3]</b></sup></a> Kaufman YJ & Gao B-C (1992). Remote sensing of water vapor in the near IR from EOS/MODIS. IEEE Trans Geosci Remote Sens 30(5): 871–84. _doi:[10.1109/36.175321](https://doi.org/10.1109/36.175321)_

<a name="Kaufman1997"><sup><b>[4]</b></sup></a> Kaufman YJ, Wald AE, Remer LA, Gao B-C, Li R-R & Flynn L (1997). The MODIS 2.1-/spl mu/m channel-correlation with visible reflectance for use in remote sensing of aerosol. IEEE Trans Geosci Remote Sens 35(5): 1286–98. _doi:[10.1109/36.628795](https://doi.org/10.1109/36.628795)_

<a name="Kurucz1992"><sup><b>[5]</b></sup></a> Kurucz RL (1992). Synthetic infrared spectra. In Rabin DM, Jefferies JT & Lindsey C (Eds.), Infrared solar physics: proceedings of the 154th symposium of the International Astronomical Union (pp. 523–31). Norwell, MA, USA: Kluwer Academic Publishers. Symposium held in Tucson, AZ, USA, March 2–6, 1992. _doi:[10.1017/S0074180900124805](https://doi.org/10.1017/S0074180900124805)_
