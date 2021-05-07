"""
Module for NTT EFOSC2

.. include:: ../include/links.rst
"""
import glob
from pkg_resources import resource_filename

import numpy as np

from pypeit import msgs
from pypeit import telescopes
from pypeit.core import parse
from pypeit.core import framematch
from pypeit.spectrographs import spectrograph
from pypeit.images import detector_container


class NTTEFOSC2Spectrograph(spectrograph.Spectrograph):
    """
    Child of Spectrograph to handle NTT/EFOSC2 specific code
    """
    ndet = 1  # Because each detector is written to a separate FITS file
    telescope = telescopes.NTTTelescopePar()
    name = 'ntt_efosc2'
    camera = 'EFOSC2'
    comment = 'The ESO Faint Object Spectrograph and Camera version 2'

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return ['dispname', 'decker', 'binning', 'datasec']
    
    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='RA', required_ftypes=['science', 'standard'])
        self.meta['dec'] = dict(ext=0, card='DEC', required_ftypes=['science', 'standard'])
        self.meta['target'] = dict(ext=0, card='OBJECT')
        self.meta['binning'] = dict(card=None, compound=True) #CDELT1 and CDELT2
        self.meta['mjd'] = dict(ext=0, card='MJD-OBS')
        
        self.meta['datasec'] = dict(card=None, compound=True)
        self.meta['oscansec'] = dict(card=None, compound=True)
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        self.meta['airmass'] = dict(ext=0, card='HIERARCH ESO TEL AIRM START', required_ftypes=['science', 'standard'])
        self.meta['decker'] = dict(card=None, compound=True, required_ftypes=['science', 'standard'])
        # Extras for config and frametyping
        self.meta['dispname'] = dict(ext=0, card='HIERARCH ESO INS GRIS1 NAME', required_ftypes=['science', 'standard'])
        #self.meta['dispangle'] = dict(ext=0, card='HIERARCH ESO INS GRIS1 WLEN', rtol=2.0, required_ftypes=['science', 'standard']) did not find dispangle
        self.meta['idname'] = dict(ext=0, card='HIERARCH ESO DPR CATG')

    
    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key == 'binning':
            binspatial = headarr[0]['CDELT1']
            binspec = headarr[0]['CDELT2']
            binning = parse.binning2string(int(binspec), int(binspatial))
            return binning
        elif meta_key == 'decker':
            try:  # Science
                decker = headarr[0]['HIERARCH ESO INS SLIT1 NAME']
            except KeyError:  # Standard!
                try:
                    decker = headarr[0]['HIERARCH ESO SEQ SPEC TARG']
                except KeyError:
                    return None
            return decker
        elif meta_key == 'datasec' or meta_key == 'oscansec':
            data_x = headarr[0]['HIERARCH ESO DET OUT1 NX'] * 2 #valid pixels along X
            data_y = headarr[0]['HIERARCH ESO DET OUT1 NY'] * 2 #valid pixels along Y
            oscan_y = headarr[0]['HIERARCH ESO DET OUT1 OVSCY'] * 2 #Overscan region in Y, no overscan in X
            pscan_x = headarr[0]['HIERARCH ESO DET OUT1 PRSCX'] * 2 #Prescan region in X, no prescan in Y  
            if meta_key == 'datasec':
                datasec = '[%s:%s,:%s]' % (pscan_x, pscan_x+data_x, data_y)
                return datasec
            else:
                oscansec = '[:%s,:%s]' % (pscan_x, data_y) # Actually two overscan regions, here I only dealing with the region on x-axis
                return oscansec
        
        else:
            msgs.error("Not ready for this compound meta")


    def get_detector_par(self, hdu, det):
        """
        Return metadata for the selected detector.

        Args:
            hdu (`astropy.io.fits.HDUList`_):
                The open fits file with the raw image of interest.
            det (:obj:`int`):
                1-indexed detector number.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # Binning
        binning = self.get_meta_value(self.get_headarr(hdu), 'binning')
        
        # Manual: https://www.eso.org/sci/facilities/lasilla/instruments/efosc/doc/manual/EFOSC2manual_v4.2.pdf
        # Instrument paper: http://articles.adsabs.harvard.edu/pdf/1984Msngr..38....9B
        detector_dict = dict(
            binning         = binning,
            det             = 1, # only one detector
            dataext         = 0,
            specaxis        = 0,
            specflip        = False,
            spatflip        = False,
            platescale      = 0.005, # focal length is 200mm, unit radian/mm, manual 2.2
            darkcurr        = 0.0,
            saturation      = 65535, # Maual Table 8
            nonlinear       = 0.80,
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(0.91), # written in hgeader['HIERARCH ESO DET OUT1 GAIN']
            ronoise         = np.atleast_1d(10.0), # manual page 108
            datasec         = np.atleast_1d(self.get_meta_value(self.get_headarr(hdu), 'datasec')),
            oscansec        = np.atleast_1d(self.get_meta_value(self.get_headarr(hdu), 'oscansec'))
            #suffix          = '_Thor',
        )
        return detector_container.DetectorContainer(**detector_dict)

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.
        
        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        # Always correct for flexure, starting with default parameters
        par['flexure']['spec_method'] = 'boxcar'

        # Median overscan
        #   IF YOU CHANGE THIS, YOU WILL NEED TO DEAL WITH THE OVERSCAN GOING ALONG ROWS
        for key in par['calibrations'].keys():
            if 'frame' in key:
                par['calibrations'][key]['process']['overscan_method'] = 'median'
        
        # Adjustments to slit and tilts for NIR
        par['calibrations']['traceframe']['process']['use_darkimage'] = False
        par['calibrations']['pixelflatframe']['process']['use_darkimage'] = False
        par['calibrations']['illumflatframe']['process']['use_darkimage'] = False
        
        # Ignore PCA
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'

        # Tilt parameters
        par['calibrations']['tilts']['tracethresh'] = 25.0
        par['calibrations']['tilts']['spat_order'] = 3
        par['calibrations']['tilts']['spec_order'] = 4

        # 1D wavelength solution
        par['calibrations']['wavelengths']['method'] = 'holy-grail'
        par['calibrations']['wavelengths']['lamps'] = ['HeI', 'ArI']  # Grating dependent
        par['calibrations']['wavelengths']['rms_threshold'] = 0.25
        par['calibrations']['wavelengths']['sigdetect'] = 10.0
        par['calibrations']['wavelengths']['fwhm'] = 4.0  # Good for 2x binning
        par['calibrations']['wavelengths']['n_final'] = 4

        # Flats
        par['calibrations']['flatfield']['tweak_slits_thresh'] = 0.90
        par['calibrations']['flatfield']['tweak_slits_maxfrac'] = 0.10

        # Extraction
        par['reduce']['skysub']['bspline_spacing'] = 0.8
        par['reduce']['skysub']['no_poly'] = True
        par['reduce']['skysub']['bspline_spacing'] = 0.6
        par['reduce']['skysub']['joint_fit'] = False
        par['reduce']['skysub']['global_sky_std']  = False

        par['reduce']['extraction']['sn_gauss'] = 4.0
        par['reduce']['findobj']['sig_thresh'] = 5.0
        par['reduce']['skysub']['sky_sigrej'] = 5.0
        par['reduce']['findobj']['find_trim_edge'] = [5,5]

        return par
    
    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        # TODO: Allow for 'sky' frame type, for now include sky in
        # 'science' category
        if ftype == 'science':
            return good_exp & ((fitstbl['idname'] == 'SCIENCE')
                                | (fitstbl['target'] == 'STD,TELLURIC')
                                | (fitstbl['target'] == 'STD,SKY'))
        if ftype == 'standard':
            return good_exp & ((fitstbl['target'] == 'STD,FLUX')
                               | (fitstbl['target'] == 'STD'))
        if ftype == 'bias':
            return good_exp & ((fitstbl['target'] == 'BIAS')
                               |(fitstbl['target'] == 'DARK'))
        if ftype in ['pixelflat', 'trace', 'illumflat']:
            # Flats and trace frames are typed together
            return good_exp & ((fitstbl['target'] == 'FLAT')
                               | (fitstbl['target'] == 'SKY,FLAT')
                               | (fitstbl['target'] == 'DOME'))
        if ftype == 'pinhole':
            # Don't type pinhole
            return np.zeros(len(fitstbl), dtype=bool)
        if ftype in ['arc', 'tilt']:
            return good_exp & ((fitstbl['target'] == 'WAVE'))

        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

