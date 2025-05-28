"""
Module for JWST NIRSpec specific methods.

.. include:: ../include/links.rst
"""
import numpy as np

from pypeit import msgs
from pypeit import telescopes
from pypeit import utils
from pypeit.core import framematch
from pypeit import io
from pypeit.par import pypeitpar
from pypeit.spectrographs import spectrograph
from pypeit.core import parse
from pypeit.images import detector_container
from IPython import embed

class MagellanLLAMASSpectograph(spectrograph.Spectrograph):
    """
    Class for handling the Magellan LLAMAS FIBER IFU spectrograph.
    """
    ndet = 24
    name = 'magellan_llamas'
    header_name = 'magellan_llamas'
    telescope = telescopes.MagellanTelescopePar()
    pypeline = 'MultiSlit'
    url = None
    supported = True

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """

        # Detector 1, i.e. NRS1 from
        # https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-detectors/nirspec-detector-performance
        detector_dict1 = dict(
            binning='1,1',
            det=1,
            dataext=1,
            specaxis=1,
            specflip=False,
            spatflip=False,
            platescale=1, #TODO:change this valyes
            darkcurr=0.0,  # e-/pixel/hour  (=0.0092 e-/pixel/s)
            saturation=2e6,
            nonlinear=0.95,  # need to look up and update
            mincounts=-1e10,
            numamplifiers=1,
            gain=np.atleast_1d(0.996),
            ronoise=np.atleast_1d(1.0),
            datasec=np.atleast_1d('[1:2048, 1:2048]'),
            oscansec=None,
        )

        # make all detectors exactly the same

        detector_dicts = [detector_dict1]

        for n in range(2, self.ndet+1):
            detector_dict2 = detector_dict1.copy()
            detector_dict2['det'] = n
            detector_dict2['dataext'] = n
            detector_dicts.append(detector_dict2)

        print(detector_dicts[det-1])
        return detector_container.DetectorContainer(**detector_dicts[det-1])

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        par = super().default_pypeit_par()


        # Reduce
        par['reduce']['trim_edge'] = [0,0]
        par['calibrations']['slitedges']['edge_thresh'] = 1.0
        par['calibrations']['slitedges']['sobel_enhance'] = 1
        par['calibrations']['slitedges']['filt_iter'] = 1
        par['calibrations']['slitedges']['fwhm_gaussian'] = 1.0
        par['calibrations']['slitedges']['fwhm_uniform'] = 1.0
        par['calibrations']['slitedges']['match_tol'] = 1
        par['calibrations']['slitedges']['smash_range'] = [0.5,0.6]
        par['calibrations']['slitedges']['length_range'] = 0.9

        par['calibrations']['wavelengths']['reid_arxiv'] = 'wvarxiv_magellan_llamas_20250526T1548.fits'
        par['calibrations']['wavelengths']['method'] = 'full_template'
        par['calibrations']['wavelengths']['lamps'] = ['ThAr_magellan_llamas']
        par['calibrations']['wavelengths']['match_toler'] = 5.0
        par['calibrations']['wavelengths']['n_final'] = 3
        par['calibrations']['wavelengths']['n_first'] = 3
        par['calibrations']['wavelengths']['nlocal_cc'] = 5
        par['calibrations']['wavelengths']['sigdetect'] = 10


        par['calibrations']['wavelengths']['nsnippet'] = 1
        par['calibrations']['wavelengths']['numsearch'] = 20
        par['calibrations']['wavelengths']['rms_thresh_frac_fwhm'] = 0.5
        par['calibrations']['wavelengths']['wvrng_arxiv'] = [3000, 6000]


        par['calibrations']['flatfield']['slit_trim'] = 0
        par['calibrations']['flatfield']['slit_illum_finecorr'] = False
        par['calibrations']['flatfield']['tweak_slits'] = False

        par['reduce']['extraction']['skip_optimal'] = True
        par['reduce']['extraction']['boxcar_radius'] = 1
        par['reduce']['extraction']['model_full_slit'] = True


        par['reduce']['skysub']['global_sky_std'] = False
        par['reduce']['skysub']['no_poly'] = False
        par['reduce']['skysub']['local_maskwidth'] = 1.0
        par['reduce']['skysub']['no_local_sky'] = True


        par['reduce']['findobj']['skip_final_global'] = True
        par['reduce']['findobj']['skip_skysub'] = True









        turn_off = dict(use_overscan=False,
                        use_darkimage=False,
                        use_illumflat=False,use_pixelflat=False,)
        par.reset_all_processimages_par(**turn_off)

        return par

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='RA')
        self.meta['dec'] = dict(ext=0, card='DEC')
        self.meta['target'] = dict(ext=0, card='OBJECT')
        #self.meta['mode'] = dict(ext=0, card='EXP_TYPE')
        #self.meta['decker'] = dict(ext=0, card='APERNAME')

        self.meta['binning'] = dict(ext=0, card=None, default='1,1')
        # dispname
        self.meta['dispname'] = dict(ext=0, card=None, default='LLAMAS-BLUE')
        # decker
        self.meta['decker'] = dict(ext=0, card=None, default='LLAMAS-IFU')
        self.meta['mjd'] = dict(ext=0, card='MJD-OBS', default=0)
        self.meta['exptime'] = dict(ext=0, card='SEXPTIME', default=1.0)
        self.meta['airmass'] = dict(ext=0, card="HIERARCH TEL AIRMASS")

        # Extras for config and frametyping
        #self.meta['dispname'] = dict(ext=0, card='GRATING')
        #self.meta['filter1'] = dict(ext=0, card='FILTER')
        #self.meta['idname'] = dict(ext=0, card=None, compound=True)
        #self.meta['dithpat'] = dict(ext=0, card=None, compound=True)
        #self.meta['dithpos'] = dict(ext=0, card='YOFFSET')

        # used for arc and continuum lamps
        #self.meta['lampstat01'] = dict(ext=0, card=None, compound=True)
        #self.meta['instrument'] = dict(ext=0, card='INSTRUME')



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

        return None

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
        # There is just one configuration for this instrument
        return []

    def pypeit_file_keys(self):
        """
        Define the list of keys to be output into a standard ``PypeIt`` file.

        Returns:
            :obj:`list`: The list of keywords in the relevant
            :class:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        pypeit_keys = super().pypeit_file_keys()
        pypeit_keys.remove('airmass')
        pypeit_keys.remove('binning')
        pypeit_keys.remove('dispname')
        pypeit_keys.remove('decker')
        return pypeit_keys


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

        #if ftype == 'science':
        #    return np.ones(len(fitstbl), dtype=bool)
        #msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)



