"""
Module for Magellan/LDSS3 specific methods.


"""
import os
from pkg_resources import resource_filename

import numpy as np
from astropy.time import Time
from astropy.io import fits

from pypeit import msgs, telescopes, io
from pypeit.core import framematch, parse
from pypeit.spectrographs import spectrograph
from pypeit.images import detector_container


class MagellanLDSS3Spectrograph(spectrograph.Spectrograph):

    # DONE
    ndet = 1
    telescope = telescopes.MagellanTelescopePar()
    camera = 'LDSS3-C'
    header_name = 'LDSS3'
    comment = None

    # DONE
    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        # Target related
        self.meta['ra'] = dict(ext=0, card='RA')
        self.meta['dec'] = dict(ext=0, card='DEC')
        self.meta['target'] = dict(ext=0, card='OBJECT')

        # Instrument related
        self.meta['dispname'] = dict(ext=0, dtype=str, card='GRISM')
        self.meta['decker'] = dict(ext=0, dtype=str, card='APERTURE')
        self.meta['binning'] = dict(ext=0, card=None, compound=True)
        self.meta['filter1'] = dict(ext=0, dtype=str, card='FILTER')
        #self.meta['detector'] = dict(ext=0, dtype=int, card='OPAMP')
        self.meta['amp'] = dict(ext=0, dtype=int, card='OPAMP')

        # Obs
        self.meta['mjd'] = dict(ext=0, card=None, compound=True)
        self.meta['airmass'] = dict(ext=0, dtype=float, card='AIRMASS')
        self.meta['exptime'] = dict(ext=0, dtype=float, card='EXPTIME')

        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='EXPTYPE')
        self.meta['instrument'] = dict(ext=0, card='INSTRUME')


    def get_detector_par(self, det, hdu=None):

        binning = '1,1' if hdu is None else self.get_meta_value(self.get_headarr(hdu), 'binning')
        # Detector 1
        detector_dict_1 = dict(
            binning         = binning,
            det             = 1,
            dataext         = 0, #in which extension is the data
            specaxis        = 0, #axis where the spectrum is located
            specflip        = False,
            spatflip        = False,
            platescale      = 0.189, # arcsec/pixel, extracted from specs docs
            darkcurr        = 0.0, # assumed to be 0, should be corrected from Darks
            saturation      = 60000., # ADU
            nonlinear       = 0.875, # non-linearity coefficient
            mincounts       = -1e10,
            numamplifiers   = 2,
            gain            = np.atleast_1d(1.5), # electrons/ADU, extracted from specs docs
            ronoise         = np.atleast_1d(6.5) # read-out noise in electrons, extracted from specs docs
        )
        '''
        detector_dict_2 = dict(
            binning         = binning,
            det             = 2,
            dataext         = 0, #in which extension is the data
            specaxis        = 0, #axis where the spectrum is located
            specflip        = False,
            spatflip        = False,
            platescale      = 0.189, # arcsec/pixel, extracted from specs docs
            darkcurr        = 0.0, # assumed to be 0, should be corrected from Darks
            saturation      = 60000., # ADU
            nonlinear       = 0.875, # non-linearity coefficient
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(1.5), # electrons/ADU, extracted from specs docs
            ronoise         = np.atleast_1d(6.5) # read-out noise in electrons, extracted from specs docs
        )'''
        detectors = [detector_dict_1]#, detector_dict_2]
        # Return
        return detector_container.DetectorContainer(**detectors[det-1])


    def pypeit_file_keys(self):
        """
        Define the list of keys to be output into a standard ``PypeIt`` file.

        Returns:
            :obj:`list`: The list of keywords in the relevant
            :class:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        pypeit_keys = super().pypeit_file_keys()
        # TODO: Why are these added here? See
        # pypeit.metadata.PypeItMetaData.set_pypeit_cols
        pypeit_keys += ['calib', 'comb_id', 'bkg_id', "manual"]
        return pypeit_keys

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
        if meta_key == 'mjd':
            time = '{:s}T{:s}'.format(headarr[0]['UT-DATE'], headarr[0]['UT-TIME'])
            ttime = Time(time, format='isot')
            return ttime.mjd
        elif meta_key == 'binning':
            binspatial, binspec = parse.parse_binning(headarr[0]['BINNING']) #1x1
            return parse.binning2string(binspec, binspatial)

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
        return ['dispname', 'decker', 'binning']

    def config_independent_frames(self):
        """
        Define frame types that are independent of the fully defined
        instrument configuration.

        Bias and dark frames are considered independent of a configuration.
        Standards are assigned to the correct configuration frame group by
        grism (i.e. ignoring that they are taken with a wider slit).
        See :func:`~pypeit.metadata.PypeItMetaData.set_configurations`.

        Returns:
            :obj:`dict`: Dictionary where the keys are the frame types that
            are configuration independent and the values are the metadata
            keywords that can be used to assign the frames to a configuration
            group.
        """
        return {'standard': 'dispname','bias': None, 'dark': None}

    '''
    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Master bias frame used to identify bad pixels. **This is
                ignored for KCWI.**

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        return
        # TODO: customize for LDSS3

        # Call the base-class method to generate the empty bpm; msbias is always set to None.
        bpm_img = super().bpm(filename, det, shape=shape, msbias=None)

        # Extract some header info
        head0 = fits.getheader(filename, ext=0)
        binning = head0['CCDSUM']

        # Construct a list of the bad columns
        bc = []
        if det == 1:
            # No bad pixel columns on detector 1
            pass
        elif det == 2:
            if binning == '1 1':
                # The BPM is based on 2x2 binning data, so the 2x2 numbers are just multiplied by two
                msgs.warn("BPM is likely over-estimated for 1x1 binning")
                bc = [[220, 222, 3892, 4100],
                      [952, 954, 2304, 4100]]
            elif binning == '2 2':
                bc = [[110, 111, 1946, 2050],
                      [476, 477, 1154, 2050]]
        else:
            msgs.warn("Bad pixel mask is not available for det={0:d} binning={1:s}".format(det, binning))
            bc = []

        # Apply these bad columns to the mask
        for bb in range(len(bc)):
            bpm_img[bc[bb][2]:bc[bb][3] + 1, bc[bb][0]:bc[bb][1] + 1] = 1

        return bpm_img'''

class MagellanLDSS3SMultiSlitpectrograph(MagellanLDSS3Spectrograph):

    name = 'magellan_ldss3_multi'
    supported = True
    pypeline = 'MultiSlit' # important for telling pypeit which pipeline to use

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        # Wavelengths
        # 1D wavelength solution with arc lines
        par['calibrations']['wavelengths']['rms_threshold'] = 0.3
        par['calibrations']['wavelengths']['sigdetect'] = 5.0
        par['calibrations']['wavelengths']['fwhm'] = 5.0
        par['calibrations']['wavelengths']['n_first'] = 2
        par['calibrations']['wavelengths']['n_final'] = 4
        par['calibrations']['wavelengths']['sigrej_final'] = 3.0
        par['calibrations']['wavelengths']['sigrej_first'] = 3.0

        # make this grating dependent
        par['calibrations']['wavelengths']['match_toler'] = 0.5


        # Set slits and tilts parameters
        par['calibrations']['slitedges']['sobel_mode'] = 'constant'

        # Processing steps
        turn_off = dict(use_overscan=False)
        par.reset_all_processimages_par(**turn_off)

        # Good exposure times, we do not limit them
        par['calibrations']['standardframe']['exprng'] = [0, None]
        par['calibrations']['arcframe']['exprng'] = [0, None]
        par['calibrations']['darkframe']['exprng'] = [0, None]
        par['scienceframe']['exprng'] = [0, None]

        # number of detectors
        par['rdx']['detnum'] = 1

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

        if ftype in ['bias']:
            return good_exp & (fitstbl['idname'] == 'Bias')
        if ftype in ['arc', 'tilt']:
            return good_exp & check_files_are_arc(fitstbl)
        if ftype in ['pixelflat', 'illumflat', 'trace']:
            return good_exp & check_files_are_pixelflat(fitstbl)
        if ftype in ['science']:
            return good_exp & check_files_are_science(fitstbl)
        if ftype in ['dark']:
            return good_exp & (fitstbl['idname'] == 'Dark')
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)


    def get_rawimage(self, raw_file, det):
        """
        Read raw images and generate a few other bits and pieces that are key
        for image processing.

        .. warning::

            - When reading multiple detectors for a mosaic, this function
              expects all detector arrays to have exactly the same shape.

        Parameters
        ----------
        raw_file : :obj:`str`
            File to read
        det : :obj:`int`, :obj:`tuple`
            1-indexed detector(s) to read.  An image mosaic is selected using a
            :obj:`tuple` with the detectors in the mosaic, which must be one of
            the allowed mosaics returned by :func:`allowed_mosaics`.

        Returns
        -------
        detector_par : :class:`~pypeit.images.detector_container.DetectorContainer`, :class:`~pypeit.images.mosaic.Mosaic`
            Detector metadata parameters for one or more detectors.
        raw_img : `numpy.ndarray`_
            Raw image for this detector.  Shape is 2D if a single detector image
            is read and 3D if multiple detectors are read.  E.g., the image from
            the first detector in the tuple is accessed using ``raw_img[0]``.
        hdu : `astropy.io.fits.HDUList`_
            Opened fits file
        exptime : :obj:`float`
            Exposure time *in seconds*.
        rawdatasec_img : `numpy.ndarray`_
            Data (Science) section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.  Shape
            is identical to ``raw_img``.
        oscansec_img : `numpy.ndarray`_
            Overscan section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.  Shape
            is identical to ``raw_img``.
        """
        # Open
        hdu = io.fits_open(raw_file)

        # Grab the detector or mosaic parameters
        det_par = self.get_detector_par(det)

        # Exposure time (used by RawImage)
        # NOTE: This *must* be (converted to) seconds.
        exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]

        image = hdu[0].data
        image = image.astype("float64")
        # Mask defining region where data is valid. Assumed to be all
        rawdatasec_img = np.ones_like(image, dtype=int)
        #rawdatasec_img = np.array(2*[rawdatasec_img])
        # Overscan region. No overscan in this case.
        oscansec_img = np.zeros_like(image, dtype=int)
        #oscansec_img = np.array(2*[oscansec_img])

        return det_par, image, hdu, exptime, rawdatasec_img, oscansec_img


def check_files_are_arc(fitstbl, lamps=['He', 'Ne', 'Ar']):
    object_names = fitstbl['target'] #OBJECT
    mask_names = fitstbl['decker']
    exp_types = fitstbl['idname']
    mask = []
    # arc object name are of the form MASK_NAME + LAMPS + EXPOSURE TYPE
    # e.g. ATM3a2_v1 HeNeAr Long

    for name, mask_name, exptype in zip(object_names, mask_names, exp_types):
        data = name.split(" ")
        # if the first element is the mask name
        if len(data) > 1 and exptype == 'Object' and data[0] == mask_name:
            # and there is a lamp in the second element
            have_lamp = False
            for lamp in lamps:
                if lamp in data[1]:
                    # the file is an arc
                    have_lamp = True

            mask.append(have_lamp)
        else:
            mask.append(False)

    return np.array(mask)

def check_files_are_science(fitstbl):
    object_names = fitstbl['target'] #OBJECT
    mask_names = fitstbl['decker']
    exp_types = fitstbl['idname']
    mask = []
    # arc object name are of the form MASK_NAME + Science
    # e.g. ATM3a2_v1 Science or ATM3a2_v1 science

    for name, mask_name, exptype in zip(object_names, mask_names, exp_types):
        data = name.split(" ")
        # if the first element is the mask name
        if  "science" in name.lower():
            # and there is a lamp in the second element

            mask.append(True)
        else:
            mask.append(False)

    return np.array(mask)

def check_files_are_trace_illum(fitstbl):
    object_names = fitstbl['target'] #OBJECT
    mask_names = fitstbl['decker']
    exp_types = fitstbl['idname']
    mask = []
    # arc object name are of the form MASK_NAME + thr
    # e.g. ATM3a2_v1 thr

    for name, mask_name, exptype in zip(object_names, mask_names, exp_types):
        data = name.split(" ")
        # if the first element is the mask name
        if len(data) > 1 and exptype == 'Object' and data[0] == mask_name and data[1].lower() == "thr":
            # and there is a lamp in the second element

            mask.append(True)
        else:
            mask.append(False)

    return np.array(mask)

def check_files_are_pixelflat(fitstbl):
    object_names = fitstbl['target'] #OBJECT
    mask_names = fitstbl['decker']
    exp_types = fitstbl['idname']
    mask = []
    # arc object name are of the form MASK_NAME + flatQhQl
    # e.g. ATM3a2_v1 flatQhQl

    for name, mask_name, exptype in zip(object_names, mask_names, exp_types):
        data = name.split(" ")
        # if the first element is the mask name
        if "flat" in name.lower():
            # and there is a lamp in the second element

            mask.append(True)
        else:
            mask.append(False)

    return np.array(mask)

