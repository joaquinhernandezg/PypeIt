"""
Module for Magellan/LDSS3 specific methods.


"""
# UPDATED: Removed unused imports (os, resource_filename, fits)
import numpy as np
from astropy.time import Time

from pypeit import msgs, telescopes, io
from pypeit.core import framematch, parse
from pypeit.spectrographs import spectrograph
from pypeit.images import detector_container

from pypeit.core import parse

from pypeit.images.mosaic import Mosaic
from pypeit.core.mosaic import build_image_mosaic_transform



class MagellanLDSS3Spectrograph(spectrograph.Spectrograph):
    """
    Base class for Magellan/LDSS3 spectrograph.
    """
    ndet = 2
    telescope = telescopes.MagellanTelescopePar()
    camera = 'LDSS3-C'
    header_name = 'LDSS3'
    # UPDATED: Added instrument URL for documentation
    url = 'https://www.lco.cl/technical-documentation/ldss-3-user-manual/'
    # UPDATED: Added descriptive comment
    comment = 'Low Dispersion Survey Spectrograph'
    supported = False

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
        self.meta['dispname'] = dict(ext=0, card='GRISM')
        self.meta['decker'] = dict(ext=0, card='APERTURE')
        self.meta['binning'] = dict(ext=0, card=None, compound=True)
        self.meta['filter1'] = dict(ext=0, card='FILTER')

        # Obs
        # UPDATED: mjd is compound metadata, computed in compound_meta()
        self.meta['mjd'] = dict(ext=0, card=None, compound=True)
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')

        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='EXPTYPE')
        self.meta['instrument'] = dict(ext=0, card='INSTRUME')

        # amplifier dependent
        #self.meta['amp'] = dict(ext=0, card='OPAMP')
        #self.meta['ronoise'] = dict(ext=0, card='RONOISE')
        #self.meta['gain'] = dict(ext=0, card='EGAIN')
        #self.meta['datasec'] = dict(ext=0, card='DATASEC')
        #self.meta['oscansec'] = dict(ext=0, card='BIASSEC')


    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        UPDATED: Added comprehensive docstring following PypeIt patterns.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest. If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
                Object with the detector metadata.
        """
        binning = '1,1' if hdu is None else self.get_meta_value(self.get_headarr(hdu), 'binning')

        # UPDATED: Added datasec and oscansec parameters (modern PypeIt format)
        detector1_dict = dict(
            binning         = binning,
            det             = 1,
            dataext         = 0,  # Extension containing image data
            specaxis        = 0,  # Spectral axis orientation
            specflip        = False,
            spatflip        = False,
            platescale      = 0.189,  # arcsec/pixel
            darkcurr        = 0.0,  # e-/pixel/hour
            saturation      = 2**16.,  # ADU
            nonlinear       = 0.9,  # Non-linearity coefficient
            mincounts       = -1e10,
            numamplifiers   = 1,
            # UPDATED: 2 elements for 2 amplifiers (was single value)
            gain            = np.array([1.65]), # TODO: make depending on readout mode
            ronoise         = np.array([4.67]), # TODO: make depending on readout mode
            # UPDATED: Added standard datasec and oscansec (full image, no overscan)
            datasec         = np.atleast_1d('[1:1024,1:4096]'),
            oscansec        = np.atleast_1d('[1025:1152,1:4096]'),
        )

        detector2_dict = dict(
            binning         = binning,
            det             = 2,
            dataext         = 1,  # Extension containing image data
            specaxis        = 0,  # Spectral axis orientation
            specflip        = True,
            spatflip        = True,
            platescale      = 0.189,  # arcsec/pixel
            darkcurr        = 0.0,  # e-/pixel/hour
            saturation      = 2**16.,  # ADU
            nonlinear       = 0.9,  # Non-linearity coefficient
            mincounts       = -1e10,
            numamplifiers   = 1,
            # UPDATED: 2 elements for 2 amplifiers (was single value)
            gain            = np.array([1.47]),
            ronoise         = np.array([5.06]),
            # UPDATED: Added standard datasec and oscansec (full image, no overscan)
            datasec         = np.atleast_1d('[1:1024,1:4096]'),
            oscansec        = np.atleast_1d('[1025:1152,1:4096]'),
        )

        detectors = [detector1_dict, detector2_dict]
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
        return ['binning', 'dispname', 'decker', 'filter1']

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
        return {'standard': 'dispname',
                'pixelflat': ["binning", "decker", 'filter1'],
                'illumflat': ["binning"],
                'arc': ["binning", "decker", "dispname", "filter1"],
                'tilt': ["binning", "decker", "dispname", "filter1"],
                'trace': ["binning", "decker"],
                'bias': 'binning',
                'dark': 'binning'}

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the PypeIt parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        # Start with instrument wide
        par = super().config_specific_par(scifile, inp_par=inp_par)

        #TODO: determine from the mask name if the science frame is taken with the MOS or longslit
        #if self.get_meta_value(scifile, 'idname') == 'OsirisMOS':
        #    par['reduce']['findobj']['find_trim_edge'] = [1,1]
        #    par['calibrations']['slitedges']['sync_predict'] = 'pca'
        #    par['calibrations']['slitedges']['det_buffer'] = 1
        #elif self.get_meta_value(scifile, 'idname') == 'OsirisLongSlitSpectroscopy':
        #    # Do not tweak the slit edges for longslit
        #    par['calibrations']['flatfield']['tweak_slits'] = False

        # Wavelength calibration and setup-dependent parameters
        if self.get_meta_value(scifile, 'dispname') == 'VPH-Red':
            par['calibrations']['wavelengths']['lamps'] = ['HeI','NeI','ArI']
            par['calibrations']['wavelengths']['reid_arxiv'] = 'magellan_ldss3_VPH-Red.fits'
            par['calibrations']['flatfield']['slit_illum_finecorr'] = False
            par['reduce']['cube']['wave_min'] = 5_500.0
            par['reduce']['cube']['wave_max'] = 11_000.0
        elif self.get_meta_value(scifile, 'dispname') == 'VPH-Blue':
            par['calibrations']['wavelengths']['lamps'] = ['HeI','NeI','ArI']
            par['calibrations']['wavelengths']['reid_arxiv'] = 'magellan_ldss3_VPH-Blue.fits'
            par['calibrations']['flatfield']['slit_illum_finecorr'] = False
            par['reduce']['cube']['wave_min'] = 3_500.0
            par['reduce']['cube']['wave_max'] = 6_500.0
        elif self.get_meta_value(scifile, 'dispname') == 'VPH-All':
            par['calibrations']['wavelengths']['lamps'] = ['HeI','NeI','ArI']
            par['calibrations']['wavelengths']['reid_arxiv'] = 'magellan_ldss3_VPH-All.fits'
            par['calibrations']['flatfield']['slit_illum_finecorr'] = False
            par['reduce']['findobj']['find_min_max'] = [500, 2051]
            par['reduce']['cube']['wave_min'] = 4_000.0
            par['reduce']['cube']['wave_max'] = 11_000.0
        else:
            msgs.warn('magellan_ldss3.py: template arc missing for this grism! Trying holy-grail...')
            par['calibrations']['wavelengths']['method'] = 'holy-grail'

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
        if ftype in ['pixelflat', 'trace']:
            return good_exp & check_files_are_pixelflat(fitstbl)
        if ftype in ['illumflat']:
            return good_exp & check_files_are_illumflat(fitstbl)
        if ftype in ['science']:
            return good_exp & check_files_are_science(fitstbl)
        if ftype in ['dark']:
            return good_exp & (fitstbl['idname'] == 'Dark')
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)


    def get_mosaic_par(self, mosaic, hdu=None, msc_ord=5):
        """
        Return the hard-coded parameters needed to construct detector mosaics
        from unbinned images.

        The parameters expect the images to be trimmed and oriented to follow
        the PypeIt shape convention of ``(nspec,nspat)``.  For returned
        lists, the length of the list is the same as the number of detectors in
        the mosaic, and they are ordered by the detector number.

        Args:
            mosaic (:obj:`tuple`):
                Tuple of detector numbers used to construct the mosaic.  Must be
                one among the list of possible mosaics as hard-coded by the
                :func:`allowed_mosaics` function.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent detector parameters are set to a
                default.  BEWARE: If ``hdu`` is not provided, the binning is
                assumed to be `1,1`, which will cause faults if applied to
                binned images!
            msc_ord (:obj:`int`, optional):
                Order of the interpolation used to construct the mosaic.

        Returns:
            :class:`~pypeit.images.mosaic.Mosaic`: Object with the mosaic *and*
            detector parameters.
        """

        # Validate the entered (list of) detector(s)
        nimg, _ = self.validate_det(mosaic)

        # Index of mosaic in list of allowed detector combinations
        mosaic_id = self.allowed_mosaics.index(mosaic)+1
        # Get the detectors
        detectors = np.array([self.get_detector_par(det, hdu=hdu) for det in mosaic])
        # Binning *must* be consistent for all detectors
        if any(d.binning != detectors[0].binning for d in detectors[1:]):
            msgs.error('Binning is somehow inconsistent between detectors in the mosaic!')

        # Collect the offsets and rotations for *all unbinned* detectors in the
        # full instrument, ordered by the number of the detector.  Detector
        # numbers must be sequential and 1-indexed.
        # See the mosaic documentattion.
        expected_shape = (2048, 4096)
        shift = np.array([(0.0, 0.0),
                          (1024,  0.0)])

        rotation = np.array([0.0, 0.0])

        # The binning and process image shape must be the same for all images in
        # the mosaic
        binning = tuple(int(b) for b in detectors[0].binning.split(','))
        shape = tuple(n // b for n, b in zip(expected_shape, binning))

        msc_sft = [None]*nimg
        msc_rot = [None]*nimg
        msc_tfm = [None]*nimg

        for i in range(nimg):
            msc_sft[i] = shift[i]
            msc_rot[i] = rotation[i]
            msc_tfm[i] = build_image_mosaic_transform(shape, msc_sft[i], msc_rot[i], binning)
        return Mosaic(mosaic_id, detectors, shape, np.array(msc_sft), np.array(msc_rot),
                      np.array(msc_tfm), msc_ord)

    @property
    def allowed_mosaics(self):
        """
        Return the list of allowed detector mosaics.

        Returns:
            :obj:`list`: List of tuples, where each tuple provides the 1-indexed
            detector numbers that can be combined into a mosaic and processed by
            PypeIt.
        """
        return [(1,2),]

    @property
    def default_mosaic(self):
        return self.allowed_mosaics[0]

    def get_rawimage(self, raw_file, det):
        """
        Read raw images and generate a few other bits and pieces
        that are key for image processing.

        Data are unpacked from the multi-extension HDU.  Function is
        based on :func:`pypeit.spectrographs.keck_deimos`

        Parameters
        ----------
        raw_file : :obj:`str`
            File to read
        det : :obj:`int`
            1-indexed detector to read

        Returns
        -------
        detector_par : :class:`pypeit.images.detector_container.DetectorContainer`
            Detector metadata parameters.
        raw_img : `numpy.ndarray`_
            Raw image for this detector.
        hdu : `astropy.io.fits.HDUList`_
            Opened fits file
        exptime : :obj:`float`
            Exposure time read from the file header
        rawdatasec_img : `numpy.ndarray`_
            Data (Science) section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        oscansec_img : `numpy.ndarray`_
            Overscan section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        """
        # Read
        msgs.info(f'Attempting to read LDSS3 file: {raw_file}')
        # NOTE: io.fits_open checks that the file exists
        hdu = io.fits_open(raw_file)

        # Validate the entered (list of) detector(s)
        nimg, _det = self.validate_det(det)

        # Grab the detector or mosaic parameters

        mosaic = None if nimg == 1 else self.get_mosaic_par(det, hdu=hdu)
        detectors = [self.get_detector_par(det, hdu=hdu)] if nimg == 1 else mosaic.detectors
        # Get post, pre-pix values
        noscan_cols = hdu[0].header['NBIASLNS']
        x_npix = hdu[0].header['NAXIS1']
        y_npix = hdu[0].header['NAXIS2']
        #x0, x_npix, y0, y_npix = np.array(parse.load_sections(detlsize)).flatten()

        # get the x and y binning factors...

        # get the chips to read in
        # DP: I don't know if this needs to still exist. I believe det is never None
        if det is None:
            chips = range(self.ndet)
        else:
            chips = [d-1 for d in _det]  # Indexing starts at 0 here

        # get final datasec and oscan size (it's the same for every chip so
        # it's safe to determine it outsize the loop)
        if det is None:
            image = np.zeros((nimg, y_npix-noscan_cols, x_npix-noscan_cols+noscan_cols))
            rawdatasec_img = np.zeros_like(image, dtype=int)
            oscansec_img = np.zeros_like(image, dtype=int)
        # One detector??
        else:
            # get final datasec and oscan size (it's the same for every chip so
            # it's safe to determine it outsize the loop)
            data, oscan = ldss3_read_1chip(hdu, chips[0] + 1)
            image = np.zeros((nimg, y_npix-noscan_cols, x_npix-noscan_cols+noscan_cols))
            rawdatasec_img = np.zeros_like(image, dtype=int)
            oscansec_img = np.zeros_like(image, dtype=int)

        def indexing(tt, noscan_cols, det=None):
            """
            Indexing for LDSS3-style single-amplifier chips with fixed sections:

            Returns x/y bounds in python (0-index, half-open) for:
            image[ii, y1:y2, x1:x2]      <- data
            image[ii, o_y1:o_y2, o_x1:o_x2] <- oscan

            if ii==0, overscan will be forced at the left
            if ii==1, overscan will be forced at the right
            This is needed for the mosaic otherwise the trimming does not work
            """
            # science region
            y1, y2 = 0, 4096

            # overscan region (postpix columns to the right)
            if tt == 0:
                o_x1, o_x2 = 0, int(noscan_cols)
                x1, x2 = int(noscan_cols), int(noscan_cols) + 1024

            elif tt == 1:
                x1, x2 = 0, 1024
                o_x1, o_x2 = x2, x2 + int(noscan_cols)

            else:
                msgs.error('Invalid chip number!')

            o_y1, o_y2 = y1, y2

            return x1, x2, y1, y2, o_x1, o_x2, o_y1, o_y2

        # Loop over the chips
        for ii, tt in enumerate(chips):
            data, oscan = ldss3_read_1chip(hdu, ii)


            x1, x2, y1, y2, o_x1, o_x2, o_y1, o_y2 = indexing(tt, noscan_cols, det=det)
            print('filling data for chip {0}, x1={1}, x2={2}, y1={3}, y2={4}'.format(tt, x1, x2, y1, y2))
            print('filling oscan for chip {0}, o_x1={1}, o_x2={2}, o_y1={3}, o_y2={4}'.format(tt, o_x1, o_x2, o_y1, o_y2))
            # Fill
            image[ii, y1:y2, x1:x2] = data
            rawdatasec_img[ii, y1:y2, x1:x2] = 1 # Amp
            image[ii, o_y1:o_y2, o_x1:o_x2] = oscan
            oscansec_img[ii, o_y1:o_y2, o_x1:o_x2] = 1 # Amp

        exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]

        if nimg == 1:
            return detectors[0], image[0], hdu, exptime, rawdatasec_img[0], oscansec_img[0]
        
        return mosaic, image, hdu, exptime, rawdatasec_img, oscansec_img


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
                Processed bias frame used to identify bad pixels

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        # Validate the entered (list of) detector(s)
        nimg, _det = self.validate_det(det)
        _det = list(_det)

        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)
        # NOTE: expand_dims does *not* copy the array.  We can edit it directly
        # because we've created it inside this function.
        _bpm_img = np.expand_dims(bpm_img, 0) if nimg == 1 else bpm_img

        return _bpm_img[0] if nimg == 1 else _bpm_img

class MagellanLDSS3MultiSlitSpectrograph(MagellanLDSS3Spectrograph):
    """
    Child class for Magellan/LDSS3 multi-slit spectroscopy mode.
    """
    name = 'magellan_ldss3_multi'
    supported = True
    pypeline = 'MultiSlit'  # Specifies the reduction pipeline
    ndet = 2

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        par['rdx']['detnum'] = [(1, 2)]

        # Wavelengths
        # UPDATED: Changed 'rms_threshold' to 'rms_thresh_frac_fwhm' (modern parameter name)
        # UPDATED: Removed deprecated parameters 'sigrej_final' and 'sigrej_first'
        par['calibrations']['wavelengths']['rms_thresh_frac_fwhm'] = 0.3
        par['calibrations']['wavelengths']['sigdetect'] = 5.0
        par['calibrations']['wavelengths']['fwhm'] = 5.0
        par['calibrations']['wavelengths']['n_first'] = 2
        par['calibrations']['wavelengths']['n_final'] = 4

        # UPDATED: Set grating-dependent match tolerance
        par['calibrations']['wavelengths']['match_toler'] = 0.5

        # Set slits and tilts parameters
        par['calibrations']['slitedges']['sobel_mode'] = 'constant'
        par['calibrations']['slitedges']['det_buffer'] = 20

        # Processing steps
        # UPDATED: Changed to 'use_overscan' (was already correct)
        turn_off = dict(use_overscan=False, use_darkimage=False)
        par.reset_all_processimages_par(**turn_off)


        # Good exposure times, we do not limit them
        par['calibrations']['standardframe']['exprng'] = [0, None]
        par['calibrations']['arcframe']['exprng'] = [0, None]
        par['calibrations']['darkframe']['exprng'] = [0, None]
        par['scienceframe']['exprng'] = [0, None]

        par["calibrations"]["biasframe"]["process"]["combine"] = "median"
        par["calibrations"]["darkframe"]["process"]["combine"] = "median"


        return par

    def list_detectors(self, mosaic=False):
        """
        List the *names* of the detectors in this spectrograph.

        This is primarily used :func:`~pypeit.slittrace.average_maskdef_offset`
        to measure the mean offset between the measured and expected slit
        locations.

        Detectors separated along the dispersion direction should be ordered
        along the first axis of the returned array.  For example, Keck/DEIMOS
        returns:

        .. code-block:: python

            dets = np.array([['DET01', 'DET02', 'DET03', 'DET04'],
                             ['DET05', 'DET06', 'DET07', 'DET08']])

        such that all the bluest detectors are in ``dets[0]``, and the slits
        found in detectors 1 and 5 are just from the blue and red counterparts
        of the same slit.

        Args:
            mosaic (:obj:`bool`, optional):
                Is this a mosaic reduction?
                It is used to determine how to list the detector, i.e., 'DET' or 'MSC'.

        Returns:
            `numpy.ndarray`_: The list of detectors in a `numpy.ndarray`_.  If
            the array is 2D, there are detectors separated along the dispersion
            axis.
        """
        dets = super().list_detectors(mosaic=mosaic)
        return dets if mosaic else dets.reshape(2,-1)




# UPDATED: Helper functions for frame classification
def check_files_are_arc(fitstbl, lamps=['He', 'Ne', 'Ar']):
    """
    UPDATED: Added docstring and improved logic with list comprehension.
    Identify arc frames by checking for lamp names in the target field.
    Arc object names are of the form: MASK_NAME + LAMP_NAMES
    Example: ATM3a2_v1 HeNeAr
    """
    object_names = fitstbl['target']
    mask_names = fitstbl['decker']
    exp_types = fitstbl['idname']
    mask = []

    for name, mask_name, exptype in zip(object_names, mask_names, exp_types):
        data = name.split(" ")
        if "arc" in name.lower() or "lamp" in name.lower():
            mask.append(True)
            continue
        if len(data) > 1 and exptype == 'Object' and data[0] == mask_name:
            # UPDATED: Simplified lamp check using any()
            have_lamp = any(lamp in data[1] for lamp in lamps)
            is_arc = "arc" in name.lower()
            if is_arc or have_lamp:
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(False)

    return np.array(mask)

def check_files_are_science(fitstbl):
    """
    UPDATED: Added docstring and simplified with list comprehension.
    Identify science frames by checking for 'science' in the target name.
    """
    object_names = fitstbl['target']
    # UPDATED: Simplified logic
    mask = ["science" in name.lower() for name in object_names]
    return np.array(mask)

def check_files_are_pixelflat(fitstbl):
    """
    UPDATED: Added docstring and simplified with list comprehension.
    Identify pixel flat frames by checking for 'flat' in the target name.
    """
    object_names = fitstbl['target']
    grating = fitstbl['dispname']
    # UPDATED: Simplified logic
    mask = [ ("flat" in name.lower() and disp.lower()!="open" and "arc" not in name.lower()) for name, disp in zip(object_names, grating)]
    return np.array(mask)

def check_files_are_illumflat(fitstbl):
    """
    UPDATED: Added docstring and simplified with list comprehension.
    Identify pixel flat frames by checking for 'flat' in the target name.
    """
    object_names = fitstbl['target']
    grating = fitstbl['dispname']
    # UPDATED: Simplified logic
    mask = [ ("flat" in name.lower() and disp.lower()=="open" and "arc" not in name.lower()) for name, disp in zip(object_names, grating)]
    return np.array(mask)

def ldss3_read_1chip(hdu,chipno):
    """ Read one of the LDSS3 detectors

    Args:
        hdu (astropy.io.fits.HDUList):
        chipno (int):

    Returns:
        np.ndarray, np.ndarray:
            data, oscan
    """

    # Extract datasec from header
    datsec = hdu[chipno].header['DATASEC']
    oscansec = hdu[chipno].header['BIASSEC']


    x1_dat, x2_dat, y1_dat, y2_dat = np.array(parse.load_sections(datsec)).flatten()
    x1_oscan, x2_oscan, y1_oscan, y2_oscan = np.array(parse.load_sections(oscansec)).flatten()
    # This rotates the image to be increasing wavelength to the top
    #data = np.rot90((hdu[chipno].data).T, k=2)
    #nx=data.shape[0]
    #ny=data.shape[1]


    # Science data
    fullimage = hdu[chipno].data
    data = fullimage[x1_dat:x2_dat,y1_dat:y2_dat]

    # Overscan
    oscan = fullimage[x1_oscan:x2_oscan,y1_oscan:y2_oscan]

    # Flip as needed
    if chipno == 1:
        data = np.flipud(data)
        oscan = np.flipud(oscan)
    # Return
    return data, oscan