#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script enables the viewing of a FITS file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import subprocess

from astropy.io import fits

from pypit import pyputils
from pypit import arlris

def parser(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file', type = str, default = None, help = 'FITS file')
    parser.add_argument("--list", default=False, help="List the extensions only?", action="store_true")
    parser.add_argument('--raw_lris', action="store_true")
    parser.add_argument('--exten', type=int, help="FITS extension")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    # List only?
    if args.list:
        hdu = fits.open(args.file)
        print(hdu.info())
        return

    kludge_fil = 'tmp_ginga.fits'

    # Setup for PYPIT imports
    msgs = pyputils.get_dummy_logger()

    # Extension
    if args.exten is not None:
        hdu = fits.open(args.file)
        img = hdu[args.exten].data
        # Write
        msgs.warn('Writing kludge file to {:s}'.format(kludge_fil))
        hdunew = fits.PrimaryHDU(img)
        hdulist = fits.HDUList([hdunew])
        hdulist.writeto(kludge_fil,clobber=True)
        #
        args.file = kludge_fil

    # RAW_LRIS??
    if args.raw_lris:
        # 
        img, head, _ = arlris.read_lris(args.file)
        # Generate hdu
        hdu = fits.PrimaryHDU(img)
        hdulist = fits.HDUList([hdu])
        # Write
        msgs.warn('Writing kludge file to {:s}'.format(kludge_fil))
        hdulist.writeto(kludge_fil,clobber=True)
        args.file = kludge_fil

    # Spawn ginga
    subprocess.call(["ginga", args.file])

    if args.raw_lris:
        msgs.warn('Removing kludge file {:s}'.format(kludge_fil))
        subprocess.call(["rm", args.file])


