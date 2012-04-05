# Copyright 2010 Sancho McCann
# Author: Sancho McCann

import os

def safe_make(directory, quiet=False):
    try:
        os.makedirs(directory)
    except OSError:
        if not quiet:
            print '%s already exists' % directory

