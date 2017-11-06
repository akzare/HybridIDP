#!/usr/bin/env python

'''
Constants.py

Description: Constant such as image path and parameters for stereo

'''

class Constants(object):
    """
    Create objects with read-only (constant) attributes.
    Example:
        Nums = Constants(ONE=1, PI=3.14159, DefaultWidth=100.0)
        print 10 + Nums.PI
        print '----- Following line is deliberate ValueError -----'
        Nums.PI = 22
    """

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    # NOTE: This is only called if self lacks the attribute.
    # So it does not interfere with get of 'self._d', etc.
    def __getattr__(self, name):
        return self._d[name]

    # ASSUMES '_..' attribute is OK to set. Need this to initialize 'self._d', etc.
    #If use as keys, they won't be constant.
    def __setattr__(self, name, value):
        if (name[0] == '_'):
            super(Constants, self).__setattr__(name, value)
        else:
            raise ValueError("setattr while locked", self)
