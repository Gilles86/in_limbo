#!/usr/bin/env python

from distutils.core import setup

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('in_limbo')
    return config

def main():
    print "YOO"
    from numpy.distutils.core import setup
    setup(name='in_limbo',
          version='0.1',
          description='Sandwich estimation for fMRI',
          author='Gilles de Hollander',
          author_email='g.dehollander@uva.nl',
          url='http://www.gillesdehollander.nl',
          packages=['in_limbo'],
          configuration=configuration
         )

if __name__ == '__main__':
    main()
