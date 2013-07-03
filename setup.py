#!/usr/bin/env python

from distutils.core import setup

SHORT_DESCR = "CAmera MOtion COMPensation using image stiching techniques to generate stabilized videos"

try:
    LONG_DESCR = open('README.rst').read()
except IOError:
    LONG_DESCR = SHORT_DESCR

setup(
    name='camocomp',
    version='0.1',
    author='Adrien Gaidon',
    author_email='easy_to_guess@googleme.com',
    keywords='camera motion compensation, video stabilization, stitching, opencv, hugin',
    packages=['camocomp'],
    url='http://pypi.python.org/pypi/camocomp/',
    license='New BSD License',
    description=SHORT_DESCR,
    long_description=LONG_DESCR,
    platforms=["Linux"],
    requires=['numpy', 'ffmpeg', 'cv2', 'hsi'],
    scripts=['scripts/camocomp_video'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
    ]
)
