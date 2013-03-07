==========================
CAmera MOtion COMPensation
==========================


What is it
==========

**camocomp** is a Python package that can stabilize videos, i.e. generate a
video copy in which the camera motion is compensated. This results in a video
where the fixed background (e.g. buildings, roads) appears to be static.


What can it be used for
=======================

Camera motion compensation is useful for a variety of tasks, including

    - **stabilizing camera shake**
    - recovering the **camera motion** for video and scene analysis
    - differentiating between the **foreground motion** (e.g. of actors) and
      the motion caused by the moving camera (for **motion analysis**)


Where to get it
===============

The source code is currently hosted on GitHub at: http://github.com/daien/camocomp

Binary installers for the latest released version are available at the Python
Package Index::

    http://pypi.python.org/pypi/camocomp/

And via ``easy_install`` or ``pip``::

    easy_install camocomp
    pip install camocomp


Dependencies
============

    - `Numpy <http://www.numpy.org>`__: 1.6.1 or higher
    - `Hugin <http://hugin.sourceforge.net>`__: a recent version (around 2012)
    - `FFmpeg <http://ffmpeg.org/download.html>`__: a recent version (around 2012)
    - `OpenCV <http://opencv.willowgarage.com/wiki/>`__: version 2.4.1 or higher
      (fixes a bug of the ffmpeg wrapper) 

Note: this package relies on Hugin's python scripting interface (HSI):
http://wiki.panotools.org/Hugin_Scripting_Interface


Installation from sources
=========================

In the ``camocomp`` directory (same one where you found this file), execute::

    python setup.py install

Note: this only works on *nix platforms.


License
=======

New BSD License


How to use it
=============

We provide a utility script called ``camocomp_video`` that can generate a
stabilized copy of a video.

The video ``example_mocomp.avi`` in the ``example`` directory contains a
stabilized video obtained with the command::

    camocomp_video -o example_mocomp.avi -c  -v p_y -f 40 example.avi

Depending on your input videos, you might need to play around with the input
field of view parameter (`-f` option) and/or the variables to optimize
(*v*iewpoint, *p*itch, *y*aw, and *r*oll).


How does it work
================

It relies on image stitching techniques similar to the ones used to create
panoramas from multiple photos. This allows to compensate for a vast array of
time-varying camera motions (e.g. camera shake, pan, zoom, tilt).


Limitations
===========

The stitching approach faces the following limitations:

    - it assumes that a large part of each frame is the background;
    - it also assumes that the background is textured (in order to detect
      ``control points`` on the background);
    - the spatial extent of the camera motion must be rather limited (i.e.
      restricted panning or translation, such that the background covered is
      limited) in order to avoid an extravagantly large output resolution;
    - some camera motions are problematic (e.g. rotation around the subject);
    - finding the correct input field of view parameter might require some
      trial and error;
    - the stitching optimization step (using hugin's autooptimizer) is VERY slow.
