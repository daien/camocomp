"""
Camera Motion Compensation
==========================

Generate a motion-stabilized video in which the camera motion is compensated.

Main function:    `generate_stabilized_video`

"""


import sys
import os
import shutil
import tempfile
import subprocess
from glob import glob

import numpy as np

import cv2
import hsi

from camocomp.control_points import gen_pairwise_surf_control_points
from camocomp.extract_valid_track import extract_track


def exec_shell(cmd_line, raise_on_err=False):
    """ Execute a shell statement in a subprocess

    Parameters
    ----------
    cmd_line: string,
              the command line to execute (verbatim)

    raise_on_err: boolean, optional, default: False,
                  whether to raise ValueError if something was dumped to stderr

    Returns
    -------
    stdout, stderr: strings containing the resulting output of the command

    """
    out, err = subprocess.Popen(
        cmd_line,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    if raise_on_err and err != "":
        raise ValueError("Error: cmd={}, stderr={}".format(cmd_line, err))
    return out.strip(), err.strip()


def duplicate_nonzero_img(img_fn):
    """ Make a copy of an image such that it has no 0-valued pixels

    Parameters
    ----------
    img_fn: string,
            path to an image file

    Returns
    -------
    out_img_fn: string,
                path to the output jpg image

    Notes
    -----
    Useful for the thresholding step in `extract_track`.
    """
    img = cv2.imread(img_fn)
    img[img == 0] = 5
    out_img_fn = img_fn[:-4] + '_dup.jpg'
    cv2.imwrite(out_img_fn, img)
    return out_img_fn


def pto_gen(img_fns, hfov, out_pto="project.pto"):
    """ Generate a Hugin .pto project file

    Parameters
    ----------
    img_fns: list,
             the (ordered) full paths to the video frames

    hfov: int,
          horizontal field of view in degrees
          (around 50 is ok for most non-fish-eye cameras)

    out_pto: string, optional, default: 'project.pto',
             output path to the generated panotools .pto file

    Notes
    -----
    Suitable as input for further tools such as the cpfind control-point
    generator.

    Inspired from pto_gen
    (http://hugin.sourceforge.net/docs/html/pto__gen_8cpp-source.html)
    but with some hacks to correct the generated m-line in the header.

    Uses the Hugin python scripting interface
    (http://wiki.panotools.org/Hugin_Scripting_Interface).
    """
    # projection type: 0 == rectilinear (2 == equirectangular)
    projection = 0
    assert projection >= 0, "Invalid projection number (%d)" % projection
    assert 1 <= hfov <= 360, "Invalid horizontal field of view (%d)" % hfov
    # hugin Panorama object
    pano = hsi.Panorama()
    # add the images in order
    for img_fn in img_fns:
        src_img = hsi.SrcPanoImage(img_fn)
        src_img.setProjection(projection)
        src_img.setHFOV(hfov)
        src_img.setExifCropFactor(1.0)
        pano.addImage(src_img)
    # check we added all of them
    n_inserted = pano.getNrOfImages()
    assert n_inserted == len(img_fns), "Didn't insert all images (%d < %d)" % \
            (n_inserted, len(img_fns))
    # output the .pto file
    pano.writeData(hsi.ofstream(out_pto + '.tmp'))  # same as pano.printPanoramaScript(...)
    # some bug in header: rewrite it manually (TODO through hsi?)
    with open(out_pto + '.tmp', 'r') as tmp_ff:
        with open(out_pto, 'w') as ff:
            # re-write the header
            ff.write(tmp_ff.readline())
            ff.write(tmp_ff.readline())
            # force jpeg for the p-line
            p_line = tmp_ff.readline().strip().split()
            assert p_line[0] == 'p', "BUG: should be a p-line"
            ff.write(' '.join(p_line[:7]) + ' n"JPEG q100"\n')
            # remove extra 'f' param in the m-line (screws everything up if left here...)
            m_line = tmp_ff.readline().strip().split()
            assert m_line[0] == 'm', "BUG: should be a m-line"
            ff.write(' '.join(m_line[:3]) + ' ' + ' '.join(m_line[4:]) + '\n')
            # write all other lines
            for l in tmp_ff.readlines():
                ff.write(l)
    os.remove(out_pto + '.tmp')


def optimize_geometry(proj_file, optim_vars, optim_ref_fov=False):
    """ Optimise the geometric parameters

    Parameters
    ----------
    proj_file: string,
               the path to the Hugin project file (.pto)

    optim_vars: string,
                the set of variables to optimize for (separated by '_')
                - v: view point
                - p: pitch
                - y: yaw
                - r: roll
                - Tr{X,Y,Z}: translation

    optim_ref_fov: boolean, optional, default: False,
                   whether to optimize the input's reference horizontal field
                   of view (risky)

    Returns
    -------
    optim_proj_file: string,
                     path of the Hugin project file containing the optimized
                     values for the desired variables.

    Notes
    -----
    This is (by far) the most time consuming operation
    (because of hugin's autooptimiser).

    This is also the most likely function where the compensation process tends
    to fail.
    """
    optim_proj_file = proj_file + '.optim.pto'
    # modify the input pto to specify the optimization variables
    pano = hsi.Panorama()
    pano.readData(hsi.ifstream(proj_file))
    n_imgs = pano.getNrOfImages()
    var_tup = tuple(optim_vars.split('_'))
    for v in var_tup:
        assert v in ('v', 'p', 'y', 'r', 'TrX', 'TrY', 'TrZ'), \
                "Unknown var {0} in {1}".format(v, optim_vars)
    optim_opts = [var_tup] * n_imgs  # fov, pitch, roll, yaw
    if optim_ref_fov:
        # optim only field of view for 1st (reference) frame
        optim_opts[0] = ('v')
        # Note: do not optim. pitch, roll and yaw for this one, weird otherwise
    else:
        # 1st reference frame has the same fov as those of the input images
        optim_opts[0] = ()
    pano.setOptimizeVector(optim_opts)
    pano.writeData(hsi.ofstream(proj_file))
    # perform the optimization (TODO through hsi?)
    cmd = "autooptimiser -n -s -o {opto} {pto}"  # leveling can screw things up
    exec_shell(cmd.format(pto=proj_file, opto=optim_proj_file))
    # check for too large output (e.g. too wide panning or traveling)
    pano = hsi.Panorama()
    pano.readData(hsi.ifstream(optim_proj_file))
    opts = pano.getOptions()
    oh = opts.getHeight()
    ow = opts.getWidth()
    if oh * ow > 1e3 * 5e3:
        raise ValueError(
            "Degenerate case: too big output size ({0}, {1})\n".format(ow, oh) + \
            "May be caused by too large panning or translations\n" + \
            "=> Possible fixes: use a different field of view parameter or " + \
            "optimize only for different variables than {0}\n".format(optim_vars))
    return optim_proj_file


# TODO also estimate the camera motion from the bounding boxes
def warp_crop_and_generate_video(out_avi, optim_proj_file, n_imgs,
                                 crop_out=False, out_codec='mjpeg',
                                 tmp_dir='/tmp'):
    """ Generate a stabilized and cropped video from remapped frames

    Parameters
    ----------
    out_avi: string,
             output path to the generated motion-stabilized video

    optim_proj_file: string,
                     path of the Hugin project file containing the optimized
                     values for the desired variables.
                     (obtained from the `optimize_geometry` function)

    n_imgs: int,
            number of images in the video

    crop_out: boolean, optional, default: False,
              automatically crop the output video.

    out_codec: string, optional, default: 'mjpeg',
               video codec to use for the output video

    tmp_dir: string, optional, default: '/tmp'
             temporary directory

    """

    # remapping to create the distorted frames in the full scene plane
    # (TODO through hsi?)
    rimgt = tmp_dir + '/remapped%04d.tif'
    cmd = "nona -m TIFF_m -o {tmp_dir}/remapped {pto}"
    exec_shell(cmd.format(tmp_dir=tmp_dir, pto=optim_proj_file))

    if crop_out:
        # get the bounding boxes for all frames
        bboxes = extract_track([rimgt % i for i in range(n_imgs)], bb_format='tblr')

        # get the global bounding box
        gt = np.min(bboxes[:, 0])
        gb = np.max(bboxes[:, 1])
        gl = np.min(bboxes[:, 2])
        gr = np.max(bboxes[:, 3])
        # make it a multiple of 2 (needed for some codecs like h.264)
        bboxes[:, 1] -= (gb - gt) % 2
        bboxes[:, 3] -= (gr - gl) % 2
        gb -= (gb - gt) % 2
        gr -= (gr - gl) % 2
        # crop all tiff files with the same global bbox
        for i in range(n_imgs):
            # load and crop color image
            img = np.ascontiguousarray(cv2.imread(rimgt % i)[gt:gb, gl:gr, :])
            # overwrite previous version with new cropped one
            cv2.imwrite(rimgt % i, img)
        # correct the bounding boxes to be wrt to the globally cropped frames
        bboxes[:, :2] -= gt
        bboxes[:, 2:4] -= gl
        # temporal smoothing
        # should do it in later stages only if necessary
        #if n_imgs > 11:
        #    from scipy.ndimage.filters import convolve1d
        #    win = np.hanning(11)
        #    win /= win.sum()
        #    bboxes = convolve1d(bboxes, win, axis=0)

        # save the bboxes in ulwh format
        track = np.copy(bboxes)
        track[:, [0, 1]] = bboxes[:, [2, 0]]
        track[:, 2] = bboxes[:, 3] - bboxes[:, 2]
        track[:, 3] = bboxes[:, 1] - bboxes[:, 0]
        out_track = out_avi[:-4] + '.npy'
        np.save(out_track, track)
        print "saved {0}".format(out_track)

    # generate the warped and cropped video
    # Note: directly cropping in ffmpeg (-croptop ...) doesn't work
    # TODO try: -vf crop=...?
    cmd = "ffmpeg -y -f image2 -i {rit} -vcodec {codec} -sameq -r 25 -an {avi}"
    exec_shell(cmd.format(rit=rimgt, codec=out_codec, avi=out_avi))
    print "saved {0}".format(out_avi)


def generate_stabilized_video(input_media, optim_vars='v_p_y', hfov=40,
                              out_avi="out.avi", crop_out=False):
    """ Motion stabilize a video using a stitching technique

    Parameters
    ----------
    input_media: string or list,
                 a video file name or the (ordered) paths to the video frames

    optim_vars: string, optional, default: 'v_p_y',
                geometric parameters to optimize (separated by '_')
                    - v: view point
                    - p: pitch
                    - y: yaw
                    - r: roll
                    - Tr{X,Y,Z}: translation

    hfov: int, optional, default: 40,
          horizontal field of view in degrees
          (around 40-60 is ok for most non-fish-eye cameras)

    out_avi: string, optional, default: 'out.avi',
             output path to the generated motion-stabilized video

    crop_out: boolean, optional, default: False,
              automatically crop the output video.

    """
    # create a temporary directory
    tmp_pref = 'tmp_mostab_' + out_avi.split('/')[-1] + '_'
    tmp_dir = tempfile.mkdtemp(prefix=tmp_pref, dir='.')
    try:
        # get the frames if necessary
        if isinstance(input_media, str):
            input_media = [input_media]
        if len(input_media) == 1:
            if input_media[0][-4:] in ('.avi', 'mpg', '.mp4'):
                # input arg == a video: dumps its frames
                exec_shell(
                    'ffmpeg -i {} -f image2 -sameq {}/origframe-%06d.jpg'.format(
                        input_media[0], tmp_dir))
                img_fns = glob('{}/origframe-*.jpg'.format(tmp_dir))
            else:
                # input arg: assume its directory containing jpg's or png's
                img_dir = input_media[0]
                img_fns = glob('{}/*.jpg'.format(img_dir))
                if len(img_fns) <= 0:
                    img_fns = glob('{}/*.png'.format(img_dir))
        else:
            img_fns = input_media

        if len(img_fns) <= 0:
            raise ValueError('Could not obtain frames from {}'.format(img_fns))

        # get the absolute paths
        img_fns = map(os.path.abspath, img_fns)

        # duplicate images such that there are no 0-valued pixels
        img_fns = map(duplicate_nonzero_img, img_fns)

        # pano tools project file
        proj_file = "%s/project.pto" % tmp_dir
        # initialize the pto project
        pto_gen(img_fns, hfov=hfov, out_pto=proj_file)

        # generate the control points with OpenCV's SURF + RANSAC
        gen_pairwise_surf_control_points(proj_file, img_fns)
        # prune the control points (TODO through hsi?)
        cmd = "cpclean -p -o {pto} {pto}"
        exec_shell(cmd.format(pto=proj_file))
        # add lines (TODO through hsi?)
        cmd = "linefind -o {pto} {pto}"
        exec_shell(cmd.format(pto=proj_file))

        # optimization of the geometry
        optim_proj_file = optimize_geometry(proj_file, optim_vars)
        # warp the frames and make a motion stabilized video
        warp_crop_and_generate_video(
            out_avi, optim_proj_file, len(img_fns), crop_out=crop_out,
            tmp_dir=tmp_dir)
    finally:
        # clean up
        shutil.rmtree(tmp_dir)
    sys.stdout.flush()
