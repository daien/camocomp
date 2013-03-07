"""
Get the valid area of a compensated video
=========================================


Extract a track (sequence of bounding boxes) which describes the area of the
video that contains only pixels of the original (non-compensated video)


Main function:   `extract_track`

"""


import numpy as np
import cv2


def get_mask(img, valid_intensity_threshold=1):
    """ Return a binary mask for the inner part of a remapped image

    Parameters
    ----------
    img: numpy array,
         an image resulting from the remapping step of the stitching process
         (contains the warped original image embedded in a bigger rectangular
         black image)

    valid_intensity_threshold: int, optional, default: 1,
         minimum pixel intensity value to be considered as "valid"

    Returns
    -------
    bin_mask: 2D numpy array,
              the binary mask where non-zero entries denote valid pixels
              corresponding to the embedded warped image,

    Notes
    -----
    Assumes that very low intensity pixels in img are invalid (i.e. belong to
    the encompassing background black image). Therefore, the original image
    (before remapping) must have its zero-valued regions shifted towards
    (slightly) higher intensity values.

    """
    # binarize the image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # histogram equalization (super important!!!)
    img = cv2.equalizeHist(img)
    # blur it with median filter to preserve edges
    img = cv2.medianBlur(img, 5)
    # binarize by thresholding
    bin_img = (img > valid_intensity_threshold).astype('u1')
    h, w = bin_img.shape
    # flood fill the external connex non-valid (zero) parts from the corners
    bin_mask = np.zeros((h + 2, w + 2), dtype='u1')  # OpenCV: need +2
    if np.all(bin_img[:3, :3] == 0):
        # use upper left corner as seed (just update bin_mask)
        cv2.floodFill(
            bin_img, bin_mask, (0, 0), 255, flags=cv2.FLOODFILL_MASK_ONLY)
    if np.all(bin_img[:3, -3:] == 0):
        # use upper right corner as seed (just update bin_mask)
        cv2.floodFill(
            bin_img, bin_mask, (w - 1, 0), 255, flags=cv2.FLOODFILL_MASK_ONLY)
    if np.all(bin_img[-3:, :3] == 0):
        # use lower left corner as seed (just update bin_mask)
        cv2.floodFill(
            bin_img, bin_mask, (0, h - 1), 255, flags=cv2.FLOODFILL_MASK_ONLY)
    if np.all(bin_img[-3:, -3:] == 0):
        # use lower right corner as seed (just update bin_mask)
        cv2.floodFill(bin_img, bin_mask, (w - 1, h - 1), 255,
                      flags=cv2.FLOODFILL_MASK_ONLY)
    # invert and crop the bin_mask
    bin_mask = (bin_mask[1:-1, 1:-1] == 0).astype('u1')
    # XXX for visualization
    #cv2.imshow("z", img)
    #cv2.imshow("y", bin_mask * 255)
    #cv2.waitKey()
    return bin_mask


def get_largest_rect_in_warped_rect(bin_mask, max_poss=2e5, ini_offset=5):
    """ Get the largest rectangle containg only non-zero pixels in a 2D mask

    Parameters
    ----------
    bin_mask: 2D numpy array,
              the binary mask where non-zero entries denote valid pixels,

    max_poss: int, optional, default: 2e5,
              maximum number of possibilities tested

    ini_offset: int, optional, default: 5,
                offset (in pixels) by which the initial starting bounding box
                is shrunk

    Returns
    -------
    t, b, l, r: a quadruplet integers,
                the top, bottom, left, and right dimensions of the largest
                rectangle containing only valid pixels

    Notes
    -----
    The binary mask corresponds to the area of a frame after the remapping
    step of the stitching process. Its shape can be warped according to a
    variety of perspective transformations. In addition, the binarization used
    to obtain the mask is not perfect: the valid area migh contain "holes"
    (corresponding to large pitch-black areas in the remapped frame).

    Therefore, we use a simple brute force approach, which sequentially tests
    smaller and smaller candidate rectangles by looking whether their edges
    contain only valid pixels. This is far from perfect, but quite robust and
    very fast with respect to the other steps of the compensation process.
    """
    tot_h, tot_w = bin_mask.shape
    # get the barycenter of the non-zero points
    assert np.any(bin_mask), "All empty mask"
    mh, mw = np.mean(np.argwhere(bin_mask), axis=0).astype(int)
    assert 0 < mh < tot_h and 0 < mw < tot_w, \
            "BUG in barycenter ({0}, {1})".format(mh, mw)
    # get all possible rectangles
    min_lims = np.array([2, mh + ini_offset, 2, mw + ini_offset])
    max_lims = np.array(
        [mh - ini_offset, tot_h - 2, mw - ini_offset, tot_w - 2])
    n_poss = np.prod(max_lims - min_lims)
    assert n_poss > 0, "Too small bin_mask size"
    if n_poss < max_poss:
        step = 1
    else:
        # subsample the possible rectangles
        step = int(0.5 + (n_poss / float(max_poss)) ** 0.25)
    possible_bbs = [(t, b, l, r)
                    for t in range(min_lims[0], max_lims[0], step)
                    for b in range(min_lims[1], max_lims[1], step)
                    for l in range(min_lims[2], max_lims[2], step)
                    for r in range(min_lims[3], max_lims[3], step)
                    ]
    # sort by increasing area
    possible_bbs.sort(key=lambda (t, b, l, r): (b - t) * (r - l), reverse=True)
    # greedily find the largest included rectangle
    for t, b, l, r in possible_bbs:
        # XXX for visualization
        #img = bin_mask * 127
        #img[t:b, [l-1, l, l+1, r-1, r, r+1]] = 255
        #img[[t-1, t, t+1, b-1, b, b+1], l:r] = 255
        #cv2.imshow('y', img)
        #cv2.waitKey(5)
        # only test the rectangle boundaries (might be holes in the middle)
        if np.all(bin_mask[t, l:r]) and np.all(bin_mask[b, l:r]) and \
                np.all(bin_mask[t:b, l]) and np.all(bin_mask[t:b, r]):
            # largest including rectangle found
            return t, b, l, r
    # if we reach here, we failed
    raise ValueError('BUG: Failed to find the largest rectangle')


def get_bbox(img, crop=-3, bb_format='ulwh'):
    """ Return the largest bounding box containing only the inner part of a remapped image

    Parameters
    ----------
    img: numpy array,
         an image resulting from the remapping step of the stitching process
         (contains the warped original image embedded in a bigger rectangular
         black image)

    crop: int, optional, default: -3,
          - if < 0 : crop the largest inner rectangle by -crop pixels
          - if >= 0 : crop the tightest outer rectangle by crop pixels
                      (no search for the largest inner rectangle)

    bb_format: string,
               bounding box format: 'tblr' or 'ulwh' (cf. below)

    Returns
    -------
    bbox: a quadruplet integers,
          the rectangle containing only valid pixels:
          - (ulx, uly, width, height, frame) if bb_format == 'ulwh'
          - (top, bottom, left, right, frame) if bb_format == 'tblr'
          (0, 0, 0, 0) if the estimation failed.

    """
    assert bb_format in ('tblr', 'ulwh'), "Unknown bbox format %s" % bb_format
    if not isinstance(img, np.ndarray):
        # assume it's the path to the image file
        img = cv2.imread(img, 0)  # load as gray-scale
    try:
        # get the mask
        bin_mask = get_mask(img)
        # get top and bottom non-empty row indexes
        hnnz = np.nonzero(bin_mask.sum(axis=1))[0]
        # get left and right non-empty column indexes
        wnnz = np.nonzero(bin_mask.sum(axis=0))[0]
        # get the encompassing bounding box (top, bottom, left, right)
        et, eb = hnnz[0] + 1, hnnz[-1]
        el, er = wnnz[0] + 0, wnnz[-1]
        if crop < 0:
            # find largest inner rectangle
            t, b, l, r = get_largest_rect_in_warped_rect(
                bin_mask[et:eb, el:er])
            t += et - crop
            b += et + crop
            l += el - crop
            r += el + crop
        else:
            # just shrink encompasssing bb
            t = et + crop
            b = eb - crop
            l = el + crop
            r = er - crop
        # XXX for visualization
        #img[t:b, [l, r]] = 255
        #img[[t, b], l:r] = 255
        #cv2.imshow("x", img)
        #cv2.waitKey(5)
        if bb_format == 'tblr':
            return t, b, l, r
        else:
            ulx = l
            uly = t
            h = b - t
            w = r - l
            return ulx, uly, w, h
    except:
        # failure case: return 0,0,0,0 (to be interpolated)
        return 0, 0, 0, 0


def extract_track(frame_seq, bb_format, sample_name=''):
    """ Return the track containing the valid bounding boxes

    Parameters
    ----------
    frame_seq: iterable,
               the path to frame images or the images themselves (as numpy arrays)

    bb_format: string,
               bounding box format: 'tblr' or 'ulwh' (cf. below)

    Returns
    -------
    track: (n_frs, 5) ndarray,
           the sequence of bounding boxes
           - (ulx, uly, width, height, frame) if bb_format == 'ulwh'
           - (top, bottom, left, right, frame) if bb_format == 'tblr'
    """
    assert bb_format in ('tblr', 'ulwh'), "Unknown bbox format %s" % bb_format

    rt = len(frame_seq)

    track = np.empty((rt, 5), dtype=np.int32)
    track[:, 4] = np.arange(1, rt + 1)  # frames start from 1 in vwgeo
    track[:, :4] = np.array([get_bbox(img, bb_format=bb_format)
                             for img in frame_seq], dtype=np.int32)

    # interpolate failure cases
    failed = track[:, :4].sum(axis=1) == 0
    assert not np.all(failed), "BUG: all failed for {0}".format(sample_name)
    if failed[0]:
        for i in range(1, len(failed)):
            if not failed[i]:
                track[0, :4] = track[i, :4]
                break
        failed[0] = False
    if failed[-1]:
        for i in range(len(failed) - 2, 0, -1):
            if not failed[i]:
                track[-1, :4] = track[i, :4]
                break
        failed[-1] = False
    for i in range(1, len(failed) - 1):
        if failed[i]:
            pt = track[i - 1, :4]
            nt = None
            for j in range(i + 1, len(failed)):
                if not failed[j]:
                    nt = track[j, :4]
                    break
            assert nt is not None, "BUG: didn't find next frame for interpolation"
            track[i, :4] = (0.5 + 0.5 * (pt + nt)).astype(np.int32)
            print '# interpolated frame {0} of {1}'.format(i + 1, sample_name)

    return track
