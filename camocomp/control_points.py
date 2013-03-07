"""
Compute control-points using OpenCV
===================================

Compute interest points (using the OpenCV's SURF detector) on pairs of images,
match them, and then filter them out using RANSAC.


main function:    `gen_pairwise_surf_control_points`

"""


import numpy as np

import cv2
import hsi


def get_surf_kps(img_fn, img=None, center_out=0,
                 cness_thresh=1000, min_pts=10, max_pts=300):
    """ Return the opened gray-scale OpenCV image and its SURF keypoints

    Points "in the middle" of the frames are left out
    (center_out = proportioned of space left out).
    """
    assert center_out < 1, "Too high center part to remove"
    # initialize the SURF keypoint detector and descriptor
    surf = cv2.SURF(cness_thresh)
    # load the gray-scale image
    if img is None:
        img = cv2.imread(img_fn, 0)
    # detect and describe SURF keypoints
    cvkp, ds = surf.detect(img, None, None)
    # re-arrange the data properly
    ds.shape = (-1, surf.descriptorSize())  # reshape to (n_pts, desc_size)
    kp = np.array([p.pt for p in cvkp])
    cness = np.array([p.response for p in cvkp])
    # filter out points in the middle (likely to be on the moving actor)
    if center_out > 0:
        rx = img.shape[1]
        lb = center_out * 0.5 * rx
        ub = (1 - center_out * 0.5) * rx
        mask = (kp[:, 0] < lb) + (kp[:, 0] > ub)
        kp = kp[mask, :]
        ds = ds[mask, :]
        cness = cness[mask]
    # check we're within the limits
    if kp.shape[0] < min_pts:
        if cness_thresh > 100:
            # redo the whole thing with a lower threshold
            _, kp, ds = get_surf_kps(img_fn, img=img, center_out=center_out,
                                     min_pts=min_pts, max_pts=max_pts,
                                     cness_thresh=0.5 * cness_thresh)
        else:
            # we lowered the threshold too much and didn't find enough points
            raise ValueError('Degenerate image (e.g. black) or too high center_out')
    if kp.shape[0] > max_pts:
        # too many points, take those with max cornerness only
        cness_order = np.argsort(cness)[::-1]
        kp = kp[cness_order[:max_pts], :]
        ds = ds[cness_order[:max_pts], :]
    return img, kp, ds


def get_pairwise_matches(pos1, descs1, pos2, descs2, up_to=30):
    """ Get the matching local features from img1 to img2
    """
    assert pos1.shape[0] * pos2.shape[0] < 1e8, \
            "Too many points: increase cornerness threshold"
    assert pos1.shape[0] > 10 and pos1.shape[0] > 10, \
            "Not enough points: lower cornerness threshold"
    # get the similarities between all descriptors
    sims = np.dot(descs1, descs2.T)
    # get the best matches
    mi2 = sims.argmax(axis=1).squeeze()
    ms = sims.max(axis=1).squeeze()
    bmi1 = ms.argsort()[::-1][:up_to]
    bmi2 = mi2[bmi1]
    # return their positions
    bp1 = pos1[bmi1]
    bp2 = pos2[bmi2]
    return bp1, bp2


def gen_pairwise_surf_control_points(proj_file, img_fns, display=False):
    """ Use OpenCV for pairwise image matching
    """
    # get the kps of the first frame
    img1, kp1, ds1 = get_surf_kps(img_fns[0])
    # match the frame t with t+1
    cpoints = []
    for i2 in range(1, len(img_fns)):
        # get the kps of frame t+1
        img2, kp2, ds2 = get_surf_kps(img_fns[i2])
        # get the control points
        cp1, cp2 = get_pairwise_matches(kp1, ds1, kp2, ds2)
        # estimate the homography
        H, mask = cv2.findHomography(cp1, cp2, cv2.RANSAC)
        mask = mask.squeeze() > 0
        # display the matches and homography
        if display:
            homo_warp_image(img1, cp1, img2, cp2, H, mask)
        # filter out the outlier matches
        cp1 = cp1[mask]
        cp2 = cp2[mask]
        # add the control points
        cpoints.extend([hsi.ControlPoint(i2 - 1, x1, y1, i2, x2, y2)
                        for (x1, y1), (x2, y2) in zip(cp1, cp2)])
        # next -> cur
        img1, kp1, ds1 = img2, kp2, ds2
    # write to pto
    pano = hsi.Panorama()
    pano.readData(hsi.ifstream(proj_file))
    pano.setCtrlPoints(cpoints)
    pano.writeData(hsi.ofstream(proj_file))


# =============================================================================
# some visualization functions based on opencv
# =============================================================================


def draw_match(img1, p1, img2, p2, mask=None, H=None):
    """ Draw the matches found from img1 to img2
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(
            cv2.perspectiveTransform(
                corners.reshape(1, -1, 2), H).reshape(-1, 2) \
            + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if mask is None:
        mask = np.ones(len(p1), np.bool_)

    green = (63, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), mask):
        col = [red, green][inlier]
        if inlier:
            cv2.line(vis, (x1, y1), (x2 + w1, y2), col)
            cv2.circle(vis, (x1, y1), 4, col, 2)
            cv2.circle(vis, (x2 + w1, y2), 4, col, 2)
        else:
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 + w1 - r, y2 - r), (x2 + w1 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 + w1 - r, y2 + r), (x2 + w1 + r, y2 - r), col, thickness)
    return vis


def homo_warp_image(img1, pts1, img2, pts2, homo, mask, prev_homo=None):
    """ Show keypoint matches and the estimated homography
    """
    # if prev_homo is given, then we update homo with it (temporal smoothing)
    if not prev_homo is None:
        homo = 0.8 * homo + 0.2 * prev_homo
        #homo = np.dot(homo, prev_homo)
    # warp img2
    if img2.ndim == 2:
        img2 = img2[:, :, np.newaxis]
    wimg2 = np.zeros_like(img2)
    for chan in range(img2.shape[2]):
        _i2 = np.ascontiguousarray(img2[:, :, chan], dtype="f4")
        #wimg2[:,:,chan] = cv2.warpPerspective(_i2, homo, _i2.T.shape)
        zz = cv2.warpPerspective(_i2, homo, _i2.T.shape)
        zx, zy = np.where(zz > 0)
        wimg2[zx, zy, chan] = zz[zx, zy]
    wimg2 = wimg2.squeeze()
    # warp the matches in img2
    wpts2 = cv2.perspectiveTransform(pts2.reshape(1, -1, 2), homo).reshape(-1, 2)
    # show the kept matches
    vis = draw_match(img1, pts1, wimg2, wpts2, mask, homo)
    cv2.imshow("match", vis)
    cv2.waitKey()
