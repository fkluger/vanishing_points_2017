import matplotlib.pyplot as plt
import numpy as np
import probability_functions as prob
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.ndimage as ndimage
import scipy.misc
import coordinate_conversion as coconv


def show_em_result(datum, image_file, maxbest=4, trueVPs=None, target_size=None, horizon=None):

    image = ndimage.imread(image_file)

    if target_size is not None:
        imshape = image.shape
        print(imshape)
        max_dim = np.max(imshape)
        resize_factor = target_size * 1. / max_dim
        image = scipy.misc.imresize(image, resize_factor, interp='bilinear', mode=None)

    width = image.shape[1]
    height = image.shape[0]
    scale = np.maximum(width, height)

    sphere_image = datum['sphere_image'] if 'sphere_image' in datum else None
    prediction = datum['cnn_prediction'][::-1, :] if 'cnn_prediction' in datum else None

    lines_dict = datum['lines'] if 'lines' in datum else None
    em_result = datum['EM_result'] if 'EM_result' in datum else None

    fig1 = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)  # original image
    ax2 = plt.subplot2grid((2, 2), (0, 1))  # all lines
    ax3 = plt.subplot2grid((2, 2), (1, 1))  # significant lines

    if image.ndim > 2:
        ax1.imshow(image)
    else:
        ax1.imshow(image, cmap='gray')
    ax1.autoscale(enable=False)

    if not (sphere_image is None):
        ax2.imshow(sphere_image, cmap='Greys_r')
        ax2.autoscale(enable=False)

    if not (prediction is None):
        ax3.imshow(prediction, cmap='Greys_r', interpolation='none')
        ax3.autoscale(enable=False)

    if not (lines_dict is None):
        ls = lines_dict['line_segments']
        lsc = ls.copy()
        lsc[:, 0] = lsc[:, 0] * scale / 2.0 + width / 2.0
        lsc[:, 2] = lsc[:, 2] * scale / 2.0 + width / 2.0
        lsc[:, 1] = -lsc[:, 1] * scale / 2.0 + height / 2.0
        lsc[:, 3] = -lsc[:, 3] * scale / 2.0 + height / 2.0

    if em_result is not None:
        vps = em_result['vp']
        counts = em_result['counts']
        vp_assoc = em_result['vp_assoc']
        angles = prob.calc_angles(vps.shape[0], vps)
        ls = lines_dict['line_segments']
        ll = lines_dict['lines']

        print("lines: ", ls.shape)

        best_vps = np.argsort(counts)
        best_vps = best_vps[::-1]
        best_vps = best_vps[0:maxbest]

        plot_result(None, ax2, vps, angles, counts, best_vps, imgSize=sphere_image.shape[0])
        plot_result(None, ax3, vps, angles, counts, best_vps, imgSize=prediction.shape[0])

        if not (trueVPs is None):
            anglesTrue = prob.calc_angles(trueVPs.shape[0], trueVPs)
            plot_result(None, ax2, trueVPs, anglesTrue, None, None, imgSize=sphere_image.shape[0], stdMark='co')
            plot_result(None, ax3, trueVPs, anglesTrue, None, None, imgSize=prediction.shape[0], stdMark='co')

        jet = plt.get_cmap('jet')
        cNormBest = colors.Normalize(vmin=0, vmax=best_vps.size - 1)
        scalarMapBest = cmx.ScalarMappable(norm=cNormBest, cmap=jet)

        langles = np.zeros(ll.shape[0])
        lcosphi = np.zeros(ll.shape[0])
        llen = np.zeros(ll.shape[0])

        for li in range(ls.shape[0]):
            if vp_assoc[li] in best_vps:
                idx_best = np.squeeze(np.where(best_vps == vp_assoc[li])[0])
                colorVal = scalarMapBest.to_rgba(idx_best)
                ax1.plot([lsc[li, 0], lsc[li, 2]], [lsc[li, 1], lsc[li, 3]], c=colorVal, lw=5)

            norm2 = np.linalg.norm(ll[li, 0:2], ord=2)
            cosphi = np.clip(ll[li, 1] / norm2, -1, 1)
            langles[li] = np.arccos(cosphi) * 180.0 / np.pi
            lcosphi[li] = cosphi

            llen[li] = np.linalg.norm([ls[li, 0] - ls[li, 2], ls[li, 1] - ls[li, 3]], ord=2)

        if horizon is not None:
            ax1.plot([horizon[0][0], horizon[1][0]], [horizon[0][1], horizon[1][1]], c='g', lw=10)

    plt.show()


def plot_result(ax1, ax2, vps, angles, vp_counts, best=None, imgSize=250, stdMark='yo', markersize=1):

    if vp_counts is not None:
        pg = vp_counts * 1.0 / np.sum(vp_counts)
    else:
        pg = np.ones(vps.shape[0])*0.1

    gvp = vps
    gang = angles

    for j in range(gvp.shape[0]):

        scaling = 100

        mark = stdMark
        if not (best is None):
            if j in best:
                mark = 'go'

        if not (ax1 is None):
            ax1.plot([gvp[j, 0]], [gvp[j, 1]], [gvp[j, 2]], mark, markersize=markersize*np.minimum(np.maximum(pg[j] * scaling,6),20))
        if not (ax2 is None):
            angle = gang[j, :]
            pos = coconv.angleToIndex(angle, (imgSize, imgSize))
            ax2.plot(pos[0], imgSize-1-pos[1], mark, markersize=markersize*np.minimum(np.maximum(pg[j] * scaling,6),20), alpha=0.6)
