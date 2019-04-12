import numpy as np
import scipy.io as io
import os
import vp_localisation as vp
import pickle
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import probability_functions as prob
import sklearn.metrics
import calc_horizon as ch
import time

from py.result_check.deprecated.camera_params import calcFocal

start = 25
end = 10000
indices = None
maxbest = 20
show_histograms = False

show_plots = False
# show_plots = False

# use_old_idx = True
use_old_idx = False

# second_em = True
second_em = False
# both_em = True
both_em = False

err_cutoff = 0.25

minVPdist = np.pi/10 #np.pi*0.1

legend_title = "YUD"
graph_color = 'g'

# dataset_name = "eurasian"
dataset_name = "york"
# dataset_name = "horizon"
# dataset_name = "kitti"
# dataset_name = "other"

# result_folder = "results-scaled"
result_folder = "results"

cnn_size = 20
# cnn_size = 40

# cnn_type = "v0"
# cnn_type = "v4"
cnn_type = "v5"
# cnn_type = "v5r"
# cnn_type = "v5p"
# cnn_type = "v6"

dist_measure = "angle"
# dist_measure = "dotprod"
use_weights = "weights"
splitmerge = ""
do_split = "%ssplit" % splitmerge
do_merge = "%smerge" % splitmerge

cnn_version = "newdata_500px_%dx%d_%s" % (cnn_size, cnn_size, cnn_type)

# dataset_list = "/home/kluger/ma/data/real_world/%s/%s/%s.pkl" % (dataset_name, result_folder, cnn_version)
# dataset_list = "/data/kluger/ma/data/real_world/%s/%s/%s_%s_%s_%s_%s.pkl" % (dataset_name, result_folder, cnn_version, dist_measure, use_weights, do_split, do_merge)
# dataset_list = "/home/kluger/tmp/%s/%s/%s_%s_%s_%s_%s.pkl" % (dataset_name, result_folder, cnn_version, dist_measure, use_weights, do_split, do_merge)
dataset_list = "/home/kluger/ma/data/real_world/%s/%s/%s_%s_%s_%s_%s.pkl" % (dataset_name, result_folder, cnn_version, dist_measure, use_weights, do_split, do_merge)

print "dataset_list: ", dataset_list

with open(dataset_list, 'rb') as fp:
    dataset = pickle.load(fp)

print "dataset name: ", dataset['name']

if dataset_name == "york":
    cameraParams = io.loadmat("/data/kluger/ma/data/real_world/york/cameraParameters.mat")

    f = cameraParams['focal'][0,0]
    ps = cameraParams['pixelSize'][0,0]
    pp = cameraParams['pp'][0,:]

    print f, ps, pp

    K = np.matrix([[f/ps, 0, 13], [0, f/ps, -11], [0,0,1]])
    S = np.matrix([[2.0/640, 0, 0], [0, 2.0/640, 0], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    print K
    print K_inv

metadata = []
if dataset_name == "horizon":
    import csv
    with open('/data/kluger/ma/data/real_world/horizon/metadata.csv', 'rb') as csvfile:
        metadata_file = csv.reader(csvfile)
        for row in metadata_file:
            row[0] = row[0].split('/')[-1]
            row[0] = row[0].split('.')[0]
            metadata.append(row)
            # print row

    # exit(0)

errors = []
angle_errors = []
z_angle_errors = []

f_errors = []

false_pos = []
false_neg = []
true_pos = []

false_pos3 = []
false_neg3 = []
true_pos3 = []

recalls = []

count = 0

if use_old_idx:
    with open("tmp/error_idx.pkl", 'rb') as fp:
        indices = pickle.load(fp)

if indices is None:
    indices = range(len(dataset['image_files']))
else:
    indices = indices
    # indices = indices[::-1]

start_time = time.time()

for idx in indices:

    # print "index: ", idx

    image_file = dataset['image_files'][idx]
    data_file = dataset['pickle_files'][idx]

    image_file = image_file.replace("scaled", "vanilla")
    if dataset_name == "eurasian":
        image_file = image_file.replace("png", "jpg")

    count += 1

    if count <= start: continue
    if count > end: break

    print "image file: ", image_file
    if not os.path.isfile(image_file):
        print "file not found"
        continue

    image = ndimage.imread(image_file)

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    # print "image size: ", imageWidth, imageHeight

    basename = os.path.splitext(image_file)[0]
    # data_file = "%s.data.pkl" % basename
    print "data file: ", data_file
    if not os.path.isfile(data_file):
        print "file not found"
        continue

    # print "basename: ", basename
    path0, imageID = os.path.split(basename)
    path1, rest = os.path.split(path0)
    # print imageID

    scale = np.maximum(imageWidth, imageHeight)

    trueVPs = None
    trueHorizon = None

    if dataset_name == "york":
        # matGTpath = "%s/%s/%sGroundTruthVP_Orthogonal_CamParams.mat" % (path1, imageID, imageID)
        matGTpath = "%s/%s/%sGroundTruthVP_CamParams.mat" % (path1, imageID, imageID)
        # print matGTpath

        GTdata = io.loadmat(matGTpath)

        # trueVPs = np.matrix(GTdata['vp_orthogonal'])
        trueVPs = np.matrix(GTdata['vp'])
        trueVPs_3d = trueVPs.copy()
        # trueVPs = np.matrix(GTdata['vp'])
        # print "trueVPs:\n", trueVPs

        trueVPs = K * trueVPs
        # print "trueVPs:\n", trueVPs

        trueVPs[:,0] /= trueVPs[2,0]
        trueVPs[:,1] /= trueVPs[2,1]
        trueVPs[:,2] /= trueVPs[2,2]

        trueVPs = S * trueVPs

        tVP1 = np.array(trueVPs[:,0])[:,0]
        tVP1 /= tVP1[2]
        tVP2 = np.array(trueVPs[:,1])[:,0]
        tVP2 /= tVP2[2]
        tVP3 = np.array(trueVPs[:,2])[:,0]
        tVP3 /= tVP3[2]

        trueHorizon= np.cross(tVP1, tVP3)

        trueVPs = np.vstack([tVP1, tVP2, tVP3])


    elif dataset_name == "eurasian":

        horizonMatPath = "%shor.mat" % basename
        vpMatPath = "%sVP.mat" % basename

        trueZenith = io.loadmat(vpMatPath)['zenith']
        trueHorVPs = io.loadmat(vpMatPath)['hor_points']

        trueVPs = np.ones((trueHorVPs.shape[0]+1, 3))
        trueVPs[:,0:2] = np.vstack([trueZenith, trueHorVPs])

        # print "true horvps: \n", trueHorVPs

        trueVPs[:,0] -= imageWidth/2
        trueVPs[:,1] -= imageHeight/2
        trueVPs[:,1] *= -1
        trueVPs[:,0:2] /= scale/2

        # print "true vps: ", trueVPs
        # print "zenith: ", trueZenith
        # print " hor vp: ", trueHorVPs

        trueHorizon = io.loadmat(horizonMatPath)['horizon']
        trueHorizon = np.squeeze(trueHorizon)

        # print "trueHorizon: ", trueHorizon

        thP1 = np.cross(trueHorizon, np.array([-1, 0, imageWidth]))
        thP2 = np.cross(trueHorizon, np.array([-1, 0, 0]))
        thP1 /= thP1[2]
        thP2 /= thP2[2]

        thP1[0] -= imageWidth/2.0
        thP2[0] -= imageWidth/2.0
        thP1[1] -= imageHeight/2.0
        thP2[1] -= imageHeight/2.0
        thP1[1] *= -1
        thP2[1] *= -1

        thP1[0:2] /= scale/2.0
        thP2[0:2] /= scale/2.0

        trueHorizon = np.cross(thP1, thP2)

    elif dataset_name == "horizon":

        image_basename = image_file.split('/')[-1]
        image_basename = image_basename.split('.')[0]
        # print "image base name: ", image_basename

        for row in metadata:
            if row[0] == image_basename:
                imageWidth_orig = float(row[2])
                imageHeight_orig = float(row[1])
                scale_orig = np.maximum(imageWidth_orig, imageHeight_orig)
                thP1 = np.array([ float(row[3]), float(row[4]), 1 ])
                thP2 = np.array([ float(row[5]), float(row[6]), 1 ])
                thP1[0:2] /= scale_orig/2.0
                thP2[0:2] /= scale_orig/2.0
                trueHorizon = np.cross(thP1, thP2)
                break



    with open(data_file, 'rb') as fp:
        datum = pickle.load(fp)

    if second_em is False:
        sphere_image = datum['sphere_image'] if 'sphere_image' in datum else None
        prediction = datum['cnn_prediction'][::-1,:] if 'cnn_prediction' in datum else None
    else:
        sphere_image = datum['sphere_image_2'] if 'sphere_image_2' in datum else None
        prediction = datum['cnn_prediction_2'][::-1,:] if 'cnn_prediction_2' in datum else None

    # scipy.misc.imsave("/home/kluger/sphere1.png", sphere_image)

    if second_em:
        lines_dict = datum['lines_2'] if 'lines_2' in datum else None
        em_result = datum['EM_result_2'] if 'EM_result_2' in datum else None
    else:
        lines_dict = datum['lines'] if 'lines' in datum else None
        em_result = datum['EM_result'] if 'EM_result' in datum else None

    if both_em:
        em_result_2 = datum['EM_result_2'] if 'EM_result_2' in datum else None

    if show_plots:
        fig2 = plt.figure()
        fig2.suptitle(image_file)
        ax2 = plt.subplot2grid((2, 2), (0, 0))
        ax3 = plt.subplot2grid((2, 2), (0, 1), projection='3d')
        # ax4 = plt.subplot2grid((2, 2), (1, 0))
        # plt.axis('off')
        # ax5 = plt.subplot2grid((2, 2), (1, 1))
        # plt.axis('off')

        fig5 = plt.figure()
        ax4 = plt.subplot()
        plt.axis('off')

        fig6 = plt.figure()
        ax5 = plt.subplot()
        plt.axis('off')

        fig3 = plt.figure()
        fig3.suptitle(image_file)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        plt.axis('off')
        ax6 = plt.subplot2grid((2, 2), (1, 0))
        ax7 = plt.subplot2grid((2, 2), (0, 1))
        ax8 = plt.subplot2grid((2, 2), (1, 1))

        fig4 = plt.figure()
        ax9 = plt.subplot()
        plt.axis('off')
        ax9.imshow(image)
        # ax9.autoscale(enable=False)

        ax1.imshow(image)
        if not (sphere_image is None):
            ax2.imshow(sphere_image)
            ax5.imshow(sphere_image, cmap='Greys_r')
            ax5.autoscale(enable=False)

        if not (prediction is None):
            ax4.imshow(prediction, interpolation='none', cmap='Greys_r')
            ax4.autoscale(enable=False)


        if not (lines_dict is None):
            ls = lines_dict['line_segments']

            for li in range(ls.shape[0]):
                ax6.plot( [ls[li,0], ls[li,2]], [ls[li,1], ls[li,3]], 'k-' )

    if not (em_result is None):

        if both_em:
            em_result['vp'] = np.vstack([em_result['vp'], em_result_2['vp']])
            em_result['counts'] = np.hstack([em_result['counts'], em_result_2['counts']])
            em_result['counts_weighted'] = np.hstack([em_result['counts_weighted'], em_result_2['counts_weighted']])
            em_result['sigma'] = np.hstack([em_result['sigma'], em_result_2['sigma']])

        ( hP1, hP2, zVP, hVP1, hVP2, best_combo ) = ch.calculate_horizon_and_ortho_vp(em_result, maxbest=maxbest, minVPdist=minVPdist) #np.pi/5.0

        vps = em_result['vp']
        counts = em_result['counts']
        # counts = em_result['counts_weighted']
        vp_assoc = em_result['vp_assoc']
        angles = prob.calc_angles(vps.shape[0], vps)
        ls = lines_dict['line_segments']
        ll = lines_dict['lines']

        num_best = np.minimum(maxbest, vps.shape[0])

        # print "no. of lines: ", ll.shape[0]

        # print "counts: \n", counts



        # print "best combo ", best_combo
        # print(vps[best_combo[0]])
        # print(vps[best_combo[1]])
        # print(vps[best_combo[2]])

        if show_plots:
            vp.plot_result(ax3, ax5, vps, angles, counts, best_combo, imgSize=sphere_image.shape[0], markersize=2)
            vp.plot_result(None, ax4, vps, angles, counts, best_combo, imgSize=prediction.shape[0], markersize=2)

            if not (trueVPs is None):
                # tvps = np.vstack([tVP1/np.linalg.norm(tVP1),tVP2/np.linalg.norm(tVP2),tVP3/np.linalg.norm(tVP3)])
                tvps = trueVPs.copy()
                for i in range(tvps.shape[0]): tvps[i,:] /= np.linalg.norm(tvps[i,:])
                # print("tvps:")
                # print(tvps)
                anglesTrue = prob.calc_angles(tvps.shape[0], tvps)
                # print(anglesTrue)
                vp.plot_result(None, ax4, tvps, anglesTrue, None, None, imgSize=prediction.shape[0], stdMark='co', markersize=2)
                vp.plot_result(None, ax5, tvps, anglesTrue, None, None, imgSize=sphere_image.shape[0], stdMark='co', markersize=2)


        horizon_line = np.cross(hP1, hP2)

        if not (trueHorizon is None):
            thP1 = np.cross(trueHorizon, np.array([1, 0, 1]))
            thP2 = np.cross(trueHorizon, np.array([-1, 0, 1]))
            thP1 /= thP1[2]
            thP2 /= thP2[2]

            max_error = np.maximum(np.abs(hP1[1]-thP1[1]), np.abs(hP2[1]-thP2[1]))/2 * scale*1.0/imageHeight

            # max_error = hP1[1]-thP1[1] if np.abs(hP1[1]-thP1[1]) > np.abs(hP2[1]-thP2[1]) else hP2[1]-thP2[1]
            # max_error = max_error / 2 * scale*1.0/imageHeight

            print "max_error: ", max_error

            errors.append(max_error)

        langles = np.zeros(ll.shape[0])
        lcosphi = np.zeros(ll.shape[0])
        llen = np.zeros(ll.shape[0])

        if dataset_name == "york":

            zVPs = zVP.copy()
            hVP1s = hVP1.copy()
            hVP2s = hVP2.copy()
            hP1s = hP1.copy()
            hP2s = hP2.copy()
            zVPs[0] = zVPs[0] * scale / 2.0 #+ imageWidth / 2.0
            hVP1s[0] = hVP1s[0] * scale / 2.0 #+ imageWidth / 2.0
            hVP2s[0] = hVP2s[0] * scale / 2.0 #+ imageWidth / 2.0
            hP1s[0] = hP1s[0] * scale / 2.0 #+ imageWidth / 2.0
            hP2s[0] = hP2s[0] * scale / 2.0 #+ imageWidth / 2.0
            zVPs[1] = -zVPs[1] * scale / 2.0 #+ imageHeight / 2.0
            hVP1s[1] = -hVP1s[1] * scale / 2.0 #+ imageHeight / 2.0
            hVP2s[1] = -hVP2s[1] * scale / 2.0 #+ imageHeight / 2.0
            hP1s[1] = -hP1s[1] * scale / 2.0 #+ imageHeight / 2.0
            hP2s[1] = -hP2s[1] * scale / 2.0 #+ imageHeight / 2.0

            h = np.cross(hP1s, hP2s)

            F = calcFocal(zVPs, hVP1s, hVP2s, h, imageWidth, imageHeight)

            F_err = np.abs(F-f/ps)

            f_errors.append(F_err)

            # K_est = cameraParamsFromOrthoVP(zVPs, hVP1s, hVP2s, imageWidth, imageHeight)
            # print K_est


            zVP_3d = np.array(K.I * S.I * np.matrix(zVP).T).squeeze()
            hVP1_3d = np.array(K.I * S.I * np.matrix(hVP1).T).squeeze()
            hVP2_3d = np.array(K.I * S.I * np.matrix(hVP2).T).squeeze()


            ztVP_3d = np.array(trueVPs_3d[:,1]).squeeze()
            htVP1_3d = np.array(trueVPs_3d[:,0]).squeeze()
            htVP2_3d = np.array(trueVPs_3d[:,2]).squeeze()

            zAng = np.abs( np.arccos( np.abs(np.dot(zVP_3d, ztVP_3d)) / ( np.linalg.norm(zVP_3d)*np.linalg.norm(ztVP_3d) ) ) ) * 180.0/np.pi
            h1Ang_a = np.abs( np.arccos( np.abs(np.dot(hVP1_3d, htVP1_3d)) / ( np.linalg.norm(hVP1_3d)*np.linalg.norm(htVP1_3d) ) ) ) * 180.0/np.pi
            h1Ang_b = np.abs( np.arccos( np.abs(np.dot(hVP1_3d, htVP2_3d)) / ( np.linalg.norm(hVP1_3d)*np.linalg.norm(htVP2_3d) ) ) ) * 180.0/np.pi
            h2Ang_a = np.abs( np.arccos( np.abs(np.dot(hVP2_3d, htVP1_3d)) / ( np.linalg.norm(hVP2_3d)*np.linalg.norm(htVP1_3d) ) ) ) * 180.0/np.pi
            h2Ang_b = np.abs( np.arccos( np.abs(np.dot(hVP2_3d, htVP2_3d)) / ( np.linalg.norm(hVP2_3d)*np.linalg.norm(htVP2_3d) ) ) ) * 180.0/np.pi

            h1Ang = np.minimum(h1Ang_a, h1Ang_b)
            h2Ang = np.minimum(h2Ang_a, h2Ang_b)

            if h1Ang_a + h2Ang_b < h1Ang_b + h2Ang_a:
                h1Ang = h1Ang_a
                h2Ang = h2Ang_b
            else:
                h1Ang = h1Ang_b
                h2Ang = h2Ang_a

            angle_errors.extend([zAng, h1Ang, h2Ang])

            z_angle_errors.append(zAng/180.0*np.pi)

            th_ = 5

            tp = 0
            if zAng < th_:
                tp += 1
            if h1Ang < th_:
                tp += 1
            if h1Ang < th_:
                tp += 1

            fn = 3-tp
            fp = 3-tp

            true_pos3.append(tp)
            false_neg3.append(fn)
            false_pos3.append(fp)

            # print "precision: ", tp/(tp+fp)
            # print "recall: ", tp/(tp+fn)

            # vps_to_test = vps[best_vps].copy()
            vps_to_test = np.vstack([hVP1, hVP2, zVP])
            vp_dir3 =  np.array(K.I * S.I * np.matrix(vps_to_test).T).T.squeeze()

            all_angle_errors = np.zeros((vp_dir3.shape[0], 3))

            for ia in range(vp_dir3.shape[0]):
                all_angle_errors[ia,0] = np.abs( np.arccos( np.abs(np.dot(vp_dir3[ia,:], ztVP_3d)) / ( np.linalg.norm(vp_dir3[ia,:])*np.linalg.norm(ztVP_3d) ) ) ) * 180.0/np.pi
                all_angle_errors[ia,1] = np.abs( np.arccos( np.abs(np.dot(vp_dir3[ia,:], htVP1_3d)) / ( np.linalg.norm(vp_dir3[ia,:])*np.linalg.norm(htVP1_3d) ) ) ) * 180.0/np.pi
                all_angle_errors[ia,2] = np.abs( np.arccos( np.abs(np.dot(vp_dir3[ia,:], htVP2_3d)) / ( np.linalg.norm(vp_dir3[ia,:])*np.linalg.norm(htVP2_3d) ) ) ) * 180.0/np.pi

            min_angles = np.min(all_angle_errors, axis=0)



            min_angles[min_angles<th_] = 1
            min_angles[min_angles>=th_] = 0

            tp = min_angles.sum()
            fp = vp_dir3.shape[0]-tp
            fn = 3-tp

            # print "precision: ", tp/(tp+fp)
            # print "recall: ", tp/(tp+fn)

            true_pos.append(tp)
            false_neg.append(fn)
            false_pos.append(fp)

        # elif dataset_name == "eurasian":
        #     num_true_vps = trueVPs.shape[0]
        #     num_recall_vps = 0
        #     for i in range(num_true_vps):
        #         trueVP = trueVPs[i,:].copy()
        #         trueVP /= trueVP[2]
        #         tVPlen = np.linalg.norm(trueVP[0:2])
        #         for j in range(vps.shape[0]):
        #             estVP = vps[j,:].copy()
        #             estVP /= estVP[2]
        #             diffLen = np.linalg.norm(estVP - trueVP)
        #             if diffLen/tVPlen < 0.20:
        #                 num_recall_vps += 1
        #                 break
        #
        #     recall = num_recall_vps*1.0/num_true_vps
        #
        #     print "recall: ", recall
        #
        #     recalls.append(recall)

        if show_plots:

            # best3_vps = best_vps[best_combo]
            best3_vps = best_combo.copy()
            rest3_vps = np.array([])
            for iv in range(vps.shape[0]):
                if not (iv in best3_vps):
                    rest3_vps = np.append(rest3_vps, iv)

            if not (trueHorizon is None):
                ax6.plot([thP1[0], thP2[0]], [thP1[1], thP2[1]], 'g-', lw=5)

                thP1s = thP1.copy()
                thP2s = thP2.copy()
                thP1s[0] = thP1s[0] * scale / 2.0 + imageWidth / 2
                thP2s[0] = thP2s[0] * scale / 2.0 + imageWidth / 2
                thP1s[1] = -thP1s[1] * scale / 2.0 + imageHeight / 2
                thP2s[1] = -thP2s[1] * scale / 2.0 + imageHeight / 2

                ax9.plot([thP1s[0], thP2s[0]], [thP1s[1], thP2s[1]], 'c--', lw=15)

            ax6.plot([hP1[0], hP2[0]], [hP1[1], hP2[1]], 'r-', lw=5)
            ax6.axis([-1,1,-1,1])

            jet = plt.get_cmap('brg')
            cNormBest = colors.Normalize(vmin=0, vmax=best3_vps.size - 1)
            cNormRest = colors.Normalize(vmin=0, vmax=rest3_vps.size - 1)
            scalarMapBest = cmx.ScalarMappable(norm=cNormBest, cmap=jet)
            scalarMapRest = cmx.ScalarMappable(norm=cNormRest, cmap=jet)

            lsc = ls.copy()
            lsc[:,0] = lsc[:,0]*scale/2.0 + imageWidth/2
            lsc[:,2] = lsc[:,2]*scale/2.0 + imageWidth/2
            lsc[:,1] = -lsc[:,1]*scale/2.0 + imageHeight/2
            lsc[:,3] = -lsc[:,3]*scale/2.0 + imageHeight/2

            for li in range(ls.shape[0]):
                vtemp = vps[vp_assoc[li]]

                if vp_assoc[li] in best3_vps:
                    idx_best = np.squeeze(np.where(best3_vps == vp_assoc[li])[0])
                    colorVal = scalarMapBest.to_rgba(idx_best)
                    if idx_best == 0: colorVal = (0,0,1)
                    elif idx_best == 1: colorVal = (0,1,0)
                    elif idx_best == 2: colorVal = (1,1,0)
                    ax7.plot([ls[li, 0], ls[li, 2]], [ls[li, 1], ls[li, 3]], c=colorVal)

                    ax9.plot([lsc[li, 0], lsc[li, 2]], [lsc[li, 1], lsc[li, 3]], c=colorVal, lw=5)

                elif vp_assoc[li] in rest3_vps:
                    idx_rest = np.squeeze(np.where(rest3_vps == vp_assoc[li])[0])
                    colorVal = scalarMapRest.to_rgba(idx_rest)
                    ax8.plot([ls[li, 0], ls[li, 2]], [ls[li, 1], ls[li, 3]], c=colorVal)
                elif vp_assoc[li] == -1:
                    idx_rest = np.squeeze(np.where(rest3_vps == vp_assoc[li])[0])
                    ax8.plot([ls[li, 0], ls[li, 2]], [ls[li, 1], ls[li, 3]], 'k-')

                norm2 = np.linalg.norm(ll[li, 0:2], ord=2)
                cosphi = np.clip(ll[li, 1] / norm2, -1, 1)
                langles[li] = np.arccos(cosphi) * 180.0 / np.pi
                lcosphi[li] = cosphi

                llen[li] = np.linalg.norm([ls[li, 0] - ls[li, 2], ls[li, 1] - ls[li, 3]], ord=2)

            sigma = em_result['sigma']
            for vi in range(best3_vps.shape[0]):
                # print "sigma: ", sigma[best3_vps[vi]]
                vptemp = vps[best3_vps[vi]]

                vptemp /= vptemp[2]

                if vptemp[0] > -1 and vptemp[0] < 1 and vptemp[1] > -1 and vptemp[1] < 1:
                    colorVal = scalarMapBest.to_rgba(vi)
                    # msize = sigma[best_vps[vi]]*1e5
                    msize = 5
                    ax7.plot([vptemp[0]], [vptemp[1]], 'o', c=colorVal, markersize=msize)


            ax6.set_aspect('equal', 'datalim')
            ax6.axis([-1,1,-1,1])
            ax7.set_aspect('equal', 'datalim')
            ax7.axis([-1,1,-1,1])
            ax8.set_aspect('equal', 'datalim')
            ax8.axis([-1,1,-1,1])


            if not (trueHorizon is None):
                thP1s = thP1.copy()
                thP2s = thP2.copy()
                thP1s[0] = thP1s[0] * scale / 2.0 + imageWidth / 2
                thP2s[0] = thP2s[0] * scale / 2.0 + imageWidth / 2
                thP1s[1] = -thP1s[1] * scale / 2.0 + imageHeight / 2
                thP2s[1] = -thP2s[1] * scale / 2.0 + imageHeight / 2

                ax9.plot([thP1s[0], thP2s[0]], [thP1s[1], thP2s[1]], 'c--', lw=20)


            hP1s = hP1.copy()
            hP2s = hP2.copy()
            hP1s[0] = hP1s[0] * scale / 2.0 + imageWidth / 2
            hP2s[0] = hP2s[0] * scale / 2.0 + imageWidth / 2
            hP1s[1] = -hP1s[1] * scale / 2.0 + imageHeight / 2
            hP2s[1] = -hP2s[1] * scale / 2.0 + imageHeight / 2

            ax9.plot([hP1s[0], hP2s[0]], [hP1s[1], hP2s[1]], 'm--', lw=20)



            if show_histograms:

                plt.figure()
                plt.hist(llen, 100)

                dm = em_result['decision_metric']

                score_ratios = np.zeros((ll.shape[0]))

                for li in range(ll.shape[0]):
                    one_dm = dm[:,li]
                    best_dm = np.argsort(one_dm)
                    best_dm = best_dm[::-1]
                    ratio = one_dm[best_dm[1]] / one_dm[best_dm[0]]
                    score_ratios[li] = ratio

                plt.figure()
                plt.hist(score_ratios, 100)



            else:
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()

            plt.show()
    else:
        print "no EM results!"

end_time = time.time()

print "time elapsed: ", end_time-start_time

# print "errors: ", errors

error_arr = np.array(errors)
error_arr_idx = np.argsort(error_arr)
error_arr = np.sort(error_arr)

num_values = len(errors)

plot_points = np.zeros((num_values,2))


for i in range(num_values):
    fraction = (i+1) * 1.0/num_values
    value = error_arr[i]
    plot_points[i,1] = fraction
    plot_points[i,0] = value
    if i > 0:
        lastvalue = error_arr[i-1]
        if lastvalue < err_cutoff and value > err_cutoff:
            midfraction = (lastvalue*plot_points[i-1,1] + value*fraction) / (value+lastvalue)

if plot_points[-1,0] < err_cutoff:
    plot_points = np.vstack([plot_points, np.array([err_cutoff,1])])
else:
    plot_points = np.vstack([plot_points, np.array([err_cutoff,midfraction])])

# for eth in range(25,205,5):
#     auc = sklearn.metrics.auc(plot_points[plot_points[:,0]<=eth/100.0,0], plot_points[plot_points[:,0]<=eth/100.0,1])
#     print "%d, auc: " % eth, auc / (eth/100.0)

auc = sklearn.metrics.auc(plot_points[plot_points[:,0]<=err_cutoff,0], plot_points[plot_points[:,0]<=err_cutoff,1])
print "auc: ", auc / err_cutoff

# print "errors: \n", error_arr
# print error_arr_idx

# print "mean error: ", np.mean(error_arr)
#
#
# if dataset_name == "york":
#     print "angle errors: %f -- %f -- %f -- %f" % (np.mean(angle_errors), np.std(angle_errors), np.min(angle_errors), np.max(angle_errors))
#
#     print "precision3: ", np.sum(true_pos3)*1.0/(np.sum(true_pos3)+np.sum(false_pos3))
#     print "recall3: ", np.sum(true_pos3)*1.0/(np.sum(true_pos3)+np.sum(false_neg3))
#     print "precision: ", np.sum(true_pos)*1.0/(np.sum(true_pos)+np.sum(false_pos))
#     print "recall: ", np.sum(true_pos)*1.0/(np.sum(true_pos)+np.sum(false_neg))
#
#     print "zAngle: ", np.mean(z_angle_errors), " / ", np.std(z_angle_errors)
#
#     f_error_arr = np.array(f_errors)
#     f_error_arr_idx = np.argsort(f_error_arr)
#     f_error_arr = np.sort(f_error_arr)
#
#     f_num_values = len(f_errors)
#
#     f_plot_points = np.zeros((f_num_values, 2))
#
#     f_err_cutoff = 250
#
#     midfraction = 1
#
#     print "F errors:\n",f_error_arr
#
#     for i in range(f_num_values):
#         fraction = (i + 1) * 1.0 / f_num_values
#         value = f_error_arr[i]
#         f_plot_points[i, 1] = fraction
#         f_plot_points[i, 0] = value
#         if i > 0:
#             lastvalue = f_error_arr[i - 1]
#             if lastvalue < f_err_cutoff and value > f_err_cutoff:
#                 midfraction = (lastvalue * f_plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)
#
#     # if f_plot_points[-1, 0] < f_err_cutoff:
#     #     f_plot_points = np.vstack([f_plot_points, np.array([f_err_cutoff, 1])])
#     # else:
#     #     f_plot_points = np.vstack([f_plot_points, np.array([f_err_cutoff, midfraction])])
#     #
#     plt.figure()
#     axf = plt.subplot()
#     # ax.plot(plot_points[:,0], plot_points[:,1], '-', lw=2, label=legend_title, c=graph_color)
#     axf.plot(f_plot_points[:, 0], f_plot_points[:, 1], '-', lw=2, c=graph_color)
#     # axf.set_xlabel('F error', fontsize=18)
#     # axf.set_ylabel('fraction of images', fontsize=18)
#     # # ax.legend(loc='upper right')
#     #
#     # plt.setp(axf.get_xticklabels(), fontsize=18)
#     # plt.setp(axf.get_yticklabels(), fontsize=18)
#     axf.axis([0, f_err_cutoff, 0, 1])
#     # plt.show()


plt.figure()
ax = plt.subplot()
# ax.plot(plot_points[:,0], plot_points[:,1], '-', lw=2, label=legend_title, c=graph_color)
ax.plot(plot_points[:,0], plot_points[:,1], '-', lw=2, c=graph_color)
ax.set_xlabel('horizon error', fontsize=18)
ax.set_ylabel('fraction of images', fontsize=18)
# ax.legend(loc='upper right')

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
ax.axis([0,err_cutoff,0,1])
plt.show()

# if len(recalls) > 0:
#     print "average recall: ", np.mean(recalls)


# with open("tmp/error_idx.pkl", 'wb') as pickle_file:
#     pickle.dump(error_arr_idx, pickle_file, -1)
