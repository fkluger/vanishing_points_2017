import config
import sys
sys.path.insert(0,config.caffe_path)
import evaluation
import scipy.io as io
import os
import pickle
import scipy.ndimage as ndimage
import probability_functions as prob
import calc_horizon as ch
import time
import matplotlib.pyplot as plt
import argparse
from auc import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--yud', dest='yud', action='store_true', help='Run benchmark on YUD')
parser.add_argument('--ecd', dest='ecd', action='store_true', help='Run benchmark on ECD')
parser.add_argument('--hlw', dest='hlw', action='store_true', help='Run benchmark on HLW')
parser.add_argument('--result_dir', default='/tmp/', type=str, help='Directory to store (intermediate) results')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
parser.add_argument('--update_datalist', dest='update_datalist', action='store_true', help='Update the dataset list')
parser.add_argument('--update_datafiles', dest='update_datafiles', action='store_true', help='Update the dataset files')
parser.add_argument('--run_cnn', dest='run_cnn', action='store_true', help='Evaluate CNN on the data')
parser.add_argument('--run_em', dest='run_em', action='store_true', help='Run EM refinement on the data')
args = parser.parse_args()

update_list = args.update_datalist
update_pickles = args.update_datafiles
update_cnn = args.run_cnn
update_em = args.run_em

GPU_ID = args.gpu

image_mean = config.cnn_mean_path
model_def = config.cnn_config_path
model_weights = config.cnn_weights_path

if args.yud:
    data_folder = {"name": "york", "source_folder": config.yud_path,
                   "destination_folder": os.path.join(args.result_dir, "york")}
elif args.ecd:
    data_folder = {"name": "eurasian", "source_folder": config.ecd_path,
                   "destination_folder": os.path.join(args.result_dir, "eurasian")}
else:
    assert False

em_config = {'distance_measure': 'angle', 'use_weights': True, 'do_split': True, 'do_merge': True}

dataset = evaluation.get_data_list(data_folder['source_folder'], data_folder['destination_folder'],
                             'default_net', "", "0",
                             distance_measure=em_config['distance_measure'],
                             use_weights=em_config['use_weights'], do_split=em_config['do_split'],
                             do_merge=em_config['do_merge'], update=update_list, dataset_name=data_folder["name"])

evaluation.create_data_pickles(dataset, update=update_pickles, cnn_input_size=500,
                               target_size=800 if (args.ecd or args.hlw) else None)

if update_cnn:
    evaluation.run_cnn(dataset, mean_file=image_mean, model_def=model_def, model_weights=model_weights, gpu=GPU_ID)

if update_em:
    evaluation.run_em(dataset)


start = 25 if (args.yud or args.ecd) else 0
end = 10000

err_cutoff = 0.25

theta_vmin = np.pi / 10
N_vp = 20

dataset_name = data_folder["name"]

print "dataset name: ", dataset['name']

if dataset_name == "york":
    cameraParams = io.loadmat(os.path.join(config.yud_path, "cameraParameters.mat"))

    f = cameraParams['focal'][0,0]
    ps = cameraParams['pixelSize'][0,0]
    pp = cameraParams['pp'][0,:]

    K = np.matrix([[f/ps, 0, 13], [0, f/ps, -11], [0,0,1]])
    S = np.matrix([[2.0/640, 0, 0], [0, 2.0/640, 0], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
metadata = []
if dataset_name == "horizon":
    import csv
    with open(os.path.join(config.hlw_path, "metadata.csv"), 'rb') as csvfile:
        metadata_file = csv.reader(csvfile)
        for row in metadata_file:
            row[0] = row[0].split('/')[-1]
            row[0] = row[0].split('.')[0]
            metadata.append(row)

errors = []

indices = range(len(dataset['image_files']))

start_time = time.time()

count = 0
for idx in indices:

    image_file = dataset['image_files'][idx]
    data_file = dataset['pickle_files'][idx]

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

    basename = os.path.splitext(image_file)[0]

    if not os.path.isfile(data_file):
        print "file not found"
        continue

    path0, imageID = os.path.split(basename)
    path1, rest = os.path.split(path0)

    scale = np.maximum(imageWidth, imageHeight)

    trueVPs = None
    trueHorizon = None

    if dataset_name == "york":
        matGTpath = "%s/%s/%sGroundTruthVP_CamParams.mat" % (path1, imageID, imageID)

        GTdata = io.loadmat(matGTpath)

        trueVPs = np.matrix(GTdata['vp'])
        trueVPs_3d = trueVPs.copy()

        trueVPs = K * trueVPs

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

        trueVPs[:,0] -= imageWidth/2
        trueVPs[:,1] -= imageHeight/2
        trueVPs[:,1] *= -1
        trueVPs[:,0:2] /= scale/2

        trueHorizon = io.loadmat(horizonMatPath)['horizon']
        trueHorizon = np.squeeze(trueHorizon)

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

        for row in metadata:
            if row[0] == image_basename:
                imageWidth_orig = float(row[2])
                imageHeight_orig = float(row[1])
                scale_orig = np.maximum(imageWidth_orig, imageHeight_orig)
                thP1 = np.array([ float(row[3]), float(row[4]), 1])
                thP2 = np.array([ float(row[5]), float(row[6]), 1])
                thP1[0:2] /= scale_orig/2.0
                thP2[0:2] /= scale_orig/2.0
                trueHorizon = np.cross(thP1, thP2)
                break

    with open(data_file, 'rb') as fp:
        datum = pickle.load(fp)

    sphere_image = datum['sphere_image'] if 'sphere_image' in datum else None
    prediction = datum['cnn_prediction'][::-1,:] if 'cnn_prediction' in datum else None

    lines_dict = datum['lines'] if 'lines' in datum else None
    em_result = datum['EM_result'] if 'EM_result' in datum else None

    assert not (em_result is None), "no EM result!"

    (hP1, hP2, zVP, hVP1, hVP2, best_combo) = ch.calculate_horizon_and_ortho_vp(em_result, maxbest=N_vp,
                                                                                theta_vmin=theta_vmin)

    vps = em_result['vp']
    counts = em_result['counts']
    vp_assoc = em_result['vp_assoc']
    angles = prob.calc_angles(vps.shape[0], vps)
    ls = lines_dict['line_segments']
    ll = lines_dict['lines']

    num_best = np.minimum(N_vp, vps.shape[0])

    horizon_line = np.cross(hP1, hP2)

    if not (trueHorizon is None):
        thP1 = np.cross(trueHorizon, np.array([1, 0, 1]))
        thP2 = np.cross(trueHorizon, np.array([-1, 0, 1]))
        thP1 /= thP1[2]
        thP2 /= thP2[2]

        max_error = np.maximum(np.abs(hP1[1]-thP1[1]), np.abs(hP2[1]-thP2[1]))/2 * scale*1.0/imageHeight

        print "max_error: ", max_error

        errors.append(max_error)

end_time = time.time()

print "time elapsed: ", end_time-start_time

error_arr = np.array(errors)
auc, plot_points = calc_auc(error_arr, cutoff=err_cutoff)

print "AUC: ", auc

plt.figure()
ax = plt.subplot()
ax.plot(plot_points[:,0], plot_points[:,1], '-', lw=2, c='b')
ax.set_xlabel('horizon error', fontsize=18)
ax.set_ylabel('fraction of images', fontsize=18)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
ax.axis([0,err_cutoff,0,1])
plt.show()
