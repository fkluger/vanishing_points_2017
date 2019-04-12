import config
import sys
sys.path.insert(0,config.caffe_path)
import evaluation
import numpy as np
import scipy.io as io
import os
import pickle
import scipy.ndimage as ndimage
import probability_functions as prob
import sklearn.metrics
import calc_horizon as ch
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--yud', dest='yud', action='store_true', help='')
parser.add_argument('--ecd', dest='ecd', action='store_true', help='')
parser.add_argument('--hlw', dest='hlw', action='store_true', help='')
parser.add_argument('--result_folder', default='/tmp/', type=str, help='')
args = parser.parse_args()

update_list = False
update_pickles = False
update_cnn = True
update_em = True

GPU_ID = 3

# model_root = "/data/kluger/ma/caffe/vp_sphere_classification/models/alexnet/newdata_500px_20x20_v5"
image_mean = config.cnn_mean_path
model_def = config.cnn_config_path #model_root + "/deploy.prototxt"
model_weights = config.cnn_weights_path

if args.yud:
    data_folder = {"name": "york", "source_folder": config.yud_path,
                   "destination_folder": os.path.join(args.result_folder, "york")}
elif args.ecd:
    data_folder = {"name": "eurasian", "source_folder": config.ecd_path,
                   "destination_folder": os.path.join(args.result_folder, "eurasian")}
else:
    assert False

em_config = {'distance_measure': 'angle', 'use_weights': True, 'do_split': True, 'do_merge': True}

dataset = evaluation.get_data_list(data_folder['source_folder'], data_folder['destination_folder'],
                             'default_net', "", "0",
                             distance_measure=em_config['distance_measure'],
                             use_weights=em_config['use_weights'], do_split=em_config['do_split'],
                             do_merge=em_config['do_merge'], update=update_list)

evaluation.create_data_pickles(dataset, update=update_pickles, cnn_input_size=500,
                               target_size=800 if (args.ecd or args.hlw) else None)

if update_cnn:
    evaluation.run_cnn(dataset, mean_file=image_mean, model_def=model_def, model_weights=model_weights, gpu=GPU_ID)

if update_em:
    evaluation.run_em(dataset)


start = 25 if (args.yud or args.ecd) else 0
end = 10000
maxbest = 20

err_cutoff = 0.25

minVPdist = np.pi/10 #np.pi*0.1

graph_color = 'g'

dataset_name = data_folder["name"]

print "dataset name: ", dataset['name']

if dataset_name == "york":
    cameraParams = io.loadmat(config.yud_camera_param_path)

    f = cameraParams['focal'][0,0]
    ps = cameraParams['pixelSize'][0,0]
    pp = cameraParams['pp'][0,:]

    K = np.matrix([[f/ps, 0, 13], [0, f/ps, -11], [0,0,1]])
    S = np.matrix([[2.0/640, 0, 0], [0, 2.0/640, 0], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

metadata = []
if dataset_name == "horizon":
    import csv
    with open(config.hlw_metadata_path, 'rb') as csvfile:
        metadata_file = csv.reader(csvfile)
        for row in metadata_file:
            row[0] = row[0].split('/')[-1]
            row[0] = row[0].split('.')[0]
            metadata.append(row)

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

indices = range(len(dataset['image_files']))

start_time = time.time()

for idx in indices:


    image_file = dataset['image_files'][idx]
    data_file = dataset['pickle_files'][idx]

    # image_file = image_file.replace("scaled", "vanilla")
    # if dataset_name == "eurasian":
    #     image_file = image_file.replace("png", "jpg")

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

    print "data file: ", data_file
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

    if not (em_result is None):

        ( hP1, hP2, zVP, hVP1, hVP2, best_combo ) = ch.calculate_horizon_and_ortho_vp(em_result, maxbest=maxbest, minVPdist=minVPdist)

        vps = em_result['vp']
        counts = em_result['counts']
        vp_assoc = em_result['vp_assoc']
        angles = prob.calc_angles(vps.shape[0], vps)
        ls = lines_dict['line_segments']
        ll = lines_dict['lines']

        num_best = np.minimum(maxbest, vps.shape[0])

        horizon_line = np.cross(hP1, hP2)

        if not (trueHorizon is None):
            thP1 = np.cross(trueHorizon, np.array([1, 0, 1]))
            thP2 = np.cross(trueHorizon, np.array([-1, 0, 1]))
            thP1 /= thP1[2]
            thP2 /= thP2[2]

            max_error = np.maximum(np.abs(hP1[1]-thP1[1]), np.abs(hP2[1]-thP2[1]))/2 * scale*1.0/imageHeight

            print "max_error: ", max_error

            errors.append(max_error)

        langles = np.zeros(ll.shape[0])
        lcosphi = np.zeros(ll.shape[0])
        llen = np.zeros(ll.shape[0])

    else:
        print "no EM results!"

end_time = time.time()

print "time elapsed: ", end_time-start_time

error_arr = np.array(errors)
error_arr_idx = np.argsort(error_arr)
error_arr = np.sort(error_arr)

num_values = len(errors)

plot_points = np.zeros((num_values,2))

midfraction = 1.
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

auc = sklearn.metrics.auc(plot_points[plot_points[:,0]<=err_cutoff,0], plot_points[plot_points[:,0]<=err_cutoff,1])
print "auc: ", auc / err_cutoff

plt.figure()
ax = plt.subplot()
ax.plot(plot_points[:,0], plot_points[:,1], '-', lw=2, c=graph_color)
ax.set_xlabel('horizon error', fontsize=18)
ax.set_ylabel('fraction of images', fontsize=18)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
ax.axis([0,err_cutoff,0,1])
plt.show()
