import sys
sys.path.insert(0,"/home/kluger/tmp/tools/caffe-rc5/python")
import evaluation
import numpy as np
import glob
import scipy.io as io
import os
import vp_localisation as vp
import pickle
import scipy.ndimage as ndimage
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import probability_functions as prob
import sklearn.metrics
import calc_horizon as ch
import time
import kitti
from auc import *
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument('--start', default=None, type=int, help='')
parser.add_argument('--end', default=None, type=int, help='')
args = parser.parse_args()

set_split = 'test'

dataset_path = '/data/scene_understanding/KITTI/rawdata/'
pickle_path = '/data/kluger/tmp/kitti_%s/' % set_split
split_file = '/home/kluger/tmp/kitti_split_5/%s.csv' % set_split

update_list = False
update_pickles = False
update_cnn = False
update_em = False

GPU_ID = 3

model_root = "/data/kluger/ma/caffe/vp_sphere_classification/models/alexnet/newdata_500px_20x20_v5"
image_mean = model_root + "/mean.binaryproto"
model_def = model_root + "/deploy.prototxt"
model_weights = model_root + '/tmp/_iter_300000.caffemodel'

data_folder = {"name": "kitti_val", "source_folder": dataset_path, "destination_folder": pickle_path}

em_config = {'distance_measure': 'angle', 'use_weights': True, 'do_split': True, 'do_merge': True}

keepAR = True

dataset = evaluation.get_data_list_kitti(data_folder['source_folder'], data_folder['destination_folder'],
                             'default_net', model_root, "0", csv_file=split_file,
                             distance_measure=em_config['distance_measure'],
                             use_weights=em_config['use_weights'], do_split=em_config['do_split'],
                             do_merge=em_config['do_merge'], update=update_list)

if update_pickles:
    evaluation.create_data_pickles_kitti(dataset, update=update_pickles, cnn_input_size=500, target_size=None)

if update_cnn:
    evaluation.run_cnn(dataset, mean_file=image_mean, model_def=model_def, model_weights=model_weights, gpu=GPU_ID)

if update_em:
    evaluation.run_em(dataset, start=args.start, end=args.end)

# exit(0)

start = 0
end = 10000
indices = None
maxbest = 20
show_histograms = False
show_plots = False
use_old_idx = False
second_em = False
both_em = False

err_cutoff = 0.25

minVPdist = np.pi/10 #np.pi*0.1

legend_title = ""
graph_color = 'g'

dataset_name = data_folder["name"]

dist_measure = "angle"
use_weights = "weights"
splitmerge = ""
do_split = "%ssplit" % splitmerge
do_merge = "%smerge" % splitmerge

print "dataset name: ", dataset['name']

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

all_angular_errors = []
all_angular_errors_per_sequence = []
all_errors_per_sequence = []
error_grads = []

last_date = None
last_drive = None

for idx in indices:

    image_file = dataset['image_files'][idx]
    data_file = dataset['pickle_files'][idx][0]

    drive = image_file[1]
    date = image_file[0]
    if last_date != date or last_drive != drive:
        print(date, drive)
        if not (last_date is None or last_drive is None):

            error_gradient = np.gradient(all_errors_per_sequence)
            abs_error_grad = np.sum(np.abs(error_gradient)) / len(all_errors_per_sequence)
            print("abs_error_grad: %.9f" % abs_error_grad)
            error_grads += [error_gradient]

        last_date = date
        last_drive = drive
        all_angular_errors_per_sequence = []
        all_errors_per_sequence = []

    # print(image_file, "\n", data_file)

    with open(data_file, 'rb') as fp:
        datum = pickle.load(fp)

    sphere_image = datum['sphere_image'] if 'sphere_image' in datum else None
    prediction = datum['cnn_prediction'][::-1,:] if 'cnn_prediction' in datum else None

    lines_dict = datum['lines'] if 'lines' in datum else None
    em_result = datum['EM_result'] if 'EM_result' in datum else None

    count += 1

    if count <= start: continue
    if count > end: break

    image = datum['lines']['image']
    image_shape = datum['lines']['image_shape']
    imageWidth = image_shape[1]
    imageHeight = image_shape[0]


    trueVPs = None
    trueHorizon = datum['lines']['horizon']
    K = datum['lines']['K']
    G = datum['lines']['G']
    # print(trueHorizon)

    scale = np.maximum(imageWidth, imageHeight)

    if not (em_result is None):

        ( hP1, hP2, zVP, hVP1, hVP2, best_combo ) = ch.calculate_horizon_and_ortho_vp(em_result, maxbest=maxbest, minVPdist=minVPdist)

        hP1[0] = hP1[0] * scale / 2.0 + imageWidth / 2.0
        hP2[0] = hP2[0] * scale / 2.0 + imageWidth / 2.0
        hP1[1] = -hP1[1] * scale / 2.0 + imageHeight / 2.0
        hP2[1] = -hP2[1] * scale / 2.0 + imageHeight / 2.0

        vps = em_result['vp']
        counts = em_result['counts']
        # counts = em_result['counts_weighted']
        vp_assoc = em_result['vp_assoc']
        angles = prob.calc_angles(vps.shape[0], vps)
        ls = lines_dict['line_segments']
        ll = lines_dict['lines']

        num_best = np.minimum(maxbest, vps.shape[0])

        horizon_line = np.cross(hP1, hP2)
        # print(hP1, hP2)

        Ge = K.T * np.matrix(horizon_line).T
        Ge /= np.linalg.norm(Ge)

        G = np.matrix(G)
        G /= np.linalg.norm(G)

        try:
            angular_error = np.abs((np.arccos(np.clip(np.abs(np.dot(Ge.T, G)), 0, 1)) * 180 / np.pi)[0, 0])
        except:
            print(Ge)
            print(G)
            print(np.dot(Ge.T, G))
            exit(0)

        all_angular_errors.append(angular_error)
        all_angular_errors_per_sequence.append(angular_error)

        if not (trueHorizon is None):
            thP1 = np.cross(trueHorizon, np.array([1, 0, 0]))
            thP2 = np.cross(trueHorizon, np.array([1, 0, -imageWidth]))
            thP1 /= thP1[2]
            thP2 /= thP2[2]

            max_error = np.maximum(np.abs(hP1[1]-thP1[1]), np.abs(hP2[1]-thP2[1]))/imageHeight

            print "max_error: ", max_error, angular_error

            errors.append(max_error)
            all_errors_per_sequence.append(max_error)

        langles = np.zeros(ll.shape[0])
        lcosphi = np.zeros(ll.shape[0])
        llen = np.zeros(ll.shape[0])

    else:
        print "no EM results!"
        assert False

error_gradient = np.gradient(all_errors_per_sequence)
abs_error_grad = np.sum(np.abs(error_gradient)) / len(all_errors_per_sequence)
print("abs_error_grad: %.9f" % abs_error_grad)
error_grads += [error_gradient]

end_time = time.time()

print "time elapsed: ", end_time-start_time


error_arr = np.array(errors)
error_arr_idx = np.argsort(error_arr)
error_arr = np.sort(error_arr)

MSE = np.mean(np.square(error_arr))
print("MSE: %.8f" % MSE)

error_grads = np.concatenate(error_grads)
abs_error_grad = np.sum(np.abs(error_grads)) / error_grads.shape[0]
sq_error_grad = np.sum(np.square(np.abs(error_grads))) / error_grads.shape[0]
print("abs_error_grad: %.9f" % abs_error_grad)
print("sq_error_grad: %.9f" % sq_error_grad)

num_values = len(errors)

auc, plot_points = calc_auc(error_arr, cutoff=0.25)
print("auc: ", auc)
print("mean error: ", np.mean(error_arr))

plt.figure()
plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
plt.xlim(0, 0.25)
plt.ylim(0, 1.0)
plt.text(0.175, 0.05, "AUC: %.8f" % auc, fontsize=12)
plt.savefig(os.path.join("/home/kluger/tmp/", "kluger_error_histogram_%s.png" % set_split), dpi=300)
plt.savefig(os.path.join("/home/kluger/tmp/", "kluger_error_histogram_%s.svg" % set_split), dpi=300)



print("angular errors:")
error_arr = np.abs(np.array(all_angular_errors))
auc, plot_points = calc_auc(error_arr, cutoff=5)
print("auc: ", auc)
plt.figure()
plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
plt.xlim(0, 5)
plt.ylim(0, 1.0)
plt.text(0.175, 0.05, "AUC: %.8f" % auc, fontsize=12)
plt.savefig(os.path.join("/home/kluger/tmp/", "kluger_error_histogram_angular_%s.png" % set_split), dpi=300)
plt.savefig(os.path.join("/home/kluger/tmp/", "kluger_error_histogram_angular_%s.svg" % set_split), dpi=300)