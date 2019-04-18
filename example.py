import config
import sys
sys.path.insert(0,"/home/kluger/tmp/tools/caffe-rc5/python")
import os
import argparse
import pickle
import scipy.ndimage as ndimage
import numpy as np
import calc_horizon

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
parser.add_argument('--show', dest='show', action='store_true', help='Show results')
args = parser.parse_args()

if args.show:
    import result_plotting
import evaluation

GPU_ID = args.gpu

image_mean = config.cnn_mean_path
model_def = config.cnn_config_path
model_weights = config.cnn_weights_path

data_folder = {"name": "examples", "source_folder": "assets/examples", "destination_folder": "assets/results"}

em_config = {'distance_measure': 'angle', 'use_weights': True, 'do_split': True, 'do_merge': True}

dataset = evaluation.get_data_list(data_folder['source_folder'], data_folder['destination_folder'],
                                   'default_net', "", "0",
                                   distance_measure=em_config['distance_measure'],
                                   use_weights=em_config['use_weights'], do_split=em_config['do_split'],
                                   do_merge=em_config['do_merge'], update=True)

if not args.show:
    evaluation.create_data_pickles(dataset, update=True, cnn_input_size=500, target_size=640)
    evaluation.run_cnn(dataset, mean_file=image_mean, model_def=model_def, model_weights=model_weights, gpu=GPU_ID)
    evaluation.run_em(dataset)

else:
    for idx in range(len(dataset['image_files'])):

        image_file = dataset['image_files'][idx]
        pickle_file = dataset['pickle_files'][idx]

        print "image file: ", image_file
        if not os.path.isfile(image_file):
            print "file not found"
            continue

        image = ndimage.imread(image_file)

        basename = os.path.splitext(image_file)[0]

        data_file = pickle_file
        print "data file: ", data_file
        if not os.path.isfile(data_file):
            print "file not found"
            continue

        with open(data_file, 'rb') as fp:
            datum = pickle.load(fp)

        (hP1, hP2, _, _, _, _) = calc_horizon.calculate_horizon_and_ortho_vp(datum['EM_result'], maxbest=20,
                                                                             theta_vmin=np.pi / 10.)
        width = image.shape[1]
        height = image.shape[0]
        scale = 640. / np.maximum(width, height)
        width *= scale
        height *= scale
        hP1[0] = hP1[0] * 640 / 2.0 + width / 2.0
        hP2[0] = hP2[0] * 640 / 2.0 + width / 2.0
        hP1[1] = -hP1[1] * 640 / 2.0 + height / 2.0
        hP2[1] = -hP2[1] * 640 / 2.0 + height / 2.0

        print(hP1)
        print(hP2)

        num_vps = 10 if 'nord_lb' in data_file else (5 if 'ihme' in data_file else 3)

        result_plotting.show_em_result(datum, image_file, maxbest=num_vps, target_size=640, horizon=(hP1, hP2))
