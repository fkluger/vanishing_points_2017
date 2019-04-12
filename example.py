import sys
sys.path.insert(0,"/home/kluger/tmp/tools/caffe-rc5/python")
import os
import evaluation as eval
import pickle
import scipy.ndimage as ndimage
import result_plotting
import calc_horizon
import numpy as np

update_list = False
update_pickles = False
update_cnn = False
update_em = False

GPU_ID = 3

model_root = "/data/kluger/ma/caffe/vp_sphere_classification/models/alexnet/newdata_500px_20x20_v5"
image_mean = model_root + "/mean.binaryproto"
model_def = model_root + "/deploy.prototxt"
model_weights = model_root + '/tmp/_iter_300000.caffemodel'

data_folder = {"name": "examples", "source_folder": "/home/kluger/tmp/gcpr_examples", "destination_folder": "/home/kluger/tmp/gcpr_examples" }

em_config = {'distance_measure': 'angle', 'use_weights': True, 'do_split': True, 'do_merge': True}


keepAR = True

dataset = eval.get_data_list(data_folder['source_folder'], data_folder['destination_folder'],
                             'default_net', model_root, "0",
                             distance_measure=em_config['distance_measure'],
                             use_weights=em_config['use_weights'], do_split=em_config['do_split'],
                             do_merge=em_config['do_merge'], update=update_list)

eval.create_data_pickles(dataset, update=update_pickles, keepAR=keepAR, cnn_input_size=500, target_size=640)

if update_cnn:
    eval.run_cnn(dataset, mean_file=image_mean, model_def=model_def, model_weights=model_weights, gpu=GPU_ID)

if update_em:
    eval.run_em(dataset)

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
                                                                                    minVPdist=np.pi/10.)  # np.pi/5.0
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

    num_vps = 10 if 'nordlb' in data_file else 3

    result_plotting.show_em_result(datum, image_file, maxbest=num_vps, target_size=640, horizon=(hP1, hP2))

