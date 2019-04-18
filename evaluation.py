import numpy as np
import sphere_mapping as sm
import os
import cPickle as pickle
import glob
import vp_localisation as vp
import lsdpython.lsd as lsd
import scipy.ndimage as ndimage
from skimage import color


def get_sphere_image(lines, size=250, alpha=0.1, f=1.0):
    sphere_image = sm.sphere_line_plot(lines, size, alpha=alpha, f=f, alternative=False)
    return sphere_image


def init_caffe(model_def, model_weights, gpu_id=0):
    import caffe
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    return net


def read_mean_blob(mean_file):
    import caffe
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file, 'rb').read()
    blob.ParseFromString(data)
    mean_arr = np.array(caffe.io.blobproto_to_array(blob))
    return mean_arr


def caffe_forward(net, image, mean_arr):
    net.blobs['data'].data[...] = image[np.newaxis,:,:] - mean_arr[0, :, :, :]
    output = net.forward()
    prediction = np.squeeze(output['sigout'])
    return prediction


def save_cnn_result(net, mean_arr, datum, file_for_basename):
    image = get_sphere_image(datum['lines'])
    prediction = caffe_forward(net, image, mean_arr)

    datum['prediction'] = prediction

    basename = os.path.splitext(file_for_basename)[0]

    pkl_file = "%s.cnn_result.pkl" % basename

    with open(pkl_file, 'wb') as pickle_file:
        pickle.dump(datum, pickle_file, -1)


def get_data_list(source_folder, destination_folder, name, cnn_model_root, cnn_model_iterations, dataset_name=None,
                  distance_measure="angle", use_weights=True, do_split=True, do_merge=True, update=False):

    print "Fetching file list for ", name

    pkl_filename = "%s/%s_%s_%sweights_%ssplit_%smerge.pkl" % (destination_folder, name, distance_measure,
                                                               "" if use_weights else "no", "" if do_split else "no",
                                                               "" if do_merge else "no")

    print "pkl_filename: ", pkl_filename

    fullname = "%s_%s_%sweights_%ssplit_%smerge" % (name, distance_measure, "" if use_weights else "no",
                                                    "" if do_split else "no", "" if do_merge else "no")

    pkl_exists = os.path.isfile(pkl_filename)

    if (not pkl_exists) or update:

        dataset = {'source_folder': source_folder, 'destination_folder': destination_folder + '/' + fullname,
                   'cnn_root':cnn_model_root, 'cnn_iterations':cnn_model_iterations, 'use_weights':use_weights,
                   'distance_measure':distance_measure, 'do_split':do_split, 'do_merge':do_merge }

        if not os.path.exists(dataset['destination_folder']):
            os.makedirs(dataset['destination_folder'])

        print "dataset: ", fullname

        if dataset_name == 'york':
            image_files = glob.glob("%s/P*/P*.jpg" % source_folder)
        elif dataset_name == 'eurasian':
            image_files = glob.glob("%s/*.jpg" % source_folder)
        elif dataset_name == 'hlw':
            image_list_file = "%s/split/test.txt" % source_folder
            image_files = []
            with open(image_list_file) as fp:
                line = fp.readline()
                while line:
                    image_files += ["%s/images/%s" % (source_folder, line.strip())]
                    line = fp.readline()
        else:
            image_files = glob.glob("%s/*.jpg" % source_folder)
            image_files += glob.glob("%s/*.png" % source_folder)
            image_files += glob.glob("%s/*.pgm" % source_folder)

        image_files.sort()

        dataset['image_files'] = image_files

        dataset['pickle_files'] = []
        for image_file in image_files:
            basename = os.path.splitext(os.path.basename(image_file))[0]
            data_file = "%s/%s.data.pkl" % (dataset["destination_folder"], basename)
            dataset['pickle_files'].append(data_file)

        dataset['name'] = fullname

        with open(pkl_filename, 'wb') as fp:
            pickle.dump(dataset, fp)

    else:
        with open (pkl_filename, 'rb') as fp:
            dataset = pickle.load(fp)

    return dataset


def create_data_pickles(dataset, update=False, cnn_input_size=250, target_size=None):

    image_files = dataset["image_files"]
    pickle_files = dataset["pickle_files"]

    for idx in range(len(image_files)):

        image_file = image_files[idx]
        data_file = pickle_files[idx]

        print "processing image: ", image_file

        basename = os.path.basename(image_file)
        tmp_file = os.path.join("/tmp/", basename + ".png")

        pkl_exists = os.path.isfile(data_file)

        if (not pkl_exists) or update:

            if target_size is not None:

                resize_command = "convert %s -resize %dx%d %s" % (image_file, target_size, target_size, tmp_file)
                os.system(resize_command)

                imageRGB = ndimage.imread(tmp_file)

            else:
                imageRGB = ndimage.imread(image_file)

            image = color.rgb2gray(imageRGB)

            datum = {"dataset": dataset["name"], "image_file": image_file, "image_shape":image.shape, "image":imageRGB}

            lsd_result = detect_lsd_lines(image)

            line_segments = lsd_result['segments']

            lines = np.zeros((line_segments.shape[0], 3))
            linelen = np.zeros((line_segments.shape[0]))

            for li in range(line_segments.shape[0]):
                ls = line_segments[li,:]
                p1 = np.array([ls[0], ls[1], 1])
                p2 = np.array([ls[2], ls[3], 1])
                linelen[li] = np.linalg.norm(p1-p2, ord=2)

                line = np.cross(p1,p2)
                lines[li,:] = line.copy()

            datum['line_segments'] = line_segments
            datum['lines'] = lines

            if not(datum['line_segments'] is None):

                sphere_image = get_sphere_image(datum['lines'], size=cnn_input_size, alpha=0.1)

                pkl_data = {'lines':datum, 'sphere_image':sphere_image}

            else:
                print "SKIPPING: incomplete data %s" % data_file
                pkl_data = {'lines':datum, 'sphere_image':None}

            with open(data_file, 'wb') as pickle_file:
                pickle.dump(pkl_data, pickle_file, -1)

    return


def create_data_dict_single(image_rgb, cnn_input_size=250):

    image = color.rgb2gray(image_rgb)

    datum = {"image_shape": image.shape, "image": image_rgb}

    lsd_result = detect_lsd_lines(image)

    line_segments = lsd_result['segments']

    lines = np.zeros((line_segments.shape[0], 3))
    linelen = np.zeros((line_segments.shape[0]))

    for li in range(line_segments.shape[0]):
        ls = line_segments[li,:]
        p1 = np.array([ls[0], ls[1], 1])
        p2 = np.array([ls[2], ls[3], 1])
        linelen[li] = np.linalg.norm(p1-p2, ord=2)

        line = np.cross(p1,p2)
        lines[li,:] = line.copy()

    datum['line_segments'] = line_segments
    datum['lines'] = lines

    if not(datum['line_segments'] is None):

        sphere_image = get_sphere_image(datum['lines'], size=cnn_input_size, alpha=0.1)

        pkl_data = {'lines':datum, 'sphere_image':sphere_image}

    else:
        print "SKIPPING: incomplete data"
        pkl_data = {'lines': datum, 'sphere_image':None}

    return pkl_data


def detect_lsd_lines(image):

    image = image.astype('float64')
    if np.max(image) <= 1:
        image *= 255

    width = image.shape[1]
    height = image.shape[0]
    scale_w = np.maximum(width, height)
    scale_h = scale_w

    lsd_lines = lsd.detect_line_segments(image)

    lsd_lines[:,0] -= width/2.0
    lsd_lines[:,1] -= height/2.0
    lsd_lines[:,2] -= width/2.0
    lsd_lines[:,3] -= height/2.0
    lsd_lines[:,0] /= (scale_w/2.0)
    lsd_lines[:,1] /= (scale_h/2.0)
    lsd_lines[:,2] /= (scale_w/2.0)
    lsd_lines[:,3] /= (scale_h/2.0)
    lsd_lines[:,1] *= -1
    lsd_lines[:,3] *= -1

    return {'segments': lsd_lines[:,0:4], 'nfa': lsd_lines[:,6]}


def run_cnn(dataset, model_def, model_weights, mean_file, gpu=0):

    import time

    start = time.time()

    mean_arr = read_mean_blob(mean_file)

    net = init_caffe(model_def, model_weights, gpu)

    end = time.time()

    print "CNN init time: ", end-start

    # base_files = dataset['image_files']
    base_files = dataset['pickle_files']

    for base_file in base_files:
        print "CNN - file: ", base_file

        if isinstance(base_file, tuple):
            data_file = base_file[0]
        else:
            data_file = base_file

        with open (data_file, 'rb') as fp:
            datum = pickle.load(fp)

        if not (datum['sphere_image'] is None):
            prediction = caffe_forward(net, datum['sphere_image'], mean_arr)

            datum['cnn_prediction'] = prediction
        else:
            datum['cnn_prediction'] = None

        with open(data_file, 'wb') as pickle_file:
            pickle.dump(datum, pickle_file, -1)

    print "finished ", dataset['destination_folder']


def run_em(dataset, start=None, end=None):

    base_files = dataset['pickle_files']

    distance_measure = dataset['distance_measure']
    use_weights = dataset['use_weights']
    do_split = dataset['do_split']
    do_merge = dataset['do_merge']

    if not (start is None or end is None):
        if end > len(base_files):
            end = len(base_files)
        base_files = base_files[start:end]

    for idx, base_file in enumerate(base_files):

        print idx+1, " / ", len(base_files)
        print "EM - processing ", base_file

        if isinstance(base_file, tuple):
            data_file = base_file[0]
        else:
            data_file = base_file

        with open (data_file, 'rb') as fp:
            datum = pickle.load(fp)

        datum = run_em_single(datum, distance_measure=distance_measure, use_weights=use_weights,
                              do_split=do_split, do_merge=do_merge)

        if datum['EM_result'] is None:
            print "SKIPPING: file %s is incomplete" % data_file

        with open(data_file, 'wb') as pickle_file:
            pickle.dump(datum, pickle_file, -1)


def run_em_single(datum, distance_measure="angle", use_weights=True, do_split=True, do_merge=True):
    lines = datum['lines']

    if not (datum['cnn_prediction'] is None):

        sphere_image = datum['sphere_image']

        l = lines['lines']
        lp = lines['line_segments']

        cnn_prediction = datum['cnn_prediction'][:, :]

        em_result = vp.expectation_maximisation(l, lp,
                                                cnn_prediction, sphere_image=sphere_image,
                                                distance_measure=distance_measure, use_weights=use_weights,
                                                do_split=do_split, do_merge=do_merge)

        datum['EM_result'] = em_result
        datum['lines'] = lines
    else:
        datum['EM_result'] = None

    return datum


def renew_cnn_result(net, mean_arr, lines, image_size):
    image = get_sphere_image(lines, size=image_size)
    prediction = caffe_forward(net, image, mean_arr)

    return (image, prediction)
