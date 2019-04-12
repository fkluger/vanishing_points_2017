import numpy as np
import sphere_mapping as sm
import os
import cPickle as pickle
import glob
import vp_localisation as vp
import lsdpython.lsd as lsd
import scipy.ndimage as ndimage
import scipy.misc
from skimage import color
import csv
import pykitti

from joblib import Parallel, delayed
import multiprocessing

def get_sphere_image(lines, size=250, alpha=0.1, f=1.0):
    sphere_image = sm.sphereLinePlot(lines, size, alpha=alpha, f=f, alternative=False)
    return sphere_image


def init_caffe(model_def, model_weights, GPU_ID=0):
    import caffe
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
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


def get_data_list(source_folder, destination_folder, name, cnn_model_root, cnn_model_iterations,
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


def get_data_list_kitti(source_folder, destination_folder, name, cnn_model_root, cnn_model_iterations, csv_file,
                  distance_measure="angle", use_weights=True, do_split=True, do_merge=True, update=False):
    sequences = []

    num_images = 0
    with open(csv_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            date = row[0]
            drive = row[1]
            total_length = int(row[2])
            start_frame = int(row[3])
            num_images += total_length

            sequences.append((date, drive, (0, total_length), start_frame))

            print(row)

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

        dataset['image_files'] = []
        dataset['pickle_files'] = []

        for idx in range(len(sequences)):
            date = sequences[idx][0]
            drive = sequences[idx][1]
            frames = sequences[idx][2]
            start_frame = sequences[idx][3]

            frame_list = list(range(frames[0], frames[1]))

            dataset['image_files'] += [(date, drive, idx + start_frame, idx) for idx in
                       frame_list]
            dataset['pickle_files'] += [((destination_folder + "/" + date + "/" + drive + "/%06d.pkl" % (idx + start_frame)), idx) for idx in
                       frame_list]

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

        # basename = os.path.splitext(os.path.basename(image_file))[0]
        # data_file = "%s/%s.data.pkl" % (dataset["destination_folder"], basename)

        pkl_exists = os.path.isfile(data_file)

        if (not pkl_exists) or update:

            # imageRGB = ndimage.imread(image_file)
            # imageRGB = Image.open(image_file)

            if target_size is not None:

                resize_command = "convert %s -resize %dx%d %s" % (image_file, target_size, target_size, tmp_file)
                os.system(resize_command)

                imageRGB = ndimage.imread(tmp_file)

                # imshape = imageRGB.shape[0:2]
                # max_dim = np.argmax(imshape).squeeze()
                # min_dim = np.argmin(imshape).squeeze()
                # resize_factor = target_size * 1. / np.max(imshape)
                # newshape = list(imshape)
                # newshape[max_dim] = target_size
                # newshape[min_dim] = int(newshape[min_dim] * resize_factor)
                # # imageRGB = scipy.misc.imresize(imageRGB, resize_factor, interp='bilinear', mode=None)
                # imageRGB = transform.resize(imageRGB, newshape, order=3, mode='reflect', cval=0, clip=True,
                #                          preserve_range=False)

                # imageRGB = imageRGB.resize(newshape, Image.ANTIALIAS)
                # imageRGB =

                # imageRGB = scipy.misc.imresize(imageRGB, resize_factor, interp='bicubic', mode=None)
                # imageRGB = scipy.misc.imresize(imageRGB, resize_factor, interp='cubic', mode=None)
                # imageRGB = scipy.misc.imresize(imageRGB, resize_factor, interp='lanczos', mode=None)
                # imageRGB = scipy.misc.imresize(imageRGB, resize_factor, interp='nearest', mode=None)

            else:
                imageRGB = ndimage.imread(image_file)

            print(imageRGB.shape)

            image = color.rgb2gray(imageRGB)

            datum = {"dataset": dataset["name"], "image_file": image_file, "image_shape":image.shape, "image":imageRGB}

            lsd_result = detect_lsd_lines(image)

            line_segments = lsd_result['segments']
            NFAs = lsd_result['nfa']

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
            datum['nfa'] = NFAs

            if not(datum['line_segments'] is None):

                sphere_image = get_sphere_image(datum['lines'], size=cnn_input_size, alpha=0.1)

                pkl_data = {'lines':datum, 'sphere_image':sphere_image}

            else:
                print "SKIPPING: incomplete data %s" % data_file
                pkl_data = {'lines':datum, 'sphere_image':None}

            with open(data_file, 'wb') as pickle_file:
                pickle.dump(pkl_data, pickle_file, -1)

    return


def create_data_pickles_kitti(dataset, update=False, cnn_input_size=250, target_size=None):

    image_files = dataset["image_files"]
    pickle_files = dataset["pickle_files"]

    num_cores = int(multiprocessing.cpu_count())
    Parallel(n_jobs=num_cores)(delayed(create_pickle_single_kitti)(i, image_files, pickle_files, dataset, update, cnn_input_size, target_size) for i in range(len(image_files)))

    # for idx in range(len(image_files)):
    #
    #     image_file = image_files[idx]
    #     # image_index = image_files[idx][1]
    #     data_file = pickle_files[idx][0]
    #
    #     print "processing image: ", image_file
    #
    #     pkl_exists = os.path.isfile(data_file)
    #
    #     if (not pkl_exists) or update:
    #
    #         date = image_file[0]
    #         drive = image_file[1]
    #         frame = image_file[2]
    #         image_index = image_file[3]
    #
    #         kittidata = pykitti.raw(dataset['source_folder'], date, drive, frames=range(frame, frame+1))
    #
    #         R_cam_imu = np.matrix(kittidata.calib.T_cam2_imu[0:3, 0:3])
    #         K = np.matrix(kittidata.calib.P_rect_20[0:3, 0:3])
    #
    #         G = np.matrix([[0.], [0.], [1.]])
    #
    #         R_imu = np.matrix(kittidata.oxts[0].T_w_imu[0:3, 0:3])
    #         G_imu = R_imu.T * G
    #         G_cam = R_cam_imu * G_imu
    #
    #         h = np.array(K.I.T * G_cam).squeeze()
    #
    #         imageRGB = np.array(kittidata.rgb[0][0])
    #
    #         data_folder = os.path.dirname(data_file)
    #         if not os.path.exists(data_folder):
    #             os.makedirs(data_folder)
    #
    #         if target_size is not None:
    #             imshape = imageRGB.shape[0:2]
    #             resize_factor = target_size * 1. / np.max(imshape)
    #             imageRGB = scipy.misc.imresize(imageRGB, resize_factor, interp='bicubic', mode=None)
    #
    #         image = color.rgb2gray(imageRGB)
    #
    #         datum = {"dataset": dataset["name"], "image_file": image_file, "image_shape":image.shape, "image":imageRGB,
    #                  "horizon": h, "index": image_index, "K": K, "G": G_cam}
    #
    #         lsd_result = detect_lsd_lines(image)
    #
    #         line_segments = lsd_result['segments']
    #         NFAs = lsd_result['nfa']
    #
    #         lines = np.zeros((line_segments.shape[0], 3))
    #         linelen = np.zeros((line_segments.shape[0]))
    #
    #         for li in range(line_segments.shape[0]):
    #             ls = line_segments[li,:]
    #             p1 = np.array([ls[0], ls[1], 1])
    #             p2 = np.array([ls[2], ls[3], 1])
    #             linelen[li] = np.linalg.norm(p1-p2, ord=2)
    #
    #             line = np.cross(p1,p2)
    #             lines[li,:] = line.copy()
    #
    #         datum['line_segments'] = line_segments
    #         datum['lines'] = lines
    #         datum['nfa'] = NFAs
    #
    #         if not(datum['line_segments'] is None):
    #
    #             sphere_image = get_sphere_image(datum['lines'], size=cnn_input_size, alpha=0.1)
    #
    #             pkl_data = {'lines':datum, 'sphere_image':sphere_image}
    #
    #         else:
    #             print "SKIPPING: incomplete data %s" % data_file
    #             pkl_data = {'lines':datum, 'sphere_image':None}
    #
    #         with open(data_file, 'wb') as pickle_file:
    #             pickle.dump(pkl_data, pickle_file, -1)

    return


def create_pickle_single_kitti(idx, image_files, pickle_files, dataset, update, cnn_input_size, target_size):

    image_file = image_files[idx]
    # image_index = image_files[idx][1]
    data_file = pickle_files[idx][0]

    print "processing image: ", image_file

    pkl_exists = os.path.isfile(data_file)

    if (not pkl_exists) or update:

        date = image_file[0]
        drive = image_file[1]
        frame = image_file[2]
        image_index = image_file[3]

        kittidata = pykitti.raw(dataset['source_folder'], date, drive, frames=range(frame, frame+1))

        R_cam_imu = np.matrix(kittidata.calib.T_cam2_imu[0:3, 0:3])
        K = np.matrix(kittidata.calib.P_rect_20[0:3, 0:3])

        G = np.matrix([[0.], [0.], [1.]])

        R_imu = np.matrix(kittidata.oxts[0].T_w_imu[0:3, 0:3])
        G_imu = R_imu.T * G
        G_cam = R_cam_imu * G_imu

        h = np.array(K.I.T * G_cam).squeeze()

        imageRGB = np.array(kittidata.rgb[0][0])

        data_folder = os.path.dirname(data_file)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        if target_size is not None:
            imshape = imageRGB.shape[0:2]
            resize_factor = target_size * 1. / np.max(imshape)
            imageRGB = scipy.misc.imresize(imageRGB, resize_factor, interp='bicubic', mode=None)

        image = color.rgb2gray(imageRGB)

        datum = {"dataset": dataset["name"], "image_file": image_file, "image_shape":image.shape, "image":imageRGB,
                 "horizon": h, "index": image_index, "K": K, "G": G_cam}

        lsd_result = detect_lsd_lines(image)

        line_segments = lsd_result['segments']
        NFAs = lsd_result['nfa']

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
        datum['nfa'] = NFAs

        if not(datum['line_segments'] is None):

            sphere_image = get_sphere_image(datum['lines'], size=cnn_input_size, alpha=0.1)

            pkl_data = {'lines':datum, 'sphere_image':sphere_image}

        else:
            print "SKIPPING: incomplete data %s" % data_file
            pkl_data = {'lines':datum, 'sphere_image':None}

        with open(data_file, 'wb') as pickle_file:
            pickle.dump(pkl_data, pickle_file, -1)


def create_data_dict_single(imageRGB, cnn_input_size=250):

    image = color.rgb2gray(imageRGB)

    datum = {"image_shape":image.shape, "image":imageRGB}

    lsd_result = detect_lsd_lines(image, True)

    line_segments = lsd_result['segments']
    NFAs = lsd_result['nfa']

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
    datum['nfa'] = NFAs

    if not(datum['line_segments'] is None):

        sphere_image = get_sphere_image(datum['lines'], size=cnn_input_size, alpha=0.1)

        pkl_data = {'lines':datum, 'sphere_image':sphere_image}

    else:
        print "SKIPPING: incomplete data"
        pkl_data = {'lines':datum, 'sphere_image':None}

    return pkl_data


def detect_lsd_lines(image):

    image = image.astype('float64')
    if np.max(image) <= 1:
        image *= 255

    width = image.shape[1]
    height = image.shape[0]
    scaleW = np.maximum(width, height)
    scaleH = scaleW

    lsd_lines = lsd.detect_line_segments(image)

    lsd_lines[:,0] -= width/2.0
    lsd_lines[:,1] -= height/2.0
    lsd_lines[:,2] -= width/2.0
    lsd_lines[:,3] -= height/2.0
    lsd_lines[:,0] /= (scaleW/2.0)
    lsd_lines[:,1] /= (scaleH/2.0)
    lsd_lines[:,2] /= (scaleW/2.0)
    lsd_lines[:,3] /= (scaleH/2.0)
    lsd_lines[:,1] *= -1
    lsd_lines[:,3] *= -1

    print("lines: ", lsd_lines.shape)

    return {'segments':lsd_lines[:,0:4], 'nfa':lsd_lines[:,6]}


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

        # basename = os.path.splitext(base_file)[0]
        # data_file = "%s.data.pkl" % basename
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


def run_cnn_kitti(dataset, model_def, model_weights, mean_file, gpu=0):

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

        # basename = os.path.splitext(base_file)[0]
        # data_file = "%s.data.pkl" % basename
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

def run_cnn_single_frame(dataset, model_def, model_weights, mean_file, gpu=0):

    import time

    start = time.time()

    mean_arr = read_mean_blob(mean_file)

    net = init_caffe(model_def, model_weights, gpu)

    end = time.time()

    for base_file in base_files:
        print "CNN - file: ", base_file

        # basename = os.path.splitext(base_file)[0]
        # data_file = "%s.data.pkl" % basename
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

    # base_files = dataset['image_files']
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

        print idx, " / ", len(base_files)
        print "EM - processing ", base_file

        # basename = os.path.splitext(base_file)[0]
        # data_file = "%s.data.pkl" % basename
        # data_file = base_file
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
