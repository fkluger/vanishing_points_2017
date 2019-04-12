import csv
import pykitti
import numpy as np

WIDTH = 1250
HEIGHT = 380


class KittiRawDataset:

    def __init__(self, csv_file, root_dir):

        self.sequences = []

        self.num_images = 0
        with open(csv_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                date = row[0]
                drive = row[1]
                total_length = int(row[2])
                start_frame = int(row[3])
                self.num_images += total_length

                self.sequences.append((date, drive, (0, total_length), start_frame))

                print(row)

        self.root_dir = root_dir

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        date = self.sequences[idx][0]
        drive = self.sequences[idx][1]
        frames = self.sequences[idx][2]

        dataset = pykitti.raw(self.root_dir, date, drive, frames=range(frames[0], frames[1]))

        R_cam_imu = np.matrix(dataset.calib.T_cam2_imu[0:3,0:3])
        K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])

        G = np.matrix([[0.], [0.], [1.]])

        images = np.zeros((len(dataset), 3, HEIGHT, WIDTH)).astype(np.float32)
        offsets = np.zeros((len(dataset), 1)).astype(np.float32)
        angles = np.zeros((len(dataset), 1)).astype(np.float32)

        for idx, image in enumerate(iter(dataset.rgb)):

            image_width = WIDTH

            pad_w = WIDTH-image[0].width
            pad_h = HEIGHT-image[0].height

            pad_w1 = int(pad_w/2)
            pad_w2 = pad_w - pad_w1
            pad_h1 = int(pad_h/2)
            pad_h2 = pad_h - pad_h1

            padded_image = np.pad(np.array(image[0]), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0,0)), 'edge')

            R_imu = np.matrix(dataset.oxts[idx].T_w_imu[0:3,0:3])
            G_imu = R_imu.T * G
            G_cam = R_cam_imu * G_imu

            h = np.array(K.I.T*G_cam).squeeze()

            padded_image = np.transpose(padded_image, [2, 0, 1]).astype(np.float32) / 255.

            hp1 = np.cross(h, np.array([1, 0, 0]))
            # hp1 = np.array([0, h[2], -h[1]])
            hp2 = np.cross(h, np.array([1, 0, -image_width]))

            hp1 /= hp1[2]
            hp2 /= hp2[2]

            mh = (0.5*(hp1[1]+hp2[1])+pad_h1) / HEIGHT - 0.5
            offset = mh

            angle = np.arctan2(h[0], h[1])
            if angle > np.pi/2:
                angle -= np.pi
            elif angle < -np.pi/2:
                angle += np.pi

            images[idx,:,:,:] = padded_image
            offsets[idx] = offset
            angles[idx] = angle

        sample = {'images': images, 'offsets': offsets, 'angles': angles}

        return sample

