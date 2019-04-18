import numpy as np


def index_to_angle(index, shape):

    angle = np.zeros(2)

    a = index[0]
    b = index[1]

    M = shape[0]
    N = shape[1]

    alpha = (a-0.5*M+0.5) * np.pi / M
    beta = (b-0.5*N+0.5) * np.pi / N

    angle[0] = alpha
    angle[1] = beta

    return angle


def angle_to_index(angle, shape):
    alpha = angle[0]
    beta = angle[1]

    M = shape[0]
    N = shape[1]

    a = (alpha / np.pi + 0.5 - 0.5/M) * M
    b = (beta / np.pi + 0.5 - 0.5/N) * N

    index = np.array([a,b])

    return index


def angle_to_point(angle):
    alpha = angle[0]
    beta = angle[1]

    point = np.zeros(3)

    point[1] = np.sin(beta)
    point[0] = np.sin(alpha)*np.cos(beta)
    point[2] = np.cos(alpha)*np.cos(beta)

    point *= np.sign(point[2])

    return point


def point_to_angle(point):
    angle = np.zeros(2)
    angle[1] = np.arcsin(point[1])
    inner = point[0] / np.cos(angle[1])
    inner = np.minimum(inner, 1)
    inner = np.maximum(inner, -1)
    angle[0] = np.arcsin(inner)

    return angle
