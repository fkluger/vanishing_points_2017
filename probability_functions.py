import numpy as np
from collections import namedtuple
from joblib import Parallel, delayed
import multiprocessing

pi = np.pi

PDFParams = namedtuple('PDFParams', 'means weights sigma')
PDF = namedtuple('PDF', 'v lv vl l lvsq angles')

def calc_pdf(pdfpar, x, y):

    # means = pdfpar['means']
    # weights = pdfpar['weights']
    # sigma = pdfpar['sigma']
    means = pdfpar.means
    weights = pdfpar.weights
    sigma = pdfpar.sigma

    N = means.shape[0]

    d = np.zeros((5))
    # d = np.zeros((2))

    # test: use uniform distribution
    # return np.ones((x.shape))

    response = np.zeros((x.shape))

    M = x.shape[0]
    ang1 = np.stack((x,y),axis=-1)
    ang2 = calc_angles(M,-calc_point(M,ang1))

    for i in range(x.shape[0]):
        for n in range(N):
            if weights[n] > 0:
                d1v = np.array([x[i] - means[n, 0], y[i] - means[n, 1]])
                d2v = np.array([x[i] - means[n, 0] + pi, y[i] + means[n, 1]])
                d3v = np.array([x[i] - means[n, 0] - pi, y[i] + means[n, 1]])
                d4v = np.array([x[i] + means[n, 0], y[i] - means[n, 1] - pi])
                d5v = np.array([x[i] + means[n, 0], y[i] - means[n, 1] - pi])

                d[0] = np.dot(d1v, d1v)
                d[1] = np.dot(d2v, d2v)
                d[2] = np.dot(d3v, d3v)
                d[3] = np.dot(d4v, d4v)
                d[4] = np.dot(d5v, d5v)

                d *= (-0.5 / (sigma * sigma))
                #
                # d1v = np.array([ang1[i,0] - means[n, 0], ang1[i,1] - means[n, 1]])
                # d2v = np.array([ang2[i,0] - means[n, 0], ang2[i,1] - means[n, 1]])
                # d[0] = np.dot(d1v, d1v)
                # d[1] = np.dot(d2v, d2v)
                # d *= (-0.5 / (sigma * sigma))

                p = np.exp(d)

                response[i] += np.sum(p) * weights[n]

    return response


def calc_pdf_grid(pdfpar, X, Y):

    # means = pdfpar['means']
    # weights = pdfpar['weights']
    # sigma = pdfpar['sigma']
    means = pdfpar.means
    weights = pdfpar.weights
    sigma = pdfpar.sigma

    N = weights.shape[0]

    if not (N == means.shape[0]):
        print "means has wrong shape!"
        return 0

    response = np.zeros((X.shape))

    for j in range(X.shape[1]):

        print j

        response[:,j] = calc_pdf(pdfpar, X[:,j], Y[:,j])

    return response


def pdf_params(cnn_response, confidence=1.282):


    A = cnn_response.shape[0]
    B = cnn_response.shape[1]
    N = A*B

    # confidence = 1.645 # 90 %
    # confidence = 1.282 # 80 %
    # confidence = 1.000 # 68 %

    sigma = pi / (confidence*A)

    alphas = np.linspace(-(A-1.0)/A*pi/2, (A-1.0)/A*pi/2, A)
    alphas = np.matlib.repmat(alphas, B, 1)
    betas = np.linspace(-(B-1.0)/B*pi/2, (B-1.0)/B*pi/2, B)
    betas  = np.matlib.repmat(betas, A, 1)
    betas = betas.T

    alphas = alphas.flatten()
    betas  = betas.flatten()

    total_weight = np.sum(cnn_response)

    weights = cnn_response.flatten() #/ total_weight

    weights_argsort = np.argsort(weights)
    weights_argsort = weights_argsort[::-1]

    # weights[weights < np.median(weights)] = 0
    weights[weights_argsort[100:]] = 0

    # weights[weights < weights.mean()/5 ] = 0
    weights /= np.sum(weights)
    weights /= (2 * pi * sigma * sigma)

    means = np.zeros((N,2))
    means[:,0] = alphas
    means[:,1] = betas

    # return {'means':means, 'weights':weights, 'sigma':sigma}
    return PDFParams(means=means, weights=weights, sigma=sigma)

def calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure="angle", outlier_stdd=1e-6 ):

    M = v.shape[1]
    N = l.shape[0]

    angles = calc_angles(M, v[i, :, :])
    p_v = calc_pdf(pdfpar, angles[:, 0], angles[:, 1])

    if distance_measure == "angle":
        lvsq = calc_lvsq_angle(v[i, :, :].T, l, lp, llen)
    elif distance_measure == "dotprod":
        lvsq = calc_lvsq_dotprod(v[i, :, :].T, l, lp, llen)
    elif distance_measure == "area":
        lvsq = calc_lvsq_area(v[i, :, :].T, l, lp, llen)

    p_lv = calc_plv(M, v[i,:,:].T, s, lvsq, lp)

    p_l = np.dot(p_lv, p_v)
    p_l = np.maximum(p_l, 1e-12)
    # p_l = np.maximum(p_l, 1e-15)
    # p_l += 1/(np.sqrt(2*np.pi)*outlier_stdd)
    # p_l = np.maximum(p_l, 1/(np.sqrt(2*np.pi)*outlier_stdd))

    p_vl = calc_pvl(M, N, p_lv, p_v, p_l)

    # return {'v':p_v, 'lv':p_lv, 'vl':p_vl, 'l':p_l, 'lvsq':lvsq, 'angles':angles}
    return PDF(v=p_v, lv=p_lv, vl=p_vl, l=p_l, lvsq=lvsq, angles=angles)

def calc_pvl(M, N, p_lv, p_v, p_l):
    p_vl = np.zeros((M, N))

    for n in range(N):
        for m in range(M):
            p_vl[m, n] = p_lv[n, m] * p_v[m] / p_l[n]

    return p_vl

def calc_plv(M, v, s, lvsq, lp):

    N = lp.shape[0]

    lve = lvsq.copy()
    for m in range(M):
        s[m] = s[m] if s[m] > 1e-200 else 1e-200
        lve[:, m] /= (2 * s[m])

    p_lv = np.exp(-lve)

    # num_cores = multiprocessing.cpu_count() / 2
    # angles = Parallel(n_jobs=num_cores)(delayed(calc_vp_line_triangles)(v[:,m], lp[:,:]) for m in range(M))
    #
    # for m in range(M):
    #     angle_array = np.array(angles[m])
    #     p_lv[angle_array > 0, m] = 0


    for m in range(M):
        p_lv[:, m] *= 1.0 / np.sqrt(2 * np.pi * s[m])

    return p_lv

def calc_lvsq_dotprod(v, l, lp, llen):
    lv = np.dot(l, v)
    lvsq = lv * lv

    return lvsq

def calc_lvsq_angle(v, l, lp, llen):

    M = v.shape[1]
    N = l.shape[0]

    lvsq = np.zeros((N,M))

    for m in range(M):
        v_ = v[0:2,m].copy()
        v_ /= v[2,m]

        for n in range(N):
            lm = 0.5 * (lp[n,0:2] + lp[n,2:4])

            vec1 = lm - v_.T
            vec2 = lp[n,0:2] - lp[n,2:4]

            lvsq[n,m] = (1-np.abs(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))**2  # 1-cos(phi)
            # lvsq[n,m] = (np.tan(np.arccos(np.abs(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))))**2  # tan(phi)
            # lvsq[n,m] = (np.arccos(np.abs(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))))**2  # (phi)

    return lvsq

def calc_lvsq_area(v, l, lp, llen):

    M = v.shape[1]
    N = l.shape[0]

    lvsq = np.zeros((N,M))

    for m in range(M):
        v_ = v[0:2,m].copy()
        v_ /= v[2,m]

        for n in range(N):


            lm = 0.5 * (lp[n,0:2] + lp[n,2:4])

            lp1 = np.ones(3)
            lp1[0:2] = lp[n,0:2].copy()

            lmh = np.ones(3)
            lmh[0:2] = lm[0:2].copy()

            vl = np.cross(v_, lmh)
            vl /= np.linalg.norm(vl[0:2])

            b = np.abs(np.dot(vl, lp1))
            c = np.linalg.norm(lm - lp[n,2:4])
            a = np.sqrt(c**2 - b**2)

            # vec1 = lm - v_.T
            # vec2 = lp[n,0:2] - lp[n,2:4]

            # cosphi = np.abs(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

            # lvsq[n,m] = (c**2 - b**2) * (b / cosphi)**2
            # lvsq[n,m] = (b*c)**2
            lvsq[n,m] = (a*(b**2)/c)**2

    return lvsq

def calc_lvsq_single(v, l, lp):

    v_ = v[0:2].copy()
    v_ /= v[2]


    lm = 0.5 * (lp[0:2] + lp[2:4])

    vec1 = lm - v_
    vec2 = lp[0:2] - lp[2:4]

    lvsq = (1-np.abs(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))**2  # 1-cos(phi)
    # lvsq = (np.tan(np.arccos(np.abs(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))))**2  # tan(phi)
    # lvsq = (np.arccos(np.abs(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))))**2  # (phi)

    return lvsq

def calc_lvsq_area_single(v, l, lp):

    v_ = v[0:2].copy()
    v_ /= v[2]

    lm = 0.5 * (lp[0:2] + lp[2:4])

    lp1 = np.ones(3)
    lp1[0:2] = lp[0:2].copy()

    lmh = np.ones(3)
    lmh[0:2] = lm[0:2].copy()

    vl = np.cross(v_, lmh)
    vl /= np.linalg.norm(vl[0:2])

    b = np.abs(np.dot(vl, lp1))
    c = np.linalg.norm(lm - lp[2:4])
    a = np.sqrt(c ** 2 - b ** 2)

    # vec1 = lm - v_.T
    # vec2 = lp[0:2] - lp[2:4]

    # cosphi = np.abs(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    # lvsq = (c ** 2 - b ** 2) * (b / cosphi) ** 2
    # lvsq = (b * c) ** 2
    lvsq = (a*(b**2)/c)**2

    return lvsq

def calc_angles(M, v):
    angle = np.zeros((M, 2))
    angle[:, 1] = np.arcsin(v[:,1])
    inner = v[:,0] / np.cos(angle[:, 1])
    inner = np.minimum(inner, 1)
    inner = np.maximum(inner, -1)
    angle[:, 0] = np.arcsin(inner)
    return angle

def calc_point(M, a):
    v = np.zeros((M, 3))
    v[:,0] = np.sin(a[:,0])*np.cos(a[:,1])
    v[:,0] = np.cos(a[:,1])
    v[:,0] = np.cos(a[:,0])*np.cos(a[:,1])
    return v


def pdf_grid(cnn_response, N=50):

    pdfpar = pdf_params(cnn_response)

    Av = N
    Bv = N
    Nv = Av*Bv

    X = np.arange(-pi/2, pi/2, pi*1.0/Av)
    Y = np.arange(-pi/2, pi/2, pi*1.0/Bv)
    X, Y = np.meshgrid(X, Y)

    aview = np.linspace(-pi/2+1.0/Av, pi/2, Av)
    aview = np.matlib.repmat(aview, Bv, 1)
    bview = np.linspace(-pi/2+1.0/Bv, pi/2, Bv)
    bview = np.matlib.repmat(bview, Av, 1)
    bview = bview.T

    aview = aview.flatten()
    bview = bview.flatten()

    x = np.zeros((Nv,2))
    x[:,0] = aview
    x[:,1] = bview

    pdf = calc_pdf_grid(pdfpar, X, Y)

    return {'X':X, 'Y':Y, 'p':pdf }


def calc_vp_line_triangles(vp, lines):

    v = vp[0:2] / vp[2]

    angles = np.zeros(lines.shape[0])
    for i in range(lines.shape[0]):

        lp1 = lines[i,0:2]
        lp2 = lines[i,2:4]

        angle_lp1 = np.dot( v-lp1, lp2-lp1 )
        if angle_lp1 > 0:
            angle_lp2 = np.dot( v-lp2, lp1-lp2 )
            angles[i] = np.minimum( angle_lp1, angle_lp2 )
        else:
            angles[i] = angle_lp1

    return angles

def vp_is_within_image(vp):
    vp2 = vp[0:2] / vp[2]
    if vp2[0] < 2 and vp2[0] > -2 and vp2[1] < 2 and vp2[1] > -2:
        return True
    else:
        return False