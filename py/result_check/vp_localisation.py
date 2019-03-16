import numpy as np
import numpy.matlib
from numpy import linalg as la
import sklearn.cluster as cluster
import probability_functions as prob
from joblib import Parallel, delayed
import multiprocessing
import coordinate_conversion as coconv

pi = np.pi


def find_maxima(cnn_response):

    maxima = np.zeros(cnn_response.shape)

    A = cnn_response.shape[1]
    B = cnn_response.shape[0]

    for b in range(B):
        for a in range(A):
            vm = cnn_response[b,a]
            vu = cnn_response[b,a+1] if a+1 < A else 0
            vd = cnn_response[b,a-1] if a-1 > 0 else 0
            vl = cnn_response[b-1,a] if b-1 > 0 else 0
            vr = cnn_response[b+1,a] if b+1 < B else 0

            if vm > vu and vm > vd and vm > vl and vm > vr:
                maxima[b,a] = 1

    return maxima


def line_rating_knn(l, lp, k1=10, k2=3, sigma=1):

    N = l.shape[0]

    lscore = np.zeros(N)

    k1 = np.minimum(k1, N)
    k2 = np.minimum(k2, N)

    num_cores = multiprocessing.cpu_count() #/ 2
    ldist = Parallel(n_jobs=num_cores)(delayed(calc_ldist_parfun)(i, lp) for i in range(N))
    ldist = np.stack(ldist)

    ldist_argsorted = np.argsort(ldist, axis=1)
    ldist_argbest = ldist_argsorted[:,0:k1]

    for li in range(N):

        l1 = l[li,:]
        lp1 = lp[li,:]

        cosphi = np.zeros(k1)
        for ki in range(k1):
            cosphi[ki] = lines_points_cosangle(lp[li,:], lp[ldist_argbest[li,ki],:], f=9)

        cosphi_argsorted = np.argsort(cosphi)
        cosphi_argsorted = cosphi_argsorted[::-1]
        cosphi_argbest = cosphi_argsorted[0:k2]

        lsim_temp = np.zeros(k2)
        for ki in range(k2):
            lj = ldist_argbest[li,cosphi_argbest[ki]]
            l2 = l[lj,:]
            lp2 = lp[lj,:]
            prox = lines_proximity(l1, lp1, l2, lp2, sigma)
            lsim_temp[ki] = prox * cosphi[cosphi_argbest[ki]]

        lscore[li] = np.sum(lsim_temp)

    lscore /= k2

    return lscore


def calc_ldist_parfun(i, lp):
    N = lp.shape[0]
    ldist = np.zeros(N)
    for j in range(N):
        if i != j:
            ldist[j] = line_distance_closest(lp[i, :], lp[j, :])
        else:
            ldist[j] = 4

    return ldist


def calc_lsim(l, lp, sigma=0.1):

    num_cores = multiprocessing.cpu_count() #/ 2
    N = l.shape[0]

    lsim = Parallel(n_jobs=num_cores)(delayed(calc_lsim_parfun)(i, l, lp, sigma) for i in range(N))
    lsim = np.stack(lsim)

    for i in range(N):
        for j in range(i, N):
            lsim[i,j] = lsim[j,i]

    return lsim


def calc_lsim_parfun(i, l, lp, sigma):
    N = l.shape[0]
    lsim = np.zeros(N)
    for j in range(i):
        lsim[j] = lines_similarity(l[i, :], lp[i, :], l[j, :], lp[j, :], sigma=sigma)

    return lsim


def find_initial_vps(sphere_image, cnn_response, num_max):

    sphere = sphere_image.copy()
    sphere = sphere[::-1,:]

    rA = cnn_response.shape[0]
    rB = cnn_response.shape[1]
    sA = sphere_image.shape[0]
    sB = sphere_image.shape[1]

    maxima = find_maxima(cnn_response).flatten()
    flat_cnn_response = cnn_response.flatten()
    best_maxima = np.argsort(flat_cnn_response[maxima==1])
    best_maxima = best_maxima[::-1]
    maxima[np.where(maxima==1)[0][best_maxima[num_max:]]] = 0
    maxima = np.reshape(maxima, cnn_response.shape)

    vps = []

    for ra in range(rA):
        for rb in range(rB):
            if maxima[ra,rb] == 1:
                sphere_slice = sphere[ra*sA/rA:(ra+1)*sA/rA, rb*sB/rB:(rb+1)*sB/rB]

                max_response = np.max(sphere_slice)
                sphere_slice_flat = sphere_slice.flatten()
                sphere_slice_flat[sphere_slice_flat < max_response] = 0
                maxed_idx = np.where(sphere_slice_flat>0)[0]
                unraveled_indices = []

                if maxed_idx.shape[0] == 0:
                    continue

                for i in range(maxed_idx.shape[0]):
                    unraveled = np.unravel_index(maxed_idx[i], sphere_slice.shape)
                    unraveled_indices.append(unraveled)

                average_index = np.zeros(2)
                for idx in unraveled_indices:
                    average_index += idx
                average_index /= len(unraveled_indices)

                max_response = average_index

                max_index = np.zeros(2)

                max_index[1] = max_response[0] + ra*sA/rA
                max_index[0] = max_response[1] + rb*sB/rB

                angle = coconv.indexToAngle(max_index, sphere_image.shape)
                vp = coconv.angleToPoint(angle)

                vps.append(vp)

    return np.vstack(vps)


def expectation_maximisation(l, lp, cnn_response, num_iter=100, sphere_image=None,
                             init_vp=None, do_merge=True, do_split=True, do_iterations=True,
                             distance_measure="angle", use_weights=True, wbias=1, num_init_vp=25, split_merge_freq=10,
                             merge_thresh=1e-3, outlier_thresh=1.96**2, final_convergence=5e-3,
                             s_thresh=1e-200, num_min_lines=3):

    N = l.shape[0]
    print "Number of line segments: ", N

    if use_weights:
        lsim = calc_lsim(l, lp, sigma=1)
    else:
        lsim = np.zeros((l.shape[0], l.shape[0]))

    lv = np.zeros((l.shape[0], 2))
    lm = np.zeros((l.shape[0], 2))

    for i in range(l.shape[0]):
        l[i,:] /= np.linalg.norm(l[i,:])
        lv[i,:] = lp[i,0:2] - lp[i,2:4]
        lm[i,:] = (lp[i,0:2] + lp[i,2:4])*0.5

    merge_thresh_final = merge_thresh*10
    merge_freq = split_merge_freq
    split_freq = split_merge_freq
    split_merge_it = 100
    splits = 1

    if distance_measure == "angle":
        max_stdd = 1e-6
        s_init_factor = 1e-6
    elif distance_measure == "dotprod":
        max_stdd = 1e-3
        s_init_factor = 1e-3
    elif distance_measure == "area":
        max_stdd = 1e-6
        s_init_factor = 1e-6
    else:
        assert False

    result = {'vp_assoc': None, 'vp': None, 'counts': None, 'count_id': None,
              'decision_metric': None, 'iterations': 0}

    v0 = find_initial_vps(sphere_image, cnn_response, num_init_vp)

    pdfpar = prob.pdf_params(cnn_response)

    if not (init_vp is None):
        v0 = init_vp.copy()
        for m in range(v0.shape[0]):
            v0[m,:] /= np.linalg.norm(v0[m,:])

    langles = lines_angles(lp)

    s_init = pdfpar.sigma * s_init_factor

    outlier_stdd = 1

    llen = np.ones(l.shape[0])

    for li in range(l.shape[0]):
        l[li,:] /= np.linalg.norm(l[li,:], ord=2)
        llen[li] = np.linalg.norm(np.array([lp[li,0]-lp[li,2], lp[li,1]-lp[li,3]]), ord=2)

    if use_weights:
        lscore = line_rating_knn(l, lp, k2=4)
        lscore = np.clip(lscore, 0.2, 1)
        lweight = llen
        lweight *= lscore
    else:
        lweight = np.ones(N)

    M = v0.shape[0]


    s = np.ones((M)) * s_init

    v = np.zeros((num_iter+1, v0.shape[0], v0.shape[1]))

    v[0,:,:] = v0.copy()

    p = prob.calc_probabilities(0, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=100)
    w = weight_matrix(p.vl, lweight, lsim, bias=wbias)
    counts, counts_weighted, vp_assoc = calc_vp_line_counts(v[0,:,:], l, lp, s, p, w, lweight, distance_measure, thresh=outlier_thresh, outlier_stdd=100)

    v = np.delete(v, np.where(counts < 3)[0], axis=1)
    s = np.delete(s, np.where(counts < 3)[0], axis=0)

    M = v.shape[1]
    print "Number of initial VPs: ", M

    for i in range(num_iter):

        if M == 0:
            print "No VPs left!"
            return result

        if np.mod(i, split_freq) == 0 and i > 0 and i < split_merge_it and do_split:
            for it in range(splits):
                p = prob.calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)
                w = weight_matrix(p.vl, lweight, lsim, bias=wbias)
                split = split_best_vp(i, v, s, linePoints=lp, lines=l, weightMatrix=w, lineWeights=lweight, lineAngles=langles, min_diff=merge_thresh)
                v = split['v'].copy()
                s = split['s'].copy()

        M = v.shape[1]

        p = prob.calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)

        max_err = 0
        max_id = 0

        to_be_removed = []

        lweight_temp = lweight.copy()

        w = weight_matrix(p.vl, lweight_temp, lsim, bias=wbias)

        for m in range(M):

            if not do_iterations:
                break

            wtemp = w[m,:]
            ltemp = l

            newVP = calc_new_vanishing_point(ltemp, wtemp)

            if newVP is None:
                to_be_removed.append(m)
                continue
            else:
                v[i + 1, m, :] = newVP

            try:
                p_vl_sum = np.sum(p.vl[m,:])

                s_log = np.log(np.sum(p.lvsq[:,m] * p.vl[m,:])) - np.log(p_vl_sum)
                s[m] = np.exp( s_log )

                s[m] = np.minimum(s[m], max_stdd)
                s[m] = np.maximum(s[m], s_thresh)

                if np.isnan(s[m]):
                    to_be_removed.append(m)
                else:
                    err = np.arccos(np.minimum(np.abs(np.dot(v[i,m,:], v[i+1,m,:])), 1.0))
                    max_err = np.maximum(max_err, err)
                    max_id = m if max_err == err else max_id

                    if err > 1.5:
                        to_be_removed.append(m)

            except np.linalg.linalg.LinAlgError as err:
                print err
                to_be_removed.append(m)
                continue

        if not do_iterations:
            v[i + 1, :, :] = v[i , :, :].copy()

        print "%03d - max. VP change: %.4f - VPs: %d" % (i, max_err, M)

        to_be_removed = np.array(to_be_removed)
        v = np.delete(v, to_be_removed, axis=1)
        s = np.delete(s, to_be_removed, axis=0)
        p = prob.calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)
        M = v.shape[1]

        if max_err < final_convergence or i == num_iter-1 or not do_iterations:
            print "reached convergence"

            if do_merge:
                merged = merge_vps(i + 1, v, p, s, l, merge_thresh_final, lweight, lsim, wbias, pdfpar, lp, llen, distance_measure, outlier_stdd=outlier_stdd)
                v = merged['v']
                s = merged['s']

            p = prob.calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)

            w = weight_matrix(p.vl, lweight_temp, lsim, bias=wbias)

            to_be_removed = []
            assoc = np.argmax(w, axis=0)

            M = v.shape[1]

            for m in range(M):

                if np.size(w[m, assoc == m]) == 0:
                    continue

                w[m, assoc == m] /= np.max(w[m, assoc == m])

                wtemp = w[m, assoc==m]
                ltemp = l[assoc==m,:]

                newVP = calc_new_vanishing_point(ltemp, wtemp)

                if newVP is None:
                    to_be_removed.append(m)
                    continue
                else:
                    v[i + 1, m, :] = newVP

                try:
                    p_vl_sum = np.sum(p.vl[m, :])

                    s_log = np.log(np.sum(p.lvsq[:, m] * p.vl[m, :])) - np.log(p_vl_sum)
                    s[m] = np.exp(s_log)

                    s[m] = np.minimum(s[m], max_stdd)

                    if np.isnan(s[m]) or s[m] < s_thresh:
                        to_be_removed.append(m)
                    else:

                        err = np.arccos(np.minimum(np.abs(np.dot(v[i, m, :], v[i + 1, m, :])), 1.0))
                        max_err = np.maximum(max_err, err)
                        max_id = m if max_err == err else max_id

                        if err > 1.5:
                            to_be_removed.append(m)

                except np.linalg.linalg.LinAlgError as err:
                    to_be_removed.append(m)
                    continue

            to_be_removed = np.array(to_be_removed)
            v = np.delete(v, to_be_removed, axis=1)
            s = np.delete(s, to_be_removed, axis=0)
            p = prob.calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)
            M = v.shape[1]

            p = prob.calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)

            decision_metric = weight_matrix(p.vl, lweight, lsim, bias=wbias)

            if decision_metric.size <= 0:
                print "decision metric is empty"
                return result

            max_decision = np.argmax(decision_metric, axis=0)

            good_vp = np.unique(max_decision)

            print "Number of VPs: ", good_vp.size

            v = v[:,good_vp,:]
            s = s[good_vp]

            M = v.shape[1]

            p = prob.calc_probabilities(i+1, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)

            decision_metric = weight_matrix(p.vl, lweight, lsim, bias=wbias)
            counts, counts_weighted, vp_assoc = calc_vp_line_counts(v[i+1,:,:], l, lp, s, p, decision_metric, lweight, distance_measure, thresh=outlier_thresh, outlier_stdd=outlier_stdd)

            M = v.shape[1]

            vidx = 0
            while vidx < M:
                if counts[vidx] < num_min_lines:
                    v = np.delete(v, vidx, axis=1)
                    s = np.delete(s, vidx)
                    M = v.shape[1]

                    p = prob.calc_probabilities(i + 1, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)
                    decision_metric = weight_matrix(p.vl, lweight, lsim, bias=wbias)
                    counts, counts_weighted, vp_assoc = calc_vp_line_counts(v[i + 1, :, :], l, lp, s, p, decision_metric, lweight, distance_measure, thresh=outlier_thresh, vp_assoc=None, outlier_stdd=outlier_stdd)

                else:
                    vidx += 1

            vp = v[i + 1, :, :]

            return {'vp_assoc':vp_assoc,'vp':vp, 'counts':counts, 'counts_weighted':counts_weighted, 'count_id':None, 'decision_metric':decision_metric, 'iterations':i, 'distribution':p, 'sigma':s }

        if np.mod(i,merge_freq) == 0 and i > 0 and i <= split_merge_it+merge_freq and do_merge:
            merged = merge_vps(i+1, v, p, s, l, merge_thresh, lweight, lsim, wbias, pdfpar, lp, llen, distance_measure, outlier_stdd=outlier_stdd)
            v = merged['v']
            s = merged['s']

    return result


def calc_new_vanishing_point(l, w):

    try:
        if np.size(w) == 0:
            return None

        if np.max(w) == 0:
            return None

        W = np.diag(w / np.max(w))
        A = l
        Mat = np.dot(W, A)

        U, S, V = la.svd(Mat)

        V = V.T

        vp = np.squeeze(V[:, 2])

        vp /= np.linalg.norm(vp, ord=2)

        vp *= np.sign(vp[2])

    except np.linalg.linalg.LinAlgError as err:
        vp = None

    return vp


def calc_vp_line_counts(vp, l, lp, s, p, decision_metric, lweights, distance_measure, thresh=2.57, vp_assoc=None, outlier_stdd=1e-6):

    N = l.shape[0]
    M = vp.shape[0]
    if vp_assoc is None:
        vp_assoc = np.argmax(decision_metric, axis=0)

    counts = np.zeros(M)
    counts_weighted = np.zeros(M)

    for n in range(N):
        m = vp_assoc[n]
        if m > -1:
            if distance_measure == "dotprod":
                dist = np.abs(np.dot(vp[m], l[n, :]))
            elif distance_measure == "angle":
                dist = prob.calc_lvsq_single(vp[m], l[n, :], lp[n, :])
            elif distance_measure == "area":
                dist = prob.calc_lvsq_area_single(vp[m], l[n, :], lp[n, :])
            else:
                assert False

            if dist > thresh * np.sqrt(s[m]):
                vp_assoc[n] = -1
            elif lweights[n] == 0:
                vp_assoc[n] = -1
            else:
                counts[m] += 1
                counts_weighted[m] += lweights[n]

    return counts, counts_weighted, vp_assoc


def weight_matrix(p_vl, lweight, lsim, bias=0.001):

    w = np.zeros(p_vl.shape)
    for m in range(w.shape[0]):
        w_ = p_vl[m, :] * lweight

        for k in range(w.shape[1]):
            w[m,k] = ( w_[k] + bias * lweight[k] * np.dot(w_, lsim[:, k]) ) / ( 1 + bias * lweight[k] * np.sum(lsim[:, k]) )

    return w


def split_best_vp(i, v, s, linePoints, lines, weightMatrix, lineWeights, lineAngles, numClusters=2, min_diff=0.0001):

    M = v.shape[1]
    N = lines.shape[0]

    mean_phi = np.zeros(M)
    stdd_phi = np.zeros(M)

    weightMatrixGreedy = np.zeros(weightMatrix.shape)
    weightIndices = weightMatrix.argmax(axis=0)
    for li in range(N):
        weightMatrixGreedy[weightIndices[li],li] = weightMatrix[weightIndices[li],li]
    weightMatrixGreedy /= weightMatrix.max()

    for m in range(M):
        # print(mean_phi.shape)
        # print(weightMatrixGreedy.shape)
        # print(lineAngles.shape)
        mean_phi[m] = np.mean(lineAngles[weightMatrixGreedy[m,:]>0])
        stdd_phi[m] = np.std(lineAngles[weightMatrixGreedy[m,:]>0])

    worstVPs = np.argsort(stdd_phi)
    worstVPs = worstVPs[::-1]

    worstVP = None
    for m in range(M):
        vpAssoc = np.argmax(weightMatrix, axis=0)
        assocLines = np.where(vpAssoc == worstVPs[m])[0]
        lp = linePoints[assocLines]
        l = lines[assocLines]
        Nworst = lp.shape[0]

        vp = v[i, m, :].copy()
        vp /= vp[2]

        if  Nworst > numClusters*4 and ( ( vp[0] > -1 and vp[1] > -1 and vp[0] < 1 and vp[1] < 1 ) ):
            worstVP = worstVPs[m]
            break

    if not (worstVP is None):

        stdd = s[worstVP] / numClusters

        Ldist = np.zeros((Nworst,Nworst))
        for li in range(Nworst):
            for lj in range(Nworst):
                if lj != li:
                    Ldist[li,lj] = 1-lines_points_cosangle(lp[li,:], lp[lj,:], f=2)


        model = cluster.AgglomerativeClustering(linkage='average', connectivity=Ldist, n_clusters=numClusters, affinity='precomputed')
        model.fit_predict(Ldist)

        labels = model.labels_

        lw = lineWeights[assocLines]

        l[:,0] *= lw
        l[:,1] *= lw
        l[:,2] *= lw

        new_vps = []

        for c in range(numClusters):

            lineSet = l[labels == c]

            if lineSet.shape[0] < 3:
                continue

            U, S, V = la.svd(lineSet)
            V = V.T
            vp = np.squeeze(V[:, 2])
            vp /= np.linalg.norm(vp, ord=2)
            if vp[2] < 0:
                vp *= -1

            new_vps.append(vp)

        too_similar = True

        for c in range(len(new_vps)):
            for d in range(c+1, len(new_vps)):
                vp1 = new_vps[c]
                vp2 = new_vps[d]

                cosphi = np.clip(np.dot(vp1, vp2.T), -1, 1)
                angle = np.abs(np.arccos(np.clip(np.abs(cosphi), -1, 1)))

                if angle > min_diff:
                    too_similar = False

        if not too_similar:
            first = True
            for c in range(len(new_vps)):
                vp = new_vps[c]
                if first:
                    v[i, worstVP, :] = vp.copy()
                    s[worstVP] = stdd
                    first = False
                else:
                    v = np.append(v, np.zeros((v.shape[0],1,v.shape[2])), axis=1)
                    s = np.append(s,stdd)
                    v[i, -1, :] = vp.copy()

    return {'v':v, 's':s}


def merge_vps(i, v, p, s, l, thresh, lweight, lsim, wbias, pdfpar, lp, llen, distance_measure, max_stdd=0.01, outlier_stdd=1e-6):

    M = v.shape[1]

    num_cores = multiprocessing.cpu_count() #/ 2

    tryAgain = True
    tries = 0

    while tryAgain and M > 1:

        tries += 1

        angles = Parallel(n_jobs=num_cores)(delayed(calc_angle_to_other_vp)(v, i, j) for j in range(M))
        angles = np.stack(angles)

        argmin_angle = numpy.unravel_index(angles.argmin(), angles.shape)
        j = argmin_angle[0]
        k = argmin_angle[1]
        min_angle = angles[j, k]

        if min_angle < thresh:

            try:
                p = prob.calc_probabilities(i, pdfpar, v, l, lp, s, llen, distance_measure, outlier_stdd=outlier_stdd)
                w = weight_matrix(p.vl, lweight, lsim, bias=wbias)

                newVP = calc_new_vanishing_point(l, w[j, :] + w[k, :])

                p_vl_sum = np.sum(p.vl[k, :] + p.vl[j, :])
                s_log = np.log(np.sum(0.5 * (p.lvsq[:, j] + p.lvsq[:, k]) * (p.vl[k, :] + p.vl[j, :]))) - np.log(p_vl_sum)
                s[k] = np.exp(s_log)

                if newVP is None or s[k] > max_stdd:
                    tryAgain = False
                    continue
                else:
                    v[i, k, :] = newVP

                v = np.delete(v, j, axis=1)
                s = np.delete(s, j, axis=0)

            except np.linalg.linalg.LinAlgError as err:
                continue
        else:
            tryAgain = False

        M = v.shape[1]

    return {'v':v, 's':s}


def calc_angle_to_other_vp(v, i, k):

    thisVP = np.squeeze(v[i,k,:])
    otherVPs = np.squeeze(v[i,:,:])
    cosphi = np.clip(np.dot(otherVPs, thisVP.T), -1, 1)
    angles = np.abs(np.arccos(np.clip(np.abs(cosphi), -1, 1)))
    if np.isscalar(angles):
        angles = pi
    else:
        angles[k] = pi
    return angles


def lines_similarity(l1, lp1, l2, lp2, sigma=0.1):
    cosphi = lines_points_cosangle(lp1, lp2, f=9)

    sim = cosphi * lines_proximity(l1, lp1, l2, lp2, sigma)

    return sim


def lines_proximity(l1, lp1, l2, lp2, sigma=0.1):
    sigma = sigma*np.minimum(line_length(lp1), line_length(lp2))
    d = line_distance_closest(lp1,lp2)
    prox = np.exp(-(d * d) / (2 * sigma * sigma))
    return prox


def lines_points_cosangle(lp1,lp2,f=1):
    v1 = lp1[0:2] - lp1[2:4]
    v2 = lp2[0:2] - lp2[2:4]

    cosdphi = np.abs(np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))

    dphi = np.abs(np.arccos(np.clip(cosdphi, -1, 1)))
    cosdphi = np.cos(np.clip(f*dphi,-pi/2,pi/2))

    return cosdphi


def line_distance_closest(lp1, lp2):
    l1p1 = np.array([lp1[0], lp1[1], 1])
    l1p2 = np.array([lp1[2], lp1[3], 1])
    l2p1 = np.array([lp2[0], lp2[1], 1])
    l2p2 = np.array([lp2[2], lp2[3], 1])

    d1 = line_segment_point_distance(lp1, l2p1)
    d2 = line_segment_point_distance(lp1, l2p2)
    d4 = line_segment_point_distance(lp2, l1p1)
    d5 = line_segment_point_distance(lp2, l1p2)

    d = np.min(np.array([d1,d2,d4,d5]))

    return d


def line_segment_point_distance(lp, p):
    lp1 = np.array([lp[0], lp[1], 1])
    lp2 = np.array([lp[2], lp[3], 1])

    param = np.dot(p-lp1, lp2-lp1) / np.square(np.linalg.norm(lp2-lp1))

    pclosest = None

    if param < 0:
        pclosest = lp1
    elif param > 1:
        pclosest = lp2
    else:
        pclosest = lp1 + param*(lp2-lp1)

    d = np.linalg.norm(pclosest-p)

    return d


def line_length(lp):
    return np.linalg.norm(lp[0:2]-lp[2:4], ord=2)


def lines_angles(lp):
    N = lp.shape[0]
    angles = np.zeros(N)

    for i in range(N):
        v = np.array([lp[i,0] - lp[i,2], lp[i,1] - lp[i,3]])
        v /= np.linalg.norm(v)
        phi = np.abs(np.arccos(np.clip(v[0], -1, 1)))
        phi = pi-phi if phi > pi/2 else phi
        angles[i] = phi

    return angles


# def plot_current_result(i, v, p_v, weight, angle, true_vp, true_vp_ang, sphere_image, cnn_prediction, vp_counts=None, good_vp_only=False, s=None):
#
#     imgSize = sphere_image.shape[0]
#     cnnSize = cnn_prediction.shape[0]
#
#     fig = plt.figure()
#     ax1 = plt.subplot2grid((2, 3), (0, 0), projection='3d', colspan=2, rowspan=2)
#     ax2 = plt.subplot2grid((2, 3), (0, 2))
#     ax3 = plt.subplot2grid((2, 3), (1, 2))
#
#     ax1.set_title("VPs on unit sphere (3D)")
#     ax2.set_title("VPs on rendered sphere image")
#
#     if true_vp is not None:
#         for j in range(true_vp.shape[0]):
#             ax1.plot(true_vp[j, 0], true_vp[j, 1], true_vp[j, 2], 'bx', markersize=20)
#             pos = coconv.angleToIndex(true_vp_ang[j,:], (imgSize, imgSize))
#             ax2.plot(pos[0], imgSize-1-pos[1], 'bx', markersize=20,
#                      mew=2)
#
#     ax2.imshow(sphere_image, cmap='Greys_r')
#     ax2.autoscale(enable=False)
#
#     ax3.imshow(cnn_prediction[::-1,:], cmap='Greys_r', interpolation="none")
#     ax3.autoscale(enable=False)
#
#     max_p = p_v.max()
#
#     if good_vp_only is False:
#         for j in range(v.shape[1]):
#             if p_v[j] > 0:
#                 ax1.plot([v[i, j, 0]], [v[i, j, 1]], [v[i, j, 2]], 'o', c=[(1 - p_v[j] / max_p), p_v[j] / max_p, 0],
#                          markersize=np.maximum(weight[j] * 12, 2))
#
#                 pos = coconv.angleToIndex(angle[j,:], (imgSize, imgSize))
#                 ax2.plot(pos[0], imgSize-1-pos[1], 'o',
#                          c=[(1 - p_v[j] / max_p), p_v[j] / max_p, 0], markersize=np.maximum(weight[j] * 12, 2))
#
#                 pos = coconv.angleToIndex(angle[j,:], (cnnSize, cnnSize))
#                 ax3.plot(pos[0], cnnSize-1-pos[1], 'o',
#                          c=[(1 - p_v[j] / max_p), p_v[j] / max_p, 0], markersize=np.maximum(weight[j] * 12, 2))
#
#
#
#     if vp_counts is not None:
#         gvp = v[i, :, :]
#         gang = angle
#
#         pg = vp_counts * 1.0 / np.sum(vp_counts)
#
#         for j in range(gvp.shape[0]):
#
#             scaling = 100
#
#             ax1.plot([gvp[j, 0]], [gvp[j, 1]], [gvp[j, 2]], 'yo', markersize=np.maximum(pg[j] * scaling,1))
#
#             angle = gang[j, :]
#             pos = coconv.angleToIndex(angle, (imgSize, imgSize))
#             ax2.plot(pos[0], imgSize-1-pos[1], 'yo', markersize=np.maximum(pg[j] * scaling,1))
#             pos = coconv.angleToIndex(angle, (cnnSize, cnnSize))
#             ax2.plot(pos[0], cnnSize-1-pos[1], 'yo', markersize=np.maximum(pg[j] * scaling,1))
#
#             if s is not None:
#                 msize = np.maximum(np.sqrt(s[j]) * 100, 1)
#                 print "s= ", s[j]
#                 ax1.plot([gvp[j, 0]], [gvp[j, 1]], [gvp[j, 2]], 'o', markersize=msize, markerfacecolor='none', markeredgecolor='b', markeredgewidth=1)
#
#                 ax2.plot(pos[0], imgSize-1-pos[1], 'o', markersize=msize, markerfacecolor='none', markeredgecolor='b', markeredgewidth=1)
#
#     figManager = plt.get_current_fig_manager()
#     figManager.window.showMaximized()
#
#     plt.show()
#

def plot_result(ax1, ax2, vps, angles, vp_counts, best=None, imgSize=250, stdMark='yo', markersize=1):
    if vp_counts is not None:
        pg = vp_counts * 1.0 / np.sum(vp_counts)

    else:
        pg = np.ones(vps.shape[0])*0.1

    gvp = vps
    gang = angles

    for j in range(gvp.shape[0]):

        scaling = 100

        mark = stdMark
        if not (best is None):
            if j in best:
                mark = 'go'

        if not (ax1 is None):
            ax1.plot([gvp[j, 0]], [gvp[j, 1]], [gvp[j, 2]], mark, markersize=markersize*np.minimum(np.maximum(pg[j] * scaling,6),20))
        if not (ax2 is None):
            angle = gang[j, :]
            pos = coconv.angleToIndex(angle, (imgSize, imgSize))
            # print("pos:", pos)
            ax2.plot(pos[0], imgSize-1-pos[1], mark, markersize=markersize*np.minimum(np.maximum(pg[j] * scaling,6),20), alpha=0.6)

# def plot_single_vp(ax1, ax2, vp, angle, vp_count, imgSize=250, stdMark='yo'):
#     scaling = 1
#
#     mark = stdMark
#
#     if not (ax1 is None):
#         ax1.plot([vp[0]], [vp[1]], [vp[2]], mark, markersize=np.minimum(np.maximum(vp_count * scaling,1),20))
#     if not (ax2 is None):
#         pos = coconv.angleToIndex(angle, (imgSize, imgSize))
#         ax2.plot(pos[0], imgSize-1-pos[1], mark, markersize=np.minimum(np.maximum(vp_count * scaling,1),20))
#         # ax2.plot((angle[0] / pi + 0.5 - 0.5/imgSize) * (imgSize), (-angle[1] / pi + 0.5 - 0.5/imgSize) * (imgSize), mark, markersize=np.minimum(np.maximum(vp_count * scaling,1),20))
