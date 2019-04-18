import numpy as np

def numCombo3(n):
    if n >= 3:
        an = 3*numCombo3(n-1) - 3*numCombo3(n-2) + numCombo3(n-3) + 1
    else:
        an = 0
    return an


def VPinImage(vp):
    vp_ = vp/vp[2]
    if vp_[0] <= 1 and vp_[0] >= -1 and vp_[1] <= 1 and vp_[1] >= -1:
        return True
    else:
        return False


def calculate_horizon_and_ortho_vp(em_result, maxbest=10, theta_vmin=np.pi/10., theta_z=np.pi/4.):

    # vanishing points:
    vps = em_result['vp'].copy()

    # vanishing point scores (unweighted):
    counts = em_result['counts']

    # number of 'best' VPs to consider:
    num_best = np.minimum(maxbest, vps.shape[0])

    # possible zenith VPs:
    zenith_candidate_idx = np.where(np.abs(vps[:, 1]) > np.sin(theta_z))[0]

    # select the 'best' VPs:
    best_vps = np.argsort(counts)
    best_vps = best_vps[::-1]
    best_vps = best_vps[0:num_best]

    # number of possible VP triplets:
    num_combinations = numCombo3(num_best)

    # create list with all possible triplets:
    combinations = np.zeros((num_combinations, 3)).astype(int)

    combo_count = 0
    for i in range(num_best):
        for j in range(i, num_best):
            for k in range(j, num_best):
                if i != k and j != k and i != j:
                    combinations[combo_count, :] = np.array([i, j, k]).astype(int)
                    combo_count += 1

    # cosine of minimum VP distance as a threshold:
    costh = np.cos(theta_vmin)

    # calculate a score for each triplet:
    score_dist = np.zeros(num_combinations)
    score_weight = np.zeros(num_combinations)
    score = np.zeros(num_combinations)

    best_score = -1
    best_combo = 0
    hlin = None

    if num_best > 2:

        possible_solutions = []

        for i in range(num_combinations):
            a = combinations[i, 0]
            b = combinations[i, 1]
            c = combinations[i, 2]

            Va = vps[best_vps[a], :]
            Vb = vps[best_vps[b], :]
            Vc = vps[best_vps[c], :]

            # cos(phi) between the VPs:
            AB = (np.dot(Va, Vb))
            BC = (np.dot(Vb, Vc))
            AC = (np.dot(Va, Vc))

            AB = np.abs(AB)
            BC = np.abs(BC)
            AC = np.abs(AC)

            # number of zenith candidates in the triplet:
            num_zenith = 0
            if best_vps[a] in zenith_candidate_idx:
                num_zenith += 1
                zenith = vps[best_vps[a], :]
            if best_vps[b] in zenith_candidate_idx:
                num_zenith += 1
                zenith = vps[best_vps[b], :]
            if best_vps[c] in zenith_candidate_idx:
                num_zenith += 1
                zenith = vps[best_vps[c], :]

            # number of possible central perspective VPs:
            num_central = 0
            if VPinImage(Va):
                num_central += 1
            if VPinImage(Vb):
                num_central += 1
            if VPinImage(Vc):
                num_central += 1


            # assign horizon and zenith VPs:
            if np.abs(Va[1]) > np.abs(Vb[1]) and np.abs(Va[1]) > np.abs(Vc[1]):
                hVP1_temp = Vb
                hVP2_temp = Vc
                zVP_temp = Va
                h1Count = counts[best_vps[b]]
                h2Count = counts[best_vps[c]]
                zVP_id_temp = best_vps[a]
            elif np.abs(Vb[1]) > np.abs(Va[1]) and np.abs(Vb[1]) > np.abs(Vc[1]):
                hVP1_temp = Va
                hVP2_temp = Vc
                zVP_temp = Vb
                h1Count = counts[best_vps[a]]
                h2Count = counts[best_vps[c]]
                zVP_id_temp = best_vps[b]
            else:
                hVP1_temp = Va
                hVP2_temp = Vb
                zVP_temp = Vc
                h1Count = counts[best_vps[a]]
                h2Count = counts[best_vps[b]]
                zVP_id_temp = best_vps[c]


            zlin_temp = np.cross(zVP_temp, np.array([0,0,1]))
            zlin_temp /= np.linalg.norm(zlin_temp[0:2])

            l1 = zlin_temp[0]
            l2 = zlin_temp[1]

            v11 = hVP1_temp[0]
            v12 = hVP1_temp[1]
            v13 = hVP1_temp[2]
            v21 = hVP2_temp[0]
            v22 = hVP2_temp[1]
            v23 = hVP2_temp[2]

            d1 = np.linalg.norm(np.array([0,0,1]) - hVP1_temp/hVP1_temp[2])
            d2 = np.linalg.norm(np.array([0,0,1]) - hVP2_temp/hVP2_temp[2])

            h1 = -l2
            h2 = l1
            h3 = ( (v11*l2-v12*l1)/v13*(d2*h1Count) + (v21*l2-v22*l1)/v23*(d1*h2Count) ) / ((d1*h2Count)+(d2*h1Count)) ###!!!

            hlin_temp = np.array([h1, h2, h3])

            # angle of the proposed horizon line:
            hvec = (hVP1_temp / hVP1_temp[2]) - (hVP2_temp / hVP2_temp[2])
            hang = np.arccos(np.abs(np.dot(hvec, np.array([1, 0, 0]))) / np.linalg.norm(hvec))

            hP1 = np.cross(hlin_temp, np.array([1, 0, 1]))
            hP2 = np.cross(hlin_temp, np.array([-1, 0, 1]))
            hP1 /= hP1[2]
            hP2 /= hP2[2]


            # score for orthogonality of horizon, and vector between zenith and principal point:
            ortho_score = 0
            if num_zenith == 1:
                cosphi = np.abs(np.dot(hvec / np.linalg.norm(hvec), zenith / np.linalg.norm(zenith)))
                ortho_score = 1 - np.clip(1.0 * cosphi, 0, 1)

            zenithPos = 1 if zVP_temp[1] > 0 else -1
            horPos = 1 if (hP1[1]+hP2[1])/2 < 0 else -1

            # set score to zero if some minimum sanity checks are not met:
            # * min. angle between VPs
            # * number of zenith and central perspective VPs
            # * angle of horizon line (< 30 deg)
            score_dist[i] = 1 if ( AB < costh and BC < costh and AC < costh and num_zenith == 1 and num_central <= 1 \
                                 # and hang < 30 * np.pi / 180 ) and min_count > 4 else 0
                                 and hang < 30 * np.pi / 180 and zenithPos*horPos==1)  else 0
                                 # and hang < 30 * np.pi / 180)  else 0

            # sum of VP weights:
            score_weight[i] = counts[best_vps[a]] + counts[best_vps[b]] + counts[best_vps[c]] #+ 1*consistency_count

            # total score:
            score[i] = score_dist[i] * score_weight[i] * ortho_score

            if score[i] > 0:
                possible_solutions.append({"score":score[i], "zVP_id":zVP_id_temp, "hVP1":hVP1_temp, "hVP2":hVP2_temp, "h1Count":h1Count, "h2Count":h2Count, "horizon":hlin_temp})

            if score[i] > best_score:
                best_combo = i
                best_score = score[i]
                hVP1 = hVP1_temp
                hVP2 = hVP2_temp
                zVP = zVP_temp
                hlin = hlin_temp

        best_combo = best_vps[combinations[best_combo]]
    elif num_best > 1:
        hVP1 = vps[0,:]
        hVP2 = vps[1,:]
        zVP = np.array([0,1,0])
        best_combo = np.array([0,1])
        hlin = np.cross(hVP1, hVP2)
    elif num_best > 0:
        hVP1 = vps[0,:]
        hVP2 = vps[0,:]
        zVP = np.array([0,1,0])
        best_combo = np.array([0,0])
        hlin = np.cross(np.array([0,0,1]), np.array([1,0,1]))
    else:
        hVP1 = np.array([-1,0,0])
        hVP2 = np.array([1,0,0])
        zVP = np.array([0,1,0])
        best_combo = np.array([0,0])
        hlin = np.cross(np.array([0,0,1]), np.array([1,0,1]))


    hP1 = np.cross(hlin, np.array([1, 0, 1]))
    hP2 = np.cross(hlin, np.array([-1, 0, 1]))
    hP1 /= hP1[2]
    hP2 /= hP2[2]

    return ( hP1, hP2, zVP, hVP1, hVP2, best_combo )
