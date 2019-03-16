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
