import numpy as np

def imr(modal_MR, lengths):
    num_sample = sum(lengths)

    count = 0
    mask = []

    # AVL
    ttt = int(num_sample*(1-modal_MR[0])*(1-modal_MR[1])*(1-modal_MR[2]))
    mask.extend([[1, 1, 1]] * (ttt))
    count += ttt

    # AV
    ttf = int(num_sample*(1-modal_MR[0])*(1-modal_MR[1])*modal_MR[2])
    mask.extend([[1, 1, 0]] * ttf)
    count += ttf

    # VL
    ftt = int(num_sample*modal_MR[0]*(1-modal_MR[1])*(1-modal_MR[2]))
    mask.extend([[0, 1, 1]] * ftt)
    count += ftt

    # AL
    tft = int(num_sample*(1-modal_MR[0])*modal_MR[1]*(1-modal_MR[2]))
    mask.extend([[1, 0, 1]] * tft)
    count += tft

    # A
    tff = int(num_sample*(1-modal_MR[0])*modal_MR[1]*modal_MR[2])
    mask.extend([[1, 0, 0]] * tff)
    count += tff

    # V
    ftf = int(num_sample*modal_MR[0]*(1-modal_MR[1])*modal_MR[2])
    mask.extend([[0, 1, 0]] * ftf)
    count += ftf

    # L
    fft = int(num_sample*modal_MR[0]*modal_MR[1]*(1-modal_MR[2]))
    mask.extend([[0, 0, 1]] * fft)
    count += fft

    # no modal
    left = num_sample - count
    tmp = left // 3
    mask.extend([[1, 0, 0]] * tmp)
    mask.extend([[0, 0, 1]] * tmp)
    mask.extend([[0, 1, 0]] * (left - 2 * tmp))
    np.random.shuffle(mask)

    return np.array(mask)


def smr(modal_MR, lengths):
    print(f'Create SMR mask with missing ratio {modal_MR[0]}')
    mr = modal_MR[0]
    one_rate = 1 - mr
    num_sample = sum(lengths)
    if one_rate == 1:
        mask = np.array([[1, 1, 1]] * num_sample)
    else:
        SINGLE_PATTERN = [[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]]
        if one_rate*3 <= 1:
            # If one rate <= 1/3 (MR upperbound=0.7 ~ each sample has one modal)
            # Generate mask such that all samples have one modal to avoid all zeros, one rate < 0.3 will be round up to 0.3
            idx = np.random.randint(0, 3, num_sample)
            mask = np.array([SINGLE_PATTERN[i] for i in idx])
        else:
            error = 1
            tmp = num_sample
            if num_sample < 32:
                num_sample = 32  # min sample number to generate mask
            # Keep re-generating until get a mask with expected one rate
            while error >= 0.005:
                # Generate 1-modal available mask
                idx = np.random.randint(0, 3, num_sample)
                preserve = np.array([SINGLE_PATTERN[i]
                                    for i in idx])

                # Remaining available mask needed = total available modal - generated available (i.e., sum(preserve)=num_sample)
                one_num = 3*num_sample*one_rate - num_sample
                ratio = one_num / (3*num_sample)
                # Generate remaining mask randomly with available probability = ratio, then combine with preserve mask
                mask_iter = (np.random.randint(0, 100, size=(
                    num_sample, 3)) < int(ratio*100))

                # Calculate overlap of 1-modal mask with remaining mask
                a = np.sum(((mask_iter + preserve) > 1))

                # Re-calculate ratio after removing overlap
                # ratio of non-overlapped ones
                one_num_iter = one_num / (1 - a / one_num)
                ratio = one_num_iter / (3 * num_sample)
                mask_iter = (np.random.randint(0, 100, size=(
                    num_sample, 3)) < int(ratio * 100))
                mask = ((mask_iter + preserve) > 0)
                ratio = np.sum(mask) / (3 * num_sample)
                error = abs(one_rate - ratio)
            mask = mask[:tmp]
            num_sample = tmp

    return np.array(mask)


def generate_modal_mask(modal_MR, lengths, uids):
    assert all(
        0 <= mr < 1 for mr in modal_MR), "Modal missing rates should be between 0 and 1."

    # generate missing index mask for each utterance, utt_mask: np.array [sum(lengths), 3-AVL]
    if len(modal_MR) == 3:
        utt_mask = imr(modal_MR, lengths)
    elif len(modal_MR) == 1:
        utt_mask = smr(modal_MR, lengths)
    else:
        raise ValueError(
            f"1 MR for SMR or 3 MRs for IMR expected, but got {len(modal_MR)}")

    missing_index = {'A': {}, 'V': {}, 'L': {}}
    count = 0
    for length, uid in zip(lengths, uids):
        sample_mask = np.array(utt_mask[count:count + length]).astype(bool)

        missing_index["A"][uid] = sample_mask[:, 0]
        missing_index["V"][uid] = sample_mask[:, 1]
        missing_index["L"][uid] = sample_mask[:, 2]

        count += length
        
    return missing_index