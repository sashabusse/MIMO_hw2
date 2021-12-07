import numpy as np
from scipy import io, signal
import matplotlib.pyplot as plt
from mimo_lib.simulate_transmission import simulate_transmission
from mimo_lib.spatial_filters import Linear, SvdFilter, DftFilter
from mimo_lib.TaskCfg import TaskCfg, load_link

'''
y expected to be in <bs_ind> x <sc_ind> format
SNR in dB
'''


def add_noise(y, SNR):
    # average signal power over antennas
    if y.ndim == 1:
        Ps = np.var(y, ddof=1)
    else:
        Ps = np.mean(np.var(y, axis=1, ddof=1))

    Dn = Ps / (10 ** (SNR / 10))

    n0 = (np.random.normal(size=y.shape) + \
          1j * np.random.normal(size=y.shape)) * np.sqrt(Dn / 2)

    return y + n0;


def noise_std(y, SNR):
    # average signal power over antennas
    if y.ndim == 1:
        Ps = np.var(y, ddof=1)
    else:
        Ps = np.mean(np.var(y, axis=1, ddof=1))

    Dn = Ps / (10 ** (SNR / 10))

    return np.sqrt(Dn)


chan_name = "chan_PATH"
link_channel = io.loadmat("Data/link_{}.mat".format(chan_name))['Link_Channel']
# new axes: <time> x <sub-carrier> x <UE antenna> x <BS antenna>
link_channel = np.moveaxis(link_channel, [3, 2], [0, 1])
# normalization of the channel
path_loss_avg = np.mean(np.abs(link_channel))
link_channel = link_channel / (np.sqrt(2.) * path_loss_avg)

# << loading channel data >>
# axes: <time> x <sub-carrier> x <UE antenna> x <BS antenna>
H = link_channel

# process all the svd at once to improve performance
H_u, H_s, H_vh = np.linalg.svd(H, full_matrices=False)
packet_cnt, sc_cnt, ue_ant_cnt, bs_ant_cnt = H.shape

# << loading SRS signals >>
srs_seqs = io.loadmat('srsSeqs.mat')['srsSeqs']
srs_sc = np.arange(12, 600-12, 2)

user_ind = 0
interf_ind = 4

user_ue = 0
interf_ue = 1

srs_user = srs_seqs[user_ind]
srs_user_t = np.fft.ifft(srs_user)
srs_interf = srs_seqs[interf_ind]
srs_interf_t = np.fft.ifft(srs_interf)


def channel_est_LS(y, srs_seq, srs_sc):
    H_LS = srs_seq.conj() * y
    return H_LS


def channel_est_LS_rect_wnd_hard(y, srs_seq, srs_sc, wnd_mid=-3, wnd_width=20):
    H_LS = srs_seq.conj() * y
    H_LS_t = np.fft.ifft(H_LS)

    wnd_st = (wnd_mid - wnd_width // 2) % y.shape[0]
    wnd_end = (wnd_st + wnd_width) % y.shape[0]

    if wnd_st <= wnd_end:
        wnd = np.zeros(y.shape, dtype=np.complex128)
        wnd[wnd_st:wnd_end] = 1.
    else:
        wnd = np.ones(y.shape, dtype=np.complex128)
        wnd[wnd_end: wnd_st] = 0.

    if wnd_width >= y.shape[0]:
        wnd = np.ones(y.shape[0], dtype=np.complex128)

    H_LS_wnd_t = wnd * H_LS_t
    H_LS_wnd = np.fft.fft(H_LS_wnd_t)

    return H_LS_wnd


def channel_est_LS_wnd_SNR(y, srs_seq, srs_sc, snr):
    H_LS = srs_seq.conj() * y
    H_LS_t = np.fft.ifft(H_LS)

    Ps_est = (y @ y.conj()).real / y.shape[0]

    # divide by N cause ifft
    Pn = (Ps_est * 10 ** (-0.1 * snr)) / y.shape[0]
    n_sigma = np.sqrt(Pn)

    window = np.ones(y.shape, dtype=np.complex128)

    window[np.abs(H_LS_t) < 3 * n_sigma] = 0.5
    window[np.abs(H_LS_t) < 2 * n_sigma] = 0.1
    window[np.abs(H_LS_t) < n_sigma] = 0

    window[18:270] = 0

    H_LS_wnd_t = window * H_LS_t
    H_LS_wnd = np.fft.fft(H_LS_wnd_t)

    return H_LS_wnd, n_sigma


def channel_est_LS_wnd_STD(y, srs_seq, srs_sc, n_sigma, s3_coef=0.5, s2_coef=0.1):
    H_LS = srs_seq.conj() * y
    H_LS_t = np.fft.ifft(H_LS)

    # divide cause ifft
    n_sigma /= np.sqrt(y.shape[0])

    window = np.ones(y.shape, dtype=np.complex128)

    window[np.abs(H_LS_t) < 3 * n_sigma] = 0.5
    window[np.abs(H_LS_t) < 2 * n_sigma] = 0.1
    window[np.abs(H_LS_t) < n_sigma] = 0

    window[18:270] = 0

    H_LS_wnd_t = window * H_LS_t
    H_LS_wnd = np.fft.fft(H_LS_wnd_t)

    return H_LS_wnd, n_sigma


Hh_user_00 = H[0, srs_sc, user_ue, 0].conj().T  # 0 time, srs, user ue, 0 bs_ant
Hh_interf_00 = H[0, srs_sc, interf_ue, 0].conj().T  # 0 time, srs, interf ue, 0 bs_ant

y = srs_user * Hh_user_00
y_interf = srs_interf * Hh_interf_00

snr_list = np.arange(-10, 31, 2)
wnd_width_list = [10, 20, 30, 280]
attempt_cnt = 50  # number of attempts for result averaging

err_LS = dict()
err_LS_interf = dict()

for wnd_width in wnd_width_list:
    err_LS[wnd_width] = np.zeros(len(snr_list), dtype=np.complex128)
    err_LS_interf[wnd_width] = np.zeros(len(snr_list), dtype=np.complex128)

    for snr_ind, snr in enumerate(snr_list):
        for q in range(attempt_cnt):
            yn = add_noise(y, snr);
            yni = yn + y_interf

            H_LS = channel_est_LS_rect_wnd_hard(yn, srs_user, srs_sc, wnd_width=wnd_width)
            H_LS_interf = channel_est_LS_rect_wnd_hard(yni, srs_user, srs_sc, wnd_width=wnd_width)

            err_LS[wnd_width][snr_ind] += np.linalg.norm(Hh_user_00 - H_LS) / np.linalg.norm(Hh_user_00)
            err_LS_interf[wnd_width][snr_ind] += np.linalg.norm(Hh_user_00 - H_LS_interf) / np.linalg.norm(Hh_user_00)

        err_LS[wnd_width][snr_ind] /= attempt_cnt
        err_LS_interf[wnd_width][snr_ind] /= attempt_cnt

# plotting results
plt.figure(figsize=(9, 5))

legend = []
for wnd_width in wnd_width_list:
    plt.plot(snr_list, np.abs(err_LS[wnd_width]) * 100, '-')
    plt.plot(snr_list, np.abs(err_LS_interf[wnd_width]) * 100, 'o')

    legend += ['LS(wnd={})'.format(wnd_width)]
    legend += ['LS(wnd={}) + interf'.format(wnd_width)]

plt.legend(legend)

plt.xlabel("SNR, dB")
plt.ylabel("Relative error, %")
plt.title("Channel Estimation error")
plt.grid(True)

plt.show()
