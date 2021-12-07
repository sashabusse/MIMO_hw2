import numpy as np
from mimo_lib.hard_detection import hard_detection
from mimo_lib.spatial_filters import Linear


# implements transmission simulation
#   -encoding
#   -channel propagation
#   -addition of AGWN
#   -decoding
#   -ber calculation
def simulate_transmission(
    t_cfg,
    link, link_u, link_s, link_vh,
    snr,
    tx_OFDM,
    sp_filter,
    use_snr_linear=False
):
    sc_cnt, ue_ant_cnt, bs_ant_cnt = (link_u.shape[1], link_u.shape[2], link_vh.shape[3])

    # complete sequence so that it is divisible by cnt of used beams
    if tx_OFDM.shape[0] % sp_filter.symbols_cnt() > 0:
        tx_OFDM = np.vstack((
            tx_OFDM,
            tx_OFDM[0:sp_filter.symbols_cnt() - (tx_OFDM.shape[0] % sp_filter.symbols_cnt())]
        ))

    ber = []
    # here simulating single transmission
    #   - single OFDM symbol
    #   - several OFDM symbols for multichannel encoding
    for cur_time in range(0, int(tx_OFDM.shape[0]/sp_filter.symbols_cnt())):
        st_symbol = cur_time * sp_filter.symbols_cnt()
        end_symbol = st_symbol + sp_filter.symbols_cnt()

        # <subcarrier> - <bs_ant>
        x = sp_filter.encode(
            tx_OFDM[st_symbol: end_symbol],
            link_u[cur_time], link_s[cur_time], link_vh[cur_time], link[cur_time]
        )

        y0 = np.zeros((t_cfg.sc_cnt, ue_ant_cnt), complex)
        for sc_ind in range(t_cfg.sc_cnt):
            y0[sc_ind] = link[cur_time, sc_ind] @ x[sc_ind]

        if use_snr_linear:
            lin_filter = Linear(enable_regularization=False)
            x_lin = lin_filter.encode(
                tx_OFDM[st_symbol: st_symbol + lin_filter.symbols_cnt()],
                link_u[cur_time], link_s[cur_time], link_vh[cur_time], link[cur_time]
            )

            y0_lin = np.zeros((t_cfg.sc_cnt, ue_ant_cnt), complex)
            for sc_ind in range(t_cfg.sc_cnt):
                y0_lin[sc_ind] = link[cur_time, sc_ind] @ x_lin[sc_ind]

        # estimate power of signal to generate noise for given SNR
        y_est = y0
        if use_snr_linear:
            y_est = y0_lin

        # Ps = np.mean(np.var(y_est, axis=1, ddof=1))  # incorrect
        Ps = np.zeros((sc_cnt, ))
        for sc_ind in range(t_cfg.sc_cnt):
            # .real to disable complex casting warning (y0.conj @ y0 is real for sure)
            Ps[sc_ind] = (y_est[sc_ind].conj() @ y_est[sc_ind]).real/len(y_est[sc_ind])

        Ps = np.mean(Ps)
        Dn = Ps / 10 ** (snr / 10)
        # adding complex gaussian noise with variation Dn and expectation mean|n0| = 1
        n0 = (
                np.random.normal(loc=0, scale=1, size=y0.shape) +
                1j * np.random.normal(loc=0, scale=1, size=y0.shape)
             ) * np.sqrt(Dn / 2)

        y = y0 + n0

        # If we have channel estimation H, we can use straight forward way MIMO equalization
        # as we sent single stream, then each receive antenna has to have similar result, and
        # to reduce noise impact, we just sum up all antennas assuming
        # COHERENT SUMMATION for SIGNAL, and AVERAGING for noise
        rx_OFDM = sp_filter.decode(y, link_u[cur_time], link_s[cur_time], link_vh[cur_time], link[cur_time], np.sqrt(Dn))

        for ofdm_ind in range(rx_OFDM.shape[0]):
            rx_OFDM[ofdm_ind] = hard_detection(rx_OFDM[ofdm_ind], "BPSK")

        ber.append(
            np.sum(
                np.abs((tx_OFDM[st_symbol: end_symbol] - rx_OFDM)/2)
            ) / (t_cfg.sc_cnt * sp_filter.symbols_cnt())
        )
    return np.mean(np.array(ber))


