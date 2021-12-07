import numpy as np

'''
This file contains classes that implement spatial encoding/decoding

for each class in this file next interface is mandatory

method: encode
    implements encoding
    -from OFDM_symbols: array of OFDM symbols: <symbol_ind> - <sc_ind>
    -to:                array of shape (sc_cnt, bs_ant_cnt) - spatially encoded subcarriers

method: decode
    implements decoding
    -from y: array of shape (sc_cnt, ue_ant_cnt) - received spatial distribution for each carrier
    -to:     array of OFDM symbols: <symbol_ind> - <sc_ind>

method: symbols_cnt
    returns count of symbols encode/decode needs as input for transmission
'''


class Linear:
    def __init__(self, reduce_singular_values_border=0.0, enable_regularization=False):
        self.svd_border = reduce_singular_values_border
        self.en_reg = enable_regularization

    def label(self):
        label = "Linear("
        label += " reg={} ".format(self.en_reg)

        if self.svd_border > 0.0:
            label += " svd_border={:.1f} ".format(self.svd_border)

        label += ")"
        return label

    @staticmethod
    def symbols_cnt():
        return 1

    @staticmethod
    def encode(OFDM_symbols, link_u, link_s, link_vh, link):
        sc_cnt, ue_ant_cnt, bs_ant_cnt = (link_u.shape[0], link_u.shape[1], link_vh.shape[2])
        assert OFDM_symbols.shape == (1, sc_cnt)

        p = np.ones((bs_ant_cnt, sc_cnt)) / np.sqrt(bs_ant_cnt)

        # each subcarrier multiplied by (bs_ant_cnt) len column
        # gives us distribution over antennas
        return np.swapaxes(OFDM_symbols[0] * p, 0, 1)

    def decode(self, y, link_u, link_s, link_vh, link, std_n):
        sc_cnt, ue_ant_cnt, bs_ant_cnt = (link_u.shape[0], link_u.shape[1], link_vh.shape[2])

        result = []
        if self.en_reg:
            for sc_ind in range(sc_cnt):
                H = link[sc_ind]
                Hh = H.T.conj()
                inv = link_vh[sc_ind].T.conj() @ np.diag(np.sqrt(link_s[sc_ind]*link_s[sc_ind].conj()/(ue_ant_cnt*(std_n**2)))) @ link_vh[sc_ind] @ Hh
                #inv = link_vh[sc_ind].T.conj() @ np.diag(1/(link_s[sc_ind].conj()*link_s[sc_ind] + 1e4*(ue_ant_cnt**2)*bs_ant_cnt*(std_n**2))) @ link_vh[sc_ind] @ Hh
                #inv = np.linalg.pinv(Hh @ H + np.sqrt(ue_ant_cnt*bs_ant_cnt)*std_n * np.eye(H.shape[1]), ) @ Hh
                #inv = link_vh[sc_ind].T.conj() @ np.diag(1/(link_s[sc_ind].conj() * link_s[sc_ind])) @ link_vh[sc_ind] @ Hh
                result.append(np.sum(inv @ y[sc_ind]) / np.sqrt(bs_ant_cnt))

        else:  # no regularization
            for sc_ind in range(sc_cnt):
                link_inv = link_vh[sc_ind].conj().T @ \
                           np.diag(1/link_s[sc_ind]) @ \
                           link_u[sc_ind].conj().T

                result.append(np.sum(link_inv @ y[sc_ind])/np.sqrt(bs_ant_cnt))

        return np.array([result])


class SvdFilter:
    def __init__(self, vectors_use=np.array([0])):
        self.vectors_use = vectors_use

    def label(self):
        return "SvdFilter({})".format(self.vectors_use)

    def symbols_cnt(self):
        return len(self.vectors_use)

    def encode(self, OFDM_symbols, link_u, link_s, link_vh, link):
        sc_cnt, ue_ant_cnt, bs_ant_cnt = (link_u.shape[0], link_u.shape[1], link_vh.shape[2])
        assert OFDM_symbols.shape == (len(self.vectors_use), sc_cnt)

        # here could be attempt for shorter code
        result = np.zeros((sc_cnt, bs_ant_cnt), complex)
        for symbol_ind in range(OFDM_symbols.shape[0]):
            vec_ind = self.vectors_use[symbol_ind]
            for sc_ind in range(sc_cnt):
                p_sc = link_vh[sc_ind][vec_ind].conj()
                result[sc_ind] += p_sc * OFDM_symbols[symbol_ind][sc_ind]

        # normalization
        result /= np.sqrt(len(self.vectors_use))

        return result

    def decode(self, y, link_u, link_s, link_vh, link, std_n):
        sc_cnt, ue_ant_cnt, bs_ant_cnt = (link_u.shape[0], link_u.shape[1], link_vh.shape[2])

        result = []
        for vec_ind in self.vectors_use:
            channel_res = []

            for sc_ind in range(sc_cnt):
                p_sc = link_u[sc_ind][:, vec_ind].conj()
                channel_res.append(p_sc @ y[sc_ind])

            result.append(channel_res)

        return np.array(result)


class DftFilter:
    def __init__(self, dft_ticks=8):
        self.dft_ticks = dft_ticks

    def label(self):
        return "DftFilter(ticks={})".format(self.dft_ticks)

    def symbols_cnt(self):
        return 1

    def encode(self, OFDM_symbols, link_u, link_s, link_vh, link):
        sc_cnt, ue_ant_cnt, bs_ant_cnt = (link_u.shape[0], link_u.shape[1], link_vh.shape[2])
        assert OFDM_symbols.shape == (1, sc_cnt)

        result = np.zeros((sc_cnt, bs_ant_cnt), complex)
        for sc_ind in range(sc_cnt):
            h_fft = np.fft.fft(link[sc_ind])

            ind_hold = np.argpartition(np.abs(h_fft[0]), -self.dft_ticks)[-self.dft_ticks:]

            h0_fft_reduced = np.zeros((bs_ant_cnt, ), complex)
            h0_fft_reduced[ind_hold] = h_fft[0][ind_hold]

            p = np.fft.ifft(h0_fft_reduced).conj()
            p /= np.linalg.norm(p)

            result[sc_ind] = OFDM_symbols[0][sc_ind] * p

        return result

    def decode(self, y, link_u, link_s, link_vh, link, std_n):
        sc_cnt, ue_ant_cnt, bs_ant_cnt = (link_u.shape[0], link_u.shape[1], link_vh.shape[2])

        result = np.zeros((sc_cnt,), complex)
        for sc_ind in range(sc_cnt):
            h_fft = np.fft.fft(link[sc_ind])
            y_fft = np.fft.fft(y[sc_ind])

            ind_hold = np.argpartition(np.abs(h_fft[0]), -self.dft_ticks)[-self.dft_ticks:]

            p = np.sum(h_fft[:, ind_hold].conj() * h_fft[0][ind_hold], axis=1)

            result[sc_ind] = y[sc_ind].dot(p)

        return np.array([result])
