import numpy as np
from scipy import io


class TaskCfg:
    def __init__(self, packet_cnt, sc_cnt, student_id):
        self.packet_cnt = packet_cnt
        self.sc_cnt = sc_cnt
        self.student_id = student_id

        # all the calculations from matlab -1 (cause 0-indexation)
        self.sc_offset = self.student_id * 12
        self.time_offset = 2 * self.student_id - 1


# reads file and returns link_channel with borders given by t_cfg
# return: (link_channel, average path loss)
def load_link(t_cfg, file_name="Data/link_chan_1.mat"):
    # <UE antenna> x <BS antenna> x <sub-carrier> x <time>
    link_channel = io.loadmat(file_name)['Link_Channel']

    # take only necessary subcarriers and time packets to reduce further usage of offsets
    link_channel = link_channel[
                   :,
                   :,
                   t_cfg.sc_offset:(t_cfg.sc_offset + t_cfg.sc_cnt),
                   t_cfg.time_offset: t_cfg.time_offset + t_cfg.packet_cnt
                   ]

    # new axes: <time> x <sub-carrier> x <UE antenna> x <BS antenna>
    link_channel = np.moveaxis(link_channel, [3, 2], [0, 1])

    # normalization of the channel
    path_loss_avg = np.mean(np.abs(link_channel))
    link_channel = link_channel / (np.sqrt(2.) * path_loss_avg)

    return link_channel, path_loss_avg

