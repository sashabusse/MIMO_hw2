import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from numba import jit

'''
This file contains routines to:
    -calculate
    -display
spatial spectrum
'''

# calculates 2d spatial spectrum
#   - arg (link): vector of len 32 that represents link (flattened 4 rows * 8 cols) (H estimation)
#   - arg (scale_mul=10): fft size scale multiplier to get more estimation points
#
#   - return: 2d spatial spectrum (2d fft over reshaped link vector)
@jit(nopython=True)
def calc_spatial_spectrum(link, scale_mul=10):
    link = np.reshape(
        link,
        (4, 8)
    )
    link_fft = np.fft.fft2(link, s=(link.shape[0] * scale_mul, link.shape[1] * scale_mul))
    return link_fft


# calculates spatial spectrum and x, y coordinates for surface
#   - arg (link): vector of len 32 that represents link (flattened 4 rows * 8 cols) (H estimation)
#   - arg (l1): distance between rows of bs antennas
#   - arg (l2): distance between cols of bs antennas
#   - arg (wave_len)
#
#   - return: arrays tet, phi, spec of shape (4, 8).
@jit(nopython=True)
def spatial_spectrum_surf(link, l1, l2, wave_len, scale_mul=100, centered=True):
    link_fft = calc_spatial_spectrum(link, scale_mul=scale_mul)

    if centered:
        link_fft = np.fft.fftshift(link_fft, axes=(0, 1))
        tet = np.fft.fftshift(np.fft.fftfreq(link_fft.shape[0]))
        phi = np.fft.fftshift(np.fft.fftfreq(link_fft.shape[1]))

    else:  # not centered
        tet = np.arange(link_fft.shape[0])/link_fft.shape[0]
        phi = np.arange(link_fft.shape[1])/link_fft.shape[1]

    tet = np.rad2deg(np.arcsin(tet * wave_len / l1))
    phi = np.rad2deg(np.arcsin(phi * wave_len / l2))

    phi, tet = np.meshgrid(phi, tet)

    return tet, phi, link_fft


# calculates link power spectrum to the given ue antenna
# (power spectrum is averaged over base antenna polarization)
#   - arg (link): !! vector of shape (64) describing bs antennas of both polarization
#   - arg (l1, l2, wave_len, scale_mul, centered): refer to the functions before
#
#   - return: tet, phi, rho. coordinates and power spectrum
@jit(nopython=True)
def spatial_power_spectrum_surf(link, l1, l2, wave_len, scale_mul=100, centered=True):
    tet, phi, spec0 = spatial_spectrum_surf(link[0:32], l1, l2, wave_len, scale_mul, centered)
    tet, phi, spec1 = spatial_spectrum_surf(link[32:64], l1, l2, wave_len, scale_mul, centered)

    return tet, phi, np.abs(0.5*spec0 + 0.5*spec1)**2


# draws 2d spectrum of the link
#   link.shape: <ue_ant> - <bs_ant>
def draw_surf(
        x, y, z,
        title="", x_label="tet", y_label='phi', z_label='power spectrum',
        fig_ax=None, c_bar=False, shade=False
):
    if fig_ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]

    ax.plot_surface(
        x,
        y,
        z,
        cmap='coolwarm', linewidth=0, antialiased=True, shade=shade
    )

    if c_bar:
        col_map = cm.ScalarMappable(cmap=cm.get_cmap('coolwarm'))
        col_map.set_array(z)
        fig.colorbar(col_map, fraction=0.03, aspect=30, pad=0.2)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
