import pandas as pd
import pdb
import numpy as np
import numpy.random as rand
import time
import os
import scipy.stats as stats
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from physt import h2 as physt_h2
import hashlib


def calc_epoch_T(epoch):
    if epoch == 'J2000':
        # Apparently from astropy
        theta   = 122.9319185680026 * np.pi/180.
        dec_ngp = 27.12825118085622 * np.pi/180.
        ra_ngp  = 192.8594812065348 * np.pi/180.

        T_theta = np.array([[np.cos(theta), np.sin(theta), 0.],
                            [np.sin(theta), -np.cos(theta), 0.],
                            [0.,0.,1.]])
        T_dec = np.array([[-np.sin(dec_ngp), 0, np.cos(dec_ngp)],
                        [0.,1.,0.],
                        [np.cos(dec_ngp), 0., np.sin(dec_ngp)]])

        T_ra = np.array([[np.cos(ra_ngp), np.sin(ra_ngp), 0.],
                        [-np.sin(ra_ngp), np.cos(ra_ngp), 0.],
                        [0.,0.,1.]])

        return np.dot(T_theta, np.dot(T_dec,T_ra))

    else:
        raise exception('Wrong epoch name')

def epoch_angles(epoch):
    if epoch == 'J2000':
        # Apparently from astropy
        theta   = 122.9319185680026 * np.pi/180.
        dec_ngp = 27.12825118085622 * np.pi/180.
        ra_ngp  = 192.8594812065348 * np.pi/180.
        return theta, dec_ngp, ra_ngp
    else:
        raise exception('Wrong epoch name')

def astrometric_to_galactocentric(ra, dec, para, pm_ra, pm_dec, vr, Rsun, phisun, Zsun, vRsun,vYsun,vZsun, epoch_T):
    """
    Frankensteined together from Jo Bovy's galpy code. Any errors most likely
    attributable to HS.
    Inputs:
      ra          rad     array
      dec         rad     array
      parallax    mas     array
      pm_ra       rad/s  array
      pm_dec      rad/s  array
      vr          km/s    array
      Rsun        pc
      phisun      rads (but zero by default)
      Zsun        pc
      vRsun       km/s
      vphisun     km/s
      vZsun       km/s
    """
    print_out = False
    #ra and dec to galactic latitude and longitude
    lb_one = np.array([np.cos(dec) * np.cos(ra),
                        np.cos(dec) * np.sin(ra),
                        np.sin(dec)])
    lb_two = np.dot(epoch_T, lb_one)

    lb_two[2][lb_two[2] > 1.]= 1.
    lb_two[2][lb_two[2] < -1.]= -1.

    b_vec = np.arcsin(lb_two[2])
    l_vec = np.arctan2(lb_two[1]/np.cos(b_vec),
                        lb_two[0]/np.cos(b_vec))
    l_vec[l_vec<0.]+= 2. * np.pi

    if print_out:
        print('l_vec: ', l_vec)
        print('b_vec: ', b_vec)
    # parallax to distance
    d_vec = 1000./para #pc
    if print_out:
        print('d_vec: ', d_vec)
        print('')

    #lbd_to_XYZ
    Xh_vec = d_vec * np.cos(b_vec) * np.cos(l_vec) # Positive towards GC
    Yh_vec = d_vec * np.cos(b_vec) * np.sin(l_vec)
    Zh_vec = d_vec * np.sin(b_vec) # NB: Heliocentric Z
    if print_out:
        print('Xh_vec: ', Xh_vec)
        print('Yh_vec: ', Yh_vec)
        print('Zh_vec: ', Zh_vec)
        print('')

    #XYZ_to_galcencyl
    dgc = np.sqrt(Rsun**2 + Zsun**2)
    h2g_rot_mat = np.array([[Rsun/dgc, 0., -Zsun/dgc],
                            [0.,1.,0.],
                            [Zsun/dgc, 0., Rsun/dgc]])
    Xg_vec, Yg_vec, Zg_vec = np.dot(h2g_rot_mat,
                                np.array([-Xh_vec + dgc,
                                            Yh_vec,
                                            np.sign(Rsun)*Zh_vec]))

    if print_out:
        print('Xg_vec: ', Xg_vec)
        print('Yg_vec: ', Yg_vec)
        print('Zg_vec: ', Zg_vec)
        print('')

    Rg_vec = np.sqrt(Xg_vec**2 + Yg_vec**2)
    phig_vec = np.arctan2(Yg_vec, Xg_vec)

    if print_out:
        print('Rg_vec: ', Rg_vec)
        print('phig_vec: ', phig_vec)
        print('Zg_vec: ', Zg_vec)
        print('')

    #pmrapmdec_to_pmllpmbb
    theta, dec_ngp, ra_ngp = epoch_angles('J2000')
    dec[dec == dec_ngp]+= 10.**-16 #deal w/ pole JB galpy

    cosphi = (np.sin(dec_ngp) * np.cos(dec) - np.cos(dec_ngp) * np.sin(dec) * np.cos(ra-ra_ngp))
    sinphi = (np.sin(ra-ra_ngp) * np.cos(dec_ngp))
    norm= np.sqrt(cosphi**2.+sinphi**2.)
    cosphi /= norm
    sinphi /= norm

    pm_ll_vec_cosb, pm_bb_vec =  (np.array([[cosphi,-sinphi],[sinphi,cosphi]]).T *np.array([[pm_ra,pm_ra],[pm_dec,pm_dec]]).T).sum(-1).T
    pm_ll_vec =  pm_ll_vec_cosb/np.cos(b_vec)

    if print_out:
        print('pm_ll_vec_cosb: ', pm_ll_vec_cosb)
        print('pm_bb_vec:', pm_bb_vec)
        print('')

    #vrpmllpmbb_to_vxvyvz
    Rmat=np.zeros((3,3,len(l_vec)))
    Rmat[0,0]= np.cos(l_vec)*np.cos(b_vec)
    Rmat[1,0]= -np.sin(l_vec)
    Rmat[2,0]= -np.cos(l_vec)*np.sin(b_vec)
    Rmat[0,1]= np.sin(l_vec)*np.cos(b_vec)
    Rmat[1,1]= np.cos(l_vec)
    Rmat[2,1]= -np.sin(l_vec)*np.sin(b_vec)
    Rmat[0,2]= np.sin(b_vec)
    Rmat[2,2]= np.cos(b_vec)

    pc2km = 3.08567758149137E13
    invr_mat = np.array([[vr,vr,vr],
                    [d_vec * pm_ll_vec_cosb * pc2km, d_vec * pm_ll_vec_cosb * pc2km, d_vec * pm_ll_vec_cosb * pc2km],
                    [d_vec * pm_bb_vec * pc2km, d_vec * pm_bb_vec * pc2km, d_vec * pm_bb_vec * pc2km]])

    vXh_vec, vYh_vec, vZh_vec = (Rmat.T * invr_mat.T).sum(-1).T

    if print_out:
        print('Velocities xyz Heliocentric')
        print('vxh: ', vxh)
        print('vyh: ', vyh)
        print('vzh: ', vzh)
        print('')

    #vxvyvz_to_galcencyl

    vXg_vec, vYg_vec, vZg_vec = np.dot(h2g_rot_mat,np.array([-vXh_vec, vYh_vec,np.sign(Rsun) * vZh_vec]))\
                                + np.array([vRsun,vYsun,vZsun]).reshape(3,1)

    if print_out:
        print('vXg_vec: ', vXg_vec)
        print('vYg_vec: ', vYg_vec)
        print('vZg_vec: ', vZg_vec)
        print('')
    # vtg vec is off
    vRg_vec = vXg_vec * np.cos(phig_vec) + vYg_vec * np.sin(phig_vec)
    vTg_vec = -vXg_vec * np.sin(phig_vec) + vYg_vec * np.cos(phig_vec)

    if print_out:
        print('vRg_vec:', vRg_vec)
        print('vTg_vec:', vTg_vec)
        print('vZg_vec: ', vZg_vec)
        print('')

    return Rg_vec, phig_vec, Zg_vec, vRg_vec, vTg_vec, vZg_vec

def binning(Rg_vec, phig_vec, Zg_vec, vRg_vec, vTg_vec, vZg_vec, phi_limits, R_edges, Z_edges):
    """
    Rg_vec, star_data_gccyl[0]
    phig_vec, star_data_gccyl[1]
    Zg_vec, star_data_gccyl[2]

    vRg_vec, star_V_gccyl[0]
    vphig_vec, star_V_gccyl[1]
    vZg_vec, star_V_gccyl[2]
    """
    vphig_vec = vTg_vec/Rg_vec #rad/s
    counts_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            Rg_vec, #dummy array for count
                                            statistic='count',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]

    #print('Counts done')
    # AVERAGE VELOCITIES
    vbar_R1_dat_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vRg_vec,
                                            statistic='mean',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    vbar_R1_std_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vRg_vec,
                                            statistic=np.std,
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    #print('vbar_x1 done')

    vbar_p1_dat_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vphig_vec,
                                            statistic='mean',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    vbar_p1_std_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vphig_vec,
                                            statistic=np.std,
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    #print('vbar_p1 done')

    vbar_Z1_dat_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vZg_vec,
                                            statistic='mean',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    vbar_Z1_std_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vZg_vec,
                                            statistic=np.std,
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    #print('vbar_z1 done')

    #AVERAGE DOUBLE VELOCITIES
    vbar_RR_dat_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vRg_vec**2,
                                            statistic='mean',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    vbar_RR_std_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vRg_vec**2,
                                            statistic=np.std,
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    #print('vbar_xx done')

    vbar_pp_dat_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vphig_vec**2,
                                            statistic='mean',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    vbar_pp_std_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vphig_vec**2,
                                            statistic=np.std,
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    #print('vbar_pp done')

    vbar_ZZ_dat_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vZg_vec**2,
                                            statistic='mean',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    vbar_ZZ_std_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vZg_vec**2,
                                            statistic=np.std,
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    #print('vbar_zz done')

    vbar_RZ_dat_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vRg_vec*vZg_vec,
                                            statistic='mean',
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]
    vbar_RZ_std_grid = stats.binned_statistic_dd([phig_vec,Rg_vec,Zg_vec],
                                            vRg_vec*vZg_vec,
                                            statistic=np.std,
                                            bins=[phi_limits,
                                            R_edges, Z_edges])[0][0]

    return np.array([counts_grid,
            vbar_R1_dat_grid, vbar_p1_dat_grid, vbar_Z1_dat_grid,
            vbar_RR_dat_grid, vbar_pp_dat_grid, vbar_ZZ_dat_grid,
            vbar_RZ_dat_grid]),\
            np.array([vbar_R1_std_grid,vbar_p1_std_grid,vbar_Z1_std_grid,
            vbar_RR_std_grid,vbar_pp_std_grid,vbar_ZZ_std_grid,
            vbar_RZ_std_grid])


def sample_transform_bin(astrometric_means, astrometric_covariances,
                            cholesky_astrometric_covariances,
                            solar_pomo_means, solar_pomo_covariances,
                            epoch_T, seed,
                            phi_limits, R_edges, Z_edges):
    """
    #https://stackoverflow.com/questions/14920272/generate-a-data-set-consisting-of-n-100-2-dimensional-samples
    """
    rand.seed(int(seed + int(time.time())%10000+1))
    # stars_sample = np.array([rand.multivariate_normal(astrometric_means[ii],
    #                     astrometric_covariances[ii]) for ii in range(Nstars)])

    #Cholesky Decomposition Method
    Nstars = len(astrometric_means)
    uncorrelated_sample = np.random.standard_normal((Nstars,6))
    stars_sample = np.array([np.dot(cholesky_astrometric_covariances[ii],\
                        uncorrelated_sample[ii]) + astrometric_means[ii] for ii in range(Nstars)])

    solar_pomo_sample = rand.multivariate_normal(solar_pomo_means, solar_pomo_covariances)

    Rg_vec, phig_vec, Zg_vec, vRg_vec, vTg_vec, vZg_vec = astrometric_to_galactocentric(
                    stars_sample[:,0], stars_sample[:,1], stars_sample[:,2],
                    stars_sample[:,3], stars_sample[:,4], stars_sample[:,5],
                    solar_pomo_sample[0], solar_pomo_sample[1], solar_pomo_sample[2],
                    solar_pomo_sample[3], solar_pomo_sample[4], solar_pomo_sample[5],
                    epoch_T)
    binned_data_vector, binned_std_vector = binning(Rg_vec, phig_vec, Zg_vec,
                                vRg_vec, vTg_vec, vZg_vec,
                                phi_limits, R_edges, Z_edges)

    return binned_data_vector.flatten(), binned_std_vector.flatten()

def plot_RZ_heatmap(R_data_coords_mesh, Z_data_coords_mesh, data_grid,
                    file_name,
                    fig_height = 9, fig_width = 13, colormap = 'magma',
                    lognorm = False, vmin=None, vmax=None,
                    ylabel = 'Z [pc]', xlabel = 'R [pc]',
                    cb_label = ' '):
    fig, axes = plt.subplots(ncols=2, nrows=1, gridspec_kw={"width_ratios":[15,1]})
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    #plt.subplots_adjust(wspace=wspace_double_cbax)
    ax = axes[0] #Plot
    cbax = axes[1] #Colorbar
    if lognorm:
        im = ax.pcolormesh(R_data_coords_mesh, Z_data_coords_mesh, data_grid,
                        cmap = colormap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax.pcolormesh(R_data_coords_mesh, Z_data_coords_mesh, data_grid,
                        cmap = colormap, vmin=vmin, vmax=vmax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    cb = fig.colorbar(im, cax=cbax)
    cb.set_label(label=cb_label)
    plt.savefig(file_name)
    return

def plot_matrix_heatmap(matrix, out_file_name,
                fig_height = 9, fig_width = 13, colormap = 'magma',
                cb_label = 'Correlation'):

    fig, axes = plt.subplots(ncols=2, nrows=1, gridspec_kw={"width_ratios":[15,1]})
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    #plt.subplots_adjust(wspace=wspace_double_cbax)
    ax = axes[0] #Plot
    cbax = axes[1] #Colorbar

    im = ax.pcolormesh(matrix, cmap = colormap)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    cb = fig.colorbar(im, cax=cbax)
    cb.set_label(label=cb_label)

    plt.savefig(out_file_name)



##########################################################################

class oscar_gaia_data:
    """
    Performs sample-bin-repeat on astrometric Gaia data
    2019-01     Hamish Silverwood, basic machinery
    2019-02     Turned into class object
    """
    def __init__(self, data_root = '../Astrometric_Data/Gaia_DR2_subsamples/',
                        data_file_name = 'GaiaDR2_RC_sample_Mcut_0p0_0p75_Ccut_1p0_1p5Nstars_1333998.csv',
                        binning_type = 'quartile', #linear #input
                        Rmin = 6000, Rmax = 10000, num_R_bins = 10,
                        Zmin = -2000, Zmax = 2000, num_Z_bins = 10,
                        input_R_edges = None, input_Z_edges = None,
                        phi_limits = [-np.pi/8,np.pi/8],
                        N_samplings = 100,
                        N_cores = 1,
                        ):
        self.data_root = data_root
        self.data_file_name = data_file_name
        self.binning_type = binning_type
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.num_R_bins = num_R_bins
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.num_Z_bins = num_Z_bins
        self.input_R_edges = input_R_edges
        self.input_Z_edges = input_Z_edges
        self.phi_limits = phi_limits
        self.N_samplings = N_samplings
        self.N_cores = N_cores


        # Set Constants and Parameters
        deg_to_rad = np.pi/180
        mas_to_rad = (np.pi/6.48E8)
        maspyr_to_radps = np.pi/(6.48E8 * 31557600)

        # Solar Position and Motion model
        self.solar_pomo_means = np.array([8200.,0.,100., 14.,238.,5.])
        self.solar_pomo_covariances = np.identity(6) * self.solar_pomo_means * 0.1 #10% errors

        # Open data file
        datab = pd.read_csv(self.data_root + self.data_file_name) #astrometric_data_table

        # Construct Means and Covarriance Matrices
        astrometric_means = np.array([datab['ra'].values * deg_to_rad, #rad
                                datab['dec'].values * deg_to_rad, #rad
                                datab['parallax'].values, #mas
                                datab['pmra'].values * maspyr_to_radps, #rad/s
                                datab['pmdec'].values * maspyr_to_radps, #rad/s
                                datab['radial_velocity'].values]).T #km/s

        Nstars = datab['ra'].values.shape[0]
        Nzeros = np.zeros(Nstars)
        astrometric_covariances = np.array([[(datab['ra_error'].values*mas_to_rad)**2,
            datab['ra_dec_corr'].values * datab['ra_error'].values * datab['dec_error'].values * mas_to_rad**2,
            datab['ra_parallax_corr'].values * datab['ra_error'].values * datab['parallax_error'].values * mas_to_rad,
            datab['ra_pmra_corr'].values * datab['ra_error'].values * datab['pmra_error'].values * mas_to_rad * maspyr_to_radps,
            datab['ra_pmdec_corr'].values * datab['ra_error'].values * datab['pmdec_error'].values * mas_to_rad * maspyr_to_radps,
            Nzeros],
            [Nzeros, (datab['dec_error'].values*mas_to_rad)**2,
            datab['dec_parallax_corr'].values * datab['dec_error'].values * datab['parallax_error'].values * mas_to_rad,
            datab['dec_pmra_corr'].values * datab['dec_error'].values * datab['pmra_error'].values * mas_to_rad * maspyr_to_radps,
            datab['dec_pmdec_corr'].values * datab['dec_error'].values * datab['pmdec_error'].values * mas_to_rad * maspyr_to_radps,
            Nzeros],
            [Nzeros, Nzeros, datab['parallax_error'].values**2,
            datab['parallax_pmra_corr'].values * datab['parallax_error'].values * datab['pmra_error'].values * maspyr_to_radps,
            datab['parallax_pmdec_corr'].values * datab['parallax_error'].values * datab['pmdec_error'].values * maspyr_to_radps,
            Nzeros],
            [Nzeros,Nzeros,Nzeros, (datab['pmra_error'].values * maspyr_to_radps)**2,
            datab['pmra_pmdec_corr'].values * datab['pmra_error'].values * datab['pmdec_error'].values * maspyr_to_radps**2,
            Nzeros],
            [Nzeros, Nzeros, Nzeros, Nzeros, (datab['pmdec_error'].values * maspyr_to_radps)**2, Nzeros],
            [Nzeros, Nzeros, Nzeros, Nzeros, Nzeros, datab['radial_velocity_error'].values**2]])

        astrometric_covariances = np.transpose(astrometric_covariances, (2,0,1)) #Rearrange
        astrometric_covariances = np.array([astrometric_covariances[ii] + astrometric_covariances[ii].T - \
                                        np.diagonal(astrometric_covariances[ii])*np.identity(6) \
                                        for ii in range(Nstars)]) #Symmetrize
        cholesky_astrometric_covariances = np.linalg.cholesky(astrometric_covariances)

        #Calculate epoch_T matrix
        epoch_T = calc_epoch_T('J2000')

        # Determine Binning
        if binning_type == 'input':
            self.R_edges = self.input_R_edges
            self.Z_edges = self.input_Z_edges
        elif binning_type == 'linear':
            self.R_edges = np.linspace(self.Rmin, self.Rmax, self.num_R_bins)
            self.Z_edges = np.linspace(self.Zmin, self.Zmax, self.num_Z_bins)
        elif binning_type == 'quartile':
            galactocentric_means = astrometric_to_galactocentric(
                    astrometric_means[:,0], astrometric_means[:,1],
                    astrometric_means[:,2], astrometric_means[:,3],
                    astrometric_means[:,4], astrometric_means[:,5],
                    self.solar_pomo_means[0], self.solar_pomo_means[1],
                    self.solar_pomo_means[2], self.solar_pomo_means[3],
                    self.solar_pomo_means[4], self.solar_pomo_means[5],
                    epoch_T)
            phi_cut_locs = np.where((galactocentric_means[1]>self.phi_limits[0]) & (galactocentric_means[1]<self.phi_limits[1]))
            Rg_vec_means = galactocentric_means[0][phi_cut_locs]
            Zg_vec_means = galactocentric_means[2][phi_cut_locs]

            physt_hist = physt_h2(Rg_vec_means, Zg_vec_means, "quantile",
                        (self.num_R_bins,self.num_Z_bins))
            self.R_edges = physt_hist.numpy_bins[0][1:-1]
            self.Z_edges = physt_hist.numpy_bins[1][1:-1]

        # Calculate bin centers and volumes
        self.R_bin_centers = (self.R_edges[1:] + self.R_edges[:-1])/2
        self.Z_bin_centers = (self.Z_edges[1:] + self.Z_edges[:-1])/2
        self.R_data_coords_mesh, self.Z_data_coords_mesh = np.meshgrid(self.R_bin_centers, self.Z_bin_centers, indexing='ij')

        self.bin_vol_grid= np.zeros([len(self.R_edges) - 1, len(self.Z_edges)-1])
        for (aa,bb), dummy in np.ndenumerate(self.bin_vol_grid):
            self.bin_vol_grid[aa,bb] = 0.5 * abs(self.phi_limits[1]-self.phi_limits[0])\
                            * abs(self.R_edges[aa+1]**2 - self.R_edges[aa]**2)\
                            * abs(self.Z_edges[bb+1] - self.Z_edges[bb])

        # Build cache file name
        if not os.path.isdir(data_root + '/oscar_cache_files/'):
            os.mkdir(data_root + '/oscar_cache_files/')

        cache_file_name = 'oscar_cache_' + hashlib.md5(self.R_edges).hexdigest()\
                            + hashlib.md5(self.Z_edges).hexdigest()\
                            + hashlib.md5(self.solar_pomo_means).hexdigest()\
                            + hashlib.md5(self.solar_pomo_covariances).hexdigest()\
                            + str(self.N_samplings)\
                            + data_file_name.split('.')[0] + '.dat'

        # Search for cache file
        if os.path.isfile(data_root + '/oscar_cache_files/' + cache_file_name):
            print('Previous sampling found, pulling data from cache.')
            cache_dataframe = pd.read_pickle(data_root + '/oscar_cache_files/' + cache_file_name)

            self.data_mean = cache_dataframe['data_mean']
            self.data_sigma2 = cache_dataframe['data_sigma2']
            self.data_cov = cache_dataframe['data_cov']
            self.data_corr = cache_dataframe['data_corr']
            self.data_mean_grids = cache_dataframe['data_mean_grids']
            self.sigma_meas_grids = cache_dataframe['sigma_meas_grids']
            self.skewness_stat_grids = cache_dataframe['skewness_stat_grids']
            self.skewness_pval_grids = cache_dataframe['skewness_pval_grids']
            self.kurtosis_stat_grids = cache_dataframe['kurtosis_stat_grids']
            self.kurtosis_pval_grids = cache_dataframe['kurtosis_pval_grids']
            self.gaussianity_stat_grids = cache_dataframe['gaussianity_stat_grids']
            self.gaussianity_pval_grids = cache_dataframe['gaussianity_pval_grids']
            self.R_data_coords_mesh = cache_dataframe['R_data_coords_mesh']
            self.Z_data_coords_mesh = cache_dataframe['Z_data_coords_mesh']
            self.counts_grid = cache_dataframe['counts_grid']
            self.nu_dat_grid = cache_dataframe['nu_dat_grid']
            self.vbar_R1_dat_grid = cache_dataframe['vbar_R1_dat_grid']
            self.vbar_p1_dat_grid = cache_dataframe['vbar_p1_dat_grid']
            self.vbar_Z1_dat_grid = cache_dataframe['vbar_Z1_dat_grid']
            self.vbar_RR_dat_grid = cache_dataframe['vbar_RR_dat_grid']
            self.vbar_pp_dat_grid = cache_dataframe['vbar_pp_dat_grid']
            self.vbar_ZZ_dat_grid = cache_dataframe['vbar_ZZ_dat_grid']
            self.vbar_RZ_dat_grid = cache_dataframe['vbar_RZ_dat_grid']



        else:
            print('No previous sampling found, running from scratch')

            if N_cores == 1:
                #Linear Sample Transform Bin
                all_binned_data_vectors = []
                start = time.time()
                for jj in range(N_samplings):
                    print('Sample ', jj, ' of ', N_samplings)
                    binned_data_vector, binned_std_vector = sample_transform_bin(
                                        astrometric_means, astrometric_covariances,
                                        cholesky_astrometric_covariances,
                                        self.solar_pomo_means, self.solar_pomo_covariances,
                                        epoch_T,jj,
                                        self.phi_limits, self.R_edges, self.Z_edges)
                    all_binned_data_vectors.append(binned_data_vector)
                all_binned_data_vectors = np.array(all_binned_data_vectors)
                print('\nLinear Sampling, Transforming, Binning takes ', time.time()-start, ' s')
                print('Time per sample: ', (time.time()-start)/N_samplings, ' s\n')

            else:
                #Multiprocessor Pool
                print('Starting Parallel Sampling')
                start = time.time()
                pool = mp.Pool(processes=self.N_cores)
                results = [pool.apply_async(sample_transform_bin,
                                        args = (astrometric_means, astrometric_covariances,
                                                cholesky_astrometric_covariances,
                                                self.solar_pomo_means, self.solar_pomo_covariances,
                                                epoch_T, seed,
                                                self.phi_limits, self.R_edges, self.Z_edges)) for seed in range(N_samplings)]

                output = [p.get() for p in results]
                all_binned_data_vectors = np.array([output[ii][0] for ii in range(N_samplings)])
                all_binned_std_vectors = np.array([output[ii][1] for ii in range(N_samplings)])
                end = time.time()
                print('Parallel Sampling, Transforming, Binning takes ', end-start, ' s')
                print('Wall time per sample: ', (end-start)/N_samplings)

            #Calculate means and covariances, Skewness, Kurtosis

            self.data_mean = np.mean(all_binned_data_vectors, axis=0)
            self.data_cov  = np.cov(all_binned_data_vectors.T)
            self.data_corr = np.corrcoef(all_binned_data_vectors.T)
            self.data_sigma2 = np.diag(self.data_cov)

            #Gaussianity test using D’Agostino and Pearson’s tests
            self.skewness_stat, self.skewness_pval = stats.skewtest(all_binned_data_vectors)
            self.kurtosis_stat, self.kurtosis_pval = stats.kurtosistest(all_binned_data_vectors)
            self.gaussianity_stat, self.gaussianity_pval = stats.normaltest(all_binned_data_vectors)

            #Reformat into individual quantities
            grid_shape = (8, len(self.R_edges)-1, len(self.Z_edges)-1)

            # Reshape
            self.data_mean_grids = self.data_mean.reshape(grid_shape)
            self.sigma_meas_grids = np.sqrt(self.data_sigma2).reshape(grid_shape)
            self.skewness_stat_grids = self.skewness_stat.reshape(grid_shape)
            self.skewness_pval_grids = self.skewness_pval.reshape(grid_shape)
            self.kurtosis_stat_grids = self.kurtosis_stat.reshape(grid_shape)
            self.kurtosis_pval_grids = self.kurtosis_pval.reshape(grid_shape)
            self.gaussianity_stat_grids = self.gaussianity_stat.reshape(grid_shape)
            self.gaussianity_pval_grids = self.gaussianity_pval.reshape(grid_shape)

            # Pull out means and errors
            self.counts_grid,\
            self.vbar_R1_dat_grid, self.vbar_p1_dat_grid, self.vbar_Z1_dat_grid,\
            self.vbar_RR_dat_grid, self.vbar_pp_dat_grid, self.vbar_ZZ_dat_grid,\
            self.vbar_RZ_dat_grid = self.data_mean_grids

            # vbar_R1_std_grid, vbar_p1_std_grid, vbar_Z1_std_grid,\
            # vbar_RR_std_grid, vbar_pp_std_grid, vbar_ZZ_std_grid,\
            # vbar_RZ_std_grid = self.data_std_grids #FIGURE OUT ERRORS

            # Calculate tracer density
            sigma_pois_counts_grid = np.sqrt(self.counts_grid)
            sigma_meas_counts_grid = 0.*sigma_pois_counts_grid #BODGE
            sigma_total_counts_grid = np.sqrt(sigma_pois_counts_grid**2 + sigma_meas_counts_grid**2)
            self.nu_dat_grid = self.counts_grid/self.bin_vol_grid
            self.nu_err_grid = sigma_total_counts_grid/self.bin_vol_grid

            # Build dictionary then save to dataframe
            dictionary = {'data_mean' : self.data_mean,
                            'data_sigma2': self.data_sigma2,
                            'data_cov': self.data_cov,
                            'data_corr' : self.data_corr,
                            'data_mean_grids' : self.data_mean_grids,
                            'sigma_meas_grids' : self.sigma_meas_grids,
                            'skewness_stat_grids' : self.skewness_stat_grids,
                            'skewness_pval_grids' : self.skewness_pval_grids,
                            'kurtosis_stat_grids' : self.kurtosis_stat_grids,
                            'kurtosis_pval_grids' : self.kurtosis_pval_grids,
                            'gaussianity_stat_grids' : self.gaussianity_stat_grids,
                            'gaussianity_pval_grids' : self.gaussianity_pval_grids,
                            'R_data_coords_mesh' : self.R_data_coords_mesh,
                            'Z_data_coords_mesh' : self.Z_data_coords_mesh,
                            'counts_grid' : self.counts_grid,
                            'nu_dat_grid' : self.nu_dat_grid,
                            'vbar_R1_dat_grid' : self.vbar_R1_dat_grid,
                            'vbar_p1_dat_grid' : self.vbar_p1_dat_grid,
                            'vbar_Z1_dat_grid' : self.vbar_Z1_dat_grid,
                            'vbar_RR_dat_grid' : self.vbar_RR_dat_grid,
                            'vbar_pp_dat_grid' : self.vbar_pp_dat_grid,
                            'vbar_ZZ_dat_grid' : self.vbar_ZZ_dat_grid,
                            'vbar_RZ_dat_grid' : self.vbar_RZ_dat_grid,
                            }

            cache_dataframe = pd.Series(dictionary)
            cache_dataframe.to_pickle(data_root + '/oscar_cache_files/' + cache_file_name)

    def plot_histograms(self):
        # PLOT RESULTS


        sigma_meas_counts_grid,\
        sigma_meas_vbar_R1_dat_grid, sigma_meas_vbar_p1_dat_grid,\
        sigma_meas_vbar_Z1_dat_grid, sigma_meas_vbar_RR_dat_grid,\
        sigma_meas_vbar_pp_dat_grid, sigma_meas_vbar_ZZ_dat_grid,\
        sigma_meas_vbar_RZ_dat_grid = self.sigma_meas_grids

        # sigma_meas_counts_grid,\
        # sigma_meas_vbar_R1_dat_grid, sigma_meas_vbar_R1_std_grid,\
        # sigma_meas_vbar_p1_dat_grid, sigma_meas_vbar_p1_std_grid,\
        # sigma_meas_vbar_Z1_dat_grid, sigma_meas_vbar_Z1_std_grid,\
        # sigma_meas_vbar_RR_dat_grid, sigma_meas_vbar_RR_std_grid,\
        # sigma_meas_vbar_pp_dat_grid, sigma_meas_vbar_pp_std_grid,\
        # sigma_meas_vbar_ZZ_dat_grid, sigma_meas_vbar_ZZ_std_grid,\
        # sigma_meas_vbar_RZ_dat_grid, sigma_meas_vbar_RZ_std_grid = self.sigma_meas_grids

        skewness_stat_counts_grid,\
        skewness_stat_vbar_R1_dat_grid, skewness_stat_vbar_p1_dat_grid,\
        skewness_stat_vbar_Z1_dat_grid, skewness_stat_vbar_RR_dat_grid,\
        skewness_stat_vbar_pp_dat_grid, skewness_stat_vbar_ZZ_dat_grid,\
        skewness_stat_vbar_RZ_dat_grid = self.skewness_stat_grids

        # skewness_stat_counts_grid,\
        # skewness_stat_vbar_R1_dat_grid, skewness_stat_vbar_R1_std_grid,\
        # skewness_stat_vbar_p1_dat_grid, skewness_stat_vbar_p1_std_grid,\
        # skewness_stat_vbar_Z1_dat_grid, skewness_stat_vbar_Z1_std_grid,\
        # skewness_stat_vbar_RR_dat_grid, skewness_stat_vbar_RR_std_grid,\
        # skewness_stat_vbar_pp_dat_grid, skewness_stat_vbar_pp_std_grid,\
        # skewness_stat_vbar_ZZ_dat_grid, skewness_stat_vbar_ZZ_std_grid,\
        # skewness_stat_vbar_RZ_dat_grid, skewness_stat_vbar_RZ_std_grid = self.skewness_stat_grids

        skewness_pval_counts_grid,\
        skewness_pval_vbar_R1_dat_grid, skewness_pval_vbar_p1_dat_grid,\
        skewness_pval_vbar_Z1_dat_grid, skewness_pval_vbar_RR_dat_grid,\
        skewness_pval_vbar_pp_dat_grid, skewness_pval_vbar_ZZ_dat_grid,\
        skewness_pval_vbar_RZ_dat_grid = self.skewness_pval_grids

        # skewness_pval_counts_grid,\
        # skewness_pval_vbar_R1_dat_grid, skewness_pval_vbar_R1_std_grid,\
        # skewness_pval_vbar_p1_dat_grid, skewness_pval_vbar_p1_std_grid,\
        # skewness_pval_vbar_Z1_dat_grid, skewness_pval_vbar_Z1_std_grid,\
        # skewness_pval_vbar_RR_dat_grid, skewness_pval_vbar_RR_std_grid,\
        # skewness_pval_vbar_pp_dat_grid, skewness_pval_vbar_pp_std_grid,\
        # skewness_pval_vbar_ZZ_dat_grid, skewness_pval_vbar_ZZ_std_grid,\
        # skewness_pval_vbar_RZ_dat_grid, skewness_pval_vbar_RZ_std_grid = self.skewness_pval_grids

        kurtosis_stat_counts_grid,\
        kurtosis_stat_vbar_R1_dat_grid, kurtosis_stat_vbar_p1_dat_grid,\
        kurtosis_stat_vbar_Z1_dat_grid, kurtosis_stat_vbar_RR_dat_grid,\
        kurtosis_stat_vbar_pp_dat_grid, kurtosis_stat_vbar_ZZ_dat_grid,\
        kurtosis_stat_vbar_RZ_dat_grid = self.kurtosis_stat_grids

        # kurtosis_stat_counts_grid,\
        # kurtosis_stat_vbar_R1_dat_grid, kurtosis_stat_vbar_R1_std_grid,\
        # kurtosis_stat_vbar_p1_dat_grid, kurtosis_stat_vbar_p1_std_grid,\
        # kurtosis_stat_vbar_Z1_dat_grid, kurtosis_stat_vbar_Z1_std_grid,\
        # kurtosis_stat_vbar_RR_dat_grid, kurtosis_stat_vbar_RR_std_grid,\
        # kurtosis_stat_vbar_pp_dat_grid, kurtosis_stat_vbar_pp_std_grid,\
        # kurtosis_stat_vbar_ZZ_dat_grid, kurtosis_stat_vbar_ZZ_std_grid,\
        # kurtosis_stat_vbar_RZ_dat_grid, kurtosis_stat_vbar_RZ_std_grid = self.kurtosis_stat_grids

        kurtosis_pval_counts_grid,\
        kurtosis_pval_vbar_R1_dat_grid, kurtosis_pval_vbar_p1_dat_grid,\
        kurtosis_pval_vbar_Z1_dat_grid, kurtosis_pval_vbar_RR_dat_grid,\
        kurtosis_pval_vbar_pp_dat_grid, kurtosis_pval_vbar_ZZ_dat_grid,\
        kurtosis_pval_vbar_RZ_dat_grid = self.kurtosis_pval_grids

        # kurtosis_pval_counts_grid,\
        # kurtosis_pval_vbar_R1_dat_grid, kurtosis_pval_vbar_R1_std_grid,\
        # kurtosis_pval_vbar_p1_dat_grid, kurtosis_pval_vbar_p1_std_grid,\
        # kurtosis_pval_vbar_Z1_dat_grid, kurtosis_pval_vbar_Z1_std_grid,\
        # kurtosis_pval_vbar_RR_dat_grid, kurtosis_pval_vbar_RR_std_grid,\
        # kurtosis_pval_vbar_pp_dat_grid, kurtosis_pval_vbar_pp_std_grid,\
        # kurtosis_pval_vbar_ZZ_dat_grid, kurtosis_pval_vbar_ZZ_std_grid,\
        # kurtosis_pval_vbar_RZ_dat_grid, kurtosis_pval_vbar_RZ_std_grid = self.kurtosis_pval_grids

        gaussianity_stat_counts_grid,\
        gaussianity_stat_vbar_R1_dat_grid, gaussianity_stat_vbar_p1_dat_grid,\
        gaussianity_stat_vbar_Z1_dat_grid, gaussianity_stat_vbar_RR_dat_grid,\
        gaussianity_stat_vbar_pp_dat_grid, gaussianity_stat_vbar_ZZ_dat_grid,\
        gaussianity_stat_vbar_RZ_dat_grid = self.gaussianity_stat_grids

        # gaussianity_stat_counts_grid,\
        # gaussianity_stat_vbar_R1_dat_grid, gaussianity_stat_vbar_R1_std_grid,\
        # gaussianity_stat_vbar_p1_dat_grid, gaussianity_stat_vbar_p1_std_grid,\
        # gaussianity_stat_vbar_Z1_dat_grid, gaussianity_stat_vbar_Z1_std_grid,\
        # gaussianity_stat_vbar_RR_dat_grid, gaussianity_stat_vbar_RR_std_grid,\
        # gaussianity_stat_vbar_pp_dat_grid, gaussianity_stat_vbar_pp_std_grid,\
        # gaussianity_stat_vbar_ZZ_dat_grid, gaussianity_stat_vbar_ZZ_std_grid,\
        # gaussianity_stat_vbar_RZ_dat_grid, gaussianity_stat_vbar_RZ_std_grid = self.gaussianity_stat_grids

        gaussianity_pval_counts_grid,\
        gaussianity_pval_vbar_R1_dat_grid, gaussianity_pval_vbar_p1_dat_grid,\
        gaussianity_pval_vbar_Z1_dat_grid, gaussianity_pval_vbar_RR_dat_grid,\
        gaussianity_pval_vbar_pp_dat_grid, gaussianity_pval_vbar_ZZ_dat_grid,\
        gaussianity_pval_vbar_RZ_dat_grid = self.gaussianity_pval_grids

        # gaussianity_pval_counts_grid,\
        # gaussianity_pval_vbar_R1_dat_grid, gaussianity_pval_vbar_R1_std_grid,\
        # gaussianity_pval_vbar_p1_dat_grid, gaussianity_pval_vbar_p1_std_grid,\
        # gaussianity_pval_vbar_Z1_dat_grid, gaussianity_pval_vbar_Z1_std_grid,\
        # gaussianity_pval_vbar_RR_dat_grid, gaussianity_pval_vbar_RR_std_grid,\
        # gaussianity_pval_vbar_pp_dat_grid, gaussianity_pval_vbar_pp_std_grid,\
        # gaussianity_pval_vbar_ZZ_dat_grid, gaussianity_pval_vbar_ZZ_std_grid,\
        # gaussianity_pval_vbar_RZ_dat_grid, gaussianity_pval_vbar_RZ_std_grid = self.gaussianity_pval_grids





        # TRACER DENSITY
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, self.nu_dat_grid,
                        'nu_data.pdf', colormap = 'magma',
                        lognorm = True, cb_label='Tracer density stars [stars pc$^{-3}$]')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, self.counts_grid,
                        'nu_data_pure_counts.pdf', colormap = 'magma',
                        lognorm = True, vmin=10,
                        cb_label='Star count [stars per bin]')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, gaussianity_pval_counts_grid,
                        'nu_gauss_pval.pdf', colormap = 'magma',
                        lognorm = True, vmin=1e-2, vmax=1.,
                        cb_label='Tracer density gaussianity p-value')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, skewness_stat_counts_grid,
                        'nu_skew_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = 'Tracer density Skewness z-score')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, kurtosis_stat_counts_grid,
                        'nu_kurt_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = 'Tracer density kurtosis z-score')




        #Vertical Velocity vZ1
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, self.vbar_Z1_dat_grid,
                        'vbar_Z1_data.pdf', colormap = 'seismic',
                        lognorm = False, vmin=-30., vmax=30.,
                        cb_label='Vertical velocity $\overline{v_Z}$ [km s$^{-1}$]')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, gaussianity_pval_vbar_Z1_dat_grid,
                        'vbar_Z1_gauss_pval.pdf', colormap = 'magma',
                        lognorm = True, vmin=1e-2, vmax=1.,
                        cb_label='$\overline{v_Z}$  gaussianity p-value')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, skewness_stat_vbar_Z1_dat_grid,
                        'vbar_Z1_skew_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_Z}$  Skewness z-score')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, kurtosis_stat_vbar_Z1_dat_grid,
                        'vbar_Z1_kurt_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_Z}$  kurtosis z-score')

        #Vertical Velocity vZZ
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, self.vbar_ZZ_dat_grid,
                        'vbar_ZZ_data.pdf', colormap = 'magma',
                        lognorm = False, vmin=0., vmax=4000.,
                        cb_label='Vertical velocity $\overline{v_Z v_Z}$ [km$^{2}$ s$^{-2}$]')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, gaussianity_pval_vbar_ZZ_dat_grid,
                        'vbar_ZZ_gauss_pval.pdf', colormap = 'magma',
                        lognorm = True, vmin=1e-2, vmax=1.,
                        cb_label='$\overline{v_Z v_Z}$  gaussianity p-value')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, skewness_stat_vbar_ZZ_dat_grid,
                        'vbar_ZZ_skew_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_Z v_Z}$  skewness z-score')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, kurtosis_stat_vbar_ZZ_dat_grid,
                        'vbar_Z1_kurt_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_Z v_Z}$  kurtosis z-score')

        #Radial Velocity vR
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, self.vbar_R1_dat_grid,
                        'vbar_R1_data.pdf', colormap = 'magma',
                        lognorm = False, vmin=0., vmax=60.,
                        cb_label='Radial velocity $\overline{v_R}$ [km s$^{-1}$]')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, gaussianity_pval_vbar_R1_dat_grid,
                        'vbar_R1_gauss_pval.pdf', colormap = 'magma',
                        lognorm = True, vmin=1e-2, vmax=1.,
                        cb_label='$\overline{v_R}$  gaussianity p-value')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, skewness_stat_vbar_R1_dat_grid,
                        'vbar_R1_skew_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_R}$  Skewness z-score')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, kurtosis_stat_vbar_R1_dat_grid,
                        'vbar_R1_kurt_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_R}$  kurtosis z-score')

        #Radial Velocity vRR
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, self.vbar_RR_dat_grid,
                        'vbar_RR_data.pdf', colormap = 'magma',
                        lognorm = False, vmin=0., vmax=5000.,
                        cb_label='Radial velocity $\overline{v_R v_R}$ [km$^{2}$ s$^{-2}$]')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, gaussianity_pval_vbar_RR_dat_grid,
                        'vbar_RR_gauss_pval.pdf', colormap = 'magma',
                        lognorm = True, vmin=1e-2, vmax=1.,
                        cb_label='$\overline{v_R v_R}$  gaussianity p-value')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, skewness_stat_vbar_RR_dat_grid,
                        'vbar_RR_skew_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_R v_R}$  Skewness z-score')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, kurtosis_stat_vbar_RR_dat_grid,
                        'vbar_RR_kurt_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_R v_R}$  kurtosis z-score')


        #Tilt Term vRvZ
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, self.vbar_RZ_dat_grid,
                        'vbar_RZ_data.pdf', colormap = 'seismic',
                        lognorm = False, vmin=-1000., vmax=1000.,
                        cb_label='RZ velocity cross term $\overline{v_R v_Z}$ [km$^{2}$ s$^{-2}$]')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, gaussianity_pval_vbar_RZ_dat_grid,
                        'vbar_RZ_gauss_pval.pdf', colormap = 'magma',
                        lognorm = True, vmin=1e-2, vmax=1.,
                        cb_label='$\overline{v_R v_Z}$  gaussianity p-value')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, skewness_stat_vbar_RZ_dat_grid,
                        'vbar_RZ_skew_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_R v_Z}$  Skewness z-score')
        plot_RZ_heatmap(self.R_data_coords_mesh, self.Z_data_coords_mesh, kurtosis_stat_vbar_RZ_dat_grid,
                        'vbar_RZ_kurt_stat.pdf', colormap = 'magma',
                        lognorm = False, vmin=1e-1, vmax=1.,
                        cb_label = '$\overline{v_R v_Z}$  kurtosis z-score')

    def plot_correlation_matrix(self):
        # Total Correlation Matrix
        plot_matrix_heatmap(self.data_corr, 'correlation_matrix_all.pdf')
        pdb.set_trace()
        #Counts correlations
        block_size = len(self.R_data_coords_mesh.flatten())
        file_name_vec = ['correlation_matrix_counts.pdf',
                            'correlation_matrix_vbar_R1.pdf',
                            'correlation_matrix_vbar_p1.pdf',
                            'correlation_matrix_vbar_Z1.pdf',
                            'correlation_matrix_vbar_RR.pdf',
                            'correlation_matrix_vbar_pp.pdf',
                            'correlation_matrix_vbar_ZZ.pdf',
                            'correlation_matrix_vbar_RZ.pdf']
        for NN in range(0,8):
            plot_matrix_heatmap(self.data_corr[NN*block_size:(NN+1)*block_size,
                                                NN*block_size:(NN+1)*block_size],
                                file_name_vec[NN],colormap='seismic')




if __name__ == "__main__":

    oscar_test = oscar_gaia_data(N_samplings = 11, N_cores=1,num_R_bins=5,num_Z_bins=10,
                                    binning_type='linear')
    oscar_test.plot_histograms()
    oscar_test.plot_correlation_matrix()






# #PROCESS DATA OR LOAD FROM CACHE
# file_name = 'cache_5_feb_full_sample_quartile_binning_test'
# codemode = 'SAVE' # 'LOAD'
#
# if codemode == 'LOAD':
#     data_mean_grids, sigma_meas_grids,\
#     skewness_stat_grids, skewness_pval_grids,\
#     kurtosis_stat_grids, kurtosis_pval_grids,\
#     gaussianity_stat_grids, gaussianity_pval_grids = np.load(file_name + '.npy')
#
# elif codemode == 'SAVE':
#     #Import Astrometric Data
#     data_folder = '../Astrometric_Data/Gaia_DR2_subsamples/'
#     #data_file = 'gaiaDR2_6D_test_sample_100k-result.csv'
#     #data_file = 'GaiaDR2_RC_sample_Mcut_0p0_0p75_Ccut_1p0_1p5_Nstars_20000.csv'
#     data_file = 'GaiaDR2_RC_sample_Mcut_0p0_0p75_Ccut_1p0_1p5Nstars_1333998.csv'
#
#     datab = pd.read_csv(data_folder + data_file) #astrometric_data_table
#
#     #Construct Means and Covarriance Matrices
#     astrometric_means = np.array([datab['ra'].values * deg_to_rad, #rad
#                             datab['dec'].values * deg_to_rad, #rad
#                             datab['parallax'].values, #mas
#                             datab['pmra'].values * maspyr_to_radps, #rad/s
#                             datab['pmdec'].values * maspyr_to_radps, #rad/s
#                             datab['radial_velocity'].values]).T #km/s
#     Nstars = datab['ra'].values.shape[0]
#     Nzeros = np.zeros(Nstars)
#
#     astrometric_covariances = np.array([[(datab['ra_error'].values*mas_to_rad)**2,
        # datab['ra_dec_corr'].values * datab['ra_error'].values * datab['dec_error'].values * mas_to_rad**2,
        # datab['ra_parallax_corr'].values * datab['ra_error'].values * datab['parallax_error'].values * mas_to_rad,
        # datab['ra_pmra_corr'].values * datab['ra_error'].values * datab['pmra_error'].values * mas_to_rad * maspyr_to_radps,
        # datab['ra_pmdec_corr'].values * datab['ra_error'].values * datab['pmdec_error'].values * mas_to_rad * maspyr_to_radps,
        # Nzeros],
        # [Nzeros, (datab['dec_error'].values*mas_to_rad)**2,
        # datab['dec_parallax_corr'].values * datab['dec_error'].values * datab['parallax_error'].values * mas_to_rad,
        # datab['dec_pmra_corr'].values * datab['dec_error'].values * datab['pmra_error'].values * mas_to_rad * maspyr_to_radps,
        # datab['dec_pmdec_corr'].values * datab['dec_error'].values * datab['pmdec_error'].values * mas_to_rad * maspyr_to_radps,
        # Nzeros],
        # [Nzeros, Nzeros, datab['parallax_error'].values**2,
        # datab['parallax_pmra_corr'].values * datab['parallax_error'].values * datab['pmra_error'].values * maspyr_to_radps,
        # datab['parallax_pmdec_corr'].values * datab['parallax_error'].values * datab['pmdec_error'].values * maspyr_to_radps,
        # Nzeros],
        # [Nzeros,Nzeros,Nzeros, (datab['pmra_error'].values * maspyr_to_radps)**2,
        # datab['pmra_pmdec_corr'].values * datab['pmra_error'].values * datab['pmdec_error'].values * maspyr_to_radps**2,
        # Nzeros],
        # [Nzeros, Nzeros, Nzeros, Nzeros, (datab['pmdec_error'].values * maspyr_to_radps)**2, Nzeros],
        # [Nzeros, Nzeros, Nzeros, Nzeros, Nzeros, datab['radial_velocity_error'].values**2]])
#
#     astrometric_covariances = np.transpose(astrometric_covariances, (2,0,1)) #Rearrange
#     astrometric_covariances = np.array([astrometric_covariances[ii] + astrometric_covariances[ii].T - \
#                                     np.diagonal(astrometric_covariances[ii])*np.identity(6) \
#                                     for ii in range(Nstars)]) #Symmetrize
#     cholesky_astrometric_covariances = np.linalg.cholesky(astrometric_covariances)
#
#
#     #Calculate epoch_T matrix
#     epoch_T = calc_epoch_T('J2000')
#
#     #Calculate Quantile Binning
#     galactocentric_means = astrometric_to_galactocentric(
#                             astrometric_means[:,0], astrometric_means[:,1],
#                             astrometric_means[:,2], astrometric_means[:,3],
#                             astrometric_means[:,4], astrometric_means[:,5],
#                             solar_pomo_means[0], solar_pomo_means[1],
#                             solar_pomo_means[2], solar_pomo_means[3],
#                             solar_pomo_means[4], solar_pomo_means[5],
#                             epoch_T)
#     phi_cut_locs = np.where((galactocentric_means[1]>phi_limit[0]) & (galactocentric_means[1]<phi_limit[1]))
#     Rg_vec_means = galactocentric_means[0][phi_cut_locs]
#     Zg_vec_means = galactocentric_means[2][phi_cut_locs]
#
#     histogram = physt_h2(Rg_vec_means, Zg_vec_means, "quantile", (40,40))
#
#     R_edges = histogram.numpy_bins[0][1:-1]
#     Z_edges = histogram.numpy_bins[1][1:-1]
#
#     R_bin_centers = (R_edges[1:] + R_edges[:-1])/2
#     Z_bin_centers = (Z_edges[1:] + Z_edges[:-1])/2
#     R_data_coords_mesh, Z_data_coords_mesh = np.meshgrid(R_bin_centers, Z_bin_centers, indexing='ij')
#
#     bin_vol_grid= np.zeros([len(R_edges) - 1, len(Z_edges)-1])
#     for (aa,bb), dummy in np.ndenumerate(bin_vol_grid):
#         bin_vol_grid[aa,bb] = 0.5 * abs(phi_limit[1]-phi_limit[0])\
#                         * abs(R_edges[aa+1]**2 - R_edges[aa]**2)\
#                         * abs(Z_edges[bb+1] - Z_edges[bb])
#
#
#     if single_core:
#         #Linear Sample Transform Bin
#         all_binned_data_vectors = []
#         start = time.time()
#         for jj in range(N_samplings):
#             print('Sample ', jj, ' of ', N_samplings)
#             binned_data_vector = sample_transform_bin(astrometric_means, astrometric_covariances,
#                                         cholesky_astrometric_covariances,
#                                         solar_pomo_means, solar_pomo_covariances,
#                                         epoch_T,jj)
#             all_binned_data_vectors.append(binned_data_vector)
#         all_binned_data_vectors = np.array(all_binned_data_vectors)
#         print('\nLinear Sampling, Transforming, Binning takes ', time.time()-start, ' s')
#         print('Time per sample: ', (time.time()-start)/N_samplings, ' s\n')
#
#     else:
#         #Multiprocessor Pool
#         print('Starting Parallel Sampling')
#         start = time.time()
#         pool = mp.Pool(processes=6)
#         results = [pool.apply_async(sample_transform_bin,
#                                 args = (astrometric_means, astrometric_covariances,
#                                         cholesky_astrometric_covariances,
#                                         solar_pomo_means, solar_pomo_covariances,
#                                         epoch_T, seed)) for seed in range(N_samplings)]
#         output = [p.get() for p in results]
#         all_binned_data_vectors = np.array(output)
#         end = time.time()
#         print('Parallel Sampling, Transforming, Binning takes ', end-start, ' s')
#         print('Wall time per sample: ', (end-start)/N_samplings)
#
#     #Calculate means and covariances, Skewness, Kurtosis
#     data_mean = np.mean(all_binned_data_vectors, axis=0)
#     data_cov  = np.cov(all_binned_data_vectors.T)
#     data_corr = np.corrcoef(all_binned_data_vectors.T)
#     data_sigma2 = np.diag(data_cov)
#
#     #Gaussianity test using D’Agostino and Pearson’s tests
#
#     skewness_stat, skewness_pval = stats.skewtest(all_binned_data_vectors)
#     kurtosis_stat, kurtosis_pval = stats.kurtosistest(all_binned_data_vectors)
#     gaussianity_stat, gaussianity_pval = stats.normaltest(all_binned_data_vectors)
#
#     #Reformat into individual quantities
#     grid_shape = (15, len(R_edges)-1, len(Z_edges)-1)
#
#     # Reshape
#     data_mean_grids = data_mean.reshape(grid_shape)
#     sigma_meas_grids = np.sqrt(data_sigma2).reshape(grid_shape)
#     skewness_stat_grids = skewness_stat.reshape(grid_shape)
#     skewness_pval_grids = skewness_pval.reshape(grid_shape)
#     kurtosis_stat_grids = kurtosis_stat.reshape(grid_shape)
#     kurtosis_pval_grids = kurtosis_pval.reshape(grid_shape)
#     gaussianity_stat_grids = gaussianity_stat.reshape(grid_shape)
#     gaussianity_pval_grids = gaussianity_pval.reshape(grid_shape)
#
#     np.save(file_name, np.array([data_mean_grids, sigma_meas_grids,
#                                     skewness_stat_grids, skewness_pval_grids,
#                                     kurtosis_stat_grids, kurtosis_pval_grids,
#                                     gaussianity_stat_grids, gaussianity_pval_grids]))
# else:
#     print('Wrong option')
#
#
#
#
#
#
#
#
#
#
#
#
# print('Oscar the Grouch... Out')
#
#
# # (counts_grid, \
# # vbar_R1_dat_grid, vbar_R1_std_grid,\
# # vbar_p1_dat_grid, vbar_p1_std_grid,\
# # vbar_Z1_dat_grid, vbar_Z1_std_grid,\
# # vbar_RR_dat_grid, vbar_RR_std_grid,\
# # vbar_pp_dat_grid, vbar_pp_std_grid,\
# # vbar_ZZ_dat_grid, vbar_ZZ_std_grid,\
# # vbar_RZ_dat_grid, vbar_RZ_std_grid) \
#
# # T= sc.dot(sc.array([[sc.cos(theta),sc.sin(theta),0.],[sc.sin(theta),-sc.cos(theta),0.],[0.,0.,1.]]),
# #             sc.dot(sc.array([[-sc.sin(dec_ngp),0.,sc.cos(dec_ngp)],[0.,1.,0.],
#[sc.cos(dec_ngp),0.,sc.sin(dec_ngp)]]),sc.array([[sc.cos(ra_ngp),sc.sin(ra_ngp),0.],[-sc.sin(ra_ngp),sc.cos(ra_ngp),0.],[0.,0.,1.]])))
