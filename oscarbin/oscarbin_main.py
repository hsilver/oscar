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
import sklearn
import sklearn.covariance as sklcov

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
      vTsun       km/s
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
    # vXh towards the sun, vYh towards direction of rotation, vZh upwards

    if print_out:
        print('Velocities xyz Heliocentric')
        print('vxh: ', vxh)
        print('vyh: ', vyh)
        print('vzh: ', vzh)
        print('')

    #vxvyvz_to_galcencyl
    vXg_vec, vYg_vec, vZg_vec = np.dot(h2g_rot_mat,np.array([-vXh_vec, vYh_vec,np.sign(Rsun) * vZh_vec]))\
                                + np.array([-vRsun,vYsun,vZsun]).reshape(3,1)

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

def astrometric_to_galactocentric_positions_only(ra, dec, para,  Rsun, phisun, Zsun, epoch_T):
    """
    Frankensteined together from Jo Bovy's galpy code. Any errors most likely
    attributable to HS.
    Inputs:
      ra          rad     array
      dec         rad     array
      parallax    mas     array
      Rsun        pc
      phisun      rads (but zero by default)
      Zsun        pc
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

    return Rg_vec, phig_vec, Zg_vec


def binning(Rg_vec, phig_vec, Zg_vec, vRg_vec, vTg_vec, vZg_vec, R_edges, phi_edges, Z_edges):
    """
    Rg_vec, star_data_gccyl[0]
    phig_vec, star_data_gccyl[1]
    Zg_vec, star_data_gccyl[2]

    vRg_vec, star_V_gccyl[0]
    vphig_vec, star_V_gccyl[1]
    vZg_vec, star_V_gccyl[2]

    0 counts_grid,
    1 vbar_R1_dat_grid,
    2 vbar_p1_dat_grid,
    3 vbar_T1_dat_grid,
    4 vbar_Z1_dat_grid,
    5 vbar_RR_dat_grid,
    6 vbar_pp_dat_grid,
    7 vbar_TT_dat_grid,
    8 vbar_ZZ_dat_grid,
    9 vbar_Rp_dat_grid,
    10 vbar_RT_dat_grid,
    11 vbar_RZ_dat_grid,
    12 vbar_pZ_dat_grid,
    13 vbar_TZ_dat_grid])

    std refers to the standard deviation of the mean, hence the extra 1/sqrt(N)
    """
    #vphig_vec = vTg_vec/(Rg_vec * 3.086E1) #picorad/s
    vphig_vec = vTg_vec/(Rg_vec * 3.086E13) #rad/s
    counts_grid = stats.binned_statistic_dd([Rg_vec,phig_vec,Zg_vec],
                                            Rg_vec, #dummy array for count
                                            statistic='count',
                                            bins=[R_edges, phi_edges, Z_edges])[0]
    counts_pois_grid = np.sqrt(counts_grid)

    # Binning Velocity Mean Calculations
    (vbar_R1_dat_grid, vbar_p1_dat_grid, vbar_T1_dat_grid, vbar_Z1_dat_grid,
    vbar_RR_dat_grid, vbar_pp_dat_grid, vbar_TT_dat_grid, vbar_ZZ_dat_grid,
    vbar_Rp_dat_grid, vbar_RT_dat_grid, vbar_RZ_dat_grid,
    vbar_pZ_dat_grid, vbar_TZ_dat_grid)\
        = np.ma.masked_invalid(stats.binned_statistic_dd([Rg_vec,phig_vec,Zg_vec],
                            [vRg_vec, vphig_vec, vTg_vec, vZg_vec,
                             vRg_vec**2, vphig_vec**2, vTg_vec**2, vZg_vec**2,
                             vRg_vec*vphig_vec, vRg_vec*vTg_vec, vRg_vec*vZg_vec,
                             vphig_vec*vZg_vec, vTg_vec*vZg_vec],
                             statistic='mean',
                             bins=[R_edges, phi_edges, Z_edges])[0])

    (vbar_R1_std_grid, vbar_p1_std_grid, vbar_T1_std_grid, vbar_Z1_std_grid,
    vbar_RR_std_grid, vbar_pp_std_grid, vbar_TT_std_grid, vbar_ZZ_std_grid,
    vbar_Rp_std_grid, vbar_RT_std_grid, vbar_RZ_std_grid,
    vbar_pZ_std_grid, vbar_TZ_std_grid)\
        = np.ma.masked_invalid(stats.binned_statistic_dd([Rg_vec,phig_vec,Zg_vec],
                            [vRg_vec, vphig_vec, vTg_vec, vZg_vec,
                             vRg_vec**2, vphig_vec**2, vTg_vec**2, vZg_vec**2,
                             vRg_vec*vphig_vec, vRg_vec*vTg_vec, vRg_vec*vZg_vec,
                             vphig_vec*vZg_vec, vTg_vec*vZg_vec],
                             statistic=np.std,
                             bins=[R_edges, phi_edges, Z_edges])[0])/([np.sqrt(counts_grid)]*13)

    return np.array([counts_grid,
            vbar_R1_dat_grid, vbar_p1_dat_grid, vbar_T1_dat_grid, vbar_Z1_dat_grid,
            vbar_RR_dat_grid, vbar_pp_dat_grid, vbar_TT_dat_grid, vbar_ZZ_dat_grid,
            vbar_Rp_dat_grid, vbar_RT_dat_grid, vbar_RZ_dat_grid, vbar_pZ_dat_grid, vbar_TZ_dat_grid]),\
            np.array([counts_pois_grid,
            vbar_R1_std_grid, vbar_p1_std_grid, vbar_T1_std_grid, vbar_Z1_std_grid,
            vbar_RR_std_grid, vbar_pp_std_grid, vbar_T1_std_grid, vbar_ZZ_std_grid,
            vbar_Rp_std_grid, vbar_RT_std_grid, vbar_RZ_std_grid, vbar_pZ_std_grid, vbar_TZ_std_grid])


def binning_positions_only(Rg_vec, phig_vec, Zg_vec, R_edges, phi_edges, Z_edges):
    counts_grid = stats.binned_statistic_dd([Rg_vec,phig_vec,Zg_vec],
                                            Rg_vec, #dummy array for count
                                            statistic='count',
                                            bins=[R_edges, phi_edges, Z_edges])[0]
    counts_pois_grid = np.sqrt(counts_grid)

    return np.array([counts_grid]), np.array([counts_pois_grid])

def sample_transform_bin(astrometric_means, astrometric_covariances,
                            cholesky_astrometric_covariances,
                            solar_pomo_means, solar_pomo_covariances,
                            epoch_T, seed,
                            R_edges, phi_edges, Z_edges,
                            positions_only=False):
    """
    #https://stackoverflow.com/questions/14920272/generate-a-data-set-consisting-of-n-100-2-dimensional-samples
    """
    rand.seed(int(seed + int(time.time())%10000+1))
    # stars_sample = np.array([rand.multivariate_normal(astrometric_means[ii],
    #                     astrometric_covariances[ii]) for ii in range(Nstars)])
    #Cholesky Decomposition Method
    Nstars = len(astrometric_means)
    if positions_only:
        uncorrelated_sample = np.random.standard_normal((Nstars,3))
        stars_sample = np.array([np.dot(cholesky_astrometric_covariances[ii],\
                            uncorrelated_sample[ii]) + astrometric_means[ii] for ii in range(Nstars)])
        solar_pomo_sample = rand.multivariate_normal(solar_pomo_means[0:3], solar_pomo_covariances[0:3,0:3])
        Rg_vec, phig_vec, Zg_vec =astrometric_to_galactocentric_positions_only(
                stars_sample[:,0], stars_sample[:,1], stars_sample[:,2],
                solar_pomo_sample[0], solar_pomo_sample[1], solar_pomo_sample[2], epoch_T)
        binned_data_vector, binned_std_vector = binning_positions_only(
                                                Rg_vec, phig_vec, Zg_vec,
                                                R_edges, phi_edges, Z_edges)

    else:
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
        binned_data_vector, binned_std_vector = binning(
                                                Rg_vec, phig_vec, Zg_vec,
                                                vRg_vec, vTg_vec, vZg_vec,
                                                R_edges, phi_edges, Z_edges)

        # Extra Quantities
        vertex_deviation_dat_grid = vertex_deviation(binned_data_vector[0], #R1
                                                    binned_data_vector[3], #T1
                                                    binned_data_vector[5], #RR
                                                    binned_data_vector[7], #TT
                                                    binned_data_vector[10])

    return binned_data_vector.flatten(), binned_std_vector.flatten(), \
            vertex_deviation_dat_grid.flatten()

def vertex_deviation(vbar_R1_dat_grid, vbar_T1_dat_grid, vbar_RR_dat_grid,
                    vbar_TT_dat_grid, vbar_RT_dat_grid):
    return -2*(180/np.pi)*(vbar_RT_dat_grid - vbar_R1_dat_grid*vbar_T1_dat_grid)\
        /(vbar_RR_dat_grid - vbar_R1_dat_grid**2 - vbar_TT_dat_grid + vbar_T1_dat_grid**2)

##########################################################################

class oscar_gaia_data:
    """
    Performs sample-bin-repeat on astrometric Gaia data
    2019-01     Hamish Silverwood, basic machinery
    2019-02     Turned into class object
    """
    def __init__(self, data_root = '../Astrometric_Data/Gaia_DR2_subsamples/',
                        data_file_name = 'GaiaDR2_RC_sample_Mcut_0p0_0p75_Ccut_1p0_1p5Nstars_1333998.csv',
                        binning_type = 'linear', #linear #input
                        Rmin = 6000, Rmax = 10000, num_R_bins = 10,
                        phimin = -np.pi/4, phimax=np.pi/4, num_phi_bins = 3,
                        Zmin = -2000, Zmax = 2000, num_Z_bins = 10,
                        input_R_edges = None,
                        input_phi_edges = None,
                        input_Z_edges = None,
                        N_samplings = 100,
                        N_cores = 1,
                        calculate_covariance = True,
                        positions_only = False
                        ):
        self.data_root = data_root
        self.data_file_name = data_file_name

        self.binning_type = binning_type
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.num_R_bins = num_R_bins
        self.phimin = phimin
        self.phimax = phimax
        self.num_phi_bins = num_phi_bins
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.num_Z_bins = num_Z_bins
        self.input_R_edges = input_R_edges
        self.input_phi_edges = input_phi_edges
        self.input_Z_edges = input_Z_edges
        self.N_samplings = N_samplings
        self.N_cores = N_cores
        self.calculate_covariance = calculate_covariance
        self.positions_only = positions_only

        # Set Constants and Parameters
        deg_to_rad = np.pi/180
        mas_to_rad = (np.pi/6.48E8)
        maspyr_to_radps = np.pi/(6.48E8 * 31557600)

        # Solar Position and Motion model
        self.solar_pomo_means = np.array([8200.,0.,20.8, 10.,248.,7.])
        self.solar_pomo_stds = np.array([100., 0., 0.3, 1., 3., 0.5])
        self.solar_pomo_covariances = np.identity(6) * self.solar_pomo_stds**2
        """
        Bland Hawthorn et al 2016 review
        R0 = 8200±100 pc
        Z0 = 25±5 pc
        Vgsun = 248±3 km/s, tangential velocity relative to Sgr A*
        Usun = 10.0±1 km/s, radial, positive towards the galactic center
        Vsun = 11.0±2 km/s, in direction of rotation
        Wsun = 7.0±0.5 km/s, vertical upwards positive

        Bennet & Bovy 2018
        Z0 = 20.8 ± 0.3 pc
        """
        # Open data file
        datab = pd.read_csv(self.data_root + self.data_file_name) #astrometric_data_table

        # Construct Means and Covarriance Matrices
        if self.positions_only:
            astrometric_means = np.array([datab['ra'].values * deg_to_rad, #rad
                                    datab['dec'].values * deg_to_rad, #rad
                                    datab['parallax'].values]).T #mas
        else:
            astrometric_means = np.array([datab['ra'].values * deg_to_rad, #rad
                                    datab['dec'].values * deg_to_rad, #rad
                                    datab['parallax'].values, #mas
                                    datab['pmra'].values * maspyr_to_radps, #rad/s
                                    datab['pmdec'].values * maspyr_to_radps, #rad/s
                                    datab['radial_velocity'].values]).T #km/s

        Nstars = datab['ra'].values.shape[0]
        Nzeros = np.zeros(Nstars)

        if self.positions_only:
            astrometric_covariances = np.array([[(datab['ra_error'].values*mas_to_rad)**2,
                datab['ra_dec_corr'].values * datab['ra_error'].values * datab['dec_error'].values * mas_to_rad**2,
                datab['ra_parallax_corr'].values * datab['ra_error'].values * datab['parallax_error'].values * mas_to_rad],
                [Nzeros, (datab['dec_error'].values*mas_to_rad)**2,
                datab['dec_parallax_corr'].values * datab['dec_error'].values * datab['parallax_error'].values * mas_to_rad],
                [Nzeros, Nzeros, datab['parallax_error'].values**2]])
            astrometric_covariances = np.transpose(astrometric_covariances, (2,0,1)) #Rearrange
            astrometric_covariances = np.array([astrometric_covariances[ii] + astrometric_covariances[ii].T - \
                                            np.diagonal(astrometric_covariances[ii])*np.identity(3) \
                                            for ii in range(Nstars)]) #Symmetrize
        else:
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
            self.phi_edges = self.input_phi_edges
            self.Z_edges = self.input_Z_edges
            self.num_R_bins = len(self.input_R_edges)-1
            self.num_phi_bins = len(self.input_phi_edges)-1
            self.num_Z_bins = len(self.input_Z_edges)-1
        elif binning_type == 'linear':
            self.R_edges = np.linspace(self.Rmin, self.Rmax, self.num_R_bins+1)
            self.phi_edges = np.linspace(self.phimin, self.phimax, self.num_phi_bins+1)
            self.Z_edges = np.linspace(self.Zmin, self.Zmax, self.num_Z_bins+1)
        elif binning_type == 'quartile':
            galactocentric_means = astrometric_to_galactocentric(
                    astrometric_means[:,0], astrometric_means[:,1],
                    astrometric_means[:,2], Nzeros,
                    Nzeros, Nzeros,
                    self.solar_pomo_means[0], self.solar_pomo_means[1],
                    self.solar_pomo_means[2], 0.,0.,0.,
                    epoch_T)
            Rg_vec_means = galactocentric_means[0]
            phig_vec_means = galactocentric_means[1]
            Zg_vec_means = galactocentric_means[2]

            physt_hist = physt_h3(Rg_vec_means, phig_vec_means, Zg_vec_means, "quantile",
                        (self.num_R_bins+2,self.num_phi_bins+2, self.num_Z_bins+2))
            self.R_edges = physt_hist.numpy_bins[0][1:-1]
            self.phi_edges = physt_hist.numpy_bins[1][1:-1]
            self.Z_edges = physt_hist.numpy_bins[2][1:-1]

        # Calculate bin centers,edge mesh, and volumes
        self.R_bin_centers = (self.R_edges[1:] + self.R_edges[:-1])/2
        self.phi_bin_centers = (self.phi_edges[1:] + self.phi_edges[:-1])/2
        self.Z_bin_centers = (self.Z_edges[1:] + self.Z_edges[:-1])/2
        self.R_data_coords_mesh, self.phi_data_coords_mesh, self.Z_data_coords_mesh\
            = np.meshgrid(self.R_bin_centers, self.phi_bin_centers, self.Z_bin_centers, indexing='ij')

        self.R_edges_mesh, self.phi_edges_mesh, self.Z_edges_mesh \
            = np.meshgrid(self.R_edges, self.phi_edges, self.Z_edges, indexing='ij')


        self.bin_vol_grid= np.zeros([len(self.R_edges) - 1, len(self.phi_edges)-1, len(self.Z_edges)-1])
        for (rr,pp,zz), dummy in np.ndenumerate(self.bin_vol_grid):
            self.bin_vol_grid[rr,pp,zz] = 0.5 * abs(self.phi_edges[pp+1]-self.phi_edges[pp])\
                            * abs(self.R_edges[rr+1]**2 - self.R_edges[rr]**2)\
                            * abs(self.Z_edges[zz+1] - self.Z_edges[zz])

        # Build cache file name
        if not os.path.isdir(data_root + '/oscar_cache_files/'):
            os.mkdir(data_root + '/oscar_cache_files/')

        if self.positions_only:
            cache_file_name = 'oscar_cache_positions_only_' \
                + hashlib.md5(np.concatenate([self.R_edges,self.phi_edges,self.Z_edges])).hexdigest()\
                + hashlib.md5(np.concatenate([self.solar_pomo_means, self.solar_pomo_covariances.flatten()])).hexdigest()\
                + '_' + str(self.N_samplings)\
                + data_file_name.split('.')[0] + '.dat'
        else:
            cache_file_name = 'oscar_cache_' \
                + hashlib.md5(np.concatenate([self.R_edges,self.phi_edges,self.Z_edges])).hexdigest()\
                + hashlib.md5(np.concatenate([self.solar_pomo_means, self.solar_pomo_covariances.flatten()])).hexdigest()\
                + '_' + str(self.N_samplings)\
                + data_file_name.split('.')[0] + '.dat'

        # Search for cache file
        if os.path.isfile(data_root + '/oscar_cache_files/' + cache_file_name):
            print('Previous sampling found, pulling data from cache.')
            cache_dataframe = pd.read_pickle(data_root + '/oscar_cache_files/' + cache_file_name)

            self.data_mean = cache_dataframe['data_mean']
            self.data_cov = cache_dataframe['data_cov']
            self.data_corr = cache_dataframe['data_corr']
            self.data_std_total = cache_dataframe['data_std_total']
            self.data_mean_grids = cache_dataframe['data_mean_grids']
            self.data_var_from_cov = cache_dataframe['data_var_from_cov']
            self.data_var_avg_from_samples = cache_dataframe['data_var_avg_from_samples']
            self.data_std_total_grids = cache_dataframe['data_std_total_grids']
            self.skewness_stat_grids = cache_dataframe['skewness_stat_grids']
            self.skewness_pval_grids = cache_dataframe['skewness_pval_grids']
            self.kurtosis_stat_grids = cache_dataframe['kurtosis_stat_grids']
            self.kurtosis_pval_grids = cache_dataframe['kurtosis_pval_grids']
            self.gaussianity_stat_grids = cache_dataframe['gaussianity_stat_grids']
            self.gaussianity_pval_grids = cache_dataframe['gaussianity_pval_grids']
            self.R_data_coords_mesh = cache_dataframe['R_data_coords_mesh']
            self.phi_data_coords_mesh = cache_dataframe['phi_data_coords_mesh']
            self.Z_data_coords_mesh = cache_dataframe['Z_data_coords_mesh']
            self.R_edges_mesh = cache_dataframe['R_edges_mesh']
            self.phi_edges_mesh = cache_dataframe['phi_edges_mesh']
            self.Z_edges_mesh = cache_dataframe['Z_edges_mesh']
            self.counts_grid = cache_dataframe['counts_grid']
            self.nu_dat_grid = cache_dataframe['nu_dat_grid']
            self.vbar_R1_dat_grid = cache_dataframe['vbar_R1_dat_grid']
            self.vbar_p1_dat_grid = cache_dataframe['vbar_p1_dat_grid']
            self.vbar_T1_dat_grid = cache_dataframe['vbar_T1_dat_grid']
            self.vbar_Z1_dat_grid = cache_dataframe['vbar_Z1_dat_grid']
            self.vbar_RR_dat_grid = cache_dataframe['vbar_RR_dat_grid']
            self.vbar_pp_dat_grid = cache_dataframe['vbar_pp_dat_grid']
            self.vbar_TT_dat_grid = cache_dataframe['vbar_TT_dat_grid']
            self.vbar_ZZ_dat_grid = cache_dataframe['vbar_ZZ_dat_grid']
            self.vbar_Rp_dat_grid = cache_dataframe['vbar_Rp_dat_grid']
            self.vbar_RT_dat_grid = cache_dataframe['vbar_RT_dat_grid']
            self.vbar_RZ_dat_grid = cache_dataframe['vbar_RZ_dat_grid']
            self.vbar_pZ_dat_grid = cache_dataframe['vbar_pZ_dat_grid']
            self.vbar_TZ_dat_grid = cache_dataframe['vbar_TZ_dat_grid']
            self.counts_std_grid = cache_dataframe['counts_std_grid']
            self.nu_std_grid = cache_dataframe['nu_std_grid']
            self.vbar_R1_std_grid = cache_dataframe['vbar_R1_std_grid']
            self.vbar_p1_std_grid = cache_dataframe['vbar_p1_std_grid']
            self.vbar_T1_std_grid = cache_dataframe['vbar_T1_std_grid']
            self.vbar_Z1_std_grid = cache_dataframe['vbar_Z1_std_grid']
            self.vbar_RR_std_grid = cache_dataframe['vbar_RR_std_grid']
            self.vbar_pp_std_grid = cache_dataframe['vbar_pp_std_grid']
            self.vbar_TT_std_grid = cache_dataframe['vbar_TT_std_grid']
            self.vbar_ZZ_std_grid = cache_dataframe['vbar_ZZ_std_grid']
            self.vbar_Rp_std_grid = cache_dataframe['vbar_Rp_std_grid']
            self.vbar_RT_std_grid = cache_dataframe['vbar_RT_std_grid']
            self.vbar_RZ_std_grid = cache_dataframe['vbar_RZ_std_grid']
            self.vbar_pZ_std_grid = cache_dataframe['vbar_pZ_std_grid']
            self.vbar_TZ_std_grid = cache_dataframe['vbar_TZ_std_grid']

            self.median_vertex_dev_vector = cache_dataframe['median_vertex_dev_vector']
            self.mean_vertex_dev_vector = cache_dataframe['mean_vertex_dev_vector']
            self.vertex_dev_3sig_lower = cache_dataframe['vertex_dev_3sig_lower']
            self.vertex_dev_2sig_lower = cache_dataframe['vertex_dev_2sig_lower']
            self.vertex_dev_1sig_lower = cache_dataframe['vertex_dev_1sig_lower']
            self.vertex_dev_1sig_upper = cache_dataframe['vertex_dev_1sig_upper']
            self.vertex_dev_2sig_upper = cache_dataframe['vertex_dev_2sig_upper']
            self.vertex_dev_3sig_upper = cache_dataframe['vertex_dev_3sig_upper']

        else:
            print('No previous sampling found, running from scratch')

            if N_cores == 1:
                #Linear Sample Transform Bin
                all_binned_data_vectors = []
                all_binned_std_vectors = []
                all_vertex_dev_vectors = []
                start = time.time()
                for jj in range(N_samplings):
                    print('Sample ', jj+1, ' of ', N_samplings)
                    binned_data_vector, binned_std_vector,\
                    vertex_deviation_vector = sample_transform_bin(
                                        astrometric_means, astrometric_covariances,
                                        cholesky_astrometric_covariances,
                                        self.solar_pomo_means, self.solar_pomo_covariances,
                                        epoch_T,jj,
                                        self.R_edges, self.phi_edges, self.Z_edges,
                                        positions_only = self.positions_only)
                    all_binned_data_vectors.append(binned_data_vector)
                    all_binned_std_vectors.append(binned_std_vector)
                    all_vertex_dev_vectors.append(vertex_deviation_vector)

                all_binned_data_vectors = np.array(all_binned_data_vectors)
                all_binned_std_vectors = np.array(all_binned_std_vectors)
                all_vertex_dev_vectors = np.array(all_vertex_dev_vectors)
                print('\nLinear Sampling, Transforming, Binning takes ', time.time()-start, ' s')
                print('Time per sample: ', (time.time()-start)/N_samplings, ' s\n')

            else:
                #Multiprocessor Pool
                print('Starting Parallel Sampling')
                start = time.time()
                pool = mp.Pool(processes=self.N_cores)
                results = [pool.apply_async(sample_transform_bin,
                                            (astrometric_means, astrometric_covariances,
                                            cholesky_astrometric_covariances,
                                            self.solar_pomo_means, self.solar_pomo_covariances,
                                            epoch_T, seed,
                                            self.R_edges, self.phi_edges, self.Z_edges),
                                            dict(positions_only = self.positions_only)) for seed in range(N_samplings)]

                output = [p.get() for p in results]
                all_binned_data_vectors = np.array([output[ii][0] for ii in range(N_samplings)])
                all_binned_std_vectors = np.array([output[ii][1] for ii in range(N_samplings)])
                all_vertex_dev_vectors = np.array([output[ii][1] for ii in range(N_samplings)])
                end = time.time()
                print('Parallel Sampling, Transforming, Binning takes ', end-start, ' s')
                print('Wall time per sample: ', (end-start)/N_samplings)

            #Calculate means and covariances, Skewness, Kurtosis
            if self.positions_only:
                grid_shape = (1, len(self.R_edges)-1, len(self.phi_edges)-1, len(self.Z_edges)-1)
            else:
                grid_shape = (14, len(self.R_edges)-1, len(self.phi_edges)-1, len(self.Z_edges)-1)
            subvector_length = (len(self.R_edges)-1)*(len(self.phi_edges)-1)*(len(self.Z_edges)-1)

            self.data_mean = np.mean(all_binned_data_vectors, axis=0)
            self.data_median = np.median(all_binned_data_vectors, axis=0)

            self.std_mean = np.mean(all_binned_std_vectors, axis=0)
            self.std_median = np.median(all_binned_std_vectors, axis=0)

            if self.calculate_covariance:
                covariance_fit = sklcov.EmpiricalCovariance().fit(all_binned_data_vectors)
                self.data_cov = covariance_fit.covariance_
                self.data_var_from_cov = np.diag(self.data_cov)
                data_sigma_inv = 1/np.sqrt(np.diag(self.data_cov))
                data_sigma_inv = data_sigma_inv.reshape(len(data_sigma_inv), 1)
                self.data_corr = np.dot(data_sigma_inv, data_sigma_inv.T) * self.data_cov
            else:
                self.data_cov = np.zeros(1)
                self.data_var_from_cov = np.var(all_binned_data_vectors, axis=0)
                self.data_corr = np.zeros(1)

            #Combine the mean sample variances with variances from the covariance fit
            #   (eg the variance between the means).
            counts_subvectors = all_binned_data_vectors[:,0:subvector_length]
            if positions_only:
                counts_repeated = np.hstack([counts_subvectors]*1)
            else:
                counts_repeated = np.hstack([counts_subvectors]*14)

            self.data_var_avg_from_samples = np.sum(counts_repeated * \
                (np.nan_to_num(all_binned_std_vectors)**2),axis=0)/np.sum(counts_repeated,axis=0)
            self.data_std_total = np.sqrt(self.data_var_from_cov + self.data_var_avg_from_samples)

            #Gaussianity test using D’Agostino and Pearson’s tests
            self.skewness_stat, self.skewness_pval = stats.skewtest(all_binned_data_vectors)
            self.kurtosis_stat, self.kurtosis_pval = stats.kurtosistest(all_binned_data_vectors)
            self.gaussianity_stat, self.gaussianity_pval = stats.normaltest(all_binned_data_vectors)

            # Reshape
            self.data_mean_grids = self.data_mean.reshape(grid_shape)
            self.data_std_total_grids = self.data_std_total.reshape(grid_shape)
            self.skewness_stat_grids = self.skewness_stat.reshape(grid_shape)
            self.skewness_pval_grids = self.skewness_pval.reshape(grid_shape)
            self.kurtosis_stat_grids = self.kurtosis_stat.reshape(grid_shape)
            self.kurtosis_pval_grids = self.kurtosis_pval.reshape(grid_shape)
            self.gaussianity_stat_grids = self.gaussianity_stat.reshape(grid_shape)
            self.gaussianity_pval_grids = self.gaussianity_pval.reshape(grid_shape)

            # Pull out means and errors
            if positions_only:
                self.counts_grid = self.data_mean_grids[0]
                self.counts_std_grid = self.data_std_total_grids[0]

                self.vbar_R1_dat_grid = self.vbar_p1_dat_grid = \
                self.vbar_T1_dat_grid = self.vbar_Z1_dat_grid = \
                self.vbar_RR_dat_grid = self.vbar_pp_dat_grid = \
                self.vbar_TT_dat_grid = self.vbar_ZZ_dat_grid = \
                self.vbar_Rp_dat_grid = self.vbar_RT_dat_grid = \
                self.vbar_RZ_dat_grid = self.vbar_pZ_dat_grid = \
                self.vbar_TZ_dat_grid = \
                self.vbar_R1_std_grid = self.vbar_p1_std_grid = \
                self.vbar_T1_std_grid = self.vbar_Z1_std_grid = \
                self.vbar_RR_std_grid = self.vbar_pp_std_grid = \
                self.vbar_TT_std_grid = self.vbar_ZZ_std_grid = \
                self.vbar_Rp_std_grid = self.vbar_RT_std_grid = \
                self.vbar_RZ_std_grid = self.vbar_pZ_std_grid = \
                self.vbar_TZ_std_grid = 0.*self.counts_grid

            else:
                self.counts_grid,\
                self.vbar_R1_dat_grid, self.vbar_p1_dat_grid,\
                self.vbar_T1_dat_grid, self.vbar_Z1_dat_grid,\
                self.vbar_RR_dat_grid, self.vbar_pp_dat_grid,\
                self.vbar_TT_dat_grid, self.vbar_ZZ_dat_grid,\
                self.vbar_Rp_dat_grid, self.vbar_RT_dat_grid,\
                self.vbar_RZ_dat_grid, self.vbar_pZ_dat_grid,\
                self.vbar_TZ_dat_grid = self.data_mean_grids

                self.counts_std_grid,\
                self.vbar_R1_std_grid, self.vbar_p1_std_grid,\
                self.vbar_T1_std_grid, self.vbar_Z1_std_grid,\
                self.vbar_RR_std_grid, self.vbar_pp_std_grid,\
                self.vbar_TT_std_grid, self.vbar_ZZ_std_grid,\
                self.vbar_Rp_std_grid, self.vbar_RT_std_grid,\
                self.vbar_RZ_std_grid, self.vbar_pZ_std_grid,\
                self.vbar_TZ_std_grid = self.data_std_total_grids

            # Calculate tracer density
            self.nu_dat_grid = self.counts_grid/self.bin_vol_grid
            self.nu_std_grid = self.counts_std_grid/self.bin_vol_grid

            # Process Vertex Deviation
            all_vertex_dev_vectors = np.ma.masked_where(np.isnan(all_vertex_dev_vectors), all_vertex_dev_vectors)

            self.median_vertex_dev_vector = np.median(all_vertex_dev_vectors, axis=0).reshape(grid_shape[1:])
            self.mean_vertex_dev_vector = np.mean(all_vertex_dev_vectors, axis=0).reshape(grid_shape[1:])

            self.vertex_dev_3sig_lower = np.percentile(all_vertex_dev_vectors, 100*0.0015, axis=0).reshape(grid_shape[1:])
            self.vertex_dev_2sig_lower = np.percentile(all_vertex_dev_vectors, 100*0.0225, axis=0).reshape(grid_shape[1:])
            self.vertex_dev_1sig_lower = np.percentile(all_vertex_dev_vectors, 100*0.158, axis=0).reshape(grid_shape[1:])
            self.vertex_dev_1sig_upper = np.percentile(all_vertex_dev_vectors, 100*0.8415, axis=0).reshape(grid_shape[1:])
            self.vertex_dev_2sig_upper = np.percentile(all_vertex_dev_vectors, 100*0.9775, axis=0).reshape(grid_shape[1:])
            self.vertex_dev_3sig_upper = np.percentile(all_vertex_dev_vectors, 100*0.9985, axis=0).reshape(grid_shape[1:])

            # Build dictionary then save to dataframe
            dictionary = {'data_mean' : self.data_mean,
                            'data_cov': self.data_cov,
                            'data_corr' : self.data_corr,
                            'data_var_from_cov' : self.data_var_from_cov,
                            'data_var_avg_from_samples' : self.data_var_avg_from_samples,
                            'data_std_total' : self.data_std_total,
                            'data_mean_grids' : self.data_mean_grids,
                            'data_std_total_grids': self.data_std_total_grids,
                            'skewness_stat_grids' : self.skewness_stat_grids,
                            'skewness_pval_grids' : self.skewness_pval_grids,
                            'kurtosis_stat_grids' : self.kurtosis_stat_grids,
                            'kurtosis_pval_grids' : self.kurtosis_pval_grids,
                            'gaussianity_stat_grids' : self.gaussianity_stat_grids,
                            'gaussianity_pval_grids' : self.gaussianity_pval_grids,
                            'R_data_coords_mesh' : self.R_data_coords_mesh,
                            'phi_data_coords_mesh' : self.phi_data_coords_mesh,
                            'Z_data_coords_mesh' : self.Z_data_coords_mesh,
                            'R_edges_mesh' : self.R_edges_mesh,
                            'phi_edges_mesh' : self.phi_edges_mesh,
                            'Z_edges_mesh' : self.Z_edges_mesh,
                            'counts_grid' : self.counts_grid,
                            'nu_dat_grid' : self.nu_dat_grid,
                            'vbar_R1_dat_grid' : self.vbar_R1_dat_grid,
                            'vbar_p1_dat_grid' : self.vbar_p1_dat_grid,
                            'vbar_T1_dat_grid' : self.vbar_T1_dat_grid,
                            'vbar_Z1_dat_grid' : self.vbar_Z1_dat_grid,
                            'vbar_RR_dat_grid' : self.vbar_RR_dat_grid,
                            'vbar_pp_dat_grid' : self.vbar_pp_dat_grid,
                            'vbar_TT_dat_grid' : self.vbar_TT_dat_grid,
                            'vbar_ZZ_dat_grid' : self.vbar_ZZ_dat_grid,
                            'vbar_Rp_dat_grid' : self.vbar_Rp_dat_grid,
                            'vbar_RT_dat_grid' : self.vbar_RT_dat_grid,
                            'vbar_RZ_dat_grid' : self.vbar_RZ_dat_grid,
                            'vbar_pZ_dat_grid' : self.vbar_pZ_dat_grid,
                            'vbar_TZ_dat_grid' : self.vbar_TZ_dat_grid,
                            'counts_std_grid' : self.counts_std_grid,
                            'nu_std_grid' : self.nu_std_grid,
                            'vbar_R1_std_grid' : self.vbar_R1_std_grid,
                            'vbar_p1_std_grid' : self.vbar_p1_std_grid,
                            'vbar_T1_std_grid' : self.vbar_T1_std_grid,
                            'vbar_Z1_std_grid' : self.vbar_Z1_std_grid,
                            'vbar_RR_std_grid' : self.vbar_RR_std_grid,
                            'vbar_pp_std_grid' : self.vbar_pp_std_grid,
                            'vbar_TT_std_grid' : self.vbar_TT_std_grid,
                            'vbar_ZZ_std_grid' : self.vbar_ZZ_std_grid,
                            'vbar_Rp_std_grid' : self.vbar_Rp_std_grid,
                            'vbar_RT_std_grid' : self.vbar_RT_std_grid,
                            'vbar_RZ_std_grid' : self.vbar_RZ_std_grid,
                            'vbar_pZ_std_grid' : self.vbar_pZ_std_grid,
                            'vbar_TZ_std_grid' : self.vbar_TZ_std_grid,
                            'median_vertex_dev_vector' : self.median_vertex_dev_vector,
                            'mean_vertex_dev_vector' : self.mean_vertex_dev_vector,
                            'vertex_dev_3sig_lower' : self.vertex_dev_3sig_lower ,
                            'vertex_dev_2sig_lower' : self.vertex_dev_2sig_lower ,
                            'vertex_dev_1sig_lower' : self.vertex_dev_1sig_lower ,
                            'vertex_dev_1sig_upper' : self.vertex_dev_1sig_upper,
                            'vertex_dev_2sig_upper' : self.vertex_dev_2sig_upper,
                            'vertex_dev_3sig_upper' : self.vertex_dev_3sig_upper
                            }

            cache_dataframe = pd.Series(dictionary)
            cache_dataframe.to_pickle(data_root + '/oscar_cache_files/' + cache_file_name)
