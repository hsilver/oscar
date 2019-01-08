import pandas as pd
import pdb
import numpy as np
import numpy.random as rand
import time
import scipy.stats as stats


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

def astrometric_to_galactocentric(ra, dec, para, pm_ra, pm_dec, vr,
    Rsun, phisun, Zsun, vRsun,vYsun,vZsun, epoch_T):
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

def binning(Rg_vec, phig_vec, Zg_vec, vRg_vec, vTg_vec, vZg_vec,
            phi_limits, R_edges, Z_edges):

    vphig_vec = vTg_vec/Rg_vec #rad/s

    """
    Rg_vec, star_data_gccyl[0]
    phig_vec, star_data_gccyl[1]
    Zg_vec, star_data_gccyl[2]

    vRg_vec, star_V_gccyl[0]
    vphig_vec, star_V_gccyl[1]
    vZg_vec, star_V_gccyl[2]
    """

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

    return counts_grid, \
            vbar_R1_dat_grid, vbar_R1_std_grid,\
            vbar_p1_dat_grid, vbar_p1_std_grid,\
            vbar_Z1_dat_grid, vbar_Z1_std_grid,\
            vbar_RR_dat_grid, vbar_RR_std_grid,\
            vbar_pp_dat_grid, vbar_pp_std_grid,\
            vbar_ZZ_dat_grid, vbar_ZZ_std_grid,\
            vbar_RZ_dat_grid, vbar_RZ_std_grid




# Set Constants and Parameters
N_samplings = 10
deg_to_rad = np.pi/180
mas_to_rad = (np.pi/6.48E8)
maspyr_to_radps = np.pi/(6.48E8 * 31557600)

#Import Astrometric Data
data_folder = '/Users/hsilver/Physics_Projects/Astrometric_Data/Gaia_DR2_subsamples/'
data_file = 'gaiaDR2_6D_test_sample_100k-result.csv'
datab = pd.read_csv(data_folder + data_file) #astrometric_data_table

#Construct Means and Covarriance Matrices
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

#Sample from multivariate Gaussian
for jj in range(N_samplings):

    start = time.time()
    sample_jj = np.array([rand.multivariate_normal(astrometric_means[ii], astrometric_covariances[ii]) for ii in range(Nstars)])
    print('Sampling takes ', time.time()-start, ' s')

    solar_posvel = np.array([8200.,0.,100., 14.,238.,5.])
    epoch_T = calc_epoch_T('J2000')

    start = time.time()
    Rg_vec, phig_vec, Zg_vec, vRg_vec, vTg_vec, vZg_vec = astrometric_to_galactocentric(
                                sample_jj[:,0], sample_jj[:,1], sample_jj[:,2],
                                sample_jj[:,3], sample_jj[:,4], sample_jj[:,5],
                                solar_posvel[0], solar_posvel[1], solar_posvel[2],
                                solar_posvel[3], solar_posvel[4], solar_posvel[5],
                                epoch_T)
    print('Transformation takes ', time.time()-start, ' s')

    phi_limit = [0,2*np.pi]
    R_edges = np.linspace(5000,10000,10)
    Z_edges = np.linspace(-2000,2000,9)

    start = time.time()
    counts_grid, \
    vbar_R1_dat_grid, vbar_R1_std_grid,\
    vbar_p1_dat_grid, vbar_p1_std_grid,\
    vbar_Z1_dat_grid, vbar_Z1_std_grid,\
    vbar_RR_dat_grid, vbar_RR_std_grid,\
    vbar_pp_dat_grid, vbar_pp_std_grid,\
    vbar_ZZ_dat_grid, vbar_ZZ_std_grid,\
    vbar_RZ_dat_grid, vbar_RZ_std_grid \
        = binning(Rg_vec, phig_vec, Zg_vec, vRg_vec, vTg_vec, vZg_vec,
                    phi_limit, R_edges, Z_edges)
    print('Binning takes ', time.time()-start, ' s')




pdb.set_trace()
print('Oscar the Grouch... Out')


# T= sc.dot(sc.array([[sc.cos(theta),sc.sin(theta),0.],[sc.sin(theta),-sc.cos(theta),0.],[0.,0.,1.]]),
#             sc.dot(sc.array([[-sc.sin(dec_ngp),0.,sc.cos(dec_ngp)],[0.,1.,0.],[sc.cos(dec_ngp),0.,sc.sin(dec_ngp)]]),sc.array([[sc.cos(ra_ngp),sc.sin(ra_ngp),0.],[-sc.sin(ra_ngp),sc.cos(ra_ngp),0.],[0.,0.,1.]])))
