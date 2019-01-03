import pandas as pd
import pdb
import numpy as np
import numpy.random as rand

N_samplings = 10

#Import Astrometric Data
pdb.set_trace()
data_folder = '/Users/hsilver/Physics_Projects/Astrometric_Data/Gaia_DR2_subsamples/'
data_file = 'gaiaDR2_6D_test_sample_100k-result.csv'
datab = pd.read_csv(data_folder + data_file) #astrometric_data_table

#Construct Means and Covarriance Matrices
astrometric_means = np.array([datab['ra'].values,
                        datab['dec'].values,
                        datab['parallax'].values,
                        datab['pmra'].values,
                        datab['pmdec'].values,
                        datab['radial_velocity'].values]).T
Nstars = datab['ra'].values.shape[0]
Nzeros = np.zeros(Nstars)
astrometric_covariances = np.array([[datab['ra_error'].values**2,
                                    datab['ra_dec_corr'].values * datab['ra_error'].values * datab['dec_error'].values,
                                    datab['ra_parallax_corr'].values * datab['ra_error'].values * datab['parallax_error'].values,
                                    datab['ra_pmra_corr'].values * datab['ra_error'].values * datab['pmra_error'].values,
                                    datab['ra_pmdec_corr'].values * datab['ra_error'].values * datab['pmdec_error'].values, Nzeros],
                                    [Nzeros, datab['dec_error'].values**2,
                                    datab['dec_parallax_corr'].values * datab['dec_error'].values * datab['parallax_error'].values,
                                    datab['dec_pmra_corr'].values * datab['dec_error'].values * datab['pmra_error'].values,
                                    datab['dec_pmdec_corr'].values * datab['dec_error'].values * datab['pmdec_error'].values,Nzeros],
                                    [Nzeros, Nzeros, datab['parallax_error'].values**2,
                                    datab['parallax_pmra_corr'].values * datab['parallax_error'].values * datab['pmra_error'].values,
                                    datab['parallax_pmdec_corr'].values * datab['parallax_error'].values * datab['pmdec_error'].values,Nzeros],
                                    [Nzeros,Nzeros,Nzeros, datab['pmra_error'].values**2,
                                    datab['pmra_pmdec_corr'].values * datab['pmra_error'].values * datab['pmdec_error'].values,Nzeros],
                                    [Nzeros, Nzeros, Nzeros, Nzeros, datab['pmdec_error'].values**2,Nzeros],
                                    [Nzeros, Nzeros, Nzeros, Nzeros, Nzeros, datab['radial_velocity_error'].values**2]])

astrometric_covariances = np.transpose(astrometric_covariances, (2,0,1)) #Rearrange
astrometric_covariances = np.array([astrometric_covariances[ii] + astrometric_covariances[ii].T - \
                                np.diagonal(astrometric_covariances[ii])*np.identity(6) \
                                for ii in range(Nstars)]) #Symmetrize

#Sample from multivariate Gaussian
#for jj in range(N_samplings):
sample_jj = np.array([rand.multivariate_normal(astrometric_means[ii], astrometric_covariances[ii]) for ii in range(Nstars)])


pdb.set_trace()

print('Oscar the Grouch... Out')
