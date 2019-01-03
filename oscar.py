import pandas as pd
import pdb
import numpy as np


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
astrometric_covariance = np.array([[datab['ra_error'].values**2,
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

astrometric_covariance = np.transpose(astrometric_covariance, (2,0,1)) #Rearrange
astrometric_covariance = np.array([astrometric_covariance[ii] + astrometric_covariance[ii].T - \
                                np.diagonal(astrometric_covariance[ii])*np.identity(6) \
                                for ii in range(0,Nstars)]) #Symmetrize

#Sample from multivariate Gaussian



print('Oscar the Grouch... Out')
