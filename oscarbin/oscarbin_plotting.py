import pandas as pd
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os


def plot_RZ_heatmap(R_data_coords_mesh, Z_data_coords_mesh,
                    R_edges_mesh, Z_edges_mesh, data_grid,
                    file_name,
                    fig_height = 9, fig_width = 13, colormap = 'magma',
                    Norm = 'linear', vmin=None, vmax=None,
                    linthresh = None, linscale = None,
                    ylabel = 'Z [pc]', xlabel = 'R [pc]',
                    cb_label = ' ', counts_mask = True):
    fig, axes = plt.subplots(ncols=2, nrows=1, gridspec_kw={"width_ratios":[15,1]})
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    #plt.subplots_adjust(wspace=wspace_double_cbax)
    ax = axes[0] #Plot
    cbax = axes[1] #Colorbar
    ax.set_aspect('equal')

    data_grid_masked = np.ma.masked_where(np.logical_not(counts_mask), data_grid)

    if Norm == 'lognorm':
        im = ax.pcolormesh(R_edges_mesh, Z_edges_mesh, data_grid_masked,
                        cmap = colormap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    elif Norm == 'symlognorm':
        im = ax.pcolormesh(R_edges_mesh, Z_edges_mesh, data_grid_masked,
                        cmap = colormap,
                        norm=colors.SymLogNorm(vmin=vmin, vmax=vmax,
                                                linthresh=linthresh,
                                                linscale=linscale))
    elif Norm== 'linear':
        im = ax.pcolormesh(R_edges_mesh, Z_edges_mesh, data_grid_masked,
                        cmap = colormap, vmin=vmin, vmax=vmax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    cb = fig.colorbar(im, cax=cbax)
    cb.set_label(label=cb_label)
    plt.savefig(file_name)
    return

def plot_RZ_heatmap_and_lines(R_data_coords_mesh, Z_data_coords_mesh,
                    R_edges_mesh, Z_edges_mesh,
                    data_grid, data_error_upper, data_error_lower,
                    file_name, fig_height = 10, fig_width = 25, colormap = 'magma',
                    Norm = 'linear', vmin=None, vmax=None,
                    linthresh = None, linscale = None,
                    ylabel = 'Z [pc]', xlabel = 'R [pc]',
                    cb_label = ' ', counts_mask=True):

    num_R_bins = len(R_data_coords_mesh[:,0])
    # width_ratios_vector = [14,1]
    # for ii in range(num_R_bins):
    #     width_ratios_vector.append(2)

    fig, axes = plt.subplots(ncols=2, nrows=2,
                gridspec_kw={"height_ratios":[10,1],"width_ratios":[15,50]})
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    plt.subplots_adjust(wspace=0.1)
    heat_ax = axes[0,0]  # Heatmap
    cbax = axes[1,0]     # Colorbar
    line_ax = axes[0,1]  # Line plot
    scale_ax = axes[1,1] # Scale for line plot
    scale_ax.get_shared_x_axes().join(line_ax, scale_ax)
    line_ax.get_shared_y_axes().join(line_ax, heat_ax)

    data_grid_masked = np.ma.masked_where(np.logical_not(counts_mask), data_grid)
    data_error_upper_masked = np.ma.masked_where(np.logical_not(counts_mask), data_error_upper)
    data_error_lower_masked = np.ma.masked_where(np.logical_not(counts_mask), data_error_lower)

    # Heat Map
    if Norm == 'lognorm':
        im = heat_ax.pcolormesh(R_edges_mesh, Z_edges_mesh, data_grid_masked,
                        cmap = colormap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    elif Norm == 'symlognorm':
        im = heat_ax.pcolormesh(R_edges_mesh, Z_edges_mesh, data_grid_masked,
                        cmap = colormap,
                        norm=colors.SymLogNorm(vmin=vmin, vmax=vmax,
                                                linthresh=linthresh,
                                                linscale=linscale))
    elif Norm== 'linear':
        im = heat_ax.pcolormesh(R_edges_mesh, Z_edges_mesh, data_grid_masked,
                        cmap = colormap, vmin=vmin, vmax=vmax)
    heat_ax.set_ylabel(ylabel)
    heat_ax.set_xlabel(xlabel)

    # Colorbar
    cb = fig.colorbar(im, cax=cbax, orientation = 'horizontal')
    cb.set_label(label=cb_label)

    # Line plot
    spacing_param = 1
    scaling_param = 1.5/np.nanmax(data_grid_masked)

    min_data_and_err = min(0.,np.nanmin(data_grid_masked), np.nanmin(data_grid_masked+data_error_lower_masked))
    max_data_and_err = max(0.,np.nanmax(data_grid_masked), np.nanmin(data_grid_masked+data_error_upper_masked))

    line_ax.axhline(0, xmin=min_data_and_err*scaling_param,
                        xmax=max_data_and_err*scaling_param,
                        ls='-')

    for RR, Rval in enumerate(R_data_coords_mesh[:,0]):
        zero_point = RR * spacing_param

        Z_values = Z_data_coords_mesh[RR,:]
        data_values = data_grid_masked[RR,:]
        data_upper = data_error_upper_masked[RR,:]
        data_lower = data_error_lower_masked[RR,:]


        #line_ax.set_aspect('equal')
        main_line = line_ax.plot(data_values*scaling_param + zero_point, Z_values, ls='-', linewidth=2)
        line_color = main_line[0].get_color()
        line_ax.axvline(zero_point,ls='-', color = line_color)
        line_ax.plot((data_values+data_upper)*scaling_param + zero_point, Z_values, ls='-',
                        color = line_color, alpha=0.5, linewidth=1)
        line_ax.plot((data_values-data_lower)*scaling_param + zero_point, Z_values, ls='-',
                        color = line_color, alpha=0.5, linewidth=1)
        line_ax.fill_betweenx(Z_values,(data_values-data_lower)*scaling_param + zero_point,
                                (data_values+data_upper)*scaling_param + zero_point,
                                color=line_color, alpha=0.1)

        line_ax.spines['top'].set_visible(False)
        line_ax.spines['left'].set_visible(False)
        line_ax.spines['bottom'].set_visible(False)
        line_ax.spines['right'].set_visible(False)
        line_ax.xaxis.set_ticks_position('none')
        line_ax.yaxis.set_ticks_position('none')
        line_ax.set_xticks(range(len(R_data_coords_mesh[:,0])))
        line_ax.xaxis.set_ticklabels(R_data_coords_mesh[:,0],
                            rotation='vertical')
        line_ax.xaxis.set_label_coords(0.08, -0.05)
        line_ax.set_xlabel(xlabel + u"\u2192")
        line_ax.yaxis.set_ticklabels([])


    # Scale
    scale_ax.yaxis.set_ticklabels([])
    scale_ax.xaxis.set_ticklabels([])
    #scale_ax.xaxis.set_ticks_position('none')
    scale_ax.yaxis.set_ticks_position('none')
    #scale_ax.set_xlabel(cb_label)
    scale_ax.spines['top'].set_visible(False)
    scale_ax.spines['left'].set_visible(False)
    scale_ax.spines['right'].set_visible(False)
    scale_ax.spines['bottom'].set_visible(False)
    scale_ax.set_ylim([-1,1])

    scale_max = 10**int(round(np.log10(np.nanmax(data_grid_masked))))
    scale_ax.plot((0,scale_max*scaling_param), (0,0), ls='-', lw=2)
    scale_ax.scatter(np.array([0,scale_max*scaling_param]), np.array([0,0]), marker='+',s=70)
    scale_ax.xaxis.set_ticklabels([0,scale_max])
    scale_ax.set_xticks([0,scale_max*scaling_param])
    scale_ax.xaxis.set_label_coords(0.25, -0.5)
    scale_ax.set_xlabel(cb_label)

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
    plt.close()



def plot_RZ_histograms(oscar_obj, counts_threshold = 50, phi_slice = 0):
    #Set up file structure
    plot_folder = oscar_obj.data_root + oscar_obj.data_file_name.split('.')[0] + \
                    '/samplings_' + str(oscar_obj.N_samplings) + \
                    '_Rlim_' + str(oscar_obj.Rmin) + '_' + str(oscar_obj.Rmax) + \
                    '_philim_' + str(oscar_obj.phimin) + '_' + str(oscar_obj.phimax) + \
                    '_Zlim_' + str(oscar_obj.Zmin) + '_' + str(oscar_obj.Zmax) + \
                    '_Rbins_' + str(oscar_obj.num_R_bins) + \
                    '_phibins_' + str(oscar_obj.num_phi_bins) + \
                    '_Zbins_' + str(oscar_obj.num_Z_bins) + \
                    oscar_obj.binning_type + '/'
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    # PLOT RESULTS
    skewness_stat_counts_grid,\
    skewness_stat_vbar_R1_dat_grid, skewness_stat_vbar_p1_dat_grid,\
    skewness_stat_vbar_Z1_dat_grid, skewness_stat_vbar_RR_dat_grid,\
    skewness_stat_vbar_pp_dat_grid, skewness_stat_vbar_ZZ_dat_grid,\
    skewness_stat_vbar_RZ_dat_grid = oscar_obj.skewness_stat_grids

    skewness_pval_counts_grid,\
    skewness_pval_vbar_R1_dat_grid, skewness_pval_vbar_p1_dat_grid,\
    skewness_pval_vbar_Z1_dat_grid, skewness_pval_vbar_RR_dat_grid,\
    skewness_pval_vbar_pp_dat_grid, skewness_pval_vbar_ZZ_dat_grid,\
    skewness_pval_vbar_RZ_dat_grid = oscar_obj.skewness_pval_grids

    kurtosis_stat_counts_grid,\
    kurtosis_stat_vbar_R1_dat_grid, kurtosis_stat_vbar_p1_dat_grid,\
    kurtosis_stat_vbar_Z1_dat_grid, kurtosis_stat_vbar_RR_dat_grid,\
    kurtosis_stat_vbar_pp_dat_grid, kurtosis_stat_vbar_ZZ_dat_grid,\
    kurtosis_stat_vbar_RZ_dat_grid = oscar_obj.kurtosis_stat_grids

    kurtosis_pval_counts_grid,\
    kurtosis_pval_vbar_R1_dat_grid, kurtosis_pval_vbar_p1_dat_grid,\
    kurtosis_pval_vbar_Z1_dat_grid, kurtosis_pval_vbar_RR_dat_grid,\
    kurtosis_pval_vbar_pp_dat_grid, kurtosis_pval_vbar_ZZ_dat_grid,\
    kurtosis_pval_vbar_RZ_dat_grid = oscar_obj.kurtosis_pval_grids

    gaussianity_stat_counts_grid,\
    gaussianity_stat_vbar_R1_dat_grid, gaussianity_stat_vbar_p1_dat_grid,\
    gaussianity_stat_vbar_Z1_dat_grid, gaussianity_stat_vbar_RR_dat_grid,\
    gaussianity_stat_vbar_pp_dat_grid, gaussianity_stat_vbar_ZZ_dat_grid,\
    gaussianity_stat_vbar_RZ_dat_grid = oscar_obj.gaussianity_stat_grids

    gaussianity_pval_counts_grid,\
    gaussianity_pval_vbar_R1_dat_grid, gaussianity_pval_vbar_p1_dat_grid,\
    gaussianity_pval_vbar_Z1_dat_grid, gaussianity_pval_vbar_RR_dat_grid,\
    gaussianity_pval_vbar_pp_dat_grid, gaussianity_pval_vbar_ZZ_dat_grid,\
    gaussianity_pval_vbar_RZ_dat_grid = oscar_obj.gaussianity_pval_grids

    # TRACER DENSITY
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], oscar_obj.nu_dat_grid[:,phi_slice,:],
                    plot_folder + 'nu_data.pdf', colormap = 'magma',
                    Norm = 'lognorm', cb_label='Tracer density stars [stars pc$^{-3}$]')

    masked_cmap = plt.cm.viridis
    masked_cmap.set_bad(color='grey')
    masked_counts = np.ma.masked_where(oscar_obj.counts_grid[:,phi_slice,:] < counts_threshold, oscar_obj.counts_grid[:,phi_slice,:])
    counts_above_threshold = oscar_obj.counts_grid[:,phi_slice,:] >= counts_threshold
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:],
                    masked_counts,
                    plot_folder + 'nu_data_pure_counts.pdf', colormap = masked_cmap,
                    Norm = 'lognorm', vmin=50.,
                    cb_label='Star count [stars per bin]',
                    counts_mask = True)
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], gaussianity_pval_counts_grid[:,phi_slice,:],
                    plot_folder + 'nu_gauss_pval.pdf', colormap = 'magma',
                    Norm = 'lognorm', vmin=1e-2, vmax=1.,
                    cb_label='Tracer density gaussianity p-value')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], skewness_stat_counts_grid[:,phi_slice,:],
                    plot_folder + 'nu_skew_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = 'Tracer density Skewness z-score')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], kurtosis_stat_counts_grid[:,phi_slice,:],
                    plot_folder + 'nu_kurt_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = 'Tracer density kurtosis z-score')

    #Vertical Velocity vZ1
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], oscar_obj.vbar_Z1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_Z1_data.pdf', colormap = 'seismic',
                    Norm = 'linear',
                    vmin=-np.nanmax(abs(oscar_obj.vbar_Z1_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    vmax=np.nanmax(abs(oscar_obj.vbar_Z1_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    cb_label='Vertical velocity $\overline{v_Z}$ [km s$^{-1}$]',
                    counts_mask = counts_above_threshold)
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], gaussianity_pval_vbar_Z1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_Z1_gauss_pval.pdf', colormap = 'magma',
                    Norm = 'lognorm', vmin=1e-2, vmax=1.,
                    cb_label='$\overline{v_Z}$  gaussianity p-value')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], skewness_stat_vbar_Z1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_Z1_skew_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_Z}$  Skewness z-score')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], kurtosis_stat_vbar_Z1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_Z1_kurt_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_Z}$  kurtosis z-score')

    #Vertical Velocity vZZ
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], oscar_obj.vbar_ZZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_ZZ_data.pdf', colormap = 'nipy_spectral',
                    Norm = 'linear',
                    vmin=np.nanmin(oscar_obj.vbar_ZZ_dat_grid[:,phi_slice,:][counts_above_threshold]),
                    vmax=np.nanmax(oscar_obj.vbar_ZZ_dat_grid[:,phi_slice,:][counts_above_threshold]),
                    cb_label='Vertical velocity $\overline{v_Z v_Z}$ [km$^{2}$ s$^{-2}$]',
                    counts_mask = counts_above_threshold)
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], gaussianity_pval_vbar_ZZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_ZZ_gauss_pval.pdf', colormap = 'magma',
                    Norm = 'lognorm', vmin=1e-2, vmax=1.,
                    cb_label='$\overline{v_Z v_Z}$  gaussianity p-value')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], skewness_stat_vbar_ZZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_ZZ_skew_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_Z v_Z}$  skewness z-score')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], kurtosis_stat_vbar_ZZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_ZZ_kurt_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_Z v_Z}$  kurtosis z-score')

    #Radial Velocity vR
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], oscar_obj.vbar_R1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_R1_data.pdf', colormap = 'seismic',
                    Norm = 'linear',
                    vmin=-np.nanmax(abs(oscar_obj.vbar_R1_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    vmax=np.nanmax(abs(oscar_obj.vbar_R1_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    cb_label='Radial velocity $\overline{v_R}$ [km s$^{-1}$]',
                    counts_mask = counts_above_threshold)
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], gaussianity_pval_vbar_R1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_R1_gauss_pval.pdf', colormap = 'magma',
                    Norm = 'lognorm', vmin=1e-2, vmax=1.,
                    cb_label='$\overline{v_R}$  gaussianity p-value')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], skewness_stat_vbar_R1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_R1_skew_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_R}$  Skewness z-score')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], kurtosis_stat_vbar_R1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_R1_kurt_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_R}$  kurtosis z-score')

    #Radial Velocity vRR
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], oscar_obj.vbar_RR_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RR_data.pdf', colormap = 'nipy_spectral',
                    Norm = 'linear',
                    vmin=np.nanmin(oscar_obj.vbar_RR_dat_grid[:,phi_slice,:][counts_above_threshold]),
                    vmax=np.nanmax(oscar_obj.vbar_RR_dat_grid[:,phi_slice,:][counts_above_threshold]),
                    cb_label='Radial velocity $\overline{v_R v_R}$ [km$^{2}$ s$^{-2}$]',
                    counts_mask = counts_above_threshold)
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], gaussianity_pval_vbar_RR_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RR_gauss_pval.pdf', colormap = 'magma',
                    Norm = 'lognorm', vmin=1e-2, vmax=1.,
                    cb_label='$\overline{v_R v_R}$  gaussianity p-value')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], skewness_stat_vbar_RR_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RR_skew_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_R v_R}$  Skewness z-score')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], kurtosis_stat_vbar_RR_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RR_kurt_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_R v_R}$  kurtosis z-score')

    #Tangential Velocity vp
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], oscar_obj.vbar_p1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_p1_data.pdf', colormap = 'magma',
                    Norm = 'linear',
                    vmin=np.nanmin(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:][counts_above_threshold]),
                    vmax=np.nanmax(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:][counts_above_threshold]),
                    cb_label='Angular Velocity $\overline{v_\phi}$ [rad s$^{-1}$]',
                    counts_mask = counts_above_threshold)
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], gaussianity_pval_vbar_p1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_p1_gauss_pval.pdf', colormap = 'magma',
                    Norm = 'lognorm', vmin=1e-2, vmax=1.,
                    cb_label='$\overline{v_p}$  gaussianity p-value')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], skewness_stat_vbar_p1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_p1_skew_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_p}$  Skewness z-score')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], kurtosis_stat_vbar_p1_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_p1_kurt_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_p}$  kurtosis z-score')

    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:],
                    oscar_obj.vbar_p1_dat_grid[:,phi_slice,:]*oscar_obj.R_data_coords_mesh[:,phi_slice,:]*3.086E1, #picorad/s *
                    plot_folder + 'vbar_T1_data.pdf', colormap = 'nipy_spectral',
                    Norm = 'linear',
                    vmin=np.nanmin((oscar_obj.vbar_p1_dat_grid[:,phi_slice,:]*oscar_obj.R_data_coords_mesh[:,phi_slice,:]*3.086E1)[counts_above_threshold]),
                    vmax=np.nanmax((oscar_obj.vbar_p1_dat_grid[:,phi_slice,:]*oscar_obj.R_data_coords_mesh[:,phi_slice,:]*3.086E1)[counts_above_threshold]),
                    cb_label='Tangential velocity $\overline{v_p}$ [km s$^{-1}$]',
                    counts_mask = counts_above_threshold)




    #Tilt Term vRvZ
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RZ_data.pdf', colormap = 'seismic',
                    Norm = 'symlognorm',
                    vmin=-np.nanmax(abs(oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    vmax=np.nanmax(abs(oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    linthresh = 200, linscale = 1.0,
                    cb_label='RZ velocity cross term $\overline{v_R v_Z}$ [km$^{2}$ s$^{-2}$]',
                    counts_mask = counts_above_threshold)
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], gaussianity_pval_vbar_RZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RZ_gauss_pval.pdf', colormap = 'magma',
                    Norm = 'lognorm', vmin=1e-2, vmax=1.,
                    cb_label='$\overline{v_R v_Z}$  gaussianity p-value')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], skewness_stat_vbar_RZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RZ_skew_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_R v_Z}$  Skewness z-score')
    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:], kurtosis_stat_vbar_RZ_dat_grid[:,phi_slice,:],
                    plot_folder + 'vbar_RZ_kurt_stat.pdf', colormap = 'magma',
                    Norm = 'linear', vmin=0., vmax=1.,
                    cb_label = '$\overline{v_R v_Z}$  kurtosis z-score')

    plot_RZ_heatmap(oscar_obj.R_data_coords_mesh[:,phi_slice,:], oscar_obj.Z_data_coords_mesh[:,phi_slice,:],
                    oscar_obj.R_edges_mesh[:,phi_slice,:], oscar_obj.Z_edges_mesh[:,phi_slice,:],
                    oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:] - oscar_obj.vbar_R1_dat_grid[:,phi_slice,:]*oscar_obj.vbar_Z1_dat_grid[:,phi_slice,:],
                    plot_folder + 'sigma_RZ_data.pdf', colormap = 'seismic',
                    Norm = 'symlognorm',
                    vmin=-np.nanmax(abs(oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    vmax=np.nanmax(abs(oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:][counts_above_threshold])),
                    linthresh = 200, linscale = 1.0,
                    cb_label='RZ velocity cross term $\sigma_{RZ} = \overline{v_R v_Z} \
                                - \overline{v_R}\,\overline{v_Z}$ [km$^{2}$ s$^{-2}$]',
                    counts_mask = counts_above_threshold)











def plot_correlation_matrix(oscar_obj):
    # Total Correlation Matrix
    plot_matrix_heatmap(oscar_obj.data_corr, 'correlation_matrix_all.pdf')

    #Counts correlations
    block_size = len(oscar_obj.R_data_coords_mesh.flatten())
    file_name_vec = ['correlation_matrix_counts.pdf',
                        'correlation_matrix_vbar_R1.pdf',
                        'correlation_matrix_vbar_p1.pdf',
                        'correlation_matrix_vbar_Z1.pdf',
                        'correlation_matrix_vbar_RR.pdf',
                        'correlation_matrix_vbar_pp.pdf',
                        'correlation_matrix_vbar_ZZ.pdf',
                        'correlation_matrix_vbar_RZ.pdf']
    for NN in range(0,8):
        plot_matrix_heatmap(oscar_obj.data_corr[NN*block_size:(NN+1)*block_size,
                                            NN*block_size:(NN+1)*block_size],
                            file_name_vec[NN],colormap='seismic')




    # plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #             oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #             oscar_obj.nu_dat_grid[:,phi_slice,:], oscar_obj.nu_std_grid[:,phi_slice,:], oscar_obj.nu_std_grid[:,phi_slice,:],
    #             plot_folder + 'nu_data_w_line.pdf',colormap = 'magma',
    #             Norm = 'lognorm', cb_label='Tracer density stars [stars pc$^{-3}$]')
    #
    # plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #                     oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #                     oscar_obj.vbar_Z1_dat_grid[:,phi_slice,:], oscar_obj.vbar_Z1_std_grid[:,phi_slice,:],oscar_obj.vbar_Z1_std_grid[:,phi_slice,:],
    #                     plot_folder + 'vbar_Z1_data_w_line.pdf', colormap = 'seismic',
    #                     Norm = 'linear',
    #                     vmin=-np.nanmax(abs(oscar_obj.vbar_Z1_dat_grid[:,phi_slice,:][counts_above_threshold])),
    #                     vmax=np.nanmax(abs(oscar_obj.vbar_Z1_dat_grid[:,phi_slice,:][counts_above_threshold])),
    #                     cb_label='Vertical velocity $\overline{v_Z}$ [km s$^{-1}$]',
    #                     counts_mask = counts_above_threshold)
    # plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #                 oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #                 oscar_obj.vbar_ZZ_dat_grid[:,phi_slice,:], oscar_obj.vbar_ZZ_std_grid[:,phi_slice,:],oscar_obj.vbar_ZZ_std_grid[:,phi_slice,:],
    #                 plot_folder + 'vbar_ZZ_data_w_line.pdf', colormap = 'nipy_spectral',
    #                 Norm = 'linear',
    #                 vmin=np.nanmin(oscar_obj.vbar_ZZ_dat_grid[:,phi_slice,:][counts_above_threshold]),
    #                 vmax=np.nanmax(oscar_obj.vbar_ZZ_dat_grid[:,phi_slice,:][counts_above_threshold]),
    #                 cb_label='Vertical velocity $\overline{v_Z v_Z}$ [km$^{2}$ s$^{-2}$]',
    #                 counts_mask = counts_above_threshold)
    #
    # plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #                 oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #                 oscar_obj.vbar_R1_dat_grid[:,phi_slice,:], oscar_obj.vbar_R1_std_grid[:,phi_slice,:],oscar_obj.vbar_R1_std_grid[:,phi_slice,:],
    #                 plot_folder + 'vbar_R1_data_w_line.pdf', colormap = 'seismic',
    #                 Norm = 'linear',
    #                 vmin=-np.nanmax(abs(oscar_obj.vbar_R1_dat_grid[:,phi_slice,:][counts_above_threshold])),
    #                 vmax=np.nanmax(abs(oscar_obj.vbar_R1_dat_grid[:,phi_slice,:][counts_above_threshold])),
    #                 cb_label='Radial velocity $\overline{v_R}$ [km s$^{-1}$]',
    #                 counts_mask = counts_above_threshold)
    #
    #     plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #                     oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #                     oscar_obj.vbar_RR_dat_grid[:,phi_slice,:], oscar_obj.vbar_RR_std_grid[:,phi_slice,:],oscar_obj.vbar_RR_std_grid[:,phi_slice,:],
    #                     plot_folder + 'vbar_RR_data_w_line.pdf', colormap = 'nipy_spectral',
    #                     Norm = 'linear',
    #                     vmin=np.nanmin(oscar_obj.vbar_RR_dat_grid[:,phi_slice,:][counts_above_threshold]),
    #                     vmax=np.nanmax(oscar_obj.vbar_RR_dat_grid[:,phi_slice,:][counts_above_threshold]),
    #                     cb_label='Radial velocity $\overline{v_R v_R}$ [km$^{2}$ s$^{-2}$]',
    #                     counts_mask = counts_above_threshold)
    #
    #
    # plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #                 oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #                 oscar_obj.vbar_p1_dat_grid[:,phi_slice,:], oscar_obj.vbar_p1_std_grid[:,phi_slice,:],oscar_obj.vbar_p1_std_grid[:,phi_slice,:],
    #                 plot_folder + 'vbar_p1_data_w_line.pdf', colormap = 'magma',
    #                 Norm = 'linear',
    #                 vmin=np.nanmin(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:][counts_above_threshold]),
    #                 vmax=np.nanmax(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:][counts_above_threshold]),
    #                 cb_label='Angular Velocity $\overline{v_\phi}$ [rad s$^{-1}$]',
    #                 counts_mask = counts_above_threshold)
    #
    # plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #                 oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #                 oscar_obj.vbar_p1_dat_grid[:,phi_slice,:]*oscar_obj.R_data_coords_mesh[:,phi_slice,:]*3.086E1,
    #                 oscar_obj.vbar_p1_std_grid[:,phi_slice,:]*oscar_obj.R_data_coords_mesh[:,phi_slice,:]*3.086E1,
    #                 oscar_obj.vbar_p1_std_grid[:,phi_slice,:]*oscar_obj.R_data_coords_mesh[:,phi_slice,:]*3.086E1,
    #                 plot_folder + 'vbar_T1_data_w_line.pdf', colormap = 'nipy_spectral',
    #                 Norm = 'linear',
    #                 vmin=None,#np.amin(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:][~np.isnan(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:])]),
    #                 vmax=None,#np.amax(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:][~np.isnan(oscar_obj.vbar_p1_dat_grid[:,phi_slice,:])]),
    #                 cb_label='Tangential velocity $\overline{v_p}$ [km s$^{-1}$]',
    #                 counts_mask = counts_above_threshold)
    #
    #     plot_RZ_heatmap_and_lines(oscar_obj.R_data_coords_mesh, oscar_obj.Z_data_coords_mesh,
    #                     oscar_obj.R_edges_mesh, oscar_obj.Z_edges_mesh,
    #                     oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:], oscar_obj.vbar_RZ_std_grid[:,phi_slice,:],oscar_obj.vbar_RZ_std_grid[:,phi_slice,:],
    #                     plot_folder + 'vbar_RZ_data_w_line.pdf', colormap = 'seismic',
    #                     Norm = 'symlognorm',
    #                     vmin=-np.nanmax(abs(oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:][counts_above_threshold])),
    #                     vmax=np.nanmax(abs(oscar_obj.vbar_RZ_dat_grid[:,phi_slice,:][counts_above_threshold])),
    #                     linthresh = 200, linscale = 1.0,
    #                     cb_label='RZ velocity cross term $\overline{v_R v_Z}$ [km$^{2}$ s$^{-2}$]',
    #                     counts_mask = counts_above_threshold)
