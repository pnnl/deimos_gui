
import pandas as pd
import numpy as np
import panel as pn

import dask.dataframe as dd
import hvplot.xarray  # noqa: API import
import hvplot.dask  # noqa: API import
import hvplot.pandas
import deimos
import logging
import colorcet as cc
import datashader as ds
from holoviews.operation.datashader import aggregate, datashade, rasterize
import os
from pathlib import Path

def exception_handler(ex):
    logging.error("Error", exc_info=ex)
    pn.state.notifications.error('Error: %s: see command line for more information' % str(ex), duration=0)

pn.extension(exception_handler=exception_handler, notifications=True)

def load_mz_h5(file_name_initial, key, columns, rt_name=None, dt_name=None, new_name = None):
        '''
        Load either mz, h5 or csv file

                Args:
                        file_name_initial (path): file path to data
                        key (str): key for uploaded data, such as ms1 or ms2
                        columns (list): list of the feature names to return from file
                        rt_name (list): name retention time accession if using mzML file
                        dt_name (list): name drift time accession if using mzML file
                Returns:
                        pd DataFrame with data 
        '''
        extension = Path(file_name_initial).suffix
        if extension == ".mzML" or extension == ".gz":
                 if os.path.exists(new_name):
                        pn.state.notifications.info("Using existing h5 file: " + new_name )
                        pn.state.notifications.info("If you wish to create a new file, rename or delete " + new_name )
                        return deimos.load(new_name, key=key)
                 else:
                        rt_name_value = deimos.get_accessions(file_name_initial)[rt_name]
                        dt_name_value = deimos.get_accessions(file_name_initial)[dt_name]
                        pn.state.notifications.info("load deimos mz using " + str({rt_name: rt_name_value, dt_name: dt_name_value}), duration=0)
                        pn.state.notifications.info("loading an mz file will take a while, will see 'done loading' when finished", duration=0)
                        pn.state.notifications.info("See https://deimos.readthedocs.io/en/latest/user_guide/loading_saving.html to convert with DEIMoS directly", duration=0)
                        load_file = deimos.load(file_name_initial, accession={'retention_time': rt_name_value, 'drift_time': dt_name_value})
                        
                        # Save ms1 to new file, use x so don't overwrite existing file
                        deimos.save(new_name, load_file['ms1'], key='ms1', mode='w')

                        # Save ms2 to same file
                        deimos.save(new_name, load_file['ms2'], key='ms2', mode='a')
                        pn.state.notifications.info("saving as h5 file in " + str(new_name))
                        pn.state.notifications.info("done loading", duration=0)
                        return load_file[key]
        elif extension in [".hdf5", ".hdf", ".h5"]:
                return deimos.load(file_name_initial, key=key, columns=columns)
        elif extension ==".csv":
                return pd.read_csv(file_name_initial)
        else:
             if extension == "":
                     extension = "Folder"
             raise Exception(extension + " used. Please only use h5, hdf, mzML, or mzML.gz files")
           
def load_initial_deimos_data(file_name_initial, feature_dt, feature_rt, feature_mz, feature_intensity, rt_name, dt_name,  new_name = None,  key= 'ms1'):
        '''
        Full function to return dataframe with load_mz_h5

                Args:
                        file_name_initial (path): file path to data
                        feature_dt (str): drift time name
                        feature_rt (str): retention time name
                        feature_mz (str): mz name
                        feature_intensity (str): intensity name
                        key (str): key for uploaded data, such as ms1 or ms2
                        rt_name (list): name retention time accession if using mzML file
                        dt_name (list): name drift time accession if using mzML file
                Returns:
                        pd DataFrame with data 

        '''
        if Path(file_name_initial).stem == 'placeholder':     
                raise Exception("Select files and adjust Args before clicking 'Rerun'")
        full_data_1 = load_mz_h5(file_name_initial, key=key, columns=[feature_mz, feature_dt, feature_rt, feature_intensity], rt_name = rt_name, dt_name = dt_name, new_name = new_name)
        full_data_1 = full_data_1[[feature_dt, feature_rt, feature_mz, feature_intensity]]
        full_data_1.reset_index(drop = True, inplace=True)
        return full_data_1

        
def create_smooth(file_name_initial, feature_mz, feature_dt, feature_rt, feature_intensity, smooth_radius, smooth_iterations, new_smooth_name, rt_name, dt_name, pre_threshold =128):
        '''
        Get the smooth data

                Args:
                        file_name_initial (path): file path to data
                        feature_dt (str): drift time name
                        feature_rt (str): retention time name
                        feature_mz (str): mz name
                        feature_intensity (str): intensity name
                        threshold_slider (int): threshold data with this value
                        radius (float or list): Radius of the sparse filter in each dimension. Values less than
                        zero indicate no connectivity in that dimension.
                        iterations (int): Maximum number of smoothing iterations to perform.
                        new_smooth_name (str): name of new smooth data
                        rt_name (list): name retention time accession if using mzML file
                        dt_name (list): name drift time accession if using mzML file
                        pre_threshold (float): pre-threshold value
                Returns:
                        pd DataFrame with data 

        '''

        ms1 = load_mz_h5(file_name_initial, key='ms1', columns=[feature_mz, feature_dt, feature_rt, feature_intensity], rt_name = rt_name, dt_name = dt_name)
        ms2 = load_mz_h5(file_name_initial, key='ms2', columns=[feature_mz, feature_dt, feature_rt, feature_intensity], rt_name = rt_name, dt_name = dt_name)

        factors = deimos.build_factors(ms1, dims='detect')
                
        # Nominal threshold
        ms1 = deimos.threshold(ms1, threshold=pre_threshold)
        # Build index
        index_ms1_peaks = deimos.build_index(ms1, factors)
        # Smooth data
        smooth_radius= [int(i) for i in list(smooth_radius.split('-'))]
        iterations = int(smooth_iterations)
        pn.state.notifications.info('Smooth MS1 data', duration=3000)
        ms1_smooth = deimos.filters.smooth(ms1, index=index_ms1_peaks, dims=[feature_mz, feature_dt, feature_rt],
                                radius=smooth_radius, iterations=iterations)
        if len(ms1_smooth) == 0:
                raise Exception("No smooth ms1 data created. Perhaps there isn't enough data in the initial files. Go to https://deimos.readthedocs.io/en/latest/getting_started/example_data.html for example data.")
           
        ## save with date and time because user won't reuse. 
        deimos.save(new_smooth_name, ms1_smooth, key='ms1', mode='w')

                # append peak ms2
        factors = deimos.build_factors(ms2, dims='detect')

        pn.state.notifications.info('Smooth MS2 data', duration=3000)
        # Nominal threshold
        ms2 = deimos.threshold(ms2, threshold=pre_threshold)
        # Build index
        index_ms2_peaks = deimos.build_index(ms2, factors)

        # Smooth data
        iterations = int(smooth_iterations)
        # Smooth data
        ms2_smooth = deimos.filters.smooth(ms2, index=index_ms2_peaks, dims=[feature_mz, feature_dt, feature_rt],
                                radius=smooth_radius, iterations=iterations)
        if len(ms2_smooth) == 0:
                raise Exception("No smooth ms2 data created. Perhaps there isn't enough data in the initial files. Go to https://deimos.readthedocs.io/en/latest/getting_started/example_data.html for example data")
        ## save with date and time because user won't reuse. 
        deimos.save(new_smooth_name, ms2_smooth, key='ms2', mode='a')
        return ms1_smooth, index_ms1_peaks, index_ms2_peaks

def create_peak(file_name_smooth, feature_mz, feature_dt, feature_rt, feature_intensity,  threshold_slider,  peak_radius, index_ms1_peaks, index_ms2_peaks, new_peak_name, rt_name = None, dt_name = None, pre_threshold=128 ):
        '''
        Get the smooth data

                Args:
                        file_name_smooth (path): file path to data
                        feature_dt (str): drift time name
                        feature_rt (str): retention time name
                        feature_mz (str): mz name
                        feature_intensity (str): intensity name
                        threshold_slider (int): threshold data with this value
                        index_ms1_peaks (dict) Index of features in original data array.
                        index_ms2_peaks (dict) Index of features in original data array.
                        peak_radius (float, list, or None) If specified, radius of the sparse weighted mean filter in each dimension.
                        Values less than one indicate no connectivity in that dimension.
                        new_peak_name (str): name of new peak data
                        rt_name (list): name retention time accession if using mzML file
                        dt_name (list): name drift time accession if using mzML file
                        pre_threshold (float): pre-threshold value
                Returns:
                        pd DataFrame with data 
        '''
        if os.path.exists(new_peak_name):
                raise Exception(new_peak_name + " already exists. Please rename before continuing or use the existing file name in the smooth file name")
                
        else:
                ms1_smooth = load_mz_h5(file_name_smooth, key='ms1', columns=[feature_mz, feature_dt, feature_rt, feature_intensity], rt_name = rt_name, dt_name = dt_name)
                ms2_smooth = load_mz_h5(file_name_smooth, key='ms2', columns=[feature_mz, feature_dt, feature_rt, feature_intensity], rt_name = rt_name, dt_name = dt_name)


                peak_radius= [int(i) for i in list(peak_radius.split('-'))]

                # Perform peak detection
                ms1_peaks = deimos.peakpick.persistent_homology(deimos.threshold(ms1_smooth,  threshold = pre_threshold),  index=index_ms1_peaks,
                                                                dims=[feature_mz, feature_dt, feature_rt],
                                                                radius=peak_radius)
                # Sort by persistence
                ms1_peaks = ms1_peaks.sort_values(by='persistence', ascending=False).reset_index(drop=True)
                # Save ms1 to new file
                ms1_peaks = deimos.threshold(ms1_peaks, by='persistence', threshold=int(threshold_slider))
                ms1_peaks = deimos.threshold(ms1_peaks, by='intensity', threshold=int(threshold_slider))
                if len(ms1_peaks) == 0:
                        raise Exception("No peak ms1 data created. Perhaps there isn't enough data in the initial files. Go to https://deimos.readthedocs.io/en/latest/getting_started/example_data.html for example data")
                deimos.save(new_peak_name, ms1_peaks, key='ms1', mode='w')


                # Perform peak detection
                ms2_peaks = deimos.peakpick.persistent_homology(deimos.threshold(ms2_smooth,  threshold = pre_threshold), index=index_ms2_peaks,
                                                                dims=[feature_mz, feature_dt, feature_rt],
                                                                radius=peak_radius)
                
                ms2_peaks = deimos.threshold(ms2_peaks, by='persistence', threshold=int(threshold_slider))
                ms2_peaks = deimos.threshold(ms2_peaks, by='intensity', threshold=int(threshold_slider))
                # Sort by persistence
                ms2_peaks = ms2_peaks.sort_values(by='persistence', ascending=False).reset_index(drop=True)
                # update list of options in file selections
                if len(ms2_peaks) == 0:
                        raise Exception("No peak ms2 data created. Perhaps there isn't enough data in the initial files. Go to https://deimos.readthedocs.io/en/latest/getting_started/example_data.html for example data")
                
                # Save ms2 to new file with _new_peak_data.h5 suffix
                deimos.save(new_peak_name, ms2_peaks, key='ms2', mode='a')
                return ms1_peaks

def align_peak_create(full_ref, theshold_presistence, feature_mz, feature_dt, feature_rt, feature_intensity, pre_threshold):
        '''
        Get the smooth data

                Args:
                        full_ref (path): file path to data
                        threshold_presistence (int): initial threshold presistence
                        feature_dt (str): drift time name
                        feature_rt (str): retention time name
                        feature_mz (str): mz name
                        feature_intensity (str): intensity name
                        pre_threshold (float): pre-threshold value
                Returns:
                        pd DataFrame with data 
        '''
        peak_ref  = deimos.peakpick.persistent_homology(deimos.threshold(full_ref,  threshold = pre_threshold),
                                                dims=[feature_mz, feature_dt, feature_rt])
        peak_ref = deimos.threshold(peak_ref, by='persistence', threshold=theshold_presistence)
        peak_ref = deimos.threshold(peak_ref, by='intensity', threshold=theshold_presistence)
        return peak_ref

def offset_correction_model(dt_ms2, mz_ms2, mz_ms1, ce=0,
                            params=[1.02067031, -0.02062323,  0.00176694]):
    '''Function to correct putative pairs while running the deconvolution function'''
    # Cast params as array
    params = np.array(params).reshape(-1, 1)
    
    # Convert collision energy to array
    ce = np.ones_like(dt_ms2) * np.log(ce)
    
    # Create constant vector
    const = np.ones_like(dt_ms2)
    
    # Sqrt
    mu_ms1 = np.sqrt(mz_ms1)
    mu_ms2 = np.sqrt(mz_ms2)
    
    # Ratio
    mu_ratio = mu_ms2 / mu_ms1
    
    # Create dependent array
    x = np.stack((const, mu_ratio, ce), axis=1)
    
    # Predict
    y = np.dot(x, params).flatten() * dt_ms2
    
    return y

# function that adjusts the x_range and y_range with either x y stream or user input
# this work-arround is necessary due to the x_range and y_range streams being set to none with new data
# if not set to none, old data ranges from previous data are still used
# when range is set to none, need to explicitely set the x and y range of the of the rasterization level and plot limits, else it will rasterize over whole image: 
# trying to avoid this bug https://github.com/holoviz/holoviews/issues/4396
# keep an eye on this, as likely will fix in datashader, which would make this work around unnecessary

def rasterize_plot(
element,
feature_intensity,
x_filter=None,
y_filter=None,
x_spacing=0,
y_spacing=0,
):

        '''
        Get rasterized plot

                Args:
                        element: graph object
                        feature_intensity (str): intensity value
                        x_filter (tuple): x range
                        y_filter (tuple): y range
                        x_spacing (flt): min size of grids
                        y_spacing (flt): min size of grids
                Returns:
                        pd DataFrame with data 
        '''
        # dynmaic false to allow the x_range and y_range to be adjusted by either
        #xy stream or manual filter rather than automaically
        rasterize_plot = rasterize(
                element,
                width=800,
                height=600,
                # summing here by intensity, rather than using the deimos.collapse and summing by intensity 
                # per feature_dt & feature_rt group, and interpolating into a heatmap
                aggregator=ds.sum(feature_intensity),
                x_sampling=x_spacing,
                y_sampling=y_spacing,
                x_range=x_filter,
                y_range=y_filter,
                dynamic=False,
        )
        ropts = dict(
                tools=["hover"],
                default_tools=[],
                colorbar=True,
                colorbar_position="bottom",
                cmap=cc.blues,
                cnorm='eq_hist')
        # actually changes the x and y limits seen, not just the rasterization levels
        if y_filter != None and x_filter != None:
                rasterize_plot.apply.opts(
                xlim=x_filter, ylim=y_filter, framewise=True, **ropts
                )
        else:
                rasterize_plot.apply.opts(framewise=True, **ropts)
        return rasterize_plot

def new_name_if_mz(mz_file_name):
        extension = Path(mz_file_name).suffix
        if extension == ".mzML" or extension == ".mzml" or extension == ".gz":
                new_name = os.path.join("created_data",  Path(mz_file_name).stem + '.h5')
        else:
                new_name = None
        return new_name

def aligment(two_matched, ref_matched, two_matched_aligned, dim, kernel, parameter_inputs ):
        '''
        Save the aligned data

                Args:
                        two_matched (df): dataframe to be aligned
                        ref_matched
                        two_matched_aligned: dataframe output with aligned dimensions
                        dim: dimensions to align
                        kernel: the kernel to use for alignment,
                        parameter_inputs: str to use in saving csv files
                Returns:
                        xy_drift_retention_time: pd DataFrame aligned dataframe 
                        matchtable reference line
        '''
        spl = deimos.alignment.fit_spline( two_matched, ref_matched, align= dim, kernel=kernel, C=1000)
        newx = np.linspace(0, max(ref_matched[ dim].max(), two_matched[ dim].max()), 1000)
        two_matched_aligned["aligned_" + dim] = spl(two_matched_aligned[dim])
        # save by peak in peak name
        two_matched_aligned.to_csv(os.path.join("created_data", parameter_inputs + "_aligned.csv"))
                # match_table includes the matched data from data a and b to compare with scatter plot (data a retention time vs data b retention time)
        matchtable = pd.concat( [
                two_matched[[ dim]].reset_index(drop=True),
                ref_matched[[ dim]].reset_index(drop=True)], axis=1,)
        matchtable.columns = ['match_a_' + dim, 'match_b_' + dim]
        xy_drift_retention_time = pd.DataFrame(
                np.hstack( ( newx[:, None], # spline applied to matching
                        spl(newx)[:, None],)))
        # rename columns
        xy_drift_retention_time.columns = ['x_' + dim, 'y_' + dim]

        xy_drift_retention_time.to_csv(os.path.join("created_data", parameter_inputs + "_xy_drift_retention_time.csv"))
        matchtable.to_csv(os.path.join("created_data", parameter_inputs + "_matchtable.csv"))
        return xy_drift_retention_time, matchtable

def get_peak_file(file_name_initial, feature_dt, feature_rt, feature_mz, feature_intensity, rt_name, dt_name,  theshold_presistence,  key= 'ms1', pre_threshold= 128):

        '''
        load the peak files for alignment

                Args:
                        file_name_initial (path): file path to data
                        feature_dt (str): drift time name
                        feature_rt (str): retention time name
                        feature_mz (str): mz name
                        feature_intensity (str): intensity name
                        key (str): key for uploaded data, such as ms1 or ms2
                        rt_name (list): name retention time accession if using mzML file
                        dt_name (list): name drift time accession if using mzML file
                        threshold: int: minimum value of intensity
                Returns:
                        pd DataFrame with peak data, name of new h5 if created from mzML file
        '''
        try:
                # load initial refence file
                new_name = new_name_if_mz(file_name_initial)
                peak_ref_initial = load_initial_deimos_data(file_name_initial, feature_dt, feature_rt, feature_mz, feature_intensity, rt_name, dt_name, key = key, new_name = new_name)
                
                peak_ref =align_peak_create(peak_ref_initial, theshold_presistence,feature_mz,  feature_dt, feature_rt, feature_intensity, pre_threshold)
        except Exception as e:
                raise Exception(str(e))
        # if thesholding makes the files have lenght 0, bring up exception
        if len(peak_ref) == 0: 
                raise Exception("No data left after thresholding: lower threshold or change data")
        return peak_ref, new_name


def decon_ms2(ms1_peaks, ms1, ms2_peaks, ms2, feature_mz, feature_dt, feature_rt, require_ms1_greater_than_ms2, drift_score_min):

        '''
        return ms2 decon values

                Args:
                        ms1_peaks: data file of ms1 peaks
                        ms1: data file of ms1 data
                        ms2_peaks: data file of ms2 peaks
                        ms2: data file of ms2 data
                        feature_dt (str): drift time name
                        feature_rt (str): retention time name
                        feature_mz (str): mz name
                        require_ms1_greater_than_ms2 (boolean): ms1 must be greater than ms2 when constructing putative pairs
                        drift_score_min (boolean): only keep drift score greater than .9
                Returns:
                        pd DataFrame with data 
        '''       

        decon = deimos.deconvolution.MS2Deconvolution(ms1_peaks, ms1, ms2_peaks, ms2)
        # use false .loc[res['drift_time_score']
        decon.construct_putative_pairs(dims=[feature_dt, feature_rt],
                        low=[-0.12, -0.1], high=[1.4, 0.1], ce=20,
                        model=offset_correction_model,
                        require_ms1_greater_than_ms2=require_ms1_greater_than_ms2,
                        error_tolerance=0.12)
        
        decon.configure_profile_extraction(dims=[feature_mz, feature_dt, feature_rt],
                                low=[-200E-6, -0.05, -0.1],
                                high=[600E-6, 0.05, 0.1],
                                relative=[True, True, False])

        res = decon.apply(dims=feature_dt, resolution=0.01)
        if drift_score_min:
                res = res.loc[res[feature_dt + '_score'] > .9].groupby(by=[x for x in res.columns if x.endswith('_ms1')],
                                                as_index=False).agg(list)
        else: 
        
                res = res.groupby(by=[x for x in res.columns if x.endswith('_ms1')],
                                                as_index=False).agg(list)
        
        
        return res