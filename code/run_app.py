
import pandas as pd
import holoviews as hv
import numpy as np
import panel as pn
import math
import ast
import colorcet as cc
import dask.dataframe as dd
import hvplot.xarray  # noqa: API import
import hvplot.dask  # noqa: API import
import hvplot.pandas
from holoviews import opts
from holoviews.operation.datashader import aggregate, datashade, rasterize
from datashader.colors import Sets1to3
import deimos
import multiprocessing as mp
import os, param as pm, holoviews as hv, panel as pn, datashader as ds
import logging
from pathlib import Path
import additional_functions as additional_functions
from datetime import datetime
# from pathlib import PurePath, PureWindowsPath
#from pyinstrument import Profiler



file_name_initial_name = "placeholder.csv"  #example_data.h5
file_name_smooth_name = "placeholder.csv"
file_name_peak_name = "placeholder.csv"
calibration_input_name = "placeholder.csv" #cal_input.csv
example_tune_file_name = "placeholder.csv" #example_tune_pos.h5
file_to_calibrate_name = "placeholder.csv" #example_tune_pos.h5 
peak_ref_name = "placeholder.csv" #example_alignment.h5

# file_name_initial_name = "example_data.h5"  #example_data.h5
# file_name_smooth_name = "placeholder.csv"
# file_name_peak_name = "placeholder.csv"
# file_name_smooth_name = "example_data_smooth_radius_0-1-0_smooth_iterations_3_feature_rt_retention_time_new_smooth_data.h5" 
# file_name_peak_name = "example_data_threshold_1000_peak_radius_2-10-0_feature_rt_retention_time_new_peak_data.h5"
# calibration_input_name = "cal_input.csv"
# example_tune_file_name = "example_tune_pos.h5" #"example_tune_pos.h5"
# file_to_calibrate_name = "example_tune_pos.h5" #"example_tune_pos.h5"
# peak_ref_name = "example_alignment.h5" #"example_alignment.h5"

hv.extension('bokeh', 'matplotlib')

# view general exception 
def exception_handler(ex):
    '''Return the error value to the user and stop running the code'''

    pn.state.notifications.position = 'top-right'
    logging.error("Error", exc_info=ex)
    pn.state.notifications.error('Error: %s: see command line for more information' % str(ex), duration=0)

pn.extension(exception_handler=exception_handler, notifications=True)


# linked selection places constants on all dimensions at once
# linked brushing supports both the box_select and lasso_select tools.
ls = hv.link_selections.instance()
ls2 = hv.link_selections.instance()
ls3 = hv.link_selections.instance()
ls4 = hv.link_selections.instance()

hv.output(backend='bokeh')

class Deimos_app(pm.Parameterized):
    '''Class to create a parameterized functions that only changes when paramaters are updated'''
    file_name_initial = pm.FileSelector(default = os.path.join("data", file_name_initial_name), path="data/*",  doc='Initial File in .h5, .mzML, or .mzML.gz format. Default: example_data.h5', label='Initial Data Default: example_data.h5')
    file_folder_initial =  pm.String(
        default= "data", doc='Please use forward slashes / and starting from / if absolute. Data folder (use / at the end of the file).', label='Data folder (use /)')
    file_folder_cal =  pm.String(
        default= "data", doc='Please use forward slashes / and starting from / if absolute ', label='Data folder (use /)')
    rt_mzML_name = pm.Selector(["scan start time"], doc='Select one of the columns within the mzML file. Only adjust if mz file selected. Select the retention time column name')
    dt_mzML_name = pm.Selector(["ion mobility drift time"], doc='Select one of the columsn within the mzML file. Only adjust if mz file selected. Select the drift time column name')
    # reset the manual filters to the data bounds and reset the rangexy of the plot
    reset_filter = pm.Action(
        lambda x: x.param.trigger('reset_filter'), doc = 'Refresh axis ranges to data min and max (will not update plot). Select (re)run to redo plot',
        label='Update axis ranges below')
  
    reset_filter_iso = pm.Action(
        lambda x: x.param.trigger('reset_filter_iso'), doc = 'Refresh axis ranges to data min and max (will not update plot). Select (re)run to redo plot',
        label='Update axis ranges below')
    
    feature_dt = pm.Selector(default='drift_time', objects = ["drift_time", 'retention_time', 'mz'], label="Drift Time", doc="This should be the name of one feature (drift time is the default) in the data. Change if data is using different column value")
    feature_rt = pm.Selector(default='retention_time', objects = ["drift_time", 'retention_time', 'mz'], label="Retention Time", doc="This should be the name of one feature (retention time is the default) in the data. Change if data is using different column value")
    feature_mz = pm.Selector(default='mz', objects = ["drift_time", 'retention_time', 'mz'], label="mz", doc="This should be the name of one feature (mz is the default) in the data. Change if data is using different column value")

    feature_intensity = pm.String(default = 'intensity', label='Intensity Feature', doc="This the value that will be summed and visualized in the plots. Change if data is using different column value")

    # manual filter bounds of plots
    feature_dt_axis_width = pm.Range(bounds=(0.0, 200.0),  step=0.1, label="Axis width: " + feature_dt.default, doc='Only clicking Recreate plot will adjust the plots. Reset by clicking Update Axis Range to data ranges')
    feature_rt_axis_width = pm.Range(bounds=(0.0, 200.0), step=0.1, label="Axis width: " + feature_rt.default, doc='Only clicking Recreate plot will adjust the plots. Reset by clicking Update Axis Range to data ranges')
    feature_mz_axis_width = pm.Range(bounds=(0.0, 200.0), step=0.1, label="Axis width: " + feature_mz.default, doc='Only clicking Recreate plot will adjust the plots. Reset by clicking Update Axis Range to data ranges')

    # set the min spacing for all the dimensions for rasterizing
    min_feature_dt_bin_size = pm.Number(default=0.2,bounds=(0,None),  label="Min bin size: " + feature_dt.default, doc= 'The grid-size will never be smaller than this if using width input to zoom in. Only clicking Recreate plot will adjust the plots')
    min_feature_rt_bin_size = pm.Number(default=0.2, bounds=(0,None), label="Min bin size: " + feature_rt.default, doc= 'The grid-size will never be smaller than this if using width input to zoom in. Only clicking Recreate plot will adjust the plots')
    min_feature_mz_bin_size = pm.Number(default=0.02,bounds=(0,None),  label="Min bin size: " + feature_mz.default, doc= 'The grid-size will never be smaller than this if using width input to zoom in. Only clicking Recreate plot will adjust the plots')

    # manual filter centers for the graphs in isotopes (smaller range than the full_plot, so don't want use the same bounds)
    
    feature_dt_axis_width_iso = pm.Range(bounds=(0.0, 200.0), step=0.1, label="Axis width: " + feature_dt.default, doc='Only clicking Recreate plot will adjust the plots. Reset by clicking Update Axis Range to data ranges')
    feature_rt_axis_width_iso = pm.Range(bounds=(0.0, 200.0), step=0.1, label="Axis width: " + feature_rt.default, doc='Only clicking Recreate plot will adjust the plots. Reset by clicking Update Axis Range to data ranges')
    feature_mz_axis_width_iso = pm.Range(bounds=(0.0, 200.0), step=0.1, label="Axis width: " + feature_mz.default, doc='Only clicking Recreate plot will adjust the plots. Reset by clicking Update Axis Range to data ranges')

    # set the min spacing for all the dimensions for isotopes
    min_feature_dt_bin_size_iso = pm.Number(default=0.2, bounds=(0,None), label="Min bin size: " + feature_dt.default, doc= 'The grid-size will never be smaller than this if using width input to zoom in. Only clicking Recreate plot will adjust the plots')
    min_feature_rt_bin_size_iso = pm.Number(default=0.2, bounds=(0,None), label="Min bin size: " + feature_rt.default, doc= 'The grid-size will never be smaller than this if using width input to zoom in. Only clicking Recreate plot will adjust the plots')
    min_feature_mz_bin_size_iso = pm.Number(default=0.02, bounds=(0,None), label="Min bin size: " + feature_mz.default, doc= 'The grid-size will never be smaller than this if using width input to zoom in. Only clicking Recreate plot will adjust the plots')

    min_feature_rt_spacing = pm.Number(default=1.5, bounds=(0,None), label="Spacing: " + feature_rt.default, doc= "Check for MS2 within the range of the location clicked the MS1 plus and minus this value")
    min_feature_dt_spacing = pm.Number(default=1.5, bounds=(0,None), label="Spacing: " + feature_dt.default, doc= "Check for MS2 within the range of the location clicked the MS1 plus and minus this value")
    min_feature_mz_spacing = pm.Number(default=20, bounds=(0,None), label="Spacing: " + feature_mz.default, doc= "Check for MS2 within the range of the location clicked the MS1 plus and minus this value")

    file_name_smooth = pm.FileSelector(default = os.path.join("created_data", file_name_smooth_name),\
                                        path="created_data/*",  doc='Automatically updated with new file name after created. View in created folder. File in .h5, .mzML, or .mzML.gz format.', label='Smooth Data (in Created_Data Folder)')
    file_name_peak = pm.FileSelector(default = os.path.join("created_data", file_name_peak_name), \
                                     path="created_data/*",  doc='Automatically updated with new file name after created. View in created folder. File in .h5, .mzML, or .mzML.gz format.', label='Peak Data (in Created_Data Folder)')
    ##TODO this is actually a lower theshold than originally in the paper - need to update the time
    threshold_slider = pm.Integer(default=1000, label='Threshold', doc= 'Filter the files to only keep peaks above this intensity')
    threshold_slider_ms1_ms2 = pm.Integer(default=100, label='Min Threshold for MS1', doc= 'Filter the files to only keep peaks above this intensity for MS1')
    smooth_radius = pm.String(
        default='0-1-0', doc='Keep - between numbers. Best practice is to increase number of iterations', label='Smoothing radius by mz, drift time, and retention time, repectively, to use when smoothing')
    smooth_iterations = pm.String(
        default='3', doc='Best practice is to increase number of iterations by mz, drift time, and retention time', label='Number of smoothing iterations')
    peak_radius = pm.String(
        default='2-10-0', doc='Keep - between numbers. A radius per dimension by mz, drift time, and retention time', label='Weighted mean kernel size, respectively, for mz, drift time, and retention time')
    #
    view_plot = pm.Action(lambda x: x.param.trigger('view_plot'), doc="Click to view new file", label='Click to rerun after changing inputs')
    
    rerun_peak = pm.Action(lambda x: x.param.trigger('rerun_peak'), doc="Click to rerun after changing inputs", label='(Re)Run peak')
    rerun_smooth = pm.Action(lambda x: x.param.trigger('rerun_smooth'), doc="Click to rerun after changing inputs", label='(Re)Run smooth')
    rerun_decon = pm.Action(lambda x: x.param.trigger('rerun_decon'), doc="Click to rerun after changing inputs", label='(Re)Run deconvolution')
    rerun_iso = pm.Action(lambda x: x.param.trigger('rerun_iso'), doc="Click to rerun after changing inputs", label='(Re)Run ison')
    rerun_calibrate = pm.Action(lambda x: x.param.trigger('rerun_calibrate'), doc="Click to rerun after changing inputs", label='(Re)Run calibrate')
    Recreate_plots_with_below_values = pm.Action(lambda x: x.param.trigger('Recreate_plots_with_below_values'), doc="Set axis ranges to ranges below")
    Recreate_plots_with_below_values_iso = pm.Action(lambda x: x.param.trigger('Recreate_plots_with_below_values_iso'), doc="Set axis ranges to ranges below")

    remove_notifications = pm.Action(lambda x: x.param.trigger('remove_notifications'), doc="Remove all notifications", label='Remove all notifications')

    # set the min spacing for all the dimensions for rasterizing 
    slice_distance_dt = pm.Number(default=0.2, bounds=(0,None), label="Slice isotopes drift time", doc = "Add distance to selected isotope drift time to get plot range")
    slice_distance_rt = pm.Number(default=0.2, bounds=(0,None), label="Slice isotopes retention time", doc = "Add distance to selected isotope retention time to get plot range")
    slice_distance_mz = pm.Number(default=5, bounds=(0,None), label="Slice isotopes mz left", doc = "Add distance to selected isotope mz to get plot range")

    calibration_input = pm.FileSelector(default = os.path.join("data", calibration_input_name), path="data/*",  doc='Calibrate input file. Data must include mz, ccs, charge, and, if not tune mix, a ta column. File in .h5, .mzML, .mzML.gz or csv format. Default: cal_input.csv', label='Calibration Input. Default: cal_input.csv')
    example_tune_file = pm.FileSelector(default = os.path.join("data", example_tune_file_name), path="data/*",  doc='Example tune file. File in .h5, .mzML, .mzML.gz or csv format. Default: example_tune_pos.h5', label='Example Tune Data, Default: example_tune_pos.h5')
    file_to_calibrate = pm.FileSelector(default = os.path.join("data", file_to_calibrate_name), path="data/*",  doc='Input that will be calibrated. Data must include mz, ta, and q values. File in .h5, .mzML, .mzML.gz or csv format. Default: example_tune_pos.h5', label='File to Calib. Default: example_tune_pos.h5')
    beta = pm.String(default = "0.12991516042484708", label='beta', doc ="Only necessary if selected fix_parameters")
    tfix = pm.String(default = "-0.03528247661068562", label='tfix', doc ="Only necessary if selected fix_parameters")
    traveling_wave = pm.Boolean(False, label='traveling_wave', doc="If true, then using travelling wave IMS, where the relationship between measurement and CCS will linearized by the natural logarithm")
    calibrate_type = pm.Selector(default = "load_all_values", objects=["load_all_values", "use_tunemix","fix_parameters"], doc= "Calibrate using one of possible functions, see user guide for more information")

    @pn.depends("remove_notifications", watch=True)
    def remove_not(self):
        '''Remove all notifications'''
        pn.state.notifications.clear()
                
    @pn.depends("file_folder_initial", watch=True)
    def update_param(self, new_name = None):
        '''With new file folder update the files available in file selector files'''
        # update all files if updating file folder
        # convert to posix
        if not os.path.isdir(self.file_folder_initial):
            pn.state.notifications.error('Folder does not exist', duration=0)
        if self.file_folder_initial[-1] == '/':
            self.param.file_name_initial.path = self.file_folder_initial + "*"
        else:
            self.param.file_name_initial.path = self.file_folder_initial + "/*"

        self.param.file_name_initial.update()
        if new_name != None:
            self.file_name_initial = new_name
        else:
            if self.file_name_initial not in self.param.file_name_initial.objects:
                self.file_name_initial = self.param.file_name_initial.objects[0]
            else:
                pass
                

    @pn.depends("file_folder_cal", watch=True)
    def update_param_cal(self):
        '''With new file folder update the files available in file selector files for calibration'''
        
        if not os.path.isdir(self.file_folder_cal):
            pn.state.notifications.error('Folder does not exist', duration=0)
        # update all files if updating file folder
        if self.calibration_input[-1] == '/':
            path = self.file_folder_cal + "*"
        else:
            path = self.file_folder_cal + "/*"
        
        self.param.calibration_input.path = path
        self.param.example_tune_file.path = path
        self.param.file_to_calibrate.path = path

        self.param.calibration_input.update()
        self.param.example_tune_file.update()
        self.param.file_to_calibrate.update()

        if self.calibration_input not in self.param.calibration_input.objects:
            self.calibration_input = self.param.calibration_input.objects[0]

        if self.example_tune_file not in self.param.example_tune_file.objects:
            self.example_tune_file = self.param.example_tune_file.objects[0]

        if self.file_to_calibrate not in self.param.file_to_calibrate.objects:
            self.file_to_calibrate = self.param.file_to_calibrate.objects[0]

    @pn.depends("file_name_initial", watch=True)
    def update_mz_accession(self):
        '''If using mzML allow users to chose name from accession file'''
        if self.file_name_initial == None:
            raise Exception("No file selected")
        extension = Path(self.file_name_initial).suffix
        if extension == ".mzML" or extension == ".mzML.gz" :
            accessin_list = list(deimos.get_accessions(self.file_name_initial).keys())
            self.param.rt_mzML_name.objects = accessin_list
            if self.rt_mzML_name not in self.param.rt_mzML_name.objects:
                self.rt_mzML_name = self.param.rt_mzML_name.objects[0]
            self.param.dt_mzML_name.objects = accessin_list
            if self.dt_mzML_name not in self.param.dt_mzML_name.objects:
                self.dt_mzML_name = self.param.dt_mzML_name.objects[0]

 # load the h5 files and load to dask             
    @pm.depends('view_plot')
    def hvdata_initial(self):
        '''Start initial data by loading data. Restart if using different file or feature names changed '''
        # don't need to recreate for sequental graphs if already created for first graphs
        # self.res will be reset to empty dataframe with rerun 
        pn.state.notifications.info('Loading initial data plots', duration=0)
        if len(self.data_initial) == 0:
            if not os.path.exists("created_data"):
                os.makedirs("created_data")
                
            self.update_mz_accession()
            # changing the label requires changing the parameter class
            self.param.feature_dt_axis_width.label = "Axis width: " + self.feature_dt
            self.param.feature_rt_axis_width.label = "Axis width: " + self.feature_rt
            self.param.feature_mz_axis_width.label = "Axis width: " + self.feature_mz
            self.param.min_feature_dt_bin_size.label = "Min bin size: " + self.feature_dt
            self.param.min_feature_rt_bin_size.label = "Min bin size: " + self.feature_rt
            self.param.min_feature_mz_bin_size.label = "Min bin size: " + self.feature_mz
            self.param.smooth_radius.label = 'Smoothing radius by ' + self.feature_mz + ', ' + self.feature_dt + ', and ' + self.feature_rt
            self.param.peak_radius.label = 'Weighted mean kernel size by ' + self.feature_mz + ', ' + self.feature_dt + ', and ' + self.feature_rt
            if self.file_name_initial == "data/placeholder.csv":
                    pn.state.notifications.info('Load data placeholder. Replace with real data', duration=10000)
                    self.data_initial = dd.from_pandas(pd.DataFrame([[0,0,0,0],[2000,200,200,4], [20,10,30,100]], columns = [self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity]), npartitions=mp.cpu_count())
            else:
                pn.state.notifications.info('In progress. Cannot make additional changes until plots update. Loading initial data ' + str(self.file_name_initial), duration=0)
                try:
                    
                    new_name = additional_functions.new_name_if_mz(self.file_name_initial)
                    full_data_1 = additional_functions.load_initial_deimos_data(self.file_name_initial, \
                                                        self.feature_dt, self.feature_rt, self.feature_mz, self.feature_intensity, rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name, new_name = new_name)
                    if new_name != None:
                        self.file_folder_initial = "created_data"
                        self.update_param(new_name)
                        pn.state.notifications.info("initial input file has changed to " + str(new_name))
                except Exception as e:
                    raise Exception(str(e))
                    
                self.data_initial = dd.from_pandas(full_data_1, npartitions=mp.cpu_count())
            self.data_initial.persist()
            self.refresh_axis_values()
            pn.state.notifications.clear()
            pn.state.notifications.info('Finished loading initial data', duration=10000)
            
        else:   
        # if the data value has already been updated from rerun, 
        #don't update again until user clicks rerun again
            pn.state.notifications.clear()
            pn.state.notifications.info('Re-trigged hvplot_initial function, loading previously loaded file', duration=10000)
            pass
        return hv.Dataset(self.data_initial)

    # resets the axis to the data's min and max
    @pm.depends('reset_filter',  watch=True)
    def refresh_axis_values(self):
        '''Let axis ranges based on the data'''
        self.reset_xy_stream()

        mz_range = (
            float(self.data_initial.mz.min().compute()),
            float(self.data_initial.mz.max().compute()) + 1.0,
        )
        #self.param sets the features within the param, while self.x sets actual value of x
        self.param.feature_mz_axis_width.bounds = mz_range
        self.feature_mz_axis_width = mz_range
        retention_range = (
            float(self.data_initial[self.feature_rt].min().compute()),
            float(self.data_initial[self.feature_rt].max().compute()) + 1.0,
        )
        self.param.feature_rt_axis_width.bounds = retention_range
        self.feature_rt_axis_width = retention_range
        drift_range = (
            float(self.data_initial[self.feature_dt].min().compute()),
            float(self.data_initial[self.feature_dt].max().compute()) + 1.0,
        )
        self.param.feature_dt_axis_width.bounds = drift_range
        self.feature_dt_axis_width = drift_range
        return
    

    def rasterize_md(
        self,
        element,
        Recreate_plots_with_below_values
    ):
        
        '''Return rasterized mz and drift retention plot
        with x and y range and x and y spacing
        Run if steam value of 
        Recreate_plots_with_below_values changes'''

        pn.state.notifications.info("Re-aggregating based on values below")
        rasterize_plot = additional_functions.rasterize_plot(
        element = element,
        feature_intensity = self.feature_intensity, 
        x_filter= self.feature_mz_axis_width,
        y_filter=self.feature_dt_axis_width,
        x_spacing=self.min_feature_dt_bin_size,
        y_spacing=self.min_feature_mz_bin_size)
        return rasterize_plot
    

    def rasterize_dr(
        self,
        element,
        Recreate_plots_with_below_values
    ):
        '''Return rasterized drift time vs retention time plot
        with x and y range and x and y spacing
        run if steam value of 
        Recreate_plots_with_below_values changes'''
        rasterize_plot = additional_functions.rasterize_plot(
        element = element,
        feature_intensity = self.feature_intensity, 
        x_filter= self.feature_dt_axis_width,
        y_filter=self.feature_rt_axis_width,
        x_spacing=self.min_feature_dt_bin_size,
        y_spacing=self.min_feature_rt_bin_size)
        return rasterize_plot
    

    def rasterize_rm(
        self,
        element,
        Recreate_plots_with_below_values
    ):
        '''Return rasterized retention time vs mz plot
        with x and y range and x and y spacing
        Run if steam value of 
        Recreate_plots_with_below_values changes'''
        rasterize_plot = additional_functions.rasterize_plot(
        element = element,
        feature_intensity = self.feature_intensity, 
        x_filter= self.feature_rt_axis_width,
        y_filter=self.feature_mz_axis_width,
        x_spacing=self.min_feature_rt_bin_size,
        y_spacing=self.min_feature_mz_bin_size)
        
        return rasterize_plot
    
        # create the hv plots 
    def hvplot_md(self, ds):
        '''Return initial points plot mz and drift retention plot'''
        element = ds.data.hvplot.points(x=self.feature_mz, y=self.feature_dt, c=self.feature_intensity)
        return element
    
    def hvplot_dr(self, ds):
        '''Return initial points drift time vs retention time plot'''
        element = ds.data.hvplot.points(x=self.feature_dt, y=self.feature_rt, c=self.feature_intensity)
        return element
    
    # create the hv plots
    def hvplot_rm(self, ds):
        '''Return initial points drift time vs retention time plot'''
        element = ds.data.hvplot.points(x=self.feature_rt, y=self.feature_mz, c=self.feature_intensity)
        return element
    
    # show plots of initial data before any smoothing, peakfinding, etc.
    @pm.depends('view_plot', watch = True)
    def initial_viewable(self, **kwargs):
        pn.state.notifications.position = 'top-right'
        '''Full function to return the initial data in three graphs'''
        #update file selector widget with new names from folder
        self.param.file_name_initial.update()
        self.param.file_name_smooth.update()
        self.param.file_name_peak.update()
        
        self.data_initial =  pd.DataFrame({'A' : []})
        # dynamic map to return hvdata after loading it with deimos - hvplot because needs to be a holoview to be returned with dynamicmap
        hvdata_full = hv.DynamicMap(self.hvdata_initial)
        # return the hvplot for mz and retentio_time
        hvplot_md_initial = hvdata_full.apply(self.hvplot_md)
        hvplot_dr_initial = hvdata_full.apply(self.hvplot_dr)
        hvplot_rm_initial = hvdata_full.apply(self.hvplot_rm)
        
        # stream to rasterize the plot. any change will cause the whole plot to reload
        stream_initial = hv.streams.Params(
            self, ['Recreate_plots_with_below_values'])
        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy stream
        
        self.rasterized_md_initial = hv.util.Dynamic(
                    hvplot_md_initial,
                    operation=self.rasterize_md,
                    streams=[stream_initial],
                )
        
        self.rasterized_dr_initial = hv.util.Dynamic(
                    hvplot_dr_initial,
                    operation=self.rasterize_dr,
                    streams=[stream_initial],
                )
        
        self.rasterized_rm_initial = hv.util.Dynamic(
                    hvplot_rm_initial,
                    operation=self.rasterize_rm,
                    streams=[stream_initial],
                )
        # profiler.stop()
        # results_file = os.path.join(TESTS_ROOT, "initial_" + str(self.file_name_initial == "placeholder.csv") + Path(self.file_name_initial).name + ".html")
        # profiler.write_html(results_file)
        return ls(self.rasterized_rm_initial  + self.rasterized_dr_initial + self.rasterized_md_initial).opts(shared_axes=True)
    

    # does not automatically reset xy axis (keeps at originally loaded, so have to do so automatically)
    @pm.depends('file_name_initial',  watch=True)
    def reset_xy_stream(self):
        '''Reset streams from hvplots so reset isn't set at original data range'''
        # plots have streams, first streams are rangexy, which
        try:
            self.rasterized_md_initial.streams[0].reset()
            self.rasterized_dr_initial.streams[0].reset()
            self.rasterized_rm_initial.streams[0].reset()
        except:
              pass
        try:
            self.rasterized_md_peak.streams[0].reset()
            self.rasterized_dr_peak.streams[0].reset()
            self.rasterized_rm_peak.streams[0].reset()
        except:
              pass
        try:
            self.rasterized_md_smooth.streams[0].reset()
            self.rasterized_dr_smooth.streams[0].reset()
            self.rasterized_rm_smooth.streams[0].reset()
        except:
              pass
        try:
            self.rasterized_md_iso.streams[0].reset()
            self.rasterized_dr_iso.streams[0].reset()
            self.rasterized_rm_iso.streams[0].reset()
        except:
              pass
        try:
            self.md_decon.streams[0].reset()
            self.dr_decon.streams[0].reset()
            self.mr_decon.streams[0].reset()
        except:
             pass

    @pm.depends('rerun_smooth')
    def create_smooth_data(self):
        
        '''Run deimos functions to get the smoothed data returned'''

        # don't need to recreate for sequental graphs if already created for first graphs
        # self.res will be reset to empty dataframe with rerun 
        if len(self.data_smooth_ms1) == 0: 
            # name will be saved as
            new_smooth_name =  os.path.join( "created_data",  Path(self.file_name_initial).stem + \
                '_smooth_radius_' + str(self.smooth_radius) +  '_smooth_iterations_' + str(self.smooth_iterations) +  "_feature_rt_" + str(self.feature_rt) +\
                   '_new_smooth_data.h5')
            if self.file_name_initial == "data/placeholder.csv":
                    self.data_smooth_ms1 = dd.from_pandas(pd.DataFrame([[0,0,0,0],[2000,200,200,4], [20,10,30,100]], columns = [self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity]), npartitions=mp.cpu_count())
            else:
                pn.state.notifications.info('In progress. Cannot make additional changes until plots update. Create smooth data from ' + str(self.file_name_initial), duration=0)
                try:
                    ms1_smooth, self.index_ms1_peaks, self.index_ms2_peaks = additional_functions.create_smooth(self.file_name_initial, self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity,  self.smooth_radius, \
                                                                                                                self.smooth_iterations, new_smooth_name, rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name)
                except Exception as e:
                    raise Exception(str(e))
                # update file selector widget with new names from folder
                self.param.file_name_smooth.update()
                # set the file_folder and name of smooth data
                self.file_name_smooth = new_smooth_name
                self.data_smooth_ms1  = dd.from_pandas(ms1_smooth, npartitions=mp.cpu_count())
                pn.state.notifications.clear()
                pn.state.notifications.info('Finished data processing. Creating plots. Created smooth data from ' + str(self.file_name_smooth), duration=10000)
                
            self.data_smooth_ms1.persist()
        else:   
# if the data value has already been updated from rerun, 
#don't update again until user clicks rerun again
            
            pn.state.notifications.info('Re-triggered create_smooth_data function, loading previously loaded file', duration=10000)
            pass
        return hv.Dataset(self.data_smooth_ms1)
  
    @pm.depends('rerun_smooth', watch = True)
    def smooth_viewable(self, **kwargs):
        '''Full function to load and process smooth function 
        If users already has peak data, this step can be skipped
        return three graphs and smooth_data.h5 in created_data'''
        pn.state.notifications.position = 'top-right'
        self.data_smooth_ms1 = pd.DataFrame({'A' : []})
        pn.state.notifications.info('Loading smooth data: ' + str(self.file_name_smooth), duration=10000)
        # dynamic map to return hvdata after loading it with deimos
        hvdata_smooth = hv.DynamicMap(self.create_smooth_data)

        # return the hvplot for mz and retention_time
        hvplot_md_smooth = hvdata_smooth.apply(self.hvplot_md)
        hvplot_dr_smooth = hvdata_smooth.apply(self.hvplot_dr)
        hvplot_rm_smooth = hvdata_smooth.apply(self.hvplot_rm)
        
        # stream to rasterize the plot. any change will cause the whole plot to reload
        stream_smooth = hv.streams.Params(
            self, ['Recreate_plots_with_below_values'])
        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy stream
        
        self.rasterized_md_smooth = hv.util.Dynamic(
                    hvplot_md_smooth,
                    operation=self.rasterize_md,
                    streams=[stream_smooth],
                )
        
        self.rasterized_dr_smooth = hv.util.Dynamic(
                    hvplot_dr_smooth,
                    operation=self.rasterize_dr,
                    streams=[stream_smooth],
                )
        
        self.rasterized_rm_smooth = hv.util.Dynamic(
                    hvplot_rm_smooth,
                    operation=self.rasterize_rm,
                    streams=[stream_smooth],
                )
        return ls2( self.rasterized_rm_smooth  + self.rasterized_dr_smooth + self.rasterized_md_smooth).opts(shared_axes=True)
    
    @pm.depends('rerun_peak')
    def create_peak_data(self):
        '''Get peak data using deimos functions
        Saves the peak value and changes the file name in the user inputs to new peak file name
        Return the peak data to make the graphs'''
        if len(self.data_peak_ms1) == 0:
            
            # name will be saved as, check if already exists, if so don't rerun
            new_peak_name = os.path.join( "created_data",  Path(self.file_name_initial).stem  + '_threshold_' + str(self.threshold_slider) + \
                '_peak_radius_' + str(self.peak_radius) +  "_feature_rt_" + str(self.feature_rt) +\
                    '_new_peak_data.h5')
            pn.state.notifications.info('In progress. Cannot make additional changes until plots update. Create peak data: ' + str(self.file_name_smooth), duration=0)
            if self.file_name_initial == "data/placeholder.csv":
                    self.data_peak_ms1 = dd.from_pandas(pd.DataFrame([[0,0,0,0],[2000,200,200,4], [20,10,30,100]], columns = [self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity]), npartitions=mp.cpu_count())
            else:
                if os.path.isfile(new_peak_name):
                    try:
                        pn.state.notifications.info('Loading previously created peak file', duration=10000)
                        pn.state.notifications.info('If you wish to recreate the file, delete or rename ' + str(new_peak_name), duration=10000)
                        ms1_peaks = additional_functions.load_mz_h5(new_peak_name, key='ms1', columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity], rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name)
                    except Exception as e:
                        raise Exception(str(e))
                else:
                    # if have smooth data from previous step
                    # add boolean to use created threshold data and created smooth data or redo entiremely
                    try:
                        ms1_peaks = additional_functions.create_peak(self.file_name_smooth, self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity, int(self.threshold_slider), self.peak_radius, self.index_ms1_peaks, self.index_ms2_peaks,\
                                                                    new_peak_name, rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name)
                    except Exception as e:
                        raise Exception(str(e))
                self.param.file_name_peak.update()
                self.file_name_peak = new_peak_name
                self.data_peak_ms1  = dd.from_pandas(ms1_peaks, npartitions=mp.cpu_count())
            pn.state.notifications.clear()
            pn.state.notifications.info('Finished: Peak data at ' + str(self.file_name_peak), duration=10000)
            self.data_peak_ms1.persist()
        else:   
        # if the data value has already been updated from rerun, 
        #don't update again until user clicks rerun again
            pn.state.notifications.info('Re-triggered create_peak_data function, loading previously loaded file', duration=10000)
            pass

        return hv.Dataset(self.data_peak_ms1)
    @pm.depends('rerun_peak', watch = True)
    def peak_viewable(self, **kwargs):
        '''Run full function to load smooth data, run peak function and return heatmaps'''
        # dynamic map to return hvdata after loading it with deimos
        pn.state.notifications.position = 'top-right'
        
        self.param.file_name_initial.update()
        self.param.file_name_smooth.update()
        self.param.file_name_peak.update()
        pn.state.notifications.info('Loading peak data: ' + str(self.file_name_peak), duration=10000)

        self.data_peak_ms1 = pd.DataFrame({'A' : []})
        hvdata_peak = hv.DynamicMap(self.create_peak_data)
        # return the hvplot for mz and retention_time

        # return the hvplot for mz and retention_time
        hvplot_md_peak = hvdata_peak.apply(self.hvplot_md)
        hvplot_dr_peak = hvdata_peak.apply(self.hvplot_dr)
        hvplot_rm_peak = hvdata_peak.apply(self.hvplot_rm)
        
        # stream to rasterize the plot. any change will cause the whole plot to reload
        stream_peak = hv.streams.Params(
            self, ['Recreate_plots_with_below_values'])
        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy stream
        
        self.rasterized_md_peak = hv.util.Dynamic(
                    hvplot_md_peak,
                    operation=self.rasterize_md,
                    streams=[stream_peak],
                )
        
        self.rasterized_dr_peak = hv.util.Dynamic(
                    hvplot_dr_peak,
                    operation=self.rasterize_dr,
                    streams=[stream_peak],
                )
        
        self.rasterized_rm_peak = hv.util.Dynamic(
                    hvplot_rm_peak,
                    operation=self.rasterize_rm,
                    streams=[stream_peak],
                )
        return ls3(self.rasterized_dr_peak + self.rasterized_rm_peak + self.rasterized_md_peak).opts(shared_axes=True)
    
    
    @pm.depends('rerun_decon', watch = True)
    def ms2_decon(self):
        '''Get the deconvoluted file
        using ms1 and ms2 data from the orignal file and peak file
        Returns decovuluted file ending in _res.csv in created_data folder'''
        
        # don't need to recreate for sequental graphs if already created for first graphs
        # self.res will be reset to empty dataframe with rerun 
        if len(self.res) == 0:
            file_name_res = os.path.join( "created_data",  Path(self.file_name_initial).stem  + '_threshold_' + str(self.threshold_slider_ms1_ms2) + \
                '_file_path_peak_' + Path(self.file_name_peak).stem  + \
                    '_res.csv')
            pn.state.notifications.info("In progress. Cannot make additional changes until plots update. Run deconvolution", duration=10000)
            if self.file_name_peak == "created_data/placeholder.csv":
                    self.res = pd.DataFrame([[1,1,2,3,1,1,2,3],[2,1,3,4,1,1,2,3], [20,10,30,100,1,1,2,3]], \
                                    columns = ["mz_ms1","drift_time_ms1","retention_time_ms1",\
                                                "intensity_ms1","mz_ms2","drift_time_ms2","retention_time_ms2","intensity_ms2"])
            else:
                if os.path.isfile(file_name_res):
                        self.res = pd.read_csv(file_name_res)
                else:
                    threshold_peak_ms1 = 10000
                    threshold_peak_ms2 = 1000
                    threshold_full_m1 = int(self.threshold_slider_ms1_ms2)
                    threshold_full_m2 = int(self.threshold_slider_ms1_ms2)
                    require_ms1_greater_than_ms2 = True
                    drift_score_min = True
                    try:
                        ms2_peaks = additional_functions.load_mz_h5(self.file_name_peak, key='ms2', columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity], rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name)
                        ms1_peaks = additional_functions.load_mz_h5(self.file_name_peak, key='ms1', columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity], rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name)
                    
                        new_name = additional_functions.new_name_if_mz(self.file_name_initial)
                        ms1 = additional_functions.load_mz_h5(self.file_name_initial, key='ms1', columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity], rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name, new_name=new_name)
                        ms2 = additional_functions.load_mz_h5(self.file_name_initial, key='ms2', columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity], rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name, new_name=new_name)
                        if new_name != None:
                            self.file_folder_initial = "created_data"
                            self.update_param(new_name)
                            pn.state.notifications.info("initial input file has changed to " + str(new_name))
                    
                    except Exception as e:

                        pn.state.notifications.error("Check if peak files have been created", duration=0)
                        raise Exception(str(e))
                    # get thresholds of ms1 and ms2 peak and full data
                    ms1 = deimos.threshold(ms1, threshold= threshold_full_m1)
                    ms2 = deimos.threshold(ms2, threshold= threshold_full_m2)  
                    
                    ms1_peaks = deimos.threshold(ms1_peaks, threshold=threshold_peak_ms1)
                    ms2_peaks = deimos.threshold(ms2_peaks, threshold=threshold_peak_ms2)

                    self.res = additional_functions.decon_ms2(ms1_peaks, ms1, ms2_peaks, ms2, self.feature_mz, self.feature_dt, self.feature_rt, require_ms1_greater_than_ms2, drift_score_min)
                    self.res.to_csv(file_name_res)
                    
                    pn.state.notifications.clear()
                    pn.state.notifications.info("Finished running deconvolution", duration=10000)
        
        else:   
            pn.state.notifications.info('Re-triggered ms2_decon function, loading previously loaded file', duration=10000)
        
        return hv.Dataset(self.res)
        
# create the hv plots
    def hvplot_md_decon(self, ds):
        '''Plot for mz vs drift'''
        plot_df = ds.data.rename(columns = {self.feature_intensity + '_ms1': self.feature_intensity})
        element = plot_df.hvplot.points(x=self.feature_mz + '_ms1', y=self.feature_dt + '_ms1', c=self.feature_intensity)
        return element# create the hv plots
    
        # create the hv plots
    def hvplot_dr_decon(self, ds):
        '''Plot for drift vs retention time'''
        plot_df = ds.data.rename(columns = {self.feature_intensity + '_ms1': self.feature_intensity})
        element = plot_df.hvplot.points(x=self.feature_dt + '_ms1', y=self.feature_rt + '_ms1', c=self.feature_intensity)
        return element
    
    def hvplot_rm_decon(self, ds):
        '''Decon plot for retention time vs mz'''
        plot_df = ds.data.rename(columns = {self.feature_intensity + '_ms1': self.feature_intensity})
        element = plot_df.hvplot.points(x=self.feature_rt + '_ms1', y=self.feature_mz + '_ms1', c=self.feature_intensity)
        return element
    
    def hvplot_mi_decon(self, ds):
        '''Plot for mz vs intensity decon'''
        data_collapse = deimos.collapse(ds.data, keep='mz')
        element2 = hv.Spikes(data_collapse , self.feature_mz, self.feature_intensity).opts(framewise = True, width=600)
        return  element2 
    
    # input data is the selection on the  dt vs rt time for ms1 data plot
    def function_return_m2_subset_decon(self, ds, m1, d1, d2, r2, r3, m3):
        '''Plot for mz vs intensity deconvolution
        Inputs are the selected values from the three scatter plots
        If there is no data associated with the scatter plot values, 
        returns random graph with small values so it's obvious it's not real
        Otherwise select the highest intensity of the selction, and return the ms2 decon values '''
        example_ms2= pd.DataFrame([[np.random.randint(0,2), np.random.randint(0,2)], [np.random.randint(0,2), np.random.randint(0,2)]],\
                                   columns = [self.feature_mz, self.feature_intensity])
        if self.file_name_peak == "created_data/placeholder.csv":
            
            return hv.Dataset(example_ms2)
        # get the decon plots if necessary
        self.md_decon_stream.reset()
        self.dr_decon_stream.reset()
        self.rm_decon_stream.reset()
        
        #get dimensions depending on which depending on which plot were selected
        if m1 != self.m1 and d1 != self.d1:
            x_column_plot = self.feature_mz + '_ms1'
            y_column_plot = self.feature_dt + '_ms1'
            #todo: make this space its own input
            space_x = self.min_feature_mz_spacing
            space_y = self.min_feature_dt_spacing
            x_range = m1
            y_range = d1
        elif d2 != self.d2 and r2 != self.r2:
            x_column_plot = self.feature_dt + '_ms1'
            y_column_plot = self.feature_rt + '_ms1'
            space_x = self.min_feature_dt_spacing
            space_y = self.min_feature_rt_spacing
            x_range = d2
            y_range = r2
        else:
            x_column_plot = self.feature_rt + '_ms1'
            y_column_plot = self.feature_mz + '_ms1'
            space_x = self.min_feature_rt_spacing
            space_y = self.min_feature_mz_spacing
            x_range = r3
            y_range = m3
        # save current values to see which values have changes and use that as input
        self.m1, self.d1, self.d2, self.r2, self.r3, self.m3 = m1, d1, d2, r2, r3, m3
        # should only be one range to use from y and x, else use none
        # using range of last selection

        pn.state.notifications.info('Slice data: ' + str(x_range) + " " + str(y_range), duration=10000)
        res = ds.data
        def lit_list(x):
            '''Get python object of list from string'''
            try:
                return ast.literal_eval(str(x))   
            except Exception as e:
                
                pn.state.notifications.info(e, duration=10000)
                return x
            
        # if string, convert to list. if list, string and back to list (does nothing)
        res[self.feature_mz + '_ms2'] = res[self.feature_mz + '_ms2'].apply(lambda x: lit_list(x))
        res[self.feature_intensity + '_ms2'] = res[self.feature_intensity + '_ms2'].apply(lambda x: lit_list(x))
        # just getting example of the first ms2 location data to use if haven't selected anything yet

        
   
        # if no range selected 
        if (x_range == None or math.isnan(float(x_range))): 
            return hv.Dataset(example_ms2)
        else:
            # slice data to get subset of ms1 data, of which will pick the highest intensity to get ms2 data
            x_range = float(x_range)
            y_range = float(y_range)
            ms2_subset = deimos.subset.slice(
                res,
                by=[x_column_plot, y_column_plot],
                low=[x_range - space_x, y_range - space_y],
                high=[x_range + space_x, y_range + space_y],
            )
            if isinstance(ms2_subset, type(None)):
                pn.state.notifications.error("No MS2 decon data within " + str(x_column_plot) + ": " + str(x_range) +  " + " + str(space_x) + " or " + str(y_column_plot) + ": " + str(y_range) + " + " + str(space_y), duration = 0)
                return hv.Dataset(example_ms2)
            else:
                # of the subset, get maximum intensity of ms1 data
                max_idx =  pd.to_numeric(ms2_subset[self.feature_intensity + '_ms1']).idxmax()
                if len(res.loc[max_idx, self.feature_mz + '_ms2']) == 1:
                    # if just one value of ms2 in list, convert to pandas dataframe rather than a series
                    numpy_dataframe = np.hstack((np.array(res.loc[max_idx, self.feature_mz + '_ms2']),
                    np.array(res.loc[max_idx, self.feature_intensity + '_ms2'])))
            
                    highest_ms2  = pd.DataFrame(numpy_dataframe.reshape(-1, len(numpy_dataframe)), columns = [self.feature_mz, self.feature_intensity])
                else:
                    highest_ms2 = pd.DataFrame({self.feature_mz: np.array(res.loc[max_idx, self.feature_mz + '_ms2']),
                    self.feature_intensity: np.array(res.loc[max_idx, self.feature_intensity + '_ms2'])})
                pn.state.notifications.clear()
                return hv.Dataset(highest_ms2)
 

    # create the hv plots with intenisty and ms2 data
    @pm.depends('rerun_decon', watch= True)
    def decon_viewable(self, **kwargs):
        '''Main function to get the deconvolution values from peak and initial data'''
        pn.state.notifications.position = 'top-right'
        # dynamic map to return hvdata after loading it with deimos
        # trigger with 'run decon' button
        self.m1, self.d1, self.d2, self.r2, self.r3, self.m3 = None, None, None, None, None, None
        # reset res to empty so get new res files if rerun
        self.res =  pd.DataFrame({'A': []})
        #get ms2 convoluted data from peak and ms2 data
        ms2_decon = hv.DynamicMap(self.ms2_decon)
    
        # return the hvplot for mz and retention_time
        self.md_decon = ms2_decon.apply(self.hvplot_md_decon)
        self.dr_decon = ms2_decon.apply(self.hvplot_dr_decon)
        self.rm_decon = ms2_decon.apply(self.hvplot_rm_decon)
   
        # xy is using the plot of ms1 data as the source of the pointer
        self.md_decon_stream = hv.streams.Tap(source=self.md_decon, rename = {'x': 'm1', 'y': 'd1'})
        self.dr_decon_stream = hv.streams.Tap(source=self.dr_decon, rename = {'x': 'd2', 'y': 'r2'})
        self.rm_decon_stream = hv.streams.Tap(source=self.rm_decon, rename = {'x': 'r3', 'y': 'm3'})
        # resample from res output stream xy_dcon is updated
        filtered_ms2_data_decon = ms2_decon.apply(self.function_return_m2_subset_decon, streams=[self.md_decon_stream, self.dr_decon_stream, self.rm_decon_stream])
        
        # make ms plot
        full_plot_1_mi_decon = hv.util.Dynamic(filtered_ms2_data_decon,  operation= self.hvplot_mi_decon)

        return hv.Layout(self.rm_decon + self.md_decon  + self.dr_decon + full_plot_1_mi_decon).opts(shared_axes=False).cols(2)
    

        

    def rasterize_md_iso(
        self,
        element,
        Recreate_plots_with_below_values_iso
    ):
        '''Aggregrate by grid for mz vs drift plot
        with x and y range and x and y spacing
        Run if steam value of 
        Recreate_plots_with_below_values_iso changes'''
        pn.state.notifications.info("Re-aggregating based on values below")
        rasterize_plot = additional_functions.rasterize_plot(
        element = element,
        feature_intensity = self.feature_intensity, 
        x_filter= self.feature_mz_axis_width_iso,
        y_filter=self.feature_dt_axis_width_iso,
        x_spacing=self.min_feature_dt_bin_size_iso,
        y_spacing=self.min_feature_mz_bin_size_iso)
        
        return rasterize_plot
    

    def rasterize_dr_iso(
        self,
        element,
        Recreate_plots_with_below_values_iso
    ):
        '''Aggregrate by grid for drift vs retention plot 
        with x and y range and x and y spacing
        Run if steam value of 
        Recreate_plots_with_below_values_iso changes'''
        rasterize_plot = additional_functions.rasterize_plot(
        element = element,
        feature_intensity = self.feature_intensity, 
        x_filter= self.feature_dt_axis_width_iso,
        y_filter=self.feature_rt_axis_width_iso,
        x_spacing=self.min_feature_dt_bin_size_iso,
        y_spacing=self.min_feature_rt_bin_size_iso)
        
        return rasterize_plot
    

    def rasterize_rm_iso(
        self,
        element,
        Recreate_plots_with_below_values_iso
    ):
        '''Aggregrate by grid for retention vs mz plot 
        with x and y range and x and y spacing
        Run if steam value of 
        Recreate_plots_with_below_values_iso changes'''
        rasterize_plot = additional_functions.rasterize_plot(
        element = element,
        feature_intensity = self.feature_intensity, 
        x_filter= self.feature_rt_axis_width_iso,
        y_filter=self.feature_mz_axis_width_iso,
        x_spacing=self.min_feature_rt_bin_size_iso,
        y_spacing=self.min_feature_mz_bin_size_iso)
        
        return rasterize_plot
    # resets the axis to the data's min and max
    @pm.depends('reset_filter_iso',  watch=True)
    def refresh_axis_values_iso(self):
        '''Reset the manual filter to the min and max values from the input data'''
        self.reset_xy_stream()
        if len(self.feature_iso) > 0:
            # since manual filter is true, tthis will reset xy stream 
            # and set the x and y range to the min and max of the new data via the rasterize functon
            mz_range = (
                int(self.feature_iso.mz.min()),
                int(self.feature_iso.mz.max() + 1),
            )
            #self.param sets the features within the param, while self.x sets actual value of x
            self.param.feature_mz_axis_width_iso.bounds = mz_range
            self.feature_mz_axis_width_iso = mz_range
            retention_range = (
                int(self.feature_iso[self.feature_rt].min()),
                int(self.feature_iso[self.feature_rt].max() + 1),
            )
            self.param.feature_rt_axis_width_iso.bounds = retention_range
            self.feature_rt_axis_width_iso = retention_range
            drift_range = (
                int(self.feature_iso[self.feature_dt].min()),
                int(self.feature_iso[self.feature_dt].max() + 1),
            )
            self.param.feature_dt_axis_width_iso.bounds = drift_range
            self.feature_dt_axis_width_iso = drift_range
        
        return
    
    @pm.depends('rerun_iso', watch = True)
    def get_isotype(self):
        '''Get the isotopes dataframe from input values of the peak data file
        Isotope dataframe will be save in created_data ending in isotopes.csv'''
        # Load data
        if len(self.isotopes_head) == 0:
            
            if self.file_name_peak == "created_data/placeholder.csv":
                self.isotopes_head = pd.DataFrame([[1,1,2,[3,2],[3,4],[3,3]],[2,2,3,[3,5],[3,6],[3,3]]], \
                                    columns = ["mz","idx","intensity","mz_iso","intensity_iso","idx_iso"])
                self.ms1_peaks = pd.DataFrame([[3,10,12,3],[4,12,13,4], [2,12,314,2]], columns = [self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity])
            else:
                pn.state.notifications.info('In progress. Cannot make additional changes until plots update. Get Isotopes', duration=10000)
                parameter_names = Path(self.file_name_peak).stem + "isotopes.csv"
                ms1_peaks = deimos.load(self.file_name_peak, key='ms1',
                                            columns=['mz', 'drift_time', 'retention_time', 'intensity'])
                self.ms1_peaks = deimos.threshold(ms1_peaks, threshold=1000)
                if os.path.exists(parameter_names):
                    self.isotopes_head = pd.read_csv(parameter_names)
                else:
                    
                    # Partition the data
                    partitions = deimos.partition(self.ms1_peaks, size=1000, overlap=5.1)
                    # Map isotope detection over partitions
                    isotopes = partitions.map(deimos.isotopes.detect,
                                            dims=['mz', 'drift_time', 'retention_time'],
                                            tol=[0.1, 0.7, 0.15],
                                            delta=1.003355,
                                            max_isotopes=5,
                                            max_charge=1,
                                            max_error=50E-6)
                    
                    self.isotopes_head = isotopes.sort_values(by=['intensity', 'n'], ascending=False)
                    self.isotopes_head.reset_index(inplace = True)
                    self.isotopes_head.to_csv(parameter_names)
                
                pn.state.notifications.clear()
                pn.state.notifications.info('Finished getting isotopes', duration=10000)  
        else:
            pn.state.notifications.info('Re-triggered rerun_iso function, loading previously loaded file', duration=10000)
            pass
        return hv.Dataset(self.isotopes_head)
    
    def get_ids(self, table, index):
        '''Return a slice of the ms1 data based on user input of range 
        and the mz values of selected row in table'''
        pn.state.notifications.info('Get index: ' + str(index) + " Click 'Recreate plots' to view with correct axis range", duration=10000) 
        if self.file_name_peak == "created_data/placeholder.csv":
            mz1 = np.random.randint(0,9)
            mz2 = np.random.randint(0,9)
            mz3 = np.random.randint(0,9)
            feature = pd.DataFrame([[mz1,10,12,3],[mz2,12,13,4], [mz3,12,314,3]], columns = [self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity])
            self.mz_iso = [mz1, mz2]
            self.mz = mz3
        else:
            if index== None or str(index) == '[]':
                index = 0
            else:
                index = index[0]
            # should it include the idx column too (Plus idx iso)
            def lit_list(x):
                '''Get python object of list from string'''
                try:
                    return ast.literal_eval(str(x))   
                except Exception as e:
                    
                    pn.state.notifications.info(e, duration=10000)
                    return x
            self.mz_iso = lit_list(self.isotopes_head.iloc[index]['mz_iso'])

            self.mz = self.isotopes_head.iloc[index]['mz']

            idx = self.isotopes_head.iloc[index]['idx']
            # get values from original mz using idx from isotopes table
            mz_idx = self.ms1_peaks.iloc[int(idx)]['mz']
            rt_idx = self.ms1_peaks.iloc[int(idx)]['retention_time']
            dt_idx = self.ms1_peaks.iloc[int(idx)]['drift_time']
            # slice based on inputs
            feature = deimos.slice(table.data, by=['mz', 'drift_time', 'retention_time'],
                        low=[mz_idx-self.slice_distance_mz, dt_idx-self.slice_distance_dt, rt_idx-self.slice_distance_rt],
                        high=[mz_idx+5*1.003355, dt_idx+self.slice_distance_dt, rt_idx+self.slice_distance_rt])
            
        self.feature_iso = feature
        self.refresh_axis_values_iso()
        # to do return slice of table from mz retention time drift time values
        return hv.Dataset(feature)
    
    def hvplot_datatable_iso(self, ds):
        '''Return datatable with isotope values for isotopes'''
        element2 = hv.Table(ds.data.applymap(str).to_dict('list'), list(ds.data.columns)).opts(framewise = True, width=600)
        return element2
       
    # create the hv plots
    def hvplot_md_iso(self, ds):
        '''Return scatter plot with mz vs drift for isotopes'''
        element = ds.data.hvplot.points(x=self.feature_mz, y=self.feature_dt, c=self.feature_intensity)
        return element
        # create the hv plots 
    def hvplot_dr_iso(self, ds):
        '''Return scatter plot with drift vs retention time for isotopes'''
        element = ds.data.hvplot.points(x=self.feature_dt, y=self.feature_rt, c=self.feature_intensity)
        return element
    # create the hv plots
    def hvplot_rm_iso(self, ds):
        '''Return scatter plot with retention time vs mz time for isotopes'''
        element = ds.data.hvplot.points(x=self.feature_rt, y=self.feature_mz, c=self.feature_intensity)
        return element
        # create the hv plots with intenisty and ms2 data
    def hvplot_mi_iso(self, ds):
        '''Return spike plot for mz values and intensies from row selected by the user'''
        data_collapse = deimos.collapse(ds.data, keep='mz')
        element2 = hv.Spikes(data_collapse , self.feature_mz, self.feature_intensity).opts(framewise = True, width=600)
        iso_points = hv.Points(np.array([(x, 0) for x in self.mz_iso])).opts(size=5)
        points = hv.Points(np.array([(self.mz, 0)])).opts(size=5)
        return  (element2 * iso_points * points).opts(xticks=5, yticks=5, xlim=(data_collapse.mz.min(), data_collapse.mz.min()), ylim=(data_collapse.intensity.min(), data_collapse.intensity.max()))
    
    @pm.depends('rerun_iso', watch = True)
    def get_ms1(self):
        '''Load ms1 data whenever either placeholder or rerun button is clicked'''
        if len(self.ms1) == 0:
            
            try:
                new_name = additional_functions.new_name_if_mz(self.file_name_initial)
                self.ms1 = additional_functions.load_mz_h5(self.file_name_initial, key='ms1', columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity], rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name, new_name = new_name)
                if new_name != None:
                        self.file_folder_initial = "created_data"

                        self.update_param(new_name)
                        pn.state.notifications.info("initial input file has changed to " + str(new_name))
                
            except Exception as e:
                    raise Exception(str(e))
        return hv.Dataset(self.ms1)
    
    @pm.depends('rerun_iso', watch = True)
    def iso_viewable(self, **kwargs):
        '''Main function to view the isotopes and if the user clicks on the isotopes table row, 
        to see the ms1 data and the mz data from that row'''
        pn.state.notifications.position = 'top-right'
        pn.state.notifications.info('Return Isotope data', duration=10000)
        # dynamic map to return hvdata after loading it with deimos
        #get isotype data from peak  when run_iso or placeholder changes
        self.isotopes_head = pd.DataFrame({'A': []})
        iso_data = hv.DynamicMap(self.get_isotype)
        
        # turn data into datatables, triggered when iso_data changes
        iso_dataframe = hv.util.Dynamic(iso_data, operation= self.hvplot_datatable_iso)
        stream_ids = hv.streams.Selection1D(source=iso_dataframe)

        # get all ms1 data again when run_iso or placeholder changes, update ms1 used in get_ids
        self.ms1 = pd.DataFrame({'A': []})
        ms1 = hv.DynamicMap(self.get_ms1)

        # filter the ms1 data by the values of the selected isotye data
        # triggered when ms1 changes or streams change
        iso_dataframe_filtered = ms1.apply(self.get_ids, streams=[stream_ids])

        
        # return the hvplot for mz, drift time and retention time retention_time
        hvplot_md_iso = iso_dataframe_filtered.apply(self.hvplot_md_iso)
        hvplot_dr_iso = iso_dataframe_filtered.apply(self.hvplot_dr_iso)
        hvplot_rm_iso = iso_dataframe_filtered.apply(self.hvplot_rm_iso)
        hvplot_mi_iso = iso_dataframe_filtered.apply(self.hvplot_mi_iso)
        
  
        # stream to rasterize the plot. any change will cause the whole plot to reload
        stream_iso = hv.streams.Params(
            self, ['Recreate_plots_with_below_values_iso'])
        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy stream
        
        self.rasterized_md_iso = hv.util.Dynamic(
                    hvplot_md_iso,
                    operation=self.rasterize_md_iso,
                    streams=[stream_iso],
                )
        
        self.rasterized_dr_iso = hv.util.Dynamic(
                    hvplot_dr_iso,
                    operation=self.rasterize_dr_iso,
                    streams=[stream_iso],
                )
        
        self.rasterized_rm_iso = hv.util.Dynamic(
                    hvplot_rm_iso,
                    operation=self.rasterize_rm_iso,
                    streams=[stream_iso],
                )

        
        pn.state.notifications.clear() 
        pn.state.notifications.info('Finished with Isotopes function', duration=10000) 
        return hv.Layout(iso_dataframe + iso_dataframe_filtered \
            +  self.rasterized_md_iso +  self.rasterized_dr_iso  + self.rasterized_rm_iso \
                + hvplot_mi_iso).opts(shared_axes=False).cols(2)
    
    @pm.depends('rerun_calibrate')
    def calibrate(self):
        '''Return the calibrated values from the user input in created_data folder
        depending on the type of calibration chosen by the user'''
        if len(self.cal_values) == 0:
            if self.calibration_input == "data/placeholder.csv":
                self.cal_values = pd.DataFrame({'reduced_ccs': np.array([1,1]), 'ta': np.array([1,1])}, columns=['reduced_ccs', 'ta'])
                #https://panel.holoviz.org/reference/global/Notifications.html
                
            # Load tune data
            #load_deimos_data
            else:
                pn.state.notifications.info('In progress. Cannot make additional changes until plots update. Start calibrating data', duration=0)
                try:
                    
                    cal_input = additional_functions.load_mz_h5(self.calibration_input, key='ms1', \
                                                                columns= ["mz", "ta", "ccs", "charge"])

                    tune = additional_functions.load_mz_h5(self.example_tune_file, key='ms1', \
                                                                columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity])
                    
                    to_calibrate = additional_functions.load_mz_h5(self.file_to_calibrate, key='ms1', \
                                                                columns=[self.feature_mz, self.feature_dt, self.feature_rt, self.feature_intensity])
                except Exception as e:
                    raise Exception(str(e))
                load_values = self.calibrate_type
                traveling_wave = self.traveling_wave
                beta = float(self.beta)
                tfix = float(self.tfix)
                if load_values == "load_all_values":

                    L1 = [x for x in ['mz', 'ccs', 'charge', 'ta'] if x not in cal_input.columns]
                    L3 = [x for x in ['mz', 'drift_time'] if x not in to_calibrate.columns]
                    if len(L1 + L3) > 0:
                        raise Exception("Make sure calibration input has columns mz, ccs, charge, and ta, and the 'file to calibrate' file has columns mz and drift time")
                    # Load data
                    ccs_cal = deimos.calibration.calibrate_ccs(mz=cal_input['mz'],
                                                        ta=cal_input['ta'],
                                                        ccs=cal_input['ccs'],
                                                        q=cal_input['charge'],
                                                        buffer_mass=28.013,
                                                        power=traveling_wave)

                elif load_values == "use_tunemix":
                    L1 = [x for x in ['mz', 'ccs', 'charge'] if x not in cal_input.columns]
                    L2 = [x for x in ['mz', 'drift_time', 'intensity'] if x not in tune.columns]
                    L3 = [x for x in ['mz', 'drift_time'] if x not in to_calibrate.columns]
                    if len(L1 + L2 + L3) > 0:
                        raise Exception("Make sure calibration input has columns mz, ccs, and charge and mz, drift_time and intensity in the tune file, and mz and drift_time in the 'file to calibrate'")
                    # Load data
                    ccs_cal = deimos.calibration.tunemix(tune,
                                                        mz=cal_input['mz'],
                                                        ccs=cal_input['ccs'],
                                                        q=cal_input['charge'],
                                                        buffer_mass=28.013,
                                                        power=traveling_wave)
                
                elif load_values == "fix_parameters":
                    # Calibrate positive mode
                    ccs_cal = deimos.calibration.calibrate_ccs(beta=beta,
                                                                tfix=tfix,
                                                        power=traveling_wave)

                calibrated_values = ccs_cal.arrival2ccs(mz=to_calibrate['mz'], ta=to_calibrate['drift_time'], q=1)
                calibration_files = os.path.join( "created_data",  Path(self.file_to_calibrate).name + '_calibrated.csv')
                pd.DataFrame(calibrated_values).to_csv(calibration_files)
                pn.state.notifications.clear()
                pn.state.notifications.info('Finished calibrating, file in created_data as ' + str(calibration_files), duration=10000)
                self.cal_values = pd.DataFrame({'reduced_ccs': ccs_cal.reduced_ccs, 'ta': ccs_cal.ta}, columns=['reduced_ccs', 'ta'])
        else:
            pn.state.notifications.info('Load previously loaded res data', duration=10000)  
        return hv.Dataset(self.cal_values)
   
       
    def hvplot_datatable_calibrate(self, ds):
        '''Return points plot of reduced ccs vs arrival time'''
        element = ds.data.hvplot.points(x='reduced_ccs', y='ta', title = "Calibration Graph")
        return element
    
    @pm.depends('rerun_calibrate', watch= True)
    def calibrate_viewable(self, **kwargs):
        '''Main calibrate function to rerun calibration value
        Return plot with reduced ccs to ta
        and file with calibrated values in the created_data folder'''
        pn.state.notifications.position = 'top-right'
        
        self.param.calibration_input.update()
        self.param.example_tune_file.update()
        self.param.file_to_calibrate.update()

        pn.state.notifications.info('Return calibration data', duration=10000)

        self.cal_values =  pd.DataFrame({'A' : []})
        #get isotype data from peak
        new_calibrated = hv.DynamicMap(self.calibrate)
        # turn data into datatables
        cal_dataframe = hv.util.Dynamic(new_calibrated, operation= self.hvplot_datatable_calibrate)
        return cal_dataframe

class Align_plots(pm.Parameterized):
    '''New class for aligning peak data to a reference file'''

    peak_ref = pm.FileSelector(default = os.path.join("data", peak_ref_name),  path="data/*",  doc='Initial File in .h5, .mzML, or .mzML.gz format. Default: example_alignment.h5. Also can change to refresh peak folder files', label='Initial Data. Default: example_alignment.h5')
    file_folder =  pm.String(
        default= 'data', doc='Please use forward slashes / and starting from / if absolute ', label='Location of data folder (use /).')
    peak_folder =  pm.String(
        default= "data", doc='Either relative path to file or absolute path to folder with peak files', label='Location of peak folder')
    align_endswith =  pm.String(default="*.h5", doc='Use * for wildcard (ie. *end.h5)', label='Only use files that end with this value')
    tolerance_text = pm.String(default = '.00002-0.03-2', doc="Keep - between numbers", label='Tolerances for alignment by mz, drift, and retention time')
    relative_text = pm.String(default = 'True-True-False',  doc="Keep - between numbers", label  = 'Relevant or abs val by mz, drift, and retention time used during tolarance')
    menu_kernel = pm.Selector(['linear',  'rbf'], default = "rbf", doc="Changes the alignment kernel", label='Support Vector Regression Kernel used during alignment')
    threshold_text = pm.String(default = '2000', label = 'Threshold', doc="Only keep values above this intensity")
    rerun_align = pm.Action(lambda x: x.param.trigger('rerun_align'), label='(Re)Run align')
    
    feature_dt = pm.Selector(default='drift_time', objects = ["drift_time", 'retention_time', 'mz'], label="Drift Time", doc="This should be the name of one feature in the data. Change if data is using different column value")
    feature_rt = pm.Selector(default='retention_time', objects = ["drift_time", 'retention_time', 'mz'], label="Retention Time", doc="This should be the name of one feature in the data. Change if data is using different column value")
    feature_mz =  pm.Selector(default='mz', objects = ["drift_time", 'retention_time', 'mz'], label="mz", doc="This should be the name of one feature in the data. Change if data is using different column value")
    feature_intensity = pm.String(default = 'intensity', label='Intensity Feature', doc="Change if data is using different column value")

    rt_mzML_name = pm.Selector(["scan start time"], label="mzML file retention time", doc='Only adjust if mz file selected. Select the retention time column name')
    dt_mzML_name = pm.Selector(["ion mobility drift time"], label="mzML file drift time", doc='Only adjust if mz file selected. Select the retention time column name')

    @pn.depends("file_folder", watch=True)
    def update_param(self, new_name = None):
        '''update the files selectable by the user after the folder updates'''
        # update all files if updating file folder
        if not os.path.isdir(self.file_folder):
            pn.state.notifications.error('Folder does not exist', duration=0)

        if self.file_folder[-1] == '/':
            self.param.peak_ref.path = self.file_folder + "*"
        else:
            self.param.peak_ref.path = self.file_folder + "/*"
        self.param.peak_ref.update()

        if new_name != None:
            self.peak_ref = new_name
        else:
            if self.peak_ref not in self.param.peak_ref.objects:
                self.peak_ref = self.param.peak_ref.objects[0]
            else:
                pass

    def viewable(self):
        '''Align the folder in peak folder with the reference folder
        Returns: 
        * plots showing the alignmnent
        * csv files to create the plots ending in matchtable.csv
        * _xy_drift_retention_time.csv
        csv file ending in alignment.csv with aligned drift and retention time'''

        
        pn.state.notifications.position = 'top-right'

        list_plots = []
        if self.peak_ref == "data/placeholder.csv":
            #return placeholder plots
            i = 0
            for file in range(2):
                    i=+1
                    coords = np.random.rand(2,2)*int(self.threshold_text)
                    plot1 = hv.Points(coords)
                    plot2 = hv.Points(coords)
                    list_plots.append((plot1 * plot2).opts(opts.Overlay(title="Placeholder")))
                    
        else:  
            # if using example_alignment, use file_key B as the "to align" file rather than the user input folder
            if Path(os.path.abspath(self.peak_ref)).stem == 'example_alignment':
                file_list = [self.peak_ref]
                ref_key = 'A'
                file_key = 'B'

            else:
                file_list = [str(f) for f in Path(os.path.abspath(self.peak_folder)).glob(self.align_endswith)]
                ref_key = 'ms1'
                file_key = 'ms1'
            pn.state.notifications.info('In progress. Cannot make additional changes until plots update. Alignment', duration=0)
            theshold_presistence = 128
            tolerance_text = [float(i) for i in list(self.tolerance_text.split('-'))]
            relative_text = [bool(i) for i in list(self.relative_text.split('-'))]
            peak_ref, new_name = additional_functions.get_peak_file(self.peak_ref, self.feature_dt, self.feature_rt, self.feature_mz,\
                                                                        self.feature_intensity, rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name, \
                                                                            theshold_presistence = theshold_presistence, key = ref_key)
            # convert to h5 if mzML
            if new_name != None:
                        self.file_folder = "created_data"
                        self.update_param(new_name)
                        pn.state.notifications.info("Ref file has changed to " + str(new_name), duration=10000)
            peak_two_list = []
            peak_file_list = []
            for file in file_list:
                peak_two, new_name = additional_functions.get_peak_file(file, self.feature_dt, self.feature_rt, self.feature_mz,\
                                                                        self.feature_intensity, rt_name = self.rt_mzML_name, dt_name = self.dt_mzML_name, \
                                                                            theshold_presistence = theshold_presistence, key = file_key)
           
                peak_two_list.append(peak_two)
                if new_name == None:
                    peak_file_list.append(file)
                else:
                    peak_file_list.append(new_name)
            list_plots = []
            i = 0
            for num, (peak_two, peak_file) in enumerate(zip(peak_two_list, peak_file_list)):
                # align each file in the folder with the reference file
                pn.state.notifications.info('Aligning' + peak_file + " " + str(num + 1) +  " out of " + str(len(peak_file_list)), duration=10000)
                # b is reference, a is peak two
                partitions = deimos.partition(deimos.threshold(peak_two, threshold=int(self.threshold_text)), split_on=self.feature_mz, size=1000, overlap=0.25)
                two_matched, ref_matched = partitions.zipmap(deimos.alignment.match, deimos.threshold(peak_ref, threshold=int(self.threshold_text)),
                                            dims=[self.feature_mz, self.feature_dt, self.feature_rt],
                                            tol=tolerance_text, relative=relative_text)
                two_matched = two_matched.copy()
                two_matched_aligned = two_matched.copy()
                i=+1
                for dim in [self.feature_dt, self.feature_rt]:
                    parameter_inputs =  Path(peak_file).stem + str(self.tolerance_text) + str(self.relative_text) + str(self.menu_kernel) + str(self.threshold_text) + str(dim)
                    # if already aligned, re-use values
                    if os.path.exists(os.path.join("created_data", parameter_inputs + "_matchtable.csv"))\
                          and os.path.exists(os.path.join("created_data", parameter_inputs + "_xy_drift_retention_time.csv")):
                        pn.state.notifications.info('Reuse existing files, rename or delete to recreate', duration=10000)
                        pn.state.notifications.info('Reuse ' + os.path.join("created_data", parameter_inputs + "_matchtable.csv"), duration=10000)
                        pn.state.notifications.info('Reuse ' + os.path.join("created_data", parameter_inputs + "_xy_drift_retention_time.csv"), duration=10000)
                        matchtable = pd.read_csv(os.path.join("created_data", parameter_inputs + "_matchtable.csv"))
                        xy_drift_retention_time = pd.read_csv(os.path.join("created_data", parameter_inputs + "_xy_drift_retention_time.csv"))
                    else: 
                        xy_drift_retention_time, matchtable = additional_functions.aligment(two_matched = two_matched, ref_matched = ref_matched, two_matched_aligned= two_matched_aligned , dim = dim, kernel = self.menu_kernel, parameter_inputs = parameter_inputs )
      
                    plot1 = hv.Points(xy_drift_retention_time, kdims=['x_' + dim, 'y_' + dim]).options(color='blue')
                    plot2 = hv.Points(matchtable, kdims=['match_a_' + dim, 'match_b_' + dim]).options(color='red')
            
                    list_plots.append((plot1 * plot2).opts(opts.Overlay(title=dim + ' vs ' + dim + ' ' + str(i))))
                
            pn.state.notifications.clear()
            pn.state.notifications.info('Finished aligning all. Recreating plot', duration=10000)
        return hv.Layout(list_plots).cols(2)
    

Deimos_app = Deimos_app()
Align_plots = Align_plots()
#using viewable() would be best practice for minimizing refresh, but need hard refresh in this case for new axis
instructions_view = "<ul> <li>Click rerun to reflect changes</li> <li>Indicate the local path of the full data below to update</li>\
    <li>Use the box selector (as seen on the bottom) to filter data in all plots based on the box's range</li>\
<li>Changing the axis widths and clicking 'Recreate plots with below values' to re-aggregrate with new widths</li>\
<li>Toolbar's zoom and reset does not re-aggregate within this tool.</li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/getting_started/example_data.html'> Example Data Located Here </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://github.com/pnnl/deimos_gui/blob/master/user_guide_deimos.md'> User Guide </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/'> DEIMoS Guide </a></li>\
    <ul> "
instructions_smooth = "<ul> <li>Click rerun to reflect changes</li> <li>Click 'Run smooth' after updating parameters to get new graph</li><li>Use the box selector (as seen on the bottom) to filter data in all plots based on the box's range</li>\
    <li>Keeping the <b>smooth radius</b> small and increasing number of iterations <br> is preferable to a larger smoothing radius, albeit at greater computational expense.</li>\
    <li>Output files will be in the created_data folder besides the run_app.py file</li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/getting_started/example_data.html'> Example Data Located Here </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://github.com/pnnl/deimos_gui/blob/master/user_guide_deimos.md'> User Guide </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/'> DEIMoS Guide </a></li>\
    <ul> "
instructions_peaks = "<p>Feature detection, also referred to as peak detection, is the process by which local maxima that fulfill certain criteria (such as sufficient signal-to-noise ratio) are located in the signal acquired by a given analytical instrument. </p><ul> <li>Click rerun to reflect changes</li> <li>Click 'Run peak' after updating parameters to get new graph</li><li>Use the box selector (as seen on the bottom) to filter data in all plots based on the box's range</li> \
    <li>The <b>radius per dimension</b> insures an intensity-weighted per-dimension coordinate will be returned for each feature.</li>\
    <li>Output files will be in the created_data folder besides the run_app.py file</li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/getting_started/example_data.html'> Example Data Located Here </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://github.com/pnnl/deimos_gui/blob/master/user_guide_deimos.md'> User Guide </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/'> DEIMoS Guide </a></li>\
    <ul> "
instructions_ms2 = "<p>With MS1 features of interest determined by peak detection, corresponding tandem mass spectra, if available, must be extracted and assigned to the MS1 parent ion feature. </p><ul> <li>The original data is a placeholder, clicking will not work without real data </li> <li>Click 'Run decon' after updating parameters to get new graph</li><li>The MS2 data associated with user-selected MS1 data, with the MS1 data with the highest intensity used if there are multiple MS1 data points within a small range of the user-click </li>\
    <li>Output files will be in the created_data folder besides the run_app.py file</li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/getting_started/example_data.html'> Example Data Located Here </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://github.com/pnnl/deimos_gui/blob/master/user_guide_deimos.md'> User Guide </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/'> DEIMoS Guide </a></li>\
    <ul> "
instructions_align = "<ul><li>Alignment is the process by which feature coordinates across samples are adjusted to account for instrument variation such that matching features are aligned to adjust for small differences in coordinates</li>\
    <li>Click rerun to reflect changes</li> <li>Indicate the reference file and folder of files to align</li><li>Determine matches within <b>tolerance</b> per feature with the alignment determined by the <b>kernel</b> by <b>relative or absolute </b> value by <b>support vector regression kernel </b> </li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/getting_started/example_data.html'> Example Data Located Here </a></li>\
    <li>Output files will be in the created_data folder besides the run_app.py file</li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://github.com/pnnl/deimos_gui/blob/master/user_guide_deimos.md'> User Guide </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/'> DEIMoS Guide </a></li>\
    <ul> "
instructions_calibrate = "<ul><li>Click 'rerun calibrate' to get the calibrated values within the created_data folder</li>\
    <li>Output files will be in the created_data folder besides the run_app.py file</li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/getting_started/example_data.html'> Example Data Located Here </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://github.com/pnnl/deimos_gui/blob/master/user_guide_deimos.md'> User Guide </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/'> DEIMoS Guide </a></li>\
    <ul> "
instructions_isotopes = "<ul><li>Click 'rerun plots' to get the isotopes within the created_data folder</li>\
    <li>Select a row to view the isotopes</li>\
    <li>Graphs will show slice of MS1 data. Plot will show isotopes</li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/getting_started/example_data.html'> Example Data Located Here </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://github.com/pnnl/deimos_gui/blob/master/user_guide_deimos.md'> User Guide </a></li>\
<li> <a target='_blank' rel='noopener noreferrer' href='https://deimos.readthedocs.io/en/latest/'> DEIMoS Guide </a></li>\
    <ul> "
param_full = pn.Column('<b>View initial Data</b>', Deimos_app.param.file_folder_initial,  Deimos_app.param.file_name_initial,  Deimos_app.param.rt_mzML_name, Deimos_app.param.dt_mzML_name, Deimos_app.param.view_plot, Deimos_app.param.remove_notifications, '<b>Adjust the plots</b>', Deimos_app.param.reset_filter, Deimos_app.param.Recreate_plots_with_below_values,
                    Deimos_app.param.feature_dt_axis_width, Deimos_app.param.feature_rt_axis_width, Deimos_app.param.feature_mz_axis_width, \
                        Deimos_app.param.min_feature_dt_bin_size, Deimos_app.param.min_feature_rt_bin_size, Deimos_app.param.min_feature_mz_bin_size, \
                            Deimos_app.param.feature_dt, Deimos_app.param.feature_rt, Deimos_app.param.feature_mz, Deimos_app.param.feature_intensity)
param_smooth = pn.Column('<b>Smooth</b>', Deimos_app.param.file_folder_initial,  Deimos_app.param.file_name_initial, Deimos_app.param.smooth_radius, Deimos_app.param.smooth_iterations,  Deimos_app.param.rerun_smooth,  Deimos_app.param.remove_notifications, '<b>Result</b>', Deimos_app.param.file_name_smooth)
param_peak = pn.Column('<b>Peak-picking</b>', '<b>Adjust the plots</b>', Deimos_app.param.file_name_smooth,   Deimos_app.param.peak_radius, Deimos_app.param.threshold_slider, Deimos_app.param.rerun_peak, Deimos_app.param.remove_notifications, '<b>Result</b>', Deimos_app.param.file_name_peak)
param_decon = pn.Column('<b>MS2 Deconvolution</b>', Deimos_app.param.file_folder_initial, Deimos_app.param.file_name_initial, Deimos_app.param.file_name_peak, Deimos_app.param.threshold_slider_ms1_ms2, Deimos_app.param.min_feature_rt_spacing, Deimos_app.param.min_feature_dt_spacing, Deimos_app.param.min_feature_mz_spacing, Deimos_app.param.rerun_decon, Deimos_app.param.remove_notifications)
param_iso = pn.Column('<b>View Isotopes</b>', Deimos_app.param.file_folder_initial,  Deimos_app.param.file_name_initial,  Deimos_app.param.file_name_peak,  Deimos_app.param.slice_distance_dt, Deimos_app.param.slice_distance_rt,  Deimos_app.param.slice_distance_mz,  Deimos_app.param.rerun_iso, Deimos_app.param.remove_notifications,'<b>Adjust the plots</b>', Deimos_app.param.reset_filter_iso, Deimos_app.param.Recreate_plots_with_below_values_iso,
                    Deimos_app.param.feature_dt_axis_width_iso, Deimos_app.param.feature_rt_axis_width_iso, Deimos_app.param.feature_mz_axis_width_iso, \
                        Deimos_app.param.min_feature_dt_bin_size_iso, Deimos_app.param.min_feature_rt_bin_size_iso, Deimos_app.param.min_feature_mz_bin_size_iso, \
                            Deimos_app.param.feature_dt, Deimos_app.param.feature_rt, Deimos_app.param.feature_mz, Deimos_app.param.feature_intensity)
param_cal = pn.Column('<b>Calibrate</b>', Deimos_app.param.file_folder_cal, Deimos_app.param.calibration_input, Deimos_app.param.example_tune_file, Deimos_app.param.file_to_calibrate, Deimos_app.param.beta,\
                        Deimos_app.param.tfix, Deimos_app.param.traveling_wave, Deimos_app.param.calibrate_type, Deimos_app.param.rerun_calibrate, Deimos_app.param.remove_notifications)

plot_text = '<p><li>Only changing the values with the widgets and clicking Recreate Plot will re-aggregate the plot and reset the width.</li><li> The color of the plots is the sum of the intensities for all ions with the same values of the plot’s x and y dimensions (i.e. retention time vs drift time).</li><li> Before zooming in, datashader will aggregate the values into grids, so the color will represent the aggregate intensity for ions within the same grid.</li> </p>'
app1 = pn.Tabs(
    ('1. Load Initial Data', pn.Row(pn.Column(instructions_view, pn.pane.PNG('box_select.png'),  pn.Row(param_full, pn.Column(Deimos_app.initial_viewable(), plot_text))))),        
                ('2. Smoothing', pn.Row(pn.Column(instructions_smooth, pn.pane.PNG('box_select.png'),  pn.Row(param_smooth, pn.Column(Deimos_app.smooth_viewable(), plot_text))))),\
               ('3. Peak Detection', pn.Row(pn.Column(instructions_peaks, pn.pane.PNG('box_select.png'),  pn.Row(param_peak, pn.Column(Deimos_app.peak_viewable(), plot_text))))),\
               ('Deconvolution', pn.Row(pn.Column(instructions_ms2,  pn.Row( param_decon, Deimos_app.decon_viewable())))),\
               ('Calibration',  pn.Row(pn.Column(instructions_calibrate,   pn.Row(param_cal, Deimos_app.calibrate_viewable())))),\
                ('Isotope Detection', pn.Row(pn.Column(instructions_isotopes, pn.Row(param_iso, Deimos_app.iso_viewable())))),\
                ('Plot Alignment', pn.Row(pn.Column(instructions_align, pn.Row(Align_plots.param, Align_plots.viewable))))\
                ).servable(title='Deimos App')

pn.serve(app1)