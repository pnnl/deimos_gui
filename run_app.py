drift_spacing = 0.15
retention_spacing = 0.15
mz_spacing = 0.02

# bokeh serve --show run_app.py
full1 = 'small_full1.h5'
peak1 = 'small_peak1.h5'

peak_ref = 'small_full1.h5'

peak_folder = 'data/peak_folder/'
data_folder = 'data/'
align_endswith = "example.h5"

# #align_endswith = "data.h5"
# #peak_ref = 'created_data/BRAVE_SC_E027_night_F_M_131_POS_40V_23Aug19_Fiji_ZIC119-06-01_threshold_500_peak_radius_2-10-0_feature2_retention_time_new_peak_data.h5'
# full1 =  'BRAVE_SC_E027_night_F_M_131_POS_40V_23Aug19_Fiji_ZIC119-06-01.h5'

feature1 = 'drift_time'
feature2 = 'retention_time'
feature3 = 'mz'
feature_intensity = 'intensity'

# conda create -n deimos -c conda-forge -c bioconda python=3.7 numpy scipy pandas matplotlib snakemake pymzml h5py statsmodels scikit-learn colorcet holoviews panel xarray hvplot datashader

import pandas as pd
import holoviews as hv
import numpy as np
import panel as pn
import math
import ast

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

hv.extension('bokeh', 'matplotlib')
pn.extension()

# linked selection places constants on all dimensions at once
# linked brushing supports both the box_select and lasso_select tools.
ls = hv.link_selections.instance()
ls2 = hv.link_selections.instance()
ls3 = hv.link_selections.instance()
ls4 = hv.link_selections.instance()

# https://github.com/holoviz/datashader/issues/698
hv.output(backend='bokeh')

def offset_correction_model(dt_ms2, mz_ms2, mz_ms1, ce=0,
                            params=[1.02067031, -0.02062323,  0.00176694]):
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

class Deimos_app(pm.Parameterized):

    file_name_raw = pm.String(
        default=full1, doc='A string', label='Initial Data'
    )

    file_folder =  pm.String(
        default= data_folder, doc='A string', label='Location of data folder'
    )

    # reset the manual filters to the data bounds and reset the rangexy of the plot
    reset_filter = pm.Action(
        lambda x: x.param.trigger('reset_filter'),
        label='Refresh Axis (necessary with new data)',
    )

    # use the manual filter
    manual_filter = pm.Boolean(
        True, label='Set axis width and bin values based on axis below'
    )
    
    feature1 = pm.String(default = feature1, label = 'Feature 1')
    feature2 = pm.String(default = feature2, label = 'Feature 2')
    feature3 = pm.String(default = feature3, label = 'Feature 3')
    feature_intensity = pm.String(default = feature_intensity, label = 'Intensity Feature')

    # manual filter centers
    feature1_axis_width = pm.Range(bounds=(0, 200), label = "Axis width: " + feature1.default)
    feature2_axis_width = pm.Range(bounds=(0, 200), label = "Axis width: " + feature2.default)
    feature3_axis_width = pm.Range(bounds=(0, 2000), label = "Axis width: " + feature3.default)

    # set the min spacing for all the dimensions
    min_feature1_bin_size = pm.Number(default=0.2, label = "Min bin size: " + feature1.default)
    min_feature2_bin_size = pm.Number(default=0.2, label = "Min bin size: " + feature1.default)
    min_feature3_bin_size = pm.Number(default=0.02, label = "Min bin size: " + feature1.default)

    file_name_smooth = pm.String(
        default=peak1, doc='A string', label='Smooth Data'
    )

    file_name_peak = pm.String(
        default=peak1, doc='A string', label='Peak Data'
    )

    threshold_slider = pm.Integer(default=500, label='Threshold')

    smooth_radius = pm.String(
        default='0-1-0', doc='A string', label='Smoothing radius by ' + feature3.default + ', ' + feature1.default + ', and ' + feature2.default
    )

    smooth_iterations = pm.String(
        default='7', doc='A string', label='Smoothing iterations'
    )

    peak_radius = pm.String(
        default='2-10-0', doc='A string', label='Weighted mean kernel size by ' + feature3.default + ', ' + feature1.default + ', and ' + feature2.default
    )
    run_peak = pm.Action(lambda x: x.param.trigger('run_peak'), label='Run peak')
    run_smooth = pm.Action(lambda x: x.param.trigger('run_smooth'), label='Run smooth') 
    run_decon = pm.Action(lambda x: x.param.trigger('run_decon'), label='Run deconvolution')


    # load the h5 files and load to dask
    @pm.depends('file_name_raw', 'feature1', 'feature2', 'feature3', 'feature_intensity')
    def hvdata_raw(self):
        if not os.path.exists(os.path.join(self.file_folder, "created_data")):
            os.makedirs(os.path.join(self.file_folder, "created_data"))
        print('load raw data')
        # changing the label requires changing the parameter class
        self.param.feature1_axis_width.label = "Axis width: " + self.feature1
        self.param.feature2_axis_width.label = "Axis width: " + self.feature2
        self.param.feature3_axis_width.label = "Axis width: " + self.feature3
        self.param.min_feature1_bin_size.label = "Min bin size: " + self.feature1
        self.param.min_feature2_bin_size.label = "Min bin size: " + self.feature2
        self.param.min_feature3_bin_size.label = "Min bin size: " + self.feature3
        self.param.smooth_radius.label = 'Smoothing radius by ' + self.feature3 + ', ' + self.feature1 + ', and ' + self.feature2
        self.param.peak_radius.label = 'Weighted mean kernel size by ' + self.feature3 + ', ' + self.feature1 + ', and ' + self.feature2

        try:
            full_data_1 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms1')
        except:
            # old load
            ms1 = deimos.io._load_hdf(os.path.join(self.file_folder,  self.file_name_raw), level='ms1')
            ms2 = deimos.io._load_hdf(os.path.join(self.file_folder,  self.file_name_raw), level='ms2')
            
            # new save
            deimos.io.save_hdf(os.path.join(self.file_folder,  self.file_name_raw), ms1, key='ms1', mode='w')
            deimos.io.save_hdf(os.path.join(self.file_folder,  self.file_name_raw), ms2, key='ms2', mode='a')
            
            # new load
            full_data_1 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms1')
 
        full_data_1.rename(
            columns={'inverse_reduced_ion_mobility': self.feature1}, inplace=True
        )
        full_data_1 = full_data_1[[self.feature1, self.feature2, self.feature3, self.feature_intensity]]
        full_data_1.reset_index(drop = True, inplace=True)


        self.data_raw = dd.from_pandas(full_data_1, npartitions=mp.cpu_count())
        self.data_raw.persist()

        self.refresh_axis_values()
        return hv.Dataset(self.data_raw)


    # resets the axis the the data's min and max
    @pm.depends('reset_filter',  watch=True)
    def refresh_axis_values(self):

        print('get min and max of data')
        self.reset_xy_stream()

        current_manual_filter_value =  self.manual_filter

        # since manual filter is true, tthis will reset xy stream 
        # and set the x and y range to the min and max of the new data via the rasterize functon
        self.manual_filter = True

        mz_range = (
            int(self.data_raw.mz.min().compute()),
            int(self.data_raw.mz.max().compute()) + 1,
        )
        #self.param sets the features within the param, while self.x sets actual value of x
        self.param.feature3_axis_width.bounds = mz_range
        self.feature3_axis_width = mz_range

        retention_range = (
            int(self.data_raw[self.feature2].min().compute()),
            int(self.data_raw[self.feature2].max().compute()) + 1,
        )
        self.param.feature2_axis_width.bounds = retention_range
        self.feature2_axis_width = retention_range

        drift_range = (
            int(self.data_raw[self.feature1].min().compute()),
            int(self.data_raw[self.feature1].max().compute()) + 1,
        )
        self.param.feature1_axis_width.bounds = drift_range
        self.feature1_axis_width = drift_range

        # back to original value
        self.manual_filter = current_manual_filter_value
        return


    # function that adjusts the x_range and y_range with either x y stream or user input
    # this work-arround is necessary due to the x_range and y_range streams being set to none with new data
    # if not set to none, old data ranges from previous data are still used
    # when range is set to none, need to explicitely set the x and y range of the of the rasterization level and plot limits, else it will rasterize over whole image: 
    # trying to avoid this bug https://github.com/holoviz/holoviews/issues/4396
    # keep an eye on this, as likely will fix in datashader, which would make this work around unnecessary
    def rasterize(
        self,
        element,
        x_range=None,
        y_range=None,
        x_filter=None,
        y_filter=None,
        x_spacing=0,
        y_spacing=0,
    ):
        # if the manual_filter is true, filter by range input
        # else reasterize by zoom box
        if self.manual_filter:
            x_range_input = x_filter
            y_range_input = y_filter
        else:
            x_range_input = x_range
            y_range_input = y_range

        # dynmaic false to allow the x_range and y_range to be adjusted by either
        #xy stream or manual filter rather than automaically
        rasterize_plot = datashade(
            element,
            width=800,
            height=600,
            aggregator=ds.sum(self.feature_intensity),
            x_sampling=x_spacing,
            y_sampling=y_spacing,
            x_range=x_range_input,
            y_range=y_range_input,
            dynamic=False,
        )

        # actually changes the x and y limits seen, not just the rasterization levels
        if y_range_input != None and x_range_input != None:
            rasterize_plot.apply.opts(
                xlim=x_range_input, ylim=y_range_input, framewise=True
            )
        else:
            pass
        return rasterize_plot


    # create the hv plots
    def hvplot_md(self, ds):
        element = ds.data.hvplot.points(x=self.feature3, y=self.feature1, c=self.feature_intensity)
        return element
        # create the hv plots 

    def hvplot_dr(self, ds):
        element = ds.data.hvplot.points(x=self.feature1, y=self.feature2, c=self.feature_intensity)
        return element

    # create the hv plots
    def hvplot_rm(self, ds):
        element = ds.data.hvplot.points(x=self.feature2, y=self.feature3, c=self.feature_intensity)
        return element
    
    # show plots of raw data before any smoothing, peakfinding, etc.
    def raw_viewable(self, **kwargs):
        print('view raw data plots')
        # dynamic map to return hvdata after loading it with deimos - hvplot because needs to be a holoview to be returned with dynamicmap
        hvdata_full = hv.DynamicMap(self.hvdata_raw)

        # return the hvplot for mz and retention_time
        hvplot_md_raw = hvdata_full.apply(self.hvplot_md)
        hvplot_dr_raw = hvdata_full.apply(self.hvplot_dr)
        hvplot_rm_raw = hvdata_full.apply(self.hvplot_rm)
        
        # stream to rasterize the plot. any change will cause the whole plot to reload
        md_stream_raw = hv.streams.Params(
            self, ['feature3_axis_width', 'feature1_axis_width', 'min_feature3_bin_size', 'min_feature1_bin_size'], \
                rename = {'feature3_axis_width': 'x_filter', 'feature1_axis_width': 'y_filter', \
                    'min_feature3_bin_size': 'x_spacing', 'min_feature1_bin_size': 'y_spacing'})

        dr_stream_raw = hv.streams.Params(
            self, ['feature1_axis_width', 'feature2_axis_width', 'min_feature1_bin_size', 'min_feature2_bin_size'], \
                rename = {'feature1_axis_width': 'x_filter', 'feature2_axis_width': 'y_filter', \
                    'min_feature1_bin_size': 'x_spacing', 'min_feature2_bin_size': 'y_spacing'})

        rm_stream_raw = hv.streams.Params(
            self, ['feature2_axis_width', 'feature3_axis_width', 'min_feature2_bin_size', 'min_feature3_bin_size'], \
                rename = {'feature2_axis_width': 'x_filter', 'feature3_axis_width': 'y_filter', \
                    'min_feature2_bin_size': 'x_spacing', 'min_feature3_bin_size': 'y_spacing'})

        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy stream
        self.rasterized_md_raw = hv.util.Dynamic(
            hvplot_md_raw,
            operation=self.rasterize,
            streams=[hv.streams.RangeXY(), md_stream_raw],
        )

        self.rasterized_dr_raw = hv.util.Dynamic(
                    hvplot_dr_raw,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), dr_stream_raw],
                )

        self.rasterized_rm_raw = hv.util.Dynamic(
                    hvplot_rm_raw,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), rm_stream_raw],
                )
        return ls(self.rasterized_rm_raw + self.rasterized_dr_raw +  self.rasterized_md_raw).opts(shared_axes=True)
    


    # does not automatically reset xy axis (keeps at originally loaded, so have to do so automatically)
    @pm.depends('manual_filter', 'file_name_raw',  watch=True)
    def reset_xy_stream(self):
        # plots have streams, first streams are rangexy, which needs to be rest
        print('start resetting')
        try:
            self.rasterized_md_raw.streams[0].reset()
            self.rasterized_dr_raw.streams[0].reset()
            self.rasterized_rm_raw.streams[0].reset()
            print('reset default stream value for xy stream')
        except:
            print('raw does not exist')
        try:
            self.rasterized_md_peak.streams[0].reset()
            self.rasterized_dr_peak.streams[0].reset()
            self.rasterized_rm_peak.streams[0].reset()
            print('reset default stream value for xy stream')
        except:
            print('peak plots does not exist')
        try:
            self.rasterized_md_smooth.streams[0].reset()
            self.rasterized_dr_smooth.streams[0].reset()
            self.rasterized_rm_smooth.streams[0].reset()
            print('reset default stream value for xy stream')
        except:
            print('smooth plots does not exist')
        print('done with reseting')


    @pm.depends('run_smooth')
    def create_smooth_data(self):
        print('create smooth data')
        print(os.path.join(self.file_folder,  self.file_name_raw))
        self.file_name_smooth = self.file_name_raw[:-3] + '_threshold_' + str(self.threshold_slider) + \
             '_smooth_radius_' + str(self.smooth_radius) +  '_smooth_iterations_' + str(self.smooth_iterations) +  "_feature2_" + str(self.feature2) +\
                '_new_smooth_data.h5'
        
        if os.path.isfile(os.path.join(self.file_folder,  "created_data", self.file_name_smooth)):
            ms1_smooth = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_smooth), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
        
            ms1 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
            ms2 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms2', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])

            factors = deimos.build_factors(ms1, dims='detect')
                    
            # Nominal threshold
            ms1 = deimos.threshold(ms1, threshold=int(self.threshold_slider))
            # Build index
            self.index_ms1_peaks = deimos.build_index(ms1, factors)
            

            factors = deimos.build_factors(ms2, dims='detect')
                    
            # Nominal threshold
            ms2 = deimos.threshold(ms2, threshold=int(self.threshold_slider))
            # Build index
            self.index_ms2_peaks = deimos.build_index(ms2, factors)

        else:
            ms1 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
            ms2 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms2', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])

            factors = deimos.build_factors(ms1, dims='detect')
                    
            # Nominal threshold
            ms1 = deimos.threshold(ms1, threshold=int(self.threshold_slider))
            # Build index
            self.index_ms1_peaks = deimos.build_index(ms1, factors)
            # Smooth data
            smooth_radius= [int(i) for i in list(self.smooth_radius.split('-'))]
            iterations = int(self.smooth_iterations)

            ms1_smooth = deimos.filters.smooth(ms1, index=self.index_ms1_peaks, dims=[self.feature3, self.feature1, self.feature2],
                                        radius=smooth_radius, iterations=iterations)
            
            # Save ms1 to new file
            deimos.save(os.path.join(self.file_folder,  "created_data", self.file_name_smooth), ms1_smooth, key='ms1', mode='w')

                    # append peak ms2
            factors = deimos.build_factors(ms2, dims='detect')
                    
            # Nominal threshold
            ms2 = deimos.threshold(ms2, threshold=int(self.threshold_slider))
            # Build index
            self.index_ms2_peaks = deimos.build_index(ms2, factors)

            # Smooth data
            smooth_radius= [int(i) for i in list(self.smooth_radius.split('-'))]
            iterations = int(self.smooth_iterations)
            # Smooth data
            ms2_smooth = deimos.filters.smooth(ms2, index=self.index_ms2_peaks, dims=[self.feature3, self.feature1, self.feature2],
                                        radius=smooth_radius, iterations=iterations)

            # Save ms1 to new file
            deimos.save(os.path.join(self.file_folder,  "created_data", self.file_name_smooth), ms2_smooth, key='ms2', mode='a')

            
        self.data_smooth_ms1  = dd.from_pandas(ms1_smooth, npartitions=mp.cpu_count())
        self.data_smooth_ms1.persist()
        return hv.Dataset(self.data_smooth_ms1)
    
    @pm.depends('run_peak')
    def create_peak_data(self):
        print('create peak data')
        print(os.path.join(self.file_folder,  "created_data", self.file_name_smooth))
        self.file_name_peak = self.file_name_raw[:-3] + '_threshold_' + str(self.threshold_slider) + \
             '_peak_radius_' + str(self.peak_radius) +  "_feature2_" +str(self.feature2) +\
                '_new_peak_data.h5'
        if os.path.isfile(os.path.join(self.file_folder,  "created_data", self.file_name_peak)) :
            ms1_peaks = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_peak), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
     
        else:
             # if have smooth data
            if  self.file_name_raw[:-3] + '_threshold_' + str(self.threshold_slider) + \
             '_smooth_radius_' + str(self.smooth_radius) +  '_smooth_iterations_' + str(self.smooth_iterations)  +  "_feature2_" + str(self.feature2) +\
                '_new_smooth_data.h5' == self.file_name_smooth:

                ms1_smooth = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_smooth), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
                ms2_smooth = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_smooth), key='ms2', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])


                peak_radius= [int(i) for i in list(self.peak_radius.split('-'))]

                # Perform peak detection
                ms1_peaks = deimos.peakpick.persistent_homology(ms1_smooth, index=self.index_ms1_peaks,
                                                                dims=[self.feature3, self.feature1, self.feature2],
                                                                radius=peak_radius)

                # Sort by persistence
                ms1_peaks = ms1_peaks.sort_values(by='persistence', ascending=False).reset_index(drop=True)

                # Save ms1 to new file
                deimos.save(os.path.join(self.file_folder,  "created_data", self.file_name_peak), ms1_peaks, key='ms1', mode='w')


                # Perform peak detection
                ms2_peaks = deimos.peakpick.persistent_homology(ms2_smooth, index=self.index_ms2_peaks,
                                                                dims=[self.feature3, self.feature1, self.feature2],
                                                                radius=peak_radius)

                # Sort by persistence
                ms2_peaks = ms2_peaks.sort_values(by='persistence', ascending=False).reset_index(drop=True)

                # Save ms2 to new file with _new_peak_data.h5 suffix
                deimos.save(os.path.join(self.file_folder,  "created_data", self.file_name_peak), ms2_peaks, key='ms2', mode='a')

            else:

                ms1_peaks = deimos.load(os.path.join(self.file_folder,  'small_peak1.h5'), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
      
            

        self.data_peak_ms1  = dd.from_pandas(ms1_peaks, npartitions=mp.cpu_count())
        self.data_peak_ms1.persist()

        return hv.Dataset(self.data_peak_ms1)


    def smooth_viewable(self, **kwargs):
        print('view smooth data')
        # dynamic map to return hvdata after loading it with deimos
        hvdata_smooth = hv.DynamicMap(self.create_smooth_data)

        # return the hvplot for mz and retention_time
        hvplot_md = hvdata_smooth.apply(self.hvplot_md)
        hvplot_dr = hvdata_smooth.apply(self.hvplot_dr)
        hvplot_rm = hvdata_smooth.apply(self.hvplot_rm)
        
        # stream to rasterize the plot. any change will cause the whole plot to reload
        md_stream= hv.streams.Params(
            self, ['feature3_axis_width', 'feature1_axis_width', 'min_feature3_bin_size', 'min_feature1_bin_size'], \
                rename = {'feature3_axis_width': 'x_filter', 'feature1_axis_width': 'y_filter', \
                    'min_feature3_bin_size': 'x_spacing', 'min_feature1_bin_size': 'y_spacing'})

        dr_stream = hv.streams.Params(
            self, ['feature1_axis_width', 'feature2_axis_width', 'min_feature1_bin_size', 'min_feature2_bin_size'], \
                rename = {'feature1_axis_width': 'x_filter', 'feature2_axis_width': 'y_filter', \
                    'min_feature1_bin_size': 'x_spacing', 'min_feature2_bin_size': 'y_spacing'})

        rm_stream = hv.streams.Params(
            self, ['feature2_axis_width', 'feature3_axis_width', 'min_feature2_bin_size', 'min_feature3_bin_size'], \
                rename = {'feature2_axis_width': 'x_filter', 'feature3_axis_width': 'y_filter', \
                    'min_feature2_bin_size': 'x_spacing', 'min_feature3_bin_size': 'y_spacing'})

        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy
        self.rasterized_md_smooth = hv.util.Dynamic(
            hvplot_md,
            operation=self.rasterize,
            streams=[hv.streams.RangeXY(), md_stream],
        )

        self.rasterized_dr_smooth = hv.util.Dynamic(
                    hvplot_dr,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), dr_stream],
                )

        self.rasterized_rm_smooth = hv.util.Dynamic(
                    hvplot_rm,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), rm_stream],
                )

        return ls2(self.rasterized_rm_smooth + self.rasterized_dr_smooth +  self.rasterized_md_smooth ).opts(shared_axes=True)


    def peak_viewable(self, **kwargs):
        # dynamic map to return hvdata after loading it with deimos
        print('view peak data')
        hvdata_peak = hv.DynamicMap(self.create_peak_data)

        # return the hvplot for mz and retention_time
        hvplot_md = hvdata_peak.apply(self.hvplot_md)
        hvplot_dr = hvdata_peak.apply(self.hvplot_dr)
        hvplot_rm = hvdata_peak.apply(self.hvplot_rm)
        
        # stream to rasterize the plot. any change will cause the whole plot to reload
        md_stream= hv.streams.Params(
            self, ['feature3_axis_width', 'feature1_axis_width', 'min_feature3_bin_size', 'min_feature1_bin_size'], \
                rename = {'feature3_axis_width': 'x_filter', 'feature1_axis_width': 'y_filter', \
                    'min_feature3_bin_size': 'x_spacing', 'min_feature1_bin_size': 'y_spacing'})

        dr_stream = hv.streams.Params(
            self, ['feature1_axis_width', 'feature2_axis_width', 'min_feature1_bin_size', 'min_feature2_bin_size'], \
                rename = {'feature1_axis_width': 'x_filter', 'feature2_axis_width': 'y_filter', \
                    'min_feature1_bin_size': 'x_spacing', 'min_feature2_bin_size': 'y_spacing'})

        rm_stream = hv.streams.Params(
            self, ['feature2_axis_width', 'feature3_axis_width', 'min_feature2_bin_size', 'min_feature3_bin_size'], \
                rename = {'feature2_axis_width': 'x_filter', 'feature3_axis_width': 'y_filter', \
                    'min_feature2_bin_size': 'x_spacing', 'min_feature3_bin_size': 'y_spacing'})

        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy
        self.rasterized_md_peak = hv.util.Dynamic(
            hvplot_md,
            operation=self.rasterize,
            streams=[hv.streams.RangeXY(), md_stream],
        )

        self.rasterized_dr_peak = hv.util.Dynamic(
                    hvplot_dr,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), dr_stream],
                )

        self.rasterized_rm_peak = hv.util.Dynamic(
                    hvplot_rm,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), rm_stream],
                )

        return ls3(self.rasterized_rm_peak + self.rasterized_dr_peak +  self.rasterized_md_peak ).opts(shared_axes=True)

    @pm.depends('run_decon')
    def load_decon_data(self):

        print('load new decon  data')
        self.ms1 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
        self.ms2 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms2', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])

        
        self.ms2_peaks = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_peak), key='ms2', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
        self.ms1_peaks = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_peak), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])

        ms1_peaks_dd = dd.from_pandas(self.ms1_peaks, npartitions=mp.cpu_count())
        ms1_peaks_dd.persist()

        return hv.Dataset(ms1_peaks_dd)
    
    @pm.depends('run_decon')
    def ms2_decon(self):
        print("run decon; shouldn't run")
        print(os.path.join(self.file_folder,  "created_data", self.file_name_peak))
        print(os.path.join(self.file_folder,  self.file_name_raw))

        self.file_name_res = self.file_name_raw[:-3] + self.file_name_peak[:-3].split('_threshold_')[-1] + self.file_name_smooth[:-3].split('_threshold_')[-1] + '_res.csv'
        
        if os.path.isfile(os.path.join(self.file_folder,  "created_data", self.file_name_res)):
            res = pd.read_csv(os.path.join(self.file_folder,  "created_data", self.file_name_res))
     
        else:
            # if using the default data, use easy filters, or haven't processed peak data yet from full
            if self.file_name_raw[:-3] + '_threshold_' + str(self.threshold_slider) + \
             '_peak_radius_' + str(self.peak_radius) +  "_feature2_" + str(self.feature2) +\
                '_new_peak_data.h5' != self.file_name_peak or not os.path.isfile(os.path.join(self.file_folder,  "created_data", self.file_name_peak)):

                res = pd.read_csv(os.path.join(self.file_folder,  "small_full1_res.csv"))


            else:
                threshold_peak_ms1 = 10000
                threshold_peak_ms2 = 1000
                threshold_full_m1 = int(self.threshold_slider)
                threshold_full_m2 = int(self.threshold_slider)
                require_ms1_greater_than_ms2 = True
                drift_score_min = True

                ms1 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
                ms2 = deimos.load(os.path.join(self.file_folder,  self.file_name_raw), key='ms2', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])

                
                ms2_peaks = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_peak), key='ms2', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
                ms1_peaks = deimos.load(os.path.join(self.file_folder,  "created_data", self.file_name_peak), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])



                # get thresholds of ms1 and ms2 peak and full data
                ms1 = deimos.threshold(ms1, threshold= threshold_full_m1)
                ms2 = deimos.threshold(ms2, threshold= threshold_full_m2)  

                
                ms1_peaks = deimos.threshold(ms1_peaks, threshold=threshold_peak_ms1)
                ms2_peaks = deimos.threshold(ms2_peaks, threshold=threshold_peak_ms2)


                decon = deimos.deconvolution.MS2Deconvolution(ms1_peaks, ms1, ms2_peaks, ms2)
                # use false .loc[res['drift_time_score']
                decon.construct_putative_pairs(dims=[self.feature1, self.feature2],
                                    low=[-0.12, -0.1], high=[1.4, 0.1], ce=20,
                                    model=offset_correction_model,
                                    require_ms1_greater_than_ms2=require_ms1_greater_than_ms2,
                                    error_tolerance=0.12)
                
                decon.configure_profile_extraction(dims=[self.feature3, self.feature1, self.feature2],
                                        low=[-200E-6, -0.05, -0.1],
                                        high=[600E-6, 0.05, 0.1],
                                        relative=[True, True, False])
            
                res = decon.apply(dims=self.feature1, resolution=0.01)

                if drift_score_min:
                    res = res.loc[res[self.feature1 + '_score'] > .9].groupby(by=[x for x in res.columns if x.endswith('_ms1')],
                                                            as_index=False).agg(list)
                else: 
                    res = res.groupby(by=[x for x in res.columns if x.endswith('_ms1')],
                                                            as_index=False).agg(list)
                
                res.to_csv( os.path.join(self.file_folder,  "created_data", self.file_name_res))
        return hv.Dataset(res)
        
# create the hv plots
    def hvplot_md_decon(self, ds):
        plot_df = ds.data.rename(columns = {self.feature_intensity + '_ms1': self.feature_intensity})
        element = plot_df.hvplot.points(x=self.feature3 + '_ms1', y=self.feature1 + '_ms1', c=self.feature_intensity)
        return element# create the hv plots
    
        # create the hv plots
    def hvplot_dr_decon(self, ds):
        plot_df = ds.data.rename(columns = {self.feature_intensity + '_ms1': self.feature_intensity})
        element = plot_df.hvplot.points(x=self.feature1 + '_ms1', y=self.feature2 + '_ms1', c=self.feature_intensity)
        return element
    
    def hvplot_rm_decon(self, ds):
        plot_df = ds.data.rename(columns = {self.feature_intensity + '_ms1': self.feature_intensity})
        element = plot_df.hvplot.points(x=self.feature2 + '_ms1', y=self.feature3 + '_ms1', c=self.feature_intensity)
        return element
    
    # input data is the selection on the  dt vs rt time for ms1 data plot
    def function_return_m2_subset_decon(self, ds, m1, d1, d2, r2, r3, m3):
        print('run return ms2 subset')
        print( m1, d1, d2, r2, r3, m3)
        if m1 != None and d1 != None:
            x_column_plot = self.feature3 + '_ms1'
            y_column_plot = self.feature1 + '_ms1'
            space_x = self.min_feature3_bin_size
            space_y = self.min_feature1_bin_size

        elif d2 != None and r2 != None:
            x_column_plot = self.feature1 + '_ms1'
            y_column_plot = self.feature2 + '_ms1'
            space_x = self.min_feature1_bin_size
            space_y = self.min_feature2_bin_size

        else:
            x_column_plot = self.feature2 + '_ms1'
            y_column_plot = self.feature3 + '_ms1'
            space_x = self.min_feature2_bin_size
            space_y = self.min_feature3_bin_size

        # should only be one range to use from y and x, else use none
        # using range of last selection
        y_list = [d1, r2, m3]
        list_y_range = [x for x in y_list if x!=None]
        if len(list_y_range) != 1:
            print('two filters y')
            y_range = None
        else:
            y_range = list_y_range[0]

        x_list = [m1, d2, r3]
        list_x_range = [x for x in x_list if x!=None]
        if len(list_x_range) != 1:
            x_range = None
        else:
            x_range = list_x_range[0]
        
        # get the decon plots if necessary
        self.md_decon.reset()
        self.dr_decon.reset()
        self.rm_decon.reset()

        #reset value to none
        m1, d1, d2, r2, r3, m3 = None, None, None, None, None, None

        res = ds.data

        def lit_list(x):
            try:
                return ast.literal_eval(str(x))   
            except Exception as e:
                print(e)
                return x
            
        # if string, convert to list. if list, string and back to list (does nothing)
        res[self.feature3 + '_ms2'] = res[self.feature3 + '_ms2'].apply(lambda x: lit_list(x))
        res[self.feature_intensity + '_ms2'] = res[self.feature_intensity + '_ms2'].apply(lambda x: lit_list(x))

        # just getting example of the first ms2 location data to use if haven't selected anything yet
        example_ms2= pd.DataFrame([[1,1],[2,2]], columns = [self.feature3, self.feature_intensity])
       
        # if no range selected 
        if (x_range == None or math.isnan(float(x_range)) or self.file_name_peak == 'placeholder_full.h5'
        ): 
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
                print(example_ms2.head())
                return hv.Dataset(example_ms2)
            else:
                # of the subset, get maximum intensity of ms1 data
                max_idx =  pd.to_numeric(ms2_subset[self.feature_intensity + '_ms1']).idxmax()
                if len(res.loc[max_idx, self.feature3 + '_ms2']) == 1:
                    # use location of maximum intensity ms1 data to get 
                    numpy_dataframe = np.hstack((np.array(res.loc[max_idx, self.feature3 + '_ms2']),
                    np.array(res.loc[max_idx, self.feature_intensity + '_ms2'])))
            
                    highest_ms2  = pd.DataFrame(numpy_dataframe.reshape(-1, len(numpy_dataframe)), columns = [self.feature3, self.feature_intensity])
                else:
                    highest_ms2 = pd.DataFrame({self.feature3: np.array(res.loc[max_idx, self.feature3 + '_ms2']),
                    self.feature_intensity: np.array(res.loc[max_idx, self.feature_intensity + '_ms2'])})
                print(x_range)
                print(y_range)
                print(highest_ms2.head())
                return hv.Dataset(highest_ms2)
 
    # create the hv plots with intenisty and ms2 data
    def hvplot_mi_decon(self, ds):
        print(ds.data)
        element2 = hv.Spikes( ds.data, self.feature3, self.feature_intensity).opts(framewise = True, width=600)
        return element2 

    # create the hv plots with intenisty and ms2 data


    def decon_viewable(self, **kwargs):
        print('return decon viewable')
        # dynamic map to return hvdata after loading it with deimos
        # trigger with 'run decon' button
       
        #get ms2 convoluted data from peak and ms2 data
        ms2_decon = hv.DynamicMap(self.ms2_decon)
       
        # return the hvplot for mz and retention_time
        hvplot_md = ms2_decon.apply(self.hvplot_md_decon)
        hvplot_dr = ms2_decon.apply(self.hvplot_dr_decon)
        hvplot_rm = ms2_decon.apply(self.hvplot_rm_decon)
        
        # stream to rasterize the plot. any change will cause the whole plot to reload
        md_stream= hv.streams.Params(
            self, ['feature3_axis_width', 'feature1_axis_width', 'min_feature3_bin_size', 'min_feature1_bin_size'], \
                rename = {'feature3_axis_width': 'x_filter', 'feature1_axis_width': 'y_filter', \
                    'min_feature3_bin_size': 'x_spacing', 'min_feature1_bin_size': 'y_spacing'})

        dr_stream = hv.streams.Params(
            self, ['feature1_axis_width', 'feature2_axis_width', 'min_feature1_bin_size', 'min_feature2_bin_size'], \
                rename = {'feature1_axis_width': 'x_filter', 'feature2_axis_width': 'y_filter', \
                    'min_feature1_bin_size': 'x_spacing', 'min_feature2_bin_size': 'y_spacing'})

        rm_stream = hv.streams.Params(
            self, ['feature2_axis_width', 'feature3_axis_width', 'min_feature2_bin_size', 'min_feature3_bin_size'], \
                rename = {'feature2_axis_width': 'x_filter', 'feature3_axis_width': 'y_filter', \
                    'min_feature2_bin_size': 'x_spacing', 'min_feature3_bin_size': 'y_spacing'})

        # normally would use rasterize(hv.DynamicMap(self.elem)) but using dynamic for a more complicated function
        # the changes to the x and y axis will also be an input due to rangexy
        self.rasterized_md_smooth = hv.util.Dynamic(
            hvplot_md,
            operation=self.rasterize,
            streams=[hv.streams.RangeXY(), md_stream],
        )

        self.rasterized_dr_smooth = hv.util.Dynamic(
                    hvplot_dr,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), dr_stream],
                )

        self.rasterized_rm_smooth = hv.util.Dynamic(
                    hvplot_rm,
                    operation=self.rasterize,
                    streams=[hv.streams.RangeXY(), rm_stream],
                )

        # xy is using the plot of ms1 data as the source of the pointer
        self.md_decon = hv.streams.Tap(source=self.rasterized_md_smooth, rename = {'x': 'm1', 'y': 'd1'})

        self.dr_decon = hv.streams.Tap(source=self.rasterized_dr_smooth, rename = {'x': 'd2', 'y': 'r2'})

        self.rm_decon = hv.streams.Tap(source=self.rasterized_rm_smooth, rename = {'x': 'r3', 'y': 'm3'})

        # resample from res output stream xy_dcon is updated
        filtered_ms2_data_decon = ms2_decon.apply(self.function_return_m2_subset_decon, streams=[self.md_decon, self.dr_decon, self.rm_decon])
        
        # make ms plot
        full_plot_1_mi_decon = hv.util.Dynamic(filtered_ms2_data_decon,  operation= self.hvplot_mi_decon)

        return hv.Layout(ls4(self.rasterized_md_smooth  + self.rasterized_dr_smooth + self.rasterized_rm_smooth).opts(shared_axes=False) + full_plot_1_mi_decon.opts(shared_axes=False)).cols(2)




class Align_plots(pm.Parameterized):
    
    full_ref = pm.String( default=peak_ref, label="Path to reference data")
    peak_folder =  pm.String( default=peak_folder, label= "Path to folder of files to align to reference")
    file_folder =  pm.String(
        default=data_folder, doc='A string', label='Location of data folder'
    )
    align_endswith =  pm.String(
        default=align_endswith, doc='A string', label='Only use files that end with this value'
    )
    tolerance_text = pm.String(default = '.00002-0.015-2', label = 'Tolerances by ' + feature3 + ', ' + feature1 + ', and ' + feature2)
    relative_text = pm.String(default = 'True-True-False', label  = 'Relevant or abs val by ' + feature3 + ', ' + feature1 + ', and ' + feature2)
    menu_kernel = pm.Selector(['linear',  'rbf'], label = 'Support Vector Regression Kernel')
    threshold_text = pm.String(default = '2000', label  = 'Threshold')


    
    feature1 = pm.String(default = feature1, label = 'Feature 1')
    feature2 = pm.String(default = feature2, label = 'Feature 2')
    feature3 = pm.String(default = feature3, label = 'Feature 3')
    feature_intensity = pm.String(default = feature_intensity, label = 'Intensity Feature')

    @pm.depends('full_ref', 'peak_folder', 'file_folder', 'align_endswith', 'tolerance_text', 'relative_text', 'menu_kernel',  'threshold_text', 'feature1', 'feature2', 'feature3', 'feature_intensity')
    def viewable(self):
        # if just test, lower threshold
        if self.full_ref == "small_full1.h5" or self.align_endswith == "example.h5":
            self.threshold_text = "0"
        tolerance_text = [float(i) for i in list(self.tolerance_text.split('-'))]
        relative_text = [bool(i) for i in list(self.relative_text.split('-'))]
        try:
            full_ref = deimos.load(os.path.join(self.file_folder, self.full_ref), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
        except:
            # old load
            ms1 = deimos.io._load_hdf(os.path.join(self.file_folder, self.full_ref), level='ms1')
            ms2 = deimos.io._load_hdf(os.path.join(self.file_folder, self.full_ref), level='ms2')
            
            # new save
            deimos.io.save_hdf(os.path.join(self.file_folder, self.full_ref), ms1, key='ms1', mode='w')
            deimos.io.save_hdf(os.path.join(self.file_folder, self.full_ref), ms2, key='ms2', mode='a')
            
            # new load
            full_ref = deimos.load(os.path.join(self.file_folder, self.full_ref), key='ms1')

        theshold_presistence =  int(self.threshold_text)
        

        full_ref.rename(columns = {'inverse_reduced_ion_mobility': self.feature2}, inplace = True)

        peak_ref  = deimos.peakpick.persistent_homology(deimos.threshold(full_ref,  threshold = theshold_presistence),
                                                 dims=[self.feature3, self.feature1, self.feature2])
        
        peak_ref['persistence_ratio'] = peak_ref['persistence'] / peak_ref[self.feature_intensity]
        peak_ref = deimos.threshold(peak_ref, by='persistence_ratio', threshold=0.75)


        peak_two_list = []
        peak_file_list = []
        for file in os.listdir(self.peak_folder):
            if file.endswith(self.align_endswith):
                try:
                    full_two = deimos.load(os.path.join(self.peak_folder,file), key='ms1', columns=[self.feature3, self.feature1, self.feature2, self.feature_intensity])
                except:
                    # old load
                    ms1 = deimos.io._load_hdf(os.path.join(self.peak_folder,file), level='ms1')
                    
                    # new save
                    deimos.io.save_hdf(file, ms1, key='ms1', mode='w')
                    
                    # new load
                    full_two = deimos.load(file, key='ms1')

                full_two.rename(columns = {'inverse_reduced_ion_mobility': self.feature2}, inplace = True)
                full_two.rename(columns = {'inverse_reduced_ion_mobility': self.feature2}, inplace = True)

                peak_two  = deimos.peakpick.persistent_homology(deimos.threshold(full_two,  threshold =  theshold_presistence),
                                                            dims=[self.feature3, self.feature1, self.feature2])
                
                peak_two['persistence_ratio'] = peak_two['persistence'] / peak_two[self.feature_intensity]
                peak_two = deimos.threshold(peak_two, by='persistence_ratio', threshold=0.75)

                peak_two_list.append(peak_two)
                peak_file_list.append(file)
    
        # deimos
        # deimos.collapse(df, keep=mz)

        list_plots = []
        i = 0
        for peak_two, peak_file in zip(peak_two_list, peak_file_list):
            # b is reference, a is peak two
            partitions = deimos.partition(deimos.threshold(peak_two, threshold=1E3),
            split_on=self.feature3,
            size=1000,
            overlap=0.25)


            two_matched, ref_matched = partitions.zipmap(deimos.alignment.match, deimos.threshold(peak_ref, threshold=int(self.threshold_text)),
                                        dims=[self.feature3, self.feature1, self.feature2],
                                        tol=tolerance_text, relative=relative_text,
                                        processes=4)
            two_matched = two_matched.copy()
            two_matched_aligned = two_matched.copy()
            i= i + 1
            for dim in [self.feature1, self.feature2]:
               
                
                spl = deimos.alignment.fit_spline( two_matched, ref_matched, align= dim, kernel=self.menu_kernel, C=1000)
                newx = np.linspace(0, ref_matched[ dim].max(), 1000)

                two_matched_aligned["aligned_" + dim] = spl(two_matched_aligned[dim])
                # save by peak in peak name
                two_matched_aligned.to_csv( os.path.join(self.file_folder,  "created_data", peak_file[:-3] + "_aligned.csv"))

                    # match_table includes the matched data from data a and b to compare with scatter plot (data a retention time vs data b retention time)
                matchtable = pd.concat(
                    [
                        two_matched[[ dim]].reset_index(drop=True),
                        ref_matched[[ dim]].reset_index(drop=True)
                    ],
                    axis=1,
                )

                matchtable.columns = [
                    'match_a_' + dim,
                   'match_b_' + dim
                ]

                xy_drift_retention_time = pd.DataFrame(
                    np.hstack(
                        (
                            newx[:, None], # spline applied to matching
                            spl(newx)[:, None],
                        )
                    )
                )

                # rename columns
                xy_drift_retention_time.columns = [
                    'x_' + dim,
                    'y_' + dim
                ]


                plot1 = hv.Points(xy_drift_retention_time, kdims=['x_' + dim, 'y_' + dim]).options(color='blue')
                plot2 = hv.Points(matchtable, kdims=['match_a_' + dim, 'match_b_' + dim]).options(color='red')
        
                list_plots.append((plot1 * plot2).opts(opts.Overlay(title=dim + ' vs ' + dim + ' ' + str(i))))

        return hv.Layout(list_plots).cols(2)

if __name__ == '__main__':
    Deimos_app = Deimos_app()
    Align_plots = Align_plots()

    #using viewable() would be best practice for minimizing refresh, but need hard refresh in this case for new axis
    instructions_view = "<ul> <li>The original data is a placeholder</li> <li>Indicate the local path of the full data below to update</li><li>Use the box selector (as seen on the bottom) to filter data in all plots based on the box's range</li>\
        <li> Data in folder</li><ul> "
    instructions_smooth = "<ul> <li>The original data is a placeholder</li> <li>Click 'Run smooth' after updating parameters to get new graph</li><li>Use the box selector (as seen on the bottom) to filter data in all plots based on the box's range</li>\
        <li>Keeping the <b>smooth radius</b> small and increasing number of iterations <br> is preferable to a larger smoothing radius, albeit at greater computational expense.</li><ul> "
    instructions_peaks = "<p>Feature detection, also referred to as peak detection, is the process by which local maxima that fulfill certain criteria (such as sufficient signal-to-noise ratio) are located in the signal acquired by a given analytical instrument. </p><ul> <li>The original data is a placeholder</li> <li>Click 'Run peak' after updating parameters to get new graph</li><li>Use the box selector (as seen on the bottom) to filter data in all plots based on the box's range</li> \
        <li>The <b>radius per dimension</b> insures an intensity-weighted per-dimension coordinate will be returned for each feature.</li><ul> "
    instructions_ms2 = "<p>With MS1 features of interest determined by peak detection, corresponding tandem mass spectra, if available, must be extracted and assigned to the MS1 parent ion feature. </p><ul> <li>The original data is a placeholder, clicking will not work without real data </li> <li>Click 'Run decon' after updating parameters to get new graph</li><li>Click the left plot to view highest intensity of MS2 within the minimum spacing of each feature at the click location </li> </ul> "
    instructions_align = "<ul><li>Alignment is the process by which feature coordinates across samples are adjusted to account for instrument variation such that matching features are aligned to adjust for small differences in coordinates</li>\
        <li>The original data is a placeholder</li> <li>Indicate the reference file and folder of files to align</li><li>Determine matches within <b>tolerance</b> per feature with the alignment determined by the <b>kernel</b> by <b>relative or absolute </b> value by <b>support vector regression kernel </b> </li></ul>"

    param_full = pn.Column('<b>View Raw Data</b>', Deimos_app.param.file_name_raw, Deimos_app.param.reset_filter, Deimos_app.param.manual_filter,
                        Deimos_app.param.feature1_axis_width, Deimos_app.param.feature2_axis_width, Deimos_app.param.feature3_axis_width, \
                            Deimos_app.param.min_feature1_bin_size, Deimos_app.param.min_feature2_bin_size, Deimos_app.param.min_feature3_bin_size, \
                                Deimos_app.param.feature1, Deimos_app.param.feature2, Deimos_app.param.feature3, Deimos_app.param.feature_intensity)

    param_smooth = pn.Column('<b>Smooth</b>', Deimos_app.param.file_name_raw,  Deimos_app.param.threshold_slider, Deimos_app.param.smooth_radius, Deimos_app.param.smooth_iterations,  Deimos_app.param.run_smooth, Deimos_app.param.file_name_smooth)

    param_peak = pn.Column('<b>Peak-picking</b>', Deimos_app.param.file_name_smooth, Deimos_app.param.peak_radius, Deimos_app.param.run_peak, Deimos_app.param.file_name_peak)

    param_decon = pn.Column('<b>MS2 Deconvolution</b>', Deimos_app.param.file_name_raw, Deimos_app.param.file_name_peak, Deimos_app.param.run_decon)



    app1 = pn.Tabs(('Filter Graphs', pn.Row(pn.Column(instructions_view, pn.pane.PNG('box_select.png'),  pn.Row(param_full, pn.Column(Deimos_app.raw_viewable())
        )))),('Smoothing', pn.Row(pn.Column(instructions_smooth, pn.pane.PNG('box_select.png'),  pn.Row(param_smooth, Deimos_app.smooth_viewable(),  
        )))),('Peak Detection', pn.Row(pn.Column(instructions_peaks, pn.pane.PNG('box_select.png'),  pn.Row(param_peak, Deimos_app.peak_viewable(),  
        )))), ('Deconvolution', pn.Row(pn.Column(instructions_ms2,  pn.pane.PNG('box_select.png'), pn.Row( param_decon, Deimos_app.decon_viewable(),  
        )))), ('Align Plots', pn.Row(pn.Column(instructions_align, pn.Row(Align_plots.param, Align_plots.viewable))))).servable(title='Deimos App')



    pn.serve(app1)
