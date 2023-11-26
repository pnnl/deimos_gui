# Adjust Visualization

## Filtering

Check "set axis width and bin values based on axis below" to filter with
user-input "Axis-width"

![Graphical user interface, text, application Description automatically
generated](./myMediaFolder/media/image1.png)


If the box isn't checked, zooming is possible via the plot input "zoom"

![Icon Description automatically
generated](./myMediaFolder/media/image2.png)


## Minimum grid size

Min spacing input sets the minimum size of the grid during rasterizing

> ![Application Description automatically generated with medium
> confidence](./myMediaFolder/media/image3.png)

>
> Output:

![](./myMediaFolder/media/image4.png)


![Application Description automatically generated with medium
confidence](./myMediaFolder/media/image5.png)


Output:

![](./myMediaFolder/media/image6.png)


You can zoom into new areas of the chart with the user-input widgets,
and the aggregation and colormap level will automatically update,
allowing you to inspect the data on an overview level or a focused area.
We disabled the re-aggregation when you zoom using the toolbar "zoom" to
only trigger re-aggregation if the user choses to refresh the plots with
the manual axis widths

## Linked Selection

Linked Selection will filter on the underlying data, even if they aren't
used in the plot

![](./myMediaFolder/media/image7.emf)


## Column Name Selection

The names of the data can be updated here:

![Graphical user interface, application Description automatically
generated](./myMediaFolder/media/image8.png)


If the initial data is mzML, the user needs to select the names from the
file to use as the retention and drift time. If there is not mzML data,
the input here doesn't matter.

![Graphical user interface, text, application Description automatically
generated](./myMediaFolder/media/image9.png)


The new h5 file created from the mzML file will be located in a folder
called "created_data" next to the deimos_gui folder. All other output
from the GUI will be saved here.

# Selecting Data

Select the files from the folder indicated in the "location of data
folder" folder

![Graphical user interface, text, application Description automatically
generated](./myMediaFolder/media/image10.png)


![Graphical user interface, text, application Description automatically
generated](./myMediaFolder/media/image11.png)


The initial data is a placeholder

# Smooth data: 

DEIMoS guide:
https://deimos.readthedocs.io/en/latest/user_guide/peak_detection.html

The original data is a placeholder. Unchecking it will run rerun the
function with the selected data.

Click \'Run smooth\' after updating parameters to get new graph.

Keeping the smooth radius small and increasing number of iterations is
preferable to a larger smoothing radius, albeit at greater computational
expense

The new smooth data file will be located in a folder called
"created_data" next to the deimos_gui folder. All other output from the
GUI will be saved here.

This fille will be re-used if the parameter inputs are the same in a
rerun to save computing time.

![Graphical user interface Description automatically generated with
medium
confidence](./myMediaFolder/media/image12.png)


![Graphical user interface, text, application, email Description
automatically
generated](./myMediaFolder/media/image13.png)

# Peak data:

DEIMoS guide:
https://deimos.readthedocs.io/en/latest/user_guide/peak_detection.html

Feature detection, also referred to as peak detection, is the process by
which local maxima that fulfill certain criteria (such as sufficient
signal-to-noise ratio) are located in the signal acquired by a given
analytical instrument.

The original data is a placeholder and unclicking it will trigger the
peak function to run.

Click \'Run peak\' after updating parameters to get new graph.

The radius per dimension insures an intensity-weighted per-dimension
coordinate will be returned for each feature

Threshold sets the persistence and persistence ratio

The new peak data file will be located in a folder called "created_data"
next to the deimos_gui folder. All other output from the GUI will be
saved here.

This fille will be re-used if the parameter inputs are the same in a
rerun to save computing time.

![Graphical user interface, diagram Description automatically generated
with medium
confidence](./myMediaFolder/media/image14.png)


![Graphical user interface, text, application Description automatically
generated](./myMediaFolder/media/image15.png)


MS deconvolution

DEIMoS guide:
https://deimos.readthedocs.io/en/latest/user_guide/ms2_extraction.html

With MS1 features of interest determined by peak detection,
corresponding tandem mass spectra, if available, must be extracted and
assigned to the MS1 parent ion feature.

The original data is a placeholder, clicking with the placeholder will
only return random data.

Click \'Run decon\' after updating parameters to get new graph.

The parameters are the initial data and the peak data created in steps
1-3 in the application.

The MS2 data associated with user-selected MS1 data, with the MS1 data
with the highest intensity used if there are multiple MS1 data points
within a small range of the user-click

The decon file is saved in a folder called "created_data" next to the
deimos_gui file and will be re-used if the parameter inputs are the same
in a rerun to save computing time.

![Chart Description automatically
generated](./myMediaFolder/media/image16.png)


![Graphical user interface, text, application Description automatically
generated](./myMediaFolder/media/image17.png)


# Calibration

DEIMoS guide:
https://deimos.readthedocs.io/en/latest/user_guide/ccs_calibration.html

The calibrated file will be saved the "created data folder". The data
file must include mz, ccs, charge, and, if not tune mix, a ta column.

The three possible ways to calculate the calibration is to

Load all values:

Make sure calibration input has columns mz, ccs, charge, and ta

file to calibrate\' file has columns mz and drift time

use_tunemix:

Make sure calibration input has columns mz, ccs, and charge

mz, drift_time and intensity in the tune file

mz and drift_time in the \'file to calibrate\'

fixed_parameters:

Provide a beta and tfix value

For all, you can chose to use travelling wave IMS, where the
relationship between measurement and CCS must be linearized by the
natural logarithm, then fit by linear regression.

![Chart, scatter chart Description automatically
generated](./myMediaFolder/media/image18.png)


![Graphical user interface, application Description automatically
generated](./myMediaFolder/media/image19.png)


# Isotopes

DEIMoS guide:
https://deimos.readthedocs.io/en/latest/user_guide/isotope_detection.html

Select a row in the data-frame to view the isotopes and a slice of the
data around the isotopes, with the range of the slices determined by the
user inputs for slice size

![Chart Description automatically generated with low
confidence](./myMediaFolder/media/image20.png)


![Chart Description automatically generated with low
confidence](./myMediaFolder/media/image20.png)


![Graphical user interface, text, application Description automatically
generated](./myMediaFolder/media/image21.png)


# Align

DEIMoS Guide:
<https://deimos.readthedocs.io/en/latest/user_guide/alignment.html>

Update the name of the file to use as the reference

Update the peak folder location and indicate (with \* as the wildcard)
the ending of the files to use.

Alignment is the process by which feature coordinates across samples are
adjusted to account for instrument variation such that matching features
are aligned to adjust for small differences in coordinates

Determine matches within tolerance per feature with the alignment
determined by the kernel by relative or absolute value by support vector
regression kernel.

![Chart, scatter chart Description automatically
generated](./myMediaFolder/media/image22.png)


![Graphical user interface, text, application, email Description
automatically
generated](./myMediaFolder/media/image23.png)


Extra:

"bokeh.core.serialization.DeserializationError: can\'t resolve reference
warning" seem to occur when the align plots are updated. According to
<https://discourse.bokeh.org/t/bug-with-deserializationerror-when-changing-content-layout-in-bokeh-3-1-1/10566>
and <https://github.com/bokeh/bokeh/issues/13229>, it occurs when a
layout is updated with new plots, and it is usually not something to be
concerned about.
