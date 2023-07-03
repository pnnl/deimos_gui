DEIMoS Visualization
=======

## Variables
Indicate 
* the file name of the raw data of interest (full1)
* peak data placeholder, which will be replaced with the actual peak data (peak1)
* the reference peak data the other files will align with (peak_ref), * the folder of the files you will align (peak_folder)
* the folder where the raw data will be located and where outputs of the application will be saved under "created_data" (data_folder)
* the string end of the files used in the folder of files you will align (align_endswith) in the first lines of the main.py file.  
You can also set the path of the data after launching the application
```
full1 = 'small_full1.h5'
peak1 = 'small_peak1.h5'

peak_ref = 'small_full1.h5'

peak_folder = 'data/peak_folder/'
data_folder = 'data/'
align_endswith = "data.h5"
```

This is also where you can set the minimum tile size for the vizualizations.
```
drift_spacing = 0.15
retention_spacing = 0.15
mz_spacing = 0.02
```

## Install DEIMoS

Follow all directions at https://deimos.readthedocs.io/en/latest/getting_started/installation.html


## Install additonal packages

``` 
conda install colorcet holoviews panel xarray hvplot datashader pandas
```

In the terminal, cd to the the downloaded folder
```
cd /path/to/downloaded/folder
```

Run app
```
python run_app.py
```


Citing DEIMoS
-------------
If you would like to reference DEIMoS in an academic paper, we ask you include the following:
* DEIMoS, version 0.1.0 http://github.com/pnnl/deimos (accessed MMM YYYY)

Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830


