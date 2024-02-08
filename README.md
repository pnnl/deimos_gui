DEIMoS Visualization
=======

Version 1.0.3

## User Guide
[User Guide](user_guide_deimos.md)


## Clone Repository

``` 

git clone https://github.com/pnnl/deimos_gui
``` 

Create environment
``` 
conda create -n deimos_env python=3.10
conda activate deimos_env
``` 
Install DEIMoS and DEIMoS GUI
``` 
cd deimos_gui
pip install -e .
```


## Run app from src folder

Run app (from deimos_gui/src folder).

Copy/paste the resulting address after 'Bokeh app running at:' in your browser
```
cd src
panel serve run_app.py
# Copy/paste the resulting address after 'Bokeh app running at:' in your browser
```


Use example data (rather than placeholder data)

Follow instructions here to download the data: 
https://deimos.readthedocs.io/en/latest/getting_started/example_data.html
Use ftp://massive.ucsd.edu/v01/MSV000091746 as the FTP Download Link

Provide path to data within the application (default: data folder within DEIMOS_GUI folder)


## Funding
This research was supported by the National Institutes of Health, National Institute of Environmental Health Sciences grant U2CES030170 and is a contribution of the Pacific Northwest Advanced Compound Identification Core. A portion of the work was conducted under the Laboratory Directed Research and Development Program at Pacific Northwest National Laboratory (PNNL). PNNL is a multi-program national laboratory operated by Battelle for the DOE under Contract DE-AC05-76RLO 1830.

Citing DEIMoS
-------------
If you would like to reference DEIMoS in an academic paper, we ask you include the following:
* DEIMoS, version 1.3.3 http://github.com/pnnl/deimos (accessed MMM YYYY)

## Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830

## Simplified BSD
____________________________________________
Copyright 2023 Battelle Memorial Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

