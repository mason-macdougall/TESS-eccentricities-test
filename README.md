# TESS-eccentricities-test

This is the complete code for my photo-eccentric model, 
including data download, light curve processing, 
transit modeling, eccentricity re-sampling, 
plotting both input data and results, and storing the results. 
While the code works well for its purpose, 
it still needs significant overhaul before making it accessible to others. 
This version is a single-system example version for HIP-97166,
with 1 transiting planet at 10.3 days. 

The impact parameter for this planet is poorly fit, 
and this leads to a significant overestimation of eccentricity. 
I'm currently working on figuring out what leads to this discrepancy 
and whether or not it can be mitigated in a self-contained way. 
When the code and data are downloaded by a user, 
the paths within the code will need to be changed to fit the user's directory setup. 
I'm still learning more about true python software development, 
but hopefully I will have a more professional set-up soon!

The construction and implementation of the transit model occurs between
lines 1105 and 1545. Before this is data download and lightcurve processing.
After this is plotting and saving the results along with performing
photo-eccentric resampling procedures to determine eccentricity constraints.

Once the data and code are downloaded and the directories are changed,
the code can be run simply with terminal command:

python3 photoecc_mod.py

A variety of different python modules are needed for the complete use of this code:

lightkurve
exoplanet
pymc3
theano
seaborn
scipy
astropy
ldtk
numpy
uncertainties
sklearn
pandas
matplotlib
chainconsumer
