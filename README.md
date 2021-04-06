# TESS-eccentricities

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
