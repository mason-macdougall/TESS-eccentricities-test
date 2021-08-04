#!/u/home/m/macdouga/miniconda3/bin/python3

# Set up system requirements
import sys
import os, fnmatch

sys.path.insert(0,'/usr/share/texmf/tex:/usr/share/texmf/tex/latex:/u/local/compilers/gcc/7.2.0/bin:/u/home/m/macdouga/miniconda3/bin:/u/home/m/macdouga/miniconda3/condabcondabin:/u/home/m/macdouga/miniconda3/pkgs/texlive-core-20180414-pl526h89d1741_1/bin:/u/home/m/macdouga/miniconda3/pkgs/texlive-core-20180414-pl526h89d1741_1')
# base_compiledir=/tmp/christian/theano.NOBACKUP,
os.environ["THEANO_FLAGS"] = "allow_gc=True,scan.allow_gc=True,scan.allow_output_prealloc=False"
print('os')
#####################################################

# Lightcurve Fitting
import lightkurve as lk
print('lk')
import exoplanet as xo
print('xo')

# Statistical & Math Tools
import pymc3 as pm
print('pm')
import theano
print('th')
theano.config.allow_gc=True
theano.config.scan.allow_gc=True
theano.config.scan.allow_output_prealloc=False
print('flags')
import seaborn as sb
import theano.tensor as tt
import scipy
import scipy.stats as sstat
from scipy import stats
from scipy.signal import savgol_filter
from astropy.timeseries import BoxLeastSquares
from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter, tess
import numpy as np
from numpy.random import normal, multivariate_normal
import uncertainties
from uncertainties import ufloat
import arviz as az
print('math')
# Data Structure
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
from tqdm import tqdm
from chainconsumer import ChainConsumer
import chainconsumer.analysis
print('data')
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import corner
import seaborn as sns
#from cgs import *
from scipy.stats import kde as kdest
from sklearn.neighbors import KernelDensity
print('plot')
def e_w_function(input_list):
    # Takes input list in the form of [secosw, sesinw]
    # Outputs f(e,w) = rho_circ/rho_obs
    # If result is complex or invalid, outputs -np.inf as result
    
    e = input_list[0]**2 + input_list[1]**2
    ratio = (1+(e**(1/2))*input_list[1])**3/(1-e**2)**(3/2)
    if type(ratio) == complex or str(ratio) == 'nan':
        ratio = -np.inf 
    return ratio

# General Constants
G = 6.6743 * 10**(-8)            # cm^3 / (g * s^2)
G_err = 1.5 * 10**(-12)

msun = 1.988409870698051*10**33           # g
msun_err = 4.468805426856864*10**28

rsun = 6.957 * 10**10           # cm
rsun_err = 1.4 * 10**7

rearth = 6.378137 * 10**8         # cm
rearth_err = 1.0 * 10**4

G_u = ufloat(G, G_err)
msun_u = ufloat(msun, msun_err)
rsun_u = ufloat(rsun, rsun_err)
rearth_u = ufloat(rearth, rearth_err)

rhosun_u = msun_u / (4/3*np.pi * rsun_u**3)
rhosun = rhosun_u.n
rhosun_err = rhosun_u.s           # g / cc

day = 86400                       # seconds






#####################################################


date = '30Jun21'

# Database for all TESS candidates import and relavent paths
lc_path = '/u/scratch/m/macdouga/tess_ecc_test-ecc/'
xl_path = '/u/home/m/macdouga/hmc_scripts/'
file_name = "tess_ecc_mod-inputs-" + '21Jan21' + ".xlsx"
sheet_name = "Sheet1"

all_data = pd.read_excel(xl_path + file_name, sheet_name=sheet_name)

tess_pl = all_data




#####################################################





pl = int(sys.argv[1])

pl -= 1

# Establish system ID (TIC, TOI, planet ID within system)
tic_tev = all_data["tic-toi"][pl]
tic = int(tic_tev)
host_tic = str(tic)


# Determine if multi-system
tess_tic = list(all_data["tic-toi"])
if tess_tic.count(tic_tev) > 1:
    print('\nMultiplanet system! ')
    
system_data = all_data.loc[all_data["tic-toi"]==tic_tev]

full_toi_ids = list(system_data["full_toi_id"])
print(full_toi_ids)

pls = []
pl_ids = []
pl_names = []
candidate_ids = []

true_vals_all = []
rp_rs_all = []

terminates = []
termination_codes = []

tks_flags = []

for full_toi_id in full_toi_ids:
    
    pl = all_data.loc[all_data["full_toi_id"]==full_toi_id].index[0]


    full_toi_id = str(full_toi_id)
    toi = full_toi_id.split('.')[0]
    host_toi = str(toi)

    pl_num = int(full_toi_id.split('.')[1])
    letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    pl_id = letters[pl_num - 1]

    # General ID string
    candidate_id = 'TIC ' + host_tic + pl_id + ' / TOI ' + host_toi + pl_id


    ########################################################################


    pl_name = 'TOI' + host_toi + pl_id + '_TIC' + host_tic + pl_id
    sys_name = 'TOI' + host_toi + '_TIC' + host_tic


    print('\n' + 'Beginning process for: ' + candidate_id)
    print("idx = " + str(pl) + '\n')


    dec_true = float(tess_pl["dec"][pl])
    ra_true = float(tess_pl["ra"][pl])
    tmag_true = float(tess_pl["t_mag"][pl])
    vmag_true = float(tess_pl["v_mag"][pl])

    t0_true = float(tess_pl["epoch"][pl])
    t0_err_true = float(tess_pl["epoch"][pl])
    per_true = float(tess_pl["period"][pl])
    per_err_true = float(tess_pl["period"][pl])

    dur_true = float(tess_pl["dur"][pl]) / 24.0
    dur_err_true = float(tess_pl["dur_err"][pl]) / 24.0
    dep_true = float(tess_pl["depth"][pl]) * 1e-6
    dep_err_true = float(tess_pl["depth_err"][pl]) * 1e-6

    rp_true = float(tess_pl["rp_exp"][pl])
    rp_err_true = float(tess_pl["rp_exp_err"][pl])

    evol_true = tess_pl["evol"][pl]
    disp_true = tess_pl["disp"][pl]
    snr_true = float(tess_pl["snr"][pl])
    
    if float(tess_pl["rad-iso_teff"][pl]) > 0:
        source_st = 1
        rstar_true = float(tess_pl["rad-iso_teff"][pl])
        rstar_err_true = np.mean([float(tess_pl["rad_err1-iso_teff"][pl]), -float(tess_pl["rad_err2-iso_teff"][pl])])

        mstar_true = float(tess_pl["mass-iso_teff"][pl])
        mstar_err_true = np.mean([float(tess_pl["mass_err1-iso_teff"][pl]), -float(tess_pl["mass_err2-iso_teff"][pl])])

        teff_true = float(tess_pl["teff-iso_teff"][pl])
        teff_err_true = np.mean([float(tess_pl["teff_err1-iso_teff"][pl]), -float(tess_pl["teff_err2-iso_teff"][pl])])

        rhostar_true = float(tess_pl["rho-iso_teff"][pl])
        rhostar_err_true = np.mean([float(tess_pl["rho_err1-iso_teff"][pl]), -float(tess_pl["rho_err2-iso_teff"][pl])])

        feh_true = float(tess_pl["feh-iso_teff"][pl])
        feh_err_true = np.mean([float(tess_pl["feh_err1-iso_teff"][pl]), -float(tess_pl["feh_err2-iso_teff"][pl])])

        logg_true = float(tess_pl["logg-iso_teff"][pl])
        logg_err_true = np.mean([float(tess_pl["logg_err1-iso_teff"][pl]), -float(tess_pl["logg_err2-iso_teff"][pl])])

    elif float(tess_pl["rad-iso_color"][pl]) > 0:
        source_st = 2
        rstar_true = float(tess_pl["rad-iso_color"][pl])
        rstar_err_true = np.mean([float(tess_pl["rad_err1-iso_color"][pl]), -float(tess_pl["rad_err2-iso_color"][pl])])

        mstar_true = float(tess_pl["mass-iso_color"][pl])
        mstar_err_true = np.mean([float(tess_pl["mass_err1-iso_color"][pl]), -float(tess_pl["mass_err2-iso_color"][pl])])

        teff_true = float(tess_pl["teff-iso_color"][pl])
        teff_err_true = np.mean([float(tess_pl["teff_err1-iso_color"][pl]), -float(tess_pl["teff_err2-iso_color"][pl])])

        rhostar_true = float(tess_pl["rho-iso_color"][pl])
        rhostar_err_true = np.mean([float(tess_pl["rho_err1-iso_color"][pl]), -float(tess_pl["rho_err2-iso_color"][pl])])

        feh_true = float(tess_pl["feh-iso_color"][pl])
        feh_err_true = np.mean([float(tess_pl["feh_err1-iso_color"][pl]), -float(tess_pl["feh_err2-iso_color"][pl])])

        logg_true = float(tess_pl["logg-iso_color"][pl])
        logg_err_true = np.mean([float(tess_pl["logg_err1-iso_color"][pl]), -float(tess_pl["logg_err2-iso_color"][pl])])
    else:
        source_st = 3
        rstar_true = float(tess_pl["tic_rs"][pl])
        rstar_err_true = float(tess_pl["tic_rs_err"][pl])

        mstar_true = float(tess_pl["tic_ms"][pl])
        mstar_err_true = float(tess_pl["tic_ms_err"][pl])

        teff_true = float(tess_pl["tic_teff"][pl])
        teff_err_true = float(tess_pl["tic_teff_err"][pl])

        mstar_true_u = ufloat(mstar_true, mstar_err_true)
        rstar_true_u = ufloat(rstar_true, rstar_err_true)
        rhostar_true_u = mstar_true_u / (4./3. * np.pi * rstar_true_u**3)
        rhostar_true = float(rhostar_true_u.n)
        rhostar_err_true = float(rhostar_true_u.s)

        feh_true = float(tess_pl["tic_feh"][pl])
        feh_err_true = float(tess_pl["tic_feh_err"][pl])

        logg_true = float(tess_pl["tic_logg"][pl])
        logg_err_true = float(tess_pl["tic_logg_err"][pl])

    rhostar_case_u = ufloat(rhostar_true, rhostar_err_true)          # rho_sun (CHECK UNITS OF INPUT SOURCE)

    rhostar_case_unc = rhostar_case_u * rhosun_u
    rhostar_case = float(rhostar_case_unc.n)
    rhostar_err_case = float(rhostar_case_unc.s)             # g / cm^3

    rhostar_true = rhostar_case
    rhostar_err_true = rhostar_err_case
    rhostar_true_err = rhostar_err_true

    true_vals = [dec_true, ra_true, tmag_true, vmag_true, t0_true, t0_err_true, per_true, per_err_true,
    dur_true, dur_err_true, dep_true, dep_err_true, rstar_true, rstar_err_true,
    rp_true, rp_err_true, teff_true, teff_err_true,
    rhostar_true, rhostar_err_true, 0, evol_true, 0, snr_true, disp_true, mstar_true]

    print(true_vals)


    terminate = []

    if float(dec_true) < -20 or float(dec_true) == -99.0 or str(dec_true) == 'nan':
        print('\nDeclination issue: dec_true < -20 or no dec_true value\nDeclination value: ' + str(dec_true))
        terminate.append(1)

    if float(vmag_true) > 13:
        print('\nVmag issue: vmag_true > 13\nVmag value: ' + str(vmag_true))
        terminate.append(2)

    elif float(vmag_true) == -99 and (float(tmag_true) > 13 or float(tmag_true) == -99):
        print('\nTmag issue: vmag_true == -99 and (tmag_true > 13 or tmag_true == -99)\nTmag value: ' + str(tmag_true))
        terminate.append(2)

    if float(per_true) == -99 or float(per_err_true) == -99:
        print('\nPeriod issue: no per_true or no per_err_true\nPeriod value: ' + str(per_true) + ' [' + str(per_err_true) + ']')
        terminate.append(3)

    elif float(t0_true) == -99:
        print('\nt0 issue: t0_true == -99\nt0 value: ' + str(t0_true))
        terminate.append(3)

    elif float(dur_true) == -99:
        print('\nDuration issue: dur_true == -99\nDuration value: ' + str(dur_true))
        terminate.append(3)

    elif float(dep_true) == -99:
        print('\nDepth issue: dep_true == -99\nDepth value: ' + str(dep_true))
        terminate.append(3)

    if float(snr_true) < 10.0:
        print('\nSNR Issue: snr_true < 10.0\nSNR value: ' + str(snr_true))
        terminate.append(4)

    if 'B' in str(disp_true):
        print('\nSuspected binary of some sort\nDisposition: ' + str(disp_true))
        terminate.append(5)

    if 'K' in str(disp_true):
        print('\nKnown planet\nDisposition: ' + str(disp_true))
        terminate.append(6)

    if str(evol_true) != 'MS':
        print('\nStellar evolution issue\nEvolutionary phase: ' + str(evol_true))
        terminate.append(7)

    #if float(neighbors) != 0:
    #    print('\nNeighbors issue: neighbors != 0\nNumber of neighbors: ' + str(neighbors))
    #    terminate.append(8)

    

    if float(dep_true) != -99 and float(dep_err_true) != -99:
        dep_true_u = ufloat(float(dep_true), np.abs(float(dep_err_true)))
        rp_rs_true_unc = dep_true_u**(1/2)
        rp_rs_true = float(rp_rs_true_unc.n)
        rp_rs_err_true = float(rp_rs_true_unc.s)
    elif float(rp_true) != -99 and float(rp_err_true) != -99 and float(rstar_true) != -99 and float(rstar_err_true) != -99:
        rp_true_u = ufloat(rp_true, rp_err_true)
        rstar_true_u = ufloat(rstar_true, rstar_err_true)
        rp_rs_true_unc = rp_true_u / rstar_true_u
        rp_rs_true = float(rp_rs_true_unc.n)
        rp_rs_err_true = float(rp_rs_true_unc.s)
    else:
        print('\nRatio of planet-to-star radius cannot be computed!')
        rp_rs_true = -99
        rp_rs_err_true = -99
        terminate.append(8)
        
        
    if len(terminate) == 0:
        terminate.append(0)

    print(terminate)

    termination_code = ''
    for t in terminate:
        termination_code += str(t)

    #print(termination_code)


    ####### LDTK Analysis ########
    teff_err_u = teff_err_true

    if teff_err_u > 0:
        teff_u = teff_true
        logg_u = logg_true
        feh_u = feh_true
        
        #teff_err_u = 10.0
        logg_err_u = logg_err_true
        feh_err_u = feh_err_true
        
        print(teff_u, teff_err_u, logg_u, logg_err_u, feh_u, feh_err_u)
        
        filters_u = [tess]

        sc_u = LDPSetCreator(teff=(teff_u, teff_err_u), logg=(logg_u, logg_err_u), z=(feh_u, feh_err_u), filters=filters_u, cache='/u/scratch/m/macdouga/.ldtk/cache')

        ps_u = sc_u.create_profiles(nsamples=2000)
        ps_u.resample_linear_z(300)

        us, us_err = ps_u.coeffs_qd(do_mc=True, n_mc_samples=10000)
        us = us[0]
        us_err = us_err[0]
        print('Original u, v errors:', us_err)
        us_err = np.array([0.01, 0.01])
    else:
        us = np.array([0.4, 0.2])
        us_err = np.array([1.0, 1.0])

    print(us)
    print(us_err)
    

    print('\n' + 'Compiling data for: ' + candidate_id)


    param0 = ['period', 't0', 'duration', 'rp', 'rstar', 'rhostar', 'u1', 'u2', 'flags', 'st_source']
    units0 = ['days', 'days', 'days', 'rstar', 'rsun', 'g/cc', '', '', '', '']
    value0 = [per_true, t0_true, dur_true, rp_rs_true, rstar_true, rhostar_true, us[0], us[1], int(termination_code), source_st]
    error0 = [per_err_true, t0_err_true, dur_err_true, rp_rs_err_true, rstar_err_true, rhostar_err_true, us_err[0], us_err[1], np.nan, np.nan]
    
    inputs = pd.DataFrame(columns = ['parameter', 'units', 'value', 'error'])
    inputs['parameter'] = param0
    inputs['units'] = units0
    inputs['value'] = value0
    inputs['error'] = error0
    inputs.to_csv(lc_path + pl_name + '-inputs.csv')
    
    
    # List of termination codes to only save input data
    tks_bad = [1, 2, 3, 5, 8]

    # List of termination codes to save input data, LC, and create folders
    tks_good = [0, 4, 6, 7]

    for t in terminate:
        if t in tks_bad:
            tks_flag = 0
            break
        elif t in tks_good:
            tks_flag = 1

    #print(tks_flag)

    
    print(full_toi_id)
    print("\nPlanet Params + Errors")
    print("t0 (days), per (days), dur(days), rp (rstar)")
    print(t0_true, per_true, dur_true, rp_rs_true)
    print(t0_err_true, per_err_true, dur_err_true, rp_rs_err_true)
    
    print("\nStar Params + Errors")
    print("rstar (R_sun), rhostar (g/cc)")
    print(rstar_true, rhostar_true)
    print(rstar_err_true, rhostar_err_true)
    
    
    if tks_flag == 0: #3 in terminate or 8 in terminate:
        continue
    
    pls.append(pl)
    pl_ids.append(pl_id)
    pl_names.append(pl_name)
    candidate_ids.append(candidate_id)
    
    
    true_vals_all.append(true_vals)
    rp_rs_all.append((rp_rs_true, rp_rs_err_true))
    
    terminates.append(terminate)
    termination_codes.append(termination_code)
    tks_flags.append(tks_flag)
    

if len(pl_ids) == 0:
    sys.exit("\nNo planets in system TIC " + str(host_tic) + " meet requirements. Ending process...")
    

#################################################


# Lightcurve data file name
lc_fname = sys_name + '_lc.fits'

print(sys_name)
print(lc_fname)
print(lc_path)

try:
    f = fits.open(lc_path + 'lightcurves/' + lc_fname)
    f.close()
    print('\nAlready saved!')

    lc_file = fits.open(lc_path + 'lightcurves/' + lc_fname)

    with lc_file as hdu:
        lc = hdu[1].data

    lc_file.close()   

except FileNotFoundError:

    fail = 0

    print('\nSaving...')

    lc_download = lk.search_lightcurvefile(target="TIC " + host_tic, mission="TESS").download_all()
    if str(type(lc_download)) == '<class \'NoneType\'>':
        print('not 1')
        lc_download = lk.search_lightcurvefile(target="TOI " + host_toi, mission="TESS").download_all()
        if str(type(lc_download)) == '<class \'NoneType\'>':
            print('not 2')
            lc_download = lk.search_lightcurvefile(target="TIC" + host_tic, mission="TESS").download_all()
            if str(type(lc_download)) == '<class \'NoneType\'>':
                print('not 3')
                lc_download = lk.search_lightcurvefile(target="TOI" + host_toi, mission="TESS").download_all()
                if str(type(lc_download)) == '<class \'NoneType\'>':
                    print('not 4')
                    coord = SkyCoord(ra=float(ra_ctl), dec=float(dec_ctl), unit=(u.degree, u.degree), frame='icrs')
                    lc_download = lk.search_lightcurvefile(coord, radius=1.5).download_all()
                    if str(type(lc_download)) == '<class \'NoneType\'>':
                        print('not 5')
                        fail = 1

    if fail == 0:

        lc_full = lc_download.PDCSAP_FLUX.stitch()
        lc_full = lc_full.remove_nans().remove_outliers(sigma_lower=100,sigma_upper=3)
        lc_full.to_fits(lc_path + 'lightcurves/' + lc_fname)

        lc_file = fits.open(lc_path + 'lightcurves/' + lc_fname)

        with lc_file as hdu:
            lc = hdu[1].data

        lc_file.close()


    elif fail == 1:
        tks_flags = -1
        print('\nFile not found! Target: ' + sys_name)

####################################################################


note = ''

if tks_flags != -1:
    for terminate, tks_flag in zip(terminates, tks_flags):
        if tks_flag == 0:
            if 5 in terminate:
                note += '-EB'
            elif 3 in terminate or 8 in terminate:
                note += '-transit'
            elif 2 in terminate:
                note += '-mag'
            elif 1 in terminate:
                note += '-dec'
            else:
                note += '-none'
        elif tks_flag == 1:
            if 6 in terminate:
                note += '-KP'
            elif 4 in terminate:
                note += '-SNR'
            else:
                note += '-TKS'

    if len(full_toi_ids) > 1:
        note += '_multi'

    if str(tess_pl['name']) != 'nan':
        note += '_tks' 

    tag1 = '-ecc_mod_ecc'
    tag2 = '-ecc_mod_dur'

    dir_path1 = lc_path + sys_name + tag1 + note + '-' + date + '-e-w_free/'
    dir_path2 = lc_path + sys_name + tag2 + note + '-' + date + '-e-w_free/'

    if os.path.isdir(dir_path1) == 0:
        os.mkdir(dir_path1)
    else:
        print(dir_path1 + ' already exists!')

    if os.path.isdir(dir_path2) == 0:
        os.mkdir(dir_path2)
    else:
        print(dir_path2 + ' already exists!')

else:
    sys.exit('\nLC file not found... Ending analysis of system: ' + sys_name)

        

#if len(full_toi_ids) > 1:
#    if all_data["Full TOI ID"][pl] != full_toi_ids[0]:
#        sys.exit('\nAnalysis of system already in progress: ' + sys_name)


###############################################


dir_path = dir_path1

x0 = np.ascontiguousarray(lc["TIME"])

texp = np.min(np.diff(x0))

print('\nExposure time: ' + str(texp))


time = lc["TIME"]
flux = lc["FLUX"]
err = lc["FLUX_ERR"]
ref_time = np.nanmin(time)

m = np.any(np.isfinite(lc['FLUX'])) & (lc['QUALITY'] == 0)


###############################################


t0s_true = []
pers_true = []
durs_true = []
rp_rss_true = []

for j in range(len(true_vals_all)):
    per_true = float(true_vals_all[j][6])
    pers_true.append(per_true)
    durs_true.append(float(true_vals_all[j][8]))
    rp_rss_true.append(float(rp_rs_all[j][0]))
    

    t0_true = float(true_vals_all[j][4])
    
    t0_true -= ref_time
    
    print(t0_true, per_true)
    
    if t0_true > 0:
            while t0_true > per_true:
                print("Why is t0 not where the first transit should be?")
                t0_true -= per_true
                print(t0_true, per_true)
    elif t0_true < 0:
            while t0_true < 0:
                print("Why is t0 not where the first transit should be?")
                t0_true += per_true
                print(t0_true, per_true)
                
    t0s_true.append(t0_true)
    
t0s_true = np.array(t0s_true)
pers_true = np.array(pers_true)
durs_true = np.array(durs_true)
rp_rss_true = np.array(rp_rss_true)
    

print("t0_true = " + str(t0s_true))
print("per_true = " + str(pers_true))

time = np.ascontiguousarray(time[m] - ref_time)
flux = np.ascontiguousarray(flux[m])
err = np.ascontiguousarray(err[m])

# Get time and flux data from TESS light curve file
x = time
y = flux
yerr = err
mu = np.nanmedian(y)
y = (y / mu - 1)
yerr = (yerr / mu)
y0 = y.copy()


###############################################


m_transits = np.full(len(x), False)
for t0_true, per_true, dur_true in zip(t0s_true, pers_true, durs_true):
    x_fold = (x - t0_true + 0.5*per_true)%per_true - 0.5*per_true
    m_transits += np.abs(x_fold) < dur_true/1.5
    
m_transits = ~m_transits
    
fig = plt.figure(figsize=(14, 7))
plt.plot(x[~m_transits], y[~m_transits], 'k.', label='data')
title = "Raw LC (transits only)"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Normalized Flux", fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=20)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlim(x.min(), x.max())
fig.savefig(dir_path + sys_name + '-transits_only-init.png')
plt.close()


################################################


m = np.full(len(x), True)

smoothf = 401

for i in range(8):
    y_prime = np.interp(x, x[m_transits], y0[m_transits])
    smooth = savgol_filter(y_prime, smoothf, polyorder=1)

    resid = y0 - smooth
    sigma = np.sqrt(np.nanmean(resid**2))
    m0 = np.abs(resid) < 3*sigma
    if m.sum() == m0.sum():
        m = m0
        break
    m = m0
    
smooth = np.zeros(len(smooth))

# Only discard positive outliers
m = resid < 3.5*sigma


for j in range(len(full_toi_ids)):
    gg = 0
    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    plt.plot(x[~m_transits],y[~m_transits],'k.', label='data')
    title = "Raw LC" 
    plt.title(title + " - TOI " + str(full_toi_ids[j]) + ' - Transit ' + str(gg) + ' only', fontsize=25, y=1.03)
    plt.xlabel("Time [days]", fontsize=24)
    plt.ylabel("Normalized Flux", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.xlim([t0s_true[j] - 3*durs_true[j] + gg*pers_true[j], t0s_true[j] + 3*durs_true[j] + gg*pers_true[j]])
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-transit0-init.png')
    plt.close()
    
    
    # Plot the data
    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    title = "Raw LC (showing smoothed curve)" 
    plt.title(title + " - TOI " + str(full_toi_ids[j]) + ' - Transit ' + str(gg), fontsize=25, y=1.03)
    plt.plot(x, y, "k", label="data")
    plt.plot(x, smooth, label="smoothed")
    plt.plot(x[~m], y[~m], "xr", label="outliers")
    plt.legend(fontsize=20, loc=4)
    plt.xlabel("Time [days]", fontsize=24)
    plt.ylabel("Normalized Flux", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlim([t0s_true[j] - 5*durs_true[j] + gg*pers_true[j], t0s_true[j] + 5*durs_true[j] + gg*pers_true[j]])
    fig.savefig(dir_path + pl_names[j] + '-lc_detrend_transit0-init.png')
    plt.close()


################################################



# Plot the data
fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
        
plt.plot(x[~m], y[~m], "xr", label="outliers")
plt.plot(x, y, "k", label="data")
plt.plot(x, smooth, linewidth=2, label='smoothed')

colors = ['b', 'orange', 'g', 'r', 'm']
for j in range(len(full_toi_ids)):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x)-np.min(x))/pers_true[j])):
        if i == 0:
            lab = 'planet ' + pl_ids[j]
        else:
            lab = None
        plt.axvline(t0s_true[j] + i * pers_true[j], alpha=0.5, ls='--', color=cc, label=lab)

plt.legend(fontsize=20, loc=4)
plt.xlim(x.min(), x.max())
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Normalized Flux", fontsize=24)
title = "Raw LC (showing smoothed curve)"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-lc_detrend_smooth-init.png')
plt.close()



#############################################


# Plot the data
fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
        
plt.plot(x[~m], y[~m], "xr", label="outliers")
plt.plot(x, y, "k.", label="data")
plt.plot(x, smooth, linewidth=2, label='smoothed')

colors = ['b', 'orange', 'g', 'r', 'm']
for j in range(len(full_toi_ids)):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x)-np.min(x))/pers_true[j])):
        if i == 0:
            lab = 'planet ' + pl_ids[j]
        else:
            lab = None
        plt.axvline(t0s_true[j] + i * pers_true[j], alpha=0.5, ls='--', color=cc, label=lab)

plt.legend(fontsize=20, loc=4)
plt.xlim(x.min(), x.max())
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Normalized Flux", fontsize=24)
title = "Raw LC"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-lc_detrend-init.png')
plt.close()


#############################################


# Plot the data
fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
        
plt.plot(x[~m], y[~m], "xr", label="outliers")
plt.plot(x, y, "k.", label="data")
plt.plot(x, smooth, linewidth=2, label='smoothed')

colors = ['b', 'orange', 'g', 'r', 'm']
for j in range(len(full_toi_ids)):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x)-np.min(x))/pers_true[j])):
        if i == 0:
            lab = 'planet ' + pl_ids[j]
        else:
            lab = None
        plt.axvline(t0s_true[j] + i * pers_true[j], alpha=0.5, ls='--', color=cc, label=lab)

plt.legend(fontsize=20, loc=4)
plt.xlim(x.min(), x.max())
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Normalized Flux", fontsize=24)
title = "Raw LC"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-lc_detrend-init.png')
plt.close()



#############################################


# Make sure that the data type is consistent
x = np.ascontiguousarray(x[m])
y = np.ascontiguousarray(y[m])
yerr = np.ascontiguousarray(yerr[m])
smooth = np.ascontiguousarray(smooth[m])

x_all = list(x)
y_all = list(y)
yerr_all = list(yerr)
ysmooth_all = list(np.array(y) - np.array(smooth))
smooth_all = list(smooth)


# Plot the data
fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
        
plt.plot(x_all, ysmooth_all, "k.", label="data")

colors = ['b', 'orange', 'g', 'r', 'm']
for j in range(len(full_toi_ids)):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x_all)-np.min(x_all))/pers_true[j])):
        if i == 0:
            lab = 'planet ' + pl_ids[j]
        else:
            lab = None
        plt.axvline(t0s_true[j] + i * pers_true[j], alpha=0.5, ls='--', color=cc, label=lab)

plt.legend(fontsize=20, loc=4)
plt.xlim(np.min(x_all), np.max(x_all))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Normalized Flux", fontsize=24)
title = "Detrended LC"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-lc_all-smooth.png')
plt.close()


#########################################


lc_data = pd.DataFrame(columns=['x_all','y_all','yerr_all','smooth_all','ysmooth_all'])
lc_data['x_all'] = x_all
lc_data['y_all'] = y_all
lc_data['yerr_all'] = yerr_all
lc_data['smooth_all'] = smooth_all
lc_data['ysmooth_all'] = ysmooth_all

lc_data.to_csv(dir_path + sys_name + '-lc_data-init.csv')


#####################################################



x_all = np.array(x_all)
y_all = np.array(y_all)
smooth_all = np.array(smooth_all)
yerr_all = np.array(yerr_all)



x = np.array(x_all)
y = np.array(y_all)
smooth = np.array(smooth_all)
yerr = np.array(yerr_all)




for j in range(len(full_toi_ids)):
    x_fold = (x - t0s_true[j] + 0.5*pers_true[j])%pers_true[j] - 0.5*pers_true[j]
    m = np.abs(x_fold) < durs_true[j]*3


    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Relative Flux", fontsize=24)
    plt.xlabel("Time Since Transit [days]", fontsize=24)
    
    plt.plot(x_fold[m], y[m]-smooth[m], 'k.', label='data')
    
    title = "Detrended Phase Folded LC"
    plt.title(title + " - TOI " + str(full_toi_ids[j]), fontsize=25, y=1.03)
    plt.legend(fontsize=20, loc=4)
    plt.xlim(-2.0*durs_true[j],2.0*durs_true[j])
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-folded-smooth.png')
    plt.close()

    
#####################################################


n_transits = int((x.max() - t0_true) / per_true) + 2
keep_transits = np.zeros(len(x), dtype=bool)

keep_transits = np.full(len(x), False)
for t0_true, per_true, dur_true in zip(t0s_true, pers_true, durs_true):
    x_fold = (x - t0_true + 0.5*per_true)%per_true - 0.5*per_true
    keep_transits += np.abs(x_fold) < dur_true*10


resid = y - smooth
rms = np.sqrt(np.median(resid**2))
mask = list(np.abs(resid) < 100 * rms)

for ii in range(len(mask)):
    if mask[ii] == False and keep_transits[ii] == True:
        mask[ii] = True

keep_transits = np.array(keep_transits)
mask = np.array(mask)



x = x[mask]
y = y[mask]
yerr = yerr[mask]
smooth = smooth[mask]


#####################################################


for j in range(len(full_toi_ids)):
    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    x_fold = (x - t0s_true[j] + 0.5 * pers_true[j]) % pers_true[j] - 0.5 * pers_true[j]
    plt.scatter(x_fold, y - smooth, c=x, s=10, alpha=0.8)
    plt.xlabel("Time Since Transit [days]", fontsize=24)
    plt.ylabel("Relative Flux", fontsize=24)
    cb = plt.colorbar()
    cb.set_label(label="Time [days]", fontsize=20)
    cb.ax.tick_params(labelsize=16)
    plt.xlim(-2.0 * durs_true[j], 2.0 * durs_true[j])
    title = "Detrended Phase Folded LC (relative coloring)"
    plt.title(title + " - TOI " + str(full_toi_ids[j]), fontsize=25, y=1.03)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-folded_smooth-init.png')
    plt.close()
    

#####################################################


def plot_light_curve(x, y, yerr, soln, mask=None, g=-1, spread=0, ylim=[], idx=''):

    figs = []
    
    values = range(len(t0s_true))
    
    for j in values:
        
        if spread == 0:
            spread = durs_true[j] * 2
        
        if mask is None:
            mask = np.ones(len(x), dtype=bool)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        plt.rc('text', usetex=False)
        ax = axes[0]
        ax.plot(x[mask], y[mask], "k.", label="Data")
        gp_mod = soln["gp_pred"] + soln["mean"]
        ax.plot(x[mask], gp_mod, color="C2", label="GP Model")
        ax.legend(fontsize=10)
        ax.set_ylabel("Relative Flux")

        if ylim != [] and g > -1:
            ax.set_ylim(np.array(ylim))

        ax = axes[1]
        ax.plot(x[mask], y[mask] - gp_mod, "k.", label="De-trended Data")
                   
        letters = ["b", "c", "d", "e", "f", "g"]
        planets = ''
        for jj in values:
            planets += letters[jj]
            
        for i, l in enumerate(planets):
            mod = soln["light_curves"][:, i]
            ax.plot(
                x[mask], mod, lw=1, label="planet {0}".format(l)
            )
            
        ax.legend(fontsize=10, loc=3)
        ax.set_ylabel("De-trended Relative Flux")

        if ylim != [] and g > -1:
            ax.set_ylim(np.array(ylim))

        ax = axes[2]
        mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
        ax.plot(x[mask], y[mask] - mod, "k.")
        ax.axhline(0, color="#aaaaaa", lw=1)
        ax.set_ylabel("Residuals of Relative Flux")
        ax.set_xlim(x[mask].min(), x[mask].max())
        ax.set_xlabel("Time [days]")

        if g > -1:
            ax.set_xlim(np.array([soln["t0"][j]+(soln["period"][j]*g)-spread, soln["t0"][j]+(soln["period"][j]*g)+spread]))

        if ylim != [] and g > -1:
            ax.set_ylim(np.array(ylim))

        if g == -1:
            name_label = sys_name
        else:
            name_label = pl_names[j]

        fig.savefig(dir_path + name_label + '-gp_mod' + str(idx) + '-mod.png')
        
        if g == -1 or len(values) == 1:
            return fig

        figs.append(fig)

        plt.close()

    return figs;


if rhostar_true > 0:
    rho_test = rhostar_true
else:
    rho_test = rhosun
    
shape = len(pers_true)

bs = [0.8]*shape


#eccs = [0.263]
#eccs_err = [0.072]

#omegas = [1.95]
#omegas_err = [0.265]

us_err = np.array([0.05, 0.05])

rho_test = 2.4530380410313612 #2.154338715772575 #2.1554032281420600
rho_test_err = 0.3427792244909791 #0.31181778821821887 #0.13716684811003000

### 2.154338715772575+/-0.31181778821821887

# Working code - no ecc/w

def build_model(x, y, yerr, start=None, mask=None, test=1, gp_flag=1): 
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    # This is the current test - modified to look like my old starry model
    with pm.Model() as model0:

        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0)

        # The time of a reference transit for each planet
        BoundedNormal_t0 = pm.Bound(pm.Normal, lower=t0s_true-0.5*pers_true, upper=t0s_true+0.5*pers_true)
        t0 = BoundedNormal_t0("t0", mu=t0s_true, sd=0.01, shape=shape)

        # The log period; also tracking the period itself
        BoundedNormal_per = pm.Bound(pm.Normal, lower=(pers_true*0.9), upper=(pers_true*1.1))
        period = BoundedNormal_per("period", mu=(pers_true), sd=0.1, shape=shape)

        
        # The stellar limb darkening parameters, using inputs from LDTK if stellar data is available
        BoundedNormal_u_star = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)
        u_star = BoundedNormal_u_star("u_star", mu=us, sd=us_err, shape=2)

        # The log stellar density; also tracking the stellar density itself
        #logrho_star = pm.Uniform("logrho_star", lower=-8.0, upper=5.0, shape=shape)
        #rho_star = pm.Deterministic("rho_star", tt.exp(logrho_star))
        rho_star = pm.Normal("rho_star", mu=rho_test, sd=rho_test_err, shape=shape)

        
        # The log planet radius ratio; also tracking the radius ratio itself
        ###logr = pm.Uniform("logr", lower=-10.0, upper=0.0, testval=np.log(rp_rss_true), shape=shape)
        ###r = pm.Deterministic("r", tt.exp(logr))


        BoundedNormal_logr = pm.Bound(pm.Normal, lower=-10.0, upper=0.0)
        logr = BoundedNormal_logr("logr", mu=np.log(rp_rss_true), sd=10.0, shape=shape)
        r = pm.Deterministic("r", tt.exp(logr))

        
        b_bound = 2.0 #pm.Deterministic("b_bound", 1+r)

        # The impact parameter as a free variable, not related to stellar radius ratio directly here
        ## ## ##b = pm.Uniform("b", lower=0.000, upper=b_bound, shape=shape, testval=bs)
        b = xo.distributions.ImpactParameter("b", ror=r, shape=shape, testval=bs)
        

        ecs = xo.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
        ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))



        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, rho_star=rho_star, ecc=ecc, omega=omega)


        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u_star).get_light_curve(
            orbit=orbit, r=r, t=x[mask], texp=texp)
        light_curve = pm.math.sum(light_curves, axis=-1) + mean

        # Here we track the value of the model light curve for plotting purposes
        pm.Deterministic("light_curves", light_curves)

        if gp_flag == 1:

            # Transit jitter & GP parameters
            logs2 = pm.Normal("logs2", mu=np.log(np.var(y[mask])), sd=0.1)
            logw0 = pm.Normal("logw0", mu=0.0, sd=0.1)
            logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y[mask])), sd=0.1)

            # GP model for the light curve
            kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2))
            gp = xo.gp.GP(kernel, x[mask], tt.exp(logs2) + tt.zeros(mask.sum()), J=2)
            pm.Potential("transit_obs", gp.log_likelihood(y[mask] - light_curve))
            pm.Deterministic("gp_pred", gp.predict())
            

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=yerr[mask], observed=y[mask])


        if start is None:
            start = model0.test_point

        if gp_flag == 0:
            test = 0

        if test == 1:
            map_soln0 = xo.optimize(start=start, vars=[logs2, logSw4, logw0])
            map_soln0 = xo.optimize(start=map_soln0, vars=[r, b], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[period, t0], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[u_star], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[r, b], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[ecs], verbose=False) 
            map_soln0 = xo.optimize(start=map_soln0, vars=[rho_star], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[mean], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[logs2, logSw4, logw0], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0)

        elif test == 0:
            map_soln0 = xo.optimize(start=start)
            map_soln0 = xo.optimize(start=map_soln0, vars=[r, b], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[period, t0], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[u_star], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[r, b], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[ecs], verbose=False) 
            map_soln0 = xo.optimize(start=map_soln0, vars=[rho_star], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0, vars=[mean], verbose=False)
            map_soln0 = xo.optimize(start=map_soln0)

    return model0, map_soln0

# Optimize all parameters
model0, map_soln0 = build_model(x, y - smooth, yerr, start=None, mask=None, gp_flag=1, test=1)

plot_light_curve(x, y - smooth, yerr, map_soln0, mask=None, g=-1);


####################################################


mod = (
    map_soln0["gp_pred"]
    + map_soln0["mean"]
    + np.sum(map_soln0["light_curves"], axis=-1)
)
resid = y - smooth - mod
rms = np.sqrt(np.median(resid ** 2))
mask = np.abs(resid) < 5 * rms

fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
plt.plot(x, resid, "k.", label="data")
plt.plot(x[~mask], resid[~mask], "xr", label="outliers")

for j in range(len(full_toi_ids)):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x)-np.min(x))/pers_true[j])):
        if i == 0:
            lab = 'planet ' + pl_ids[j]
        else:
            lab = None
        plt.axvline(t0s_true[j] + i * pers_true[j], alpha=0.5, ls='--', color=cc, label=lab)

plt.axhline(0, color="#aaaaaa", lw=1)
plt.ylabel("Residuals of Relative Flux", fontsize=24)
plt.xlabel("Time [days]", fontsize=24)
plt.legend(fontsize=20, loc=4)
plt.xlim(x.min(), x.max())
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
title = "Residuals of LC Fit"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-gp_residuals-mod.png')
plt.close()


#################################################


n_transits = int((x.max() - t0_true) / per_true) + 2
keep_transits = np.zeros(len(x), dtype=bool)

keep_transits = np.full(len(x), False)
for t0_true, per_true, dur_true in zip(t0s_true, pers_true, durs_true):
    x_fold = (x - t0_true + 0.5*per_true)%per_true - 0.5*per_true
    keep_transits += np.abs(x_fold) < dur_true*2

for j in range(len(mask)):
    if mask[j] == False:
        keep_transits[j] = False



# Subtract latest GP model from data and set as new data
gp_mod_final = map_soln0["gp_pred"] + map_soln0["mean"]

time = x[keep_transits]
flux = y[keep_transits] - smooth[keep_transits] - gp_mod_final[keep_transits]
err = yerr[keep_transits]


##################################################



fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
plt.plot(x[mask], y[mask] - smooth[mask] - gp_mod_final[mask], 'k.', label='data')

j = 0
for j in range(len(full_toi_ids)):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x[mask])-np.min(x[mask]))/pers_true[j])):
        if i == 0:
            lab = 'planet ' + pl_ids[j]
        else:
            lab = None
        plt.axvline(t0s_true[j] + i * pers_true[j], alpha=0.5, ls='--', color=cc, label=lab)
    j += 1
    
plt.legend(fontsize=20, loc=1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Relative Flux", fontsize=24)
title = "Fully Detrended LC"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.xlim(x[mask].min(), x[mask].max());
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-lc_all-mod.png')
plt.close()


#######################################################


lc_data = pd.DataFrame(columns=['x','y','yerr'])
lc_data['x'] = x
lc_data['y'] = y - smooth - gp_mod_final
lc_data['yerr'] = yerr

lc_data.to_csv(dir_path + sys_name + '-lc_data-detrended_full.csv')


##################################################



plot_light_curve(x, y-smooth, yerr, map_soln0, mask=None, g=0, idx=0);


print("period = ", map_soln0["period"])
print("t0 = ", map_soln0["t0"])

fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
plt.plot(time, flux, 'k.', label='data')

j = 0
for t0_true, per_true, dur_true in zip(t0s_true, pers_true, durs_true):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x)-np.min(x))/per_true)):
        if i == 0:
            lab = 'planet ' + pl_ids[j]
        else:
            lab = None
        plt.axvline(t0_true + i * per_true, alpha=0.5, ls='--', color=cc, label=lab)
    j += 1
    
plt.legend(fontsize=20, loc=1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Relative Flux", fontsize=24)
title = "Fully Detrended LC (transits only)"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.xlim(x.min(), x.max());
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-lc_transits-mod.png')
plt.close()


#######################################################


print(t0s_true, pers_true, durs_true, rp_rss_true, rhostar_true)

x=time
y=flux
yerr=err
start=map_soln0
mask=None

lc_data = pd.DataFrame(columns=['x','y','yerr'])
lc_data['x'] = x
lc_data['y'] = y
lc_data['yerr'] = yerr

lc_data.to_csv(dir_path + sys_name + '-lc_data-mod.csv')


#######################################################


for j in range(len(map_soln0["period"])):
    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    x_fold = (x - map_soln0["t0"][j] + 0.5 * map_soln0["period"][j]) % map_soln0["period"][j] - 0.5 * map_soln0["period"][j]

    plt.scatter(x_fold, y, c=x, s=10, alpha=0.8)
    plt.xlabel("Time Since Transit [days]", fontsize=24)
    plt.ylabel("Relative Flux", fontsize=24)
    cb = plt.colorbar()
    cb.set_label(label="Time [days]", fontsize=20)
    cb.ax.tick_params(labelsize=16)
    title = "Fully Detrended Phase Folded LC (relative coloring)"
    plt.title(title + " - TOI " + str(full_toi_ids[j]), fontsize=25, y=1.03)
    plt.xlim(-2.0 * durs_true[j], 2.0 * durs_true[j])
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-folded_smooth-mod.png')
    plt.close()
    


#######################################################


if mask is None:
    mask = np.ones(len(x), dtype=bool)


with pm.Model() as model:

    # The baseline flux
    mean = pm.Normal("mean", mu=0.0, sd=1.0)

    # The time of a reference transit for each planet
    BoundedNormal_t0 = pm.Bound(pm.Normal, lower=t0s_true-0.5*pers_true, upper=t0s_true+0.5*pers_true)
    t0 = BoundedNormal_t0("t0", mu=t0s_true, sd=0.1, shape=shape)

    # The log period; also tracking the period itself
    BoundedNormal_per = pm.Bound(pm.Normal, lower=(pers_true*0.9), upper=(pers_true*1.1))
    period = BoundedNormal_per("period", mu=(pers_true), sd=0.1, shape=shape)

    
    # The stellar limb darkening parameters, using inputs from LDTK if stellar data is available
    BoundedNormal_u_star = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)
    u = BoundedNormal_u_star("u", mu=us, sd=us_err, shape=2)
    
    # The log stellar density; also tracking the stellar density itself
    #logrho = pm.Uniform("logrho", lower=-8.0, upper=5.0, shape=shape)
    #rho = pm.Deterministic("rho", tt.exp(logrho))
    rho = pm.Normal("rho", mu=rho_test, sd=rho_test_err, shape=shape)

    
    # The log planet radius ratio; also tracking the radius ratio itself
    ##logr = pm.Uniform("logr", lower=-10.0, upper=0.0, testval=np.log(rp_rss_true), shape=shape)
    ##r = pm.Deterministic("r", tt.exp(logr))

    BoundedNormal_logr = pm.Bound(pm.Normal, lower=-10.0, upper=0.0)
    logr = BoundedNormal_logr("logr", mu=np.log(rp_rss_true), sd=10.0, shape=shape)
    r = pm.Deterministic("r", tt.exp(logr))
    
    b_bound = 2.0 #pm.Deterministic("b_bound", 1+r)

    # The impact parameter as a free variable, not related to stellar radius ratio directly here
    ## ## ##b = pm.Uniform("b", lower=0.000, upper=b_bound, shape=shape, testval=bs)
    b = xo.distributions.ImpactParameter("b", ror=r, shape=shape, testval=bs)


    ecs = xo.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
    ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
    omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))


    # Set up a Keplerian orbit for the planets
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, rho_star=rho, ecc=ecc, omega=omega) 

    # Compute the model light curve using starry 
    light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
        orbit=orbit, r=r, t=x[mask], texp=texp)
    light_curve = pm.math.sum(light_curves, axis=-1) + mean

    ###pm.Deterministic("light_curves", light_curves) ###### CHANGED!!!!!

    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=light_curve, sd=yerr[mask], observed=y[mask])

    map_soln = xo.optimize(start=start)
    map_soln = xo.optimize(start=map_soln, vars=[r, b], verbose=False)
    map_soln = xo.optimize(start=map_soln, vars=[period, t0], verbose=False)
    map_soln = xo.optimize(start=map_soln, vars=[u], verbose=False)
    map_soln = xo.optimize(start=map_soln, vars=[r, b], verbose=False)
    map_soln = xo.optimize(start=map_soln, vars=[ecs], verbose=False) 
    map_soln = xo.optimize(start=map_soln, vars=[rho], verbose=False) 
    map_soln = xo.optimize(start=map_soln, vars=[mean], verbose=False)
    map_soln = xo.optimize(start=map_soln, verbose=False)

#####################################################

for j in range(len(pers_true)):
    print('Comparing input per and t0 to initial model...\n')
    print(map_soln["period"][j], pers_true[j])
    print(map_soln["t0"][j], t0s_true[j])
    if np.abs(map_soln["period"][j] - pers_true[j]) > 0.15 or np.abs(map_soln["t0"][j] - t0s_true[j]) > 0.1:
        print('MODELED PERIOD OR T0 DOES NOT MATCH GIVEN VALUES\nCheck output plots for: ' + candidate_ids[j])
        per_true = map_soln["period"][j]
        t0_true = map_soln["t0"][j]
    else:
        print(candidate_ids[j] + ' passes check!\n')

    # Plot the folded transit
    x_fold = (x - map_soln["t0"][j] + 0.5*map_soln["period"][j])%map_soln["period"][j] - 0.5*map_soln["period"][j]
    m = np.abs(x_fold) < 2.0*durs_true[j]

    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Time [days]", fontsize=24)
    plt.ylabel("Relative Flux", fontsize=24)
    plt.plot(x_fold[m], y[m], 'k.', label='data')
    plt.legend(fontsize=20, loc=4)
    plt.xlim(-2.0*durs_true[j],2.0*durs_true[j])
    title = "Fully Detrended Phase Folded LC"
    plt.title(title + " - TOI " + str(full_toi_ids[j]), fontsize=25, y=1.03)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-folded-mod.png')
    plt.close()


print('Impact parameter: ', map_soln["b"])


#####################################################


if len(pers_true) == 1:
    chainz = 4
    corez = 2
    tunez = 9000
    drawz = 6000
else:
    chainz = 2
    corez = 2
    tunez = 10000
    drawz = 6000
    

np.random.seed(42)
with model:
    tr = pm.sample(chains=chainz,
                   cores=corez,
                   start=map_soln,
                   step=xo.get_dense_nuts_step(target_accept=0.95,start=map_soln),
                   tune=tunez,
                   draws=drawz,
                   progressbar=True)


######################################################


varnames=["t0","period","r","b","ecc","omega","rho","mean","u"]
colnames=[]

for vv in varnames[:7]:
    for j in range(len(pers_true)):
        colnames.append(vv + '__' + str(j))
        
colnames += ["mean","u__0","u__1"]

df = pm.trace_to_dataframe(tr, varnames=varnames)
df.columns = colnames
df.to_csv(dir_path + sys_name + "-trace.csv")

df = pd.read_csv(dir_path + sys_name + "-trace.csv")

            
#####################################################


fig1 = pm.traceplot(tr, var_names=varnames, compact=False)
plt.rc('text', usetex=False)
figall = fig1[0][0].figure
figall.savefig(dir_path + sys_name + '-trace_plots-all.png')
plt.close()

if len(pers_true) > 1:
    single_vars = ["r", "b", "ecc", "omega", "rho"]
    for var in single_vars:
        fig0 = pm.traceplot(tr, var_names=[var], compact=False)
        plt.rc('text', usetex=False)
        figall = fig0[0][0].figure
        figall.savefig(dir_path + sys_name + '-trace_plots-' + var + '.png')
        plt.close()
        

#####################################################


c = ChainConsumer()
c.configure(usetex=False)
c.add_chain(np.array(df[colnames]),parameters=colnames)
c.configure(usetex=False, label_font_size=10, tick_font_size=8)
plt.rc('text', usetex=False)
plt.gcf().subplots_adjust(bottom=0.01, left=0.01)

fig2 = c.plotter.plot()
fig2.tight_layout()
fig2.savefig(dir_path + sys_name + '-corner_full.png')
plt.close()


#####################################################


for j in range(len(pers_true)):
    c = ChainConsumer()
    c.configure(usetex=False)
    colsmain = ["r__" + str(j),"b__" + str(j),"ecc__" + str(j),"omega__" + str(j),"rho__" + str(j)]
    colsmain_labels = ["r", "b", "ecc", "omega", "rhocirc"]
    c.add_chain(np.array(df[colsmain]),parameters=colsmain_labels)
    c.configure(usetex=False, label_font_size=10, tick_font_size=8)
    plt.rc('text', usetex=False)
    plt.gcf().subplots_adjust(bottom=0.01, left=0.01)

    fig3 = c.plotter.plot()
    fig3.tight_layout()
    fig3.savefig(dir_path + pl_names[j] + '-corner_main.png')
    plt.close()

    
###################################################


cases = ['iso_teff',
         'iso_color',
         'tic'
        ]

p = []
t0 = []

for j in range(len(pers_true)):
    p.append(np.median(tr["period"][:, j]))
    t0.append(np.median(tr["t0"][:, j]))
    

##################################################
trace = tr
trace0 = tr

data_df = pd.DataFrame()
data_df['b'] = list(df['b__0'])
data_df['r'] = list(df['r__0'])

r_low = list(data_df.sort_values(by='b')['r'])[0]


for j in range(len(pers_true)):
    # Plot the folded data
    
    mask = np.full(len(x), False)
    
    for jj in range(len(pers_true)):
        if j != jj:
            x_fold_jj = (x - t0[jj] + 0.5 * p[jj]) % p[jj] - 0.5 * p[jj]
            mask += np.abs(x_fold_jj) < durs_true[jj] * 0.75
            
    mask = ~mask

    masktemp = np.full(len(x), False)

    for jj in range(len(pers_true)):
        if j == jj:
            x_fold_j = (x - t0[jj] + 0.5 * p[jj]) % p[jj] - 0.5 * p[jj]
            masktemp += np.abs(x_fold_j) < durs_true[jj] * 0.75

    for mm in range(len(mask)):
        if mask[mm] == False and masktemp[mm] == True:
            mask[mm] = True
    
    x_fold = (x[mask] - t0[j] + 0.5 * p[j]) % p[j] - 0.5 * p[j]
    

    # Overplot the phase binned light curve
    if 4.0*durs_true[j] < 0.5*p[j]:
        bound = 4.0*durs_true[j]
    else:
        bound = 0.5*p[j]
    bins = np.linspace(-1*bound, bound, 85)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=y[mask] - np.median(tr["mean"]))
    denom[num == 0] = 1.0

    samples = np.empty((50, len(x[mask])))
    with model:
        orbit = xo.orbits.KeplerianOrbit(period=p, t0=t0, b=b, ecc=ecc, omega=omega, rho_star=rho)
        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=r, t=x[mask], texp=texp)
        light_curve = pm.math.sum(light_curves, axis=-1) + mean
        y_grid = light_curve
        for i, sample in enumerate(xo.get_samples_from_trace(tr, size=50)):
            samples[i] = xo.eval_in_model(y_grid, sample)

    isort = np.argsort(x_fold)


    rho_cases = [] 
    cases_tags = cases


    for c in ['iso_teff', 'iso_color']:
        rho_cases.append('rho-' + c)
        rho_cases.append('rho_err1-' + c)
        rho_cases.append('rho_err2-' + c)

    rho_cases += ["tic_ms", "tic_ms_err", "tic_rs", "tic_rs_err"]

    rho_data = np.array(all_data.loc[all_data["full_toi_id"] == all_data["full_toi_id"][pl]][rho_cases])[0]


    ecc_data_array = []
    
    ew_data = 0

    for i in range(len(cases)):

        case = cases[i]
        case_tag = cases_tags[i]

        if case != 'tic':
            rhostar_true = rho_data[3*i]
            rhostar_true_err1 = rho_data[3*i+1]
            rhostar_true_err2 = -1*rho_data[3*i+2]
            rhostar_true_err = np.mean([rhostar_true_err1, rhostar_true_err2])


            rhostar_case_u = ufloat(rhostar_true, rhostar_true_err)          # rho_sun (CHECK UNITS OF INPUT SOURCE)

            rhostar_case_unc = rhostar_case_u * rhosun_u
            rhostar_case = float(rhostar_case_unc.n)
            rhostar_err_case = float(rhostar_case_unc.s)             # g / cm^3

            rhostar_true = rhostar_case
            rhostar_err_true = rhostar_err_case
        elif case == 'tic':
            mass_u = ufloat(rho_data[-4], rho_data[-3])
            rad_u = ufloat(rho_data[-2], rho_data[-1])

            rhostar_case_unc = mass_u * msun_u / (4./3. * np.pi * (rad_u * rsun_u)**3)
            rhostar_case = float(rhostar_case_unc.n)
            rhostar_err_case = float(rhostar_case_unc.s)             # g / cm^3

            rhostar_true = rhostar_case
            rhostar_err_true = rhostar_err_case

        if rhostar_case > 0 and rhostar_err_case > 0:

            if rhostar_err_case < 0.05 * rhostar_case:
                print(case, ' error bar unreasonably small: ', rhostar_case, rhostar_err_case)
                rhostar_err_case = 0.05 * rhostar_case

            print(rhostar_case)
            print(rhostar_err_case)


            tr_temp = pd.read_csv(dir_path + sys_name + "-trace.csv")
            rho_tr = np.array(list(tr_temp['rho__' + str(j)]))

            rho_obs = (rhostar_case, rhostar_err_case)
            
            rp_f0 = np.median(tr_temp["r__" + str(j)])
            rp_f0_err1 = np.percentile(q=[15.865], a=tr_temp["r__" + str(j)])[0]
            rp_f0_err2 = np.percentile(q=[84.135], a=tr_temp["r__" + str(j)])[0]
            
            rp_rs_fin = ufloat(rp_f0, np.nanmean([rp_f0_err2-rp_f0, -1*(rp_f0_err1-rp_f0)]))
            if case != 'tic':
                rad_case = ufloat(all_data["rad-"+case][pl], np.nanmean([all_data["rad_err1-"+case][pl], -1*all_data["rad_err2-"+case][pl]]))
                rad_st_case = float(rad_case.n)
                rad_st_err_case = float(rad_case.s)
                
                mass_case = ufloat(all_data["mass-"+case][pl], np.nanmean([all_data["mass_err1-"+case][pl], -1*all_data["mass_err2-"+case][pl]]))    
                mass_st_case = float(mass_case.n)
                mass_st_err_case = float(mass_case.s)
                
            elif case == 'tic':
                rad_case = rad_u
                rad_st_case = float(rad_u.n)
                rad_st_err_case = float(rad_u.s)
                
                mass_st_case = float(mass_u.n)
                mass_st_err_case = float(mass_u.s)
                
                
            if rad_st_case > 0 and rad_st_err_case > 0:
                rad_pl_fin = rad_case * rp_rs_fin * rsun_u/rearth_u
                rad_pl_case = float(rad_pl_fin.n)
                rad_pl_err_case = float(rad_pl_fin.s)
            else:
                rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case, rad_pl_case, rad_pl_err_case = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
             
            
            lower_rho, upper_rho = 0.00001, 250.0
            mu_rho, sigma_rho = rho_obs
            XX = sstat.truncnorm(
                (lower_rho - mu_rho) / sigma_rho, (upper_rho - mu_rho) / sigma_rho, loc=mu_rho, scale=sigma_rho)

            rho_obs_array = XX.rvs(len(rho_tr))

            rho_ratio = rho_tr/rho_obs_array
            rho_df = pd.DataFrame(np.array([list(rho_ratio), list(rho_tr), list(rho_obs_array)]).T, columns=["rho_ratio", "rho_circ", "rho_obs"])

            tot_len = len(host_toi)
            if tot_len == 4:
                prefix = 'T00'
            elif tot_len == 3:
                prefix = 'T000'
            
            rho_df.to_csv(dir_path + prefix + host_toi + pl_ids[j] + '-rho_ratio_posterior-' + case + '.csv')

            
            
            fupsample = 1000 # duplicate chains by this amount
            rho_circ = np.hstack([rho_tr.flatten()]*fupsample)
            ecc00 = np.random.uniform(0, 1, len(rho_circ))
            omega00 = np.random.uniform(-0.5*np.pi, 1.5*np.pi, len(rho_circ))
            g = (1 + ecc00 * np.sin(omega00)) / np.sqrt(1 - ecc00 ** 2)
            rho00 = rho_circ / g ** 3



            #####################################################


            # Build up interpolated KDE
            samples00 = rho_tr.flatten()
            smin, smax = np.min(samples00), np.max(samples00)
            width = smax - smin
            xx = np.linspace(smin, smax, 1000)
            yy = stats.gaussian_kde(samples00)(xx)
            xi = np.linspace(xx[0],xx[-1],1000)
            rhocircpost = lambda xi: np.interp(xi,xx,yy,left=0,right=0)

            # Plot it
            fig = plt.figure(figsize=[10,7])
            plt.hist(samples00,bins=100,density=True)
            plt.plot(xi,rhocircpost(xi), label=r"modeled $\rho_{circ}$ posterior", color="g")


            label0 = r"derived $\rho_{obs}$ (isoclassify)"

            xspace0 = []
            for n in range(10000):
                xspace0.append(np.random.normal(loc=rho_obs[0], scale=rho_obs[1]))
                
            if rho_obs[0] - rho_obs[1] * 5 < 0:
                left = 0
            else:
                left = rho_obs[0] - rho_obs[1] * 5
            x0 = np.linspace(left, rho_obs[0] + rho_obs[1] * 5, 1000)
            y0 = stats.gaussian_kde(xspace0)(x0)
            xi0 = np.linspace(x0[0], x0[-1], 1000)
            rhocircpost0 = lambda xi: np.interp(xi0,x0,y0)

            plt.plot(xi0,rhocircpost0(xi0), color='r', label=label0)
            plt.fill_between(xi0, rhocircpost0(xi0), y2=0, interpolate=True, color="r", alpha=0.1, hatch="/", zorder=1000)

            title = 'TOI ' + str(full_toi_ids[j]) + r': $\rho_{circ}$ vs $\rho_{obs}$ ' + ' - ' + case
            plt.title(title, fontsize=25, y=1.03)
            plt.xlabel(r'$\rho$ [g/cc]', fontsize=24)
            plt.ylabel('Count density', fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            #plt.vlines(x=rhostar_case, ymin=0, ymax=np.max(rhocircpost(xi)), color='k', ls='--', lw=3, label=label)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-rho_circ_hist-' + case + '.png')
            plt.close()

            # Generate a grid of e, omega, g
            emin, emax, esamp = 0,0.99,99
            omegamin, omegamax, omegasamp = -0.5*np.pi,1.5*np.pi,100
            ecc00 = np.linspace(emin,emax,esamp)
            omega00 = np.linspace(omegamin,omegamax,omegasamp)
            ecc2d,omega2d = np.meshgrid(ecc00,omega00,indexing='ij')
            g2d = (1+ecc2d*np.sin(omega2d))/np.sqrt(1-ecc2d**2) 

            # Compute the posterior probability as a function of g
            def func(rho00, g):
                rhocircobs = np.exp(-0.5 * ((rho00 - rho_obs[0]*g**3)/(rho_obs[1]*g**3))**2)
                return rhocircpost(rho00) * rhocircobs
            gp = np.logspace(np.log10(np.min(g)),np.log10(np.max(g)),1000)
            probgp = [scipy.integrate.quad(func,smin,smax,args=(_g),full_output=1)[0] for _g in gp]
            probg = lambda g: np.interp(g,gp,probgp)
            prob = probg(g2d)
            
            
            rho_post = rho_ratio

            rho_d = np.linspace(np.min(rho_post), np.max(rho_post), 1000)

            bw = 0.9 * np.min([np.std(rho_post), scipy.stats.iqr(rho_post)/1.34]) * (len(rho_d)**(-1/5))

            kde = KernelDensity(bandwidth=bw, kernel='gaussian')
            kde.fit(rho_post[:,None])

            logprob = kde.score_samples(rho_d[:, None])

            fig = plt.figure(figsize=[10,7])
            plt.plot(rho_d, np.exp(logprob), 'k', lw=3, label='KDE: ' + r'$\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc')
            plt.hist(rho_post,50,alpha=0.5, density=True, label='from photoecc posterior')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18) 
            plt.xlabel(r'$\rho_{circ}$/$\rho_{obs}$', fontsize=24)
            plt.ylabel('Count Density', fontsize=24)
            title = 'TOI ' + str(full_toi_ids[j]) + r': $\rho_{circ}$/$\rho_{obs}$ ' + ' - ' + case
            plt.title(title, fontsize=25, y=1.03)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-rho_ratio_hist-' + case + '.png')
            plt.close()

            interp_df = pd.DataFrame(columns=["rho_ratio", "weight", "logprob"])
            interp_df["rho_ratio"] = rho_d
            interp_df["weight"] = np.exp(logprob)
            interp_df["logprob"] = logprob

            interp_df.to_csv(dir_path + prefix + host_toi + pl_ids[j] + '-rho_ratio_interp-' + case + '.csv')

            posts = interp_df
            rho_ratios_in = np.array(list(posts["rho_ratio"]))
            weights_in = np.array(list(posts["weight"]))
            logprobs_in = np.array(list(posts["logprob"]))

            combos = []
            weights = []

            for ee in ecc00:
                for ww in omega00:
                    combos.append([ee,ww])
                    rho_ratio = e_w_function([np.sqrt(ee)*np.cos(ww), np.sqrt(ee)*np.sin(ww)])
                    weight = np.interp(rho_ratio, rho_ratios_in, weights_in)
                    weights.append(weight)

            esinws = []
            ecosws = []    

            finite_weights = []
            for n in weights:
                if n > -np.inf:
                    finite_weights.append(n)

            finite_weights = np.array(finite_weights)*50.0/np.max(finite_weights)
            for combo, weight in zip(combos, finite_weights):
                weight = int(weight)
                esinw = combo[0]*np.sin(combo[1])
                ecosw = combo[0]*np.cos(combo[1])
                esinws += [esinw]*weight
                ecosws += [ecosw]*weight


            data_kde = np.array([ecosws, esinws])
            x_kde, y_kde = data_kde
            xnbins_kde = 100
            ynbins_kde = 100


            # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
            kd_xy = kdest.gaussian_kde(data_kde)
            xi_kde_lin = np.linspace(x_kde.min(),x_kde.max(),xnbins_kde)
            yi_kde_lin = np.linspace(y_kde.min(),y_kde.max(),ynbins_kde)

            xi_kde, yi_kde = np.meshgrid(xi_kde_lin, yi_kde_lin,indexing='ij')

            zi_kde = kd_xy(np.vstack([xi_kde.flatten(), yi_kde.flatten()]))

            c = ChainConsumer()
            #c.configure(usetex=False) 
            plt.rc('text', usetex=False)

            c.add_chain([xi_kde_lin,yi_kde_lin],grid=True,weights=zi_kde.reshape((len(xi_kde_lin),len(yi_kde_lin))),kde=True,smooth=True,parameters=['$e cos(\omega)$','$e sin(\omega)$'])
            c.configure(usetex=False, label_font_size=20, tick_font_size=16)
            fig = c.plotter.plot(figsize=[8,8])

            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + " (" + case + ")"
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(105, -1),
                rotation=270,
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=18,
                annotation_clip=False
            )

            #fig.set_size_inches(6.2 + fig.get_size_inches())  
            plt.gcf().subplots_adjust(bottom=0.2, right=0.85)
            plt.tight_layout()
            fig.savefig(dir_path + pl_names[j] + '-ecosw_esinw-' + case + '.png')
            plt.close()





            if np.sum(prob) > 0:
                try:
                    c = ChainConsumer()
                    #c.configure(usetex=False) #, label_font_size=22, tick_font_size=16)
                    plt.rc('text', usetex=False)

                    c.add_chain([ecc00,omega00*180/np.pi],grid=True,weights=prob,kde=True,smooth=True,parameters=['$e$','$\omega$'])
                    c.configure(usetex=False, label_font_size=26, tick_font_size=18)
                    fig = c.plotter.plot(figsize=[8,8])

                    # Annotate the plot with the planet's period
                    txt = "TOI " + str(full_toi_ids[j]) + " (" + case + ")"
                    plt.annotate(
                        txt,
                        (0, 0),
                        xycoords="axes fraction",
                        xytext=(110, 0),
                        rotation=270,
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=18,
                        annotation_clip=False
                    )

                    #fig.set_size_inches(3.0 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
                    plt.gcf().subplots_adjust(bottom=0.15, right=0.8)
                    plt.tight_layout()
                    fig.savefig(dir_path + pl_names[j] + '-ecc-' + case + '.png')
                    plt.close()

                    fig.savefig(dir_path + pl_names[j] + '-ecc-' + case + '.png')
                    plt.close()

                    if ew_data == 0:
                        post = pd.DataFrame(columns = ['ecc'])
                        post['ecc'] = ecc00
                        post.to_csv(dir_path + pl_names[j] + '-posteriors_e.csv')

                        post = pd.DataFrame(columns = ['omega'])
                        post['omega'] = omega00*180/np.pi
                        post.to_csv(dir_path + pl_names[j] + '-posteriors_w.csv')
                        ew_data += 1
                    

                    post = pd.DataFrame(prob)
                    post.to_csv(dir_path + pl_names[j] + '-posteriors_prob-' + case + '.csv')
                    

                except IndexError:
                    rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]
                    rhostar_case, rhostar_err_case, w_f, w_f_err1, w_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]

                    case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]

                    print("\nWeights sum to zero! Not deriving ecc-omega for " + host_tic + " " + case + "\n")
                    continue   
            else:
                rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]
                rhostar_case, rhostar_err_case, w_f, w_f_err1, w_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]

                case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]

                print("\nWeights sum to zero! Not deriving ecc-omega for " + host_tic + " " + case + "\n")
                continue




            ecc_out = c.analysis.get_summary()['$e$']
            w_out = c.analysis.get_summary()['$\omega$']


            ecc_f = ecc_out[1]
            if str(ecc_out[0]) != 'None' and str(ecc_out[1]) != 'None':
                ecc_f_err1 = ecc_out[0] - ecc_out[1]
            else:
                ecc_f_err1 = np.nan

            if str(ecc_out[2]) != 'None' and str(ecc_out[1]) != 'None':
                ecc_f_err2 = ecc_out[2] - ecc_out[1]
            else:
                ecc_f_err2 = np.nan



            w_f = w_out[1]
            if str(w_out[0]) != 'None' and str(w_out[1]) != 'None':
                w_f_err1 = w_out[0] - w_out[1]
            else:
                w_f_err1 = np.nan

            if str(w_out[2]) != 'None' and str(w_out[1]) != 'None':
                w_f_err2 = w_out[2] - w_out[1]
            else:
                w_f_err2 = np.nan

            print('\necc + errs:', ecc_f, ecc_f_err1, ecc_f_err2)
            print('omega + errs:', w_f, w_f_err1, w_f_err2)

            case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]



            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)

            plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".", color='silver', zorder=-1000)#, alpha=0.3)
            plt.plot(-0.11, 0, ".", ms=10, color='silver', zorder=-1000)#, alpha=0.3)

            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", ms=12, alpha=1.0, zorder=-100) ###### NEW!


            samples0 = np.empty((2, len(x[mask])))
            with model:
                orbit = xo.orbits.KeplerianOrbit(period=p, t0=t0, b=0, rho_star=rho_test, ecc=0, omega=np.pi/2.)
                # Compute the model light curve using starry
                light_curves = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=r_low, t=x[mask], texp=texp)
                light_curve = pm.math.sum(light_curves, axis=-1) + np.median(tr['mean'])
                y_grid = light_curve
                for i, sample in enumerate(xo.get_samples_from_trace(tr, size=2)):
                    samples0[i] = xo.eval_in_model(y_grid, sample)

            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.5, label="Modeled Posterior", zorder=-1)
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.2)

            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.5, lw=3, label="Flat, Circular Model", zorder=-10)
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.5, lw=3)


            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$" + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Detrended relative flux", fontsize=24)
            plt.xlim(-0.1, 0.1); #-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_circ-' + case_tag + '.png')
            plt.close()


            ###########



            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)

            plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".", color='silver', zorder=-1000)#, alpha=0.3)
            plt.plot(-0.11, 0, ".", ms=10, color='silver', zorder=-1000)#, alpha=0.3)

            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", ms=12, alpha=1.0, zorder=-100) ###### NEW!


            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.5, label="Modeled Posterior", zorder=-1)
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.2)

            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.3, lw=3, label="Flat, Circular Model", zorder=-10)
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.1, lw=3)

            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$" + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Detrended relative flux", fontsize=24)
            plt.xlim(-0.1, 0.1); #-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_circ-orig-' + case_tag + '.png')
            plt.close()



            ##### NEW

            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)
            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned", ms=15, alpha=0.6) ###### NEW!

            plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".k", label="data", zorder=-1000)


            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.3, label="Circular Posterior")
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.1)


            #plt.plot(x_fold[isort], y, ".k", label="data")
            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.3, label="Modeled Posterior")
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.1)

            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$" + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Detrended relative flux", fontsize=24)
            plt.xlim(-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_circ-15ms-' + case_tag + '.png')
            plt.close()

            ##

            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)
            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned", ms=15, alpha=0.4) ###### NEW!

            plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".k", label="data", zorder=-1000)


            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.3, label="Circular Posterior")
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.1)


            #plt.plot(x_fold[isort], y, ".k", label="data")
            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.3, label="Modeled Posterior")
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.1)

            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$" + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Detrended relative flux", fontsize=24)
            plt.xlim(-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_circ-15ms-4alph-' + case_tag + '.png')
            plt.close()

            ##

            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)
            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned", ms=12, alpha=0.4) ###### NEW!

            plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".k", label="data", zorder=-1000)


            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.3, label="Circular Posterior")
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.1)


            #plt.plot(x_fold[isort], y, ".k", label="data")
            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.3, label="Modeled Posterior")
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.1)

            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$" + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Detrended relative flux", fontsize=24)
            plt.xlim(-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_circ-4alph-' + case_tag + '.png')
            plt.close()

            #### NEW ABOVE

            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)
            #plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".k", label="data", zorder=-1000)
            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned")
            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.3, label="Circular Posterior")
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.1)
            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.3, label="Modeled Posterior")
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.1)

            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$"  + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = ' + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Detrended relative flux", fontsize=24)
            plt.xlim(-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_binned-' + case_tag + '.png')
            plt.close()


        else:
            rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2 = [np.nan, np.nan, np.nan, np.nan, np.nan]
            rhostar_case, rhostar_err_case, w_f, w_f_err1, w_f_err2 = [np.nan, np.nan, np.nan, np.nan, np.nan]
            rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case, rad_pl_case, rad_pl_err_case = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

            case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]

            print("\nNo valid stellar density available for case: " + case)

        
        ecc_data_array.append(case_data)
        
    ecc_output = pd.DataFrame(ecc_data_array, columns=['tic','full_toi_id','case','rho','rho_err','ecc','ecc_err1','ecc_err2','omega','omega_err1','omega_err2', 'rad_pl', 'rad_pl_err', 'rad_st', 'rad_st_err', 'mass_st', 'mass_st_err'])
    ecc_output.to_csv(dir_path +  pl_names[j] + '-ecc_data.csv')



    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".k", label="data", zorder=-1000)
    plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned")
    plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.3, label="Modeled Posterior")
    plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.1)

    # Annotate the plot with the planet's period
    txt = "TOI " + str(full_toi_ids[j]) + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
        np.mean(tr["period"][:,j]), np.std(tr["period"][:,j])
    )
    plt.annotate(
        txt,
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=20,
    )

    plt.legend(fontsize=20, loc=4)
    plt.title("Model Fit for " + candidate_ids[j], fontsize=25, y=1.03)
    plt.xlabel("Time since mid-transit [days]", fontsize=24)
    plt.ylabel("Detrended relative flux", fontsize=24)
    plt.xlim(-1*bound/2.0, bound/2.0);
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-folded_mod.png')
    plt.close()




    mod = samples[0,:]
    resid = y[mask] - mod
    rms = np.sqrt(np.median(resid ** 2))
    mask00 = np.abs(resid) < 5 * rms



    fig = plt.figure(figsize=[14,4])

    # Overplot the phase binned light curve
    if 4.0*durs_true[j] < 0.5*p[j]:
        bound = 4*durs_true[j]
    else:
        bound = 0.5*p[j]
    bins = np.linspace(-1*bound, bound, 85)
    denom, _ = np.histogram(x_fold[mask00], bins)
    num, _ = np.histogram(x_fold[mask00], bins, weights=resid[mask00] - np.median(tr["mean"]))
    denom[num == 0] = 1.0
    plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", ms=10, zorder=1000, alpha=1.0)


    # Plot the folded transit
    x_fold = (x[mask] - t0[j] + 0.5*p[j])%p[j] - 0.5*p[j]
    m = np.abs(x_fold) < durs_true[j] * 2
    #plt.plot(x_fold[m], resid[mask00][m], 'k.', label='data')

    plt.plot(x_fold[m], resid[m], '.', color='silver')
    plt.plot(-0.11, 0, '.', ms=10, color='silver')
    
    #if len(resid[~mask00]) > 0:
    #    x_fold = (x[mask][~mask00] - t0[j] + 0.5*p[j])%p[j] - 0.5*p[j]
    #    m = np.abs(x_fold) < durs_true[j] * 2
    #    plt.plot(x_fold[m], resid[~mask00][m], color='r', marker='x', lw=0, label="outliers")
        
    plt.legend(fontsize=20, loc=4)
    plt.title("Model Residuals for " + candidate_ids[j], fontsize=25, y=1.03)
    plt.xlabel("Time since mid-transit [days]", fontsize=24)
    plt.ylabel("Residuals of Relative Flux", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(-0.1, 0.1);
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-residuals_mod.png')
    plt.close()
    


#################################################


summary = pm.summary(tr, var_names=varnames)
summary.to_csv(dir_path + sys_name + "-summary.csv")
print( summary ) 


tr_fin = pd.read_csv(dir_path + sys_name + '-trace.csv')

for j in range(len(pers_true)):
    mean_f, rp_f, b_f, rhostar_f, t0_f, per_f, u1_f, u2_f, ecc_f, omega_f = [np.median(tr_fin["mean"]), np.median(tr_fin["r__" + str(j)]), np.median(tr_fin["b__" + str(j)]), np.median(tr_fin["rho__" + str(j)]), np.median(tr_fin["t0__" + str(j)]), np.median(tr_fin["period__" + str(j)]), np.median(tr_fin["u__0"]), np.median(tr_fin["u__1"]), np.median(tr_fin["ecc__" + str(j)]), np.median(tr_fin["omega__" + str(j)])]
    mean_f_err1, rp_f_err1, b_f_err1, rhostar_f_err1, t0_f_err1, per_f_err1, u1_f_err1, u2_f_err1, ecc_f_err1, omega_f_err1  = [np.percentile(q=[15.865], a=tr_fin["mean"])[0], np.percentile(q=[15.865], a=tr_fin["r__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["b__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["rho__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["t0__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["period__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["u__0"])[0], np.percentile(q=[15.865], a=tr_fin["u__1"])[0], np.percentile(q=[15.865], a=tr_fin["ecc__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["omega__" + str(j)])[0]]
    mean_f_err2, rp_f_err2, b_f_err2, rhostar_f_err2, t0_f_err2, per_f_err2, u1_f_err2, u2_f_err2, ecc_f_err2, omega_f_err2  = [np.percentile(q=[84.135], a=tr_fin["mean"])[0], np.percentile(q=[84.135], a=tr_fin["r__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["b__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["rho__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["t0__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["period__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["u__0"])[0], np.percentile(q=[84.135], a=tr_fin["u__1"])[0], np.percentile(q=[84.135], a=tr_fin["ecc__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["omega__" + str(j)])[0]]
    mean_f_u, rp_f_u, b_f_u, rhostar_f_u, t0_f_u, per_f_u, u1_f_u, u2_f_u, ecc_f_u, omega_f_u  = ['','rstar','','g/cc','days','days','', '', '', 'radians']

    outputs_all = [['period', per_f_u, per_f, per_f_err2-per_f, per_f_err1-per_f],
                   ['t0', t0_f_u, t0_f, t0_f_err2-t0_f, t0_f_err1-t0_f],
                   ['rp', rp_f_u, rp_f, rp_f_err2-rp_f, rp_f_err1-rp_f],
                   ['rhostar', rhostar_f_u, rhostar_f, rhostar_f_err2-rhostar_f, rhostar_f_err1-rhostar_f],
                   ['b', b_f_u, b_f, b_f_err2-b_f, b_f_err1-b_f],
                   ['mean', mean_f_u, mean_f, mean_f_err2-mean_f, mean_f_err1-mean_f],
                   ['u1', u1_f_u, u1_f, u1_f_err2-u1_f, u1_f_err1-u1_f],
                   ['u2', u2_f_u, u2_f, u2_f_err2-u2_f, u2_f_err1-u2_f],
                   ['ecc', ecc_f_u, ecc_f, ecc_f_err2-ecc_f, ecc_f_err1-ecc_f],
                   ['omega', omega_f_u, omega_f, omega_f_err2-omega_f, omega_f_err1-omega_f]]
                   #['dur', dur_f_u, dur_f, dur_f_err, dur_f_err]]

    ecc_output = pd.DataFrame(outputs_all, columns=['parameter','units','value','error','error_lower'])
    ecc_output.to_csv(dir_path + pl_names[j] + '-outputs.csv')


for j in range(len(full_toi_ids)):
    
    ecc_data_all = pd.read_csv(dir_path + pl_names[j] + "-ecc_data.csv")
    case_names = list(ecc_data_all["case"])
    rho_vals = list(ecc_data_all["rho"])
    rho_errs = list(ecc_data_all["rho_err"])
    
    for n in range(len(rho_vals)):
        if rho_vals[n] > 0 and rho_errs[n] > 0:
            case_name = case_names[n]
            rhostar_case = rho_vals[n]
            rhostar_err_case = rho_errs[n]

            rho_obs = (rhostar_case, rhostar_err_case)

            rho_tr = np.array(list(tr_temp['rho__' + str(j)]))
            b_tr = np.array(list(tr_temp['b__' + str(j)]))
            r_tr = np.array(list(tr_temp['r__' + str(j)]))

            lower_rho, upper_rho = 0.00001, 250.0
            mu_rho, sigma_rho = rho_obs
            XX = sstat.truncnorm(
                (lower_rho - mu_rho) / sigma_rho, (upper_rho - mu_rho) / sigma_rho, loc=mu_rho, scale=sigma_rho)

            rho_obs_array = XX.rvs(len(rho_tr))

            rho_ratio = rho_tr/rho_obs_array

            b_need = (1+r_tr)*(1-1/rho_ratio**(2/3))**(1/2)

            fig = plt.figure(figsize=[14,7])
            plt.title("Degeneracy test between e and b given " + r"$\rho_{circ}/\rho_{obs}$" + " - " + case_name, fontsize=25, y=1.03)
            plt.hist(b_need, bins=100, label="Circular orbit - " + case_name)
            plt.hist(b_tr, bins=100, alpha=0.8, label='Modeled orbit')
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(fontsize=20)
            plt.xlabel("impact parameter (b)", fontsize=24)
            plt.ylabel("Count density", fontsize=24)
            fig.savefig(dir_path + pl_names[j] + '-e_b-degeneracy-' + case_name + '.png')
            plt.close()

print(sys_name, ' - Complete!')
