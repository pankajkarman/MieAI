import sys
sys.path.append('/home/b/b382177/python/icon-art/')

import artist
import xarray as xr
import numpy as np
import pandas as pd
import glob
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import joblib
import sklearn
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TerminateOnNaN

from mei import Size_Distribution_Optics, MieCoated, Mie, LogNormal


def get_density(ds, core_acc):
    """
    Calculation of density using volume averaging approach.
    """
    dens = 0
    rho = {'dust':2.65e3,'na':2.2e3, 'cl': 2.2e3, 'soot':1.3e3, 'ash':2.65e3, 'h2o':1e3, 'so4':1.8e3, 'nh4':1.8e3, 'no3':1.8e3}
    tot_c = ds[core_acc].to_array(dim='comp').sum(dim='comp')
    for vari in core_acc:
        particle = vari.split('_')[0]
        dens += rho[particle] * ds[vari]
    dens /= tot_c
    return dens

def get_shell_fraction(ds, core_acc, shell_acc): 
    """
    Calculation of Shell thickness as a fration of total diameter.
    
    """
    dens_core = get_density(ds, core_acc)       # average density of core
    dens_shell = get_density(ds, shell_acc)     # average density of shell
    total_mass = ds[core_acc+shell_acc].to_array(dim='comp').sum(dim='comp')           # total mass of the coated particles
    core_mass = ds[core_acc].to_array(dim='comp').sum(dim='comp')                      # mass of the core
    fc_mass =  core_mass / total_mass                                                  # mass fraction of core
    
    fc_vol = fc_mass / (fc_mass + (1-fc_mass) * dens_core / dens_shell)                # conversion of mass fraction to volume fraction
    dc_dt = np.cbrt(fc_vol)                                                            # calculation of core diameter from volume fraction of core
    frac = 1 - dc_dt                                                                   # shell thickness from core diameter
    try:
        frac = frac.rename({'height_2':'height'}) 
    except:
        pass
    return frac 


# Load refractive indices
def read_refractive_indices(filename, columns=["lam", "n", "k"]):
    data = pd.read_csv(
        filename,
        sep="\s+",
        names=columns,
    )
    return data


def interpolate_indices(B_blc1, wavel):
    f1 = interp1d(B_blc1.lam, B_blc1["n"], fill_value="extrapolate", kind="linear")
    f2 = interp1d(B_blc1.lam, B_blc1["k"], fill_value="extrapolate", kind="linear")

    new1 = pd.DataFrame([])
    new1["lam"] = wavel
    new1["n"] = f1(wavel)
    new1["k"] = f2(wavel)
    return new1

def get_ri(B_dust_in, B_seas_in, B_soot_in, lam=0.5, real=True):
    n1, k1 = B_dust_in.set_index('lam').loc[lam]
    n2, k2 = B_seas_in.set_index('lam').loc[lam]
    n3, k3 = B_soot_in.set_index('lam').loc[lam]
    
    if real:
        return n1, n2, n3
    else:
        return k1, k2, k3
    
    
def mod2bin(mu, sig, Ntot=1, nbins=54):    
    dlogd = np.log10(sig) * 0.25
    limit = np.floor(3 * np.log(sig) / dlogd) * dlogd
    dx = np.logspace(-limit, limit, nbins+1)[:-1]
    x_range = mu * dx    
    df = LogNormal(x_range, mu, sig)
    return df, x_range, dlogd


def run_mie_bin(df, index, outpath):
    dx = df.iloc[index, :]
    m_shell = dx['n_shell'] + dx['k_shell'] * 1j
    m_core = dx['n_core'] + dx['k_core'] * 1j

    mr = 1
    mc = m_shell
    mp = m_core

    coating = dx['coating']
    iscoated = coating != 0
    
    const = np.pi * mr / dx['lambda']

    xval = dx['x']
    yval = xval * (1 + coating)

    if iscoated:
        #print('Miecoated called')
        one_result = MieCoated(mp / mr, mc / mr, xval, yval)
    else:
        one_result = Mie(mp / mr, xval)
        
    dx["Extinction"] = one_result[0]
    dx["Scattering"] = one_result[1]
    dx["Absorption"] = one_result[2]
    dx["Asym"] = one_result[4]
    dx["SSA"] = dx["Scattering"] / dx["Extinction"]

    if not outpath:
        return dx
    else:
        dx.to_csv('%s/mie_%s.csv'%(outpath, index), index=True)
        return None
    
    
def get_optical_properties(ex):
    new = pd.Series(dtype='float64')    
    columns = ['Latitude', 'Longitude', 'ilat', 'ilon', 'mu', 'coating', 'lambda', 'n_core', 'k_core', 'n_shell', 'k_shell']
    for col in columns:
        new[col] = ex.iloc[0, :][col]
        
    tvol = (np.pi / 6) * np.sum((ex['tdia']**3) * ex['pdf'] * ex['dlogd'])
    const = ex['area'] * ex['pdf'] * ex['dlogd'] 
    scats = ex["Scattering"] * const
    asymav = np.sum(ex['Asym'] * scats) / np.sum(scats)
    
    cols = ['Extinction', 'Scattering']  
    for col in cols:
        new[col] = (ex[col] * const).sum() / tvol * 1e3
        
    new["SSA"] = new["Scattering"] / new["Extinction"]
    new['g'] = asymav
    return pd.DataFrame(new).T

def to_grid(opt, prop_name, ref):
    gg = np.empty((int(opt.ilat.max())+1, int(opt.ilon.max())+1))
    for idx in opt.index:
        ilt = int(opt.loc[idx, 'ilat'])
        iln = int(opt.loc[idx, 'ilon'])
        gg[ilt, iln] = opt.loc[idx, prop_name]
    gg = xr.DataArray(gg, dims=['Latitude', 'Longitude'], coords=[ref.Latitude, ref.Longitude])
    return gg

def grid_all(opt, ref):
    prop = xr.Dataset()
    oprops = ['mu', 'coating', 'n_core', 'k_core', 'n_shell', 'k_shell', 'Extinction', 'Scattering', 'SSA', 'g']
    for col in oprops:
        prop[col] = to_grid(opt, col, ref) 
    return prop

def get_component(shell, mode='acc'):    
    shell_names = []
    for comp in shell:
        if '_mixed' in comp:
            shell_names.append(comp+'_%s'%mode)
        else:
            shell_names.append(comp+'_mixed_%s'%mode)
    return shell_names

def chem2ri(ds, lam=0.55, mode='acc', core=['ash'], shell=['h2o', 'so4', 'nh4', 'no3'], soot='ash'):
    rho_dust = 2.60
    rho_soot = 1.25
    rho_seas = 1.70

    rho_sul = 1.80
    rho_wat = 1
    rho_org = 1.35
    
    wavel_interp = np.concatenate(
        (
            np.arange(100, 1000, 50),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100001, 10000),
        )
    )

    wavel_interp = np.sort(wavel_interp) / 1000
    
    columns = ["lam", "n", "k"]

    B_dust = read_refractive_indices(
        ".\\input\\BI_USE_NorthSahara_newformat.txt".replace('\\', '/'), columns=columns
    )

    B_wate = read_refractive_indices(".\\input\\BI_USE_H2O.txt".replace('\\', '/'), columns=columns)

    B_sulf = read_refractive_indices(".\\input\\BI_USE_75Sulf215K.txt".replace('\\', '/'), columns=columns)

    B_orga = read_refractive_indices(".\\input\\BI_USE_SOA.txt".replace('\\', '/'), columns=columns)

    B_soot = read_refractive_indices(".\\input\\RI_OCBC30.txt".replace('\\', '/'), columns=columns)

    B_seas = read_refractive_indices(".\\input\\BI_USE_SS_RH70.txt".replace('\\', '/'), columns=columns)
    
    B_wate_in = interpolate_indices(B_wate, wavel_interp)
    B_sulf_in = interpolate_indices(B_sulf, wavel_interp)
    B_orga_in = interpolate_indices(B_orga, wavel_interp)
    B_soot_in = interpolate_indices(B_soot, wavel_interp)
    B_seas_in = interpolate_indices(B_seas, wavel_interp)
    B_dust_in = interpolate_indices(B_dust, wavel_interp)
    
    n1, n2, n3 = get_ri(B_dust_in, B_seas_in, B_soot_in, lam=lam, real=True)
    k1, k2, k3 = get_ri(B_dust_in, B_seas_in, B_soot_in, lam=lam, real=False)

    n4, n5, n6 = get_ri(B_wate_in, B_sulf_in, B_orga_in, lam=lam, real=True)
    k4, k5, k6 = get_ri(B_wate_in, B_sulf_in, B_orga_in, lam=lam, real=False)
    
    f_soot = 0
    f_dust = 0
    f_salt = 0
    
    core_acc = get_component(core, mode=mode)
    shell_acc = get_component(shell, mode=mode)

    core_part = ds[core_acc].to_array().sum(dim='variable')
    shell_part = ds[shell_acc].to_array().sum(dim='variable')

    if '%s_mixed_%s'%(soot, mode) in core_acc:
        f_soot = (ds['%s_mixed_%s'%(soot, mode)] / core_part).squeeze()
    
    if 'dust_mixed_%s'%mode in core_acc:
        f_dust = (ds['dust_mixed_%s'%mode] / core_part).squeeze()
        
    if 'na_mixed_%s'%mode in core_acc:
        f_salt = ((ds['na_mixed_%s'%mode] + ds['cl_mixed_%s'%mode]) / core_part).squeeze()  
    
    f_wat = (ds['h2o_mixed_%s'%mode]  / shell_part).squeeze()
    f_sul = (ds['so4_mixed_%s'%mode]  / shell_part).squeeze()
    f_org = ((ds['nh4_mixed_%s'%mode] + ds['no3_mixed_%s'%mode])  / shell_part).squeeze()
    
    rho_core =  (f_dust * rho_dust + f_salt * rho_seas + f_soot * rho_soot) 
    rho_shell =  (f_org * rho_org + f_wat * rho_wat + f_sul * rho_sul) 
    
    real_ri_core =  f_dust * n1 + f_salt * n2 + f_soot * n3 
    imag_ri_core =  f_dust * k1 + f_salt * k2 + f_soot * k3

    real_ri_shell =  f_wat * n4 + f_sul * n5 + f_org * n6 
    imag_ri_shell =  f_wat * k4 + f_sul * k5 + f_org * k6
    
    xri = xr.Dataset()
    xri['real_core'] = real_ri_core
    xri['imag_core'] = imag_ri_core
    xri['real_shell'] = real_ri_shell
    xri['imag_shell'] = imag_ri_shell    
    
    ds_dt = xr.Dataset()
    ds_dt[mode] = get_shell_fraction(ds, core_acc, shell_acc)
    coat = ds_dt.squeeze()[mode]
    return xri, coat

def get_bins(ds, gridfile, mode='acc', bins=15, map_ext=(-61.5, -58, 11.5, 15.0), dxx=0.5):
    rho = {'dust':2.65e3,'na':2.2e3, 'cl': 2.2e3, 'soot':1.3e3, 'ash':2.65e3}
    #sigma = {'ait':1.7, 'acc':1.7, 'coa':2.2}  # Heike
    sigma = {'ait':1.7, 'acc':2.0, 'coa':2.2}  # Julia
    
    lon_vec = np.arange(map_ext[0], map_ext[1], dxx)
    lat_vec = np.arange(map_ext[2], map_ext[3], dxx)
    
    ntot = ds['nmb_mixed_%s'%mode].icon.regrid(gridfile, lon_vec, lat_vec)
    mu = 0.5*ds['diam_mixed_%s'%mode].icon.regrid(gridfile, lon_vec, lat_vec) * 1e9   # convert to nano meter
    sig = sigma[mode]    
   
    rads = np.empty((bins, len(mu.Latitude), len(mu.Longitude)))
    size_dist = np.empty((bins, len(mu.Latitude), len(mu.Longitude)))
    dxx = np.empty((len(mu.Latitude), len(mu.Longitude)))
    ilt = np.empty((len(mu.Latitude), len(mu.Longitude)))
    iln = np.empty((len(mu.Latitude), len(mu.Longitude)))

    rrads = np.empty((500, len(mu.Latitude), len(mu.Longitude)))
    rsize_dist = np.empty((500, len(mu.Latitude), len(mu.Longitude)))

    for j in np.arange(len(mu.Latitude)): 
        for i in np.arange(len(mu.Longitude)):
            size_dist1, rads1, dx1 = mod2bin(mu[j, i].values, sig, ntot[j, i].values, 500)
            size_dist2, rads2, dx2 = mod2bin(mu[j, i].values, sig, ntot[j, i].values, bins)

            size_dist[:, j, i] = size_dist2
            rads[:, j, i] = rads2
            dxx[j, i] = dx2
            ilt[j, i] = j
            iln[j, i] = i

            rsize_dist[:, j, i] = size_dist1
            rrads[:, j, i] = rads1
            
    new = xr.Dataset()
    new['size_dist'] = xr.DataArray(size_dist, dims=['bins', 'lat', 'lon'], coords=[np.arange(bins), mu.Latitude, mu.Longitude])
    new['radius'] = xr.DataArray(rads, dims=['bins', 'lat', 'lon'], coords=[np.arange(bins), mu.Latitude, mu.Longitude])
    new['dx'] = xr.DataArray(dxx, dims=['lat', 'lon'], coords=[mu.Latitude, mu.Longitude])
    new['ilat'] = xr.DataArray(ilt, dims=['lat', 'lon'], coords=[mu.Latitude, mu.Longitude])
    new['ilon'] = xr.DataArray(iln, dims=['lat', 'lon'], coords=[mu.Latitude, mu.Longitude])
    new['mu'] = xr.DataArray(mu.values, dims=['lat', 'lon'], coords=[mu.Latitude, mu.Longitude])
    #new.to_netcdf('/work/bb1070/b382177/mie_data/bins_julia.nc')
    return new.rename({'lon':'Longitude', 'lat':'Latitude'})

def mie_call(gridfile, ri, coat, tmp, lam=0.55, map_ext=(-61.5, -58, 11.5, 15.0), dxx=0.5):
    lam = lam * 1000
    rads = tmp['radius'] 
    sz = tmp['size_dist']
    gap = tmp['dx']
    mu = tmp['mu']
    ilat = tmp['ilat']
    ilon = tmp['ilon']
    
    lon_vec = np.arange(map_ext[0], map_ext[1], dxx)
    lat_vec = np.arange(map_ext[2], map_ext[3], dxx)
    
    coat = coat.icon.regrid(gridfile, lon_vec, lat_vec)
    #coat = xr.where(coat<=0.5, coat, np.nan)   
    
    real_ri_core = ri['real_core'].icon.regrid(gridfile, lon_vec, lat_vec)
    imag_ri_core = ri['imag_core'].icon.regrid(gridfile, lon_vec, lat_vec)
    real_ri_shell = ri['real_shell'].icon.regrid(gridfile, lon_vec, lat_vec)
    imag_ri_shell = ri['imag_shell'].icon.regrid(gridfile, lon_vec, lat_vec)
    
    nbins = tmp.bins.shape[0]
    data = pd.DataFrame()
    for i in np.arange(nbins):
        new = xr.Dataset()
        new['dia'] = rads.isel(bins=i)
        new['area'] = (np.pi / 4) * (new['dia']**2)
        new['tdia'] = new['dia'] * (1+coat)
        new['pdf'] = sz.isel(bins=i)
        new['dlogd'] = gap
        new['mu'] = mu
        new['ilat'] = ilat
        new['ilon'] = ilon
        new['x'] = np.pi*rads.isel(bins=i) / lam
        new['coating'] = coat
        new['lambda'] = lam
        new['n_core'] = real_ri_core
        new['k_core'] = imag_ri_core
        new['n_shell'] = real_ri_shell
        new['k_shell'] = imag_ri_shell
        data = pd.concat([data, new.to_dataframe().reset_index()], axis=0)
    data.index = np.arange(data.shape[0])
    data = data[~data.coating.isna()]  
    print(data.shape)    
    print('Input data preparation done...')    
    print('Performing Mie calculations now...')
    
    ndf = pd.DataFrame([])
    start = time.time()
    for idx in data.index:
        try:
            tm = run_mie_bin(data, idx, outpath='')
            ndf = pd.concat([ndf, pd.DataFrame(tm).T], axis=0)
        except:
            pass  
    end = time.time()
    print('Elapsed time in Mie calculation: %s'%(end - start))
    print('Calculating bulk properties now...')
    gp = ndf.groupby(by=['Latitude', 'Longitude'])
    opt = pd.DataFrame([])
    for group, tdata in gp:
        opt1 = get_optical_properties(tdata)
        opt = pd.concat([opt, opt1], axis=0)

    opt.index = np.arange(opt.shape[0])  
    
    print('Gridding optical properties now...')
    prop = grid_all(opt, mu)
    print('Mie calcualtions finished.')
    return prop, ndf  

class Mie():
    def __init__(self, data, gridfile, wavelength, map_extent, grid_resolution, mode='acc', nbins=15, core=['ash'], shell=['h2o', 'so4', 'nh4', 'no3'], soot='ash'):
        self.data = data
        self.mode = mode
        self.nbins = nbins
        self.gridfile = gridfile
        self.wavelength = wavelength
        self.map_extent = map_extent
        self.grid_resolution = grid_resolution
        self.core = core
        self.shell = shell
        self.soot = soot
        
    def preprocess(self):
        print('Calculating Shell thickness and Mapping composition to Refractive Indices...')
        self.ri, self.coat = chem2ri(self.data, lam=self.wavelength, mode=self.mode, core=self.core, shell=self.shell, soot=self.soot)
        
        map_ext = self.map_extent
        lon_vec = np.arange(map_ext[0], map_ext[1], self.grid_resolution)
        lat_vec = np.arange(map_ext[2], map_ext[3], self.grid_resolution)
        self.rcoat = self.coat.icon.regrid(self.gridfile, lon_vec, lat_vec)
        
        print('Mapping Mode to Bins...')
        self.bins = get_bins(self.data, self.gridfile, mode=self.mode, bins=self.nbins, map_ext=self.map_extent, dxx=self.grid_resolution)
        
    def calculate_optics(self):
        self.preprocess()
        print('Calculating Optical properties now...')
        self.optics, self.ndf = mie_call(self.gridfile, self.ri, self.coat, self.bins, lam=self.wavelength, map_ext=self.map_extent, dxx=self.grid_resolution)  
        return self.optics
    
    def emulate2(self, model1='/work/bb1070/b382177/mie/icon/26-06-2023/model/model_26_06_2023_x1.h5', model2='/work/bb1070/b382177/mie/icon/26-06-2023/model/model_26_06_2023_x2.h5', fscale1='/work/bb1070/b382177/mie/icon/26-06-2023/mlp_min_max_x1.csv', fscale2='/work/bb1070/b382177/mie/icon/26-06-2023/mlp_min_max_x2.csv', x_criteria=0.2):
        ff1 = pd.read_csv(fscale1, names=['col', 'max', 'min'], skiprows=[0]).set_index('col')
        ff2 = pd.read_csv(fscale2, names=['col', 'max', 'min'], skiprows=[0]).set_index('col')
        
        out_col = ["Extinction", "Scattering", "Asym"]
        cols = ['x', 'coating', 'n_core', 'k_core', 'n_shell', 'k_shell']
        
        df = self.ndf.drop(out_col, axis=1)        
        df1 = ff1.loc[cols, :]
        df2 = ff2.loc[cols, :]
        
        
        dff1 = ff1.loc[out_col, :]       
        dff2 = ff2.loc[out_col, :]
        
        dfx1 = (df[cols] - df1['min']) / (df1['max'] - df1['min'])
        dfx2 = (df[cols] - df2['min']) / (df2['max'] - df2['min'])
        
        idx1 = df.index[df.x <= x_criteria]
        idx2 = df.index[df.x > x_criteria]
        
        print('Performing MLP emulation using model checkpoints: %s and %s'%(model1, model2))        
        lmodel1 = tf.keras.models.load_model(model1)
        lmodel2 = tf.keras.models.load_model(model2)        
        
        in1 = dfx1.loc[idx1,:]
        pred1 = lmodel1.predict(in1)
        pred1 = pd.DataFrame(pred1, columns=out_col, index=idx1)  
        pred1 = (dff1['max'][out_col] - dff1['min'][out_col]) * pred1 + dff1['min'][out_col]        
        
        in2 = dfx2.loc[idx2,:]
        pred2 = lmodel2.predict(in2)
        pred2 = pd.DataFrame(pred2, columns=out_col, index=idx2)
        pred2 = (dff2['max'][out_col] - dff2['min'][out_col]) * pred2 + dff2['min'][out_col]

        y1 = pd.concat([pred1, pred2], axis=0).sort_index()       
        df[out_col] = y1[out_col]

        gp = df.groupby(by=['Latitude', 'Longitude'])
        opt = pd.DataFrame([])
        for group, tdata in gp:
            opt1 = get_optical_properties(tdata)
            opt = pd.concat([opt, opt1], axis=0)    
        opt.index = np.arange(opt.shape[0])
        self.emu = grid_all(opt, self.bins['mu'])
        return self.emu
        
    def emulate1(self, model_name='/work/bb1070/b382177/mie_data/model_x1_17_04_2023.h5', fscale='/work/bb1070/b382177/mie_data/mlp_min_max.csv', qt_model='/work/bb1070/b382177/mie/icon/05-07-2023/model/transformer.pkl'):
        qt = joblib.load(qt_model)
        ff = pd.read_csv(fscale, names=['col', 'max', 'min'], skiprows=[0]).set_index('col')
        
        out_col = ["Extinction", "Scattering", "Asym"]
        cols = ['coating', 'x', 'n_core', 'k_core', 'n_shell', 'k_shell', 'lambda']
        
        df = self.ndf.drop(out_col, axis=1)        
        df1 = ff.loc[cols, :]
        df2 = ff.loc[out_col, :]
        dfx = (df[cols] - df1['min']) / (df1['max'] - df1['min'])

        lmodel1 = tf.keras.models.load_model(model_name)
        print('Performing MLP emulation using model checkpoint: %s'%model_name)
        start = time.time()
        pred1 = lmodel1.predict(dfx, batch_size=8192)
        end = time.time()
        print('Elapsed time in emulation: %s'%(end - start))
        pred1 = qt.inverse_transform(pred1)
        
        y1 = pd.DataFrame(pred1, columns=out_col, index=dfx.index)
        y1 = (df2['max'][out_col] - df2['min'][out_col]) * y1 + df2['min'][out_col]
        df[out_col] = y1[out_col]

        gp = df.groupby(by=['Latitude', 'Longitude'])
        opt = pd.DataFrame([])
        for group, tdata in gp:
            opt1 = get_optical_properties(tdata)
            opt = pd.concat([opt, opt1], axis=0)    
        opt.index = np.arange(opt.shape[0])
        self.emu = grid_all(opt, self.bins['mu'])
        return self.emu

@xr.register_dataarray_accessor('mie')
class DataAccessor(object):
    def __init__(self, da):
        self._obj = da
        
    def cut(self, gridfile='/scratch/b/b380982/ICON-OUTPUT/2_Gnu_Age/iconR2B06_DOM01.nc', map_ext=(-61.5, -58, 11.5, 15.0), lev=70, dxx=0.05):     
        new = self._obj
        lon_vec = np.arange(map_ext[0], map_ext[1], dxx)
        lat_vec = np.arange(map_ext[2], map_ext[3], dxx)
        # try:
        #     new = self._obj.isel(height_2=lev)
        # except:
        #     new = self._obj.isel(height=lev) 
        # else:
        #     pass
        return new.icon.regrid(gridfile, lon_vec, lat_vec)