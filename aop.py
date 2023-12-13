import cartopy.crs as ccrs
import cartopy.feature as cf

from numba import jit, njit, vectorize, guvectorize, prange
import dask.dataframe as dd
# import modin.pandas as pd
# import swifter

import glob
import joblib
import string
# import xesmf as xe
from mie_icon_art import *
from scipy.interpolate import griddata

from joblib import Parallel, delayed
import multiprocessing

import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)


def get_dataset_grid(grid):
    # mathematical and physical constants
    rad2deg = 180.0/np.pi

    # grid dataset
    ds_grid = ( xr.open_dataset(grid, autoclose=True)
               [['cell_area','clat','clon','clon_vertices','clat_vertices']].
               rename({'cell': 'ncells'}) )

    # convert grid from radians to degrees
    ds_grid['clon'] = ds_grid['clon']*rad2deg
    ds_grid['clat'] = ds_grid['clat']*rad2deg
    ds_grid['clon_vertices'] = ds_grid['clon_vertices']*rad2deg
    ds_grid['clat_vertices'] = ds_grid['clat_vertices']*rad2deg
    return ds_grid

def get_dataset_icon_chem1(file, ds_grid, al):
    # 3-d tracer fields
    ds_3dtracer = (xr.open_mfdataset(file, autoclose=True)[al])

    # merge datasets
    ds = xr.merge([ds_3dtracer, ds_grid])
    ds = ds.isel(time=0)
    return ds

def load_indices(lam=0.55):
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
    
    n = [n1, n2, n3, n4, n5, n6]
    k = [k1, k2, k3, k4, k5, k6]
    return n, k

def prepare(ds, n, k, lam=0.55, mode='acc', core=['ash'], shell=['h2o', 'so4', 'nh4', 'no3'], soot='ash'):
    rho_dust = 2.60
    rho_soot = 1.25
    rho_seas = 1.70

    rho_sul = 1.80
    rho_wat = 1
    rho_org = 1.35
    
    f_soot = 0
    f_dust = 0
    f_salt = 0
    
    [n1, n2, n3, n4, n5, n6] = n
    [k1, k2, k3, k4, k5, k6] = k
    
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

@jit(nopython=True)
def mod2bin1(mu, sig, ou, Ntot=1, nbins=54):  
    loggsd = np.log10(sig)
    const = loggsd * np.sqrt(2 * np.pi)
    dlogd = np.log10(sig) * 0.25
    limit = np.floor(3 * np.log(sig) / dlogd) * dlogd
    dx = np.logspace(-limit, limit, nbins+1)[:-1]
    
    x_range = mu * dx    
    pdf = np.exp(-np.log10(x_range / mu) ** 2 / (2 * loggsd**2)) / const
    
    ou[:, 0] = pdf
    ou[:, 1] = x_range
    ou[:, 2] = dlogd 
    
@jit(nopython=True, parallel=True)
def mod2bin2(xx1, inp):        
    nn = xx1.shape[0] 
    szs = np.zeros((15*nn, 3))    
    out = np.array_split(szs, nn, axis=0)    
    for i in prange(nn):
        mod2bin1(inp[i], 2, out[i], nbins=15)
    return szs
         
@jit(nopython=True, parallel=True)
def q2k(arr, opt):
    def eff2co(ex, opt): 
        tvol = np.sum(ex[:, 5])
        opt[:, 0] = np.sum(ex[:, -3] * ex[:, 6]) /  tvol * 1e3
        opt[:, 1] = np.sum(ex[:, -2] * ex[:, 6]) /  tvol * 1e3
        opt[:, 3] = np.sum(ex[:, -1] * ex[:, 7]) / np.sum(ex[:, 7])
        opt[:, 2] = opt[:, 1] / opt[:, 0]
    
    nn = opt.shape[0]
    inp = np.array_split(arr, nn, axis=0)
    out = np.array_split(opt, nn, axis=0)    
    for i in prange(nn):
        eff2co(inp[i], out[i])

@xr.register_dataset_accessor('icon3')
@xr.register_dataarray_accessor('icon3')
class IconAccessor(object):
    def __init__(self, ds):
        self._obj = ds
        
    def regrid(self, lon, lat, method='linear', ltranslon=True):
        y, x = np.meshgrid(lat, lon)
        nda = griddata((self._obj.clon, self._obj.clat), self._obj, (x, y), method=method)
        nda = xr.DataArray(nda, dims=['Longitude', 'Latitude'], coords=[lon, lat])   
        return nda.T
        
    @property    
    def dz(self):
        dz1 = -1 * self._obj.z_ifc.diff('height')
        dz1 = dz1.assign_coords(height=(dz1.height - 1))
        return dz1.rename({'height':'height_2'}) 

class MieAI():
    def __init__(self, data, wavelength, mode='acc', 
                 core=['ash'], shell=['h2o', 'so4', 'nh4', 'no3'], soot='ash',
                 model = '/work/bb1070/b382177/mie/icon/17-08-2023/model/model_17-08-2023.h5',
                 fscale = "/work/bb1070/b382177/mie/icon/05-07-2023/model/mlp_min_max.csv",
                 qt_model = '/work/bb1070/b382177/mie/icon/17-08-2023/model/transformer_17-08-2023.pkl',
                 nbins=15, verbose=1, ncpus=10):
        self.data = data
        self.mode = mode
        self.nbins = nbins
        self.wavelength = wavelength
        self.core = core
        self.shell = shell
        self.soot = soot
        self.fscale = fscale
        self.qt_model = qt_model
        self.qt = joblib.load(self.qt_model)
        self.scale = pd.read_csv(self.fscale, names=['col', 'max', 'min'], skiprows=[0]).set_index('col') 
        self.model = tf.keras.models.load_model(model)
        self.verbose = verbose
        self.ncpus = ncpus
        print('Running MieAI on mode: %s'%self.mode)
        
    def preprocess(self):
        self.n, self.k = load_indices(lam=self.wavelength)
        self.input, self.coat = prepare(self.data, self.n, self.k, lam=self.wavelength, mode=self.mode, core=self.core, shell=self.shell, soot=self.soot)
        self.input['coating'] = self.coat.rename({'height':'height_2'})
        self.input['med_diam'] = self.data['diam_mixed_%s'%self.mode]
        self.input['x'] = 1e6 * np.pi * self.data['diam_mixed_%s'%self.mode] / self.wavelength
        self.input = self.input.squeeze().rename({'real_core':'n_core', 'imag_core':'k_core', 'real_shell':'n_shell', 'imag_shell':'k_shell'})
        return self.input 
    
    def get_aod(self):
        modes = self.data.icon.get_modes(mode_type='mixed_%s'%self.mode)[1:-1]
        conc = self.data[modes].to_array(dim='species').sum(dim='species')        
        const = (self.data.icon3.dz * self.data['rho'] * conc * 1e-6).squeeze()
        
        inp = self.preprocess()
        ext = self.get_aop(inp).isel(aop=0).squeeze()
        
        aod = (ext*const).integrate('height_2')
        aod.name = 'mixed_%s_aod'%self.mode
        return aod
    
    def get_aop(self, xx):
        ext = np.zeros((len(xx.ncells), 4, len(xx.height_2)))          
        for h in np.arange(len(xx.height_2)):
            if self.verbose:
                print('Working on level %s'%h)
            ext[:, :, h] = self.emulate(xx.isel(height_2=h))           
        ext = xr.DataArray(ext, dims=['ncells', 'aop', 'height_2'], coords=[xx.ncells, ['ext', 'sca', 'ssa', 'asy'], xx.height_2])
        return ext
    
    def emulate(self, df):        
        out_col = ["Extinction", "Scattering", "Asym"]
        cols = ['coating', 'x', 'n_core', 'k_core', 'n_shell', 'k_shell', 'lambda']
        gcols = ['tdia', 'pdf', 'dlogd', 'area', 'ncells', 'vol', 'const', 'scats', "Extinction", "Scattering", "Asym"]
        
        nn = len(self.data.ncells)
        opt = np.empty((nn, 4)) 
        
        df1 = df.to_pandas().drop('time', axis=1)
        cnames = df1.columns
        inp = df1.med_diam.values
        df1 = df1.values

        szs = mod2bin2(df1, inp)
        
        df = np.repeat(df1, repeats=15, axis=0)
        df = pd.DataFrame(df, columns=cnames)
        
        df['pdf'] = szs[:, 0]
        df['dia'] = szs[:, 1] * 1e9
        df['dlogd'] = szs[:, 2]

        df['tdia'] = df['dia'] * (1 + df.coating)
        df['area'] = (np.pi / 4) * (df['dia']**2)
        df['ncells'] = df.index        
        df['lambda'] = self.wavelength
     
        df1 = self.scale.loc[cols, :]
        df2 = self.scale.loc[out_col, :]        
        dfx = (df[cols] - df1['min']) / (df1['max'] - df1['min'])

        # print('Performing MLP emulation using model checkpoint: %s'%model_name)
        # start = time.time()
        pred1 = self.model.predict(dfx, verbose=0, batch_size=50000, use_multiprocessing=True, workers=self.ncpus)
        # end = time.time()
        # print('Elapsed time in emulation: %s'%(end - start))
        pred1 = self.qt.inverse_transform(pred1)

        y1 = pd.DataFrame(pred1, columns=out_col, index=dfx.index)
        y1 = (df2['max'][out_col] - df2['min'][out_col]) * y1 + df2['min'][out_col]
        df[out_col] = y1[out_col]
        
        df['vol'] = (np.pi / 6) * (df['tdia']**3) * df['pdf'] * df['dlogd']
        df['const'] = df['area'] * df['pdf'] * df['dlogd'] 
        df['scats'] = df["Scattering"] * df['const'] 
            
        arr = df[gcols].values 
        q2k(arr, opt)
        return opt