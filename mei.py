import numpy as np
import pandas as pd
from scipy import interpolate
import PyMieScatt as ps

def degree2radian(ang):
    rad = ang * np.pi / 180.0
    return rad


def LogNormal(diam, mu, gsd):
    """Log Normal distribution"""
    x = diam / mu
    loggsd = np.log10(gsd)
    const = loggsd * np.sqrt(2 * np.pi)
    pdf = np.exp(-np.log10(x) ** 2 / (2 * loggsd**2)) / const
    return pdf


def LogNormal_dexp(diam, mu, gsd):
    """Log Normal distribution"""
    x = diam / mu
    loggsd = np.log(gsd)
    const = diam * loggsd * np.sqrt(2 * np.pi)
    pdf = np.exp(-np.log(x) ** 2 / (2 * loggsd**2)) / const
    return pdf


def ReadNephSens(filename):
    """
    reads Nephelometer data
    """
    df = pd.read_excel(filename)
    return df


# In[6]:


def Mie(m, x):
    # nMedium = nMedium.real
    # m /= nMedium
    # wavelength /= nMedium
    # x = np.pi * diameter / wavelength

    if x == 0:
        return 0, 0, 0, 1.5, 0, 0, 0
    elif x > 0:
        nmax = np.round(2 + x + 4 * (x ** (1 / 3)))
        n = np.arange(1, nmax + 1)
        n1 = 2 * n + 1
        n2 = n * (n + 2) / (n + 1)
        n3 = n1 / (n * (n + 1))
        x2 = x**2

        an, bn = ps.Mie_ab(m, x)

        qext = (2 / x2) * np.sum(n1 * (an.real + bn.real))
        qsca = (2 / x2) * np.sum(
            n1 * (an.real**2 + an.imag**2 + bn.real**2 + bn.imag**2)
        )
        qabs = qext - qsca

        g1 = [
            an.real[1 : int(nmax)],
            an.imag[1 : int(nmax)],
            bn.real[1 : int(nmax)],
            bn.imag[1 : int(nmax)],
        ]
        g1 = [np.append(x, 0.0) for x in g1]
        g = (4 / (qsca * x2)) * np.sum(
            (
                n2
                * (
                    an.real * g1[0]
                    + an.imag * g1[1]
                    + bn.real * g1[2]
                    + bn.imag * g1[3]
                )
            )
            + (n3 * (an.real * bn.real + an.imag * bn.imag))
        )

        qpr = qext - qsca * g
        qback = (1 / x2) * (np.abs(np.sum(n1 * ((-1) ** n) * (an - bn))) ** 2)
        qratio = qback / qsca
        # return qext, qsca, qabs, g, qpr, qback, qratio
        return qext, qsca, qabs, qback, g, qratio


# In[7]:


def MieCoated(
    mCore,
    mShell,
    dCore,
    dShell,
    nMedium=1.0,
    asDict=False,
    asCrossSection=False,
):
    if nMedium != 1.0:
        warnings.warn(
            "Note: the use of nMedium was incorporated naievely and the result should be carefully scrutinized."
        )
    xCore = dCore
    xShell = dShell
    if xCore == xShell:
        return Mie(mCore, dShell)

    elif xCore == 0:
        return Mie(mShell, dShell)

    elif mCore == mShell:
        return Mie(mCore, dShell)

    elif xCore > 0:
        nMedium = nMedium.real
        # wavelength /= nMedium  # The choice was either to redefine the wavelength, or the xCore & xShell, I left just for indication, your call.
        mCore /= nMedium
        mShell /= nMedium
        xCore = (
            dCore * nMedium
        )  # Not ideal to redefine xCore and xShell, but it seems need in order to keep MieQ conditions in place
        xShell = dShell * nMedium

        nmax = np.round(2 + xShell + 4 * (xShell ** (1 / 3)))
        n = np.arange(1, nmax + 1)
        n1 = 2 * n + 1
        n2 = n * (n + 2) / (n + 1)
        n3 = n1 / (n * (n + 1))
        xShell2 = xShell**2

        an, bn = ps.CoreShell.CoreShell_ab(mCore, mShell, xCore, xShell)

        qext = (2 / xShell2) * np.sum(n1 * (an.real + bn.real))
        qsca = (2 / xShell2) * np.sum(
            n1 * (an.real**2 + an.imag**2 + bn.real**2 + bn.imag**2)
        )
        qabs = qext - qsca

        g1 = [
            an.real[1 : int(nmax)],
            an.imag[1 : int(nmax)],
            bn.real[1 : int(nmax)],
            bn.imag[1 : int(nmax)],
        ]
        g1 = [np.append(x, 0.0) for x in g1]
        g = (4 / (qsca * xShell2)) * np.sum(
            (
                n2
                * (
                    an.real * g1[0]
                    + an.imag * g1[1]
                    + bn.real * g1[2]
                    + bn.imag * g1[3]
                )
            )
            + (n3 * (an.real * bn.real + an.imag * bn.imag))
        )

        qpr = qext - qsca * g
        qback = (1 / xShell2) * (np.abs(np.sum(n1 * ((-1) ** n) * (an - bn))) ** 2)
        qratio = qback / qsca
        return qext, qsca, qabs, qback, g, qratio


# In[8]:


def Mie_S12(m, x, mu):
    mie = ps.MieS1S2(m, x, mu)
    return mie


def Miecoated_S12(mCore, mShell, xCore, xShell, mu):
    mie = ps.CoreShellS1S2(mCore, mShell, xCore, xShell, mu)
    return mie


# In[9]:


def scattering_weights(*args):
    """
    returns vectors of dtheta and weights for calculating angular-weighted scattering.
    """
    varargin = args
    nargin = len(varargin)

    angres = 0.25
    nang = 180 / angres
    degs = np.arange(0, 180, angres)
    theta = degree2radian(degs)

    dtheta = np.zeros_like(theta)
    dtheta[1:-1] = degree2radian(angres)
    dtheta[0] = degree2radian(angres * 0.5)
    dtheta[-1] = degree2radian(angres * 0.5)

    wtlist = {}
    wt1 = np.sin(theta)
    bsflag = np.zeros_like(theta)
    bsflag[degs > 90] = 1
    bsflag[degs == 90] = 0.5

    wtlist[0] = wt1
    wtlist[1] = wt1 * bsflag

    for i in np.arange(nargin):
        wtmat = np.asarray(varargin[i]).astype(np.float)
        f = interpolate.interp1d(wtmat[0], wtmat[1])
        wtlist[i + 2] = f(degs)

    return theta, dtheta, wtlist


def weighted_scattering(m, x, theta, dtheta, wtvecs, m_coating=0, ycoat=1):
    """returns integrated, weighted scattering"""

    mc = m_coating
    ycoat = ycoat
    iscoat = mc != 0 and ycoat > x

    smag = np.zeros((2, len(theta)))

    for i in np.arange(len(theta)):
        if not iscoat:
            smag[:, i] = Mie_S12(m, x, np.cos(theta[i]))
        else:
            smag[:, i] = Miecoated_S12(m, mc, x, ycoat, np.cos(theta[i]))

    ssq = np.abs(smag) ** 2
    ssq = ssq[0, :] + ssq[1, :]
    ret = {}

    for i in np.arange(len(wtvecs)):
        mieret = np.sum(ssq * wtvecs[i] * np.asarray(dtheta))
        if iscoat:
            mieret = mieret / (ycoat**2)
        else:
            mieret = mieret / (x**2)
        ret[i] = mieret
    return ret


# In[10]:


def Forcing_Efficiency(vac_bscat, vac_abs, surf_alb=0.16, cloud_frac=0.6):
    """
    returns forcing efficiency in watt/cm3 aerosol
    """
    atrans = 0.79
    s0 = 1370.0
    smult = -(s0 / 4) * (atrans**2) * (1 - cloud_frac)
    feff = smult * ((1 - surf_alb) ** 2 * 2 * vac_bscat - 4 * surf_alb * vac_abs)
    return feff


# In[11]:


def Size_Distribution_Optics(
    mp,
    sizepar1,
    sizepar2,
    wavelength,
    m_medium=1,
    m_coating=1,
    density=1,
    nobackscat=False,
    nephscats=False,
    nephsensfile="AndersonOgren1998.csv",
    cut=1e9,
    coating=0,
    effcore=False,
    normalized=True,
    resolution=10.0,
    vectorout=False,
):
    """
    Arguments:
    mp                   particle refractive index
    sizepar1, sizepar2   count mean dia in nm, geometric std dev if scalar
    or                   d, dNdlogD(cm-3) if vector
    wavelength           wavelength (nm)

    Optional arguments:
    'm_medium'   refractive index of surrounding medium (default 1)
    'm_coating'  refractive index of coating (default mp)
    'density'    particle density in g/cm3 (default 1)
    'nobackscat' if true, don't calculate back scattering (quicker; default False)
    'nephscats'  produce truncated scattering for TSI 3563 (default False)
    'nephsensfile'   file name for neph angular sensitivity (default AndersonOgren1998.csv).
     nephsens file must have columns angle, scat_angsens, ref_sine, backscat_angsens, ref_backscatsine
    'cut'        actual cut size (removes large particles; default None)
    'coating'    fractional increase in diameter due to coating (default zero; can be scalar or vector)
    'effcore'    calculates cross-section as m2/(g of core) (default True)
    'normalized' normalized to m2/g particles (default True). Non-normalized only works with (d, dNdlogD)
    'resolution' bins per decade (no effect if distribution is given; default 10)
    'vectorout'  output a vector instead of a structure (default False)

    Output:
    If normalized, optical cross-sections per mass (m2/g); otherwise Bep, Bsp, Bap (Mm-1).
    Also ssa (dimensionless), forcing efficiency (W/g) as described in Bond&Bergstrom 2006.

    Based on MATLAB code written by Tami Bond, University of Illinois, yark@uiuc.edu
    """
    if np.shape(sizepar1) != np.shape(sizepar2):
        raise Exception("Size parameter arrays must have the same dimensions")

    mr = m_medium
    mc = m_coating
    fcoat = coating
    dens = density
    nobackscat = nobackscat
    nephsensfile = nephsensfile
    cut = cut
    resol = resolution
    norm2core = effcore
    norm2volume = normalized
    res = 1 / resol
    vecout = vectorout

    # Read neph angular sensitivities if required
    if nephscats:
        nephdat = ReadNephSens(nephsensfile)
        [theta, dtheta, scatwts] = scattering_weights(
            [nephdat.angle, nephdat.scat_angsens], [nephdat.angle, nephdat.bs_angsens]
        )
        scatidx = [1, 2, 3]
    else:
        [theta, dtheta, scatwts] = scattering_weights()
        scatidx = 1

    if np.isscalar(sizepar1):
        dlogd = min(res * 1.0, np.log10(sizepar2) * 0.25)
        limit = np.floor(3 * np.log(sizepar2) / dlogd) * dlogd

        dx = np.arange(-limit, limit, dlogd)
        x_range = sizepar1 * (10.0**dx)
        print(x_range)
        df = LogNormal(x_range, sizepar1, sizepar2)
        dexp = LogNormal_dexp(x_range, sizepar1, sizepar2)
        dexp = np.array(dexp)

        dDp = np.zeros_like(x_range)
        Nx = len(x_range)
        for i in range(Nx - 1):
            if i in [0, Nx - 1]:
                dDp[i] = x_range[i + 1] - x_range[i]
            else:
                dDp[i] = 0.5 * (x_range[i + 1] - x_range[i - 1])

        dDp[-1] = x_range[-1] - x_range[-2]

        # Example of size distribution properties
        N = 3000  #          % [1/m^3], number of particles per unit volume
        Dp = x_range * 1e-9  #          % [m], diameter of size bins
        rho_p = dens * 1000  #          % [kg/m^3], density of particles in
    # else:
    #     ln = len(sizepar1)
    #     x_range = sizepar1
    #     df = sizepar2
    #     Nx = len(x_range)
    #     dlogd = []
    #     for i in range(Nx - 1):
    #         dlogd.append(np.log10(x_range[i + 1] / x_range[i]))
    #     dlogd = np.array(dlogd)

    df = np.array(df)
    x_range = np.array(x_range)

    if cut:
        idx = x_range <= cut
        x_range = x_range[idx]
        df = df[idx]
        dexp = dexp[idx]
        if not np.isscalar(dlogd):
            dlogd = dlogd[idx]
            dDp = dDp[idx]

    if np.isscalar(fcoat):
        fcoat = np.ones(len(x_range)) * fcoat

    iscoated = np.max(fcoat) != 0
    y_range = x_range * (1 + fcoat)  # coating diameters

    if norm2core:
        vol_tot = (np.pi / 6) * np.sum(
            (x_range**3) * df * dlogd
        )  # [nm^3], relative total Volume
        Vol = (np.pi / 6) * np.sum(
            (Dp**3) * N * df * dlogd
        )  # [m^3 / m^3], particle volume per unit volume
        M = dens * 1e3 * Vol  # [kg/m^3], Mass of particles per unit volume
    else:
        vol_tot = (np.pi / 6) * np.sum((y_range**3) * df * dlogd)

    x_areas = (np.pi / 4) * (x_range**2)
    A = x_areas * 1e-18

    # Start of Mie calculations
    Mie_result = []
    asym = []

    for i in range(len(x_range)):
        xval = np.pi * mr * x_range[i] / wavelength
        yval = np.pi * mr * y_range[i] / wavelength

        if iscoated:
            one_result = MieCoated(mp / mr, mc / mr, xval, yval)
        else:
            one_result = Mie(mp / mr, xval)
        one_eff = list(one_result[:3])
        asym.append(one_result[4])

        # All scattering, including truncated neph
        if ~nobackscat:
            if iscoated:
                scatcalc = weighted_scattering(
                    mp / mr, xval, theta, dtheta, scatwts, m_coating=mc / mr, ycoat=yval
                )
            else:
                scatcalc = weighted_scattering(mp / mr, xval, theta, dtheta, scatwts)
            one_eff.append(scatcalc[scatidx])
        Mie_result.append(one_eff)

    Mie_result = pd.DataFrame(
        Mie_result,
        columns=["Extinction", "Scattering", "Absorption", "Weighted Scattering"],
    )

    const = x_areas * df * dlogd
    scats = Mie_result["Scattering"] * const
    asymav = np.sum(asym * scats) / np.sum(scats)

    Mie_tots = pd.Series([], dtype="float")
    for col in Mie_result.columns:
        Mie_tots[col] = np.sum(Mie_result[col] * const)

    if norm2volume:
        Mie_tots = Mie_tots / vol_tot * 1.0e3 / dens
    else:
        Mie_tots = Mie_tots * 1.0e-6

    if ~nobackscat:
        Mie_tots["Forcing Efficiency"] = Forcing_Efficiency(
            Mie_tots["Weighted Scattering"], Mie_tots["Absorption"]
        )

    Mie_tots["Extinction Coefficient"] = Mie_result["Extinction"].values
    Mie_tots["Scattering Coefficient"] = Mie_result["Scattering"].values
    Mie_tots["Absorption Coefficient"] = Mie_result["Absorption"].values
    Mie_tots["Asym"] = asym
    Mie_tots["SSA"] = Mie_tots["Scattering"] / Mie_tots["Extinction"]
    Mie_tots["g"] = asymav
    Mie_tots["coating"] = fcoat
    Mie_tots["core_dia"] = x_range
    return Mie_tots