import numpy as np
from scipy.constants import pi, epsilon_0

def linde(vwc, bd, sand, clay, water_ec, pdn=2.65, m=1.5, n=2):
    """        
        Parameters
        ----------
        vwc: float
            volumetric water content [%]
        
        bd: float
            bulk density [g/cm3]

        clay: float
            Soil volumetric clay content [%]

        water_ec: float
            Soil water real electrical conductivity [mS/m]

        pdn: float
            particle density [g/cm3]

        m: float
            cementation exponent [-]

        n: float
            saturation exponent [-]

        Returns
        -------
        bulk_ec: float
            Soil bulk real electrical conductivity [mS/m]
    """  

    por = 1-(bd/pdn) # porosity
    sat_w = (vwc/100)/por # water saturation
    f_form = por**(-m) # formation factor
    
    water_ec = water_ec/1000

    silt = 100 - clay - sand

    radius_clay = 0.002/2000
    radius_silt = 0.025/2000
    radius_sand = 0.75/2000

    solid_ec = 1*(10**-7) # Solid electrical conductivity
    clay_ec= 3*(solid_ec/radius_clay)  # clay electrical conductivity
    silt_ec = 3*(solid_ec/radius_silt) # Silt electrical conductivity
    sand_ec = 3*(solid_ec/radius_sand) # Sand electrical conductivity

    surf_ec = np.average([clay_ec*(clay/100), 
                          sand_ec*(sand/100), 
                          silt_ec*(silt/100)])
    bulk_ec = (((sat_w**n)*water_ec) 
               + (f_form - 1)*(surf_ec))/f_form 
    
    return bulk_ec*1000


def logsdon(fc, rperm, iperm):
    """        
        Parameters
        ----------
        fc: float
            central electromagnetic frequency [Hz]
        
        rperm: float
            Real bulk dielectric permittivity [-]

        iperm: float
            Imaginary bulk dielectric permittivity [-]

        Returns
        -------
        bulk_ec: float
            Soil bulk real electrical conductivity [mS/m]
    
    Logsdon 2010 equation 8
    """
    arc = np.arctan(2.0*iperm/rperm)
    bulk_ec = arc*rperm*pi*fc*epsilon_0
    return bulk_ec


def hp_bulk_ec(fc, rperm, iperm):
    """        
        Parameters
        ----------
        fc: float
            central electromagnetic frequency [Hz]
        
        rperm: float
            Real bulk dielectric permittivity [-]

        iperm: float
            Imaginary bulk dielectric permittivity [-]

        Returns
        -------
        bulk_ec: float
            Soil bulk real electrical conductivity [mS/m]
    
    Logsdon 2010 equation 8
    """
    bulk_ec = iperm*2*pi*fc*epsilon_0
    return bulk_ec


def Fu(water, clay, bd, pd, wc, solid_ec, dry_ec, sat_ec, s=1, w=2):
    """
    Calculate the soil bulk real electrical conductivity using the Fu model.

    This is a volumetric mixing model that takes into account various soil properties 
    such as clay content, bulk density, particle density, and water content. 
    It was exhaustively validated using several soil samples [1]. Reported R2 = 0.98

    Parameters
    ----------
    water : array_like
        Soil volumetric water content [m**3/m**3].
    clay : array_like
        Soil clay content [g/g]*100.
    bd : array_like 
        Soil bulk density (g/cm^3).
    pd : array_like
        Soil particle density (g/cm^3).
    wc : array_like
        Soil water real electrical conductivity [S/m].
    solid_ec : array_like
        Soil solid real electrical conductivity [S/m].
    dry_ec : array_like
        Soil bulk real electrical conductivity at zero water content [S/m].
    sat_ec : array_like
        Soil bulk real electrical conductivity at saturation water content [S/m].
    s : float, optional
        Phase exponent of the solid, default is 1.
    w : float, optional
        Phase exponent of the water, default is 2.

    Returns
    -------
    array_like
        The estimated bulk electrical conductivity [S/m].

    Notes
    -----
    The method uses default values for s and w, which are 1 and 2 respectively, 
    but can be modified if necessary. Three different forms of the model are used 
    depending on the soil data availability. The soil electrical conductivity of solid surfaces 
    is calculated as in [1] using the formula of Doussan and Ruy (2009) [2]

    References
    ----------
    .. [1] Yongwei Fu, Robert Horton, Tusheng Ren, J.L. Heitman,
    A general form of Archie's model for estimating bulk soil electrical conductivity,
    Journal of Hydrology, Volume 597, 2021, 126160, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2021.126160.
    .. [2] Doussan, C., and Ruy, S. (2009), 
    Prediction of unsaturated soil hydraulic conductivity with electrical conductivity, 
    Water Resour. Res., 45, W10408, doi:10.1029/2008WR007309.

    Example
    -------
    >>> Fu(0.3, 30, 1.3, 2.65, 0.3, 0, np.nan, np.nan)
    0.072626

    """
    d = 0.6539
    e = 0.0183
    por = 1 - bd/pd
    surf_ec = (d*clay/(100-clay))+e # Soil electrical conductivity of solid surfaces

    if np.isnan(dry_ec) & np.isnan(sat_ec):
        bulk_ec = solid_ec*(1-por)**s + (water**(w-1))*(por*surf_ec) + wc*water**w

    elif ~(np.isnan(dry_ec)) & ~(np.isnan(sat_ec)):
        bulk_ec = dry_ec + ((dry_ec-sat_ec)/(por**w) - surf_ec)*water**w + (water**(w-1))*(por*surf_ec)

    elif ~(np.isnan(dry_ec)) & np.isnan(sat_ec):
        sat_ec = dry_ec + (wc+surf_ec)*por**w
        bulk_ec = dry_ec + ((dry_ec-sat_ec)/(por**w) - surf_ec)*water**w + (water**(w-1))*(por*surf_ec)

    return bulk_ec


def Hilhorst(bulk_ec, bulk_perm, water_perm, offset_perm):
    """
    Calculate the soil bulk real relative dielectric permittivity using Hilhorst's model.

    This function calculates the soil bulk real relative dielectric permittivity of a 
    soil-water mixture based on Hilhorst's model. The relation 
    connects the bulk electrical conductivity of the mixture with the permittivity 
    of the water phase and an offset for the permittivity.

    Parameters
    ----------
    bulk_ec : array_like
        Soil bulk real relative electrical conductivity [S/m].
    bulk_perm : array_like
        Soil bulk real relative dielectric permittivity [-].
    water_perm : array_like
        Soil water phase real dielectric permittivity [-]. 
    offset_perm : array_like
        Soil bulk real relative dielectric permittivity when soil bulk real electrical conductivity is zero [-].

    Returns
    -------
    water_ec : array-like
        Soil water real electrical conductivity [S/m]
    
    References
    ----------
    .. [1] Hilhorst, M.A. (2000), A Pore Water Conductivity Sensor. 
    Soil Sci. Soc. Am. J., 64: 1922-1925. https://doi.org/10.2136/sssaj2000.6461922x   

    Example
    -------
    >>> Hilhorst(0.05, 0.5, 80, 4)
    12.0

    """
    water_ec = bulk_ec*water_perm/(bulk_perm - offset_perm) 

    return water_ec


def LongmireSmithEC(bulk_ec_dc, frequency_ec):
    """
    Calculate the soil bulk real electrical conductivity using the Longmire-Smith model.

    This is a semiempirical model that calculates the soil bulk real electrical conductivity at different
    electromagnetic frequencies [1].

    Parameters
    ----------
    bulk_ec_dc : array_like
        Soil bulk real direct current electrical conductivity [S/m].
    frequency_ec : array_like
        Frequency of electric conductivity measurement [Hz].

    Returns
    -------
    array_like
        Soil bulk real electrical conductivity [S/m].

    Notes
    -----
    The Longmire-Smith equation uses a set of coefficients to account for the 
    frequency-dependent dielectric dispersion. If all values in the `bulk_ec_dc` 
    array are zero, the function returns 0.

    Global Variables Used
    ---------------------
    epsilon_0 : float
        The vacuum permittivity constant.

    References
    ----------
    .. [1] K. S. Smith and C. L. Longmire, “A universal impedance for soils,” 
    Defense Nuclear Agency, Alexandria, VA, USA, Topical 
    Report for Period Jul. 1 1975-Sep. 30 1975, 1975.

    Example
    -------
    >>> LongmireSmithEC(np.array([0.05, 0.10]), 130)
    array([0.05153802, 0.10245936])

    """
    if (bulk_ec_dc == 0).all():
        return 0
    
    else: 
        a = [3.4e6, 2.74e5, 2.58e4, 3.38e3, 5.26e2, 1.33e2, 2.72e1, 1.25e1, 4.8, 2.17, 9.8e-1, 3.92e-1, 1.73e-1]
        f = (125*bulk_ec_dc)**0.8312
        bulk_eci_ = []
        
        for i in range(len(a)):
            F_ = f*(10**i)
            bulk_eci = 2*pi*epsilon_0*(a[i]*F_*(frequency_ec/F_)**2/(1+(frequency_ec/F_)**2))                     
            bulk_eci_.append(bulk_eci)

        bulk_ec = bulk_ec_dc + sum(bulk_eci_)
        return bulk_ec
    
    
def SheetsHendrickxEC(ECa, temp):
    """
    Calculate the temperature-corrected bulk real electrical conductivity of soil using the Sheets-Hendricks model.

    This function adjusts the apparent electrical conductivity (ECa) of soil to a standard temperature of 25°C. The adjustment is based on the Sheets-Hendricks model.

    Parameters
    ----------
    ECa : array_like
        Apparent electrical conductivity of soil at the measurement temperature [S/m].
    temp : array_like or float
        Temperature at which the ECa was measured [°C].

    Returns
    -------
    array_like
        Temperature-corrected electrical conductivity at 25°C [S/m].

    Notes
    -----
    The Sheets-Hendricks model applies a temperature correction factor to adjust the apparent electrical conductivity to a standard temperature of 25°C. This correction is particularly important in precision agriculture and soil science studies where temperature fluctuations can significantly affect conductivity measurements.

    Example
    -------
    >>> SheetsHendrickxEC(np.array([1.2, 2.5]), 20)
    array([0.13352103, 0.27816881])
    """
    ft = 0.447+1.4034*np.exp(-temp/26.815) # Temperature conversion factor
    EC25 = ECa*ft
    return EC25


def WraithOr(ECw, temp):
    """
    
    """
    diff = temp-25
    ECw25 = ECw*np.exp(-diff*(2.033e-2 - 1.266e-4*diff + 2.464e-6*diff**2))
    return ECw25


def RMSE(predictions, targets):
    """
    Compute the Root Mean Square Error.

    Parameters:
    - predictions: array-like, predicted values
    - targets: array-like, true values

    Returns:
    - RMSE value
    """
    differences = np.array(predictions) - np.array(targets)
    return np.sqrt(np.mean(differences**2))