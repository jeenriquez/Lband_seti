import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from matplotlib.patches import Polygon
from scipy import stats


def calc_DishArea(d):
    """ Compute dish area
    d = dish diameter
    """
    return np.pi * (d/2)**2

# def calc_Area_eff(d):
#     """ Compute dish area
#     d = dish diameter
#     G = gain
#     l = wavelenght
#
#     """
#
#     A*Aeff = G * l**2 / (4*np.pi)
#
#     return

def calc_BeamSize(d,v,verbose=False):
    """ Compute BeamSize
    d = dish diameter
    v = frequency
    """

    c = 2.998e8  #speed of light

    if verbose:
        print '\nBeam is: %f \n'%(1.22* (c/(d*v)) *57.2958*60.)

    return (1.22* (c/(d*v)) *57.2958*60./2)**2*np.pi

def calc_SEFD(A, Tsys, eff=1.0):
    """ Calculate SEFD
    Tsys = system temperature
    A = collecting area
    Ae = effective collecting area
    eff = aperture efficency (0.0 to 1.0)
    """
    kb = 1.3806488e3  # 1.38064852e-23    Boltzmann constant
    Ae = A*eff
    return 2 * Tsys * kb / Ae

def calc_Sensitivity(m, nu, t, SEFD=0, Tsys=10,eff=1.0,A=100,npol=2.,narrow=True):
    """ Minimum detectable luminosity for narrowband emission
    Tsys = system temperature
    A = collecting area
    m = threshold, (e.g. 10)
    nu = channel bandwidth
    t = observing time
    narrow = True if signal is narrower than spectral resolution.
    """
    if not SEFD:
        sefd = calc_SEFD(A, Tsys, eff=eff)
    else:
        sefd = SEFD


    if narrow:
        sens = m * sefd * np.sqrt(nu/(npol*t))
    else:
        sens = m * sefd / np.sqrt(npol*nu*t)

    return sens

def calc_EIRP_min(d,Sens):
    """ Minimum detectable luminosity (EIRP) for narrowband emission.
    d = distance to target star []
    Sens = sensitivity of the obs (Jy)
    """

    #1 Jy = 1e-26 W/m2/Hz)

    return 4 * np.pi * d**2 * Sens *1e-26

def calc_gain(ll, d):
    """ Gain of a dish telescope
    ll = wavelength (lambda)
    d = dish diameter (m)
    """
    return (np.pi * d / ll)**2

def calc_NP_law(alpha,P):
    ''' Calcualtes the power law, given a the power exponend and an array of EIRP powers in W. Based on Gray&Mooley
    '''

#    No = 1e25**alpha   # Not sure how I got 1e25 for the transmitter power, but from the plot I get a lower number.
    No = 1.e21**alpha
    NP = No*(1./P**(alpha))
    NPn = NP*1e3/4e11   # Normalized to stars in MW (4e11), and BW (1e3 .. ).

    return NPn

def calc_NP_law2(alpha,P):
    ''' Calcualtes the power law, given a the power exponend and an array of EIRP powers in W. Based on BL-Lband.
    '''

    No = 706146012574.0**alpha / 304.48
    NP = No*(1./P**(alpha))

    return NP

def calc_NP_law3(P):
    ''' Calcualtes the power law, given a the power exponend and an array of EIRP powers in Watts. Based on fit from both above.
    '''

#    E1 = 706146012574.0
    E1 = 5.16411376743e+12 #7511821434031.0742
    S1  = 304.48
    E2 = 1.98792219e+21  #2.75879335e+21
    S2 = 7.14285714e+08
#    N1 = 1.95395684e+21**alpha / 1.48809524e+07

    # Solving for alpha
    alpha = np.log10(S2/S1) /np.log10(E2/E1)
    print 'The exponent (alpha) = ', alpha
    # Solving for No
    No = E1**alpha / S1

    NP = No*(1./P**(alpha))

    return NP


#---------------------------
# Standardizing the sensitivity of telescopes by figuring out the max EIRP of a transmitter they could have found.
# For this, I need the sensitivity of the observation, given SEFD and so on.

#Standard distance (100 ly)
dist_std = (100. * u.lyr.to('m'))
#So decided not to use the distance above, it does makes sense if the original distance is shorter, but not if the distance is farther away (like in Siemion2013)


#nomalization to L_AO = 2e13 W, fraq_freq = 1/2 and N_stars = 1k
zeta_AO  =  1 #np.log10(1e3*.5)/ np.log10(2e13)
zeta_AO  =  1e3*0.5/ 1e13

#---------------------------
#BL
telescope = 'GBT'
BL_stars= 692
band = 660e6  #(1.1-1.2,1.34-1.9)
freq_range_norm = (band/1.5e9)

BL_SEFD = calc_SEFD(calc_DishArea(100), 20, eff=0.72) # 10 Jy (GBT)
m=25.; nu =3.; t=300.
Sens = calc_Sensitivity(m, nu,t,SEFD=BL_SEFD)

dist_std = (50*3.26156 * u.lyr.to('m'))  # Max distance approx
BL_EIRP = calc_EIRP_min(dist_std,Sens)
BL_rarity = BL_stars*freq_range_norm

iband = 800e6
BL_speed = BL_SEFD**2*nu/iband

BL_sky = BL_stars * calc_BeamSize(100,1.5e9)
BL_DFM = BL_sky * band / Sens**(3/2.)

print 'SEFD (BL):', BL_SEFD
print 'Sens (BL):', Sens
print 'EIRP (BL):', BL_EIRP
print 'BeamSize (BL):', calc_BeamSize(100,1.5e9)
print 'Sky Coverage (BL):', BL_sky/ 3600.
#print 'CWTFM (BL):',  zeta_AO *np.log10(BL_EIRP) / np.log10(BL_rarity)
print 'CWTFM (BL):',  zeta_AO *(BL_EIRP) / (BL_rarity)
print '~o~'

#----------
# Gray & Mooley 2017

telescope=['VLA']
GM_stars= 1e12

band =  np.array([1e6, 0.125e6])
central_freq = np.array([1.4e9,8.4e9])
freq_range_norm = (band/central_freq)

SEFD = calc_SEFD(calc_DishArea(25),35, eff=0.45) / np.sqrt(27*26.)   #Perley 2009
m=7.0; nu =[122., 15.3]; t=[20*60,5*60]
Sens = np.array([calc_Sensitivity(m, nu[0],t[0],SEFD=SEFD),calc_Sensitivity(m, nu[1],t[1],SEFD=SEFD)])

dist_std = (2.5e6 * u.lyr.to('m'))  # Max distance approx
GM_EIRP = np.array([calc_EIRP_min(dist_std,Sen) for Sen in Sens])

GM_rarity = GM_stars*freq_range_norm

GM_rarity_tot = GM_rarity.sum()
GM_EIRP_tot = GM_EIRP.max()

iband = 1e6
GM_speed = SEFD**2*nu[0]/iband

GM_sky = 8*(0.95*60)**2*np.pi   #0.95deg images
GM_DFM = GM_sky * band / Sens**(3/2.)

print 'SEFD (Gray & Mooley 2017):', SEFD
print 'Sens (Gray & Mooley 2017):', Sens
print 'EIRP (Gray & Mooley 2017):', GM_EIRP
print 'BeamSize (Gray & Mooley 2017):',
print 'Sky Coverage (Gray & Mooley 2017):', GM_sky / 3600.
#print 'CWTFM (Gray & Mooley 2017):',  zeta_AO * np.log10(GM_EIRP_tot)/ np.log10(GM_rarity_tot)
print 'CWTFM (Gray & Mooley 2017):',  zeta_AO * (GM_EIRP_tot)/ (GM_rarity_tot) #,'or', zeta_AO*stats.hmean(GM_EIRP/GM_rarity)


print '~o~'



#----------
#Phoenix
telescope = ['Arecibo','Arecibo; Parkes,Parkes,NRAO140']

Ph_stars = np.array([290,371,206,105,195]) # From Harp2016

#180MHz skip Backus2002; band from Harp2016
band = np.array([(1.75-1.2)*1e9 - 180e6,(3.0-1.75)*1e9,(1.75-1.2)*1e9, (3.0-1.75)*1e9, (3.0-1.2)*1e9])

central_freq = np.array([1.5e9,2.375e9,1.5e9,2.375e9,2.1e9])
freq_range_norm = (band/central_freq)

#Dish_D = np.array([305,225,64,64,43])  # Email from Gerry.
Dish_D = np.array([305,305,64,64,43])  # Email from Gerry.


# SEFD = np.array([calc_SEFD(calc_DishArea(305), 40, eff=0.5),
#     calc_SEFD(calc_DishArea(305), 40, eff=0.5),
#     calc_SEFD(calc_DishArea(64), 30, eff=0.7),
#     calc_SEFD(calc_DishArea(64), 30, eff=0.7),
#     calc_SEFD(calc_DishArea(43), 25, eff=0.6)])

SEFD = np.array([calc_SEFD(calc_DishArea(Dish_D[0]), 40, eff=0.7),
    calc_SEFD(calc_DishArea(Dish_D[1]), 40, eff=0.7),
    calc_SEFD(calc_DishArea(Dish_D[2]), 35, eff=0.7),
    calc_SEFD(calc_DishArea(Dish_D[3]), 35, eff=0.7),
    calc_SEFD(calc_DishArea(Dish_D[4]), 35, eff=0.7)])

#Apperture Efficiency for Arecibo : "Radio-Frequency Electronics: Circuits and Applications" By Jon B. Hagen
#43-m Tsys:  https://fits.gsfc.nasa.gov/dishfits/dishfits.9006    ---> not really....


m=1; nu =1.0; t=[276,195,276,138,552]
Sens1 = np.array([calc_Sensitivity(m,nu,t[i],SEFD=SEFD[i],narrow=False) for i in range(len(SEFD))])
Sens = np.array([16,16,100,100,100])  # From Harp2016


# Max distance approx    ; 147Ly median distance Shostalk(2000), ~700 farthest; Somewhere I got an 80 pc distance...
dist_std = (700 * u.lyr.to('m'))

Ph_EIRP = np.array([calc_EIRP_min(dist_std,Sen) for Sen in Sens])
Ph_rarity = Ph_stars*freq_range_norm
Ph_stars_tot = Ph_stars.sum()
Ph_rarity_tot = Ph_rarity.sum()
Ph_EIRP_tot = Ph_EIRP.max()

iband = 20e6
Ph_speed = SEFD.mean()**2*nu/iband    #NOTE: I should change this, and back-engineer from Gerry's values.

Ph_sky = Ph_stars * np.array([calc_BeamSize(Dish_D[i],central_freq[i]) for i in range(len(Dish_D))])
Ph_DFM = Ph_sky * band / Sens**(3/2.)

print 'SEFD (Phoenix):', SEFD
print 'Sens (Phoenix):', Sens1
#print 'Sens_Harp (Phoenix):', Sens_Harp
print 'EIRP (Phoenix):', Ph_EIRP
print 'BeamSize (Phoenix):', np.array([calc_BeamSize(Dish_D[i],central_freq[i]) for i in range(len(Dish_D))])
print 'Sky Coverage (Phoenix):', Ph_sky.sum()/ 3600.
#print 'CWTFM (Phoenix):',  zeta_AO * np.log10(Ph_EIRP_tot)/ np.log10(Ph_rarity_tot)
print 'CWTFM (Phoenix):',  zeta_AO * (Ph_EIRP)/ (Ph_rarity)

print 'CWTFM (Phoenix):',  zeta_AO * (Ph_EIRP_tot)/ (Ph_rarity_tot)

print '~o~'

#----------
#ATA
telescope = 'ATA'

ATA_stars= np.array([65,1959,2822,7459])

band = np.array([8000.e6,2040.e6,337.e6,268.e6])    # Ignoring the 73MHz which are RFI flagged.
central_freq = 5e9    # 1-9 GHz
freq_range_norm = (band/central_freq)

#Tsys = (80+120+95+137)/4. = 108
print 'old SEFD',calc_SEFD(calc_DishArea(6.1)*27, 108, eff=0.58)
SEFD = calc_SEFD(calc_DishArea(6.1), 108, eff=0.58) / np.sqrt(27*26)
SEFDs = np.array([SEFD,SEFD,SEFD,SEFD])

m=6.5; nu =0.7; t=93.
#m=9.0; nu =0.7; t=192.
#Sens = calc_Sensitivity(m, nu, t, SEFD=SEFD)
#ATA_EIRP = np.array(calc_EIRP_min(dist_std,Sen))

#dist_std = np.array([(1.1e3 * u.lyr.to('m')),(1.1e3 * u.lyr.to('m')),(300 * u.lyr.to('m')),(300 * u.lyr.to('m'))])  #Turnbull 2003 for HabCat
dist_std = np.array([(1.4e3*3.26156 * u.lyr.to('m')),(1.1e3*3.26156 * u.lyr.to('m')),(300 * u.lyr.to('m')),(500 * u.lyr.to('m'))])  #Turnbull 2003 for HabCat
Sens = np.array([calc_Sensitivity(m,nu,t,SEFD=SEF,narrow=False) for SEF in SEFDs])
ATA_EIRP = np.array([calc_EIRP_min(dist_std[i],Sens[i]) for i in range(len(Sens))])

ATA_rarity = ATA_stars*freq_range_norm
ATA_rarity_tot = ATA_rarity.sum()
ATA_stars_tot = ATA_stars.sum()
ATA_EIRP_tot = ATA_EIRP.max()

iband = 70e6
ATA_speed = SEFD**2*nu/iband

ATA_sky = ATA_stars * (3*6.*np.pi)  # beam 3'x6'  at 1.4GHz
ATA_DFM = ATA_sky * band / Sens**(3/2.)

print 'SEFD (ATA):', SEFD
print 'Sens (ATA):', Sens
print 'EIRP (ATA):', ATA_EIRP
print 'BeamSize (ATA):',
print 'Sky Coverage (ATA):', ATA_sky.sum()/ 3600.
#print 'CWTFM (ATA):',  zeta_AO * np.log10(ATA_EIRP)/ np.log10(ATA_rarity_tot)
print 'CWTFM (ATA):',  zeta_AO * (ATA_EIRP_tot)/ (ATA_rarity_tot)

print '~o~'

#----------
#Siemion 2013
telescope = 'GBT'
Siemion_stars= 86

band = 800e6 - 130e6  #(1.1-1.2,1.33-1.9)
freq_range_norm = (band/1.5e9)

SEFD= calc_SEFD(calc_DishArea(100), 20, eff=0.72) # 10 Jy (GBT)
m=25.; nu =1.; t=300.
Sens = calc_Sensitivity(m, nu,t,SEFD=SEFD)

dist_std = (1.1e3*3.26156 * u.lyr.to('m'))  # Max distance approx
Siemion_EIRP = calc_EIRP_min(dist_std,Sens)

Siemion_rarity = Siemion_stars*freq_range_norm

iband = 800e6
Siemion_speed = (SEFD/0.85)**2*nu/iband    # The 0.85 comes from Andrew, since he used 2 bit data format.

Siemion_sky = Siemion_stars * calc_BeamSize(100,1.5e9)
Siemion_DFM = Siemion_sky * band / Sens**(3/2.)

print 'SEFD (Siemion2013):', SEFD
print 'Sens (Siemion2013):', Sens
print 'EIRP (Siemion2013):', Siemion_EIRP
print 'BeamSize (Siemion2013):',calc_BeamSize(100,1.5e9)
print 'Sky Coverage (Siemion2013):', Siemion_sky/ 3600.
#print 'CWTFM (Siemion2013):',  zeta_AO * np.log10(Siemion_EIRP)/ np.log10(Siemion_rarity)
print 'CWTFM (Siemion2013):',  zeta_AO * (Siemion_EIRP)/ (Siemion_rarity)

print '~o~'

#----------
#Valdes 1986
telescope='HCRO'
Valdes_stars = np.array([53, 12])

band = np.array([256*4883, 1024*76])
freq_range_norm = (band/1.516e9)

SEFD = calc_SEFD(calc_DishArea(26), 100, eff=0.5)
m=3.0; nu =[4883., 76.]; t=3000.
Sens = np.array([calc_Sensitivity(m, nu[0],t,SEFD=SEFD,npol=1.),calc_Sensitivity(m, nu[1],t,SEFD=SEFD,npol=1.)])

dist_std = (20 * u.lyr.to('m'))  # Max distance approx
Valdes_EIRP = np.array([calc_EIRP_min(dist_std,Sen) for Sen in Sens])
Valdes_rarity = Valdes_stars*freq_range_norm

Valdes_rarity_tot = Valdes_rarity.sum()
Valdes_EIRP_tot = Valdes_EIRP.max()

iband = 256*4883
Valdes_speed = SEFD**2*nu[0]/iband

Valdes_sky = (Valdes_stars * calc_BeamSize(26,1.5e9)).sum()
Valdes_DFM = Valdes_sky * band / Sens**(3/2.)

print 'SEFD (Valdes 1986):', SEFD
print 'Sens (Valdes 1986):', Sens
print 'EIRP (Valdes 1986):', Valdes_EIRP
print 'BeamSize (Valdes 1986):',calc_BeamSize(26,1.5e9)
print 'Sky Coverage (Valdes 1986):', Valdes_sky/ 3600.
#print 'CWTFM (Valdes 1986):',  zeta_AO * np.log10(Valdes_EIRP)/ np.log10(Valdes_rarity)
print 'CWTFM (Valdes 1986):',  zeta_AO * (Valdes_EIRP_tot)/ (Valdes_rarity_tot)

print '~o~'

#----------
#Tarter 1980
telsecope = 'NRAO 91m'
Tarter_stars=201

band = 360e3*4.
freq_range_norm = (band/1.666e9)

SEFD = calc_SEFD(calc_DishArea(91), 70, eff=0.6)
m=12.0; nu =5.5; t= 45
Sens = calc_Sensitivity(m, nu,t,SEFD=SEFD)

dist_std = (25*3.26156* u.lyr.to('m'))  # Max distance approx
Tarter_EIRP = calc_EIRP_min(dist_std,Sens)
Tarter_rarity = Tarter_stars*freq_range_norm

iband = 360e3
Tarter_speed = SEFD**2*nu/iband

Tarter_sky = Tarter_stars * calc_BeamSize(91,1.666e9)
Tarter_DFM = Tarter_sky * band / Sens**(3/2.)

print 'SEFD (Tarter1980):', SEFD
print 'Sens (Tarter1980):', Sens
print 'EIRP (Tarter1980):', Tarter_EIRP
print 'BeamSize (Tarter1980):', calc_BeamSize(91,1.666e9)
print 'Sky Coverage (Tarter1980):', Tarter_sky/ 3600.
#print 'CWTFM (Tarter1980):',  zeta_AO * np.log10(Tarter_EIRP)/ np.log10(Tarter_rarity)
print 'CWTFM (Tarter1980):',  zeta_AO * (Tarter_EIRP)/ (Tarter_rarity)

print '~o~'


#----------
#Verschuur1973
telescope=['300ft Telescope', '140ft Telescope']
Verschuur_stars=np.array([3,8])

band = np.array([0.6e6,20e6])
freq_range_norm = (band/1.426e9)

SEFD = np.array([calc_SEFD(calc_DishArea(300*0.3048),110, eff=0.75),calc_SEFD(calc_DishArea(140*0.3048),48, eff=0.75)]) #**NOTE** the 0.75 for the 140' is not real
m=3.0; nu =[490.,7.2e3]; t= [4*60.,5*60.]
Sens = np.array([calc_Sensitivity(m, nu[0],t[0],SEFD=SEFD[0]),calc_Sensitivity(m, nu[1],t[1],SEFD=SEFD[1])])

dist_std = (5*3.26156 * u.lyr.to('m'))
Verschuur_EIRP = np.array([calc_EIRP_min(dist_std,Sen) for Sen in Sens])

Verschuur_rarity = Verschuur_stars*freq_range_norm

Verschuur_rarity_tot = Verschuur_rarity.sum()
Verschuur_EIRP_tot = Verschuur_EIRP.max()

iband = np.array([0.6e6, 2.5e6])  #300 ft: Two 192-channel receivers (at 130 km/s with 4.74kHz=1km/s at this freq.)
Verschuur_speed = SEFD.min()**2*nu[0]/iband[0]

Verschuur_sky = (Verschuur_stars * np.array([calc_BeamSize(300*0.3048,1.42e9),calc_BeamSize(140*0.3048,1.42e9)])).sum()*2  # The two comes from the off beam.
Verschuur_DFM = Verschuur_sky * band / Sens**(3/2.)

print 'SEFD (Verschuur1973):', SEFD
print 'Sens (Verschuur1973):', Sens
print 'EIRP (Verschuur1973):', Verschuur_EIRP
print 'BeamSize (Verschuur1973):', np.array([calc_BeamSize(300*0.3048,1.42e9),calc_BeamSize(140*0.3048,1.42e9)])
print 'Sky Coverage (Verschuur1973):', Verschuur_sky/ 3600.
#print 'CWTFM (Verschuur1973):',  zeta_AO * np.log10(Verschuur_EIRP_tot)/ np.log10(Verschuur_rarity_tot)
print 'CWTFM (Verschuur1973):',  zeta_AO * (Verschuur_EIRP_tot)/ (Verschuur_rarity_tot)

print '~o~'

#----------
#META Horowitz&Sagan

telescope=''
Horowitz_stars= 1e7

band = 1.2e6
freq_range_norm = (band/1.42e9)

SEFD = calc_SEFD(calc_DishArea(26),85, eff=0.5)    #**NOTE** the 0.5 is just a wild guess.
m=30; nu =0.05; t=20
Sens = calc_Sensitivity(m, nu,t,SEFD=SEFD,narrow=False)

dist_std = (700*3.26156 * u.lyr.to('m'))  # Max distance approx
Horowitz_EIRP = calc_EIRP_min(dist_std,Sens)
Horowitz_rarity = Horowitz_stars*freq_range_norm

iband = 400e3
Horowitz_speed = SEFD**2*nu/iband

Horowitz_sky = 41253*60**2*.68 # Horowitz_stars * calc_BeamSize(26,1.42e9)
Horowitz_DFM = Horowitz_sky * band / Sens**(3/2.)

print 'SEFD (Horowitz):', SEFD
print 'Sens (Horowitz):', Sens
print 'EIRP (Horowitz):', Horowitz_EIRP
print 'BeamSize (Horowitz):',
print 'Sky Coverage (Horowitz):', Horowitz_sky / 3600.
#print 'CWTFM (Horowitz):',  zeta_AO * np.log10(Horowitz_EIRP)/ np.log10(Horowitz_rarity)
print 'CWTFM (Horowitz):',  zeta_AO * (Horowitz_EIRP)/ (Horowitz_rarity)
print '~o~'


#---------------------------------------------------------------------------------
# plotting setup
plt.ion()

#EIRP values in watts.
P = np.array([1e10,1e12,1e14,1e16,1e18,1e20,1e23])

#---------------------------
# Luminosity function, of putative transmitters.

plt.figure(figsize=(15, 10))
alpha = 0.7
#plt.plot(np.log10(P),np.log10(calc_NP_law(alpha,P)),lw=40,color='gray',alpha=0.5)#,label=r'$\alpha$: %s'%alpha)
#plt.plot(np.log10(P),np.log10(calc_NP_law2(alpha,P)),lw=15,color='k',alpha=0.5)#,label=r'$\alpha$: %s'%alpha)
plt.plot(np.log10(P),np.log10(calc_NP_law3(P)),lw=20,color='gray',alpha=0.5)#,label=r'$\alpha$: %s'%alpha)

plt.plot([17,17],[-11,4],'--',lw=5,color='orange',alpha=0.7)#,label='Kardashev Type I')
plt.plot([13,13],[-11,4],lw=5,color='orange',alpha=0.7)#,label='AO Planetary Radar')

markersize = 20
fontsize = 20
ticksize = fontsize - 2
dot_size = markersize - 12

plt.plot([np.log10(BL_EIRP)],[np.log10(1./BL_rarity)],'h', color = 'orange',markeredgecolor='#1c608e',markersize = markersize, label='This work')

plt.plot(np.log10(GM_EIRP),np.log10(1./GM_rarity),'o',color ='#690182',markeredgecolor='w',markersize = markersize,label='Gray&Mooley (2017)')
#plt.plot(np.log10(GM_EIRP_tot),np.log10(1./GM_rarity_tot),'ow',markersize = markersize,markeredgecolor='#440154',alpha=0.7,label='Gray&Mooley (2017) All*')
#plt.plot(np.log10(GM_EIRP_tot),np.log10(1./GM_rarity_tot),'ow',markersize = markersize,markeredgecolor='#440154',label='Gray&Mooley (2017) All*')

#plt.plot(np.log10(ATA_EIRP),np.log10(1./ATA_rarity),'^',color ='#a1c625',markersize = markersize,markeredgecolor='w',label='Harp (2016)')
plt.plot(np.log10(ATA_EIRP[0:2]),np.log10(1./ATA_rarity[0:2]),'^',color ='#a1c625',markeredgecolor='w',markersize = markersize,label='Harp (2016) a,b')
plt.plot(np.log10(ATA_EIRP[2]),np.log10(1./ATA_rarity[2]),'s',color ='#a1c625',markeredgecolor='w',markersize = markersize,label='Harp (2016) c')
plt.plot(np.log10(ATA_EIRP[3]),np.log10(1./ATA_rarity[3]),'h',color ='#a1c625',markeredgecolor='w',markersize = markersize,label='Harp (2016) d')
plt.plot(np.log10(ATA_EIRP[0]),np.log10(1./ATA_rarity_tot),'^w',markersize = markersize,markeredgecolor='#a1c625',alpha=0.7,label='Harp (2016) All*')
#plt.plot(np.log10(ATA_EIRP[0]),np.log10(1./ATA_rarity_tot),'^w',markersize = markersize,markeredgecolor='#a1c625',label='Harp (2016) All*')
plt.plot(np.log10(ATA_EIRP[0:1]),np.log10(1./ATA_rarity[0:1]),'ow',markersize = dot_size, markeredgecolor='#a1c625')

plt.plot([np.log10(Siemion_EIRP)],[np.log10(1./Siemion_rarity)],'>',color ='#26828e',markeredgecolor='w',markersize = markersize,label='Siemion (2013)')
plt.plot([np.log10(Siemion_EIRP)],[np.log10(1./Siemion_rarity)],'ow',markersize = dot_size,markeredgecolor='#440154')
#plt.scatter(np.log10(Siemion_EIRP),np.log10(1./Siemion_rarity),marker='>',s=markersize+300,hatch='/',facecolor ='#440154',label='Siemion (2013)')

plt.plot(np.log10(Ph_EIRP),np.log10(1./Ph_rarity),'<b',color ='#31688e',markeredgecolor='w',markersize = markersize,label='Phoenix')
plt.plot([np.log10(Ph_EIRP_tot)],[np.log10(1./Ph_rarity_tot)],'<w',markeredgecolor='#31688e',markersize = markersize,label='Phoenix All*')

plt.plot(np.log10(Horowitz_EIRP),np.log10(1./Horowitz_rarity),'oc',color ='#35b779',markeredgecolor='w',markersize = markersize,label='Horowitz&Sagan (1993)')
plt.plot(np.log10(Valdes_EIRP),np.log10(1./Valdes_rarity),'sy',color ='#440154',markeredgecolor='w',markersize = markersize,label='Valdes (1986)')
#plt.plot(np.log10(Valdes_EIRP_tot),np.log10(1./Valdes_rarity_tot),'sw',markersize = markersize,alpha=0.7,markeredgecolor='#1f9e89',label='Valdes (1986) All*')
#plt.plot(np.log10(Valdes_EIRP_tot),np.log10(1./Valdes_rarity_tot),'sw',markersize = markersize,markeredgecolor='#1f9e89',label='Valdes (1986) All*')

plt.plot([np.log10(Tarter_EIRP)],[np.log10(1./Tarter_rarity)],'vc',color ='#1f9e89',markeredgecolor='w',markersize = markersize,label='Tarter (1980)')
plt.plot(np.log10(Verschuur_EIRP),np.log10(1./Verschuur_rarity),'sm',color ='#efda21',markeredgecolor='w',markersize = markersize,label='Verschuur (1973)')
#plt.plot(np.log10(Verschuur_EIRP_tot),np.log10(1./Verschuur_rarity_tot),'sw',markersize = markersize,alpha=0.7,markeredgecolor='#efda21',label='Verschuur (1973) All*')
#plt.plot(np.log10(Verschuur_EIRP_tot),np.log10(1./Verschuur_rarity_tot),'sw',markersize = markersize,markeredgecolor='#efda21',label='Verschuur (1973) All*')


plt.xlabel('EIRP [log(W)]',fontsize = fontsize)
#plt.ylabel('Transmiter Galactic Rarity [log((Nstars*BW)^-1)]',fontsize=fontsize)
plt.ylabel('Transmitter Rate \n [log(1/(Nstars * rel_BW))]',fontsize=fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.ylim(-10,4)
plt.xlim(10,23)
plt.legend(numpoints=1,scatterpoints=1,fancybox=True, shadow=True)

plt.savefig('Transmitter_Rarity_FoM.eps', format='eps', dpi=300,bbox_inches='tight')

#---------------------------
# Survey speed.

markersize = 20
fontsize = 18
ticksize = fontsize - 2
#fig = plt.subplots(2, sharex=True,figsize=(15, 10))
#plt.subplot(2,1,1)
#cax = fig[0].add_axes([0.2, 0.6, 0.7, 0.3])   #l,b,w,h

fig  = plt.figure(figsize=(10, 10))
axSpeed = plt.axes([0.15, 0.6, 0.75, 0.3])

plt.plot([2017],[np.log10(BL_speed/BL_speed)],'h', color = 'orange',markeredgecolor='#1c608e',markersize = markersize,label='This work')
plt.plot([2017],[np.log10(BL_speed/GM_speed)],'o',color ='#690182',markeredgecolor='w',markersize = markersize,label='Gray&Mooley (2017)')
plt.plot([2016],[np.log10(BL_speed/ATA_speed)],'^',color ='#a1c625',markeredgecolor='w',markersize = markersize,label='ATA')
plt.plot([2013],[np.log10(BL_speed/Siemion_speed)],'>',color ='#26828e',markeredgecolor='w',markersize = markersize,label='Siemion (2013)')
plt.plot([1998],[np.log10(BL_speed/Ph_speed)],'<b',color ='#31688e',markeredgecolor='w',markersize = markersize,label='Phoenix')
plt.plot([1993],[np.log10(BL_speed/Horowitz_speed)],'oc',color ='#35b779',markeredgecolor='w',markersize = markersize,label='Horowitz&Sagan (1993)')
plt.plot([1986],[np.log10(BL_speed/Valdes_speed)],'sy',color ='#440154',markeredgecolor='w',markersize = markersize,label='Valdes (1986)')
plt.plot([1980],[np.log10(BL_speed/Tarter_speed)],'vc',color ='#1f9e89',markeredgecolor='w',markersize = markersize,label='Tarter (1980)')
plt.plot([1973],[np.log10(BL_speed/Verschuur_speed)],'sm',color ='#efda21',markeredgecolor='w',markersize = markersize,label='Verschuur (1973)')

#plt.plot([2018],[np.log10(5e9/800e6)],'+k',markersize = markersize,label='BL - GBT ')
plt.plot([2018],[np.log10(5e9/800e6)],'hw',markeredgecolor='k',markersize = markersize,label='BL - GBT ')

plt.plot([1970,2020],[0,0],'k--')

plt.xlabel('Year',fontsize=fontsize)
plt.ylabel('Relative \n Survey Speed \n[log]',fontsize=fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
#plt.ylim(-8,2)
plt.xlim(1970,2020)

#---------------------------
# Drake figure of merit.
#DFM  = Total Frequency coverage * Sky Coverage / Sensitivity^{3/2}

#plt.figure()
#plt.subplot(2,1,2)
axDFM = plt.axes([0.15, 0.2, 0.75, 0.3])

plt.plot([2017],[np.log10(BL_DFM/BL_DFM)],'h', color = 'orange',markeredgecolor='#1c608e',markersize = markersize,label='This work')
#plt.plot([2017,2017],np.log10(GM_DFM/BL_DFM),'o',color ='#690182',markeredgecolor='w',markersize = markersize,label='Gray&Mooley (2017)')
plt.plot([2017],np.log10(GM_DFM[0]/BL_DFM),'o',color ='#690182',markeredgecolor='w',markersize = markersize,label='Gray&Mooley (2017)')

#plt.plot([2016,2016,2016,2016],np.log10(ATA_DFM/BL_DFM),'^',color ='#a1c625',markeredgecolor='w',markersize = markersize,label='Harp (2016)')
#plt.plot([2016], [np.log10(ATA_DFM.sum()/BL_DFM)],'^w',markeredgecolor='#a1c625',markersize = markersize,label='Harp (2016) All*')
plt.plot([2016], [np.log10(ATA_DFM.sum()/BL_DFM)],'^',color='#a1c625',markeredgecolor='w',markersize = markersize,label='Harp (2016) All*')

plt.plot([2013],[np.log10(Siemion_DFM/BL_DFM)],'>',color ='#26828e',markeredgecolor='w',markersize = markersize,label='Siemion (2013)')
#plt.plot([1998,1998,1998,1998,1998],np.log10(Ph_DFM/BL_DFM),'<b',color ='#31688e',markeredgecolor='w',markersize = markersize,label='Phoenix')
#plt.plot([1998],np.log10(Ph_DFM.sum()/BL_DFM),'<w',markeredgecolor='#31688e',markersize = markersize,label='Phoenix All*')
plt.plot([1998],np.log10(Ph_DFM.sum()/BL_DFM),'<',color='#31688e',markeredgecolor='w',markersize = markersize,label='Phoenix All*')
plt.plot([1993],[np.log10(Horowitz_DFM/BL_DFM)],'oc',color ='#35b779',markeredgecolor='w',markersize = markersize,label='Horowitz&Sagan (1993)')

#plt.plot([1986,1986],np.log10(Valdes_DFM/BL_DFM),'sy',color ='#440154',markeredgecolor='w',markersize = markersize,label='Valdes (1986) ')
plt.plot([1986],np.log10(Valdes_DFM[0]/BL_DFM),'sy',color ='#440154',markeredgecolor='w',markersize = markersize,label='Valdes (1986) ')
plt.plot([1980],[np.log10(Tarter_DFM/BL_DFM)],'vc',color ='#1f9e89',markeredgecolor='w',markersize = markersize,label='Tarter (1980)')
#plt.plot([1973,1973],np.log10(Verschuur_DFM/BL_DFM),'sm',color ='#efda21',markeredgecolor='w',markersize = markersize,label='Verschuur (1973)')
plt.plot([1973],np.log10(Verschuur_DFM[0]/BL_DFM),'sm',color ='#efda21',markeredgecolor='w',markersize = markersize,label='Verschuur (1973)')

# plt.plot([2017],[np.log10(BL_DFM/BL_DFM)],'^', color = 'orange',markeredgecolor='#1c608e',markersize = markersize,label='This work')
# plt.plot([2017,2017],np.log10(GM_DFM/BL_DFM),'s',color ='#690182',markeredgecolor='w',markersize = markersize,label='Gray&Mooley (2017)')
# plt.plot([2016,2016,2016,2016],np.log10(ATA_DFM/BL_DFM),'^',color ='#a1c625',markeredgecolor='w',markersize = markersize,label='ATA')
# plt.plot([2016], [np.log10(ATA_DFM.sum()/BL_DFM)],'^w',markeredgecolor='#a1c625',markersize = markersize,label='ATA All*')
#
# plt.plot([2013],[np.log10(Siemion_DFM/BL_DFM)],'^',color ='#26828e',markeredgecolor='w',markersize = markersize,label='Siemion (2013)')
# plt.plot([1998,1998,1998,1998,1998],np.log10(Ph_DFM/BL_DFM),'^',color ='#31688e',markeredgecolor='w',markersize = markersize,label='Phoenix')
# plt.plot([1998],np.log10(Ph_DFM.sum()/BL_DFM),'^w',markeredgecolor='#31688e',markersize = markersize,label='Phoenix All*')
# plt.plot([1993],[np.log10(Horowitz_DFM/BL_DFM)],'^',color ='#35b779',markeredgecolor='w',markersize = markersize,label='Horowitz&Sagan (1993)')
#
# plt.plot([1986,1986],np.log10(Valdes_DFM/BL_DFM),'sy',color ='#440154',markeredgecolor='w',markersize = markersize,label='Valdes (1986) ')
# plt.plot([1980],[np.log10(Tarter_DFM/BL_DFM)],'^c',color ='#1f9e89',markeredgecolor='w',markersize = markersize,label='Tarter (1980)')
# plt.plot([1973,1973],np.log10(Verschuur_DFM/BL_DFM),'om',color ='#efda21',markeredgecolor='w',markersize = markersize,label='Verschuur (1973)')

#plt.plot([2018],[np.log10(5e9/800e6)],'*k',markersize = markersize,label='BL - GBT ')
#plt.plot([2018],[100],'hw',markeredgecolor='k',markersize = markersize,label='BL - GBT ')


plt.plot([1970,2020],[0,0],'k--')

#plt.xlabel('Year',fontsize=fontsize)
plt.ylabel('Relative \n Drake Figure of Merit \n[log]',fontsize=fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)

axes = plt.gca()
plt.xlim(1970,2020)
axes.set_ylim([-7,1])
axes.xaxis.set_tick_params(labelsize=ticksize)
axes.xaxis.tick_top()

# ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(20)

plt.legend(numpoints=1,loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=3)

plt.savefig('Other_FoM.eps', format='eps', dpi=300,bbox_inches='tight')

#---------------------------
# Drake figure of merit.
#DFM  = Total Frequency coverage * Sky Coverage / Sensitivity^{3/2}

# markersize = 10
#
# plt.figure()
#
# plt.plot([0],[np.log10(BL_DFM/BL_DFM)],'ob',markersize = markersize,label='BL L-band')
# plt.plot([1,1,1,1],np.log10(ATA_DFM/BL_DFM),'+r',markersize = markersize,label='ATA')
# plt.plot([1], [np.log10(ATA_DFM.sum()/BL_DFM)],'+k',markersize = markersize,label='ATA Total')
#
# plt.plot([2],[np.log10(Siemion_DFM/BL_DFM)],'*g',markersize = markersize,label='Siemion (2013)')
# plt.plot([3,3,3,3,3],np.log10(Ph_DFM/BL_DFM),'or',markersize = markersize,label='Phoenix')
# plt.plot([3],np.log10(Ph_DFM.sum()/BL_DFM),'ok',markersize = markersize,label='Phoenix Total')
#
# plt.plot([4,4],np.log10(Valdes_DFM/BL_DFM),'<y',markersize = markersize,label='Valdes (1986) ')
# plt.plot([5],[np.log10(Tarter_DFM/BL_DFM)],'<c',markersize = markersize,label='Tarter (1980)')
# plt.plot([6,6],np.log10(Verschuur_DFM/BL_DFM),'>m',markersize = markersize,label='Verschuur (1973)')
# plt.plot([7],[np.log10(Horowitz_DFM/BL_DFM)],'dc',markersize = markersize,label='Horowitz&Sagan (1993)')
# plt.plot([8,8],np.log10(GM_DFM/BL_DFM),'dy',markersize = markersize,label='Gray&Mooley (2017)')
#
# #plt.plot([9],[np.log10(660e6/5e9)],'vk',label='BL - current GBT')
#
# plt.plot([0,8],[0,0],'k--')
#
# plt.xlabel('Meaningless value, add your favorite ')
# plt.ylabel('Drake Figure of Merit')
# plt.legend(loc=3)
#
#

#---------------------------
# Luminosity function, of putative transmitters.
#
# plt.figure()
# alpha = 0.5
# plt.plot(np.log10(P),np.log10(calc_NP_law(alpha,P)),label=r'$\alpha$: %s'%alpha)
# alpha = 1.0
# plt.plot(np.log10(P),np.log10(calc_NP_law(alpha,P)),label=r'$\alpha$: %s'%alpha)
# alpha = 0.65
# plt.plot(np.log10(P),np.log10(calc_NP_law(alpha,P)),label=r'$\alpha$: %s'%alpha)
#
#
# plt.plot([np.log10(BL_EIRP)],[np.log10(1./BL_stars)],'ob',label='BL L-band')
# plt.plot(np.log10(ATA_EIRP),np.log10(1./ATA_stars),'+r',label='ATA')
# plt.plot([np.log10(ATA_EIRP[0])],[np.log10(1./ATA_stars_tot)],'+k',label='ATA All*')
# plt.plot([np.log10(Siemion_EIRP)],[np.log10(1./Siemion_stars)],'*g',label='Siemion (2013)')
# plt.plot(np.log10(Ph_EIRP),np.log10(1./Ph_stars),'or',label='Phoenix')
# plt.plot([np.log10(Ph_EIRP_tot)],[np.log10(1./Ph_stars_tot)],'ok',label='Phoenix All*')
# plt.plot(np.log10(Valdes_EIRP),np.log10(1./Valdes_stars),'<y',label='Valdes (1986)')
# plt.plot([np.log10(Tarter_EIRP)],[np.log10(1./Tarter_stars)],'<c',label='Tarter (1980)')
# plt.plot(np.log10(Verschuur_EIRP),np.log10(1./Verschuur_stars),'>m',label='Verschuur (1973)')
#
# plt.plot(np.log10(BL_EIRP),np.log10(1./1e5),'vk',label='BL ++')
#
#
# plt.xlabel('EIRP [log(W)]')
# plt.ylabel('Transmiter Galactic Rarity [log(1/stars)]')
# plt.legend()
#




