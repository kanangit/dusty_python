##Import Libraries##
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import cv2
import os
from scipy.stats import norm
import statistics
from matplotlib.widgets import Slider
import pims
from rdfpy import rdf
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from PIL import Image, ImageDraw
from numpy import asarray
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy import constants
import scipy.ndimage as ndi



##Optionally, tweak styles.##
scl = 1236/1376 #!!!INPUT
mpl.rc('figure',  figsize=(10, 10*(scl))) #!!!INPUT
mpl.rc('image', cmap='gray')
mpl.rcParams['agg.path.chunksize'] = 10000

##______________________________________________________________________________________________________________________________________________##

##Define pathes and file types
#File path to video that needs converted
folderPath = r'C:\Users\Connor\Documents\DustyPlasmaAnalysis\ImageJAnalysis' #!!!INPUT
#Name of video
videoName = r'\103v' #!!!INPUT
#Analysis Group
group = r'\DataSet1'
#Video Type
vidType = '.avi' #!!!INPUT
#Image Type
imgType = '.PNG' #!!!INPUT
#Number of frames plus one
currentframe = 972

##______________________________________________________________________________________________________________________________________________##

###Create Directory for Plots###

plots = folderPath + r'\Plots' + group + videoName

if not os.path.exists(plots):
    os.makedirs(plots)    
    
plotsVelBin = plots + r'\VelocityBinData'

if not os.path.exists(plotsVelBin):
    os.makedirs(plotsVelBin)    
    
##______________________________________________________________________________________________________________________________________________##

###Create Directory for csvData###

if not os.path.exists(folderPath + r'\csvData'):
    os.makedirs(folderPath + r'\csvData')

csvD = folderPath + r'\csvData\ParticleData' + group

if not os.path.exists(csvD):
    os.makedirs(csvD)      

csvD2 = folderPath + r'\csvData\ParticleDataFramChoice' + group

if not os.path.exists(csvD2):
    os.makedirs(csvD2)      

###Create Directory for csv velocity data###

csvV = folderPath + r'\csvData\Velocity' + group + videoName

if not os.path.exists(csvV):
    os.makedirs(csvV) 
    
##______________________________________________________________________________________________________________________________________________##

###Define variables for velocity calculation conversion###

##Variables
#Frames per second of video
fps = 99 #!!!INPUT
#Conversion factor from pixel to mm [10*(1cm/#pixels)]   
pixeltomm = 10*(1/403) #!!!INPUT
#Step size
stepSize = 15 #!!!INPUT
##Bins for histogram
binwidth = 150 #!!!INPUT

##______________________________________________________________________________________________________________________________________________##

#Import location data found in ImageJ
colnames=['x','y','frame']
t_csv = pd.read_csv(csvD + videoName + r'.csv',usecols=[3,4,8],names=colnames,skiprows=1)
t_large = tp.link(t_csv,3,memory=1) #!!!INPUT
#Start Frame
startFram = 101 #!!!INPUT
#End Frame
endFram = 151 #!!!INPUT
#Create subset for frames being analyzed
t_small = t_large[((t_large['frame'] > (startFram-stepSize-1) ) & (t_large['frame'] < (endFram+1) ))].copy()
t_small = t_small.copy()
t_small.to_csv(csvD2 + videoName + '_Frames' + str(startFram) + '-' + str(endFram) +r'.csv')

##______________________________________________________________________________________________________________________________________________##

##Plot of particle trajectories
#Create Plot of filtered trajectories
plt.figure()
tp.plot_traj(t_small)
plt.show()
plt.savefig(plots + r'\TrajectoryPlot' + imgType)
#Zoomed Plot of filtered trajectories
plt.figure()
tp.plot_traj(t_small)
plt.xlim(200,250) #!!!INPUT
plt.ylim(250,200) #!!!INPUT
plt.show()
plt.savefig(plots + r'\ZoomTrajectoryPlot' + imgType)

##______________________________________________________________________________________________________________________________________________##

###Define function for particle difference in X and Y direction###

##Now cycle through several frames (more than two)

#function get_velocities(df, step) calculates velocites of particles
# with a given step step.
def get_velocities(df, step):
#initialize empy arrays to store data:
    arr_particle = np.array([])
    arr_x = np.array([])
    arr_y = np.array([])
    arr_vx = np.array([])
    arr_vy = np.array([])
    arr_frame = np.array([])
    # get an array containing all frame numbers in the input dataframe:
    frames_listing = np.unique(np.array(df['frame']))
    #cycle throught all those frames:
    for iFrame in range(step, len(frames_listing)):
        #get current frame:
        cur_frame = frames_listing[iFrame]
        #select a dataframe containing data ONLY for that frame:
        df_front_frame = df[(df['frame'] == cur_frame)]
        print(iFrame)
        #cycle throught all particles in the frame and find their velocities as
        # position of the particle in that frame minus position of the same particles
        # step frames ago:
        for i in range(0, len(df_front_frame)):
            #take i-th particle in a frame
            cur_p = df_front_frame['particle'].iloc[i]
            cur_x = df_front_frame['x'].iloc[i]
            cur_y = df_front_frame['y'].iloc[i]
            #find a row with the same particle in a frame step frames ago:
            prev_frame_cur_row = df[((df['frame'] == cur_frame - step) & (df['particle'] == cur_p))]
            #if that particle excisted back then, we will get exactly ONE row:
            if (len(prev_frame_cur_row) == 1):
                #if this row exists, we can take position of that particle in that, previous, frame:
                prev_x = prev_frame_cur_row['x'].iloc[0]
                prev_y = prev_frame_cur_row['y'].iloc[0]
                # so we can calculate velocities:
                cur_vx = cur_x - prev_x
                cur_vy = cur_y - prev_y
                cur_particle = df_front_frame['particle'].iloc[i]
                #and append all parameters of that particle to our data arrays
                arr_vx = np.append(arr_vx, cur_vx)
                arr_vy = np.append(arr_vy, cur_vy)
                arr_particle = np.append(arr_particle, cur_particle)
                arr_x = np.append(arr_x, cur_x)
                arr_y = np.append(arr_y, cur_y)
                arr_frame = np.append(arr_frame, cur_frame)
    #save output as a dataframe containing all the info we need:
    data = {'frame':arr_frame, 'particle':arr_particle, 'x': arr_x, 'y': arr_y, 'vx': arr_vx, 'vy':arr_vy}
    ret_df = pd.DataFrame(data)
                
    return ret_df

##______________________________________________________________________________________________________________________________________________##

##Checking for pixel walking##

#Rounding down x and y coordinates
floor_x = np.floor(t_small['x'])
floor_y = np.floor(t_small['y'])

#Non-rounded x and y coordinates
x_float = t_small['x']
y_float = t_small['y']

#Subtract float from integer
subtra_x = x_float - floor_x
subtra_y = y_float - floor_y

#Plot of subtracted values to check for pixel walking
plt.scatter(subtra_x,subtra_y)
plt.show()
plt.savefig(plots + r'\PixelWalkCheck' + imgType)


##______________________________________________________________________________________________________________________________________________##

###Plots for particle motion###

#Variables to look at particles trajectory over specific coordinate grid
xmin = 175
xmax = 200
ymin = 125
ymax = 150

#Filtering out particles in grid
t_particle =  t_large[((t_large['x'] > xmin) & (t_large['x'] < xmax ))].copy()
t_particle2 =  t_particle[((t_particle['y'] > ymin ) & (t_particle['y'] < ymax ))].copy()
t_particle3 =  t_particle2[((t_particle2['frame'] > endFram ) & (t_particle2['frame'] > startFram ))].copy()

#Converting coordinates to mm
posX = t_particle3['x']*pixeltomm
posY = t_particle3['y']*pixeltomm

#Plotting coordinate grid trajectory
fig, ax = plt.subplots()
color = ax.scatter(posX, posY, s=100,  c=t_particle3['frame'], cmap='plasma')
cbar = fig.colorbar(color)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.set_ylabel('Frame Number', rotation=270, fontweight ='bold', fontsize = 14)
ax.set_xlabel('x (mm)', fontweight ='bold', fontsize = 14)
ax.set_ylabel('y (mm)', fontweight ='bold', fontsize = 14)
plt.savefig(plots + r'\Trajectory' + '_Frames' + str(startFram) + '-' + str(endFram) + '_x' + str(xmin) + '-' + str(xmax) + '_y' + str(ymin) + '-' + str(ymax) + imgType)

#Variable to look at trajectory for single particle
particle_num = 450

#Filter out coordinates for single particle
t_particle4 =  t_large[(t_large['particle'] == particle_num)].copy()
t_particle5 =  t_particle4[((t_particle4['frame'] < endFram ) & (t_particle4['frame'] > startFram ))].copy()

#Converting coordinates to mm
posX = t_particle5['x']*pixeltomm
posY = t_particle5['y']*pixeltomm

#Plot particle movement in y-direction
fig, ax = plt.subplots()
ax.plot(t_particle5['frame'], posY)
ax.set_xlabel('Frame Number', fontweight ='bold', fontsize = 14)
ax.set_ylabel('y (mm)', fontweight ='bold', fontsize = 14)
plt.savefig(plots + r'\ParticleTrajectoryY' + '_Frames' + str(startFram) + '-' + str(endFram) + '_particle' + str(particle_num) + imgType)

#Plot particle movement in x-direction
fig, ax = plt.subplots()
ax.plot(t_particle5['frame'], posX)
ax.set_xlabel('Frame Number', fontweight ='bold', fontsize = 14)
ax.set_ylabel('x (mm)', fontweight ='bold', fontsize = 14)
plt.savefig(plots + r'\ParticleTrajectoryX' + '_Frames' + str(startFram) + '-' + str(endFram) + '_particle' + str(particle_num) + imgType)

##______________________________________________________________________________________________________________________________________________##

###Calculate pixel difference for dataframe###

##Calculation for sample dataframe
check = get_velocities(t_small, stepSize)
#Convert and save data to CSV
check.to_csv(csvV + videoName + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + r'.csv')

##Create Subplots for x-y velocity for sample dataframe##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.set_xlabel(r'v$_{x}$ $pix/s$')
ax1.set_ylabel('counts')
#Histogram x velocity
n, bins, patches = ax1.hist(check['vx'], bins=binwidth)  # output is two arrays
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.set_xlabel(r'v$_{y}$ $pix/s$')
#Histogram y velocity
n2, bins2, patches2 = ax2.hist(check['vy'], bins=binwidth)  # output is two arrays
##Show and save subplot
plt.show()
plt.savefig(plots + r'\VelHist' + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + imgType)


#Save desired velocity data to variable
pixVelX = check['vx'] #!!!INPUT
pixVelY = check['vy'] #!!!INPUT

##Conversion from pixel to mm/s
velX = pixVelX*pixeltomm*fps*(1/stepSize)
velY = pixVelY*pixeltomm*fps*(1/stepSize)

#Converting series to array and then sorting x-y velocities numerically
velxSorted = (sorted(np.array(velX)))
velySorted = (sorted(np.array(velY)))

#Calculating mean and standard deviation for x velocities
meanX = statistics.mean(velX)
stdX = statistics.stdev(velX)
#Calculating mean and standard deviation for y velocities
meanY = statistics.mean(velY)
stdY = statistics.stdev(velY)


##Create Subplots for x-y velocity in mm/s (just bar histogram)##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Histogram x velocity
n, bins, patches = ax1.hist(velX, bins=binwidth)  # output is two arrays
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
#Histogram y velocity
n2, bins2, patches2 = ax2.hist(velY, bins=binwidth)  # output is two arrays
##Show and save subplot
plt.show()
plt.savefig(folderPath + r'\Plots' + group + videoName + r'\VelocityHistogram' + imgType)

##______________________________________________________________________________________________________________________________________________##

###Plot velocity distribution with gaussian fit###

#Define gaussian function
def gauss(velocity,a,sigma,mu):
    return a*((1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(velocity-mu)**2/sigma**2))

#Calculate bins and n
fig = plt.figure()
ax1 = fig.add_subplot(121)
n, bins, patches = ax1.hist(velxSorted, bins=binwidth)  # output is two arrays
ax2 = fig.add_subplot(122)
n2, bins2, patches2 = ax2.hist(velySorted, bins=binwidth)  # output is two arrays
plt.close(fig)

#Find the center of each bin from the bin edges for x-vel
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
#Find the center of each bin from the bin edges for y-vel
bins_mean2 = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]

poptx, pcovx = curve_fit(gauss, bins_mean, n, p0 = [max(n), meanX, stdX])
popty, pcovy = curve_fit(gauss, bins_mean2, n2, p0 = [max(n2), meanY, stdY])

#Find gaussian curve for x and y velocity
gaussX = gauss(pd.Series(velxSorted),poptx[0],stdX,meanX)
gaussY = gauss(pd.Series(velySorted),popty[0],stdY,meanY)

##Create Subplots for x-y velocity (just dot histogram)##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Scatter plot of x velocity count 
ax1.scatter(bins_mean, n, s=20)
#Plot Gaussian fit for x-vel
ax1.plot(velxSorted, gaussX, 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax1.set_ylim(0,1.05*max(max(gaussX),max(gaussY)))
ax1.set_xlim(-0.5,0.5)
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.grid(True)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
#Scatter plot of y velocity count 
ax2.scatter(bins_mean2, n2, s=20)
#Plot Gaussian for y-vel
ax2.plot(velySorted, gaussY, 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax2.set_ylim(0,1.05*max(max(gaussX),max(gaussY)))
ax2.set_xlim(-0.5,0.5)
plt.savefig(folderPath + r'\Plots' + group + videoName + r'\VelocityCalcGaussian' + imgType)


##Create Subplots for x-y velocity (just dot histogram)##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
# ax1.set_xlim(-0.35,0.35)
#Find the center of each bin from the bin edges for x-vel
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
#Scatter plot of x velocity count 
ax1.scatter(bins_mean, n, s=20)
#Plot Gaussian fit for x-vel
ax1.plot(velxSorted, poptx[0]*norm.pdf(velxSorted,meanX,stdX), 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax1.set_ylim(0,1.05*max(max(n),max(n2)))
ax1.set_xlim(-0.5,0.5)
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.grid(True)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
# ax2.set_xlim(-0.35,0.35)
#Find the center of each bin from the bin edges for y-vel
bins_mean = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]
#Scatter plot of y velocity count 
ax2.scatter(bins_mean, n2, s=20)
#Plot Gaussian for y-vel
ax2.plot(velySorted, popty[0]*norm.pdf(velySorted,meanY,stdY), 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax2.set_ylim(0,1.05*max(max(n),max(n2)))
ax2.set_xlim(-0.5,0.5)
##Show and save subplot
plt.show()
plt.savefig(folderPath + r'\Plots' + group + videoName + r'\VelocityCalcGaussian' + imgType)

##______________________________________________________________________________________________________________________________________________##

###Calculate Plasma Temperature###

#Particle diameter (m)
d_par = 7.14e-6
#Particle radius (m)
r_par = d_par/2
#Density of particle
rho_mf = 1510 #kg/m^3
#Particle mass (kg) = Volume of sphere * density
mass_par = (4/3)*np.pi*(r_par**3)*(rho_mf) 


# number of points in the temperature profile
N_points = 15 #!!!INPUT
# frequency for calculating particle speeds
rate_for_vel = fps / stepSize #!!!INPUT
#number of bins in the hystograms
N_bins = 100 #!!!INPUT


#cx is a variable that represnst center of mass of the particles
#units in pixels
cx = 150

#Path for velocity csv data
csvV = folderPath + r'\csvData\Velocity' + group + videoName
#Name of csv velocity data
vel_csv_path = csvV + videoName + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + r'.csv'
#Reading csv data
df_vel = pd.read_csv(vel_csv_path, usecols=range(1,7))


def get_FWHM(arr_1dvel, N_bins):
    FWHM = False
    mean_v =  np.mean(arr_1dvel)
    arr_1dvel_noDrift = arr_1dvel - mean_v
    counts, bins = np.histogram(arr_1dvel_noDrift, bins = N_bins)
    arr_bins_centers = 0.5 * (bins[1:] + bins[:-1])
    arr_freqs = counts / len(arr_1dvel_noDrift)
    arr_bins_dense = np.linspace(np.min(arr_bins_centers), np.max(arr_bins_centers), num = N_bins * 100)
    f = interp1d(arr_bins_centers, arr_freqs, kind = 'cubic')
    arr_f_interp = f(arr_bins_dense)
    f_halfmax = np.max(arr_f_interp) / 2.0
    arr_bins_above_halfmax = arr_bins_dense[(arr_f_interp > f_halfmax)]
    left_bound = np.min(arr_bins_above_halfmax)
    right_bound = np.max(arr_bins_above_halfmax)
    FWHM = right_bound - left_bound    
    return FWHM

def get_temper_eV(FWHM, rate_betw_pos, mm_per_pix, part_mass):
    sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_SI = sigma * mm_per_pix / 1000.0 * rate_betw_pos
    temper_eV = part_mass * sigma_SI**2 / constants.elementary_charge 
    return temper_eV, sigma_SI

def get_temper_profile_FWHM(df_vel, coord_axis, vel_component, N_points):
    left_x = np.min(df_vel[coord_axis])
    right_x = np.max(df_vel[coord_axis])
    x_grid = np.linspace(left_x, right_x, num = N_points)
    delta_x = x_grid[1] - x_grid[0]
    x_grid = x_grid + delta_x/2
    x_grid = x_grid[:-1]
    arr_temper = np.array([])
    vel_bin = {}
    sigma = []
    i = 1
    for el_x in x_grid:
        sub_df = df_vel[(df_vel[coord_axis] > el_x - delta_x / 2) & (df_vel[coord_axis] < el_x + delta_x / 2)]
        vel_bin[str(i)]=[sub_df[vel_component].to_list()]
        cur_fwhm = get_FWHM(np.array(sub_df[vel_component]), N_bins)
        cur_temper, sigma_SI = get_temper_eV(cur_fwhm, rate_for_vel, pixeltomm, mass_par)
        sigma.append(sigma_SI)
        arr_temper = np.append(arr_temper, cur_temper)
        i += 1
    return x_grid, arr_temper, left_x, sigma, vel_bin

x_grid, arr_temper_x, left_x, sigma_x, vel_x = get_temper_profile_FWHM(df_vel, 'x', 'vx',  N_points)
x_grid, arr_temper_y, left_x, sigma_y, vel_y  = get_temper_profile_FWHM(df_vel, 'x', 'vy', N_points)
df_temper = pd.DataFrame({'x_px':x_grid, 'temper_x_eV':arr_temper_x, 'temper_y_eV':arr_temper_y})
df_temper.head()
temper_csv_path = folderPath + r'\csvData\Temperatures' + group + videoName
if not os.path.exists(temper_csv_path):
    os.makedirs(temper_csv_path)
    
df_temper.to_csv(temper_csv_path + videoName + r'_temps_FWHM' + r'.csv')

df_temper.head()


meanVx = np.mean(df_vel['vx'])
meanVy = np.mean(df_vel['vy'])
arr_vx_noDrift = np.array(df_vel['vx']) - meanVx
arr_vy_noDrift = np.array(df_vel['vy']) - meanVy

#calculating histogram
counts_x, bins_x = np.histogram(arr_vx_noDrift, bins = N_bins)
arr_bins_x_centers = 0.5 * (bins_x[1:] + bins_x[:-1]) # array of the bin centers
arr_vx_freqs = counts_x / len(arr_vx_noDrift) # histogram frequencies

#plotting the histogram
fig_hist_x, ax_hist_x = plt.subplots()
ax_hist_x.bar(arr_bins_x_centers, arr_vx_freqs, width = (arr_bins_x_centers[1] - arr_bins_x_centers[0]))

# ten times more dense bins:
arr_bins_dense = np.linspace(np.min(arr_bins_x_centers), np.max(arr_bins_x_centers), num = N_bins * 10)
#interpolating function made of cubic splines:
fx = interp1d(arr_bins_x_centers, arr_vx_freqs, kind = 'cubic')
#interpolate the data:
arr_f_x_interp = fx(arr_bins_dense)
#plot
ax_hist_x.plot(arr_bins_dense, arr_f_x_interp, color = 'red')
fig_hist_x

f_vx_halfmax = np.max(arr_f_x_interp) / 2.0 #calculate the halfmaximum of the interpolated data
#find all x-values corresponding to y-valies laying ABOVE the halfmaximum
arr_vx_above_halfmax = arr_bins_dense[(arr_f_x_interp > f_vx_halfmax)]
#minimum of the x-values will be the left boundary
left_bound_vx = np.min(arr_vx_above_halfmax)
# maximum is the right boundary
right_bound_vx = np.max(arr_vx_above_halfmax)
#FWHM is rgith boundary minus left boundary
FWHM_vx = right_bound_vx - left_bound_vx
FWHM_vx

#quick check, for Gaussian distribution, FWHM should coincide with
sigma_vx = np.std(df_vel['vx'])
sigma_vx * 2 * np.sqrt(2 * np.log(2))

#check if we calculated FWHM correctly by plotting right and left boundary ontop of the previous plot
ax_hist_x.plot(np.zeros(50) + left_bound_vx, np.linspace(0, 0.3, num = 50), color = 'yellow')
ax_hist_x.plot(np.zeros(50) + right_bound_vx, np.linspace(0, 0.3, num = 50), color = 'yellow')
fig_hist_x

plt.savefig(plots + r'\CalcFWHM' + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + imgType)


x_grid, arr_temper_x, left_x, sigma_SI, vel_x  = get_temper_profile_FWHM(df_vel, 'x', 'vx',  N_points)
x_grid, arr_temper_y, left_x, sigma_SI, vel_y  = get_temper_profile_FWHM(df_vel, 'x', 'vy', N_points)
df_temper = pd.DataFrame({'x_px':x_grid, 'temper_x_eV':arr_temper_x, 'temper_y_eV':arr_temper_y})
df_temper.head()
    
df_temper.to_csv(temper_csv_path + videoName + r'_temps_FWHM' + r'.csv')

df_temper.head()


def get_temper_profile_direct(df_vel, coord_axis, vel_component, N_points):
    left_x = np.min(df_vel[coord_axis])
    right_x = np.max(df_vel[coord_axis])
    x_grid = np.linspace(left_x, right_x, num = N_points)
    delta_x = x_grid[1] - x_grid[0]
    x_grid = x_grid + delta_x/2
    x_grid = x_grid[:-1]
    arr_temper_frosigmas = np.array([])
    for el_x in x_grid:
        sub_df = df_vel[(df_vel[coord_axis] > el_x - delta_x / 2) & (df_vel[coord_axis] < el_x + delta_x / 2)]
        sub_df.head()
        cur_sigma = np.std(np.array(sub_df[vel_component]))
        sigma_SI = cur_sigma * pixeltomm / 1000.0 * rate_for_vel
        temper_eV = mass_par * sigma_SI**2 / constants.elementary_charge
        arr_temper_frosigmas = np.append(arr_temper_frosigmas, temper_eV)
    return x_grid, arr_temper_frosigmas, left_x

x_grid, arr_temper_x_dir, left_x = get_temper_profile_direct(df_vel, 'x', 'vx',  N_points)
x_grid, arr_temper_y_dir, left_x = get_temper_profile_direct(df_vel, 'x', 'vy',  N_points)
df_temper = pd.DataFrame({'x_px':x_grid, 'T_x_eV':arr_temper_x_dir, 'T_y_eV':arr_temper_y_dir})
df_temper.head()

df_temper.to_csv(temper_csv_path + videoName + r'_temps_direct' + r'.csv')

df_temper.head()

delta_sig_x = sigma_x/np.sqrt(2*(N_points-1))
delta_sig_y = sigma_y/np.sqrt(2*(N_points-1))

err_x = ((mass_par/ constants.elementary_charge)*2*delta_sig_x*sigma_x)
err_y = ((mass_par/ constants.elementary_charge)*2*delta_sig_y*sigma_y)
x_grid_mm = (x_grid-np.min(x_grid)-left_x-cx)*pixeltomm

fig, ax = plt.subplots()
ax.errorbar(x_grid_mm, arr_temper_x_dir, label = '$T_x$, direct method', yerr=err_x, fmt='o', capsize=10)
ax.errorbar(x_grid_mm, arr_temper_y_dir, label = '$T_y$, direct method', yerr=err_y, fmt='o', capsize=10)
ax.set_xlabel('Distance (mm)', fontsize = 14)
ax.set_ylabel('$T$ (eV)', fontsize = 14)
ax.legend(prop={'size':14})
plt.savefig(plots + r'\TempDirect' + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + imgType)

data_1 = {
    'x' : x_grid_mm,
    'y' : arr_temper_x_dir,
    'yerr' : err_x,
    'label' : '$T_x$, direct method'
    }
data_2 = {
    'x' : x_grid_mm,
    'y' : arr_temper_y_dir,
    'yerr' : err_y,
    'label' : '$T_y$, direct method'
    }

# plot
plt.figure()
# only errorbar
plt.subplot(211)
plt.ylabel('T (ev)', fontsize = 14)
for data in [data_1, data_2]:
    plt.errorbar(**data, fmt='o', capsize=3, capthick=1)
    plt.legend(prop={'size':14})
# errorbar + fill_between
plt.subplot(212)
for data in [data_1, data_2]:
    plt.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]
        }
    plt.fill_between(**data, alpha=.25)
plt.xlabel('Distance (mm)', fontsize = 14)
plt.ylabel('T (ev)', fontsize = 14)
plt.savefig(plots + r'\TempDirect2' + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + imgType)


x_grid_mm = (x_grid-np.min(x_grid)-left_x-cx)*pixeltomm

fig, ax = plt.subplots()
ax.errorbar(x_grid_mm, arr_temper_x, label = '$T_x$, FWHM method', yerr=err_x, fmt='o', capsize=10)
ax.errorbar(x_grid_mm, arr_temper_y, label = '$T_y$, FWHM method', yerr=err_y, fmt='o', capsize=10)
ax.set_xlabel('distance (px)', fontsize = 14)
ax.set_ylabel('$T$ (eV)', fontsize = 14)
ax.legend(prop={'size':14})
plt.savefig(plots + r'\TempFWHM' + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + imgType)

data_1 = {
    'x' : x_grid_mm,
    'y' : arr_temper_x,
    'yerr' : err_x,
    'label' : '$T_x$, FWHM method'
    }
data_2 = {
    'x' : x_grid_mm,
    'y' : arr_temper_y,
    'yerr' : err_y,
    'label' : '$T_y$, FWHM method'
    }

# plot
plt.figure()
# only errorbar
plt.subplot(211)
plt.ylabel('T (ev)', fontsize = 14)
for data in [data_1, data_2]:
    plt.errorbar(**data, fmt='o', capsize=3, capthick=1)
    plt.legend(prop={'size':14})
# errorbar + fill_between
plt.subplot(212)
for data in [data_1, data_2]:
    plt.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]
        }
    plt.fill_between(**data, alpha=.25)
plt.xlabel('Distance (mm)', fontsize = 14)
plt.ylabel('T (ev)', fontsize = 14)
plt.savefig(plots + r'\TempFWHM2' + '_Frames' + str(startFram) + '-' + str(endFram) + '_SS' + str(stepSize) + imgType)


#Plotting velocity data for each bin
for i in range(1,15):
    #Save desired velocity data to variable
    pixVelX = vel_x[str(i)][0] #!!!INPUT
    pixVelY = vel_y[str(i)][0] #!!!INPUT
    
    ##Conversion from pixel to mm/s
    velX = np.asarray(pixVelX)*pixeltomm*fps*(1/stepSize)
    velY = np.asarray(pixVelY)*pixeltomm*fps*(1/stepSize)
    
    #Converting series to array and then sorting x-y velocities numerically
    velxSorted = (sorted(np.array(velX)))
    velySorted = (sorted(np.array(velY)))
    
    #Calculating mean and standard deviation for x velocities
    meanX = statistics.mean(velX)
    stdX = statistics.stdev(velX)
    #Calculating mean and standard deviation for y velocities
    meanY = statistics.mean(velY)
    stdY = statistics.stdev(velY)
    
    
    ##Create Subplots for x-y velocity (just bar histogram)##
    fig = plt.figure()
    ##First Subaxis
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'v$_{x}$ $mm/s$')
    ax1.set_ylabel('counts')
    #Histogram x velocity
    n, bins, patches = ax1.hist(velX, bins=binwidth)  # output is two arrays
    ##Second Subaxis
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel(r'v$_{y}$ $mm/s$')
    #Histogram y velocity
    n2, bins2, patches2 = ax2.hist(velY, bins=binwidth)  # output is two arrays
    plt.show()
    plt.savefig(plotsVelBin +  r'\VelocityHistogram_Bin' + str(i) + imgType)
    plt.close()


##______________________________________________________________________________________________________________________________________________##

###Save Data for Individual Frame Data###

frameD = folderPath + r'\csvData\FrameData' + group + videoName

#Create Directory for Individual Frame Data
if not os.path.exists(frameD):
    os.makedirs(frameD)

#Loop to Save Data for Individual Frames
for i in range(currentframe):
    #Pull data for specific frame# from dataframe
    framData = t_large[t_large['frame'] == i]
    #Save specific columns
    xyData = framData[['y','x','frame','particle']]
    #Save columns to csv
    xyData.to_csv(frameD + videoName + 'Frame' + str(i) + r'.csv', index=False)
    print("Frame ", i)
