"""
Created on Fri Feb 3 21:58:41 2023 EST
Last Updated on Sat 25 13:31 2023 EST

@author: Connor Belt
         connorbelt125@gmail.com
         (219)798-5999

Trackpy software used to track dusty plasma particles.
The data created by TrackPy is saved to a CSV.
Data is used to find velocity distribution function.

If line ends with "#!!!INPUT" that means there is a variable that may
need changed by user to run code. Line will be marked with purple checkmark

I have broke up the code into important sections, it is best to run code
section by section. Section breaks are given by: ##______________##

For some plots to save properly, graphics mode must be in auto.
Code will still run if in inline mode but not all plots will save.
To turn on auto mode copy this in console:
    %matplotlib auto
To turn on inline mode copy this in console:
    %matplotlib inline
"""


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


##Optionally, tweak styles.##
scl = 1236/1376 #!!!INPUT
mpl.rc('figure',  figsize=(10, 10*(scl))) #!!!INPUT
mpl.rc('image', cmap='gray')
mpl.rcParams['agg.path.chunksize'] = 10000

##______________________________________________________________________________________________________________________________________________##

###Convert Video to Image Sequence and save to directory###

##Define pathes amd file types
#File path to video that needs converted
folderPath = r'C:\Users\Connor\Documents\TrackPy' #!!!INPUT
#Name of video
videoName = r'\2mA_420V_12sccm_111mtorr_135volt_bias' #!!!INPUT
#Video Type
vidType = '.avi' #!!!INPUT
#Image Type
imgType = '.PNG' #!!!INPUT

##Create new directory for image sequence
#Read the video from specified path
cam = cv2.VideoCapture(folderPath + r'\Video' + videoName + vidType)
#Attempt to create folder
try:  
    # creating a folder named 
    direcPath = folderPath + r'\ImageSequences' + videoName
    if not os.path.exists(direcPath):
        os.makedirs(direcPath)
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')

##Begin creating image sequence
#Starting number for frame counter
currentframe = 0
#While statement to save frame as image
while(True):
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        #if video is still left continue creating images
        name = direcPath + r'\frame' + str(currentframe) + imgType
        print ('Creating...' + name)
        #writing the extracted images
        cv2.imwrite(name, frame)
        #increasing counter so that it will show frames created
        currentframe += 1
    else:
        #ends the loop
        break 

#Release all space and windows once done
cam.release()
cv2.destroyAllWindows()

##______________________________________________________________________________________________________________________________________________##

###Create Directory for Plots###

if not os.path.exists(folderPath + r'\Plots' + videoName):
    os.makedirs(folderPath + r'\Plots' + videoName)    

##______________________________________________________________________________________________________________________________________________##

###Import image sequence and convert to grayscale###

##Define function that converts to grayscale
@pims.pipeline
def gray(image):
    #Take just the green channel
    return image[:, :, 1]  

##Import and crop images to scale
#Creates an array for grayscale frames
frames = []
#Access first image in directory as test image
testim = gray(cv2.imread(direcPath + r'\frame' + str(0) + imgType))
print(" ")
print("View of current image before cropping provided in plots tab")
print(" ")
plt.imshow(testim)
plt.show()
#Pauses and closes plot for auto grapic mode (not needed for inline mode)
#plt.pause(pause time)
plt.pause(5)
plt.close()
print("Recommended starting scale is x=525:900 y=375:775")
print("(Press enter when prompted for input to use recommended scale)")
#Loop to test cropping scale and then apply to entire directory
while True:
    #Input for cropping scale
    xFrames1 = int(input("Input min x pixel coordinate: ") or "550")
    xFrames2 = int(input("Input max x pixel coordinate: ") or "925")
    yFrames1 = int(input("Input min y pixel coordinate: ") or "450")
    yFrames2 = int(((xFrames2-xFrames1)*(scl))+yFrames1)
    #Crops test images
    exFrame = testim[yFrames1:yFrames2, xFrames1:xFrames2]
    #Shows cropped image
    plt.imshow(exFrame)
    plt.show()
    #Pauses and closes plot for auto grapic mode (not needed for inline mode)
    plt.pause(5)
    plt.close()
    #Request to proceed
    answer = input("Would you like to reset scale? (yes or no): ")
    #If/Else statement to restart or break loop depending on input
    if answer=="yes":
        #Restarts loop for new crop scale
        continue 
    else:
        #Loop to apply cropping to all images in directory
        for i in range(currentframe):
            #Access image in directory
            im = gray(cv2.imread(direcPath + r'\frame' + str(i) + imgType))
            #Crops image and saves to array
            frames.append(im[yFrames1:yFrames2, xFrames1:xFrames2])
        
        #ends the loop
        break

#Show first frame in cropped image array
plt.imshow(frames[0])

##______________________________________________________________________________________________________________________________________________##

###Locate Features for single frame###
    
print(" ")
print("Recommended starting values for tp.locate(): PixSize=3, minmass=15, seperation=3, threshhold=0, invert=False")
print("(Press enter when prompted for input to use recommended values)")
#Loop to test locate feature variables and then apply to entire directory
while True:
    ##Locating pixels
    #Asking for input from user for locate feature, if no input use recommended settings
    fra = int(input("Frame number: ") or "0")
    pix = int(input("Feature size in pixels (odd#>2): ") or "3")
    minM = int(input("Minmass: ") or "10")
    sep = int(input("Seperation: ") or "5")
    thresh = int(input("Threshold: ") or "0")
    inv = input("Invert: ") or "False"
    #Locates particles using defined settings
    #tp.locate(frame,estimated feature size in pixels, minmass removes features below value, invert frame color)
    fSing = tp.locate(frames[fra], pix, minmass=minM, separation=sep, threshold=thresh, invert=inv)
    #prints data head
    print("Head of DataFrame: ")
    print(fSing.head(5))
    
    ##Check the subpixel accuracy
    #This checks that decimal part of the x and or position are evenly distributed
    #If mask size is too small, histogram shows dip in middle
    tp.subpx_bias(fSing)
    plt.show()
    #Pauses and closes plot for auto grapic mode (not needed for inline mode)
    plt.pause(1)
    plt.close()
    ##Create plot and circles features
    fig = plt.figure()
    tp.annotate(fSing, frames[fra],plot_style={'markersize':7}) 
    plt.show()
    plt.pause(30)
    plt.close()
    #Save fig (only works in inline mode due to trackpy)
    fig.savefig(folderPath + r'\Plots' + videoName + r'\Annotation' + imgType)

    ##Request to proceed
    answer2 = input("Would you like to change values? (yes or no): ") or "yes"
    if answer2 == "yes":
        #Restarts loop for new values
        continue 
    else:
        #ends the loop
        break
    
##______________________________________________________________________________________________________________________________________________##

###Link features into Particle Trajectories###

##Track particle from frame to frame
#Batch of frames
fTraj = tp.batch(frames[:], pix, minmass=minM, separation=sep, threshold=thresh, invert=inv)
# Turn off progress reports for best performance
tp.quiet()
#Convert and save data to CSV
fTraj.to_csv(folderPath + r'\csvData' + videoName + r'.csv')

##Link particle across frames
#tp.link(batch frames, max distance in pixels particle can travel betwen frame, 
#        number of frames memory maintains ID for disappeared particles)
t = tp.link(fTraj,3,memory=1) #!!!INPUT
#Head of new DataSet
t.head()

##Filter spurious trajectories
#tp.filter(batch DataSet, Keeps trajectories that last longer than this number of frames)
t1 = tp.filter_stubs(t, 5) #!!!INPUT
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

##Plot of particle trajectories
#Create Plot of filtered trajectories
plt.figure()
tp.plot_traj(t1)
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\TrajectoryPlot' + imgType)
#Zoomed Plot of filtered trajectories
plt.figure()
tp.plot_traj(t1)
plt.xlim(200,250) #!!!INPUT
plt.ylim(250,200) #!!!INPUT
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\ZoomTrajectoryPlot' + imgType)

##Plot average appearance throughout trajectory
#Create plot size vs mass
plt.figure()
tp.mass_size(t.groupby('particle').mean())
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\SizeMassPlot' + imgType)

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

###Define variables for velocity calculation conversion###

##Variables
#Frames per second of video
fps = 99 #!!!INPUT
#Conversion factor from pixel to mm [10*(1cm/#pixels)]   
pixeltomm = 10*(1/403) #!!!INPUT
#Step size
stepSize = 10 #!!!INPUT
##Bins for histogram
binwidth = 150 #!!!INPUT

##Sample dataframe##
##Selecting a subset of frames (for example, from 0 to 40, helps to decrease run time for test)
#Start Frame
startFram = 0 #!!!INPUT
#End Frame
endFram = 40 #!!!INPUT
#Create subset
t_small = t[((t['frame'] > startFram ) & (t['frame'] < endFram ))].copy()
t_small = t_small.copy()

##______________________________________________________________________________________________________________________________________________##

###Calculate pixel difference for dataframe###

##Calculation for sample dataframe
check = get_velocities(t_small, stepSize)

##Create Subplots for x-y velocity for sample dataframe##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Histogram x velocity
n, bins, patches = ax1.hist(check['vx'], bins=binwidth)  # output is two arrays
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
#Histogram y velocity
n2, bins2, patches2 = ax2.hist(check['vy'], bins=binwidth)  # output is two arrays
##Show and save subplot
plt.show()

##Calculation for full dataframe
check_all = get_velocities(t, stepSize)

##Create Subplots for x-y velocity for full dataframe##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Histogram x velocity
n, bins, patches = ax1.hist(check_all['vx'], bins=binwidth)  # output is two arrays
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
#Histogram y velocity
n2, bins2, patches2 = ax2.hist(check_all['vy'], bins=binwidth)  # output is two arrays
##Show and save subplot
plt.show()

##______________________________________________________________________________________________________________________________________________##

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
stdX = statistics.stdev(velY)
#Calculating mean and standard deviation for y velocities
meanY = statistics.mean(velX)
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
##Show and save subplot
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\VelocityHistogram' + imgType)

##Create Subplots for x-y velocity (just dot histogram)##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Find the center of each bin from the bin edges for x-vel
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
#Scatter plot of x velocity count 
ax1.scatter(bins_mean, n, s=20)
#Plot Gaussian fit for x-vel
ax1.plot(velxSorted, 3.5*len(velX)*norm.pdf(velxSorted,meanX,stdX), 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax1.set_ylim(0,1.05*max(max(n),max(n2)))
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.grid(True)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
#Find the center of each bin from the bin edges for y-vel
bins_mean = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]
#Scatter plot of y velocity count 
ax2.scatter(bins_mean, n2, s=20)
#Plot Gaussian for y-vel
ax2.plot(velySorted, 3.5*len(velY)*norm.pdf(velySorted,meanY,stdY), 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax2.set_ylim(0,1.05*max(max(n),max(n2)))
##Show and save subplot
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\VelocityGaussian' + imgType)

##______________________________________________________________________________________________________________________________________________##

###Gaussian Plot###

#Define gaussian function
def gauss(sigma,mu,velocity):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*(velocity-mu)**2/sigma**2)

#Find gaussian curve for x and y velocity
gaussX = gauss(stdX,meanX,velX)
gaussY = gauss(stdY,meanY,velY)

##Create Subplots for x-y velocity (just dot histogram)##
fig = plt.figure()
##First Subaxis
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Find the center of each bin from the bin edges for x-vel
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
#Scatter plot of x velocity count 
ax1.scatter(bins_mean, n, s=20)
#Plot Gaussian fit for x-vel
ax1.plot(velX, 3.5*len(velX)*gaussX, 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax1.set_ylim(0,1.05*max(max(n),max(n2)))
##Second Subaxis
ax2 = fig.add_subplot(122)
ax2.grid(True)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
#Find the center of each bin from the bin edges for y-vel
bins_mean = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]
#Scatter plot of y velocity count 
ax2.scatter(bins_mean, n2, s=20)
#Plot Gaussian for y-vel
ax2.plot(velY, 3.5*len(velY)*gaussY, 'r',linewidth=1)
#Set y-lim for max of both x and y velocity
ax2.set_ylim(0,1.05*max(max(n),max(n2)))
##Show and save subplot
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\VelocityCalcGaussian' + imgType)

##______________________________________________________________________________________________________________________________________________##

###Calculate Radial Distribution###

#Create 2d array with x and y data points
coords = np.stack((velX,velY),axis=1)
#Use RDF library to find radius distribution variables
g_r, radii = rdf(coords, dr=0.25)

##Create radial distribution plot
plt.figure()
plt.plot(radii,g_r)
plt.xlabel('r')
plt.ylabel('g(r)')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\RadialDistribution' + imgType)

##Create radial distribution plot with smoothing
plt.figure()
X_Y_Spline = make_interp_spline(radii,g_r)
X_ = np.linspace(radii.min(), radii.max(), 200)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_)
plt.xlabel('r')
plt.ylabel('g(r)')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\RadDistWSmoothing' + imgType)

##______________________________________________________________________________________________________________________________________________##

###Calculate Plasma Temperature###

#Particle diameter (m)
d_par = 7.14e-6
#Particle radius (m)
r_par = d_par/2
#Density of particle
rho_mf = 1500 #kg/m^3
#Particle mass (kg) = Volume of sphere * density
mass_par = (4/3)*np.pi*(r_par**3)*(rho_mf) 
#FWHM (mm)
width1 = 2*np.sqrt(2*np.log(2))*stdX 
width2 = 2*np.sqrt(2*np.log(2))*stdY
#Kinetic energy
energyx = 0.5*mass_par*(width1*1e-3)**2
energyy = 0.5*mass_par*(width2*1e-3)**2
#Convert to ev
Tdx = energyx/(1.6e-19)
Tdy = energyy/(1.6e-19)
#Temperature (ev)
Temp = np.sqrt(Tdx**2+Tdy**2)
print(Temp)

##______________________________________________________________________________________________________________________________________________##

###Create csv for different step sizes when caculating velocity###

#Create directory for step size data
if not os.path.exists(folderPath + r'\StepSizeData' + videoName):
    os.makedirs(folderPath + r'\StepSizeData' + videoName)

for i in np.arange(1,36,1):
    #Calculate velocity for range of step values
    check_all = get_velocities(t, i)
    #Save data to csv
    check_all.to_csv(folderPath + r'\StepSizeData' + videoName + r'\StepSize' + str(i) + r'.csv')

##______________________________________________________________________________________________________________________________________________##

###Import and analyze velcoity data for different step sizes###

#Create array of step values
step = np.arange(26,36,1)

#Create dictionary for step data
Stepdata = {}
##Loop to save data to dicitonary
for i in step:
    #Create key name for dictionary
    name = 'Step' + str(i)
    #Same imported csv dat5a to key
    Stepdata[name] = pd.read_csv(folderPath + r'\StepSizeData' + videoName + r'\StepSize' + str(i) + r'.csv', usecols=[1,2,3,4,5,6])

#Create list to append temp data
tempAll = []
sigmaxAll = []
sigmayAll = []
##Loop to calculate temp data for different step sizes
for i in step:
    #Key name to access dict
    name = 'Step' + str(i)
    
    #Extract velcoity data and convert to mm/s
    velXStep = (Stepdata[name]['vx'])*pixeltomm*fps*(1/i)
    velYStep = (Stepdata[name]['vy'])*pixeltomm*fps*(1/i)
    
    #Calculating mean and standard deviation for x velocities
    meanX = statistics.mean(velXStep)
    stdX = statistics.stdev(velXStep)
    #Calculating mean and standard deviation for y velocities
    meanY = statistics.mean(velYStep)
    stdY = statistics.stdev(velYStep)
    
    #Particle diameter (m)
    d_par = 7.14e-6
    #Particle radius (m)
    r_par = d_par/2
    #Density of particle
    rho_mf = 1500 #kg/m^3
    #Particle mass (kg) = Volume of sphere * density
    mass_par = (4/3)*np.pi*(r_par**3)*(rho_mf) 
    #FWHM (mm)
    width1 = 2*np.sqrt(2*np.log(2))*stdX 
    width2 = 2*np.sqrt(2*np.log(2))*stdY
    #Kinetic energy
    energyx = 0.5*mass_par*(width1*1e-3)**2
    energyy = 0.5*mass_par*(width2*1e-3)**2
    #Convert to ev
    Tdx = energyx/(1.6e-19)
    Tdy = energyy/(1.6e-19)
    #Temperature (ev)
    Temp = np.sqrt(Tdx**2+Tdy**2)
    
    #Append data to list
    tempAll.append(Temp)
    sigmaxAll.append(stdX)
    sigmayAll.append(stdY)

##Create plot for plasma temp vs step size
plt.figure()
plt.plot(step,tempAll)
plt.xlabel('Step Size')
plt.ylabel('Temp (ev)')
plt.title('Plasma Temp vs Frame Step Size')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\PlasmaTempvsStSize' + imgType)

plt.figure()
plt.plot(step,tempAll)
plt.xlabel('Step Size')
plt.ylabel('Temp (ev)')
plt.ylim(0,2)
plt.title('Plasma Temp vs Frame Step Size Zoomed In')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\PlasmaTempvsStSizeZoom' + imgType)

plt.figure()
plt.plot(step,sigmaxAll)
plt.xlabel('Step Size')
plt.ylabel('Sigma X')
plt.title('Sigma X vs Frame Step Size')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\SigmaXFraStSize' + imgType)

plt.figure()
plt.plot(step,sigmaxAll)
plt.xlabel('Step Size')
plt.ylabel('Sigma X')
plt.ylim(0,0.5)
plt.title('Sigma X vs Frame Step Size Zoomed In')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\SigmaXFraStSizeZoom' + imgType)

plt.figure()
plt.plot(step,sigmayAll)
plt.xlabel('Step Size')
plt.ylabel('Sigma Y')
plt.title('Sigma Y vs Frame Step Size')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\SigmaYFraStSize' + imgType)

plt.figure()
plt.plot(step,sigmayAll)
plt.xlabel('Step Size')
plt.ylabel('Sigma Y')
plt.ylim(0,0.5)
plt.title('Sigma Y vs Frame Step Size Zoomed In')
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\SigmaYFraStSizeZoom' + imgType)

##Create directory for step size data
if not os.path.exists(folderPath + r'\Plots' + videoName + r'\StepSizeVelocity'):
    os.makedirs(folderPath + r'\Plots' + videoName + r'\StepSizeVelocity')

##Loop to create gaussian plot for x and y velocities for varying step zises
for i in step:
    #Key name to access dict
    name = 'Step' + str(i)
    
    #Extract and sort velcoity data and convert to mm/s
    velXStep = sorted(np.array((Stepdata[name]['vx'])*pixeltomm*fps*(1/i)))
    velYStep = sorted(np.array((Stepdata[name]['vy'])*pixeltomm*fps*(1/i)))
    
    #Calculating mean and standard deviation for x velocities
    meanX = statistics.mean(velXStep)
    stdX = statistics.stdev(velXStep)
    #Calculating mean and standard deviation for y velocities
    meanY = statistics.mean(velYStep)
    stdY = statistics.stdev(velYStep)

    #Define gaussian function
    def gauss(velocity,a,sigma,mu):
        return a*(1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*(velocity-mu)**2/sigma**2))

    #Calculate bins and n
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    n, bins, patches = ax1.hist(velXStep, bins=binwidth)  # output is two arrays
    ax2 = fig.add_subplot(122)
    n2, bins2, patches2 = ax2.hist(velYStep, bins=binwidth)  # output is two arrays
    plt.close(fig)
    
    #Find the center of each bin from the bin edges for x-vel
    bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
    #Find the center of each bin from the bin edges for y-vel
    bins_mean2 = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]
    
    poptx, pcovx = curve_fit(gauss, bins_mean, n, p0 = [max(n), meanX, stdX])
    popty, pcovy = curve_fit(gauss, bins_mean2, n2, p0 = [max(n2), meanY, stdY])
    
    #Find gaussian curve for x and y velocity
    gaussX = gauss(pd.Series(velXStep),poptx[0],poptx[2],poptx[1])
    gaussY = gauss(pd.Series(velYStep),popty[0],popty[2],popty[1])

    ##Create Subplots for x-y velocity (just dot histogram)##
    fig = plt.figure()
    title = 'Particle Velocity Step=' + str(i)
    fig.suptitle(title, fontsize=22, y=0.92)
    ##First Subaxis
    ax1 = fig.add_subplot(121)
    ax1.grid(True)
    ax1.set_xlabel(r'v$_{x}$ $mm/s$')
    ax1.set_ylabel('counts')
    #Scatter plot of x velocity count 
    ax1.scatter(bins_mean, n, s=20)
    #Plot Gaussian fit for x-vel
    ax1.plot(velXStep, gaussX, 'r',linewidth=1)
    #Set y-lim for max of both x and y velocity
    ax1.set_ylim(0,1.05*max(max(gaussX),max(gaussY)))
    ##Second Subaxis
    ax2 = fig.add_subplot(122)
    ax2.grid(True)
    ax2.set_xlabel(r'v$_{y}$ $mm/s$')
    #Scatter plot of y velocity count 
    ax2.scatter(bins_mean2, n2, s=20)
    #Plot Gaussian for y-vel
    ax2.plot(velYStep, gaussY, 'r',linewidth=1)
    #Set y-lim for max of both x and y velocity
    ax2.set_ylim(0,1.05*max(max(gaussX),max(gaussY)))
    ##Show and save subplot
    plt.savefig(folderPath + r'\Plots' + videoName + r'\StepSizeVelocity' + r'\Vel' + name + imgType)

##______________________________________________________________________________________________________________________________________________##

###Save Data for Individual Frame Data###

#Create Directory for Individual Frame Data
if not os.path.exists(folderPath + r'\FrameData' + videoName):
    os.makedirs(folderPath + r'\FrameData' + videoName)

#Loop to Save Data for Individual Frames
for i in range(currentframe):
    #Pull data for specific frame# from dataframe
    framData = t[t['frame'] == i]
    #Save specific columns
    xyData = framData[['y','x','frame','particle']]
    #Save columns to csv
    xyData.to_csv(folderPath + r'\FrameData' + videoName + videoName + 'Frame' + str(i) + r'.csv', index=False)
    print("Frame ", i)

##______________________________________________________________________________________________________________________________________________##

###Create Directory for Annotations###

if not os.path.exists(folderPath + r'\Annotations' + videoName):
    os.makedirs(folderPath + r'\Annotations' + videoName)
     
##______________________________________________________________________________________________________________________________________________##

###THIS PART OF THE CODE IS VERY GLITCHY, STILL EDITING TO BECOME MORE EFFICIENT###
###Create Interactive Annimation to View Annotations###

##Save annotated plots
#Loop over every frame
for i in range(currentframe):
    plt.ioff()
    #Locate particles for frame
    fSing2 = tp.locate(frames[i], pix, minmass=minM, separation=sep, threshold=thresh, invert=inv)
    fig = plt.figure()
    #Annotate frame
    tp.annotate(fSing2, frames[i], plot_style={'markersize':5})
    plt.close()
    #Save annotated frame
    fig.savefig(folderPath + r'\Annotations' + videoName + r'\frame' + str(i) + imgType)
    name = 'frame' + str(i)
    print ('Creating...' + name)

##Import annotated frames
#Create list
annot = []
#Loop over every frame
for i in range(currentframe):
    #Append annotaed frame to list
    annot.append(cv2.imread(folderPath + r'\Annotations' + videoName + r'\frame' + str(i) + imgType, 1))
    name = 'frame' + str(i)
    print ('Creating...' + name)
#Convert BGR to RGB
annot = np.flip(annot, axis=-1)

##Interactive plot with frame slider
#Variables for interactive plot
startFra = int(input("Starting frame ") or 0)
endFra = int(input("Ending Frame: ") or currentframe)
jump = int(input("Gap between frames: ") or 1)
#Create interactive plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
plt.axis('off')
#Show first frame
ax.imshow(annot[startFra])
#Create axes for slider
axframe = plt.axes([0.25, 0.15, 0.55, 0.03]) #!!!INPUT
##Create slider
framepos = Slider(axframe, 'Frame', startFra, endFra, 0, valstep=jump)
#Create function to be called when slider value is changed
def update(val):
    j = int(framepos.val)
    ax.imshow(annot[j])
    fig.canvas.draw_idle()
#Call update function when slider value is changed
framepos.on_changed(update)
# display graph
plt.show()    

##______________________________________________________________________________________________________________________________________________##

###THIS PART OF THE CODE IS VERY GLITCHY, STILL EDITING TO BECOME MORE EFFICIENT###
###Create GIF to View Annotations###

##Create time stepping annotation plot
fig, ax = plt.subplots()
plt.axis('off')
for i in range(len(annot)):
    ax.imshow(annot[i])
    plt.pause(0.5)

##______________________________________________________________________________________________________________________________________________##





###REMOVED CODE###
###CODE IS TEMPORARILY BEING KEPT###


# ##Create Subplots for x velocity (bar and dot histogram)##
# fig = plt.figure()
# fig.suptitle('X Velocity Distribution Function')
# ##First Subaxis
# ax1 = fig.add_subplot(121)
# #Histogram x velocity
# n, bins, patches = ax1.hist(velxSorted, bins=binwidth) # output is two arrays
# ax1.grid(True)
# ax1.set_xlabel(r'v$_{x}$ $mm/s$')
# ax1.set_ylabel('counts')
# ##Second Subaxis
# ax2 = fig.add_subplot(122)
# #Find the center of each bin from the bin edges
# bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
# #Scatter plot of velocity count 
# ax2.scatter(bins_mean, n, s=5)
# #Plot Gaussian
# ax2.plot(velxSorted, 3.5*len(velxSorted)*norm.pdf(velxSorted,meanX,sdX),'r',linewidth=1)
# ax2.grid(True)
# ax2.set_xlabel(r'v$_{x}$ $mm/s$')
# ##Show subplot
# plt.show()

# ##Create Subplots for y velocity (bar and dot histogram)##
# fig = plt.figure()
# fig.suptitle('Y Velocity Distribution Function')
# ##First Subaxis
# ax1 = fig.add_subplot(121)
# #Histogram y velocity
# n, bins, patches = ax1.hist(velySorted, bins=binwidth) # output is two arrays
# ax1.grid(True)
# ax1.set_xlabel(r'v$_{y}$ $mm/s$')
# ax1.set_ylabel('counts')
# ##Second Subaxis
# ax2 = fig.add_subplot(122)
# #Find the center of each bin from the bin edges
# bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
# #Scatter plot of y velocity count 
# ax2.scatter(bins_mean, n, s=5)
# #Plot Gaussian
# ax2.plot(velySorted, 3.5*len(velySorted)*norm.pdf(velySorted,meanY,sdY),'r',linewidth=1)
# ax2.grid(True)
# ax2.set_xlabel(r'v$_{y}$ $mm/s$')
# ##Show subplot
# plt.show()




# meanX = statistics.mean((check['vx']*pixeltocm*fps)/100)
# meanY = statistics.mean((check['vy']*pixeltocm*fps)/100)
# sdX = statistics.stdev(velxSorted)
# aX = meanX / (2*np.sqrt(2/np.pi))
# aY = meanY / (2*np.sqrt(2/np.pi))
# Boltz = (8.62*10**(-5))
# diam = 7.14e-6
# radi = diam/2
# rho = 1500 
# ParMass = 4.0*np.pi*radi**3*rho/3
# tempX = ((aX**2)*(ParMass))/Boltz
# tempY = ((aY**2)*(ParMass))/Boltz

# ##Pull relevant data from filtered trajectories
# #Renaming frame column (index and column have same name)
# t1 = t1.rename(columns = {'frame':'framenum'})
# #Sort by particle number then frame number
# t2 = t1.sort_values(by=['particle','framenum'])
# #Pull x and y values from batch of frames dataset
# x = t2['x'][:]
# y = t2['y'][:]
# #Pull particle column from batch of frames dataset
# partSer = t2['particle'][:]
# part = partSer.values.tolist()
# #Pull framenum column from batch of frames dataset
# framSer = t2['framenum'][:]
# fram = framSer.values.tolist()

# ##Convert particle location from pixels
# #Convert x and y pixels to cm
# xcm = x*pixeltocm
# ycm = y*pixeltocm
# #Convert position in cm to mm
# xmmSer = xcm*10
# ymmSer = ycm*10
# #Convert data series to list
# xmm = xmmSer.values.tolist()
# ymm = ymmSer.values.tolist()

# ##Calculate velocity in x and y direction
# #Create dict that can be appended to as lists
# velDict = defaultdict(list)
# #Loop that calculates velocities one particle at a time
# for i in range(len(x)-1):
#     #if statement prevents velocity calc from particle to particle
#     if int(part[i+1+start])-int(part[i+start]) == 0:
#         #if statement prevents velocity calc if linking feature skips particle frames
#         if int(fram[i+1+start])-int(fram[i+start]) == 1:
#             #Appends frame number to dict
#             velDict['framenum'].append(fram[i+1+start])
#             #Appends particle number to dict
#             velDict['particle'].append(part[i+1+start])
#             #Appends velocities to dict
#             velDict['velx'].append((xmm[i+1+start]-xmm[i+start])*fps)
#             velDict['vely'].append((ymm[i+1+start]-ymm[i+start])*fps)
# #Convert dict back to pandas dataframe
# velPD = pd.DataFrame(velDict)
# #Sort by frame number then particle number
# vel = velPD.sort_values(by=['framenum','particle'])
        

# ###Create Velocity Distribution Function with Gaussian Fit###

# ##Save data for singular frame and filter outliers
# pltFram = int(input("Input Frame Number to plot on Hist: ") or "1")
# velfranum = vel[vel['framenum'] == pltFram]
# print('Number of Particles in Frame: ', len(velfranum))
# #Pull x and y velocities for singular frame
# velfrax = velfranum['velx'][:]
# velfray = velfranum['vely'][:]
# #Convert data series to list
# velxFil = velfrax.values.tolist()
# velyFil = velfray.values.tolist()
# #Max and min velolcity limit
# filVel = 10 #!!!INPUT
# #Create list for filtered velocities
# velx = []
# vely = []
# #Loop to remove velocities outside of limit
# for i in range(len(velxFil)):
#     #If statement for x velocities
#     if -filVel<velxFil[i]<filVel:
#         velx.append(velxFil[i])
#     #If statement for y velocities
#     if -filVel<velyFil[i]<filVel:
#         vely.append(velyFil[i])
        
# ##Bins for histogram
# binwidth = 30 #!!!INPUT

# ##Defining variables for Gaussian Fit
# #Sorting x-y velocities numerically
# velxSorted = sorted(velx)
# velySorted = sorted(vely)
# #Calculating mean and standard deviation for x velocities
# meanX = statistics.mean(velxSorted)
# sdX = statistics.stdev(velxSorted)
# #Calculating mean and standard deviation for y velocities
# meanY = statistics.mean(velySorted)
# sdY = statistics.stdev(velySorted)

# ##Create Subplots for x velocity (bar and dot histogram)##
# fig = plt.figure()
# fig.suptitle('X Velocity Distribution Function')
# ##First Subaxis
# ax1 = fig.add_subplot(121)
# #Histogram x velocity
# n, bins, patches = ax1.hist(velx, bins=binwidth) # output is two arrays
# ax1.grid(True)
# ax1.set_xlabel(r'v$_{x}$ $mm/s$')
# ax1.set_ylabel('counts')
# ##Second Subaxis
# ax2 = fig.add_subplot(122)
# #Find the center of each bin from the bin edges
# bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
# #Scatter plot of velocity count 
# ax2.scatter(bins_mean, n, s=5)
# #Plot Gaussian
# ax2.plot(velxSorted, norm.pdf(velxSorted,meanX,sdX),'r',linewidth=1)
# ax2.grid(True)
# ax2.set_xlabel(r'v$_{x}$ $mm/s$')
# ##Show subplot
# plt.show()

# ##Create Subplots for y velocity (bar and dot histogram)##
# fig = plt.figure()
# fig.suptitle('Y Velocity Distribution Function')
# ##First Subaxis
# ax1 = fig.add_subplot(121)
# #Histogram y velocity
# n, bins, patches = ax1.hist(vely, bins=binwidth) # output is two arrays
# ax1.grid(True)
# ax1.set_xlabel(r'v$_{y}$ $mm/s$')
# ax1.set_ylabel('counts')
# ##Second Subaxis
# ax2 = fig.add_subplot(122)
# #Find the center of each bin from the bin edges
# bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
# #Scatter plot of y velocity count 
# ax2.scatter(bins_mean, n, s=5)
# #Plot Gaussian
# ax2.plot(velySorted, norm.pdf(velySorted,meanY,sdY),'r',linewidth=1)
# ax2.grid(True)
# ax2.set_xlabel(r'v$_{y}$ $mm/s$')
# ##Show subplot
# plt.show()

# ##Create Subplots for x-y velocity (just bar histogram)##
# fig = plt.figure()
# ##First Subaxis
# ax1 = fig.add_subplot(121)
# ax1.set_xlabel(r'v$_{x}$ $mm/s$')
# ax1.set_ylabel('counts')
# #Histogram x velocity
# n, bins, patches = ax1.hist(velx, bins=binwidth)  # output is two arrays
# ##Second Subaxis
# ax2 = fig.add_subplot(122)
# ax2.set_xlabel(r'v$_{y}$ $mm/s$')
# #Histogram y velocity
# n2, bins2, patches2 = ax2.hist(vely, bins=binwidth)  # output is two arrays
# ##Show and save subplot
# plt.show()
# plt.savefig(folderPath + r'\Plots' + videoName + r'\VelocityHistogram' + imgType)

# ##Create Subplots for x-y velocity (just dot histogram)##
# fig = plt.figure()
# ##First Subaxis
# ax1 = fig.add_subplot(121)
# ax1.grid(True)
# ax1.set_xlabel(r'v$_{x}$ $mm/s$')
# ax1.set_ylabel('counts')
# #Find the center of each bin from the bin edges for x-vel
# bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
# #Scatter plot of x velocity count 
# ax1.scatter(bins_mean, n, s=20)
# #Plot Gaussian fit for x-vel
# ax1.plot(velxSorted, norm.pdf(velxSorted,meanX,sdX), 'r',linewidth=1)
# ##Second Subaxis
# ax2 = fig.add_subplot(122)
# ax2.grid(True)
# ax2.set_xlabel(r'v$_{y}$ $mm/s$')
# #Find the center of each bin from the bin edges for y-vel
# bins_mean = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]
# #Scatter plot of y velocity count 
# ax2.scatter(bins_mean, n2, s=20)
# #Plot Gaussian for y-vel
# ax2.plot(velySorted, norm.pdf(velySorted,meanY,sdY), 'r',linewidth=1)
# ##Show and save subplot
# plt.show()
# plt.savefig(folderPath + r'\Plots' + videoName + r'\VelocityGaussian' + imgType)




# xyz = velxSorted*int(pixeltocm)*int(fps)
# zyx = velySorted*int(pixeltocm)*int(fps)
# from scipy.stats import maxwell
# paramsx = maxwell.fit(xyz)
# paramsy = maxwell.fit(zyx)
# ##Create Subplots for x-y velocity (just dot histogram)##
# fig = plt.figure()
# ##First Subaxis
# ax1 = fig.add_subplot(121)
# ax1.grid(True)
# ax1.set_xlabel(r'v$_{x}$ $mm/s$')
# ax1.set_ylabel('counts')
# #Find the center of each bin from the bin edges for x-vel
# n, bins, patches = ax1.hist(check['vx'], bins=binwidth, density=True)
# bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
# #Scatter plot of x velocity count 
# # ax1.scatter(bins_mean, n, s=20)
# #Plot Gaussian fit for x-vel
# ax1.plot(velxSorted, maxwell.pdf(velxSorted, paramsx[0], paramsx[1]), lw=3)
# ##Second Subaxis
# ax2 = fig.add_subplot(122)
# ax2.grid(True)
# ax2.set_xlabel(r'v$_{y}$ $mm/s$')
# #Find the center of each bin from the bin edges for y-vel
# n, bins, patches = ax2.hist(check['vy'], bins=binwidth, density=True)
# bins_mean = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]
# #Scatter plot of y velocity count 
# #ax2.scatter(bins_mean, n2, s=20)
# #Plot Gaussian for y-vel
# ax2.plot(velySorted, maxwell.pdf(velySorted, paramsy[0], paramsy[1]), lw=3)
# ##Show and save subplot
# plt.show()
# plt.savefig(folderPath + r'\Plots' + videoName + r'\MaxwellGaussian' + imgType)





# import scipy.stats as stats

# def vmag_hist_maxwell(vel_bin_size,vmag):
#     plt.ioff()
#     maxwell = stats.maxwell
#     params = maxwell.fit(vmag, floc=0)
#     max_bin = int(np.max(vmag)/vel_bin_size) + 2
#     bins_re= np.array([n*vel_bin_size for n in range(0,max_bin)])
#     n, bins, patches = plt.hist(vmag, bins_re, histtype = 'bar', facecolor='blue')  #n = counts, bins = bin locations, patches = ?
#     bins_m = [(bins_re[i]+bins_re[i+1])/2.0 for i in range(0,len(bins_re)-1)]
#     y_fit = len(vmag)*vel_bin_size*maxwell.pdf(bins_m, *params)
#     x_points = np.arange(0,len(vmag))
#     y_fit_smooth = len(vmag)*vel_bin_size*maxwell.pdf(x_points, *params)
#     plt.plot(x_points, y_fit_smooth, lw=2, color = 'red')
#     chi_sq = stats.chisquare(n, y_fit)
#     plt.xlabel("Velocity magnitude in km/s", size = 10)                 #Sets title
#     plt.ylabel("Normalized histogram with maxwellian fit", size = 10)       #Sets title
#     plt.close()

# check2 = vmag_hist_maxwell(binwidth, check['vx'])







#yFrames2= int(input("Input max y pixel coordinate: ") or "775")




# #Create Histogram for x velocities
# plt.hist(velx, bins=binwidth)
# plt.title('X Velocity Distribution Function')
# plt.xlabel('x velocity $mm/s$')
# plt.ylabel('counts')
# plt.grid()
# plt.show()

# #Create Histogram for y velocities
# plt.hist(vely, bins=binwidth)
# plt.title('Y Velocity Distribution Function')
# plt.xlabel('y velocity $mm/s$')
# plt.ylabel('counts')
# plt.grid()
# plt.show()


# startFra = int(input("Starting frame ") or 0)
# endFra = int(input("Ending Frame: ") or currentframe)
# jump = int(input("Gap between frames: ") or 1)

# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)
# plt.axis('off')
# print(tp.annotate(fSing2, frames[startFra], plot_style={'markersize':9}))

# axframe = plt.axes([0.25, 0.15, 0.55, 0.03])

# framepos = Slider(axframe, 'Frame', startFra, endFra, 0, valstep=jump)

# def update(val):
#     j = int(framepos.val)
#     tp.annotate(fSing2, frames[j], plot_style={'markersize':9})
#     fig.canvas.draw_idle()
    
# # Call update function when slider value is changed
# framepos.on_changed(update)
 
# # display graph
# plt.show()  





# import random as r
# xm=[]
# ym=[]
# for i in range(500):
#     xm.append(r.randrange(-10,10))
#     ym.append(r.randint(-10,10))


#import Video 
# video = pims.PyAVVideoReader(r"C:\Users\conno\Documents\TrackPy\ImageSequences\2p78mA_420V_12sccm_111mtorr_115volt_bias")  # or .mov, etc.
#Show frame with axes and control over scalin
# plt.imshow(video[0])

# frames = gray(pims.open(r"C:\Users\conno\Documents\TrackPy\ImageSequences\2p78mA_420V_12sccm_111mtorr_115volt_bias\*.png"))

# #Loops over number of frames
# for i in range(currentframe):
#     #Opens all frames in directory one at a time and converts to grayscale
#     if i < currentframe-1:
#         im = gray(cv2.imread(direcPath + r'\frame' + str(i) + imgType))
#     else:
#         print("Recommended starting scale x=375:775 y=525:900 ")
#         while True:
#             #Input for cropping scale
#             xFrames1 = int(input("Input min x scale: "))
#             xFrames2= int(input("Input max x scale: "))
#             yFrames1 = int(input("Input min y scale: "))
#             yFrames2= int(input("Input max y scale: "))
#             #Crops images
#             exFrame = im[xFrames1:xFrames2, yFrames1:yFrames2]
#             #Shows cropped image
#             plt.imshow(exFrame)
#             plt.show()
#             #Request to proceed
#             answer = input("Would you like to reset scale? (yes or no): ")
#             if answer=="yes":
#                 continue 
#             else:
#                 frames.append(im[xFrames1:xFrames2, yFrames1:yFrames2])
#                 break

# fSing = tp.locate(frames[0] ,7 ,minmass=100, seperation=2, threshold=5, invert=True)

# # shows the first few rows of data
# print(fSing.head())

# #Circles features on plot
# tp.annotate(fSing, frames[0])

# ##Check the subpixel accuracy
# #This checks that decimal part of the x and or position are evenly distributed
# #If mask size is too small, histogram shows dip in middle
# tp.subpx_bias(fSing)

# ##Histogram to examine DataFrame
# #Variable being examined
# var = 'mass'
# #Create Plot
# fig, ax = plt.subplots()
# ax.hist(fSing[var], bins=20)
# # Optionally, label the axes.
# ax.set(xlabel=var, ylabel='count');

# ###Locate Features for single frame###
# fig, ax = plt.subplots()
# ax.hist(fSing['mass'], bins=20)

# # Optionally, label the axes.
# ax.set(xlabel='mass', ylabel='count');

# fig, ax = plt.subplots()
# ax.hist(fSing['size'], bins=20)

# # Optionally, label the axes.
# ax.set(xlabel='size', ylabel='count');



# drawings = []
# fig, ax = plt.subplots()
# for frame in np.arange(0, currentframe, 50):  # show every 50th frame to keep file size low
#     fSing = tp.locate(frames[frame], pix, minmass=minM, separation=sep, threshold=thresh, invert=inv)
#     tp.annotate(fSing, frames[frame], plot_style={'markersize':9})
#     ax.set(yticks=[], xticks=[])
#     fig.tight_layout(pad=0)
#     fig.canvas.draw()
#     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#     drawings.append(data)

# fig.clf()
# framerate = 13
# pims.export(drawings, "0_video.avi", rate=framerate)


# def pos(lst):
#     return [x for x in lst if x >= 0] or None
# def neg(lst):
#     return [x for x in lst if x < 0] or None

# xposVel = pos(velx)
# xnegVel = np.abs(neg(velx))
# yposVel = pos(vely)
# ynegVel = np.abs(neg(vely))
# xbinPos = np.bincount(xposVel)
# xbinNeg = np.bincount(xnegVel.astype(np.int64))
# ybinPos = np.bincount(yposVel)
# ybinNeg = np.bincount(ynegVel.astype(np.int64))

# xBin = len(xbinPos) + len(xbinNeg)
# yBin = len(ybinPos) + len(ybinNeg)