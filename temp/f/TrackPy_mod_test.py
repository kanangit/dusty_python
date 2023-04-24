"""
Created on Fri Feb  3 21:58:41 2023

@author: Connor Belt

Trackpy software used to track dusty plasma particles
"""


##Import Libraries##
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
from pandas import DataFrame, Series  # for convenience
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
import cv2
import os
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
from scipy.stats import norm
import statistics
from matplotlib.widgets import Slider
import random


##Optionally, tweak styles.##
mpl.rc('figure',  figsize=(10, 10*(1236/1376))) #400/375
mpl.rc('image', cmap='gray')

##______________________________________________________________________________________________________________________________________________##

###Convert Video to Image Sequence and save to directory###

#File path of video that needs converted
folderPath = r'g:\My Drive\workAppState\prj_shocks\expData\data20230419temperatureProfileTest\inputs'
#folderPath = r'g:\My Drive\workAppState\prj_shocks\expData\data20230322velDistribTest_Connor\inputs'
#Filepath for directory to save image sequence
videoName = r'\2mA_420V_12sccm_111mtorr_135volt_bias'
#Video Type
vidType = '.avi'
#Image Type
imgType = '.png'

##CODE DOES NOT NEED ALTERED FOR VIDEO/IMAGE-SEQUENCE CONVERTER PAST THIS POINT
#Read the video from specified path
    
cam = cv2.VideoCapture(folderPath + r'\Video' + videoName + vidType)
  
try:
      
    # creating a folder named 
    direcPath = folderPath + r'\ImageSequences' + videoName
    if not os.path.exists(direcPath):
        os.makedirs(direcPath)
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
currentframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        name = direcPath + r'\frame' + str(currentframe) + imgType
        print ('Creating...' + name)
  
        # writing the extracted images
        cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()

##______________________________________________________________________________________________________________________________________________##

###Create Directory for Plots###

if not os.path.exists(folderPath + r'\Plots' + videoName):
    os.makedirs(folderPath + r'\Plots' + videoName)
    

##______________________________________________________________________________________________________________________________________________##

###Import image sequence and convert to grayscale###

#Define function that converts to grayscale
@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


#Creates an array for grayscale frames
frames = []

#Access first image in directory as test image
testim = gray(cv2.imread(direcPath + r'\frame' + str(0) + imgType))
print(" ")
print("View of current image before cropping provided in plots tab")
print(" ")
plt.imshow(testim)
plt.show()
print("Recommended starting scale is x=525:900 y=375:775")
print("(Press enter when prompted for input to use recommended scale)")
#Loop to test cropping scale and then apply to entire directory
while True:
    #Input for cropping scale
    xFrames1 = int(input("Input min x pixel coordinate: ") or "550")
    xFrames2 = int(input("Input max x pixel coordinate: ") or "925")
    yFrames1 = int(input("Input min y pixel coordinate: ") or "450")
    #yFrames2= int(input("Input max y pixel coordinate: ") or "775")
    yFrames2 = int(((xFrames2-xFrames1)*(1236/1376))+yFrames1)
    #Crops test images
    exFrame = testim[yFrames1:yFrames2, xFrames1:xFrames2]
    #Shows cropped image
    plt.imshow(exFrame)
    plt.show()
    #Request to proceed
    answer = input("Would you like to reset scale? (yes or no): ")
    if answer=="yes":
        #Restarts loop for new crop scale
        continue 
    else:
        #Loop to apply cropping to all images in directory
        for i in range(currentframe):
            im = gray(cv2.imread(direcPath + r'\frame' + str(i) + imgType))
            frames.append(im[yFrames1:yFrames2, xFrames1:xFrames2])
        break

plt.imshow(frames[0])

##______________________________________________________________________________________________________________________________________________##

###Locate Features for single frame###

#Single frame
print(" ")
print("Recommended values for tp.locate(): PixSize=3, minmass=15, seperation=3, threshhold=0, invert=False")
print("(Press enter when prompted for input to use recommended values)")
while True:
    #Asking for input from user, if no input use recommended settings
    fra = int(input("Frame number: ") or "0")
    pix = int(input("Feature size in pixels (odd#>2): ") or "3")
    minM = int(input("Minmass: ") or "10")
    sep = int(input("Seperation: ") or "5")
    thresh = int(input("Threshold: ") or "0")
    inv = input("Invert: ") or "False"
    
    #Locates particles using defined settings
    ##tp.locate(frame,estimated feature size in pixels, minmass removes features below value, invert frame color)
    fSing = tp.locate(frames[fra], pix, minmass=minM, separation=sep, threshold=thresh, invert=inv)
    #prints data head
    print("Head of DataFrame: ")
    print(fSing.head(5))
    ##Check the subpixel accuracy
    #This checks that decimal part of the x and or position are evenly distributed
    #If mask size is too small, histogram shows dip in middle
    tp.subpx_bias(fSing)
    plt.show()
    plt.pause(1)
    plt.close()
    #Circles features on plot
    fig = plt.figure()
    tp.annotate(fSing, frames[fra], plot_style={'markersize':9}) 
    plt.show()
    plt.pause(5)
    plt.close()
    fig.savefig(folderPath + r'\Plots' + videoName + r'\Annotation' + imgType)
    #Request to proceed
    answer2 = input("Would you like to change values? (yes or no): ") or "yes"
    if answer2 == "yes":
        #Restarts loop for new values
        continue 
    else:
        #ends the loop
        break
    
##______________________________________________________________________________________________________________________________________________##

###Link features into Particle Trajectories###

#Batch of frames
fTraj = tp.batch(frames[:], pix, minmass=minM, separation=sep, threshold=thresh, invert=inv)
#Convert and save data to CSV
fTraj.to_csv(folderPath + r'\csvData' + videoName + r'.csv')

##Track particle from frame to frame
# Turn off progress reports for best performance
tp.quiet()
#tp.link(batch frames, max distance in pixels particle can travel betwen frame, 
#        number of frames memory maintains ID for disappeared particles)
t = tp.link(fTraj,3,memory=1)
#Head of new DataSet
t.head()


#to check the method, calculate speeds of particles just for 2 frames:

df_second_frame = t[(t['frame'] == 1)]
arr_particles_first_frame = np.array(df_first_frame['particle'])



df_second_frame = t[(t['frame'] == 40)]

vx_index = t.columns.get_loc('vx')
arr_vx = np.array([])

for i in range(0, len(df_second_frame)):
    cur_p = df_second_frame['particle'].iloc[i]
    cur_x = df_second_frame['x'].iloc[i]
    prev_frame_cur_row = t[((t['frame'] == 0) & (t['particle'] == cur_p))]
    if (len(prev_frame_cur_row) == 1):
        prev_x = prev_frame_cur_row['x'].iloc[0]
        cur_vx = cur_x - prev_x
        df_second_frame.iloc[i, vx_index] = cur_vx
        print(i)
        arr_vx = np.append(arr_vx, cur_vx)
    
    

fig, ax = plt.subplots()
ax.hist(arr_vx, bins = 15)


#Now cycle through several frames (more than two)

# since finding velocities of all frames make take some time, 
# we are selecting a subset of frames (for example, from 0 to 40):
t_small = t[((t['frame'] > 0 ) & (t['frame'] < 40 ))].copy()
t_small = t_small.copy()

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

check = get_velocities(t_small, 35)

fig2, ax2 = plt.subplots()
ax2.hist(check['vy'], bins = 20)



check_all = get_velocities(t, 35)




fig3, ax3 = plt.subplots()
ax3.hist(check_all['vy'], bins = 20)

fig4, ax4 = plt.subplots()
ax4.hist(check_all['vx'], bins = 20)


##Filter spurious trajectories
#tp.filter(batch DataSet, Keeps trajectories that last longer than this number of frames)
t1 = tp.filter_stubs(t, 35)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())
#Create Plot of filtered trajectories
plt.figure()
tp.plot_traj(t1)
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\TrajectoryPlot' + imgType)

#Zoomed Plot of filtered trajectories
plt.figure()
tp.plot_traj(t1)
plt.xlim(200,250)
plt.ylim(250,200)
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\ZoomTrajectoryPlot' + imgType)

##Plot average appearance throughout trajectory
#Create plot size vs mass
plt.figure()
tp.mass_size(t.groupby('particle').mean())
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\SizeMassPlot' + imgType)


##______________________________________________________________________________________________________________________________________________##

###Create Velocity Distribution Function with Gaussian Fit###

#Starting Frame
start = 0
#Frames per second of video
fps = 99.0  
#Conversion factor from pixel to cm    
pixeltocm = 1/1218

#Renaming frame column (index and column have same name)
t1 = t1.rename(columns = {'frame':'framenum'})
#Sort by particle number then frame number
t2 = t1.sort_values(by=['particle','framenum'])
#Pull x and y values from batch of frames dataset
x=t2['x'][:]
y=t2['y'][:]
#Pull particle column from batch of frames dataset
partSer=t2['particle'][:]
part=partSer.values.tolist()


#Convert x and y pixels to cm
xcm=x*pixeltocm
ycm=y*pixeltocm
#Convert position in cm to mm
xmmSer=xcm*10
ymmSer=ycm*10
#Convert series to list
xmm=xmmSer.values.tolist()
ymm=ymmSer.values.tolist()

#Calculate velocity in x and y direction
velx=[]
vely=[]
for i in range(len(x)-1):
    if int(part[i+1+start])-int(part[i+start])==0:
        velx.append((xmm[i+1+start]-xmm[i+start])*fps)
        vely.append((ymm[i+1+start]-ymm[i+start])*fps)  
        
#Randomize velocities
np.random.shuffle(velx)
np.random.shuffle(vely)

#Pull x and y velocities from batch of dataset
velxFil=velx[:1000]
velyFil=vely[:1000]

#Filter out large velocities
filVel = 1
velx=[]
vely=[]
for i in range(len(velxFil)):
    if -filVel<velxFil[i]<filVel:
        velx.append(velxFil[i])
    if -filVel<velyFil[i]<filVel:
        vely.append(velyFil[i])
        

##Begin plotting velocity distribution function
#Bins for histogram
binwidth=100
    
#Create Histogram for x velocities
plt.hist(velx, bins=binwidth)
plt.title('X Velocity Distribution Function')
plt.xlabel('x velocity $mm/s$')
plt.ylabel('counts')
plt.grid()
plt.show()

#Create Histogram for y velocities
plt.hist(vely, bins=binwidth)
plt.title('Y Velocity Distribution Function')
plt.xlabel('y velocity $mm/s$')
plt.ylabel('counts')
plt.grid()
plt.show()

##Defining variables for Gaussian Fit
#Sorting x velocities numerically
velxSorted = sorted(velx)
#Calculating mean and standard deviation for x velocities
meanX = statistics.mean(velxSorted)
sdX = statistics.stdev(velxSorted)
#Sorting y velocities numerically
velySorted = sorted(vely)
#Calculating mean and standard deviation for y velocities
meanY = statistics.mean(velySorted)
sdY = statistics.stdev(velySorted)

##Create Subplots for x velocity
fig = plt.figure()
fig.suptitle('X Velocity Distribution Function')
#Create normal histogram
ax1 = fig.add_subplot(121)
n, bins, patches = ax1.hist(velx, bins=binwidth) # output is two arrays
ax1.grid(True)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Scatter plot of count values with Gaussian Fit
#Find the center of each bin from the bin edges
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
ax2 = fig.add_subplot(122)
ax2.scatter(bins_mean, n, s=5)
ax2.plot(velxSorted, norm.pdf(velxSorted,meanX,sdX),'r',linewidth=1)
ax2.grid(True)
ax2.set_xlabel(r'v$_{x}$ $mm/s$')
plt.show()


##Create Subplots for y velocity
fig = plt.figure()
fig.suptitle('Y Velocity Distribution Function')
#Create normal histogram
ax1 = fig.add_subplot(121)
n, bins, patches = ax1.hist(vely, bins=binwidth)  # output is two arrays
ax1.grid(True)
ax1.set_xlabel(r'v$_{y}$ $mm/s$')
ax1.set_ylabel('counts')
#Scatter plot of count values with Gaussian Fit
#Find the center of each bin from the bin edges
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
ax2 = fig.add_subplot(122)
ax2.scatter(bins_mean, n, s=5)
ax2.plot(velySorted, norm.pdf(velySorted,meanY,sdY), 'r',linewidth=1)
ax2.grid(True)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
n, bins, patches = ax1.hist(velx, bins=binwidth)  # output is two arrays
ax2 = fig.add_subplot(122)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
n2, bins2, patches2 = ax2.hist(vely, bins=binwidth)  # output is two arrays
plt.savefig(folderPath + r'\Plots' + videoName + r'\VelocityHistogram' + imgType)


##Create subaxis plot
fig = plt.figure()
#Create hisotgram for vel x
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax1.set_xlabel(r'v$_{x}$ $mm/s$')
ax1.set_ylabel('counts')
#Scatter plot of count values with Gaussian Fit
#Find the center of each bin from the bin edges
bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
ax1.scatter(bins_mean, n, s=20)
ax1.plot(velxSorted, norm.pdf(velxSorted,meanX,sdX), 'r',linewidth=1)
#Create hisotgram for vel y
ax2 = fig.add_subplot(122)
ax2.grid(True)
ax2.set_xlabel(r'v$_{y}$ $mm/s$')
#Scatter plot of count values with Gaussian Fit
#Find the center of each bin from the bin edges
bins_mean = [0.5 * (bins2[i] + bins2[i+1]) for i in range(len(n2))]
ax2.scatter(bins_mean, n2, s=20)
ax2.plot(velySorted, norm.pdf(velySorted,meanY,sdY), 'r',linewidth=1)
plt.show()
plt.savefig(folderPath + r'\Plots' + videoName + r'\VelocityGaussian' + imgType)

##______________________________________________________________________________________________________________________________________________##

if not os.path.exists(folderPath + r'\Annotations' + videoName):
    os.makedirs(folderPath + r'\Annotations' + videoName)
     
for i in range(currentframe):
    plt.ioff()
    fSing2 = tp.locate(frames[i], pix, minmass=minM, separation=sep, threshold=thresh, invert=inv)
    fig = plt.figure()
    tp.annotate(fSing2, frames[i], plot_style={'markersize':9})
    plt.close()
    fig.savefig(folderPath + r'\Annotations' + videoName + r'\frame' + str(i) + imgType)
    name = 'frame' + str(i)
    print ('Creating...' + name)

annot = []
for i in range(currentframe):
    annot.append(cv2.imread(folderPath + r'\Annotations' + videoName + r'\frame' + str(i) + imgType, 1))
    name = 'frame' + str(i)
    print ('Creating...' + name)

#Convert BGR to RGB
annot = np.flip(annot, axis=-1)

startFra = int(input("Starting frame ") or 0)
endFra = int(input("Ending Frame: ") or currentframe)
jump = int(input("Gap between frames: ") or 1)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
plt.axis('off')
ax.imshow(annot[startFra])

axframe = plt.axes([0.25, 0.15, 0.55, 0.03])

framepos = Slider(axframe, 'Frame', startFra, endFra, 0, valstep=jump)

def update(val):
    j = int(framepos.val)
    ax.imshow(annot[j])
    fig.canvas.draw_idle()
    
# Call update function when slider value is changed
framepos.on_changed(update)
 
# display graph
plt.show()    

  
#Create time stepping annotation plot
fig, ax = plt.subplots()
plt.axis('off')
for i in range(len(annot)):
    ax.imshow(annot[i])
    plt.pause(0.1)








##REMOVED CODE




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