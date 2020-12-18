
# coding: utf-8

# # XPCS&XSVS Pipeline for Single-(Gi)-SAXS Run
# "This notebook corresponds to version {{ version }} of the pipeline tool: https://github.com/NSLS-II/pipelines"
# 
# This notebook begins with a raw time-series of images and ends with $g_2(t)$ for a range of $q$, fit to an exponential or stretched exponential, and a two-time correlation functoin.
# 
# ## Overview
# 
# * Setup: load packages/setup path
# * Load Metadata & Image Data
# * Apply Mask
# * Clean Data: shutter open/bad frames
# * Get Q-Map
# * Get 1D curve
# * Define Q-ROI (qr, qz)
# * Check beam damage
# * One-time Correlation
# * Fitting
# * Two-time Correlation
# The important scientific code is imported from the [chxanalys](https://github.com/yugangzhang/chxanalys/tree/master/chxanalys) and [scikit-beam](https://github.com/scikit-beam/scikit-beam) project. Refer to chxanalys and scikit-beam for additional documentation and citation information.
# 
# ### DEV
# * V8: Update visbility error bar calculation using pi = his/N +/- sqrt(his_i)/N
# *     Update normlization in g2 calculation uing 2D-savitzky golay (SG ) smooth
# 
# ## CHX Olog NoteBook
# CHX Olog (https://logbook.nsls2.bnl.gov/11-ID/)
# 
# ## Setup
# 
# Import packages for I/O, visualization, and analysis.

# In[1]:


from pyCHX.chx_packages import *
get_ipython().magic('matplotlib notebook')
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({ 'image.origin': 'lower'   })
plt.rcParams.update({ 'image.interpolation': 'none'   })
import pickle as cpk
from pyCHX.chx_xpcs_xsvs_jupyter_V1 import *
from pyCHX.chx_generic_functions import get_qval_qwid_dict,get_roi_mask_qval_qwid_by_shift
import itertools


# In[2]:


#%run /home/yuzhang/pyCHX_link/pyCHX/chx_generic_functions.py


# In[3]:


get_ipython().magic('matplotlib notebook')
#%matplotlib inline


# ## Control Runs Here

# In[4]:


scat_geometry = 'saxs'  #suport 'saxs', 'gi_saxs', 'ang_saxs' (for anisotropics saxs or flow-xpcs)
#scat_geometry = 'ang_saxs' 
#scat_geometry = 'gi_waxs' 
#scat_geometry = 'gi_saxs'


roi_auto = False #True #False #True #if True, will automatically create a roi based on the roi_type ( iso/aniso etc), currently only works for SAXS


analysis_type_auto = False #True #if True, will take "analysis type" option from data acquisition func series
qphi_analysis = True  #if True, will do q-phi (anisotropic analysis for transmission saxs)

isotropic_Q_mask = 'normal' #'wide' # 'normal' # 'wide'  ## select wich Q-mask to use for rings: 'normal' or 'wide'
phi_Q_mask = 'flow' #'phi_4x_20deg'   ## select wich Q-mask to use for phi analysis
q_mask_name = ''

force_compress = False #True   #force to compress data 
bin_frame = False   #generally make bin_frame as False
para_compress = True    #parallel compress
run_fit_form = False    #run fit form factor 
run_waterfall =  False #True   #run waterfall analysis
run_profile_plot = False  #run prolfile plot for gi-saxs
run_t_ROI_Inten = True  #run  ROI intensity as a function of time
run_get_mass_center = False  # Analysis for mass center of reflective beam center
run_invariant_analysis = False
run_one_time =  True  #run  one-time
cal_g2_error =  False  #True  #calculate g2 signal to noise
#run_fit_g2 = True       #run  fit one-time, the default function is "stretched exponential"
fit_g2_func = 'stretched'
run_two_time =   True    #run  two-time
run_four_time = False #True #True #False   #run  four-time
run_xsvs=  False #False         #run visibility analysis
att_pdf_report = True    #attach the pdf report to CHX olog
qth_interest = 3 #the intested single qth             
use_sqnorm = True    #if True, use sq to normalize intensity
use_SG = True #False #True # False        #if True, use the Sawitzky-Golay filter of the avg_img for normalization
use_SG_bin_frames =  False   #if True, use the Sawitzky-Golay filter of the (binned) frame for normalization 

use_imgsum_norm= True  #if True use imgsum to normalize intensity for one-time calculatoin
pdf_version='_%s'%get_today_date()     #for pdf report name
run_dose = True # False #True # True #False  #run dose_depend analysis


if scat_geometry == 'gi_saxs':run_xsvs= False;use_sqnorm=False
if scat_geometry == 'gi_waxs':use_sqnorm = False
if scat_geometry != 'saxs':qphi_analysis = False;scat_geometry_ = scat_geometry;roi_auto = False  
else:scat_geometry_ = ['','ang_'][qphi_analysis]+ scat_geometry   
if scat_geometry != 'gi_saxs':run_profile_plot = False


# In[5]:


scat_geometry


# In[6]:


taus=None;g2=None;tausb=None;g2b=None;g12b=None;taus4=None;g4=None;times_xsv=None;contrast_factorL=None; lag_steps = None 


# ## Make a directory for saving results

# In[7]:


CYCLE= '2019_2'  #change clycle here
path = '/XF11ID/analysis/%s/masks/'%CYCLE


# ## Load Metadata & Image Data
# 
# 

# ### Change this line to give a uid

# In[8]:


username      =  getpass.getuser()


uid = 'c50a8807' #(scan num: 12159 (Measurement: 10ms 1000fraCoralpor
uid = 'ab2b703b' #(scan num: 12162 (Measurement: T=0.11 10ms 1000fraCoralpor
uid = 'e9cc00a4' #(scan num: 12195 (Measurement: T=0.11 10ms 1000fraCoralpor   
uid = 'd767d284-eb5e-4b61-a446-f9f8d242d630'
uid = '9327c18b' #(scan num: 12919 (Measurement: making mask for 4M
uid = '4f25490c' #(scan num: 12918 (Measurement: Series 0.00134s X 1000 -sample: CoralPor xh=1.4 yh=-1.789 Series 0.00134s X 1000 -sample: CoralPor
uid = '96aff7b4' #(scan num: 12923 (Measurement: Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 xh=0.8 yh=-1.433 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2  
uid = '4f9ad2bc' #(scan num: 12925 (Measurement: Series 0.00134s X 5000 -sample: SiO2 D250nm NP:Tw=1:2 xh=0.8 yh=-1.327 Series 0.00134s X 5000 -sample: SiO2 D250nm NP:Tw=1:2
uid = '4e7902fb' #(scan num: 12926 (Measurement: Series 0.0134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 xh=0.8 yh=-1.22 Series 0.0134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2  
uid = '048b3c39' #(scan num: 12927 (Measurement: Series 0.134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 xh=0.8 yh=-1.114 Series 0.134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2   
uid = '8fc303ae' #(scan num: 12928 (Measurement: Series 0.00134s X 500 -sample: SiO2 D250nm NP:Tw=1:2 xh=0.8 yh=-1.008 Series 0.00134s X 500 -sample: SiO2 D250nm NP:Tw=1:2
uid = '03d3aff2' #(scan num: 12929 (Measurement: Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 att2: 0.1904 xh=0.8 yh=-0.901 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2        "
uid = '8d62684c' #(scan num: 12930 (Measurement: Series 0.00134s X 2000 -sample: SiO2 D250nm NP:Tw=1:2 xh=0.8 yh=-0.794 Series 0.00134s X 2000 -sample: SiO2 D250nm NP:Tw=1:2        "
uid = 'bccf4825' #(scan num: 12931 (Measurement: Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 att2: 0.1904 xh=0.8 yh=-0.688 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2        "
uid = '704fd88d' #(scan num: 12933 (Measurement: Series 0.00134s X 250 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell xh=-0.1 yh=-0.582 Series 0.00134s X 250 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell        "
uid = '8e4ad0b1' #(scan num: 12934 (Measurement: Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell xh=-0.1 yh=-0.476 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell        "
print(uid)
username = 'dnandyala'



run_two_time  =   False #True 
run_dose      =   False 


# In[9]:


uid='e9716a17' #flow-0.5 at center
uid='b7bf428d' # center(scan num: 13088) (Measurement: Frate=0.25 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='a8a329fa'#center (scan num: 13117) (Measurement: Frate=0.75 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='e0789d35'#count : 1 ['e0789d35'] (scan num: 13038) (Measurement: Frate=1 ul/min xh=-0.35 yh=-1.377 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell )

#uid='1928d9b9'#count : 1 ['1928d9b9'] (scan num: 13132) (Measurement: Frate=0.15 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='9a9a572e'#count : 1 ['9a9a572e'] (scan num: 13207) (Measurement: Frate=0.15 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='ad5b1d33'#count : 1 ['ad5b1d33'] (scan num: 13222) (Measurement: Frate=0.35 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )


# In[10]:


uid='a8a329fa'#center (scan num: 13117) (Measurement: Frate=0.75 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='ad5b1d33'#count : 1 ['ad5b1d33'] (scan num: 13222) (Measurement: Frate=0.35 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='1928d9b9'#Frate=0.15 ul/min bad data set-Don't run analysis for this UID
uid='9a9a572e'#count : 1 ['9a9a572e'] (scan num: 13207) (Measurement: Frate=0.15 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='e0789d35'#count : 1 ['e0789d35'] (scan num: 13038) (Measurement: Frate=1 ul/min xh=-0.35 yh=-1.377 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell )


# In[11]:


uid='798649ea'#count : 1 ['798649ea'] (scan num: 13215) (Measurement: Frate=0.35 ul/min xh=-0.7 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='d863fc96'#count : 1 ['d863fc96'] (scan num: 13216) (Measurement: Frate=0.35 ul/min xh=-0.65 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='2b9c32cb'#count : 1 ['2b9c32cb'] (scan num: 13217) (Measurement: Frate=0.35 ul/min xh=-0.6 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='833edaf6'#count : 1 ['833edaf6'] (scan num: 13218) (Measurement: Frate=0.35 ul/min xh=-0.55 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='be3d2c32'#count : 1 ['be3d2c32'] (scan num: 13219) (Measurement: Frate=0.35 ul/min xh=-0.5 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='2a156a24'#count : 1 ['2a156a24'] (scan num: 13220) (Measurement: Frate=0.35 ul/min xh=-0.45 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='745c8091'#count : 1 ['745c8091'] (scan num: 13221) (Measurement: Frate=0.35 ul/min xh=-0.4 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='d7f9efbc'#count : 1 ['d7f9efbc'] (scan num: 13223) (Measurement: Frate=0.35 ul/min xh=-0.3 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='32e5a3a8'#count : 1 ['32e5a3a8'] (scan num: 13224) (Measurement: Frate=0.35 ul/min xh=-0.25 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='1e56220b'#count : 1 ['1e56220b'] (scan num: 13225) (Measurement: Frate=0.35 ul/min xh=-0.2 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='9ee57b56'#count : 1 ['9ee57b56'] (scan num: 13226) (Measurement: Frate=0.35 ul/min xh=-0.15 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='bd942c1b'#count : 1 ['bd942c1b'] (scan num: 13227) (Measurement: Frate=0.35 ul/min xh=-0.1 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='69a0d5b6'#count : 1 ['69a0d5b6'] (scan num: 13228) (Measurement: Frate=0.35 ul/min xh=-0.05 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='25a67326'#count : 1 ['25a67326'] (scan num: 13229) (Measurement: Frate=0.35 ul/min xh=0.0 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )


# In[12]:


uid='ad5b1d33'#count : 1 ['ad5b1d33'] (scan num: 13222) (Measurement: Frate=0.35 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
#uid='e0789d35'#count : 1 ['e0789d35'] (scan num: 13038) (Measurement: Frate=1 ul/min xh=-0.35 yh=-1.377 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell )
#uid='e9716a17'#count : 1 ['e9716a17'] (scan num: 13017) (Measurement: Frate=0.5 ul/min xh=-0.35 yh=-1.377 Series 0.00134s X 1000 -sample: SiO2 D250nm NP:Tw=1:2 Flow Cell )
#uid='9a9a572e'#count : 1 ['9a9a572e'] (scan num: 13207) (Measurement: Frate=0.15 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )


# In[13]:


#uid='9a9a572e'#count : 1 ['9a9a572e'] (scan num: 13207) (Measurement: Frate=0.15 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )


# In[14]:


data_dir0  = create_user_folder(CYCLE, username)
print( data_dir0 )


# In[15]:


uid = uid[:8]
print('The current uid for analysis is: %s...'%uid)


# In[16]:


#get_last_uids( -1)


# In[17]:


sud = get_sid_filenames(db[uid])
for pa in sud[2]:
    if 'master.h5' in pa:
        data_fullpath = pa
print ('scan_id, full-uid, data path are:  %s--%s--%s'%(sud[0], sud[1], data_fullpath ))

#start_time, stop_time = '2017-2-24  12:23:00', '2017-2-24  13:42:00' 
#sids, uids, fuids  = find_uids(start_time, stop_time)


# In[18]:


data_dir = os.path.join(data_dir0, '%s/'%(sud[1]))
os.makedirs(data_dir, exist_ok=True)
print('Results from this analysis will be stashed in the directory %s' % data_dir)
uidstr = 'uid=%s'%uid


# # Don't Change the lines below here

# * get metadata

# In[19]:


md = get_meta_data( uid )
md_blue = md.copy()
#md_blue


# In[20]:


#md_blue['detectors'][0]
#if md_blue['OAV_mode'] != 'none':
#    cx , cy = md_blue[md_blue['detectors'][0]+'_beam_center_x'], md_blue[md_blue['detectors'][0]+'_beam_center_x']
#else: 
#    cx , cy = md_blue['beam_center_x'], md_blue['beam_center_y']
#print(cx,cy)


# In[21]:


detectors = sorted(get_detectors(db[uid]))
print('The detectors are:%s'%detectors)
if len(detectors) >1:
    md['detector'] = detectors[1]
    print( md['detector'])


# In[22]:


if md['detector'] =='eiger4m_single_image' or md['detector'] == 'image':    
    reverse= True
    rot90= False
elif md['detector'] =='eiger500K_single_image':    
    reverse= True
    rot90=True
elif md['detector'] =='eiger1m_single_image':    
    reverse= True
    rot90=False
print('Image reverse: %s\nImage rotate 90: %s'%(reverse, rot90))    


# In[23]:


try:
    cx , cy = md_blue['beam_center_x'], md_blue['beam_center_y']
    print(cx,cy)
except:
    print('Will find cx,cy later.')


# ### Load ROI defined by "XPCS_Setup" Pipeline

# #### Define data analysis type

# In[24]:


if analysis_type_auto:#if True, will take "analysis type" option from data acquisition func series
    try:
        if scat_geometry != 'gi_saxs':
            qphi_analysis_ = md['analysis'] #if True, will do q-phi (anisotropic analysis for transmission saxs)
            print(md['analysis'])
            if qphi_analysis_ == 'iso':
                qphi_analysis = False
            elif qphi_analysis_ == '':
                qphi_analysis = False
            else:
                qphi_analysis = True
            #for other analysis type, in case of GiSAXS, low_angle/high_anlge for instance    
        else:
            gisaxs_inc_type = md['analysis'] 
        
    except:
        gisaxs_inc_type = None
        print('There is no analysis in metadata.')
        
print('Will %s qphis analysis.'%['NOT DO','DO'][qphi_analysis]) 

if scat_geometry != 'saxs':qphi_analysis = False;scat_geometry_ = scat_geometry  
else:scat_geometry_ = ['','ang_'][qphi_analysis]+ scat_geometry   
if scat_geometry != 'gi_saxs':run_profile_plot = False     
    
print(scat_geometry_)    


# In[25]:


scat_geometry


# In[26]:


#%run /home/yuzhang/pyCHX_link/pyCHX/chx_generic_functions.py


# * get data

# In[27]:


imgs = load_data( uid, md['detector'], reverse= reverse, rot90=rot90  )
md.update( imgs.md );Nimg = len(imgs);
#md['beam_center_x'], md['beam_center_y']  = cx, cy
#if 'number of images'  not in list(md.keys()):
md['number of images']  = Nimg
pixel_mask =  1- np.int_( np.array( imgs.md['pixel_mask'], dtype= bool)  )
print( 'The data are: %s' %imgs )

#md['acquire period' ] = md['cam_acquire_period']
#md['exposure time'] =  md['cam_acquire_time']
mdn = md.copy()


# #### Load Chip mask depeding on detector

# In[28]:


if md['detector'] =='eiger1m_single_image':
    Chip_Mask=np.load( '/XF11ID/analysis/2017_1/masks/Eiger1M_Chip_Mask.npy')
elif md['detector'] =='eiger4m_single_image' or md['detector'] == 'image':    
    Chip_Mask= np.array(np.load( '/XF11ID/analysis/2017_1/masks/Eiger4M_chip_mask.npy'), dtype=bool)
    BadPix =     np.load('/XF11ID/analysis/2018_1/BadPix_4M.npy'  )  
    Chip_Mask.ravel()[BadPix] = 0
elif md['detector'] =='eiger500K_single_image':
    #print('here')
    Chip_Mask=  np.load( '/XF11ID/analysis/2017_1/masks/Eiger500K_Chip_Mask.npy')  #to be defined the chip mask
    Chip_Mask = np.rot90(Chip_Mask)
    pixel_mask = np.rot90(  1- np.int_( np.array( imgs.md['pixel_mask'], dtype= bool))   )
    
else:
    Chip_Mask = 1
print(Chip_Mask.shape, pixel_mask.shape)


# In[29]:


use_local_disk = True
import shutil,glob


# In[30]:


save_oavs = False
if len(detectors)==2:
    if '_image' in md['detector']:
        pref = md['detector'][:-5]
    else:
        pref=md['detector']
    for k in [ 'beam_center_x', 'beam_center_y','cam_acquire_time','cam_acquire_period','cam_num_images',
             'wavelength', 'det_distance', 'photon_energy']:
        md[k] =  md[ pref + '%s'%k]    
    
    if 'OAV_image' in detectors:
        try:
            save_oavs_tifs(  uid, data_dir )
            save_oavs = True
        except:
            pass
        


# In[31]:


print_dict( md,  ['suid', 'number of images', 'uid', 'scan_id', 'start_time', 'stop_time', 'sample', 'Measurement',
                  'acquire period', 'exposure time',  
         'det_distance', 'beam_center_x', 'beam_center_y', ] )


# ## Overwrite Some Metadata if Wrong Input

# ### Define incident beam center (also define reflection beam center for gisaxs)

# In[32]:


if scat_geometry =='gi_saxs':
    inc_x0 =  md['beam_center_x'] 
    inc_y0 =  imgs[0].shape[0] - md['beam_center_y']     
    
    refl_x0 =     md['beam_center_x']  
    refl_y0 =     imgs[0].shape[0] - 673   
    
    print( "inc_x0, inc_y0, ref_x0,ref_y0 are: %s %s %s %s."%(inc_x0, inc_y0, refl_x0, refl_y0) )
else:
    if md['detector'] =='eiger4m_single_image' or md['detector'] == 'image' or md['detector']=='eiger1m_single_image':    
        inc_x0 =  imgs[0].shape[0] - md['beam_center_y']   
        inc_y0=   md['beam_center_x']
    elif md['detector'] =='eiger500K_single_image':    
        inc_y0 =  imgs[0].shape[1] - md['beam_center_y']   
        inc_x0 =   imgs[0].shape[0] - md['beam_center_x']
    
    print(inc_x0, inc_y0)
    ###for this particular uid, manually give x0/y0
    #inc_x0 = 1041
    #inc_y0 = 1085


# In[33]:


dpix, lambda_, Ldet,  exposuretime, timeperframe, center = check_lost_metadata(
    md, Nimg, inc_x0 = inc_x0, inc_y0=   inc_y0, pixelsize = 7.5*10*(-5) )
if scat_geometry =='gi_saxs':center=center[::-1]
setup_pargs=dict(uid=uidstr, dpix= dpix, Ldet=Ldet, lambda_= lambda_, exposuretime=exposuretime,
        timeperframe=timeperframe, center=center, path= data_dir)
print_dict( setup_pargs )


# In[34]:


setup_pargs


# ## Apply Mask
# * load and plot mask if exist 
# * otherwise create a mask using Mask pipeline
# * Reverse the mask in y-direction due to the coordination difference between python and Eiger software
# * Reverse images in y-direction
# * Apply the mask

# # Change the lines below to give mask filename

# In[35]:


if scat_geometry == 'gi_saxs':
    mask_path = '/XF11ID/analysis/2019_2/masks/'    
    mask_name =  'Mar28_2019_4M_GISAXS.npy'
    
elif scat_geometry == 'saxs':
    mask_path = '/XF11ID/analysis/2019_2/masks/'
    if md['detector'] =='eiger4m_single_image' or md['detector'] == 'image':    
        mask_name = 'May30_2019_4M_SAXS.npy' 
        #mask_name=None
    elif md['detector'] =='eiger500K_single_image':    
        mask_name = 'May30_2019_500K_SAXS.npy'    
        
elif scat_geometry == 'gi_waxs':    
    mask_path = '/XF11ID/analysis/2019_2/masks/'    
    mask_name =  'May7_2019_1M_WAXS.npy'  


# In[36]:


mask = load_mask(mask_path, mask_name, plot_ =  False, image_name = uidstr + '_mask', reverse= reverse, rot90=rot90  ) 
mask =  mask * pixel_mask * Chip_Mask
show_img(mask,image_name = uidstr + '_mask', save=True, path=data_dir, aspect=1, center=center[::-1])
mask_load=mask.copy()
imgsa = apply_mask( imgs, mask )


# # Check several frames average  intensity

# #### Load ROI mask depending on data analysis type

# In[37]:


#%run ~/pyCHX_link/pyCHX/chx_generic_functions.py


# In[38]:


print(roi_auto, qphi_analysis, isotropic_Q_mask)


# In[39]:


##For SAXS 
roi_path = '/XF11ID/analysis/2019_2/masks/'
roi_date = 'May30B'

if scat_geometry =='saxs':
    ## For auto load roi mask
    if roi_auto: 
        general_path = '/nsls2/xf11id1/analysis/2017_1/masks/'
        roi_mask_norm = general_path + 'general_roi_mask_norm.npy' 
        roi_mask_wide = general_path + 'general_roi_mask_wide.npy' 
        roi_mask_phi_4x_20deg = general_path + 'general_roi_mask_phi_4x_20deg.npy' 
        roi_mask_12x_30deg = general_path + 'general_roi_mask_12x_30deg.npy' 
        roi_mask_12x_15deg_flow = general_path + 'general_roi_mask_12x_15deg_flow.npy'
        if qphi_analysis ==  False:
            if isotropic_Q_mask == 'normal':
                fp = roi_mask_norm
            elif isotropic_Q_mask == 'wide':
                fp = roi_mask_wide
        elif qphi_analysis:
            if phi_Q_mask == 'phi_4x_20deg':
                fp = roi_mask_phi_4x_20deg
            elif phi_Q_mask == 'phi_12x_30deg': 
                fp = roi_mask_phi_12x_30deg
            elif phi_Q_mask == 'phi_12x_15deg_flow': 
                fp = roi_mask_phi_12x_15deg_flow 
        roi_mask0 = np.load(fp)    
        old_cen=[4000,4000]        
        roi_mask, qval_dict, qwid_dict = get_roi_mask_qval_qwid_by_shift( 
                        new_cen=center, new_mask= mask, old_cen=old_cen,
                                    old_roi_mask=roi_mask0, limit_qnum= None,
                                    setup_pargs= setup_pargs,  geometry = scat_geometry_,
                                    ) 
 
    else: 
        if qphi_analysis ==  False:
            if isotropic_Q_mask == 'normal':
                #print('Here')
                q_mask_name='rings'
                if md['detector'] =='eiger4m_single_image' or md['detector'] == 'image': #for 4M 
                    fp = roi_path + 'roi_mask_%s_4M_norm.pkl'%roi_date  
                    

                elif md['detector'] =='eiger500K_single_image': #for 500K   
                    fp = roi_path + 'roi_mask_%s_500K_norm.pkl'%roi_date   


            elif isotropic_Q_mask == 'wide':
                q_mask_name='wide_rings'
                if md['detector'] =='eiger4m_single_image' or md['detector'] == 'image': #for 4M   
                    fp = roi_path + 'roi_mask_%s_4M_wide.pkl'%roi_date   
                elif md['detector'] =='eiger500K_single_image': #for 500K   
                    fp = roi_path + 'roi_mask_%s_500K_wide.pkl'%roi_date   


        elif qphi_analysis:
            if phi_Q_mask =='phi_4x_20deg':
                q_mask_name='phi_4x_20deg'
                if md['detector'] =='eiger4m_single_image' or md['detector'] == 'image': #for 4M 
                    fp = roi_path + 'roi_mask_%s_4M_phi_4x_20deg.pkl'%roi_date   
                elif md['detector'] =='eiger500K_single_image': #for 500K 
                    fp = roi_path + 'roi_mask_%s_500K_phi_4x_20deg.pkl'%roi_date
            elif  phi_Q_mask =='flow':    
                q_mask_name='flow'
                if md['detector'] =='eiger4m_single_image' or md['detector'] == 'image': #for 4M 
                    fp  = roi_path + 'roi_mask_%s_4M_flow.pkl'%roi_date
                    fp  = roi_path + 'roi_mask_%s_4M_flow.pkl-2'%roi_date
                    print(fp)
                elif md['detector'] =='eiger500K_single_image': #for 500K
                    fp  = roi_path + 'roi_mask_%s_500K_flow.pkl'%roi_date
                


        #fp = 'XXXXXXX.pkl'            
        roi_mask,qval_dict = cpk.load( open(fp, 'rb' )  )  #for load the saved roi data
        #print(fp)
        #print(roi_mask)

## Gi_SAXS 
elif scat_geometry =='gi_saxs':    
    # static mask    
    ss = '4f48c2c9'
    ss = '9910469a'
    fp = '/XF11ID/analysis/2019_1/masks/uid=%s_roi_masks.pkl'%ss
    roi_masks,qval_dicts = cpk.load( open(fp, 'rb' )  )  #for load the saved roi data    
    print('The static mask is: %s.'%fp)    

    fp = '/XF11ID/analysis/2019_1/masks/uid=%s_roi_mask.pkl'%(ss)
    roi_mask,qval_dict = cpk.load( open(fp, 'rb' )  )  #for load the saved roi data
    print('The dynamic mask is: %s.'%fp)
    # q-map
    fp = '/XF11ID/analysis/2019_1/masks/uid=%s_qmap.pkl'%(ss)
    #print(fp)
    qr_map, qz_map, ticks, Qrs, Qzs,  Qr, Qz, inc_x0,refl_x0, refl_y0 = cpk.load( open(fp, 'rb' )  )
    print('The qmap is: %s.'%fp)
    
## WAXS 
elif scat_geometry =='gi_waxs': 
    #fp = '/XF11ID/analysis/2019_1/masks/uid=ca2ccb14_roi_mask_5PTO_130C_PTO.pkl'     
    fp =  '/XF11ID/analysis/2019_2/masks/uid=278c0df1_roi_mask.pkl'
    
    roi_mask,qval_dict = cpk.load( open(fp, 'rb' )  )  #for load the saved roi data

print(roi_mask.shape)


# In[40]:


qval_dict


# In[43]:


#roi_mask = shift_mask(roi_mask, 10,30)  #if shift mask to get new mask
show_img(roi_mask, aspect=1.0, image_name = fp)#, center=center[::-1]


# In[43]:


img_choice_N = 1
img_samp_index = random.sample( range(len(imgs)), img_choice_N) 
avg_img =  get_avg_img( imgsa, img_samp_index, plot_ = False, uid =uidstr)
if avg_img.max() == 0:
    print('There are no photons recorded for this uid: %s'%uid)
    print('The data analysis should be terminated! Please try another uid.')


# In[44]:


#show_img( imgsa[1000],  vmin=.1, vmax= 1e1, logs=True, aspect=1,
#         image_name= uidstr + '_img_avg',  save=True, path=data_dir,  cmap = cmap_albula )
print(center[::-1])


# In[45]:


show_img( imgsa[ 15],  vmin = -1, vmax = 20, logs=False, aspect=1, #save_format='tif',
         image_name= uidstr + '_img_avg',  save=True, path=data_dir, cmap=cmap_albula,center=center[::-1])
# select subregion, hard coded center beam location
#show_img( imgsa[180+40*3/0.05][110:110+840*2, 370:370+840*2],  vmin = 0.01, vmax = 20, logs=False, aspect=1, #save_format='tif',
#         image_name= uidstr + '_img_avg',  save=True, path=data_dir, cmap=cmap_albula,center=[845,839])


# ## Compress Data
# * Generate a compressed data with filename
# * Replace old mask with a new mask with removed hot pixels
# * Do average image
# * Do each image sum
# * Find badframe_list for where image sum above bad_pixel_threshold
# * Check shutter open frame to get good time series
# 

# In[46]:


compress=True
photon_occ = len( np.where(avg_img)[0] ) / ( imgsa[0].size)
#compress =  photon_occ < .4  #if the photon ocupation < 0.5, do compress
print ("The non-zeros photon occupation is %s."%( photon_occ))
print("Will " + 'Always ' + ['NOT', 'DO'][compress]  + " apply compress process.")


# In[47]:


if  md['detector'] =='eiger4m_single_image' or md['detector'] == 'image':    
    good_start =    5  #make the good_start at least 0
elif md['detector'] =='eiger500K_single_image': 
    good_start = 100  #5  #make the good_start at least 0
    
elif  md['detector'] =='eiger1m_single_image' or md['detector'] == 'image':    
    good_start =    5    


# In[48]:


bin_frame =  False # True  #generally make bin_frame as False
if bin_frame:
    bin_frame_number=4
    acquisition_period = md['acquire period']
    timeperframe = acquisition_period * bin_frame_number
else:
    bin_frame_number =1


# In[49]:


force_compress = False
#force_compress = True


# In[50]:


import time
t0= time.time()

if not use_local_disk:
    cmp_path = '/nsls2/xf11id1/analysis/Compressed_Data'
else:
    cmp_path = '/tmp_data/compressed'
    
    
cmp_path = '/nsls2/xf11id1/analysis/Compressed_Data'    
if bin_frame_number==1:   
    cmp_file = '/uid_%s.cmp'%md['uid']
else:
    cmp_file = '/uid_%s_bined--%s.cmp'%(md['uid'],bin_frame_number)
    
filename = cmp_path + cmp_file  
mask2, avg_img, imgsum, bad_frame_list = compress_eigerdata(imgs, mask, md, filename, 
         force_compress= force_compress,  para_compress= para_compress,  bad_pixel_threshold = 1e14,
                                    reverse=reverse, rot90=rot90,
                        bins=bin_frame_number, num_sub= 100, num_max_para_process= 500, with_pickle=True,
                        direct_load_data =use_local_disk, data_path = data_fullpath, )                                  
                                                         
min_inten = 10    
good_start = max(good_start, np.where( np.array(imgsum) > min_inten )[0][0] )    
print ('The good_start frame number is: %s '%good_start)

FD = Multifile(filename, good_start, len(imgs)//bin_frame_number )
#FD = MultifileBNLCustom(filename, good_start, len(imgs)//bin_frame_number )

###For test purpose to use the first 1000 frames
####################################################################
#FD = MultifileBNLCustom(filename, good_start, 500 )
#FD = Multifile(filename, good_start, 500)
############################################################


uid_ = uidstr + '_fra_%s_%s'%(FD.beg, FD.end)
print( uid_ )
plot1D( y = imgsum[ np.array( [i for i in np.arange(good_start, len(imgsum)) if i not in bad_frame_list])],
       title =uidstr + '_imgsum', xlabel='Frame', ylabel='Total_Intensity', legend='imgsum'   )
Nimg = Nimg/bin_frame_number
run_time(t0)

mask2 =  mask * pixel_mask * Chip_Mask
mask_copy = mask.copy()
mask_copy2 = mask.copy()

avg_img *= mask


# In[51]:


#%run ~/pyCHX_link/pyCHX/chx_generic_functions.py


# In[86]:


try:
    if md['experiment']=='printing':
        #p = md['printing'] #if have this printing key, will do error function fitting to find t_print0
        find_tp0 = True
        t_print0 = ps(  y = imgsum[:400] ) * timeperframe
        print( 'The start time of print: %s.' %(t_print0  ) )
    else:
        find_tp0 = False
        print('md[experiment] is not "printing" -> not going to look for t_0')
        t_print0 = None
except:
    find_tp0 = False
    print('md[experiment] is not "printing" -> not going to look for t_0')
    t_print0 = None


# In[87]:


show_img( avg_img*mask,  vmin=1e-3, vmax= 1e7, logs=True, aspect=1, #save_format='tif',
         image_name= uidstr + '_img_avg',  save=True, 
         path=data_dir, center=center[::-1], cmap = cmap_albula )


# ## Get bad frame list by a polynominal fit

# In[88]:


good_end= None # 2000  
if good_end is not None:
    FD = Multifile(filename, good_start, min( len(imgs)//bin_frame_number, good_end) )
    uid_ = uidstr + '_fra_%s_%s'%(FD.beg, FD.end)
    print( uid_ )
    


# In[89]:


re_define_good_start =False
if re_define_good_start:
    good_start = 180
    #good_end = 19700
    good_end = len(imgs)
    FD = Multifile(filename, good_start, good_end) 
    uid_ = uidstr + '_fra_%s_%s'%(FD.beg, FD.end)
    print( FD.beg, FD.end)


# In[90]:


bad_frame_list =  get_bad_frame_list( imgsum, fit='both',  plot=True,polyfit_order = 30,                                      
                        scale= 3.5,  good_start = good_start, good_end=good_end, uid= uidstr, path=data_dir)

print( 'The bad frame list length is: %s'%len(bad_frame_list) )


# ### Creat new mask by masking the bad pixels and get new avg_img

# In[91]:


imgsum_y = imgsum[ np.array( [i for i in np.arange( len(imgsum)) if i not in bad_frame_list])]
imgsum_x = np.arange( len( imgsum_y))
save_lists(  [imgsum_x, imgsum_y], label=['Frame', 'Total_Intensity'],
           filename=uidstr + '_img_sum_t', path= data_dir  )


# ### Plot time~ total intensity of each frame

# In[92]:


plot1D( y = imgsum_y, title = uidstr + '_img_sum_t', xlabel='Frame', c='b',
       ylabel='Total_Intensity', legend='imgsum', save=True, path=data_dir)


# ## Get Dynamic Mask (currently designed for 500K)

# In[93]:


if  md['detector'] =='eiger4m_single_image' or md['detector'] == 'image':    
    pass
elif md['detector'] =='eiger500K_single_image':  
    #if md['cam_acquire_period'] <= 0.00015:  #will check this logic
    if imgs[0].dtype == 'uint16':
        print('Create dynamic mask for 500K due to 9K data acquistion!!!')
        bdp = find_bad_pixels_FD( bad_frame_list, FD, img_shape = avg_img.shape, threshold=20 )    
        mask = mask_copy2.copy()
        mask *=bdp 
        mask_copy = mask.copy()
        show_img(  mask, image_name='New Mask_uid=%s'%uid )


# # Static Analysis

# ## SAXS Scattering Geometry

# In[94]:


setup_pargs 


# In[95]:


if scat_geometry =='saxs':
    ## Get circular average| * Do plot and save q~iq
    mask = mask_copy.copy()
    hmask = create_hot_pixel_mask( avg_img, threshold = 1e8, center=center, center_radius= 10)
    
    qp_saxs, iq_saxs, q_saxs = get_circular_average( avg_img * Chip_Mask , mask * hmask, save=True,
                                                    pargs=setup_pargs  )
    plot_circular_average( qp_saxs, iq_saxs, q_saxs,  pargs=setup_pargs,  show_pixel=True,
                      xlim=[qp_saxs.min(), qp_saxs.max()*1.0], ylim = [iq_saxs.min(), iq_saxs.max()*2] )
    mask =np.array( mask * hmask, dtype=bool) 


# In[96]:


#qval_dict


# In[97]:


uid


# In[98]:


if scat_geometry =='saxs':    
    if run_fit_form:        
        form_res = fit_form_factor( q_saxs,iq_saxs,  guess_values={'radius': 2500, 'sigma':0.05, 
         'delta_rho':1E-10 },  fit_range=[0.0001, 0.015], fit_variables={'radius': T, 'sigma':T, 
         'delta_rho':T},  res_pargs=setup_pargs, xlim=[0.0001, 0.015])  
        
    qr = np.array( [qval_dict[k][0] for k in sorted( qval_dict.keys())] )
    if qphi_analysis ==  False:
        try:
            qr_cal, qr_wid = get_QrQw_From_RoiMask( roi_mask, setup_pargs  ) 
            print(len(qr))         
            if (qr_cal - qr).sum() >=1e-3:
                print( 'The loaded ROI mask might not be applicable to this UID: %s.'%uid)
                print('Please check the loaded roi mask file.')
        except:
            print('Something is wrong with the roi-mask. Please check the loaded roi mask file.')            
            
    
    show_ROI_on_image( avg_img*roi_mask, roi_mask, center, label_on = False, rwidth = 840, alpha=.9,  
                 save=True, path=data_dir, uid=uidstr, vmin= 1e-3,
                 vmax= 1e-1, #np.max(avg_img),
                 aspect=1,
                 show_roi_edge=True,     
                 show_ang_cor = True) 
    plot_qIq_with_ROI( q_saxs, iq_saxs, np.unique(qr), logs=True, uid=uidstr, 
                      xlim=[q_saxs.min(), q_saxs.max()*1.02],#[0.0001,0.08],
                  ylim = [iq_saxs.min(), iq_saxs.max()*1.02],  save=True, path=data_dir)
    
    roi_mask = roi_mask * mask


# # Time Depedent I(q) Analysis

# In[99]:


if scat_geometry =='saxs':
    Nimg = FD.end - FD.beg 
    time_edge = create_time_slice( Nimg, slice_num= 10, slice_width= 1, edges = None )
    time_edge =  np.array( time_edge ) + good_start
    #print( time_edge )    
    qpt, iqst, qt = get_t_iqc( FD, time_edge, mask*Chip_Mask, pargs=setup_pargs, nx=1500, show_progress= False )
    plot_t_iqc( qt, iqst, time_edge, pargs=setup_pargs, xlim=[qt.min(), qt.max()],
           ylim = [iqst.min(), iqst.max()], save=True )


# In[103]:


if run_invariant_analysis:
    if scat_geometry =='saxs':
        invariant = get_iq_invariant( qt, iqst )
        time_stamp = time_edge[:,0] * timeperframe

    if scat_geometry =='saxs':
        plot_q2_iq( qt, iqst, time_stamp,pargs=setup_pargs,ylim=[ -0.001, 0.01] , 
                   xlim=[0.007,0.2],legend_size= 6  )

    if scat_geometry =='saxs':
        plot_time_iq_invariant( time_stamp, invariant, pargs=setup_pargs,  )

    if False:
        iq_int = np.zeros( len(iqst) )
        fig, ax = plt.subplots()
        q = qt
        for i in range(iqst.shape[0]):
            yi = iqst[i] * q**2
            iq_int[i] = yi.sum()
            time_labeli = 'time_%s s'%( round(  time_edge[i][0] * timeperframe, 3) )
            plot1D( x = q, y = yi, legend= time_labeli, xlabel='Q (A-1)', ylabel='I(q)*Q^2', title='I(q)*Q^2 ~ time',
                   m=markers[i], c = colors[i], ax=ax, ylim=[ -0.001, 0.01] , xlim=[0.007,0.2],
                  legend_size=4)

        #print( iq_int )


# # GiSAXS Scattering Geometry

# In[104]:


if scat_geometry =='gi_saxs':    
    plot_qzr_map(  qr_map, qz_map, inc_x0, ticks = ticks, data= avg_img, uid= uidstr, path = data_dir   )


# ## Static Analysis for gisaxs

# In[105]:


if scat_geometry =='gi_saxs':    
    #roi_masks, qval_dicts = get_gisaxs_roi( Qrs, Qzs, qr_map, qz_map, mask= mask )
    show_qzr_roi( avg_img, roi_masks, inc_x0, ticks[:4], alpha=0.5, save=True, path=data_dir, uid=uidstr )


# In[106]:


if  scat_geometry =='gi_saxs':    
    Nimg = FD.end - FD.beg 
    time_edge = create_time_slice( N= Nimg, slice_num= 3, slice_width= 2, edges = None )
    time_edge =  np.array( time_edge ) + good_start
    print( time_edge )    
    qrt_pds = get_t_qrc( FD, time_edge, Qrs, Qzs, qr_map, qz_map, mask=mask, path=data_dir, uid = uidstr )    
    plot_qrt_pds( qrt_pds, time_edge, qz_index = 0, uid = uidstr, path =  data_dir )


# # Make a Profile Plot

# In[107]:


if  scat_geometry =='gi_saxs':
    if run_profile_plot:
        xcorners= [ 1100, 1250, 1250, 1100 ]
        ycorners= [ 850, 850, 950, 950 ]   
        waterfall_roi_size = [ xcorners[1] - xcorners[0],  ycorners[2] - ycorners[1]  ]
        waterfall_roi =  create_rectangle_mask(  avg_img, xcorners, ycorners   )
        #show_img( waterfall_roi * avg_img,  aspect=1,vmin=.001, vmax=1, logs=True, )
        wat = cal_waterfallc( FD, waterfall_roi, qindex= 1, bin_waterfall=True,
                              waterfall_roi_size = waterfall_roi_size,save =True, path=data_dir, uid=uidstr)
    


# In[108]:


if  scat_geometry =='gi_saxs':
    if run_profile_plot:
        plot_waterfallc( wat, qindex=1, aspect=None, vmin=1, vmax= np.max( wat), uid=uidstr, save =True, 
                        path=data_dir, beg= FD.beg)


# ## Dynamic Analysis for gi_saxs

# In[109]:


if scat_geometry =='gi_saxs':       
    show_qzr_roi( avg_img, roi_mask, inc_x0, ticks[:4], alpha=0.5, save=True, path=data_dir, uid=uidstr )        
    ## Get 1D Curve (Q||-intensity¶)
    qr_1d_pds = cal_1d_qr( avg_img, Qr, Qz, qr_map, qz_map, inc_x0= None, mask=mask, setup_pargs=setup_pargs )
    plot_qr_1d_with_ROI( qr_1d_pds, qr_center=np.unique( np.array(list( qval_dict.values() ) )[:,0] ),
                    loglog=True, save=True, uid=uidstr, path = data_dir)


# # GiWAXS Scattering Geometry

# In[110]:


if scat_geometry =='gi_waxs':
    #badpixel = np.where( avg_img[:600,:] >=300 )
    #roi_mask[badpixel] = 0
    show_ROI_on_image( avg_img, roi_mask, label_on = True,  alpha=.5,
                 save=True, path=data_dir, uid=uidstr, vmin=0.1, vmax=500)


# * Extract the labeled array

# In[111]:


qind, pixelist = roi.extract_label_indices(roi_mask)
noqs = len(np.unique(qind))
print(noqs)


# * Number of pixels in each q box

# In[112]:


nopr = np.bincount(qind, minlength=(noqs+1))[1:]
nopr


# ## Check one ROI intensity

# In[113]:


roi_inten = check_ROI_intensity( avg_img, roi_mask, ring_number= 3, uid =uidstr ) #roi starting from 1


# ## Do a waterfall analysis

# In[114]:


qth_interest = 1 #the second ring. #qth_interest starting from 1
if scat_geometry =='saxs' or scat_geometry =='gi_waxs' or scat_geometry == 'gi_saxs':
    if run_waterfall:    
        wat = cal_waterfallc( FD, roi_mask, qindex= qth_interest, save =True, path=data_dir, uid=uidstr)
        plot_waterfallc( wat, qth_interest, aspect= None, vmin=1e-1, vmax= wat.max(), uid=uidstr, save =True, 
                        path=data_dir, beg= FD.beg, cmap = cmap_vge )
 


# In[115]:


#show_img(roi_mask)


# In[116]:


ring_avg = None    
if run_t_ROI_Inten:
    times_roi, mean_int_sets = cal_each_ring_mean_intensityc(FD, roi_mask, timeperframe = None, multi_cor= True )#False )#True  ) 
    mean_int_setsF = np.zeros( [FD.end, mean_int_sets.shape[1] ] )
    mean_int_setsF[FD.beg:FD.end] =mean_int_sets
    plot_each_ring_mean_intensityc( times_roi, mean_int_sets,  uid = uidstr, save=True, path=data_dir )
    roi_avg = np.average( mean_int_sets, axis=0)


# ## Analysis for mass center of reflective beam center

# In[117]:


if run_get_mass_center:
    cx, cy = get_mass_center_one_roi(FD, roi_mask, roi_ind=25)
    


# In[118]:


if run_get_mass_center:
    fig,ax=plt.subplots(2)
    plot1D( cx, m='o', c='b',ax=ax[0], legend='mass center-refl_X', 
           ylim=[940, 960], ylabel='posX (pixel)')
    plot1D( cy, m='s', c='r',ax=ax[1], legend='mass center-refl_Y', 
           ylim=[1540, 1544], xlabel='frames',ylabel='posY (pixel)')


# ## One time Correlation
# 
# Note : Enter the number of buffers for Muliti tau one time correlation
# number of buffers has to be even. More details in https://github.com/scikit-beam/scikit-beam/blob/master/skbeam/core/correlation.py

# ### if define another good_series

# In[119]:


good_start_g2 = FD.beg
good_end_g2 = FD.end
define_good_series = False
#define_good_series = True
if define_good_series:
    good_start_g2 = 399
    good_end_g2 = 1486
    FD = Multifile(filename, beg = good_start_g2, end = good_end_g2) #end=1000)
    uid_ = uidstr + '_fra_%s_%s'%(FD.beg, FD.end)
    print( uid_ )


# In[120]:


#use_SG = False # True #False #True
#use_SG_bin_frames = True


# In[121]:


if use_sqnorm:#for transmision SAXS
    norm = get_pixelist_interp_iq( qp_saxs, iq_saxs, roi_mask, center)
    print('Using circular average in the normalization of G2 for SAXS scattering.')
elif use_SG:#for Gi-SAXS or WAXS
    avg_imgf = sgolay2d( avg_img, window_size= 11, order= 5) * mask
    norm=np.ravel(avg_imgf)[pixelist]    
    print('Using smoothed image by SavitzkyGolay filter in the normalization of G2.')      
elif use_SG_bin_frames:#for Gi-SAXS or WAXS
    avg_imgf = sgolay2d( avg_img, window_size= 11, order= 5) * mask
    norm_avg=np.ravel(avg_imgf)[pixelist]    
    #print('Using smoothed image by SavitzkyGolay filter in the normalization of G2.')
    bins_number = 4 
    norm = get_SG_norm( FD, pixelist, bins=bins_number, mask=mask, window_size= 11, order=5 )   
    print('Using smoothed bined: %s frames  by SavitzkyGolay filter in the normalization of G2.'%bins_number)     
else:     
    norm= None
    print('Using simple (average) normalization of G2.')      

if use_imgsum_norm:
    imgsum_ = imgsum
    print('Using frame total intensity for intensity normalization in g2 calculation.')      
else:
    imgsum_ = None    
import time


# In[122]:


#np.save( data_dir + 'norm_SG_per_frame',norm)
#norm = np.load(  data_dir + 'norm_SG_per_frame.npy'   )


# In[123]:


#%run -i ~/pyCHX_link/pyCHX/chx_correlationc.py


# In[124]:


if run_one_time: 
    t0 = time.time()     
    if cal_g2_error:          
        g2,lag_steps,g2_err = cal_g2p(FD,roi_mask,bad_frame_list,good_start, num_buf = 8,
                            num_lev= None,imgsum= imgsum_, norm=norm, cal_error= True )
    else:   
        g2,lag_steps    =     cal_g2p(FD,roi_mask,bad_frame_list,good_start, num_buf = 8,
                            num_lev= None,imgsum= imgsum_, norm=norm, cal_error= False )

    run_time(t0)
    


# In[125]:


lag_steps = lag_steps[:g2.shape[0]]
g2.shape[1]


# In[126]:


if run_one_time:
    
    taus = lag_steps * timeperframe         
    try:
        g2_pds = save_g2_general( g2, taus=taus,qr= np.array( list( qval_dict.values() ) )[:g2.shape[1],0],
                                            qz = np.array( list( qval_dict.values() ) )[:g2.shape[1],1],
                             uid=uid_+'_g2.csv', path= data_dir, return_res=True )
    except:
        g2_pds = save_g2_general( g2, taus=taus,qr= np.array( list( qval_dict.values() ) )[:g2.shape[1],0],                                             
                             uid=uid_+'_'+q_mask_name+'_g2.csv', path= data_dir, return_res=True )   
    if cal_g2_error:    
        try:
            g2_err_pds = save_g2_general( g2_err, taus=taus,qr= np.array( list( qval_dict.values() ) )[:g2.shape[1],0],
                                                qz = np.array( list( qval_dict.values() ) )[:g2.shape[1],1],
                                 uid=uid_+'_g2_err.csv', path= data_dir, return_res=True )
        except:
            g2_err_pds = save_g2_general( g2_err, taus=taus,qr= np.array( list( qval_dict.values() ) )[:g2.shape[1],0],                                             
                                 uid=uid_+'_'+q_mask_name+'_g2_err.csv', path= data_dir, return_res=True )         
    


# # Fit g2

# In[127]:


from scipy.special import erf


# In[128]:


def get_short_long_labels_from_qval_dict(qval_dict, geometry='saxs'):
    '''Y.G. 2016, Dec 26
        Get short/long labels from a qval_dict
        Parameters
        ----------  
        qval_dict, dict, with key as roi number,
                        format as {1: [qr1, qz1], 2: [qr2,qz2] ...} for gi-saxs
                        format as {1: [qr1], 2: [qr2] ...} for saxs
                        format as {1: [qr1, qa1], 2: [qr2,qa2], ...] for ang-saxs
        geometry:
            'saxs':  a saxs with Qr partition
            'ang_saxs': a saxs with Qr and angular partition
            'gi_saxs': gisaxs with Qz, Qr            
    '''

    Nqs = len( qval_dict.keys())
    len_qrz = len( list( qval_dict.values() )[0] )
    #qr_label = sorted( np.array( list( qval_dict.values() ) )[:,0] )
    qr_label =  np.array( list( qval_dict.values() ) )[:,0] 
    if geometry=='gi_saxs' or geometry=='ang_saxs':# or geometry=='gi_waxs':
        if len_qrz < 2:
            print( "please give qz or qang for the q-label")
        else:
            #qz_label = sorted( np.array( list( qval_dict.values() ) )[:,1]  )
            qz_label =   np.array( list( qval_dict.values() ) )[:,1]  
    else:
        qz_label = np.array(   [0]    ) 
        
    uqz_label = np.unique( qz_label )
    num_qz = len( uqz_label)
    
    uqr_label = np.unique( qr_label )
    num_qr = len( uqr_label)       
    
    #print( uqr_label, uqz_label )
    if len( uqr_label ) >=  len( uqz_label ):
        master_plot= 'qz'  #one qz for many sub plots of each qr 
    else:
        master_plot= 'qr' 

    mastp=  master_plot    
    if geometry == 'ang_saxs':
        mastp= 'ang'   
    num_short = min(num_qz, num_qr)
    num_long =  max(num_qz, num_qr)
    
    #print( mastp, num_short, num_long)
    if num_qz != num_qr:
        short_label = [qz_label,qr_label][ np.argmin( [num_qz, num_qr]    ) ]
        long_label  = [qz_label,qr_label][ np.argmax( [num_qz, num_qr]    ) ]
        short_ulabel = [uqz_label,uqr_label][ np.argmin( [num_qz, num_qr]    ) ]
        long_ulabel  = [uqz_label,uqr_label][ np.argmax( [num_qz, num_qr]    ) ]
    else:
        short_label = qz_label
        long_label  = qr_label
        short_ulabel = uqz_label
        long_ulabel  = uqr_label        
    #print( long_ulabel )    
    #print( qz_label,qr_label )
    #print( short_label, long_label ) 
        
    if geometry == 'saxs' or geometry == 'gi_waxs':
        ind_long = [ range( num_long )  ] 
    else:
        ind_long = [ np.where( short_label == i)[0] for i in short_ulabel ] 
        
        
    if Nqs  == 1:
        long_ulabel = list( qval_dict.values() )[0]
        long_label = list( qval_dict.values() )[0]
    return qr_label, qz_label, num_qz, num_qr, num_short,num_long, short_label, long_label,short_ulabel,long_ulabel, ind_long, master_plot, mastp


# In[129]:


(qr_label, qz_label, num_qz, num_qr, num_short,
num_long, short_label, long_label,short_ulabel,
long_ulabel,ind_long, master_plot,
mastp) = get_short_long_labels_from_qval_dict(qval_dict,
geometry='ang_saxs')
#long_label


# In[130]:


def flow_coated( x, beta, Diffusion,flow_velocity,baseline,):
    (qr_label, qz_label, num_qz, num_qr, num_short,num_long, short_label, long_label,short_ulabel,
    long_ulabel,ind_long, master_plot,
    mastp) = get_short_long_labels_from_qval_dict(qval_dict,
    geometry='ang_saxs')
    num_rings = g2.shape[1]
    #s_ulabel=short_ulabel.shape[0]
    result=[]
    import math
    for s_ind in range( num_short ):
        ind_long_i = ind_long[ s_ind ]
        q_rings=short_ulabel[s_ind]
        #for i in range(s_ulabel):
         #   q_rings=short_ulabel[i]
        for i, l_ind in enumerate( ind_long_i ):
                regions=long_label[l_ind]
                #print(regions)
                phi=abs(math.cos(math.radians(regions)-math.radians(90)))
                Diff_part= np.exp(-2 *q_rings**2 * x*Diffusion)
                Flow_part= np.pi**2/(16*x*q_rings*flow_velocity*phi) * abs(erf( np.sqrt( 4/np.pi * 1j* x *q_rings*flow_velocity*phi ) ) )**2
                #result.append(beta*Diff_part * Flow_part + baseline)

    return beta*Diff_part * Flow_part + baseline


# In[131]:


import pandas as pd


# In[132]:


def newflow_qphi( x, beta, Diffusion,flow_velocity,baseline,rings=1,regions=1):
    (qr_label, qz_label, num_qz, num_qr, num_short,num_long, short_label, long_label,short_ulabel,
    long_ulabel,ind_long, master_plot,
    mastp) = get_short_long_labels_from_qval_dict(qval_dict,
    geometry='ang_saxs')
    num_rings = g2.shape[1]
    import math
    #abs(np.radians( qval_dict[i][1] - ang_init) )
    phi=abs(math.cos(math.radians(regions)-math.radians(90)))
    Diff_part= np.exp(-2 *rings**2 * x*Diffusion)
    #Flow_part= np.pi**2/(16*x*rings*flow_velocity*phi) * abs(erf( np.sqrt( 4/np.pi * 1j* x *rings*flow_velocity*phi ) ) )**2
    Flow_part= np.pi/(8*rings*flow_velocity*x)* abs(  erf((1+1j)/2*  np.sqrt(   4* rings*flow_velocity*x ) ) )**2 
    return beta*Diff_part * Flow_part + baseline


# In[133]:


def g2_global_fit_general( g2, taus,  function='simple_exponential',sequential_fit=False, 
                       qval_dict = None, ang_init = 90, *argv,**kwargs):
  
    
    if 'fit_range' in kwargs.keys():
        fit_range = kwargs['fit_range'] 
    else:
        fit_range=None    

        
    num_rings = g2.shape[1]  
    #print(num_rings)
    if 'fit_variables' in kwargs:
        additional_var  = kwargs['fit_variables']        
        _vars =[ k for k in list( additional_var.keys()) if additional_var[k] is False]
    else:
        _vars = []        
    if function=='simple_exponential' or function=='simple':
        _vars = np.unique ( _vars + ['alpha']) 
        mod = Model(stretched_auto_corr_scat_factor)#,  independent_vars= list( _vars)   )        
    elif function=='flow_coated' :
        mod=Model(flow_coated)
    elif function=='newflow_qphi':
        mod=Model(newflow_qphi)
        
    else:
        print ("The %s is not supported.The supported functions include simple_exponential and stretched_exponential"%function)    
        
    mod.set_param_hint( 'baseline',   min=0.5, max= 2.5 )
    mod.set_param_hint( 'beta',   min=0.0,  max=1.0 )
    mod.set_param_hint( 'alpha',   min=0.0 )
    mod.set_param_hint( 'relaxation_rate',   min=0.0,  max= 1000  )  
    
    if 'guess_limits' in kwargs:         
        guess_limits  = kwargs['guess_limits']         
        for k in list(  guess_limits.keys() ):
            mod.set_param_hint( k,   min=   guess_limits[k][0], max= guess_limits[k][1] )           
  
    if function=='flow_coated':
        mod.set_param_hint( 'flow_velocity', min=0 , max=1e7)
        mod.set_param_hint( 'Diffusion', min=0 ,max=1e7)
        
    if function=='newflow_qphi':
        mod.set_param_hint( 'flow_velocity', min=0 , max=1e7)
        mod.set_param_hint( 'Diffusion', min=1.0 ,max=1e7)   
        
    _guess_val = dict( beta=.1, alpha=1.0, relaxation_rate =0.005, baseline=1.0,
                    )    
    if 'guess_values' in kwargs:         
        guess_values  = kwargs['guess_values']         
        _guess_val.update( guess_values )  
   
    _beta=_guess_val['beta']
    _alpha=_guess_val['alpha']
    _relaxation_rate = _guess_val['relaxation_rate']
    _baseline= _guess_val['baseline']    
    pars  = mod.make_params( beta=_beta, alpha=_alpha, relaxation_rate =_relaxation_rate, baseline= _baseline)
    if function=='flow_para_function' or  function=='flow_para':
        _flow_velocity =_guess_val['flow_velocity']    
        pars  = mod.make_params( beta=_beta, alpha=_alpha, flow_velocity=_flow_velocity,
                                relaxation_rate =_relaxation_rate, baseline= _baseline)
        
        
    if  function=='flow_vibration':
        _flow_velocity =_guess_val['flow_velocity']    
        _freq =_guess_val['freq'] 
        _amp = _guess_val['amp'] 
        pars  = mod.make_params( beta=_beta,  freq=_freq, amp = _amp,flow_velocity=_flow_velocity,
                                relaxation_rate =_relaxation_rate, baseline= _baseline) 
    if  function=='flow_coated' :
        _flow_velocity =_guess_val['flow_velocity']    
        _Diffusion =_guess_val['Diffusion']
        _beta =_guess_val['beta']
        _baseline =_guess_val['baseline']
        
        pars  = mod.make_params( beta=_beta,  flow_velocity=_flow_velocity,
                                Diffusion =_Diffusion, baseline= _baseline) 
        
    if  function=='newflow_qphi' :
        _flow_velocity =_guess_val['flow_velocity']    
        _Diffusion =_guess_val['Diffusion']
        _beta =_guess_val['beta']
        _baseline =_guess_val['baseline']
        _guess_val['rings']   = 1
        _guess_val['regions']  = 0
        
        pars  = mod.make_params( beta=_beta,  flow_velocity=_flow_velocity,
                                Diffusion =_Diffusion, baseline= _baseline,rings=1,regions=0)    
       
       
   
        
    for v in _vars:
        pars['%s'%v].vary = False
    #print( pars )
    fit_res = []
    model_data = []    
    for i in range(num_rings):  
        if fit_range is not None:
            y=g2[1:, i][fit_range[0]:fit_range[1]]
            lags=taus[1:][fit_range[0]:fit_range[1]] 
        else: 
            y=g2[1:, i] 
            lags=taus[1:]     
        #print( _relaxation_rate )
        for k in list(pars.keys()):
            #print(k, _guess_val[k]  )
            if isinstance( _guess_val[k], (np.ndarray, list) ):
                pars[k].value = _guess_val[k][i]  
            if False:
                if isinstance( _beta, (np.ndarray, list) ):
                    pars['beta'].value = _guess_val['beta'][i]      
                if isinstance( _baseline, (np.ndarray, list) ):
                    pars['baseline'].value = _guess_val['baseline'][i]                
                if isinstance( _relaxation_rate, (np.ndarray, list) ):
                    pars['relaxation_rate'].value = _guess_val['relaxation_rate'][i]               
                if isinstance( _alpha, (np.ndarray, list) ):
                     pars['alpha'].value = _guess_val['alpha'][i] 
                if isinstance( _flow_velocity, (np.ndarray, list) ):
                     pars['flow_velocity'].value = _guess_val['flow_velocity'][i] 
                if isinstance( _Diffusion, (np.ndarray, list) ):
                     pars['Diffusion'].value = _guess_val['Diffusion'][i]    
          
            if function=='newflow_qphi':  
                if qval_dict is None:
                    print("Please provide qval_dict, a dict with qr and ang (in unit of degrees).")
                else:
                    pars  = mod.make_params(  
                           beta=_beta, flow_velocity=_flow_velocity,
                               Diffusion=_Diffusion, baseline= _baseline, 
                          rings = qval_dict[i][0], regions =  qval_dict[i][1] )
                    pars['rings'].vary = False
                    pars['regions'].vary = False
                  
                    
             
                
        #if function=='newflow_qphi':  
         #   if qval_dict is None:
          #      print("Please provide qval_dict, a dict with qr and ang (in unit of degrees).")
           # else:
            #    pars  = mod.make_params(  
             #       beta=_beta, flow_velocity=_flow_velocity,
              #                 Diffusion=_Diffusion, baseline= _baseline, 
               #     rings = qval_dict[i][0], regions =  qval_dict[i][1] )
                #pars['rings'].vary = False
                #pars['regions'].vary = False
                
          
              
                #pars['beta'].vary = False
            #if False:
             #   if isinstance( _beta, (np.ndarray, list) ):
              #          pars['beta'].value = _guess_val['beta'][i]
                
                #pars['beta'].value = _guess_val['beta'][i]
                
            #for k in list(pars.keys()):
                #print(k, _guess_val[k]  )
            # pars[k].value = _guess_val[k][i]  
         
        result1 = mod.fit(y, pars, x =lags ) 
        if sequential_fit:
            for k in list(pars.keys()):
                #print( pars )
                if k in list(result1.best_values.keys()):
                    pars[k].value = result1.best_values[k]  
                
        fit_res.append( result1) 
        model_data.append(  result1.best_fit )
    return fit_res, lags, np.array( model_data ).T


# In[134]:


if run_one_time:
    if scat_geometry =='ang_saxs' or 'saxs':
        fit_g2_func ='newflow_qphi' #for parallel
    g2_fit_result, taus_fit, g2_fit = g2_global_fit_general( g2, taus,function = fit_g2_func, vlim=[0.95, 1.05], fit_range= None,
    qval_dict=qval_dict,fit_variables={'baseline':True, 'beta':True,'Diffusion':True,'flow_velocity':True},
    guess_values={'baseline':1.0,'beta':0.2,'Diffusion':2e5,'flow_velocity':1.25e5, }
    )
g2_fit_paras = save_g2_fit_para_tocsv(g2_fit_result, filename= uid_+'_g2_fit_paras.csv', path=data_dir )


# In[135]:


RUN_GUI =False
from scipy.interpolate import interp1d


# In[160]:


def plot_g2_global_general( g2_dict, taus_dict, qval_dict, fit_res=None,  geometry='saxs',filename='g2', 
                    path=None, function='simple_exponential',  g2_labels=None, 
                    fig_ysize= 12, qth_interest = None,
                    ylabel='g2',  return_fig=False, append_name='', outsize=(2000, 2400), 
                    max_plotnum_fig=16, figsize=(10, 12), show_average_ang_saxs=True,
                    qphi_analysis = True,
                    *argv,**kwargs):    
   

    if ylabel=='g2':
        ylabel='g_2'
    if ylabel=='g4':
        ylabel='g_4' 
        
    if geometry =='saxs':
        if qphi_analysis:
            geometry = 'ang_saxs'
    (qr_label, qz_label, num_qz, num_qr, num_short,
     num_long, short_label, long_label,short_ulabel,
     long_ulabel,ind_long, master_plot,
     mastp) = get_short_long_labels_from_qval_dict(qval_dict, geometry=geometry)  
    fps = [] 
    num_rings=g2.shape[1]
    #$print( num_short, num_long )
    
    #data=flattened_data.reshape(num_rings,len(taus[1:]))
    
    for s_ind in range( num_short  ):
        ind_long_i = ind_long[ s_ind ]
        num_long_i = len( ind_long_i )
        #print(num_long_i)
        #if show_average_ang_saxs:
        #    if geometry=='ang_saxs':
        #        num_long_i += 1        
        if RUN_GUI:
            fig = Figure(figsize=(10, 12))            
        else:
            #fig = plt.figure( )
            if num_long_i <=4:
                if master_plot != 'qz':
                    fig = plt.figure(figsize=(8, 6))   
                else:
                    if num_short>1:
                        fig = plt.figure(figsize=(8, 4))
                    else:
                        fig = plt.figure(figsize=(10, 6))
                    #print('Here')
            elif num_long_i > max_plotnum_fig:
                num_fig = int(np.ceil(num_long_i/max_plotnum_fig)) #num_long_i //16
                #print(num_fig)
                fig = [ plt.figure(figsize=figsize)  for i in range(num_fig) ]
                #print( figsize )
            else:
                #print('Here')
                if master_plot != 'qz':
                    fig = plt.figure(figsize=figsize)
                else:
                    fig = plt.figure(figsize=(10, 10))
        
        if master_plot == 'qz':
            if geometry=='ang_saxs':
                title_short = 'Angle= %.2f'%( short_ulabel[s_ind] )  + r'$^\circ$'                               
            elif geometry=='gi_saxs':
                title_short = r'$Q_z= $' + '%.4f'%( short_ulabel[s_ind] ) + r'$\AA^{-1}$'
            else:
                title_short = ''            
        else: #qr
            if geometry=='ang_saxs' or geometry=='gi_saxs':
                title_short =   r'$Q_r= $' + '%.5f  '%( short_ulabel[s_ind] ) + r'$\AA^{-1}$'            
            else:
                title_short=''        
                
        #filename =''
        til = '%s:--->%s'%(filename,  title_short )

        if num_long_i <=4:            
            plt.title( til,fontsize= 14, y =1.15)  
        else:
            plt.title( til,fontsize=20, y =1.06) 
            #plt.title( til,fontsize=14, y =1.15)
        
        #print( num_long )
        
        if num_long!=1:   
            #print( 'here')
            plt.axis('off')             
            #sy =   min(num_long_i,4) 
            #sy=3
            sy =   min(num_long_i,  int( np.ceil( min(max_plotnum_fig,num_long_i)/4))   ) 
            #print(sy)
            #fig.set_size_inches(10, 12)
            #fig.set_size_inches(10, fig_ysize )
        else: 
            sy =1
            #fig.set_size_inches(8,6)
            
            #plt.axis('off') 
        sx = min(4, int( np.ceil( min(max_plotnum_fig,num_long_i)/float(sy) ) ))
        
        temp = sy
        sy = sx
        sx = temp
        
        #print( num_long_i, sx, sy )
        #print( master_plot )
        #print(ind_long_i, len(ind_long_i) )
        
        for i, l_ind in enumerate( ind_long_i ):  
            if num_long_i <= max_plotnum_fig:
                #print('Here')
                #print(i, l_ind,long_label[l_ind] )                
                ax = fig.add_subplot(sx,sy, i + 1 ) 
            else:
                #fig_subnum = l_ind//max_plotnum_fig
                #ax = fig[fig_subnum].add_subplot(sx,sy, i + 1 - fig_subnum*max_plotnum_fig) 
                fig_subnum = i//max_plotnum_fig
                #print(fig_subnum)
                #print(  i, sx,sy, fig_subnum, max_plotnum_fig, i + 1 - fig_subnum*max_plotnum_fig )
                ax = fig[fig_subnum].add_subplot(sx,sy, i + 1 - fig_subnum*max_plotnum_fig) 
                
                
                  
            ax.set_ylabel( r"$%s$"%ylabel + '(' + r'$\tau$' + ')' ) 
            ax.set_xlabel(r"$\tau $ $(s)$", fontsize=16)         
            if master_plot == 'qz' or master_plot == 'angle':                 
                title_long =  r'$Q_r= $'+'%.5f  '%( long_label[l_ind]  ) + r'$\AA^{-1}$'  
                #print(  long_label   )
            else:             
                if geometry=='ang_saxs':
                    title_long = 'Ang= ' + '%.2f'%(  long_label[l_ind] ) #+ r'$^\circ$' + '( %d )'%(l_ind)
                    #chisquare=np.sum((g2[1:,l_ind]-g2_fit[:,l_ind])**2/g2_fit[:,l_ind])
                    
                    #goodness_fit=np.sum((g2[1:,l_ind]-g2glob_fit[:,l_ind])**2/g2glob_fit[:,l_ind])
                    ax.set_title(title_long, y =1.1, fontsize=12)
                elif geometry=='gi_saxs':
                    title_long =   r'$Q_z= $'+ '%.5f  '%( long_label[l_ind]  ) + r'$\AA^{-1}$'                  
                else:
                    title_long = ''    

            if master_plot != 'qz':
                ax.set_title(title_long + ' (%s  )'%(1+l_ind), y =1.1, fontsize=12) 
            else:                
                ax.set_title(title_long + ' (%s  )'%(1+l_ind), y =1.05, fontsize=12) 
            
            #if not function=='flowglobal'
            for ki, k in enumerate( list(g2_dict.keys()) ):                
                if ki==0:
                    c='b'
                    if fit_res is None:
                        m='-o'                        
                    else:
                        m='o'                        
                elif ki==1:
                    c='g'
                    if fit_res is None:
                        m='s'                        
                    else:
                        m='-'                
                elif ki==2:
                    c='r'
                    m='-'
                else:
                    c = colors[ki+2]
                    m= '-%s'%markers[ki+2] 
                try:
                    dumy = g2_dict[k].shape
                    #print( 'here is the shape' )
                    islist = False 
                except:
                    islist_n = len( g2_dict[k] )
                    islist = True
                    #print( 'here is the list' )                    
                if islist:
                    for nlst in range( islist_n ):
                        m = '-%s'%markers[ nlst ]  
                        #print(m)
                        y=g2_dict[k][nlst][:, l_ind ]
                        x = taus_dict[k][nlst]
                        if ki==0:
                            ymin,ymax = min(y), max(y[1:])
                        if g2_labels is None:                             
                            ax.semilogx(x, y, m, color=c,  markersize=6) 
                        else:
                            #print('here ki ={} nlst = {}'.format( ki, nlst ))
                            if nlst==0:
                                ax.semilogx(x, y, m,  color=c,markersize=6, label=g2_labels[ki]) 
                            else:
                                ax.semilogx(x, y, m,  color=c,markersize=6)
                                
                        if nlst==0:
                            if l_ind==0:
                                ax.legend(loc='best', fontsize = 8, fancybox=True, framealpha=0.5)             
                
                else:    
                    y=g2_dict[k][:, l_ind ] 
                    #print(l_ind)
                    x = taus_dict[k]
                    if ki==0:
                        ymin,ymax = min(y), max(y[1:])
                    if g2_labels is None:    
                        ax.semilogx(x, y, m, color=c,  markersize=6) 
                    else:
                        ax.semilogx(x, y, m,  color=c,markersize=6, label=g2_labels[ki]) 
                        if l_ind==0:
                            ax.legend(loc='best', fontsize = 8, fancybox=True, framealpha=0.5)  
            #else:
                

            if fit_res is not None:
                #result1 = fit_res[l_ind] 
                result1= fit_res
                
               
                #
                if function=='simple_exponential' or function=='simple':
                    alpha =1.0 
                elif function=='stretched_exponential' or function=='stretched':
                    alpha = result1.best_values['alpha']
                elif function=='stretched_vibration': 
                    alpha = result1.best_values['alpha']
                    freq = result1.best_values['freq'] 
                elif function=='flow_vibration': 
                    freq = result1.best_values['freq'] 
                if function=='flow_para_function' or  function=='flow_para' or  function=='flow_vibration': 
                    flow = result1.best_values['flow_velocity'] 
                     
                if function=='flowglobal':
                    flow = fit_res[0].best_values['flow_velocity']
                    Diffusion = fit_res[0].best_values['Diffusion']      
                    
                if function=='flow_coated' :
                    flow = result1.best_values['flow_velocity']
                    Diffusion = result1.best_values['Diffusion']
                    beta = result1.best_values['beta'] 
                    baseline =  result1.best_values['baseline'] 
                
                if function=='newflow_qphi' :
                    flow = result1.best_values['flow_velocity']
                    Diffusion = result1.best_values['Diffusion']
                    beta = result1.best_values['beta'] 
                    baseline =  result1.best_values['baseline'] 
                 

              
                x=0.1
                y0=0.9
                fontsize = 10
                dt=0
             
                    
                
    
                from decimal import Decimal
                if function=='flow_coated' :
                    j=flow
                    k=Decimal(str(j))
                    l='{:.2e}'.format(Decimal(str(j)))
                #print(l)
                    txts = r'$v$' + r'$ = %.3$'+' %s '%(l) + r'$\AA^{}/s$'
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize,transform=ax.transAxes)
                    #if function=='flow_coated' or 'newflow_qphi':
                    n= Diffusion
                    o='{:.2e}'.format(Decimal(str(n)))
                    txts = r'$D$' + r'$ = %.3$'+' %s '%(o) +r'$\AA^{2}/s$'
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize,transform=ax.transAxes)
                    
                    txts = r'$baseline$' + r'$ = %.3f$'%( baseline) 
                    dt +=0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                    
                    txts = r'$\beta$' + r'$ = %.3f$'%( beta ) 
                    dt +=0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                    
                if function=='newflow_qphi' :
                    j=flow
                    k=Decimal(str(j))
                    l='{:.2e}'.format(Decimal(str(j)))
                #print(l)
                    txts = r'$v$' + r'$ = %.3$'+' %s '%(l) + r'$\AA^{}/s$'
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize,transform=ax.transAxes)
                    #if function=='flow_coated' or 'newflow_qphi':
                    n= Diffusion
                    o='{:.2e}'.format(Decimal(str(n)))
                    txts = r'$D$' + r'$ = %.3$'+' %s '%(o) +r'$\AA^{2}/s$'
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize,transform=ax.transAxes)
                    
                    txts = r'$baseline$' + r'$ = %.3f$'%( baseline) 
                    dt +=0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                    
                    txts = r'$\beta$' + r'$ = %.3f$'%( beta ) 
                    dt +=0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                
                
                if function=='flowglobal' :
                    j=flow
                    k=Decimal(str(j))
                    l='{:.2e}'.format(Decimal(str(j)))
                    txts = r'$v$' + r'$ = %.3$'+' %s '%(l) + r'$\AA^{}/s$'
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize,transform=ax.transAxes)
                if function=='flowglobal' :
                    n= Diffusion
                    o='{:.2e}'.format(Decimal(str(n)))
                    txts = r'$D_0$' + r'$ = %.3$'+' %s '%(o) + r'$\AA^{2}/s$'
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize,transform=ax.transAxes)
                if function=='flowglobal' :
                    goodness_fit=np.sum((g2[1:,l_ind]-g2glob_fit[:,l_ind])**2/g2glob_fit[:,l_ind])
                    txts = r'$\chi^2$' + r'$ = %.3$'+' %s '%round(goodness_fit,4)
                    dt += 0.1
                    ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize,transform=ax.transAxes)
                 
                #txts = r'$\beta$' + r'$ = %.3f$'%( beta ) 
                #dt +=0.1
                #ax.text(x =x, y= y0- dt, s=txts, fontsize=fontsize, transform=ax.transAxes)
                

            if 'ylim' in kwargs:
                ax.set_ylim( kwargs['ylim'])
            elif 'vlim' in kwargs:
                vmin, vmax =kwargs['vlim']
                ax.set_ylim([ymin*vmin, ymax*vmax ]) 
                
            else:
                pass
            if 'xlim' in kwargs:
                ax.set_xlim( kwargs['xlim'])
        if   num_short == 1: 
            fp = path + filename 
        else:
            fp = path + filename + '_%s_%s'%(mastp, s_ind)   
            
        if append_name is not '':
            fp = fp + append_name
        fps.append( fp  + '.png' )  
        #if num_long_i <= 16:
        if num_long_i <= max_plotnum_fig:            
            fig.set_tight_layout(True)    
            #fig.tight_layout()
            plt.savefig( fp + '.png', dpi=fig.dpi) 
        else:
            fps=[]
            for fn, f in enumerate(fig):
                f.set_tight_layout(True)
                fp = path + filename + '_q_%s_%s'%(fn*16, (fn+1)*16) 
                if append_name is not '':
                    fp = fp + append_name
                fps.append( fp  + '.png' )  
                f.savefig( fp + '.png', dpi=f.dpi)
        #plt.savefig( fp + '.png', dpi=fig.dpi)        
    #combine each saved images together
    
    if (num_short !=1) or (num_long_i > 16):
        outputfile =  path + filename + '.png'
        if append_name is not '':
            outputfile =  path + filename  + append_name + '__joint.png'
        else:
            outputfile =  path + filename   + '__joint.png'
        combine_images( fps, outputfile, outsize= outsize )    
    if return_fig:
        return fig 

              


# In[139]:


plot_g2_global_general( g2_dict={1:g2, 2:g2_fit}, taus_dict={1:taus, 2:taus_fit},vlim=[0.95, 1.15],qval_dict = qval_dict, fit_res= g2_fit_result,
geometry= 'ang_saxs',max_plotnum_fig=20,filename= uid_+'_g2',path= data_dir, function= fit_g2_func, ylabel='g2',fig_ysize =12,
append_name= '_fit',)


# # Global fitting

# In[140]:


flow_vel = np.array( [ g2_fit_result[i].params['flow_velocity'].value for i in range( g2.shape[1] )])


# In[141]:


Diffusion = np.array( [ g2_fit_result[i].params['Diffusion'].value for i in range( g2.shape[1] )])


# In[142]:


beta = np.array( [ g2_fit_result[i].params['beta'].value for i in range( g2.shape[1] )])


# In[143]:


baseline= np.array( [ g2_fit_result[i].params['baseline'].value for i in range( g2.shape[1] )])


# In[144]:


np.mean(Diffusion)


# In[145]:


np.mean(flow_vel)


# In[146]:


flattened_data=g2[1:].T.flatten()


# In[147]:


time_values=np.array([taus[1:]]*len(qval_dict))
flattened_time_values=time_values.flatten()


# In[148]:


beta_values=np.array([beta[:]]*len(taus[1:])).T
flattened_beta_values=beta_values.flatten()


# In[149]:


baseline_values=np.array([baseline[:]]*len(taus[1:])).T
flattened_baseline_values=baseline_values.flatten()


# In[150]:


import pandas as pd
region_values=np.array([pd.DataFrame(qval_dict).T[1]]*len(taus[1:])).T
flattened_region_values=region_values.flatten()


# In[151]:


q_ring_values=np.array([pd.DataFrame(qval_dict).T[:][0]]*len(taus[1:])).T
flattened_q_ring_values=q_ring_values.flatten()


# In[152]:


def flowglobal( x,Diffusion,flow_velocity):
    import math
    #baseline_values_globalfunction=flattened_baselinevalues;
    #beta_values_globalfunction=flattened_betavalues;
    #regions=flattened_regionvalues;
    phi=abs(np.cos(np.radians(flattened_region_values)-np.radians(90.0)));
    Diff_part= np.exp(-2 *flattened_q_ring_values**2 *x*Diffusion)
    
    #print(Diff_part)
    #Flow_part= np.pi**2/(16*x*flattened_q_ring_values*flow_velocity*phi) * abs( erf( np.sqrt( 4/np.pi * 1j* x *flattened_q_ring_values*flow_velocity *phi) ) )**2
    Flow_part= np.pi/(8*x*flattened_q_ring_values*flow_velocity*phi)* abs(  erf((1+1j)/2*  np.sqrt(   4*x* flattened_q_ring_values*flow_velocity*phi ) ) )**2 
    #print(Flow_part.shape)
    result=(flattened_beta_values*Diff_part * Flow_part) +flattened_baseline_values
    #print (result.shape)
    return result


# In[153]:


from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import Model
import math
fit_res=[]
model_data=[]
#gmodel.set_param_hint( 'Diffusion', min=1.00 ,max=1e6)

import matplotlib.pyplot as plt
gmodel=Model(flowglobal)
#gmodel.set_param_hint( 'Diffusion', min=1.00 ,max=1e6)
result=gmodel.fit(flattened_data, x=flattened_time_values,Diffusion=2e5,flow_velocity=1.05e5)
#res=result.minimize()
print(result.fit_report())


# In[154]:


def g2_second_fit( flattened_data, flattened_timevalues,  function='flowglobal', 
                       sequential_fit=False, *argv,**kwargs):
    from numpy import sqrt, pi, exp, linspace, loadtxt
    from lmfit import Model
    import math
    fit_res=[]
    model_data=[]
    num_rings=g2.shape[1]
    #data=flattened_data.reshape(num_rings,len(taus[1:]))
    import matplotlib.pyplot as plt#
    gmodel = Model(flowglobal)
    result = gmodel.fit(flattened_data, x=flattened_timevalues,Diffusion=2e5,flow_velocity=1e5)
    fit_res.append( result)
    model_data.append( result.best_fit )
    fit_values=np.array([model_data]).reshape((num_rings,len(taus[1:])))
    
    return fit_res,taus,fit_values.T


# In[155]:


if run_one_time:
    if scat_geometry =='ang_saxs' or 'saxs':
        fit_g2_func ='flowglobal' #for parallel
    g2_globfit_result, tausglob_fit, g2glob_fit = g2_second_fit(flattened_data, flattened_time_values,function = fit_g2_func, vlim=[0.95, 1.05], fit_range= None,
       fit_variables={'Diffusion':True,'flow_velocity':True,},
    guess_values={'Diffusion':2e5,'flow_velocity':1e5, }
    )
g2glob_fit_paras = save_g2_fit_para_tocsv(g2_globfit_result, filename= uid_+'_g2glob_fit_paras.csv', path=data_dir )


# In[156]:


fit_g2_func


# In[157]:


g2_globfit_result[0].best_values['flow_velocity']


# In[158]:


g2glob_fit.shape


# In[159]:


g2[1:].shape


# In[161]:


plot_g2_global_general( g2_dict={1:g2, 2:g2_fit ,3:g2glob_fit }, taus_dict={1:taus, 2:taus_fit, 3:taus_fit},vlim=[0.95, 1.15],qval_dict = qval_dict, fit_res= g2_globfit_result, geometry= 'ang_saxs',
                  filename= uid_+'_g2'+'_secondfit',max_plotnum_fig=20,path= data_dir, function= fit_g2_func,ylabel='g2', fig_ysize =12,
append_name= '_fit',)


# In[149]:


uid


# In[ ]:


def get(g2_data,phi,figsize):
    R=len(phi)
    fig=plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 10,y=1.0 )
    for k in range(R):
        g2_data=g2[1:,phi[k]]
        g2fit=g2glob_fit[:,phi[k]]
        f = interp1d(taus[1:], g2fit,kind='cubic')
        xnew = np.linspace(taus[1], taus[-1], num=10000, endpoint=False)
        index=phi[k]
        sy=4 
        sx=3
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus[1:],g2_data,'bo',label='Experiment')
        ax.semilogx(xnew, f(xnew) ,'r-',label='Analytic')
        #ax.set_title('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 10,y=1.0 )
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(loc=1,prop={'size': 11})
        #ax.set_ylabel(r'$g^{(2)}$', fontsize=12)
        #ax.set_xlabel(r'time(s)',fontsize=10)
        if k==2:
            ax.set_xlabel(r'time(s)',fontsize=11)
        #ax.set_ylim([0,1.02])
        #ax.set_title('$\phi $=%.1f'% regions[index] + '$^\circ $')
        #ax.set_title(r'$g^{(2)}$ in the bulk flow direction')


# In[ ]:


phi=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,45])


# In[ ]:


get(g2,phi,figsize=(4.5,10))


# In[ ]:


#xhrange=np.arange(-0.7,0.05,0.05)


# In[265]:


Q=0.15 #experimentally measured


# In[266]:


def get_vel(Q):
    R=0.4 #inner radius
    A=np.pi*R**2
    vel=Q/A*1e-9/(1e-6*60)*1e10 #Avg_vel in A/s 
    return vel


# In[267]:


get_vel(Q)


# In[268]:


g2_globfit_result[0].best_values['flow_velocity']/get_vel(Q)


# In[ ]:


1e4*80e-3/(3600*np.pi*0.66**2*1e-6) #Q in micro meter/sec


# In[ ]:


1.6e-3*162386*2


# In[ ]:


(1.16e5-1.04e5)/1.16e5*100


# # For two-time

# In[ ]:


if run_four_time:
    plot_g2_general( g2_dict={1:g4}, taus_dict={1:taus4},vlim=[0.95, 1.05], qval_dict=qval_dict, fit_res= None, 
                geometry=scat_geometry_,filename=uid_+'_g4',path= data_dir,   ylabel='g4')


# # Speckle Visiblity

# In[ ]:


if run_xsvs:    
    max_cts = get_max_countc(FD, roi_mask )    
    #max_cts = 15 #for eiger 500 K
    qind, pixelist = roi.extract_label_indices(   roi_mask  )
    noqs = len( np.unique(qind) )
    nopr = np.bincount(qind, minlength=(noqs+1))[1:]
    #time_steps = np.array( utils.geometric_series(2,   len(imgs)   ) )
    time_steps = [0,1]  #only run the first two levels
    num_times = len(time_steps)    
    times_xsvs = exposuretime + (2**(  np.arange( len(time_steps) ) ) -1 ) * timeperframe   
    print( 'The max counts are: %s'%max_cts )


# ### Do historam 

# In[ ]:


if run_xsvs:
    if roi_avg is  None:
        times_roi, mean_int_sets = cal_each_ring_mean_intensityc(FD, roi_mask, timeperframe = None,  ) 
        roi_avg = np.average( mean_int_sets, axis=0)
    
    t0=time.time()
    spec_bins, spec_his, spec_std, spec_sum  =  xsvsp( FD, np.int_(roi_mask), norm=None,
                max_cts=int(max_cts+2),  bad_images=bad_frame_list, only_two_levels=True )    
    spec_kmean =  np.array(  [roi_avg * 2**j for j in  range( spec_his.shape[0] )] )
    run_time(t0)
    spec_pds =  save_bin_his_std( spec_bins, spec_his, spec_std, filename=uid_+'_spec_res.csv', path=data_dir ) 


# ### Do historam fit by negtive binominal function with maximum likehood method

# In[ ]:


if run_xsvs:    
    ML_val, KL_val,K_ = get_xsvs_fit(  spec_his, spec_sum, spec_kmean, 
                        spec_std, max_bins=2, fit_range=[1,60], varyK= False  )
    #print( 'The observed average photon counts are: %s'%np.round(K_mean,4))
    #print( 'The fitted average photon counts are: %s'%np.round(K_,4)) 
    print( 'The difference sum of average photon counts between fit and data are: %s'%np.round( 
            abs(np.sum( spec_kmean[0,:] - K_ )),4))
    print( '#'*30)
    qth=   0 
    print( 'The fitted M for Qth= %s are: %s'%(qth, ML_val[qth]) )
    print( K_[qth])
    print( '#'*30)


# ## Plot fit results

# In[ ]:


if run_xsvs:   
    qr = [qval_dict[k][0] for k in list(qval_dict.keys()) ]
    plot_xsvs_fit(  spec_his, ML_val, KL_val, K_mean = spec_kmean, spec_std=spec_std,
                  xlim = [0,10], vlim =[.9, 1.1],
        uid=uid_, qth= qth_interest, logy= True, times= times_xsvs, q_ring_center=qr, path=data_dir)
    
    plot_xsvs_fit(  spec_his, ML_val, KL_val, K_mean = spec_kmean, spec_std = spec_std,
                  xlim = [0,15], vlim =[.9, 1.1],
        uid=uid_, qth= None, logy= True, times= times_xsvs, q_ring_center=qr, path=data_dir )


# ### Get contrast

# In[ ]:


if run_xsvs:
    contrast_factorL = get_contrast( ML_val)
    spec_km_pds = save_KM(  spec_kmean, KL_val, ML_val, qs=qr, level_time=times_xsvs, uid=uid_, path = data_dir )
    #spec_km_pds


# ### Plot contrast with g2 results

# In[ ]:


if run_xsvs:    
    plot_g2_contrast( contrast_factorL, g2b, times_xsvs, tausb, qr, 
                     vlim=[0.8,1.2], qth = qth_interest, uid=uid_,path = data_dir, legend_size=14)

    plot_g2_contrast( contrast_factorL, g2b, times_xsvs, tausb, qr, 
                     vlim=[0.8,1.2], qth = None, uid=uid_,path = data_dir, legend_size=4)


# In[ ]:


#from chxanalys.chx_libs import cmap_vge, cmap_albula, Javascript


# # Export Results to a HDF5 File

# In[ ]:


md['mask_file']= mask_path + mask_name
md['roi_mask_file']= fp
md['mask'] = mask
#md['NOTEBOOK_FULL_PATH'] =  data_dir + get_current_pipeline_fullpath(NFP).split('/')[-1]
md['good_start'] = good_start
md['bad_frame_list'] = bad_frame_list
md['avg_img'] = avg_img
md['roi_mask'] = roi_mask
md['setup_pargs'] = setup_pargs
if scat_geometry == 'gi_saxs':        
    md['Qr'] = Qr
    md['Qz'] = Qz
    md['qval_dict'] = qval_dict
    md['beam_center_x'] =  inc_x0
    md['beam_center_y']=   inc_y0
    md['beam_refl_center_x'] = refl_x0
    md['beam_refl_center_y'] = refl_y0


elif scat_geometry == 'gi_waxs':
    md['beam_center_x'] =  center[1]
    md['beam_center_y']=  center[0]
else:
    md['qr']= qr
    #md['qr_edge'] = qr_edge
    md['qval_dict'] = qval_dict
    md['beam_center_x'] =  center[1]
    md['beam_center_y']=  center[0]            

md['beg'] = FD.beg
md['end'] = FD.end
md['qth_interest'] = qth_interest
md['metadata_file'] = data_dir + 'uid=%s_md.pkl'%uid
psave_obj(  md, data_dir + 'uid=%s_md.pkl'%uid ) #save the setup parameters
save_dict_csv( md,  data_dir + 'uid=%s_md.csv'%uid, 'w')


###############Only work for Randy
try:
    md['beg_OneTime'] = good_start_g2
    md['end_OneTime'] = good_end_g2
    print(md['beg_OneTime'])
    print(md['end_OneTime'])
except:
    pass
################################



Exdt = {} 
if scat_geometry == 'gi_saxs':  
    for k,v in zip( ['md', 'roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list', 'qr_1d_pds'], 
                [md,    roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list, qr_1d_pds] ):
        Exdt[ k ] = v
elif scat_geometry == 'saxs': 
    for k,v in zip( ['md', 'q_saxs', 'iq_saxs','iqst','qt','roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list'], 
                [md, q_saxs, iq_saxs, iqst, qt,roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list] ):
        Exdt[ k ] = v
elif scat_geometry == 'gi_waxs': 
    for k,v in zip( ['md', 'roi_mask','qval_dict','avg_img','mask','pixel_mask', 'imgsum', 'bad_frame_list'], 
                [md,       roi_mask, qval_dict, avg_img,mask,pixel_mask, imgsum, bad_frame_list] ):
        Exdt[ k ] = v
        
if run_waterfall:Exdt['wat'] =  wat
if run_t_ROI_Inten:Exdt['times_roi'] = times_roi;Exdt['mean_int_sets']=mean_int_sets
if run_one_time:
    if run_invariant_analysis:
        for k,v in zip( ['taus','g2','g2_fit_paras', 'time_stamp','invariant'], [taus,g2,g2_fit_paras,time_stamp,invariant] ):Exdt[ k ] = v
    else:
        for k,v in zip( ['taus','g2','g2_fit_paras'  ], [taus,g2,g2_fit_paras ] ):Exdt[ k ] = v
            
if run_two_time:
    for k,v in zip( ['tausb','g2b','g2b_fit_paras', 'g12b'], [tausb,g2b,g2b_fit_paras,g12b] ):Exdt[ k ] = v
    #for k,v in zip( ['tausb','g2b','g2b_fit_paras', ], [tausb,g2b,g2b_fit_paras] ):Exdt[ k ] = v    
if run_dose:
    for k,v in zip( [ 'taus_uids', 'g2_uids' ], [taus_uids, g2_uids] ):Exdt[ k ] = v
if run_four_time:
    for k,v in zip( ['taus4','g4'], [taus4,g4] ):Exdt[ k ] = v
if run_xsvs:
    for k,v in zip( ['spec_kmean','spec_pds','times_xsvs','spec_km_pds','contrast_factorL'], 
                   [ spec_kmean,spec_pds,times_xsvs,spec_km_pds,contrast_factorL] ):Exdt[ k ] = v 


# In[ ]:


export_xpcs_results_to_h5( 'uid=%s_%s_Res.h5'%(md['uid'],q_mask_name), data_dir, export_dict = Exdt )
#extract_dict = extract_xpcs_results_from_h5( filename = 'uid=%s_Res.h5'%md['uid'], import_dir = data_dir )


# In[ ]:


#g2npy_filename =  data_dir  + '/' + 'uid=%s_g12b.npy'%uid
#print(g2npy_filename)
#if os.path.exists( g2npy_filename):
#    print('Will delete this file=%s.'%g2npy_filename)
#    os.remove( g2npy_filename  )


# In[ ]:


#extract_dict = extract_xpcs_results_from_h5( filename = 'uid=%s_Res.h5'%md['uid'], import_dir = data_dir )


# In[ ]:


#extract_dict = extract_xpcs_results_from_h5( filename = 'uid=%s_Res.h5'%md['uid'], import_dir = data_dir )


# # Creat PDF Report

# In[ ]:


pdf_out_dir = os.path.join('/XF11ID/analysis/', CYCLE, username, 'Results/')

pdf_filename = "XPCS_Analysis_Report2_for_uid=%s%s%s.pdf"%(uid,pdf_version,q_mask_name)
if run_xsvs:
    pdf_filename = "XPCS_XSVS_Analysis_Report_for_uid=%s%s%s.pdf"%(uid,pdf_version,q_mask_name)


# In[ ]:


#%run /home/yuzhang/pyCHX_link/pyCHX/Create_Report.py


# In[ ]:


make_pdf_report( data_dir, uid, pdf_out_dir, pdf_filename, username, 
                    run_fit_form,run_one_time, run_two_time, run_four_time, run_xsvs, run_dose,
                report_type= scat_geometry, report_invariant= run_invariant_analysis,
               md = md )


# ## Attach the PDF report to Olog 

# In[ ]:


#%run  /home/yuzhang/chxanalys_link/chxanalys/chx_olog.py


# In[ ]:


if att_pdf_report:
    lops=0
    while lops<5:
        try:
            os.environ['HTTPS_PROXY'] = 'https://proxy:8888'
            os.environ['no_proxy'] = 'cs.nsls2.local,localhost,127.0.0.1'
            update_olog_uid_with_file( uid[:6], text='Add XPCS Analysis PDF Report', 
                              filename=pdf_out_dir + pdf_filename, append_name='_R1' )
            lops=5
        except:
            print("Failed to attach PDF report to Olog...try again in 10s")
            time.sleep(10)
            lops+=1
            if lops == 5:
                print("Failed final attempt to attach PDF to Olog!")


# ## Save the OVA image

# In[ ]:


#save_oavs= False #True


# In[ ]:


if save_oavs:
    os.environ['HTTPS_PROXY'] = 'https://proxy:8888'
    os.environ['no_proxy'] = 'cs.nsls2.local,localhost,127.0.0.1'
    update_olog_uid_with_file( uid[:6], text='Add OVA images', 
                              filename= data_dir + 'uid=%s_OVA_images.png'%uid, append_name='_img' )
       # except:


# # The End!

# # 
