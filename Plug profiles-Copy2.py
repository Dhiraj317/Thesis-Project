
# coding: utf-8

# # This pipeline consists of steps required to read saved data,obtain Plug Profile and plot flow profiles at different Fourier modes

# # Step:1-Reading data for obtaining experimental parameters

# In[2]:


#from chxanalys.chx_packages import *
from pyCHX.chx_packages import *
import matplotlib.pyplot as plt
#%matplotlib notebook
plt.rcParams.update({'figure.max_open_warning': 0})
import pandas as pds
from scipy.special import erf 
import scipy.special as sp
#%reset -f  #for clean up things in the memory


# In[3]:


uid='ad5b1d33'#count : 1 ['ad5b1d33'] (scan num: 13222) (Measurement: Frate=0.35 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='ad5b1d33-912f-4c7d-8ef3-76cdbd5cc693'#Q=0.35 at center
uid='e0789d35-4a7a-4058-a61d-466d4370af64'#Q=1 at center
#uid='e9716a17-767c-4360-acf6-b03d284f28cd'#Q=0.5 at center


data_dir = '/XF11ID/analysis/2019_2/dnandyala/Results/'+ uid
#data_dir = '/XF11ID/analysis/2017_3/dnandyala/Results/' + uid


# In[4]:


uid='ad5b1d33'
uid='e0789d35'
#uid='e9716a17'
fit_params=pds.read_csv(data_dir + '/' + 'uid='+ uid+'_fra_5_1000_g2_fit_paras.csv') 
data=pds.read_csv(data_dir + '/' + 'uid=' + uid+'_fra_5_1000_g2.csv')
taus=data['tau'][1:]


# In[5]:


import numba


# In[6]:


@numba.njit
def fast_double_sum(flow_vel,qphit):
    result = np.zeros_like(qphit)
    for i, vi in enumerate(flow_vel):
        for j, vj in enumerate(flow_vel):
            result += np.cos(qphit*(vj-vi))
    return result


# In[7]:


def g2_numeric(flow_vel,time,q_rings,regions,N):
    phi=abs(np.cos(np.radians(regions)-np.radians(90)))
    #phi=abs(np.cos(np.radians(regions)-np.radians(45)))
    qphi=np.dot(q_rings[:,None],phi[None,:])
    qphit=np.dot(qphi[:,:,None],time[None,:])
    doublesum = fast_double_sum(flow_vel, qphit)
    return doublesum/N**2


# In[8]:


def mse(g2_exp,g2_num):
    if len(g2_num)!=len(g2_exp):
        raise ValueError("data size does not match with size of numeric g2")
    diff=(g2_exp-g2_num)/g2_exp
    squared=diff**2
    #print(squared)
    mean_of_differences_squared = squared.mean()  
    #mean_of_differences_squared = np.sum(squared)  
    #root=np.sqrt(mean_of_differences_squared)
    return mean_of_differences_squared


# In[9]:


start=0
end=30
regions=np.unique(fit_params['regions'][start:end])
q_rings=np.unique(fit_params['rings'][start:end])
beta=np.array(fit_params['beta'][start:end])
baseline=np.array(fit_params['baseline'][start:end])
M=len(q_rings)


# In[10]:


baseline_values=np.array([baseline[:]]*len(taus)).T
flattened_baseline_values=baseline_values.flatten()
beta_values=np.array([beta[:]]*len(taus)).T
flattened_beta_values=beta_values.flatten()


# In[11]:


R=0.4
Np=50 #No.of divisions
pos=np.linspace(-R,R,Np) 
#vmax=4.8e4 #arbitrary average flow velocity
v0_fit=pds.read_csv(data_dir + '/' + 'uid=' + uid+'_fra_5_1000_g2glob_fit_paras.csv')['flow_velocity'][0]
#v0_expt=116050.4
v0=v0_fit#v0_expt #v0_fit
parabolic_prof=2*v0*(1-(pos/R)**2)
Diffusion=pds.read_csv(data_dir + '/' + 'uid=' + uid+'_fra_5_1000_g2glob_fit_paras.csv')['Diffusion'][0]


# In[12]:


v0


# In[13]:


def get_diff_part(Diffusion,taus,q_rings,regions):
    q_r=np.array([q_rings[:]]*len(regions)).T
    qr_squared=q_r**2
    flattened_qr= qr_squared.flatten()
    
    qr_t=np.dot(flattened_qr[:,None],taus[None,:])
    flattened_qrt=qr_t.flatten()
    Diff_part=np.exp(-2 *Diffusion*flattened_qrt)
    return Diff_part
    


# In[14]:


Diff=get_diff_part(Diffusion,taus,q_rings,regions)


# In[15]:


def get_analytic(q_rings,regions,time,vel):
    phi=abs(np.cos(np.radians(regions)-np.radians(90)))
    #phi=abs(np.cos(np.radians(regions)-np.radians(45)))
    qphi=np.dot(q_rings[:,None],phi[None,:])
    qphit=np.dot(qphi[:,:,None],time[None,:])
    Flow_part= np.pi/(8*qphit*vel)* abs(  erf((1+1j)/2*  np.sqrt(   4* qphit*vel) ) )**2 
    return Flow_part

#From Pawel's Thesis--Does not work
result = np.zeros_like(y)
for n in range(1,modes+1):
    for m in range(1,modes+1):
        #result += v0/4*np.sin(n*np.pi*y/w)*np.sin(m*np.pi*z/h)
        #result += v0/(n*m*((n/w)**2+(m/h)**2))*np.sin(n*np.pi*y/w)*np.sin(m*np.pi*z/h)
        result += 16*dp/(np.pi**4*mu*L)*1/(n*m*((n/w)**2+(m/h)**2))*np.sin(n*np.pi*y/w)*np.sin(m*np.pi*z/h)
        #result += 16*dp/(np.pi**4*mu*L)*1/(n*m*((n/w)**2+(m/h)**2))*np.cos(n*np.pi*y/w)*np.cos(m*np.pi*z/h)#Notes(Velocity distribution in a rectangular duct) from video lecture series
np.seterr(divide='ignore', invalid='ignore')
result=np.zeros_like(y)
for n in range(modes):
    result+=(16*dp*h**2)/(mu*L*np.pi**3)*(-1)**n/(2*n+1)**3*((-np.cosh((n+0.5)*np.pi*y/h)/(np.cosh((n+0.5)*np.pi*w/h))))*np.cos((n+0.5)*np.pi*z/h)
    #result+=(16*dp*h**2)/(mu*L*np.pi**3)*(-1)**n/(2*n+1)**3*((-sy.cosh((n+0.5)*np.pi*y/h)/(sy.cosh((n+0.5)*np.pi*w/h))))*sy.cos((n+0.5)*np.pi*z/h)
    #result+=v0*(-1)**n/(2*n+1)**3*((-np.cosh((n+0.5)*np.pi*y/h)/(np.cosh((n+0.5)*np.pi*w/h))))*np.cos((n+0.5)*np.pi*z/h)
EO= v0 + result
#new= v0*(1-(z/h)**2)+result
# # Generate flow profile for a pseudoplastic flow(n<1)

# In[16]:


n=0.25


# In[17]:


new_Np=25
x=np.linspace(-R,R,Np)
new_x=np.linspace(0,R,new_Np)


# In[18]:


pseudoplastic= (3*n+1)/(n+1)*v0*(1-(new_x/R)**(n+1/n))
#pseudoplastic= ((3*n+1)/(n+1))*v0*(1-(np.sign(x)*(np.abs(x))/R)**(n+1/n))


# In[19]:


other_side=np.flip(pseudoplastic,axis=0)


# In[20]:


pseudoplastic_flow=np.concatenate((other_side,pseudoplastic))


# In[21]:


fig,ax=plt.subplots()
ax.plot(x,parabolic_prof,'bo', label='parabolic flow')
ax.plot(x,pseudoplastic_flow,'r-',label='pseudoplastic flow')
ax.legend(loc=8,prop={'size': 12})
ax.yaxis.major.formatter.set_powerlimits((0,0))


# # Obtain g2's numerically(all regions and phi) using the pseudoplastic profile from above 

# In[22]:


from scipy.interpolate import interp1d


# In[23]:


g2_pseudo=g2_numeric(flow_vel=pseudoplastic_flow,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
g2num_pseudo=g2_pseudo*Diff*flattened_beta_values+flattened_baseline_values


# In[24]:


reshape_g2_pseudo=g2num_pseudo.reshape(len(q_rings),len(regions),len(taus))


# In[25]:


M=len(q_rings)


# In[26]:


New_taus=np.array([ 7*1e-4  ,  0.00268   ,  0.00402   ,  0.00536   ,  0.0067    ,
        0.00804   ,  0.00938   ,  0.01072   ,  0.0134    ,  0.01608   ,
        0.01876   ,  0.02144   ,  0.0268    ,  0.03216   ,  0.03752   ,
        0.04288   ,  0.0536    ,  0.06432   ,  0.07504   ,  0.08576   ,
        0.1072    ,  0.12864   ,  0.15008   ,  0.17151999,  0.21439999,
        0.25727999,  0.30015999,  0.34303999,  0.42879999,  0.51455998,
        0.60031998,  0.68607998,  0.85759997,  1.02911997])


# In[27]:


for j in range(M):
    g2_j=reshape_g2_pseudo[j,:,:]
    I=len(g2_j)
    #print(R)
    fig=plt.figure(figsize=(25, 20))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 70,y=1.0 )
    #plt.title('q_ring=%.5f' %q_rings[j]  +  '$\AA^{-1}$',fontsize= 15,y=1.05)

    #noise=reshape_g2_theoretical[j,:,:]
    for k in range(I):
        g2_pseudo=g2_j[k,:]
        #g2_noisedata=noise[k,:]
        #g2_parabolicfit=g2_parabolic[k,:]
        f = interp1d(New_taus, g2_pseudo,kind='cubic')
        xnew = np.linspace(New_taus[0],New_taus[-1], num=10000, endpoint=False)
        sy=4 
        sx=4
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(xnew, f(xnew) ,'g-',label='pseudoplastic flow')
        #ax.semilogx(taus, g2_plug ,'g-',label='plug')
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(loc=1,prop={'size': 12})
        ax.set_xlabel(r'time(s)',fontsize=15)
        ax.set_ylabel(r'$g^{(2)}(t)$', fontsize=15)
        #ax.set_ylim([1,1.20])
        ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $', fontsize =20, y=1.0)
        #fp = new_data_dir+ 'Numeric fit-' + 'q=%.5f' %q_rings[j]+'$\AA^{-1}$'+'.png'  
        #plt.savefig( fp, dpi=fig.dpi)


# # Extracting Fourier modes 

# In[28]:


modes=7


# In[29]:


mode=[]
for n in range(modes):
    #print(n)
    #An=2*v0*0.31831*np.sin(3.14159*(n+0.5))/(n+0.5)
    An=v0*0.63662*np.sin(3.14159*(n+0.5))/(n+0.5)
    mode.append(An)
    #print(An)


# In[30]:


EO_amplitudes=np.asarray(mode)


# In[31]:


EO_amplitudes


# In[32]:


def fourierSeries(x,amplitudes):
    profiles=[]
    for k in range(len(amplitudes)):
        #partialSums =amplitudes[k,0]
        partialSums=0
        Num_amp= len(amplitudes[0])
        for n in range(Num_amp):
            mode=amplitudes[k,n]
            partialSums= partialSums  + (mode* np.cos((n+0.5)*np.pi/R*x))
        #partialSums=abs(partialSums)
        #print(partialSums)
        profiles.append(partialSums)
    return profiles 


# In[33]:


EO_amplitudes[0]


# In[34]:


new_data_dir='/XF11ID/analysis/2019_2/dnandyala/Results/Numeric/%s/Plug profiles/'%uid


# In[35]:


new_data_dir


# # Fourier decomposition- One Mode

# In[36]:


C=1
chops=10
from itertools import product
#minX, maxX, = EO_amplitudes[0]-0.2*EO_amplitudes[0],EO_amplitudes[0]+0.2*EO_amplitudes[0]
minX, maxX, = EO_amplitudes[0]-C*EO_amplitudes[0],EO_amplitudes[0]+C*EO_amplitudes[0]
X= np.linspace(minX, maxX, chops)
Initial_points=np.array(list(product(X)))


# In[37]:


X


# In[38]:


Initial_points.shape


# In[39]:


Num_profile=len(Initial_points)
import random
#Initial_amplitudes=[random.choice(points)  for i in range(Num_profile)]
Initial_amplitudes=Initial_points
Initial_amplitudes=np.asarray(Initial_amplitudes)


# In[40]:


import time
start_time = time.time()
epsilon=1*1e-6
import operator
import pandas as pd
previous_value=[]
Errors_alliterations=[]
Global_chisquare=[]
count=0
R=0.4
x=np.linspace(-R,R,50)
#while True:
for a in range(1):
    fourier_profiles=fourierSeries(x=x,amplitudes=Initial_amplitudes)
    errors=[]
    for j,value in enumerate(fourier_profiles):
        flow_prof=fourier_profiles[j]
        #print(flow_prof)
        g2_NUM=g2_numeric(flow_vel=flow_prof,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        g2_num=g2_NUM*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_expt,g2_num=g2_num)
        error=mse(g2_exp=g2num_pseudo,g2_num=g2_num)
        errors.append(error)   

        
    d=dict(enumerate(errors[:]))

    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    #print(sorted_d)
    error_best_case=(sorted_d)[0][1] 
    Global_chisquare.append(error_best_case)
    #error_best_case=round(error_best_case,6)
    #best_profile=profiles[(sorted_d)[0][0]]
    #errors_all=list(d.values())
    errors_all=[t[1] for t in sorted_d[:]]
    Errors_alliterations.append(errors_all)
    #print((sorted_d)[1][0])
    best=sorted_d[0][0]
    
    #if Initial_amplitudes[best][0]>6e4 and Initial_amplitudes[best][0]<6.5e4:
     #   print(Initial_amplitudes[best][0])
      #  Initial_amplitudes[best][0]=parabola_amplitudes[0][0]
       # 
        #best_amp=np.array([[Initial_amplitudes[best][0],  Initial_amplitudes[best][1]]])
        #Num_profile=1
        #best_profile=fourierSeries(x=x,amplitudes=best_amp)
        #best_profile=np.asarray(best_profile).flatten()
        #g2_nume=g2_numeric(flow_vel=best_profile,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        #g2_num=g2_nume*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_theoretical,g2_num=g2_num,)
        #print(error)
        
    #else:
     #   Initial_amplitudes[best][0]=Initial_amplitudes[best][0]
      #  best_profile=fourier_profiles[(sorted_d)[0][0]]
       # print(error_best_case)
    #print(sorted_d)
    best_profile=fourier_profiles[(sorted_d)[0][0]]
    print(error_best_case) 
    
    #fig,ax=plt.subplots()
    #ax.plot(x,best_profile)
    #ax.set_title('$\chi^{2}$=%s'% )
    #ax.set_title('Best profile', size=16)
    #ax.set_xlabel('r/R', size=16)
    #ax.set_ylabel('Avg flow velocity ($\AA/s$)', size=16)
    #ax.yaxis.major.formatter.set_powerlimits((0,0))  
    


# In[41]:


new_data_dir


# In[42]:


fig,ax=plt.subplots(figsize=(6,4))
#x=np.linspace(-R,R,60)

xp=np.linspace(-R,R,50)
ax.plot(x, best_profile,'bo-',label='Numeric from algorithm ')
ax.plot(xp,pseudoplastic_flow,'g-',label='pseudoplastic flow')
#ax.plot(xp,parabolic,'g-',label='Poiseuille flow')
ax.legend(loc=0,prop={'size': 14})
ax.tick_params(axis="x", labelsize=13)
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel('r/R',fontsize=16)
ax.set_ylabel('Avg flow velocity ($\AA/s$)',fontsize=16)
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.set_title(r'Pseudoplastic profile-1 Mode', size =16)
#fp = new_data_dir+ 'Plug profile-1 Mode' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)

#ax.set_title('Newscore',size=16)
#ax.set_title('Expt data-Modes-4')
#fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 20,y=1.0 )
 #ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $')


# In[43]:


best_amp=sorted_d[0][0]


# In[44]:


best_amplitude=Initial_amplitudes[best_amp]


# In[45]:


best_amplitude


# In[46]:


EO_amplitudes


# In[47]:


Avgvel_EO=np.trapz(pseudoplastic_flow,x)
Q_EO=60*1e-7*np.pi*R**2*Avgvel_EO


# In[48]:


Avgvel_alg=np.trapz(best_profile,x)
Q_alg=60*1e-7*np.pi*R**2*Avgvel_alg #in mu liters/ hour
Ratio=Q_alg/Q_EO
Ratio


# In[49]:


g2_NUM=g2_numeric(flow_vel=best_profile,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
g2_num=g2_NUM*Diff*flattened_beta_values+flattened_baseline_values


# In[50]:


reshape_g2_num=g2_num.reshape(len(q_rings),len(regions),len(taus))


# In[51]:


for j in range(M):
    g2_j=reshape_g2_pseudo[j,:,:]
    I=len(g2_j)
    #print(R)
    fig=plt.figure(figsize=(25, 20))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.suptitle('One-Mode-'+'q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 70,y=1.0 )
    #plt.title('q_ring=%.5f' %q_rings[j]  +  '$\AA^{-1}$',fontsize= 15,y=1.05)

    numeric=reshape_g2_num[j,:,:]
    for k in range(I):
        g2_pseudo=g2_j[k,:]
        g2_nume=numeric[k,:]
        #g2_parabolicfit=g2_parabolic[k,:]
        f = interp1d(New_taus, g2_nume,kind='cubic')
        xnew = np.linspace(New_taus[0],New_taus[-1], num=10000, endpoint=False)
        sy=4 
        sx=4
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus, g2_pseudo ,'g^',label='pseudo plastic flow')
        ax.semilogx(xnew, f(xnew) ,'r-',label='numeric')
        
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(loc=1,prop={'size': 12})
        ax.set_xlabel(r'time(s)',fontsize=15)
        ax.set_ylabel(r'$g^{(2)}(t)$', fontsize=15)
        #ax.set_ylim([1,1.20])
        ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $', fontsize =20, y=1.0)
        #fp = new_data_dir+ 'Numeric fit-' + 'q=%.5f' %q_rings[j]+'$\AA^{-1}$'+'.png'  
        #plt.savefig( fp, dpi=fig.dpi)


# In[52]:


R


# # 3-Modes

# In[53]:


C=1
chops=10
from itertools import product
#minX, maxX, = EO_amplitudes[0]-0.2*EO_amplitudes[0],EO_amplitudes[0]+0.2*EO_amplitudes[0]
minX, maxX, = EO_amplitudes[0]-C*EO_amplitudes[0],EO_amplitudes[0]+C*EO_amplitudes[0] 
minY, maxY  = EO_amplitudes[1]-C*EO_amplitudes[1], EO_amplitudes[1]+C*EO_amplitudes[1]
minZ, maxZ = EO_amplitudes[2]-C*EO_amplitudes[2],EO_amplitudes[2]+C*EO_amplitudes[2]

X= np.linspace(minX, maxX, chops)
Y = np.linspace(minY, maxY, chops)
Z= np.linspace(minZ,maxZ,chops)
Initial_points=np.array(list(product(X,Y,Z)))
#Initial_points=np.array(list(product(X,Y,Z,K,M,P)))
#plt.plot(Initial_points[:,0],Initial_points[:,1],'yo')
#plt.plot(EO_amplitudes[0],EO_amplitudes[1],'go',)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.rc('xtick',labelsize=15)
#plt.rc('ytick',labelsize=18)
#plt.xlabel('A0 ($\AA^{}/s)$', size =20)
#plt.ylabel('A1 ($\AA^{}/s)$', size =20)


# In[54]:


X


# In[55]:


Initial_points.shape


# In[56]:


Num_profile=len(Initial_points)
import random
#Initial_amplitudes=[random.choice(points)  for i in range(Num_profile)]
Initial_amplitudes=Initial_points
Initial_amplitudes=np.asarray(Initial_amplitudes)


# In[57]:


import time
start_time = time.time()
epsilon=1*1e-6
import operator
import pandas as pd
previous_value=[]
Errors_alliterations=[]
Global_chisquare=[]
count=0
R=0.4
x=np.linspace(-R,R,50)
#while True:
for a in range(1):
    fourier_profiles=fourierSeries(x=x,amplitudes=Initial_amplitudes)
    errors=[]
    for j,value in enumerate(fourier_profiles):
        flow_prof=fourier_profiles[j]
        #print(flow_prof)
        g2_NUM=g2_numeric(flow_vel=flow_prof,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        g2_num=g2_NUM*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_expt,g2_num=g2_num)
        error=mse(g2_exp=g2num_pseudo,g2_num=g2_num)
        errors.append(error)   

        
    d=dict(enumerate(errors[:]))

    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    #print(sorted_d)
    error_best_case=(sorted_d)[0][1] 
    Global_chisquare.append(error_best_case)
    #error_best_case=round(error_best_case,6)
    #best_profile=profiles[(sorted_d)[0][0]]
    #errors_all=list(d.values())
    errors_all=[t[1] for t in sorted_d[:]]
    Errors_alliterations.append(errors_all)
    #print((sorted_d)[1][0])
    best=sorted_d[0][0]
    
    #if Initial_amplitudes[best][0]>6e4 and Initial_amplitudes[best][0]<6.5e4:
     #   print(Initial_amplitudes[best][0])
      #  Initial_amplitudes[best][0]=parabola_amplitudes[0][0]
       # 
        #best_amp=np.array([[Initial_amplitudes[best][0],  Initial_amplitudes[best][1]]])
        #Num_profile=1
        #best_profile=fourierSeries(x=x,amplitudes=best_amp)
        #best_profile=np.asarray(best_profile).flatten()
        #g2_nume=g2_numeric(flow_vel=best_profile,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        #g2_num=g2_nume*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_theoretical,g2_num=g2_num,)
        #print(error)
        
    #else:
     #   Initial_amplitudes[best][0]=Initial_amplitudes[best][0]
      #  best_profile=fourier_profiles[(sorted_d)[0][0]]
       # print(error_best_case)
    #print(sorted_d)
    best_profile=fourier_profiles[(sorted_d)[0][0]]
    print(error_best_case) 
    
    #fig,ax=plt.subplots()
    #ax.plot(x,best_profile)
    #ax.set_title('$\chi^{2}$=%s'% )
    #ax.set_title('Best profile', size=16)
    #ax.set_xlabel('r/R', size=16)
    #ax.set_ylabel('Avg flow velocity ($\AA/s$)', size=16)
    #ax.yaxis.major.formatter.set_powerlimits((0,0))  
    


# In[58]:


(sorted_d)[0][0]


# In[59]:


fig,ax=plt.subplots(figsize=(6,4))
#x=np.linspace(-R,R,60)

xp=np.linspace(-R,R,50)
ax.plot(x, best_profile,'bo-',label='Numeric from algorithm ')
ax.plot(xp,pseudoplastic_flow,'g-',label='pseudoplastic flow')
#ax.plot(xp,parabolic,'g-',label='Poiseuille flow')
ax.legend(loc=0,prop={'size': 14})
ax.tick_params(axis="x", labelsize=13)
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel('r/R',fontsize=16)
ax.set_ylabel('Avg flow velocity ($\AA/s$)',fontsize=16)
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.set_title(r'pseudoplastic flow profile-3 Modes', size =16)
#fp = new_data_dir+ 'Plug profile-3 Modes' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)

#ax.set_title('Newscore',size=16)
#ax.set_title('Expt data-Modes-4')
#fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 20,y=1.0 )
 #ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $')


# In[60]:


best_amp=sorted_d[0][0]


# In[61]:


best_amplitude=Initial_amplitudes[best_amp]


# In[62]:


best_amplitude


# In[63]:


g2_NUM=g2_numeric(flow_vel=best_profile,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
g2_num=g2_NUM*Diff*flattened_beta_values+flattened_baseline_values


# In[64]:


EO_amplitudes


# In[65]:


Avgvel_alg=np.trapz(best_profile,x)
Q_alg=60*1e-7*np.pi*R**2*Avgvel_alg #in mu liters/ hour
Ratio=Q_alg/Q_EO
Ratio


# In[66]:


reshape_g2_num=g2_num.reshape(len(q_rings),len(regions),len(taus))


# In[67]:


for j in range(M):
    g2_j=reshape_g2_pseudo[j,:,:]
    I=len(g2_j)
    #print(R)
    fig=plt.figure(figsize=(25, 20))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.suptitle('3-Modes-'+'q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 70,y=1.0 )
    #plt.title('q_ring=%.5f' %q_rings[j]  +  '$\AA^{-1}$',fontsize= 15,y=1.05)

    numeric=reshape_g2_num[j,:,:]
    for k in range(I):
        g2_pseudo=g2_j[k,:]
        g2_nume=numeric[k,:]
        #g2_parabolicfit=g2_parabolic[k,:]
        f = interp1d(New_taus, g2_nume,kind='cubic')
        xnew = np.linspace(New_taus[0],New_taus[-1], num=10000, endpoint=False)
        sy=4 
        sx=4
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus, g2_pseudo ,'g^',label='pseudoplastic flow')
        ax.semilogx(xnew, f(xnew) ,'r-',label='numeric')
        
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(loc=1,prop={'size': 12})
        ax.set_xlabel(r'time(s)',fontsize=15)
        ax.set_ylabel(r'$g^{(2)}(t)$', fontsize=15)
        #ax.set_ylim([1,1.20])
        ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $', fontsize =20, y=1.0)
        #fp = new_data_dir+ '3 Modes-'+'Numeric fit-' + 'q=%.5f' %q_rings[j]+'$\AA^{-1}$'+'.png'  
        #plt.savefig( fp, dpi=fig.dpi)


# # 5 Modes

# In[68]:


C=1
chops=10
from itertools import product
#minX, maxX, = EO_amplitudes[0]-0.2*EO_amplitudes[0],EO_amplitudes[0]+0.2*EO_amplitudes[0]
minX, maxX, = EO_amplitudes[0]-C*EO_amplitudes[0],EO_amplitudes[0]+C*EO_amplitudes[0]
minY, maxY  = EO_amplitudes[1]-C*EO_amplitudes[1], EO_amplitudes[1]+C*EO_amplitudes[1]
minZ, maxZ = EO_amplitudes[2]-C*EO_amplitudes[2],EO_amplitudes[2]+C*EO_amplitudes[2]
minK, maxK = EO_amplitudes[3]-C*EO_amplitudes[3],EO_amplitudes[3]+C*EO_amplitudes[3]
minM, maxM = EO_amplitudes[4]-C*EO_amplitudes[4],EO_amplitudes[4]+C*EO_amplitudes[4]
#minL, maxL = EO_amplitudes[5]-C*EO_amplitudes[5],EO_amplitudes[5]+C*EO_amplitudes[5]
#minO, maxO = EO_amplitudes[6]-C*EO_amplitudes[6],EO_amplitudes[6]+C*EO_amplitudes[6]

X= np.linspace(minX, maxX, chops)
Y = np.linspace(minY, maxY, chops)
Z= np.linspace(minZ,maxZ,chops)
K= np.linspace(minK,maxK,chops)
M= np.linspace(minM,maxM,chops)
#L= np.linspace(minL,maxL,chops)
#O= np.linspace(minO,maxO,chops)

Initial_points=np.array(list(product(X,Y,Z,K,M)))
#Initial_points=np.array(list(product(X,Y,Z,K)))


# In[69]:


Initial_points.shape


# In[70]:


Num_profile=len(Initial_points)
import random
#Initial_amplitudes=[random.choice(points)  for i in range(Num_profile)]
Initial_amplitudes=Initial_points
Initial_amplitudes=np.asarray(Initial_amplitudes)


# In[71]:


import time
start_time = time.time()
epsilon=1*1e-6
import operator
import pandas as pd
previous_value=[]
Errors_alliterations=[]
Global_chisquare=[]
count=0
R=0.4
x=np.linspace(-R,R,50)
#while True:
for a in range(1):
    fourier_profiles=fourierSeries(x=x,amplitudes=Initial_amplitudes)
    errors=[]
    for j,value in enumerate(fourier_profiles):
        flow_prof=fourier_profiles[j]
        #print(flow_prof)
        g2_NUM=g2_numeric(flow_vel=flow_prof,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        g2_num=g2_NUM*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_expt,g2_num=g2_num)
        error=mse(g2_exp=g2num_pseudo,g2_num=g2_num)
        errors.append(error)   

        
    d=dict(enumerate(errors[:]))

    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    #print(sorted_d)
    error_best_case=(sorted_d)[0][1] 
    Global_chisquare.append(error_best_case)
    #error_best_case=round(error_best_case,6)
    #best_profile=profiles[(sorted_d)[0][0]]
    #errors_all=list(d.values())
    errors_all=[t[1] for t in sorted_d[:]]
    Errors_alliterations.append(errors_all)
    #print((sorted_d)[1][0])
    best=sorted_d[0][0]
    
    #if Initial_amplitudes[best][0]>6e4 and Initial_amplitudes[best][0]<6.5e4:
     #   print(Initial_amplitudes[best][0])
      #  Initial_amplitudes[best][0]=parabola_amplitudes[0][0]
       # 
        #best_amp=np.array([[Initial_amplitudes[best][0],  Initial_amplitudes[best][1]]])
        #Num_profile=1
        #best_profile=fourierSeries(x=x,amplitudes=best_amp)
        #best_profile=np.asarray(best_profile).flatten()
        #g2_nume=g2_numeric(flow_vel=best_profile,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        #g2_num=g2_nume*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_theoretical,g2_num=g2_num,)
        #print(error)
        
    #else:
     #   Initial_amplitudes[best][0]=Initial_amplitudes[best][0]
      #  best_profile=fourier_profiles[(sorted_d)[0][0]]
       # print(error_best_case)
    #print(sorted_d)
    best_profile=fourier_profiles[(sorted_d)[0][0]]
    print(error_best_case) 
    
    #fig,ax=plt.subplots()
    #ax.plot(x,best_profile)
    #ax.set_title('$\chi^{2}$=%s'% )
    #ax.set_title('Best profile', size=16)
    #ax.set_xlabel('r/R', size=16)
    #ax.set_ylabel('Avg flow velocity ($\AA/s$)', size=16)
    #ax.yaxis.major.formatter.set_powerlimits((0,0))  
    


# In[72]:


best_amp=sorted_d[0][0]


# In[73]:


best_amplitude=Initial_amplitudes[best_amp]


# In[74]:


best_amplitude


# In[75]:


g2_NUM=g2_numeric(flow_vel=best_profile,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
g2_num=g2_NUM*Diff*flattened_beta_values+flattened_baseline_values


# In[76]:


fig,ax=plt.subplots(figsize=(6,4))
#x=np.linspace(-R,R,60)

xp=np.linspace(-R,R,50)
ax.plot(x, best_profile,'bo-',label='Numeric from algorithm ')
ax.plot(xp,pseudoplastic_flow,'g-',label='pseudoplastic ')
#ax.plot(xp,parabolic,'g-',label='Poiseuille flow')
ax.legend(loc=0,prop={'size': 14})
ax.tick_params(axis="x", labelsize=13)
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel('r/R',fontsize=16)
ax.set_ylabel('Avg flow velocity ($\AA/s$)',fontsize=16)
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.set_title(r'Pseudoplastic profile-5 Modes', size =16)
#fp = new_data_dir+ 'Plug profile-6 Modes' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)

#ax.set_title('Newscore',size=16)
#ax.set_title('Expt data-Modes-4')
#fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 20,y=1.0 )
 #ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $')


# In[77]:


Avgvel_alg=np.trapz(best_profile,x)
Q_alg=60*1e-7*np.pi*R**2*Avgvel_alg #in mu liters/ hour
Ratio=Q_alg/Q_EO
Ratio


# In[78]:


reshape_g2_num=g2_num.reshape(len(q_rings),len(regions),len(taus))


# In[79]:


M=len(q_rings)


# In[80]:


for j in range(M):
    g2_j=reshape_g2_pseudo[j,:,:]
    I=len(g2_j)
    #print(R)
    fig=plt.figure(figsize=(25, 20))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.suptitle('5-Modes-'+'q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 70,y=1.0 )
    #plt.title('q_ring=%.5f' %q_rings[j]  +  '$\AA^{-1}$',fontsize= 15,y=1.05)

    numeric=reshape_g2_num[j,:,:]
    for k in range(I):
        g2_pseudo=g2_j[k,:]
        g2_nume=numeric[k,:]
        #g2_parabolicfit=g2_parabolic[k,:]
        f = interp1d(New_taus, g2_nume,kind='cubic')
        xnew = np.linspace(New_taus[0],New_taus[-1], num=10000, endpoint=False)
        sy=4 
        sx=4
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus, g2_pseudo ,'g^',label='pseudoplastic flow')
        ax.semilogx(xnew, f(xnew) ,'r-',label='numeric')
        
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(loc=1,prop={'size': 12})
        ax.set_xlabel(r'time(s)',fontsize=15)
        ax.set_ylabel(r'$g^{(2)}(t)$', fontsize=15)
        #ax.set_ylim([1,1.20])
        ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $', fontsize =20, y=1.0)
        #fp = new_data_dir+ '5 Modes-'+'Numeric fit-' + 'q=%.5f' %q_rings[j]+'$\AA^{-1}$'+'.png'  
        #plt.savefig( fp, dpi=fig.dpi)


# In[81]:


M

