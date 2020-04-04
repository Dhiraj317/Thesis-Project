
# coding: utf-8

# In[1]:


#from chxanalys.chx_packages import *
from pyCHX.chx_packages import *
import matplotlib.pyplot as plt
#%matplotlib notebook
plt.rcParams.update({'figure.max_open_warning': 0})
import pandas as pds
from scipy.special import erf 
import scipy.special as sp
#%reset -f  #for clean up things in the memory


# In[2]:


uid='3dbd85'
uid='7c0be440'
uid='24f178'


#New data-2019_2
uid='ad5b1d33'#count : 1 ['ad5b1d33'] (scan num: 13222) (Measurement: Frate=0.35 ul/min xh=-0.35 yh=-1.057 Series 0.00134s X 1000 -sample: SiO2 D250 NP:Tw=1:2 flow cell )
uid='ad5b1d33-912f-4c7d-8ef3-76cdbd5cc693'#Q=0.35 at center
uid='e0789d35-4a7a-4058-a61d-466d4370af64'#Q=1 at center
#uid='e9716a17-767c-4360-acf6-b03d284f28cd'#Q=0.5 at center

#uid='0ba709'
#uid='dec7f8'
#uid='4f57c2'
#uid='56e5db'#2017-cycle-1 data-Q-20ul/hr
#uid='c289cf'#Q-40 ul/hr-2017-cycle 1 data
#uid='1e3556'

data_dir = '/XF11ID/analysis/2019_2/dnandyala/Results/'+ uid
#data_dir = '/XF11ID/analysis/2017_3/dnandyala/Results/' + uid


# In[3]:


data_dir


# In[4]:


import numba


# In[5]:


@numba.njit
def fast_double_sum(flow_vel,qphit):
    result = np.zeros_like(qphit)
    for i, vi in enumerate(flow_vel):
        for j, vj in enumerate(flow_vel):
            result += np.cos(qphit*(vj-vi))
    return result


# In[6]:


def g2_numeric(flow_vel,time,q_rings,regions,N):
    phi=abs(np.cos(np.radians(regions)-np.radians(90)))
    #phi=abs(np.cos(np.radians(regions)-np.radians(45)))
    qphi=np.dot(q_rings[:,None],phi[None,:])
    qphit=np.dot(qphi[:,:,None],time[None,:])
    doublesum = fast_double_sum(flow_vel, qphit)
    return doublesum/N**2


# In[7]:


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


# # New score if you want to impose flow rate 

# In[8]:


#def mse(g2_exp,g2_num,flow_prof):
 #   if len(g2_num)!=len(g2_exp):
  #      raise ValueError("data size does not match with size of numeric g2")
   # diff=(g2_exp-g2_num)/g2_exp
   # squared=diff**2
    #print(squared)
    #mean_of_differences_squared = squared.mean()
    #R=0.4
    #Avgvel_alg=np.trapz(flow_prof,x)
    #Q_alg=360*1e-6*np.pi*R**2*Avgvel_alg #in mu liters/ hour
    #score_Q=((Q_poiseuille- Q_alg)/ Q_poiseuille)**2
    #New_score=mean_of_differences_squared  + score_Q
    #return New_score


# In[9]:


#fit_params=pds.read_csv(data_dir + '/' + 'uid='+ uid+'_fra_5_1000_g2_fit_paras.csv') 
#data=pds.read_csv(data_dir + '/' + 'uid=' + uid+'_fra_5_1000_g2.csv')
#taus=data['tau'][1:]
uid='ad5b1d33'
uid='e0789d35'
#uid='e9716a17'
fit_params=pds.read_csv(data_dir + '/' + 'uid='+ uid+'_fra_5_1000_g2_fit_paras.csv') 
data=pds.read_csv(data_dir + '/' + 'uid=' + uid+'_fra_5_1000_g2.csv')
taus=data['tau'][1:]


# In[10]:


fit_params


# In[11]:


start_column=2
end_column= 32

expt_data=data.as_matrix(columns=data.columns[start_column:end_column])[1:].T
g2_exp=np.asarray(expt_data)
g2_expt=(g2_exp.flatten())
len(g2_expt)


# In[12]:


start=0
end=30
regions=np.unique(fit_params['regions'][start:end])
q_rings=np.unique(fit_params['rings'][start:end])
beta=np.array(fit_params['beta'][start:end])
baseline=np.array(fit_params['baseline'][start:end])
M=len(q_rings)


# In[13]:


q_rings


# In[14]:


len(regions)


# In[15]:


baseline_values=np.array([baseline[:]]*len(taus)).T
flattened_baseline_values=baseline_values.flatten()
beta_values=np.array([beta[:]]*len(taus)).T
flattened_beta_values=beta_values.flatten()


# In[16]:


baseline


# In[17]:


beta


# In[18]:


regions


# In[19]:


q_rings


# In[20]:


len(baseline)


# In[21]:


Q=1


# In[22]:


def get_vel(Q):
    R=0.4 #inner radius
    A=np.pi*R**2
    vel=Q/A*1e-9/(1e-6*60)*1e10 #Avg_vel in A/s
    return vel


# In[23]:


v0_expt=get_vel(Q)
v0_expt


# In[24]:


R=0.4
Np=50 #No.of divisions
pos=np.linspace(-R,R,Np) 
#vmax=4.8e4 #arbitrary average flow velocity
v0_fit=pds.read_csv(data_dir + '/' + 'uid=' + uid+'_fra_5_1000_g2glob_fit_paras.csv')['flow_velocity'][0]
#v0_expt=116050.4
v0=v0_fit#v0_expt #v0_fit
parabolic_prof=2*v0*(1-(pos/R)**2)
Diffusion=pds.read_csv(data_dir + '/' + 'uid=' + uid+'_fra_5_1000_g2glob_fit_paras.csv')['Diffusion'][0]
plot1D(parabolic_prof,pos)


# In[25]:


v0_fit


# In[26]:


R


# In[27]:


def get_diff_part(Diffusion,taus,q_rings,regions):
    q_r=np.array([q_rings[:]]*len(regions)).T
    qr_squared=q_r**2
    flattened_qr= qr_squared.flatten()
    
    qr_t=np.dot(flattened_qr[:,None],taus[None,:])
    flattened_qrt=qr_t.flatten()
    Diff_part=np.exp(-2 *Diffusion*flattened_qrt)
    return Diff_part
    
    


# In[28]:


Diff=get_diff_part(Diffusion,taus,q_rings,regions)


# In[29]:


regions


# In[30]:


v0


# In[31]:


#old one
def get_analytic(q_rings,regions,time,vel):
    phi=abs(np.cos(np.radians(regions)-np.radians(90)))
    qphi=np.dot(q_rings[:,None],phi[None,:])
    qphit=np.dot(qphi[:,:,None],time[None,:])
    Flow_part= np.pi**2/(16*qphit*vel) * abs(erf( np.sqrt( 4/np.pi * 1j *qphit*vel ) ) )**2
    return Flow_part


# In[32]:


def get_analytic(q_rings,regions,time,vel):
    phi=abs(np.cos(np.radians(regions)-np.radians(90)))
    #phi=abs(np.cos(np.radians(regions)-np.radians(45)))
    qphi=np.dot(q_rings[:,None],phi[None,:])
    qphit=np.dot(qphi[:,:,None],time[None,:])
    Flow_part= np.pi/(8*qphit*vel)* abs(  erf((1+1j)/2*  np.sqrt(   4* qphit*vel) ) )**2 
    return Flow_part


# In[33]:


g2num_parabolic=g2_numeric(flow_vel=parabolic_prof,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
g2num_parabolic=g2num_parabolic*Diff*flattened_beta_values+flattened_baseline_values


# In[34]:


len(q_rings)*len(regions)


# In[35]:


g2_theoretical=get_analytic(q_rings=q_rings,regions=regions,time=taus,vel=v0).flatten()
g2_theoretical=g2_theoretical*Diff*flattened_beta_values+flattened_baseline_values


# In[36]:


reshape_g2_data=g2_exp.reshape(len(q_rings),len(regions),len(taus))
reshape_g2_num=g2num_parabolic.reshape(len(q_rings),len(regions),len(taus))
reshape_g2_theoretical=g2_theoretical.reshape(len(q_rings),len(regions),len(taus))


# In[37]:


from scipy.interpolate import interp1d


# In[38]:


taus=np.asarray(taus)


# In[39]:


taus


# In[40]:


New_taus=np.array([ 7*1e-4  ,  0.00268   ,  0.00402   ,  0.00536   ,  0.0067    ,
        0.00804   ,  0.00938   ,  0.01072   ,  0.0134    ,  0.01608   ,
        0.01876   ,  0.02144   ,  0.0268    ,  0.03216   ,  0.03752   ,
        0.04288   ,  0.0536    ,  0.06432   ,  0.07504   ,  0.08576   ,
        0.1072    ,  0.12864   ,  0.15008   ,  0.17151999,  0.21439999,
        0.25727999,  0.30015999,  0.34303999,  0.42879999,  0.51455998,
        0.60031998,  0.68607998,  0.85759997,  1.02911997])


# In[41]:


for j in range(M):
    g2_j=reshape_g2_num[j,:,:]
    R=len(g2_j)
    #print(R)
    fig=plt.figure(figsize=(40, 30))
    fig=plt.figure(figsize=(25, 20))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 70,y=1.0 )
    #plt.title('q_ring=%.5f' %q_rings[j]  +  '$\AA^{-1}$',fontsize= 15,y=1.05)


    data=reshape_g2_data[j,:,:]
    analytic=reshape_g2_theoretical[j,:,:]
    for k in range(R):
        g2=g2_j[k,:]
        g2_data=data[k,:]
        g2_th=analytic[k,:]
        f = interp1d(New_taus, g2_th,kind='cubic')
        xnew = np.linspace(New_taus[0], New_taus[-1], num=10000, endpoint=False)
        #f = interp1d(taus, g2_th,kind='cubic')
        #xnew = np.linspace(taus[0], taus[-1], num=10000, endpoint=False)
        sy=4 
        sx=5
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus,g2_data,'bo')
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        #ax.semilogx(taus,g2_th,'r-')
        ax.semilogx(xnew, f(xnew) ,'r-',label='Analytic')
        #ax.semilogx(taus,g2,'r-')
        #ax.set_ylim([1,1.20])
        ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $',fontsize= 20,y=1.0)


# In[42]:


def get(g2_data,phi,figsize):
    R=len(phi)
    fig=plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 10,y=1.0 )
    for k in range(R):
        g2_data=g2_exp[phi[k],:]
        g2_analytic=g2_theoretical.reshape(len(q_rings)*len(regions),len(taus))
        g2=g2_analytic[phi[k],:]
        index=phi[k]
        f = interp1d(New_taus, g2,kind='cubic')
        xnew = np.linspace(New_taus[0],New_taus[-1], num=10000, endpoint=False)
        #f = interp1d(taus, g2,kind='cubic')
        #xnew = np.linspace(taus[0], taus[-1], num=10000, endpoint=False)
        sy=1 
        sx=4
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus,g2_data,'bo', label='Expt data')
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        #ax.semilogx(taus,g2_th,'r-')
        ax.semilogx(xnew, f(xnew) ,'r-',label='Numeric')
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim([0.99,1.21])
        #ax.set_ylabel(r'$g^{(2)}$', fontsize=12)
        #ax.set_xlabel(r'time(s)',fontsize=10)
        if k==2:
            ax.set_xlabel(r'time(s)',fontsize=15)
        #ax.set_ylim([0,1.02])
        #ax.set_title('$\phi $=%.1f'% regions[index] + '$^\circ $')
        #ax.set_title(r'$g^{(2)}$ in the bulk flow direction')


# In[43]:


phi=np.array([20,25,29])


# In[44]:


#get(g2_expt,phi,figsize=(4,13))
get(g2_expt,phi,figsize=(4,12))


# In[45]:


R=0.4


# #  Fourier series

#  $\ F_{n_x}$= $A_{0}$ + $\sum_{n=1}^{\infty} An cos(n\pi x/L) $ 
#  
# where 
# $ A_0= \frac{1}{L} \int\limits_0^L f(x)  dx $
# 
# 

# $ A_n= \frac{2}{L} \int\limits_0^L f(x) cos(n\pi x/L)  dx $

# $\ f(x)=2v_0(1-x^2/R^2)$                   #Parabolic profile

# $ A_0= \frac{1}{L}\int\limits_0^L 2v_0(1-x^2/R^2)  dx   $ 

# $ A_n= \frac{2}{L}\int\limits_0^L 2v_0(1-x^2/R^2) cos(n\pi x/L) dx   $ 

# # Using Mathematica ,we compute the above integrals to extract the A0 and An's for parabolic profile

# $ A_0= \frac{-4Rv_0}{6R} $   where L=2R

# $ A_n= \frac{-4v_0}{(\pi n)^3}[(3n \pi)^2-8 sin(n \pi) + 8n \pi cos(n \pi)] $ 

# In[46]:


a0=64*v0/(np.pi**3)


# In[47]:


v0


# In[48]:


modes=2


# In[49]:


mode=[]
for n in range(1,modes+1):
    #An=(4*v0)/(2*np.pi*n+np.pi)**3*((3*(2*np.pi*n+np.pi)**2-8)*np.sin(2*n*np.pi) + 8*np.pi*(2*n+1)*np.cos(2*n*np.pi))
    An=(32*v0)/(2*np.pi*n+np.pi)**3*(np.pi*(2*n+1)*np.sin(n*np.pi) + 2*np.cos(n*np.pi))
    mode.append(An)
    print(An)


# In[50]:


parabola_amplitudes=np.asarray(mode)
parabola_amplitudes=parabola_amplitudes.reshape(1,modes)


# In[51]:


parabola_amplitudes=np.concatenate((np.array([a0])[None,:],parabola_amplitudes),axis=1)


# In[52]:


parabola_amplitudes


# In[53]:


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


# In[54]:


R=0.4
Np=50
x=np.linspace(-R,R,Np)
Num_profile=1


# In[55]:


fourier=fourierSeries(x=x,amplitudes=parabola_amplitudes )


# In[56]:


fourier=np.asarray(fourier) 
parabolic=fourier.flatten()


# In[57]:


Avgvel_poiseuille=np.trapz(parabolic,x)
Q_poiseuille=60*1e-7*np.pi*R**2*Avgvel_poiseuille


# In[58]:


fig,ax=plt.subplots()
plt.plot(x,parabolic,'bo-')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


# In[59]:


Avgvel_poiseuille=np.trapz(parabolic_prof,x)
Q_poiseuille=60*1e-7*np.pi*R**2*Avgvel_poiseuille
Q_poiseuille


# In[60]:


Avgvel_poiseuille


# In[61]:


mode=parabola_amplitudes[0]


# In[62]:


mode


# # Fine meshing-Static(Brute Force)

# In[63]:


mode


# In[64]:


C1=1
C2=1.5
Sigma_1=1
Sigma_2=1


# In[65]:


chops=10
from itertools import product
#minX1, maxX1, = mode[0]-0.2*mode[0],mode[0]+0.2*mode[0] 

minX1, maxX1, = C1*mode[0]-C1*mode[0],C1*mode[0]+C1*mode[0] 
minY1, maxY1  = C2*mode[1]-C2*mode[1], C2*mode[1]+C2*mode[1]
minZ1, maxZ1 = C1*mode[2]-C1*mode[2],C1*mode[2]+C1*mode[2]
#minK, maxK = mode[3]-C*mode[3],mode[3]+C*mode[3]
#minM, maxM = mode[4]-C*mode[4],mode[4]+C*mode[4]

X1= np.linspace(minX1, maxX1, chops)
Y1 = np.linspace(minY1, maxY1, chops)
Z1= np.linspace(minZ1,maxZ1,chops)

Initial_points=np.array(list(product(X1,Y1,Z1)))
fig,ax=plt.subplots(figsize=(6,4.5))
plt.plot(Initial_points[:,0],Initial_points[:,1],'yo')


#plt.plot(Initial_points[:,1],Initial_points[:,2],'yo')
plt.plot(parabola_amplitudes[0][0],parabola_amplitudes[0][1],'go',)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=18)
plt.xlabel('A1 ($\AA^{}/s)$', size =20)
plt.ylabel('A2 ($\AA^{}/s)$', size =20)


# In[66]:


mode


# In[67]:


Initial_points.shape


# In[68]:


Num_profile=len(Initial_points)
import random
#Initial_amplitudes=[random.choice(points)  for i in range(Num_profile)]
Initial_amplitudes=Initial_points
Initial_amplitudes=np.asarray(Initial_amplitudes)


# In[69]:


g2_expt


# In[70]:


X1


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
        g2_num=g2_numeric(flow_vel=flow_prof,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        g2_num=g2_num*Diff*flattened_beta_values + flattened_baseline_values
        error=mse(g2_exp=g2_expt,g2_num=g2_num)
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


fig,ax=plt.subplots(figsize=(6,4))
#x=np.linspace(-R,R,60)

xp=np.linspace(-R,R,50)
ax.plot(x, best_profile,'b^',label='Numeric from algorithm ')
ax.plot(xp,parabolic,'g-',label='Poiseuille flow')
ax.legend(loc=0,prop={'size': 14})
ax.tick_params(axis="x", labelsize=13)
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel('r/R',fontsize=16)
ax.set_ylabel('Avg flow velocity ($\AA/s$)',fontsize=16)
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.set_title(r'10$\times$10 grid -3 Modes', size =16)
#fp = new_data_dir+ 'Plug profile-5 Modes' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)

#ax.set_title('Newscore',size=16)
#ax.set_title('Expt data-Modes-4')
#fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 20,y=1.0 )
 #ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $')


# In[73]:


Avgvel_alg=np.trapz(best_profile,x)
Q_alg=60*1e-7*np.pi*R**2*Avgvel_alg #in mu liters/ hour
Ratio=Q_alg/Q_poiseuille
Ratio


# In[74]:


fig,ax=plt.subplots(figsize=(6,4.5))
#Initial_points=np.array(list(product(X,Y,Z,K,M,P,Q)))
#Initial_points=np.array(list(product(X,Y,Z,K)))
Initial_points=np.array(list(product(X1,Y1,Z1)))
#plt.plot(Initial_points,'yo')
#plt.plot(parabola_amplitudes[0][0],'go',)
plt.plot(Initial_points[:,0],Initial_points[:,1],'yo')
plt.plot(Initial_amplitudes[best][0],Initial_amplitudes[best][1],'bo', label='Iter_1-best')

#plt.plot(Initial_points[:,1],Initial_points[:,2],'yo')
plt.plot(parabola_amplitudes[0][0],parabola_amplitudes[0][1],'go',label='parabolic')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=18)
plt.xlabel('A1 ($\AA^{}/s)$', size =20)
plt.ylabel('A2 ($\AA^{}/s)$', size =20)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(0.85, 1.09))


# In[75]:


Initial_best=Initial_amplitudes[best]


# In[76]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(parabola_amplitudes[0][0],parabola_amplitudes[0][1],parabola_amplitudes[0][2],'go',
           sizes=np.array([30]),facecolor='green', label='parabolic ')
ax.scatter(Initial_best[0],Initial_best[1],Initial_best[2],'bo',
           sizes=np.array([30]),facecolor='blue', label='Iter-1_best')
ax.plot(Initial_points[:,0],Initial_points[:,1],'y.')

ax.xaxis.major.formatter.set_powerlimits((0,0))   
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.tick_params(axis="z", labelsize=14)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(0.85, 1.09))
#fp = new_data_dir+ '10 by 10 -3D' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)


# In[77]:


Initial_amplitudes[best]


# In[78]:


Avgvel_poiseuille=np.trapz(parabolic,x)
Q_poiseuille=360*1e-6*np.pi*R**2*Avgvel_poiseuille


# In[79]:


Avgvel_alg=np.trapz(best_profile,x)
Q_alg=360*1e-6*np.pi*R**2*Avgvel_alg #in mu liters/ hour
Ratio=Q_alg/Q_poiseuille
Ratio


# In[80]:


Y1


# In[81]:


Z1


# In[82]:


Sigma_1


# In[83]:


chops=20


# In[92]:


for ind in range(len(X1)):
    if X1[ind]==Initial_amplitudes[best][0]:
        try:
            if X1[ind]>=X1[ind-Sigma_1] and X1[ind]<=X1[ind+Sigma_1]:
                min_X1,max_X1=X1[ind-Sigma_1],X1[ind+Sigma_1]
                X2=np.linspace(min_X1,max_X1,chops)
        except:
                X2=X1
for ind in range(len(Y1)):            
    if Y1[ind]==Initial_amplitudes[best][1]:
        try:
            if Y1[ind]<=Y1[ind-Sigma_1] and Y1[ind]>=Y1[ind+Sigma_1]:
                min_Y1,max_Y1=Y1[ind-Sigma_1],Y1[ind+Sigma_1]
                Y2=np.linspace(min_Y1,max_Y1,chops)
                
        except:
                Y2=Y1
for ind in range(len(Z1)):            
    if Z1[ind]==Initial_amplitudes[best][2]:
        try:
            if Z1[ind]>=Z1[ind-Sigma_1] and Z1[ind]<=Z1[ind+Sigma_1]:
                min_Z1,max_Z1=Z1[ind-Sigma_1],Z1[ind+Sigma_1]
                Z2=np.linspace(min_Z1,max_Z1,chops)
                
        except:
                Z2=Z1
        else:
                Z2=Z1
       


# In[85]:


Initial_amplitudes[best]


# In[86]:


Z1


# In[87]:


Y1


# In[88]:


X1


# In[89]:


X2


# In[93]:


Y2


# In[94]:


Z2


# # Refining Grid around best one from Intial grid

# In[95]:


fig,ax=plt.subplots(figsize=(6,4.5))
second_points=np.array(list(product(X2,Y2,Z2)))
plt.plot(second_points[:,0],second_points[:,1],'ro')
#plt.plot(parabola_amplitudes[0][0],parabola_amplitudes[0][1],'go',)
ax.plot(Initial_amplitudes[best][0],Initial_amplitudes[best][1],'bo')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=18)
plt.xlabel('A1 ($\AA^{}/s)$', size =20)
plt.ylabel('A2 ($\AA^{}/s)$', size =20)


# In[96]:


Num_profile=len(second_points)
import random
#Initial_amplitudes=[random.choice(points)  for i in range(Num_profile)]
second_amplitudes=second_points
second_amplitudes=np.asarray(second_amplitudes)


# In[97]:


second_points.shape


# In[98]:


R


# In[99]:


import time
start_time = time.time()
epsilon=1*1e-6
import operator
import pandas as pd
previous_value=[]
Errors_alliterations=[]
Global_chisquare=[]
count=0
x=np.linspace(-R,R,50)
#while True:
for a in range(1):
    fourier_profiles=fourierSeries(x=x,amplitudes=second_amplitudes)
    errors=[]
    for j,value in enumerate(fourier_profiles):
        flow_prof=fourier_profiles[j]
        #print(flow_prof)
        g2_num=g2_numeric(flow_vel=flow_prof,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        g2_num=g2_num*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_exp,g2_num=g2_num)
        error=mse(g2_exp=g2_expt,g2_num=g2_num,)
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
    


# In[100]:


fig,ax=plt.subplots(figsize=(6,4))
#x=np.linspace(-R,R,60)

xp=np.linspace(-R,R,50)
ax.plot(x, best_profile,'b^',label='Numeric from algorithm ')
ax.plot(xp,parabolic,'g-',label='Poiseuille flow')
ax.legend(loc=0,prop={'size': 14})
ax.tick_params(axis="x", labelsize=13)
ax.tick_params(axis="y", labelsize=15)
ax.set_xlabel('r/R',fontsize=16)
ax.set_ylabel('Avg flow velocity ($\AA/s$)',fontsize=16)
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.set_title(r'20 $\times 20 $  grid', size =16)

#fp = new_data_dir+ 'Three-modes_20 by 20 profile' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)


# In[101]:


#Making colors for visualization
rng = np.random.RandomState(0)
colors_1 = rng.rand(len(Initial_points))
sizes_1 = 100 * rng.rand(len(Initial_points))


colors_2 = rng.rand(len(second_points))
sizes_2 = 100 * rng.rand(len(second_points))


# In[102]:


fig,ax=plt.subplots(figsize=(6,4.5))
ax.scatter(Initial_points[:,0],Initial_points[:,1],c=colors_1,s=sizes_1,alpha=0.01, cmap='pink')
ax.scatter(second_points[:,0],second_points[:,1], c=colors_2,s=sizes_2, alpha=0.01,cmap='plasma')
ax.scatter(Initial_best[0],Initial_best[1],marker='o', color='b', label='Iter-1_best')
ax.scatter(second_amplitudes[best][0],second_amplitudes[best][1],marker='o', color='k',label='Iter-2_best')
ax.xaxis.major.formatter.set_powerlimits((0,0))   
ax.yaxis.major.formatter.set_powerlimits((0,0))
#'
ax.set_title(r'20 $\times 20 $  grid', size =16)
#ax.set_xlabel('A0 ($\AA^{}/s)$', size =10)
#ax.set_ylabel('A1 ($\AA^{}/s)$', size =10)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=18)
#ax.set_title('Res-20')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(0.85, 1.09))
#fp = new_data_dir+ '20 by 20 grid' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)

plt.show()


# In[103]:


Avgvel_poiseuille=np.trapz(parabolic,x)
Q_poiseuille=360*1e-6*np.pi*R**2*Avgvel_poiseuille


# In[104]:


mode


# In[105]:


second_best=second_amplitudes[best]


# In[106]:


second_best


# In[107]:


Avgvel_alg=np.trapz(best_profile,x)
Q_alg=360*1e-6*np.pi*R**2*Avgvel_alg #in mu liters/ hour
Ratio=Q_alg/Q_poiseuille
Ratio


# In[108]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Initial_best[0],Initial_best[1],Initial_best[2],'bo',
           sizes=np.array([30]),facecolor='blue',label='Iter_1-best')
ax.scatter(second_best[0],second_best[1],second_best[2],'ko',
           sizes=np.array([30]),facecolor='black',label='Iter-2_best')

ax.plot(Initial_points[:,0],Initial_points[:,1],'y.')

ax.xaxis.major.formatter.set_powerlimits((0,0))   
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.tick_params(axis="z", labelsize=14)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(0.85, 1.09))


#fp = new_data_dir+ '10 by 10 -3D' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)


# In[109]:


second_best


# In[110]:


chops=30


# In[119]:


for ind in range(len(X2)):
    if X2[ind]==second_amplitudes[best][0]:
        if X2[ind]>=X2[ind-Sigma_2] and X2[ind]<=X2[ind+Sigma_2]:
            min_X2,max_X2=X2[ind-Sigma_2],X2[ind+Sigma_2]
            X3=np.linspace(min_X2,max_X2,chops)
        else:
            X3=X2
for ind in range(len(Y2)):            
    if Y2[ind]==second_amplitudes[best][1]:
        try:
            if Y2[ind]<=Y2[ind-Sigma_2] and Y2[ind]>=Y2[ind+Sigma_2]:
                min_Y2,max_Y2=Y2[ind-Sigma_2],Y2[ind+Sigma_2]
                Y3=np.linspace(min_Y2,max_Y2,chops)
        except:
                Y3=Y2
for ind in range(len(Z2)):         
    if Z2[ind]==second_amplitudes[best][2]:
        try:
            if Z2[ind]>=Z2[ind-Sigma_2] and Z2[ind]<=Z2[ind+Sigma_2]:
                min_Z2,max_Z2=Z2[ind-Sigma_2],Z2[ind+Sigma_2]
                Z3=np.linspace(min_Z2,max_Z2,chops)
        except:
            Z3=Z2
      


# In[112]:


second_amplitudes[best]


# In[113]:


X2


# In[114]:


X3


# In[115]:


Y2


# In[122]:


Y3


# In[121]:


Z2


# In[120]:


Z3


# In[123]:


fig,ax=plt.subplots(figsize=(6,4.5))
third_points=np.array(list(product(X3,Y3,Z3)))
plt.plot(third_points[:,0],third_points[:,1],'co')
#plt.plot(parabola_amplitudes[0][1],parabola_amplitudes[0][2],'go',)
plt.plot(second_amplitudes[best][0],second_amplitudes[best][1],'bo')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.xlabel('A0 ($\AA^{}/s)$', size =20)
plt.ylabel('A1 ($\AA^{}/s)$', size =20)


# In[124]:


Num_profile=len(third_points)
import random
#Initial_amplitudes=[random.choice(points)  for i in range(Num_profile)]
third_amplitudes=third_points
third_amplitudes=np.asarray(third_amplitudes)


# In[125]:


third_points.shape


# In[126]:


R


# In[127]:


import time
start_time = time.time()
epsilon=1*1e-6
import operator
import pandas as pd
previous_value=[]
Errors_alliterations=[]
Global_chisquare=[]
count=0
x=np.linspace(-R,R,50)
#while True:
for a in range(1):
    fourier_profiles=fourierSeries(x=x,amplitudes=third_amplitudes)
    errors=[]
    for j,value in enumerate(fourier_profiles):
        flow_prof=fourier_profiles[j]
        #print(flow_prof)
        g2_num=g2_numeric(flow_vel=flow_prof,q_rings=q_rings,regions=regions,time=taus,N=Np).flatten()
        g2_num=g2_num*Diff*flattened_beta_values + flattened_baseline_values
        #error=mse(g2_exp=g2_exp,g2_num=g2_num)
        error=mse(g2_exp=g2_expt,g2_num=g2_num,)
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
    


# In[128]:


len(g2_expt)


# In[129]:


R


# In[130]:


fig,ax=plt.subplots(figsize=(6,4))
#x=np.linspace(-R,R,60)

xp=np.linspace(-R,R,50)
ax.plot(x, best_profile,'b^',label='Numeric from algorithm ')
ax.plot(xp,parabolic,'g-',label='Poiseuille flow')
ax.legend(loc=0,prop={'size': 14})
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
#ax.set_xlabel('r/R',fontsize=16)
#ax.set_ylabel(r'$\ v_0   (\AA^{}/s) $',fontsize=22)
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.set_title(r'30 $\times30 $  grid', size =16)
#fp = new_data_dir+ 'Three-modes_30 by 30 profile' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)


# In[131]:


colors_3 = rng.rand(len(third_points))
sizes_3 = 10 * rng.rand(len(third_points))


# In[133]:


fig,ax=plt.subplots(figsize=(6,4.5))

plt.scatter(Initial_points[:,0],Initial_points[:,1],c=colors_1,s=sizes_1,alpha=0.003,
            cmap='pink')

plt.scatter(second_points[:,0],second_points[:,1], c=colors_2,s=sizes_2, alpha=0.007,cmap='plasma')
plt.scatter(third_points[:,0],third_points[:,1],c=colors_3,s=sizes_3, alpha=0.5,cmap='Reds')


plt.scatter(second_best[0],second_best[1],marker='.', color='b',label='Iter_2-best')
plt.scatter(third_amplitudes[best][0],third_amplitudes[best][1],marker='.', color='k',label='Iter_3-best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(0.85, 1.09))

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
#plt.xlabel('A0 ($\AA^{}/s)$', size =20)
#plt.ylabel('A1 ($\AA^{}/s)$', size =20)
#fp = new_data_dir+ '20 by 20 grid' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)


# In[134]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(Initial_points[:,0],Initial_points[:,1],c=colors_1,s=sizes_1,alpha=0.01,
            cmap='pink')

ax.scatter(second_points[:,0],second_points[:,1], c=colors_2,s=sizes_2, alpha=0.01,cmap='plasma')

ax.scatter(third_points[:,0],third_points[:,1],color='c', marker='.')


#ax.scatter(parabola_amplitudes[0][0],parabola_amplitudes[0][1], parabola_amplitudes[0][2],
 #          'go',sizes=np.array([30]),facecolor='green', label= 'parabolic')

ax.scatter(second_best[0],second_best[1],second_best[2],'b^',
           sizes=np.array([30]),facecolor='blue', label= 'Iter_2-best')

ax.scatter(third_amplitudes[best][0],third_amplitudes[best][1],third_amplitudes[best][2],'ko',
           sizes=np.array([30]),facecolor='black', label= 'Iter_3-best')

ax.xaxis.major.formatter.set_powerlimits((0,0))   
ax.yaxis.major.formatter.set_powerlimits((0,0))
ax.tick_params(axis="x", labelsize=17)
ax.tick_params(axis="y", labelsize=19)
ax.tick_params(axis="z", labelsize=17)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(0.85, 1.09))
#fp = new_data_dir+ '30 by 30- 3D' +'.png'  
#plt.savefig( fp, dpi=fig.dpi)


# In[135]:


third_amplitudes[best]


# In[136]:


parabola_amplitudes


# In[137]:


Avgvel_poiseuille=np.trapz(parabolic,x)
Q_poiseuille=360*1e-6*np.pi*R**2*Avgvel_poiseuille


# In[138]:


Avgvel_alg=np.trapz(best_profile,x)
Q_alg=360*1e-6*np.pi*R**2*Avgvel_alg #in mu liters/ hour
Ratio=Q_alg/Q_poiseuille
Ratio


# In[ ]:


uid


# # 10 X 10 grid only

# In[ ]:


one_mode=9.41932982787e-06
Two_modes=6.15120916673e-06
Three_modes=4.67873747524e-06


# In[ ]:


# change it to Noise

#reshape_g2_theoretical=g2_noise.reshape(len(q_rings),len(regions),len(taus))
reshape_g2_data=g2_expt.reshape(len(q_rings),len(regions),len(taus))
reshape_g2_theoretical=g2_theoretical.reshape(len(q_rings),len(regions),len(taus))


# In[ ]:


for j in range(M):
    g2_j=reshape_g2_num[j,:,:]
    R=len(g2_j)
    #print(R)
    fig=plt.figure(figsize=(25, 20))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 70,y=1.0 )
    #plt.title('q_ring=%.5f' %q_rings[j]  +  '$\AA^{-1}$',fontsize= 15,y=1.05)

    #noise=reshape_g2_theoretical[j,:,:]
    g2_parabolic=reshape_g2_theoretical[j,:,:]
    data=reshape_g2_data[j,:,:]
    for k in range(R):
        num_g2=g2_j[k,:]
        g2_data=data[k,:]
        #g2_noisedata=noise[k,:]
        #g2_parabolicfit=g2_parabolic[k,:]
        f = interp1d(New_taus, num_g2,kind='cubic')
        xnew = np.linspace(New_taus[0],New_taus[-1], num=10000, endpoint=False)
        sy=4 
        sx=4
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus,g2_data,'bo', label='Expt data')
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        #ax.semilogx(taus,g2_th,'r-')
        ax.semilogx(xnew, f(xnew) ,'r-',label='Numeric')
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(loc=1,prop={'size': 12})
        ax.set_xlabel(r'time(s)',fontsize=15)
        ax.set_ylabel(r'$g^{(2)}(t)$', fontsize=15)
        #ax.set_ylim([1,1.20])
        ax.set_title('$\phi $=%.1f'% regions[k] + '$^\circ $', fontsize =20, y=1.0)
        fp = new_data_dir+ 'Numeric fit-' + 'q=%.5f' %q_rings[j]+'$\AA^{-1}$'+'.png'  
        plt.savefig( fp, dpi=fig.dpi)


# In[ ]:


def get(g2_data,phi,figsize):
    R=len(phi)
    fig=plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle('q_ring=%.5f' %q_rings[j] +  '$\AA^{-1}$',fontsize= 10,y=1.0 )
    for k in range(R):
        g2_data=reshape_g2_data[0,phi[k],:]
        #g2_theoretical=reshape_g2_theoretical[0,phi[k],:]
        g2=reshape_g2_num[0,phi[k],:]
        index=phi[k]
        f = interp1d(New_taus, g2,kind='cubic')
        xnew = np.linspace(New_taus[0],New_taus[-1], num=10000, endpoint=False)
        sy=1 
        sx=4
        ax=fig.add_subplot(sx,sy,k+1)
        ax.semilogx(taus,g2_data,'bo', label='Expt data')
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        #ax.semilogx(taus,g2_th,'r-')
        ax.semilogx(xnew, f(xnew) ,'r-',label='Numeric')
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        #ax.set_ylabel(r'$g^{(2)}$', fontsize=12)
        #ax.set_xlabel(r'time(s)',fontsize=10)
        if k==2:
            ax.set_xlabel(r'time(s)',fontsize=11)
        #ax.set_ylim([0,1.02])
        #ax.set_title('$\phi $=%.1f'% regions[index] + '$^\circ $')
        #ax.set_title(r'$g^{(2)}$ in the bulk flow direction')


# In[ ]:


phi=np.array([0,5,9])


# In[ ]:


#get(g2_expt,phi,figsize=(4,13))
get(g2_expt,phi,figsize=(4,15))


# In[ ]:


uid

