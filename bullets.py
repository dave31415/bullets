#bullets
import numpy as np
from matplotlib import pyplot as plt

def make_blank_target(radius=150):
    '''make the empty target'''
    width=2*radius+1
    shape=(width,width)
    target=np.ones(shape,dtype=int)
    #show_target(target)
    return target
    
def show_target(target):
    '''display the target in it's current state'''
    image=target.copy()
    width=target.shape[0]
    center=width/2
    #make cross hairs in the middle
    height=width/10
    image[center,center-height:center+height]=2
    image[center-height:center+height,center]=2
    plt.imshow(image)
    plt.show()
    
def shoot_target(target,x,y,bullet_radius=10):
    '''shoot a hole in the target'''
    #consider this the minimum fraction of newly removed paper
    #to be able to recognize this as a recorded shot, otherwise
    #we will call it a missing shot
    new_frac_min=0.33
    width=target.shape[0]
    center=width/2
    xarr=np.arange(width*width) % width
    yarr=np.arange(width*width) / width
    xb=x+center
    yb=y+center
    dist=np.sqrt((xarr-xb)**2 + (yarr-yb)**2)
    target_flat=target.reshape(width*width)
    new_hole_pixels=dist < bullet_radius
    new_removed_area=sum(target_flat[new_hole_pixels])
    new_removed_fraction=new_removed_area/(np.pi*bullet_radius**2)
    print "new removed fraction %s"%new_removed_fraction
    
    missing=False
    if new_removed_fraction < new_frac_min:
        missing=True
        
    if missing:
        print "Missing shot!!"
    else:
        print "Recorded shot"
    target_flat[new_hole_pixels] =0
    target=target_flat.reshape(width,width)   
    #show_target(target)
    return missing
    
def shoot_target_random(target,sigma,bullet_radius=10):
    '''shoot a random point in the target'''
    x=np.random.randn()*sigma
    y=np.random.randn()*sigma
    missing=shoot_target(target,x,y,bullet_radius=bullet_radius)
    return (x,y,missing)
    
def make_pattern(sigma,num=10,radius=150,prompt=False,showit=False):
    '''make a whole target pattern''' 
    target=make_blank_target(radius=radius)    
    positions=[]
    n_missing=0
    for i in xrange(num):
        (x,y,missing)=shoot_target_random(target,sigma,bullet_radius=10)
        if not missing:
            positions.append((x,y))
        n_missing+=int(missing)
        if prompt:
            show_target(target)
            a=raw_input("ok?")
            if a == 'q' : break
    if showit: show_target(target)
    print "\nN missing: %s"%n_missing 
    print "N recorded: %s"%(num-n_missing) 
    print "\nrecorded positions:"
    for p in positions: print p
    width=width=2*radius+1
    return (positions,n_missing,num,width,sigma)
    
def sigma_likeihood(data,sig_min=5,sig_max=30,doplot=False):
    #so far, ignoring misssing shots
    (positions,n_missing,num,width,sigma_true)=data
    sigmas=sig_min+np.arange(sig_max-sig_min)
    center=width/2
    xarr=np.arange(width*width) % width
    yarr=np.arange(width*width) / width
    xb=center
    yb=center
    dist2=(xarr-xb)**2 + (yarr-yb)**2
    print width,num
    data=[]
    for sig in sigmas:
        print sig
        tau=1.0/sig**2
        #skip normalization on here
        #gaussian_image=np.exp(-0.5*dist2*tau)
        #likelihood for recorded points
        dist2_recorded=np.array([x**2+y**2 for x,y in positions])
        log_likelihood_each=-0.5*dist2_recorded*tau-2.0*np.log(sig)-np.log(2*np.pi)
        log_likelihood=log_likelihood_each.sum()
        likelihood=np.exp(log_likelihood)
        data.append((sig,log_likelihood))
        print "sig: %s, log_likelihood: %s"%(sig,log_likelihood)
    likes=np.array([np.exp(d[1]) for d in data])
    #normalize
    likes=likes/likes.sum()
    #get maximum likelihoods
    peak=max(likes)
    sigma_ML=sigmas[np.argmax(likes)]
    sigma_ML_variance=(likes*(sigmas-sigma_ML)**2).sum() 
    sigma_ML_err=np.sqrt(sigma_ML_variance)
    zscore=(sigma_ML-sigma_true)/sigma_ML_err
    print "sigma_ML: %s +/- %s   zscore: %s"%(sigma_ML,sigma_ML_err,zscore)
    print "N_missing: %s"%n_missing
    if doplot:
        #plt.figure()
        plt.plot(sigmas,likes)
        plt.xlabel("sigma")
        plt.ylabel("Likelihood")
        plt.plot([sigma_true,sigma_true],[0,peak*1.1],'--')
        plt.plot([sigma_ML,sigma_ML],[0,peak*1.1],'--',color='red')
    return (sigma_ML,sigma_ML_err)
            
def simulate(sigma=10.0,num=100):
    data=[]
    for i in xrange(num): 
        data.append(sigma_likeihood(make_pattern(sigma)))            
    sig_ml=np.array([d[0] for d in data])
    diff=(sig_ml-sigma)
    bias=diff.mean()
    std=diff.std()
    bias_stderr=std/np.sqrt(num)
    rms=np.sqrt((diff**2).mean())
    percent_error=100.0*bias/sigma
    percent_error_uncert=100.0*bias_stderr/sigma
    print "\nsimulation result: sigma_true=%s,num_sim=%s"%(sigma,num)
    print "bias: %s +/- %s , rms: %s"%(bias,bias_stderr,rms)
    print "percent error: %0.2f +/- %0.2f"%(percent_error,percent_error_uncert)
    
           
            
            
            