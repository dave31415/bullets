#bullets
import numpy as np
from matplotlib import pyplot as plt
import time

class target(object):
    def __init__(self,width=401,sigma_prior=50.0,sigma_true=12.0,bullet_width=10.0):
        self.sigma_true=sigma_true
        width_old=int(width)
        width=2*(width/2)+1
        if width != width_old:
            print "Warning, target width rounded up to nearest ODD integer, width=%s"%width
        self.width=width
        self.sigma_prior=sigma_prior
        self.bullet_width=bullet_width
        self.bullet_locations_true=[]
        self.bullet_locations_recorded=[]
        self.new_frac_min=0.35   #when a bullet removes this much area, we will recognize it 
                                 #as a recordable shot, not great but OK for now
        self.npix=width**2      #assumed to be square
        self.n_missing=0
        self.n_shots=0
        self.name='TARGET'
        #make blank taget
        shape=(width,width)
        self.target=np.ones(shape,dtype=int)
        
    def show(self):
        '''display the target in it's current state'''
        image=self.target.copy()
        center=float(self.width/2)
        #make cross hairs in the middle when displaying
        x_height=self.width/10
        image[center,center-x_height:center+x_height]=2
        image[center-x_height:center+x_height,center]=2
        plt.clf()   #clears display
        plt.imshow(image)
        plt.draw()
        plt.show()
        
    def shoot_point(self,point):
        '''Shoot a hole in the target at position x,y w.r.t the center being (0,0)'''
        x,y=point
        self.bullet_locations_true.append((x,y))
        self.n_shots+=1
        #consider this the minimum fraction of newly removed paper
        #to be able to recognize this as a recorded shot, otherwise
        #we will call it a missing shot
        center=float(self.width/2)
        #define the coordinate arrays (flattened)
        xarr=np.arange(self.npix) % self.width
        yarr=np.arange(self.npix) / self.width
        xb=x+center     #in pixel coordinates where center is not (0,0) but (center,center)
        yb=y+center 
        dist2=(xarr-xb)**2 + (yarr-yb)**2       #distance-squared 
        target_flat=self.target.reshape(self.npix)     #flatten it
        bullet_rad2=(self.bullet_width/2.0)**2
        bullet_area=bullet_rad2*np.pi
        new_hole_pixels=dist2 < bullet_rad2
        new_removed_area=sum(target_flat[new_hole_pixels])   #how many of the new pixels get removed
        new_removed_fraction=new_removed_area/bullet_area
        #print "new removed fraction %s"%new_removed_fraction
        if new_removed_fraction > self.new_frac_min:
            #we will use this simple huristic for now to determine whether this will be a recordable shot
            self.bullet_locations_recorded.append((x,y))
        else:
            #bullet hole unclear, call it a missing shot
            self.n_missing+=1
        #remove the pixels
        target_flat[new_hole_pixels] =0
        self.target=target_flat.reshape(self.width,self.width)  
    
    def bang(self,num_shots=1,show_slow=0.0):
        '''shoot a random point in the target'''
        #TODO: add a boundary check and warning (or raise error)
        if num_shots > 1:
            #multiple shots
            for i in xrange(num_shots): 
                self.bang()
                if show_slow > 0.0:
                    self.show()
                    #print "BANG!"
                    time.sleep(show_slow)
        else :
            #one shot fired
            x=np.random.randn()*self.sigma_true
            y=np.random.randn()*self.sigma_true
            point=(x,y)
            self.shoot_point(point)

    def __repr__(self):
        '''for printing'''
        printables=(self.name,self.width,self.sigma_true,self.n_shots,self.n_missing)
        return "name=%s, width=%s, sigma_true=%s, n_shots=%s, n_missing=%s"%printables
        
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
    
           
            
            
            