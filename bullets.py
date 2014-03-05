#bullets
import numpy as np
from matplotlib import pyplot as plt
import time

class target(object):
    def __init__(self,width=201,sigma_true=12.0,bullet_width=10.0):
        #TODO: sigma_prior not yet used
        self.sigma_true=sigma_true
        width_old=int(width)
        width=2*(width/2)+1
        if width != width_old:
            print "Warning, target width rounded up to nearest ODD integer, width=%s"%width
        self.width=width
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
        self.center=float(self.width/2)
        #define the coordinate arrays (flattened)
        self.xarr=np.arange(self.npix) % self.width
        self.yarr=np.arange(self.npix) / self.width
        
    def show(self):
        '''display the target in it's current state'''
        image=self.target.copy()
        #make cross hairs in the middle when displaying
        x_height=self.width/10
        image[self.center,self.center-x_height:self.center+x_height]=2
        image[self.center-x_height:self.center+x_height,self.center]=2
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
        xb=x+self.center     #in pixel coordinates where center is not (0,0) but (center,center)
        yb=y+self.center 
        dist2=(self.xarr-xb)**2 + (self.yarr-yb)**2       #distance-squared 
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
        
    def sigma_likelihood(self,sig_min=0.3,sig_max=30,n_sigmas=200,doplot=True):
        #so far, ignoring misssing shots
        #sigmas=sig_min+np.arange(sig_max-sig_min)
        sigmas=np.linspace(sig_min,sig_max,n_sigmas)
        #dist2=(self.xarr-self.center)**2 + (self.yarr-self.center)**2
        data=[]
        for sig in sigmas:
            tau=1.0/sig**2
            #skip normalization on here
            #gaussian_image=np.exp(-0.5*dist2*tau)
            #likelihood for recorded points
            dist2_recorded=np.array([x**2+y**2 for x,y in self.bullet_locations_recorded])
            log_likelihood_each=-0.5*dist2_recorded*tau-2.0*np.log(sig)-np.log(2*np.pi)
            log_likelihood=log_likelihood_each.sum()
            likelihood=np.exp(log_likelihood)
            data.append((sig,log_likelihood))
            #print "sig: %s, log_likelihood: %s"%(sig,log_likelihood)
        likes=np.array([np.exp(d[1]) for d in data])
        #normalize
        likes=likes/likes.sum()
        #get maximum likelihoods
        peak=max(likes)
        sigma_ML=sigmas[np.argmax(likes)]
        sigma_ML_variance=(likes*(sigmas-sigma_ML)**2).sum() 
        sigma_ML_err=np.sqrt(sigma_ML_variance)
        zscore=(sigma_ML-self.sigma_true)/sigma_ML_err
        self.sigma_ML=sigma_ML
        self.sigma_ML_err=sigma_ML_err
        self.sigmas=sigmas
        self.sigma_likelihood=likes
        print "sigma_ML: %s +/- %s   zscore: %s"%(sigma_ML,sigma_ML_err,zscore)
        print "N_missing: %s"%self.n_missing
        if doplot:
            #plt.figure()
            plt.clf()
            plt.plot(sigmas,likes)
            plt.xlabel("sigma")
            plt.ylabel("Likelihood")
            plt.plot([self.sigma_true,self.sigma_true],[0,peak*1.1],color="green")
            plt.plot([sigma_ML,sigma_ML],[0,peak*1.1],'--',color='red')
            plt.draw()
            plt.show()

def exp_robust_renorm(x):
    #assumes numpy array
    #re-normalizes to max of 1
    #zero out any exponential that is 50 orders of mag (base e) smaller
    y=x-x.max()
    mask=1.0*(y > -50)
    return mask*np.exp(mask*y)
    
def simulate(sigma_true=10.0,n_targets=100,width=201,bullet_width=10.0,num_shots=10,show_slow=0):
    for i in xrange(n_targets):
        targ=target(width=width,sigma_true=sigma_true,bullet_width=bullet_width)
        targ.bang(num_shots=num_shots,show_slow=show_slow)
        targ.sigma_likelihood(doplot=False)
        log_like=np.log(targ.sigma_likelihood)
        if i ==0 : 
            sigmas=targ.sigmas
            log_likes=log_like
        else:
            log_likes+=log_like
    #likes=exp_robust_renorm(log_likes)
    likes=np.exp(log_likes)
    
    #re-normalize to unit area
    likes=likes/likes.sum()
    peak=max(likes)
    sigma_ML=sigmas[np.argmax(likes)]
    plt.clf()
    plt.plot(sigmas,likes)
    plt.xlabel("sigma")
    plt.ylabel("Likelihood")
    plt.plot([targ.sigma_true,targ.sigma_true],[0,peak*1.1],color="green")
    plt.plot([sigma_ML,sigma_ML],[0,peak*1.1],'--',color='red')
    plt.xlim(0,targ.sigma_true*2)
    plt.draw()
    plt.show()
        
    bias=sigma_ML-sigma_true
    #error about the ML value
    std=np.sqrt(((targ.sigmas-sigma_ML)**2 *likes).sum())
    
    print "\nsimulation result: sigma_true=%s,num_targets=%s, sigma_ML=%s"%(sigma_true,n_targets,sigma_ML)
    print "bias: %s +/- %s"%(bias,std)
    #print "percent error: %0.2f +/- %0.2f"%(percent_error,percent_error_uncert)
    
def test():
    prompt=True
    print "Testing everything\n"
    print "Making a target and displaying it"
    targ=target()
    targ.show()
    if prompt : 
        input=raw_input("Ok? (if not in ipython, you have to kill the window first)")
        if input == 'q' : return 1
    print "Shooting a bullet, BANG!"
    targ.bang()
    targ.show()
    if prompt : 
        input=raw_input("Ok? (if not in ipython, you have to kill the window first)")
        if input == 'q' : return 1
    print "Shooting 5 more bullets (with time delay)"
    targ.bang(num_shots=5,show_slow=.1)
    targ.show()
    if prompt : 
        input=raw_input("Ok? (if not in ipython, you have to kill the window first)")
        if input == 'q' : return 1
    print "Trying to calculate the likelihood of sigma from the data"
    targ.sigma_likelihood()
    if prompt : 
        input=raw_input("Ok? (if not in ipython, you have to kill the window first)")
        if input == 'q' : return 1
    print "Now going to simulate the Bayesian way of combining 20 targets"
    simulate(n_targets=20)
    if prompt : 
        input=raw_input("Ok? (if not in ipython, you have to kill the window first)")
        if input == 'q' : return 1
    sigma_true=3.0
    print "Now going to simulate the Bayesian way of combining 20 targets for a very small sigma_true=%s"%sigma_true
    simulate(n_targets=20,sigma_true=sigma_true)
    print "Now it is clear that it is biased because it is ignoring bullets that didn't leave obvious marks"
    if prompt : 
        input=raw_input("Ok? (if not in ipython, you have to kill the window first)")
        if input == 'q' : return 1
    print "Lets see this visually"
    targ=target(sigma_true=sigma_true)
    targ.bang(num_shots=10,show_slow=0.2)
    targ.show()
    print "See they are all bunched up and overlapping"
    print "The current method is biased. We need to correct for this bias which is a TODO"
            
            
            