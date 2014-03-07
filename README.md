bullets
=======

This currenly requires that you use ipython in pylab mode. Otherwise the plotting will block and be 
very annoying. You also need the package lmfit. (e.g. sudo pip install lmfit).

ipython --pylab

ipython> import bullets

run the test demo which shows Bayesian statistics in action

ipython> bullets.test()

The Problem:
============

We are analyzing targets from a shooting range and we want to assign an accuracy score to every shooter
so that we can determine who are the best amongst them. 

We're going to start with the following assumptions:

We assume that the shooters all shoot from a 2D Normal (circularly symmetric) distribution but that 
each has a different value for sigma, the dispersion parameter of the Normal distribution. 
We'll also assume that the mean is zero, that is, there
is no bias away from the center of the bulls-eye - there is just a scatter about it. 

The data is the image of the shot target where 10 shots have been fired. The major complication that we
wish to address is that the bullet holes are not point-like. They have a diameter of 1cm. For the best shooters,
the ones we are most interested in, the bullet holes overlap at the center and it isn't clear exactly where all the
bullets went. 

[See example image here](Target.jpg)

How can we come up with an optimal measure of shooting accuracy, given the assumptions stated above, simply from 
the images of spent tagtes. 

Sketch of the Solution
=======================
We are going to offer a solution with the real goal of teaching Bayesian statistics without a lot of math. That is, we will
simply try to use numerical computation applied directly to Bayes Theorem. We do this in Python but could easily do it in 
R or pretty much any language.  

Simple case of no overlap
---------------------------

We can break this problem up into several pieces. The simplest case is when all 10 bullet holes are apparent. 
In this case, there are two subproblems to solve. One is the image processing task of locating the positions of the
holes on the image of the target. This problem is not very hard and doesn't have much to do with Bayesian 
statistics so we will put it aside for now and assume that it is done. 

Now with the 10 bullet locations we need to find the estimator of accuracy. Since we have already defined the distribution to
be Normal, this is just a question of calculating the posterior probability of the parameter sigma. One we have done that, our
work as Bayesians is done. We can pick out the mean or mode of that distribution if we need a single number and measure
the variance of that distribution to give us an uncertainty. But calculating the posterior is our goal.  

Calculating the posterior of sigma with known mean is a problem studies in almost every book on Bayesian statistics as
one of the first examples of a non-trivial use of Bayes Theorem. However, this typically done with 1D Normal distributions
not 2D. However, the changes required to do this in 2D are minimal. See section 5 at this link
http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf. That reference explains many variance on this problem as well as how
to incorporate conjugate priors. 

Solving these two subproblems solves the simple case of separated bullets. All you need to record are the radii of the
shots. In fact all you really need to keep is the sufficient statistic of the average of the square of the radii plus the
product of the radii (this second one, the product is required for the solution in any dimension higher than 1.

When there is overlap
---------------------
This is solved again by breaking the problem into two parts. Some of the bullets leave recordable locations on the target. 
Some of them are missing.

Note that the data, the target image, will look the same regardless of the order of the shots.
This is important as we can then assume without loss of generality that all the bullets that created recordable holes
came first and the others which left no mark came last. 

So we can treat this as two separate experiments. Let's say that of the 10 shots, we can easily identify 6 of them. They are
either disconnected completely from other holes or at least leave a recognizable semi-circular outline on the target. This
again is a separate image processing task that we will put aside for the moment. 

Now dealing with these 6 bullets is reduced to the simple case describe above. If we stopped there and ignored the
other four bullets, we would get an estimate of accuracy that is too low as we are not giving them credit 
for the last 4 bullets and so we get a biased solution. 

We need to then consider the second experiment where four bullets were fired and did not leave a new hole. 

We combine experiments by multiplying likelihoods since each shot is statistically independent but comes from the 
same distribution, i.e., the shooter is the same and so is sigma. For the simple problem above we knew to calculate 
the likelihood as a product of Normal distributions. We do the same for the problem here with the 6 bullets. 

What is the likelihood for the 4 bullets? We don't know where they went. All we know is where
they DIDN'T go. They didn't make new holes. So, what is the probability that someone firing with a normal 
distribution (and some particular sigma) will fire at the target and not create new holes? That is just the 
integral (or sum) over the 2D Gaussian distribution multiplied by the mask defined by the target as it stands after
those 6 recognizable shots have been fired. Well, given that the bullets have finite size, it is a little different 
as you have to buffer the edges a bit. Still, it is a fairly easy thing to calculate numerically. 

So there you have it. We have a curve for the likelihood of sigma from the recorded bullets. We shall see, that 
this looks like a skewed bell shape function that peaks around the average of the square of the radii. We also have a
likelihood curve for the 4 bullets that landed in the masked region. This curve drops monotonically with sigma not very 
different from an exponential function P(sigma) ~ exp(-sigma / scale_parameter). Multiplying these curves together
results in a similar skewed, bell-shaped curve but one that peaks at lower sigma (higher accuracy). Thus, the
curve from the four missing bullets does what we had hoped: it gives the shooter credit for those good shots and gives
us more confidence about their shooting accuracy. 

The code included here performs this basic calculation. It has some visualizations to actually see the targets as
well as seeing the posterior probabilities. I have cheated a bit and used a heuristic to determine which bullets are 
recordable and which are not and used their input coordinates when I have decided they would be recordable. This just
allows me to skip the image processing tasks for now and concentrate on the rest of the problem involving Bayesian 
statistics.  


















