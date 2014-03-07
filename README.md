bullets
=======

This currenly requires that you use ipython in pylab mode. Otherwise the plotting will block and be very annoying. 

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

How can we come up with an optimal measure of shooting accuracy, given the assumptions stated above, simply from 
the images of spent tagtes. 

Sketch of the Solution
=============
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
product of the radii (this second one, the product is required for the solution in any dimension higher than 1). 

When there is overlap
---------------------






