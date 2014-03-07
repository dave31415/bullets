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

We assume that the shooters all shoot from a 2D Normal distribution but that each has a different value for 
sigma, the dispersion parameter of the Normal distribution. We'll also assume that the mean is zero, that is, there
is no bias away from the center of the bulls-eye - there is just a scatter about it. 

The data is the image of the shot target where 10 shots have been fired. The major complication that we
wish to address is that the bullet holes are not point-like. They have a diameter of 1cm. For the best shooters,
the ones we are most interested in, the bullet holes overlap at the center and it isn't clear exactly where all the
bullets went. 

How can we come up with an optimal measure of shooting accuracy, given the assumptions stated above?

The Solution
=============
We are going to offer a solution with the real goal of teaching Bayesian statistics without a lot of math. That is, we will
simply try to use numerical computation applied directly to Bayes Theorem. We do this in Python but could easily do it in 
R or pretty much any language.  

