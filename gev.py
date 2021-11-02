#imports relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import genextreme as gev
from sympy import *                    

#uses sympy analytic library to take the analytic derivatives of the likelihood
z, mu, xi, sigma, m = symbols('z mu xi sigma m')

#we will split the likelihood into the part non-dependent in z, called non-summation, and the part that depends in z, as we will need to sum it all over
#when the shape parameter xi is non-zero
likelihood_xi_neq_0_non_summation = -m*log(sigma)
likelihood_xi_neq_0_summation = -(1+1/xi)*log(1+xi/sigma*(z-mu))-(1+xi/sigma*(z-mu))**(-1/xi)
likelihood_xi_neq_0 = likelihood_xi_neq_0_non_summation+likelihood_xi_neq_0_summation
l_xi_neq_0_summation = lambdify([z,xi,mu,sigma, m],likelihood_xi_neq_0_summation)
l_xi_neq_0_non_summation = lambdify([z,xi,mu,sigma, m],likelihood_xi_neq_0_non_summation)

#when the shape parameter xi is zero
likelihood_xi_0_non_summation = -m*log(sigma)
likelihood_xi_0_summation = -log(1/sigma*(z-mu))-exp(-1/sigma*(z-mu))
likelihood_xi_0 = likelihood_xi_0_non_summation+likelihood_xi_0_summation
l_xi_0_summation = lambdify([z,xi,mu,sigma, m],likelihood_xi_0_summation)
l_xi_0_non_summation = lambdify([z,xi,mu,sigma, m],likelihood_xi_0_non_summation)


#calculates the log_likelihood when the shape parameter xi is non-zero
def log_likelihood_neq_0(list_of_z, xi, mu, sigma):
    m = len(list_of_z)
    summation = 0
    for z in list_of_z:
        summation += l_xi_neq_0_summation(z,xi,mu,sigma, m)
    return summation+l_xi_neq_0_non_summation(10,xi, mu,sigma, m)

#calculates the log_likelihood when the shape parameter xi is zero
def log_likelihood_0(list_of_z, xi, mu, sigma):
    m = len(list_of_z)
    summation = 0
    for z in list_of_z:
        summation += l_xi_0_summation(z,xi,mu,sigma, m)
    return summation+l_xi_0_non_summation(10,xi, mu,sigma, m)

#calculates all second order derivatives to compute the observed Fisher information matrix

#second order derivatives when the shape parameter xi is non-zero
l_xixi_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_summation, xi,2))
l_xixi_non_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_non_summation, xi,2))

def l_xixi_neq_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_xixi_sum(z,xi,mu,sigma, m)
    return summation+l_xixi_non_sum(10,xi, mu,sigma, m) #as the non-summation part does not depend on z, we can set it to whatever

l_ximu_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_summation, xi,mu))
l_ximu_non_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_non_summation, xi,mu))

def l_ximu_neq_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_ximu_sum(z,xi,mu,sigma, m)
    return summation+l_ximu_non_sum(10,xi, mu,sigma, m) 

l_xisigma_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_summation, xi,sigma))
l_xisigma_non_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_non_summation, xi,sigma))

def l_xisigma_neq_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_xisigma_sum(z,xi,mu,sigma, m)
    return summation+l_xisigma_non_sum(10,xi, mu,sigma, m) 

l_mumu_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_summation, mu,2))
l_mumu_non_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_non_summation, mu, 2))

def l_mumu_neq_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_mumu_sum(z,xi,mu,sigma, m)
    return summation+l_mumu_non_sum(10,xi, mu,sigma, m) 

l_musigma_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_summation, mu,sigma))
l_musigma_non_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_non_summation, mu, sigma))

def l_musigma_neq_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_musigma_sum(z,xi,mu,sigma, m)
    return summation+l_musigma_non_sum(10,xi, mu,sigma, m) 

l_sigmasigma_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_summation, sigma,2))
l_sigmasigma_non_sum = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_neq_0_non_summation, sigma, 2))

def l_sigmasigma_neq_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_sigmasigma_sum(z,xi,mu,sigma, m)
    return summation+l_sigmasigma_non_sum(10,xi, mu,sigma, m) 

#second order derivatives when the shape parameter xi is zero
l_xixi_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_summation, xi,2))
l_xixi_non_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_non_summation, xi,2))

def l_xixi_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_xixi_sum_0(z,xi,mu,sigma, m)
    return summation+l_xixi_non_sum_0(10,xi, mu,sigma, m) #as the non-summation part does not depend on z, we can set it to whatever

l_ximu_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_summation, xi,mu))
l_ximu_non_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_non_summation, xi,mu))

def l_ximu_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_ximu_sum_0(z,xi,mu,sigma, m)
    return summation+l_ximu_non_sum_0(10,xi, mu,sigma, m) 

l_xisigma_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_summation, xi,sigma))
l_xisigma_non_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_non_summation, xi,sigma))

def l_xisigma_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_xisigma_sum_0(z,xi,mu,sigma, m)
    return summation+l_xisigma_non_sum_0(10,xi, mu,sigma, m) 

l_mumu_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_summation, mu,2))
l_mumu_non_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_non_summation, mu, 2))

def l_mumu_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_mumu_sum_0(z,xi,mu,sigma, m)
    return summation+l_mumu_non_sum_0(10,xi, mu,sigma, m) 

l_musigma_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_summation, mu,sigma))
l_musigma_non_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_non_summation, mu, sigma))

def l_musigma_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_musigma_sum_0(z,xi,mu,sigma, m)
    return summation+l_musigma_non_sum_0(10,xi, mu,sigma, m) 

l_sigmasigma_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_summation, sigma,2))
l_sigmasigma_non_sum_0 = lambdify([z,xi,mu,sigma, m], -diff(likelihood_xi_0_non_summation, sigma, 2))

def l_sigmasigma_0(list_of_z, xi, mu, sigma, m):
    summation = 0
    for z in list_of_z:
        summation += l_sigmasigma_sum_0(z,xi,mu,sigma, m)
    return summation+l_sigmasigma_non_sum_0(10,xi, mu,sigma, m) 

#calculates the observed Fisher information matrix
def observed_Fisher(data, xi, mu, sigma, patience = 0.0001):
    m = len(data)
    if abs(xi)<patience:
        return np.array([[l_xixi_0(data, xi, mu, sigma, m), l_ximu_0(data, xi, mu, sigma, m), l_xisigma_0(data, xi, mu, sigma, m)],
                    [l_ximu_0(data, xi, mu, sigma, m),l_mumu_0(data, xi, mu, sigma, m), l_musigma_0(data, xi, mu, sigma, m)],
                    [l_xisigma_0(data, xi, mu, sigma, m), l_musigma_0(data, xi, mu, sigma, m), l_sigmasigma_0(data, xi, mu, sigma, m)]])
    else:
        return np.array([[l_xixi_neq_0(data, xi, mu, sigma, m), l_ximu_neq_0(data, xi, mu, sigma, m), l_xisigma_neq_0(data, xi, mu, sigma, m)],
                    [l_ximu_neq_0(data, xi, mu, sigma, m),l_mumu_neq_0(data, xi, mu, sigma, m), l_musigma_neq_0(data, xi, mu, sigma, m)],
                    [l_xisigma_neq_0(data, xi, mu, sigma, m), l_musigma_neq_0(data, xi, mu, sigma, m), l_sigmasigma_neq_0(data, xi, mu, sigma, m)]])

#calculates the variance of the estimated parameters based on the observed Fisher information, which has validity based on the CLT for maximum likelihood estimators 
def variance(data, xi, mu, sigma, patience = 0.0001):
    return np.linalg.inv(observed_Fisher(data, xi, mu, sigma))

class GEV:
    '''
    Model a distribuin of data using the generalized extreme value family, where we take the definition given by scipy.stats.genextreme.
    Notice that scipy takes the opposite value for the shape parameter xi as taken by Coles, S. (2013). An introduction to statistical modeling of extreme values. Springer Science & Business Media.  
    Input: data: dataset, supposed to be modeled by GEV family; patience = 0.0001: value in which we take the xi=0, that is, if abs(xi)<0.0001, we take xi = 0 (so the distribution comes from a Gubel family).
    '''
    #initiates the GEV class calculating the maximum likelihood fit parameters and variance
    def __init__(self, data, patience = 0.0001):
        self.data = data
        self.MLE_parameters = gev.fit(data)
        self.xi = -self.MLE_parameters[0] #the scipy package uses the opposite convesion for the xi shape parameter for the GEV distribution
        self.mu = self.MLE_parameters[1]
        self.sigma = self.MLE_parameters[2]
        self.variance = variance(self.data, self.xi, self.mu, self.sigma, patience)
    
    def parameters(self, alpha = 0.05, verbose = True):
        #calculates z_alpha/2 value
        z_crit = stats.norm.ppf(1-alpha/2)
        if verbose == True:
            print('xi =', self.xi,'    ','[', self.xi-z_crit*np.sqrt(self.variance[0,0]), ',',self.xi+z_crit*np.sqrt(self.variance[0,0]), ']')
            print('mu =', self.mu,'    ','[', self.mu-z_crit*np.sqrt(self.variance[0,0]), ',',self.mu+z_crit*np.sqrt(self.variance[0,0]), ']')
            print('sigma =', self.sigma,'    ','[', self.sigma-z_crit*np.sqrt(self.variance[0,0]), ',',self.sigma+z_crit*np.sqrt(self.variance[0,0]), ']')
        return [(self.xi, self.xi-z_crit*np.sqrt(self.variance[0,0]), self.xi+z_crit*np.sqrt(self.variance[0,0])),(self.mu, self.mu-z_crit*np.sqrt(self.variance[1,1]), self.mu+z_crit*np.sqrt(self.variance[1,1])),(self.sigma, self.sigma-z_crit*np.sqrt(self.variance[1,1]), self.sigma+z_crit*np.sqrt(self.variance[1,1]))]
    
    def log_likelihood(self, patience = 0.0001):
        '''
        Computes the maximum log-likelihood for the dataset
        Input: patience = 0.0001: if abs(xi)<=patience, uses the shape xi = 0 model
        Output: the maximum log-likelihood for the dataset
        '''
        if abs(self.xi)>patience:
            return log_likelihood_neq_0(self.data, self.xi, self.mu, self.sigma)
        else:
            return log_likelihood_0(self.data, self.xi, self.mu, self.sigma)
    
    def histogram(self):
        '''
        Plots the histogram of the data together with the maximum likelihood estimation of density
        Input: None
        '''
        xi = self.xi
        mu = self.mu
        sigma = self.sigma
        x = np.linspace(min(self.data),max(self.data),100)
        
        #recall the opposite convention for the shape parameter xi
        plt.plot(x,gev.pdf(x, -xi, mu, sigma))
        plt.hist(self.data, density=True)
        plt.title('Histogram')
        plt.show()

    def return_level(self, p, patience = 0.0001, confidence = False, alpha = 0.05):
        '''
        Returns the return level z_p of the maximum likelihood estimated distribution. If confidence == True, then also returns the (1-alpha)-confidence interval of z_p
        Input: p: the probability where to calculate the return level; patience = 0.0001: value in which we take the xi=0, that is, if abs(xi)<0.0001, we take xi = 0 (so the distribution comes from a Gubel family); confidence = False: if True,  returns the (1-alpha)-confidence interval of z_p; alpha = 0.05: if confidence = True, measures the (1-alpha)-confidence interval
        Output: if confidence == False, returns level z_p
        if confidence == True, returns (level z_p, min value of confidence interval, max value of confidence interval)
        '''
        xi = self.xi
        mu = self.mu
        sigma = self.sigma
        variance = self.variance
        y = -np.log(1-p)
        
        if abs(xi)>patience:
            zp = mu-sigma/xi*(1-y**-xi) 
        else:
            zp = mu-sigma*log(y)

        if confidence == False:
            return zp
        else:
            delta_z = np.array([1, -xi**(-1)*(1-y**(-xi)),sigma*xi**(-2)*(1-y**(-xi))-sigma*xi**(-1)*y**(-xi)*np.log(y)])
            part_1 = variance.dot(delta_z.transpose())
            variance_return = delta_z.dot(part_1)
            
            #calculates z_alpha/2 value
            z_crit = stats.norm.ppf(1-alpha/2)
            confidence_level = z_crit*np.sqrt(variance_return)
            return (zp, zp-confidence_level,zp+confidence_level)
        
    def return_plot(self, patience = 0.0001, confidence = False, alpha = 0.05):
        '''
        Plots the return plot, that is, the values for -log(y_p) vs. z_p for the maximum likelihood estimation model. Also plots the maximum likelihood estimation for the Gubel family (where the shape xi = 0).
        If confidence = True, also plots the region of (1-alpha)-confidence interval for the z_p estimates at each p valeu
        Input:  patience = 0.0001: value in which we take the xi=0, that is, if abs(xi)<0.0001, we take xi = 0 (so the distribution comes from a Gubel family); confidence = False: if True,  plots the (1-alpha)-confidence region of z_p for all p; alpha = 0.05: if confidence = True, measures the (1-alpha)-confidence interval
        '''
        xi = self.xi
        mu = self.mu
        sigma = self.sigma
        variance = self.variance

        probabilities = np.linspace(0.001,0.999,1000)
        y = -np.log(1-probabilities)
        zero_epsilon = mu-sigma*np.log(y)

        if confidence == False:
            levels = [self.return_level(prob, patience = patience) for prob in probabilities]    
            plt.plot(-np.log(y),levels)
            plt.plot(-np.log(y), zero_epsilon, linestyle='dashed', linewidth=0.5, label ='Zero Shape')
            plt.legend(loc='lower right')
            plt.title('Return Plot')
            plt.ylabel('Return Level')
            plt.xlabel('-log y')
            plt.show()
        else: 
            levels = [self.return_level(prob, patience = patience, confidence = True, alpha = alpha)[0] for prob in probabilities]
            max_value = [self.return_level(prob, patience = patience, confidence = True, alpha = alpha)[2] for prob in probabilities]
            min_value = [self.return_level(prob, patience = patience, confidence = True, alpha = alpha)[1] for prob in probabilities]

            plt.plot(-np.log(y), levels)
            plt.plot(-np.log(y), zero_epsilon, linestyle='dashed', linewidth=0.5, label ='Zero Shape')
            plt.plot(-np.log(y), min_value, linestyle='solid', color = 'green', linewidth=0.7)
            plt.plot(-np.log(y), max_value,  linestyle='solid', color = 'green', linewidth=0.7)
            plt.fill_between(-np.log(y), min_value, max_value, color = 'green', alpha =0.05)
            plt.legend(loc='lower right')
            plt.title('Return Plot')
            plt.ylabel('Return Level')
            plt.xlabel('-log y')
            plt.show()
        
    def probability_plot(self):
        '''
        Plots the probability diagram, which is the graph of (G(z_(i),i/(m+1)), where z_(i) are the order statistics of the input data, of length m, and G the cumulative distribution.
        The better the model, the closer the points lie to the 45-degree line 
        Output: None
        '''
        xi = self.xi
        mu = self.mu
        sigma = self.sigma

        order_data = list(self.data)
        order_data.sort()

        G_empirical = np.array(range(1,len(self.data)+1))/(len(self.data)+1)
        G_model = np.exp(-(1+xi/sigma*(order_data-mu))**(-1/xi))

        plt.scatter(G_empirical, G_model, color = 'red', marker =  'x')
        plt.plot([0,1],[0,1], linestyle='dashed', linewidth=0.5)
        plt.title('Probability Plot')


    def quantile_plot(self):
        '''
        Plots the quantile diagram, which is the graph of (G^(-1)(z_(i),z_(i)), where z_(i) are the order statistics of the input data, of length m, and G the cumulative distribution.
        The better the model, the closer the points lie to the 45-degree line 
        Output: None
        '''
        xi = self.xi
        mu = self.mu
        sigma = self.sigma

        order_data = list(self.data)
        order_data.sort()

        i_empirical = np.array(range(1,len(self.data)+1))/(len(self.data)+1)
        G_model = mu-sigma/xi*(1-(-np.log(i_empirical))**(-xi))

        plt.scatter(G_model, order_data, color = 'red', marker =  'x')
        plt.plot([min(G_model),max(G_model)],[min(G_model),max(G_model)], linestyle='dashed', linewidth=0.5)
        plt.title('Quantile Plot')