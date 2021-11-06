import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto as genpar
from scipy import stats
from sympy import symbols, log, lambdify, exp, diff                    

#uses sympy analytic library to take the analytic derivatives of the likelihood
y, xi, sigma, k = symbols('y xi sigma k')

#we will split the likelihood into the part non-dependent in y, called non-summation, and the part that depends in y, as we will need to sum it all over
#when the shape parameter xi is non-zero
likelihood_xi_neq_0_non_summation = -k*log(sigma)
likelihood_xi_neq_0_summation = -(1+1/xi)*log(1+xi*y/sigma)
likelihood_xi_neq_0 = likelihood_xi_neq_0_non_summation+likelihood_xi_neq_0_summation
l_xi_neq_0_summation = lambdify([y, xi, sigma, k],likelihood_xi_neq_0_summation)
l_xi_neq_0_non_summation = lambdify([y, xi, sigma, k],likelihood_xi_neq_0_non_summation)

#when the shape parameter xi is zero
likelihood_xi_0_non_summation = -k*log(sigma)
likelihood_xi_0_summation = -1/sigma*log(y)
likelihood_xi_0 = likelihood_xi_0_non_summation+likelihood_xi_0_summation
l_xi_0_summation = lambdify([y, xi, sigma, k],likelihood_xi_0_summation)
l_xi_0_non_summation = lambdify([y, xi, sigma, k],likelihood_xi_0_non_summation)

#calculates the log_likelihood when the shape parameter xi is non-zero
def log_likelihood_neq_0(list_of_y, xi, sigma):
    k = len(list_of_y)
    summation = 0
    for y in list_of_y:
        summation += l_xi_neq_0_summation(y, xi, sigma, k)
    return summation+l_xi_neq_0_non_summation(10, xi, sigma, k)

#calculates the log_likelihood when the shape parameter xi is zero
def log_likelihood_0(list_of_y, xi, sigma):
    k = len(list_of_y)
    summation = 0
    for y in list_of_y:
        summation += l_xi_0_summation(y, xi, sigma, k)
    return summation+l_xi_0_non_summation(10, xi, sigma, k)

#calculates all second order derivatives to compute the observed Fisher information matrix

#second order derivatives when the shape parameter xi is non-zero
l_xixi_sum = lambdify([y, xi, sigma, k], -diff(likelihood_xi_neq_0_summation, xi,2))
l_xixi_non_sum = lambdify([y, xi, sigma, k], -diff(likelihood_xi_neq_0_non_summation, xi,2))

def l_xixi_neq_0(list_of_y, xi, sigma, k):
    summation = 0
    for y in list_of_y:
        summation += l_xixi_sum(y, xi, sigma, k)
    return summation+l_xixi_non_sum(10, xi, sigma, k) #as the non-summation part does not depend on z, we can set it to whatever

l_xisigma_sum = lambdify([y, xi, sigma, k], -diff(likelihood_xi_neq_0_summation, xi,sigma))
l_xisigma_non_sum = lambdify([y, xi, sigma, k], -diff(likelihood_xi_neq_0_non_summation, xi,sigma))

def l_xisigma_neq_0(list_of_y, xi, sigma, k):
    summation = 0
    for y in list_of_y:
        summation += l_xisigma_sum(y, xi, sigma, k)
    return summation+l_xisigma_non_sum(10, xi, sigma, k) 

l_sigmasigma_sum = lambdify([y, xi, sigma, k], -diff(likelihood_xi_neq_0_summation, sigma,2))
l_sigmasigma_non_sum = lambdify([y, xi, sigma, k], -diff(likelihood_xi_neq_0_non_summation, sigma, 2))

def l_sigmasigma_neq_0(list_of_y, xi, sigma, k):
    summation = 0
    for y in list_of_y:
        summation += l_sigmasigma_sum(y, xi, sigma, k)
    return summation+l_sigmasigma_non_sum(10, xi, sigma, k) 

#second order derivatives when the shape parameter xi is zero
l_xixi_sum_0 = lambdify([y, xi, sigma, k], -diff(likelihood_xi_0_summation, xi,2))
l_xixi_non_sum_0 = lambdify([y, xi, sigma, k], -diff(likelihood_xi_0_non_summation, xi,2))

def l_xixi_0(list_of_y, xi, sigma, k):
    summation = 0
    for y in list_of_y:
        summation += l_xixi_sum_0(y, xi, sigma, k)
    return summation+l_xixi_non_sum_0(10, xi, sigma, k) #as the non-summation part does not depend on z, we can set it to whatever

l_xisigma_sum_0 = lambdify([y, xi, sigma, k], -diff(likelihood_xi_0_summation, xi,sigma))
l_xisigma_non_sum_0 = lambdify([y, xi, sigma, k], -diff(likelihood_xi_0_non_summation, xi,sigma))

def l_xisigma_0(list_of_y, xi, sigma, k):
    summation = 0
    for y in list_of_y:
        summation += l_xisigma_sum_0(y, xi, sigma, k)
    return summation+l_xisigma_non_sum_0(10, xi, sigma, k) 

l_sigmasigma_sum_0 = lambdify([y, xi, sigma, k], -diff(likelihood_xi_0_summation, sigma,2))
l_sigmasigma_non_sum_0 = lambdify([y, xi, sigma, k], -diff(likelihood_xi_0_non_summation, sigma, 2))

def l_sigmasigma_0(list_of_y, xi, sigma, k):
    summation = 0
    for y in list_of_y:
        summation += l_sigmasigma_sum_0(y, xi, sigma, k)
    return summation+l_sigmasigma_non_sum_0(10, xi,sigma, k)

#calculates the observed Fisher matrix of only xi and sigma
def observed_Fisher(data, xi, sigma, patience = 0.0001):
    k = len(data)
    if abs(xi)<patience:
        return np.array([[float(l_xixi_0(data, xi, sigma, k)), float(l_xisigma_0(data, xi, sigma, k))],
                    [float(l_xisigma_0(data, xi, sigma, k)), float(l_sigmasigma_0(data, xi, sigma, k))]])
    else:
        return np.array([[float(l_xixi_neq_0(data, xi, sigma, k)), float(l_xisigma_neq_0(data, xi, sigma, k))],
                    [float(l_xisigma_neq_0(data, xi, sigma, k)),float(l_sigmasigma_neq_0(data, xi, sigma, k))]])
    
#calculates the variance of onlt the xi and sigma parameters
def variance_xisigma(data, xi, sigma, patience = 0.0001):
    return np.linalg.inv(observed_Fisher(data, xi, sigma))

#calculates the full variance, including uncertainty on the probabilty of exceeding the threshold prob
def variance(data, full_data, xi, sigma, prob, patience = 0.0001):
    '''
    Calculates the covariance matrix of the estimation parameters in order prob, xi, sigma
    '''
    n = len(full_data)
    return np.array([[prob*(1-prob)/n, 0, 0],
                    [0, variance_xisigma(data, xi, sigma)[0,0], variance_xisigma(data, xi, sigma)[0,1]],
                    [0, variance_xisigma(data, xi, sigma)[1,0], variance_xisigma(data, xi, sigma)[1,1]]])

class GeneralizedPareto:
    '''
    Model a distribuin above a threshold (i.e. of form (data-threshold|data>threshold) using a 2-parameter the generalized Pareto family, based on Pickands–Balkema–De Haan theorem, where we take the definition given by scipy.stats.genextreme.
    Input: data: dataset, supposed to be modeled by generalized Pareto family; thresholh: value for which picking the threshold, see genpareto.mean_residual or genpareto.choose_threshold for choosing a good value of threshold; patience = 0.0001: value in which we take the xi=0, that is, if abs(xi)<0.0001, we take xi = 0 (so the distribution comes from a Gubel family).
    '''
    def __init__(self, data, threshold, patience = 0.0001):
        self.data = data
        self.threshold = threshold
        
        self.filtered_data = data[data > threshold]-threshold
        
        parameters = genpar.fit(self.filtered_data, floc = 0.00) #by Pickands–Balkema–De Haan theorem, the threshold distribution approaches a generalized Pareto with location parameter mu = 0
        
        self.xi = parameters[0]
        self.sigma = parameters[2]
        self.prob = len(self.filtered_data)/len(self.data)
        
        self.variance = variance(self.filtered_data, self.data, self.xi, self.sigma, self.prob, patience = patience)
    
    def parameters(self, alpha = 0.05, verbose = True):
        #calculates z_alpha/2 value
        z_crit = stats.norm.ppf(1-alpha/2)
        if verbose == True:
            print('prob =', self.prob,'    ','[', self.prob-z_crit*np.sqrt(self.variance[0,0]), ',',self.prob+z_crit*np.sqrt(self.variance[0,0]), ']')
            print('xi =', self.xi,'    ','[', self.xi-z_crit*np.sqrt(self.variance[1,1]), ',',self.xi+z_crit*np.sqrt(self.variance[1,1]), ']')
            print('sigma =', self.sigma,'    ','[', self.sigma-z_crit*np.sqrt(self.variance[2,2]), ',',self.sigma+z_crit*np.sqrt(self.variance[2,2]), ']')
        return [(self.prob, self.prob-z_crit*np.sqrt(self.variance[0,0]),self.prob+z_crit*np.sqrt(self.variance[0,0])),(self.xi, self.xi-z_crit*np.sqrt(self.variance[1,1]), self.xi+z_crit*np.sqrt(self.variance[1,1])),(self.sigma, self.sigma-z_crit*np.sqrt(self.variance[2,2]), self.sigma+z_crit*np.sqrt(self.variance[2,2]))]
    
    def log_likelihood(self, patience = 0.0001):
        '''
        Computes the maximum log-likelihood for the dataset
        Input: patience = 0.0001: if abs(xi)<=patience, uses the shape xi = 0 model
        Output: the maximum log-likelihood for the dataset
        '''
        if abs(self.xi)>patience:
            return log_likelihood_neq_0(self.filtered_data, self.xi, self.sigma)
        else:
            return log_likelihood_0(self.filtered_data, self.xi, self.sigma)
    
    def histogram(self):
        '''
        Plots the histogram of the data together with the maximum likelihood estimation of density
        Input: None
        '''
        xi = self.xi
        sigma = self.sigma
        x = np.linspace(min(self.filtered_data),max(self.filtered_data),100)
        
        #recall the opposite convention for the shape parameter xi
        plt.plot(x,genpar.pdf(x, xi, 0, sigma))
        plt.hist(self.filtered_data, density=True)
        plt.title('Histogram')
        plt.show()

        
    def return_level(self, m, patience = 0.0001, confidence = False, alpha = 0.05):
        '''
        Returns the return level x_m of the maximum likelihood estimated distribution. If confidence == True, then also returns the (1-alpha)-confidence interval of x_m
        Input: m: the return period in observations for the maximum value to be exceeded; patience = 0.0001: value in which we take the xi=0, that is, if abs(xi)<0.0001, we take xi = 0 (so the distribution comes from a Gubel family); confidence = False: if True,  returns the (1-alpha)-confidence interval of z_p; alpha = 0.05: if confidence = True, measures the (1-alpha)-confidence interval
        Output: if confidence == False, returns level x_m
        if confidence == True, returns (level x_m, min value of confidence interval, max value of confidence interval)
        '''
        xi = self.xi
        sigma = self.sigma
        variance = self.variance
        threshold = self.threshold
        prob = self.prob
        
        if abs(xi)>patience:
            xm = threshold+sigma/xi*((m*prob)**xi-1) 
        else:
            xm = threshold+sigma*np.log(m*prob)

        if confidence == False:
            return xm
        
        else:
            delta_z = np.array([sigma*m**(xi)*prob**(xi-1),  -sigma*xi**(-2)*((m*prob)**xi-1)+sigma/xi*(m*prob)**xi*np.log(m*prob), 1/xi*((m*prob)**xi-1)])
            part_1 = variance.dot(delta_z.transpose())
            variance_return = delta_z.dot(part_1)
            
            #calculates z_alpha/2 value
            z_crit = stats.norm.ppf(1-alpha/2)
            confidence_level = z_crit*np.sqrt(variance_return)
            return (xm, xm-confidence_level,xm+confidence_level)
        
    def return_plot(self, patience = 0.0001, confidence = False, alpha = 0.05):
        '''
        Plots the return plot, that is, the values for -log(m) vs. xm for the maximum likelihood estimation model. Also plots the maximum likelihood estimation for the Gubel family (where the shape xi = 0).
        If confidence = True, also plots the region of (1-alpha)-confidence interval for the xm estimates at each p valeu
        Input:  patience = 0.0001: value in which we take the xi=0, that is, if abs(xi)<0.0001, we take xi = 0 (so the distribution comes from a Gubel family); confidence = False: if True,  plots the (1-alpha)-confidence region of xm for all m; alpha = 0.05: if confidence = True, measures the (1-alpha)-confidence interval
        '''
        xi = self.xi
        sigma = self.sigma
        variance = self.variance
        threshold = self.threshold
        prob = self.prob

        m_list = np.linspace(100, 100*10000, 10000)
        zero_xi = threshold+sigma*np.log(m_list*prob)

        if confidence == False:
            levels = [self.return_level(m, patience = patience) for m in m_list]    
            plt.plot(np.log(m_list), levels)
            plt.plot(np.log(m_list), zero_xi, linestyle='dashed', linewidth=0.5, label ='Zero Shape')
            plt.legend(loc='lower right')
            plt.title('Return Plot')
            plt.ylabel('Return Level')
            plt.xlabel('log m')
            plt.show()
        else: 
            levels = [self.return_level(m, patience = patience, confidence = True, alpha = alpha)[0] for m in m_list]
            max_value = [self.return_level(m, patience = patience, confidence = True, alpha = alpha)[2] for m in m_list]
            min_value = [self.return_level(m, patience = patience, confidence = True, alpha = alpha)[1] for m in m_list]

            plt.plot(np.log(m_list), levels)
            plt.plot(np.log(m_list), zero_xi, linestyle='dashed', linewidth=0.5, label ='Zero Shape')
            plt.plot(np.log(m_list), min_value, linestyle='solid', color = 'green', linewidth=0.7)
            plt.plot(np.log(m_list), max_value,  linestyle='solid', color = 'green', linewidth=0.7)
            plt.fill_between(np.log(m_list), min_value, max_value, color = 'green', alpha =0.05)
            plt.legend(loc='lower right')
            plt.title('Return Plot')
            plt.ylabel('Return Level')
            plt.xlabel('log m')
            plt.show()
        
        
    def probability_plot(self):
        '''
        Plots the probability diagram, which is the graph of (H(y_(i),i/(m+1)), where y_(i) are the order statistics of the input data, of length k, and H the cumulative distribution.
        The better the model, the closer the points lie to the 45-degree line 
        Output: None
        '''
        xi = self.xi
        sigma = self.sigma
        threshold = self.threshold

        order_data = list(self.filtered_data)
        order_data.sort()
        order_data = np.array(order_data)

        H_empirical = np.array(range(1,len(self.filtered_data)+1))/(len(self.filtered_data)+1)
        H_model = 1-(1+xi*order_data/sigma)**(-1/xi)

        plt.scatter(H_empirical, H_model, color = 'red', marker =  'x')
        plt.plot([0,1],[0,1], linestyle='dashed', linewidth=0.5)
        plt.title('Probability Plot')


    def quantile_plot(self):
        '''
        Plots the quantile diagram, which is the graph of (H^(-1)(z_(i),z_(i)), where z_(i) are the order statistics of the input data, of length m, and H the cumulative distribution.
        The better the model, the closer the points lie to the 45-degree line 
        Output: None
        '''
        xi = self.xi
        sigma = self.sigma
        threshold = self.threshold

        order_data = list(self.filtered_data)
        order_data.sort()
        order_data = np.array(order_data)

        i_empirical = np.array(range(1,len(order_data)+1))/(len(order_data)+1)
        H_model = sigma/xi*((1-i_empirical)**(-xi)-1)

        plt.scatter(H_model, order_data, color = 'red', marker =  'x')
        plt.plot([min(H_model),max(H_model)],[min(H_model),max(H_model)], linestyle='dashed', linewidth=0.5)
        plt.title('Quantile Plot')

def mean_residual(data, min_value = 0, mesh = 100):
    '''
    Plots the mean residual plot for a dataset, which should help picking a good threshold value as the beginning of linear regions, together with the 1 standard deviation interval value
    Inputs: data: dataset to be tested; min_value = 0, minimum threshold value to be tested, the maximum is max(data); mesh = 100: number of threshold points to be used in making the plot
    '''
    means = []
    stdvs = []
    us = np.linspace(min_value, max(data), mesh)
    for threshold in us:
        filtered_data = data[data>threshold]-threshold
        means.append(np.mean(filtered_data))
        stdvs.append(np.std(filtered_data))

    means = np.array(means)
    stdvs = np.array(stdvs)
    
    plt.plot(us, means)
    plt.plot(us, means+stdvs, linestyle='dashed', color = 'green', linewidth=0.5)
    plt.plot(us, means-stdvs, linestyle='dashed', color = 'green', linewidth=0.5)
    plt.title('Mean Residual Plot')
    return

def choose_threshold(data, min_value = 0, max_value = 50, mesh = 30):
    '''
    Plots some figures that help choosing perfect values of threshold. These are plots of the shape value xi and of a changed scale value sigma_st = sigma-xi*threshold. If some value of threshold u_0 will produce a filtered dataset that respects the generalized Pareto family, then these two quantities should be kept approximately constant for thresholds us bigger than u_0 (although not for too big values of u, as we break assymptotic convergence at these levels). Also plots 1 standard deviation error bars.
    Inputs: data: dataset to be tested; min_value = 0, minimum threshold value to be tested; max_value = 50: maximum threshold value to be tested; mesh = 100: number of threshold points to be used in making the plot
    '''
    xi = []
    stdvs_xi = []
    sigma_st = []
    stdvs_sigma = []
    
    us = np.linspace(min_value, max_value, mesh)
    
    for threshold in us:
        model = GeneralizedPareto(data, threshold)
        variance = model.variance
        stdvs_xi.append(np.sqrt(variance[1,1]))
        xi.append(model.xi)
        
        sigma_st.append(model.sigma-model.xi*threshold)
        #to find the error bars on sigma_st, we must use the delta method to calculate its variance (here, delta_sigma = [-u,1]) for only xi and sigma dependece, respectively
        #notice that we need only the variance matrix for the xi and sigma parameters
        variance_matrix = variance[1:3,1:3]
        det_sigma_st = np.array([-threshold, 1])
        part_1 = variance_matrix.dot(det_sigma_st.transpose())
        variance_return = det_sigma_st.dot(part_1)
        stdvs_sigma.append(np.sqrt(variance_return))
        
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 7))
    ax1.plot(us, xi)
    ax1.errorbar(us,xi, yerr=stdvs_xi, alpha = 0.3)
    ax1.title.set_text('Shape')
    ax2.plot(us, sigma_st)
    ax2.errorbar(us, sigma_st, yerr=stdvs_sigma, alpha = 0.3)
    ax2.title.set_text('Modified Scale')
    return 