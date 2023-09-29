# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:55:13 2023

@author: Anna, Carmen
"""
import numpy as np
import matplotlib.pyplot as plt

class Disease:
    def __init__(self, a=1/9, b=0.21, g=0, 
                 W=np.array([[0,0.53,2.4],[0.93,0,0.3],[8.2,0.58,0]])*10E-4,
                 pop_0=np.array([7E6,100,0]), T=200, h=1 ):
        
        self.alpha = a
        self.beta = b
        self.gamma = g
        # Transport matrix: [default as in Jakobs example]
        self.W = W
        
        # Time and population 
        self.t0 = 0
        self.T = T # total time [default: 200 days]
        self.population_0 = pop_0 # inital population [S0, I0, R0] (y0)
        self.N = round(np.sum(self.population_0), 0) # Overall population
        
        #Euler & RK parameters
        self.h = h # step size [default: 1 day]
        # Number of timesteps and proof if h_step and overall number gives a whole number of time steps.
        # [X] This used to be in RungeKuttaLoop, but I need it for Euler as well
        self.n = self.T/self.h
        if self.n != int(self.n):                  
            print('Take another overall time pls.')
            exit()
        self.n = int(self.n)
        
        #Results
        self.x = np.linspace(self.t0, self.T, (self.n)+1) #array of times
        # RK
        self.y_RK_SIS = np.empty(((self.n)+1,3))
        self.theo_RK_SIS = np.empty(((self.n)+1,3))
        self.y_RK_SIR = np.empty(((self.n)+1,3))
        self.y_RK_SIT = np.empty(((self.n)+1,3))
        
        # plain Euler
        self.y_pEu_SIS = np.empty(((self.n)+1,3))
        self.y_pEu_SIR = np.empty(((self.n)+1,3))
        self.y_pEu_SIT = np.empty(((self.n)+1,3))
        
        # other Euler TODO
        
        
    ########## Equations for models (You can see that I wrote the equations into 
    # a Numpy array i.e. [dS/dT, dI/dt, dR/dt] = [equation 1, equation 2, equation 3]  
    # S = pop[0], I = pop[1], R = pop[2], pop means population

    ## SIS model: (time, population) 
    def SIS_model(self, t, pop):
        return np.array([-self.beta*pop[0]*pop[1]/self.N + self.alpha*pop[1], 
                         self.beta*pop[0]*pop[1]/self.N  - self.alpha*pop[1], 0]) 
    # added a zero as third place to prepare for three variables for the other 
    #models and generalize runge kutta
    
    ## SIR model:
    def SIR_model(self, t, pop):
        return np.array([-self.beta*pop[0]*pop[1]/self.N - self.gamma*pop[0], 
                         self.beta*pop[0]*pop[1]/self.N  - self.alpha*pop[1], 
                         self.gamma*pop[0] + self.alpha*pop[1]])
    
    ## SIR model with travel:
    def SIT_model(self, t, pop):
            return #TO DO
    
    # Runge Kutta Step (start time, pop, model) 
    def RungeKuttaStep(self, t, y, model): #y == pop
        
        # Overall of people:
        n = sum(y) # Number of people has to stay the same all time

        # ALgorithm stops if overall population changes (TODO: maybe has to be changed)
        if round(n,0) != self.N:
             print('Population number changed')
             exit()

        # Calculation of slopes (k_i are 1 X 3 vectors, where the last row is always zero in case of SIS model) 
        k1,k2,k3,k4 = 0,0,0,0
        
        if model == "SIS_model":
            k1 = self.SIS_model(t, y) 
            k2 = self.SIS_model(t + self.h/2, y + k1*self.h/2)
            k3 = self.SIS_model(t + self.h/2, y + k2*self.h/2)
            k4 = self.SIS_model(t + self.h, y + k3*self.h)
        
        elif model == "SIR_model":
            k1 = self.SIR_model(t, y) 
            k2 = self.SIR_model(t + self.h/2, y + k1*self.h/2)
            k3 = self.SIR_model(t + self.h/2, y + k2*self.h/2)
            k4 = self.SIR_model(t + self.h, y + k3*self.h)
        
        elif model == "SIT_model":
            k1 = self.SIT_model(t, y) 
            k2 = self.SIT_model(t + self.h/2, y + k1*self.h/2)
            k3 = self.SIT_model(t + self.h/2, y + k2*self.h/2)
            k4 = self.SIT_model(t + self.h, y + k3*self.h)

        # average slope
        K = 1/6*(k1 + 2.0*k2 + 2.0*k3 + k4)

        # return next time step value ([S, I, R]) 
        return  y + K*self.h
    
    
    
    # Runge Kutta Loop (model)
    #y0 is population_0
    def RungeKuttaLoop(self, model = "SIS_model"):  
        
        #checking that the model introduced is correct
        if model != "SIS_model" and model != "SIR_model" and model != "SIT_model":
            print('That model does not exist')
            exit()
        
        if model == "SIS_model":
            
            # inital condition and RungeKuttaStep applied n times
            y0 = self.population_0 #to be consistent with your notation
            self.y_RK_SIS[0] = y0
            
            for i in range(1, (self.n)+1):
                self.y_RK_SIS[i] = self.RungeKuttaStep(self.x[i-1], self.y_RK_SIS[i-1], model)
                
        elif model == "SIR_model":
            
            y0 = self.population_0 
            self.y_RK_SIR[0] = y0
            
            for i in range(1, (self.n)+1):
                self.y_RK_SIR[i] = self.RungeKuttaStep(self.x[i-1], self.y_RK_SIR[i-1], model)
        
        elif model == "SIT_model":
            
            y0 = self.population_0 
            self.y_RK_SIT[0] = y0
            
            for i in range(1, (self.n)+1):
                self.y_RK_SIT[i] = self.RungeKuttaStep(self.x[i-1], self.y_RK_SIT[i-1], model)
        
    
        
    # plain Euler (model)
    def Euler(self, model = "SIS_model"):
        
        #checking that the model introduced is correct
        if model != "SIS_model" and model != "SIR_model" and model != "SIT_model":
            print('That model does not exist')
            exit()
        
        if model == "SIS_model":
            
            # inital condition and EulerStep applied n times
            y0 = self.population_0 #to be consistent with your notation
            self.y_pEu_SIS[0] = y0
            
            for i in range(1, (self.n)+1):
                self.y_pEu_SIS[i] = self.y_pEu_SIS[i-1] + self.h * self.SIS_model(self.x[i-1], self.y_pEu_SIS[i-1])
        
        elif model == "SIR_model":
            
            y0 = self.population_0 
            self.y_pEu_SIR[0] = y0
            
            for i in range(1, (self.n)+1):
                self.y_pEu_SIR[i] = self.y_pEu_SIR[i-1] + self.h * self.SIR_model(self.x[i-1], self.y_pEu_SIR[i-1])
        
        elif model == "SIT_model":
            
            y0 = self.population_0 
            self.y_pEu_SIT[0] = y0
            
            for i in range(1, (self.n)+1):
                self.y_pEu_SIT[i] = self.y_pEu_SIT[i-1] + self.h * self.SIT_model(self.x[i-1], self.y_pEu_SIT[i-1])
        
            
    #[] where does this come from? It is a great idea!
    def Theo_SIS_model(self):
                        
        self.theo_RK_SIS[0]  = self.population_0

        I_inf = self.N*(1-self.alpha/self.beta)
        r = I_inf/self.population_0[1] -1
        delta = self.beta - self.alpha

        for i in range(1,len(self.theo_RK_SIS)): 
            I_i = I_inf/(1 + r*np.exp(delta*self.x[i]))
            self.theo_RK_SIS[i] = [ self.N - I_i, I_i , 0]

 
#includes several results in the same figure  
# xs is a list of x-type lists
# ys is a list of y-type lists   
#labels is the list of labels, one label per each line in the plot  
# xlabel and ylabel refer to the whole plot     
def plot(xs, ys, labels, title, xlabel, ylabel):
    plt.style.use('rc.mplstyle')
    fig, ax = plt.subplots(figsize = (3,4))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for line in range(len(xs)):
        ax.plot(xs[line], ys[line], label = labels[line])
        
    ax.legend()
    plt.show()
    plt.savefig(title)       
