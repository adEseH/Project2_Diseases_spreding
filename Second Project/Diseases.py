# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:55:13 2023

@author: Anna, Carmen
"""
import numpy as np
import matplotlib.pyplot as plt

#### City 1: Mexico City, City 2: Vancouver, City 3: New York

class Disease:
    def __init__(self, T=300, a=1/6, R0 = [1.72, 1.6, 1.7], g = [0.00253,0.0029, 0.0018],
                         pop_0=np.array([[19.985E6,10,0],[2.238E6,0,0], [18.309E6,0,0]]), h=0.2 ):
        
        # Time and population 
        self.R0 = R0
        self.a = a
        self.g = g

        self.t0 = 0
        self.T = T # total time [default: 200 days]
        self.population_0 = pop_0 # inital population [S0, I0, R0] (y0)
        self.N = round(sum(pop_0[0]) +sum(pop_0[1]) +sum(pop_0[2]), 0) # Overall population


        #### different choices of beta and gamma

        # with vaccination, constant beta, constant gamma
        if False:
            self.alpha = a
            # Calculation beta
            b = [a*R0[0], a*R0[1], a*R0[2]]
            self.beta_array = np.array([np.repeat(b[0], T/h),np.repeat(b[1], T/h),np.repeat(b[2], T/h)])
            self.gamma_array =  np.array([np.repeat(g[0], T/h),np.repeat(g[1], T/h),np.repeat(g[2], T/h)])

        elif True: # with vaccination, constant beta, gauss gamma
            self.T = T
            self.alpha = 1/6
            T_vec = np.arange(0,T,h)
            gamma_mexico = 180*self.g[0] / (np.sqrt(2.0 * np.pi) * 30) * np.exp(-np.power((T_vec - 90) / 30, 2.0) / 2)
            gamma_van = 180*self.g[1] / (np.sqrt(2.0 * np.pi) * 30) * np.exp(-np.power((T_vec - 90) / 30, 2.0) / 2)
            gamma_NY = 180*self.g[2] / (np.sqrt(2.0 * np.pi) * 30) * np.exp(-np.power((T_vec - 90) / 30, 2.0) / 2)
            # Calculation beta
            b = [a*R0[0], a*R0[1], a*R0[2]]
            self.beta_array= np.array([np.repeat(b[0], T/h),np.repeat(b[1], T/h),np.repeat(b[2], T/h)])

            self.gamma_array =  np.array([gamma_mexico,gamma_van,gamma_NY])

        elif False: # with vaccination, linear beta, gauss gamma
            self.T = T
            self.alpha = 1/6
            T_vec = np.arange(0,T,h)
            #beta
            b = [a*R0[0], a*R0[1], a*R0[2]]
            b1 = [np.repeat(b[0], 60*1/h), np.repeat(b[1], 60*1/h), np.repeat(b[2], 60*1/h)]       # until day 60 constant 
            b2M = -(R0[0]-0.3)*a/120*T_vec[int(60*1/h):int(180/h)] + b[0] - -(R0[0]-0.3)*a/120*60  # linear decrease
            b2V = -(R0[1]-0.3)*a/120*T_vec[int(60*1/h):int(180/h)] + b[0] - -(R0[1]-0.3)*a/120*60
            b2N = -(R0[2]-0.3)*a/120*T_vec[int(60*1/h):int(180/h)] + b[0] - -(R0[2]-0.3)*a/120*60
            b3 = np.repeat(0.3*a, T*1/h-(180)*1/h)# until end constant
            self.beta_array= np.array([np.hstack((b1[0], b2M, b3)),np.hstack((b1[1], b2V, b3)),np.hstack((b1[2], b2N, b3))])

            #gamma
            gamma_mexico = 180*self.g[0] / (np.sqrt(2.0 * np.pi) * 30) * np.exp(-np.power((T_vec - 90) / 30, 2.0) / 2)
            gamma_van = 180*self.g[1] / (np.sqrt(2.0 * np.pi) * 30) * np.exp(-np.power((T_vec - 90) / 30, 2.0) / 2)
            gamma_NY = 180*self.g[2] / (np.sqrt(2.0 * np.pi) * 30) * np.exp(-np.power((T_vec - 90) / 30, 2.0) / 2)
            self.gamma_array = g= np.array([gamma_mexico,gamma_van,gamma_NY])
        
        elif False: # no vaccination, linear beta
            self.T = T
            self.alpha = 1/6
            T_vec = np.arange(0,T,h)
            #beta
            b = [a*R0[0], a*R0[1], a*R0[2]]
            b1 = [np.repeat(b[0], 60*1/h), np.repeat(b[1], 60*1/h), np.repeat(b[2], 60*1/h)]       # until day 60 constant 
            b2M = -(R0[0]-0.3)*a/120*T_vec[int(60*1/h):int(180/h)] + b[0] - -(R0[0]-0.3)*a/120*60  # linear decrease
            b2V = -(R0[1]-0.3)*a/120*T_vec[int(60*1/h):int(180/h)] + b[0] - -(R0[1]-0.3)*a/120*60
            b2N = -(R0[2]-0.3)*a/120*T_vec[int(60*1/h):int(180/h)] + b[0] - -(R0[2]-0.3)*a/120*60
            b3 = np.repeat(0.3*a, T*1/h-(180)*1/h)# until end constant
            self.beta_array= np.array([np.hstack((b1[0], b2M, b3)),np.hstack((b1[1], b2V, b3)),np.hstack((b1[2], b2N, b3))])

            #gamma
            self.gamma_array =  np.array([np.repeat(0, T/h),np.repeat(0, T/h),np.repeat(0, T/h)])

        # No vaccination 
        elif False: # adapted values for mexico, PEAK FOR FIRST WAVE 
            self.alpha = 1/6
            #calculation beta
            b = [a*R0[0], a*R0[1], a*R0[2]]
            #self.beta_array= np.array([np.repeat(b[0], T/h),np.repeat(b[1], T/h),np.repeat(b[2], T/h)])
            self.beta_array= np.array([np.repeat(b[0], T/h),np.repeat(b[1], T/h),np.repeat(b[2], T/h)])
            self.gamma_array =  np.array([np.repeat(0, T/h),np.repeat(0, T/h),np.repeat(0, T/h)])


        elif False: # Quarantine in Mexico included, PEAK FOR FIRST WAVE  ### Recomments: infected = 10, no vaccination
            # TO
            self.T = 75
            R0 = [1.72, 1.6, 1.74]
            self.alpha = 1/6
            T_vec = np.arange(0,T,h)
            #calculation beta
            b = [a*R0[0], a*R0[1], a*R0[2]]
            b1 = np.repeat(b[0], 37*1/h)       # until day 37 constant 
            b15= -1.22*a/3*T_vec[int(37*1/h):int(40/h)] + 2.79 # until day 40 linear decrease
            b2 = np.repeat(0.5*a,11*5)         # until day 51 constant
            b25 = 0.4*a/5*T_vec[int(51*1/h):int(56/h)] - 0.596 # until day 56 linear increase
            b3 = np.repeat(0.9*a, T*1/h-(40 + 11 + 5)*1/h) # until end constant
            self.beta_array= np.array([np.hstack((b1, b15, b2, b25, b3)),np.repeat(b[1], T/h),np.repeat(b[2], T/h)])
            self.gamma_array =  np.array([np.repeat(0, T/h),np.repeat(0, T/h),np.repeat(0, T/h)])


        else: # No vaccination, constant beta
            self.alpha = a
            self.g = [0,0,0]
            b = [a*R0[0], a*R0[1], a*R0[2]]
            self.beta_array= np.array([np.repeat(b[0], T/h),np.repeat(b[1], T/h),np.repeat(b[2], T/h)])
            self.gamma_array =  np.array([np.repeat(0, T/h),np.repeat(0, T/h),np.repeat(0, T/h)])


        #print values
        print('T = ', T, 'Population0 = ', pop_0, 'beta = ' ,self.beta_array[:,0], 'gamma  = ', self.gamma_array[:,0], 'alpha = ', self.alpha)

        # initialise first values:
        self.beta = self.beta_array[:,0]
        self.gamma = self.gamma_array[:,0]

        # Transport matrix: 
        if True: # Traveling            
            
            #calculating probabilities
            W_NM = 0.14*6400000*0.06/pop_0[2,0]
            W_NV = 0.11*0.8*2.89E6/pop_0[2,0]
            W_MV = 0.7*0.8*63.5E3/pop_0[0,0]

            W_MN = W_NM*pop_0[2,0]/pop_0[0,0]
            W_VN = W_NV*pop_0[2,0]/pop_0[1,0]
            W_VM = W_MV*pop_0[0,0]/pop_0[1,0]

            self.W = np.array([[0, W_VM, W_NM],[W_MV ,0, W_NV],[W_MN , W_VN,0]])
            print(self.W)

        else: #No travaling
            self.W = np.array([[0, 0, 0],[0,0, 0],[0 , 0,0]]) 

        #Euler & RK parameters
        self.h = h # step size [default: 1 day]

        # Number of timesteps and proof if h_step and overall number gives a whole number of time steps
        self.n = self.T/self.h
        if self.n != int(self.n):                  
            print('Take another overall time pls.')
            exit()
        self.n = int(self.n)
        
        #Results
        self.x = np.linspace(self.t0, self.T, (self.n)+1) #array of times
        # RK
        self.y_RK_SIS = np.empty(((self.n)+1,3,3))
        self.theo_SIS = np.empty(((self.n)+1,3,3))
        self.y_RK_SIR = np.empty(((self.n)+1,3,3))
        self.y_RK_SIT = np.empty(((self.n)+1,3,3))
        
        # plain Euler
        self.y_pEu_SIS = np.empty(((self.n)+1,3,3))
        self.y_pEu_SIR = np.empty(((self.n)+1,3,3))
        self.y_pEu_SIT = np.empty(((self.n)+1,3,3))
        
        # other Euler TODO
        
        
    ########## Equations for models (You can see that I wrote the equations into 
    # a Numpy array i.e. [dS/dT, dI/dt, dR/dt] = [equation 1, equation 2, equation 3]  
    # S = pop[0], I = pop[1], R = pop[2], pop means population

    ## SIS model: (time, population) 
    def SIS_model(self,t, pop):
        pop0 = np.array([-self.beta[0]*pop[0,0]*pop[0,1]/self.N_i[0] + self.alpha*pop[0,1], self.beta[0]*pop[0,0]*pop[0,1]/self.N_i[0] - self.alpha*pop[0,1], 0]) # added a zero as third place to prepare for three variables for the other models and generalize runge kutta
        pop1 = np.array([0,0,0])
        pop2 = np.array([0,0,0])
        return np.array([pop0, pop1, pop2])

## SIR model:
    def SIR_model(self,t, pop):
        i = 0  # Choose city (0 = Mexico City, 1 = Vancouver, 3 = New York)
        pop0 = np.array([-self.beta[i]*pop[i,0]*pop[i,1]/self.N_i[i] - self.gamma[i]*pop[i,0], self.beta[i]*pop[i,0]*pop[i,1]/self.N_i[i]  - self.alpha*pop[i,1], self.gamma[i]*pop[i,0] + self.alpha*pop[i,1]])
        pop1 = np.array([0,0,0])
        pop2 = np.array([0,0,0])
        return np.array([pop0, pop1, pop2]) 

## SIR model with travel: 
    def SIT_model(self,t, pop):
            pop0 = np.array([-self.beta[0]*pop[0,0]*pop[0,1]/self.N_i[0] - self.gamma[0]*pop[0,0] +   (self.W[0,1]*pop[1,0] + self.W[0,2]*pop[2,0])-(self.W[1,0] + self.W[2,0])*pop[0,0], 
                                self.beta[0]*pop[0,0]*pop[0,1]/self.N_i[0]  - self.alpha*pop[0,1]+    (self.W[0,1]*pop[1,1] + self.W[0,2]*pop[2,1])-(self.W[1,0] + self.W[2,0])*pop[0,1], 
                                    self.gamma[0]*pop[0,0] + self.alpha*pop[0,1] +              (self.W[0,1]*pop[1,2] + self.W[0,2]*pop[2,2])-(self.W[1,0] + self.W[2,0])*pop[0,2]])
            
            pop1 = np.array([-self.beta[1]*pop[1,0]*pop[1,1]/self.N_i[1] - self.gamma[1]*pop[1,0] + (self.W[1,0]*pop[0,0] +self.W[1,2]*pop[2,0])-(self.W[0,1] + self.W[2,1])*pop[1,0], 
                                self.beta[1]*pop[1,0]*pop[1,1]/self.N_i[1]  - self.alpha*pop[1,1]+  (self.W[1,0]*pop[0,1] + self.W[1,2]*pop[2,1])-(self.W[0,1] + self.W[2,1])*pop[1,1], 
                                                self.gamma[1]*pop[1,0] + self.alpha*pop[1,1] + (self.W[1,0]*pop[0,2] + self.W[1,2]*pop[2,2])-(self.W[0,1] + self.W[2,1])*pop[1,2]])
            
            pop2 = np.array([-self.beta[2]*pop[2,0]*pop[2,1]/self.N_i[2] - self.gamma[2]*pop[2,0] + (self.W[2,0]*pop[0,0] + self.W[2,1]*pop[1,0])-(self.W[0,2] + self.W[1,2])*pop[2,0], 
                                self.beta[2]*pop[2,0]*pop[2,1]/self.N_i[2]  - self.alpha*pop[2,1]+  (self.W[2,0]*pop[0,1] + self.W[2,1]*pop[1,1])-(self.W[0,2] + self.W[1,2])*pop[2,1], 
                                                self.gamma[2]*pop[2,0] + self.alpha*pop[2,1] + (self.W[2,0]*pop[0,2] + self.W[2,1]*pop[1,2])-(self.W[0,2] + self.W[1,2])*pop[2,2]])
            

            return np.array([pop0, pop1, pop2]) 

    
    # Runge Kutta Step (start time, pop, model) 
    def RungeKuttaStep(self, t, y, model): #y == pop
        
        # Overall of people:
        n = round(sum(y[0]) +sum(y[1]) +sum(y[2]), 0) # Number of people has to stay the same all time

        # ALgorithm stops if overall population changes  
        if round(n,0) != self.N:
             print('Population number changed')
             exit()
        
        #Number people different cities
        self.N_i = np.array([round(sum(y[0]),0), round(sum(y[1]),0) , round(sum(y[2]), 0)])

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
    # y0 is population_0
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
                self.beta = self.beta_array[:,i-1]
                self.gamma = self.gamma_array[:,i-1]
                self.y_RK_SIS[i] = self.RungeKuttaStep(self.x[i-1], self.y_RK_SIS[i-1], model)
                
        elif model == "SIR_model":
            
            y0 = self.population_0 
            self.y_RK_SIR[0] = y0
            print(self.n)
            for i in range(1, (self.n) +1):
                self.beta = self.beta_array[:,i-1]
                self.gamma = self.gamma_array[:,i-1]
                self.y_RK_SIR[i] = self.RungeKuttaStep(self.x[i-1], self.y_RK_SIR[i-1], model)
        
        elif model == "SIT_model":
            
            y0 = self.population_0 
            self.y_RK_SIT[0] = y0
            
            for i in range(1, (self.n)+1):
                self.beta = self.beta_array[:,i-1]
                self.gamma = self.gamma_array[:,i-1]
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
                # Time Dependence Beta and gamma
                self.beta = self.beta_array[:,i-1]
                self.gamma = self.gamma_array[:,i-1]
                #Number people different cities
                self.N_i = np.array([round(sum(self.y_pEu_SIS[i-1,0]),0), round(sum(self.y_pEu_SIS[i-1,1]),0) , round(sum(self.y_pEu_SIS[i-1,2]), 0)])
                self.y_pEu_SIS[i] = self.y_pEu_SIS[i-1] + self.h * self.SIS_model(self.x[i-1], self.y_pEu_SIS[i-1])
        
        elif model == "SIR_model":
            y0 = self.population_0 
            self.y_pEu_SIR[0] = y0
            
            for i in range(1, (self.n)+1):
                # Time Dependence Beta and gamma
                self.beta = self.beta_array[:,i-1]
                self.gamma = self.gamma_array[:,i-1]
                #Number people different cities
                self.N_i = np.array([round(sum(self.y_pEu_SIS[i-1,0]),0), round(sum(self.y_pEu_SIS[i-1,1]),0) , round(sum(self.y_pEu_SIS[i-1,2]), 0)])
                self.y_pEu_SIR[i] = self.y_pEu_SIR[i-1] + self.h * self.SIR_model(self.x[i-1], self.y_pEu_SIR[i-1])
        
        elif model == "SIT_model":
            
            y0 = self.population_0 
            self.y_pEu_SIT[0] = y0
            
            for i in range(1, (self.n)+1):
                #Time Dependence beta and gamma
                self.beta = self.beta_array[:,i-1]
                self.gamma = self.gamma_array[:,i-1]
                #Number people different cities
                self.N_i = np.array([round(sum(self.y_pEu_SIS[i-1,0]),0), round(sum(self.y_pEu_SIS[i-1,1]),0) , round(sum(self.y_pEu_SIS[i-1,2]), 0)])
                self.y_pEu_SIT[i] = self.y_pEu_SIT[i-1] + self.h * self.SIT_model(self.x[i-1], self.y_pEu_SIT[i-1])
    
        
    def Theo_SIS_model(self):
                        
        self.theo_SIS[0]  = self.population_0[0]

        I_inf = sum(self.population_0[0])*(1-self.alpha/self.beta[0])
        r = I_inf/self.population_0[0,1] -1
        delta = self.beta[0] - self.alpha

        for i in range(1,len(self.theo_SIS)): 
            I_i = I_inf/(1 + r*np.exp(-delta*self.x[i]))
            self.theo_SIS[i] = np.array([sum(self.population_0[0]) - I_i, I_i , 0])

 
#includes several results in the same figure  
# xs is a list of x-type lists
# ys is a list of y-type lists   
#labels is the list of labels, one label per each line in the plot  
# xlabel and ylabel refer to the whole plot     
    def plot(self, xs, ys, labels, colors, linestyles, title, xlabel, ylabel):

        plt.style.use('rc.mplstyle')

        fig, ax = plt.subplots(figsize = (3,4))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    
        for line in range(len(xs)):
            ax.plot(xs[line], ys[line], label = labels[line], 
                    color = colors[line], linestyle = linestyles[line])
        ax.set_xlim(xs[0][0], xs[0][-1])

        
        #Mexico Quarantäne
        ax.text(0.22,0.78, f' $\\beta_1  $ = {self.R0[0]*self.a:.3f}\n $\\beta_2  $ = {0.4*self.a:.3f}\n $\\beta_3  $ = {0.9*self.a:.3f}\n$\\gamma  $ = {self.g[0]*self.a:.3f}',fontsize=9, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='lightgrey', alpha=0.2, edgecolor='lightgrey'))
        ax.vlines(x = 37, ymin = 0, ymax = 1, transform=ax.get_xaxis_transform(), colors= 'slategrey')
        ax.vlines(x = 40, ymin = 0, ymax = 1, transform=ax.get_xaxis_transform(), colors= 'darkgrey')
        ax.vlines(x = 51, ymin = 0, ymax = 1, transform=ax.get_xaxis_transform(), colors= 'darkgrey')
        ax.vlines(x = 56, ymin = 0, ymax = 1, transform=ax.get_xaxis_transform(), colors= 'slategrey')
        ax.text(0.22,0.45, f'$\\beta_1  $ ',fontsize=10, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')
        ax.text(0.6,0.45, f'$\\beta_2  $ ',fontsize=10, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')
        ax.text(0.85,0.45, f'$\\beta_3  $ ',fontsize=10, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')
        

        # const beta, gamma
        #ax.text(0.22,0.55, f'$\\beta$ = {self.R0[0]*self.a:.3f}\n $\\gamma  $ = {self.g[0]*self.a:.3f}\n',fontsize=12, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')
        
        # Gauss
        #ax.text(0.22,0.3, f'$\\beta$ = {self.R0[0]*self.a:.3f}\n \t\t\t$\\gamma = 0.003 \cdot 180 \cdot N(90, 30)$ \n',fontsize=12, transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')

        ax.legend()      
        
        fig.tight_layout()    
        plt.savefig(save_title + '.pdf') 
        plt.show()
      

if __name__ == '__main__':
    
   
    #Styles for plots with different methods and population types (same model asumed)
   
    #S(t) -> yellow
    #I(t) -> red
    #R(t) -> green 
   
    #RK -> solid line
    #Euler -> dashed line
    #theo -> dotted line
    
    #co for colour
    S_co = "purple"
    I_co = "red"
    R_co = "green"

    S_theo = "grey"
    I_theo = "grey"

    
    
    #ls for linestyle
    RK_ls = "-"
    pEu_ls = "--"
    theo_ls =  '-.'
    
    #la for labels, order is S, I, R
    RK_la = ("$S(t)$", "$I(t)$", "$R(t)$")
    pEu_la = ("Eu $S(t)$", "Eu $I(t)$", "Eu $R(t)$")
    theo_la = ("exact $S(t)$", "exact $I(t)$", "exact $R(t)$")
    
    
    #test 1:
    test1 = Disease()
    
    ###########################################################################

    if False: #SIS model
        #plot S(t) and I(t) on SIS: RK, Euler and theo vs days
        model = "SIS_model"
        
        test1.RungeKuttaLoop(model)
        test1.Euler(model)
        test1.Theo_SIS_model()
        
        title = "S(t) and I(t) on SIS: RK, Euler and theo vs days"
        save_title = "SIS"
        
        y_axis = [test1.y_RK_SIS[:,0,0], test1.y_RK_SIS[:,0,1], 
                test1.y_pEu_SIS[:,0,0], test1.y_pEu_SIS[:,0,1],
                test1.theo_SIS[:,0,0], test1.theo_SIS[:,0,1]]
        
        labels = [RK_la[0], RK_la[1], pEu_la[0], pEu_la[1], theo_la[0], theo_la[1]]
        colors = [S_co, I_co, S_co, I_co, S_theo, I_theo]
        linestyles = [RK_ls, RK_ls, pEu_ls, pEu_ls, theo_ls, theo_ls]
        
        x_axis = [test1.x for i in range(len(y_axis))]
        
        plot(x_axis, y_axis, labels, colors, linestyles, title, "Days", "Population")
    
    ###########################################################################

    #plot S(t), I(t) and R(t) on SIR: RK vs days
    if False: #SIS model - one plot (Mexico City)
        model = "SIR_model"
        
        test1.RungeKuttaLoop(model)
        
        title = ""
        print(test1.beta)
        save_title = "SIR-Mexico-quaran" + str(test1.beta[0]) +"_" +   str(test1.gamma[0])  + "_" + str(test1.alpha) + "_" + str(test1.h) +"_" +  str(test1.T) 
        
        y_axis = [test1.y_RK_SIR[:,0,2] , test1.y_RK_SIR[:,0,1]]#, test1.y_RK_SIR[:,0,2]]
        
        labels = [RK_la[2], RK_la[1], RK_la[2]]
        colors = [S_co, I_co, R_co]
        linestyles = [RK_ls, RK_ls, RK_ls]
        
        x_axis = [test1.x for i in range(len(y_axis))]
        
        test1.plot(x_axis, y_axis, labels, colors, linestyles, title, "Days", "Population")

    ###############################################################################

    if False: #SIR with travel (one plot)

        #plot S(t), I(t) and R(t) on SIR: RK vs days for three cities
        model = "SIT_model"
        
        test1.RungeKuttaLoop(model)
        
        title = "S(t), I(t) and R(t) on SIT: RK vs days"
        save_title = "SIT"
        
        y_axis = [test1.y_RK_SIT[:,0,0], test1.y_RK_SIT[:,0,1], test1.y_RK_SIT[:,0,2],
                test1.y_RK_SIT[:,1,0], test1.y_RK_SIT[:,1,1], test1.y_RK_SIT[:,1,2],
                test1.y_RK_SIT[:,2,0], test1.y_RK_SIT[:,2,1], test1.y_RK_SIT[:,2,2]]
        
        labels = [RK_la[0], RK_la[1], RK_la[2],RK_la[0], RK_la[1], RK_la[2],RK_la[0], RK_la[1], RK_la[2]]
        colors = [S_co, I_co, R_co,S_co, I_co, R_co,S_co, I_co, R_co]
        linestyles = [RK_ls, RK_ls, RK_ls,RK_ls, RK_ls, RK_ls,RK_ls, RK_ls, RK_ls]
        
        x_axis = [test1.x for i in range(len(y_axis))]
        
        plot(x_axis, y_axis, labels, colors, linestyles, title, "Days", "Population")

##############################################


#plot S(t), I(t) and R(t) on SIR: RK vs days for three cities - Subplots

if True:
    from matplotlib import rc
    rc('text', usetex=True)

    # Cities
    txt = [' (a)  Mexico City', '(b)  Vancouver', '(c)  New York City'] 

    model = "SIT_model"
    test1.RungeKuttaLoop(model)

    plt.style.use('rc.mplstyle')
    fig, ax = plt.subplots( ncols=3,figsize=(5, 3))

    ax[0].set_ylabel("Population")
    i = 0
    for col in ax:
     
        col.set_xlabel('Days')
        col.text(0.5, -0.3, txt[i], transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        col.grid(True)

        #Plots
        col.plot(test1.x, test1.y_RK_SIT[:,i,0], label = f'$S(t)$', color = S_co) 
        col.plot(test1.x, test1.y_RK_SIT[:,i,1], label = f'$I(t)$', color = I_co)
        col.plot(test1.x, test1.y_RK_SIT[:,i,2], label = f'$R(t)$', color = R_co)
        col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        col.set_xticks([0,100,200,300])
    
        col.set_xlim(test1.x[0], test1.x[-1])

        #Plot settings for different beta and gamma

        # const beta, gamma
        #col.text(0.33,0.9, f'$\\beta$ = {test1.R0[i]*test1.a:.3f}\n $\\gamma  $ = {test1.g[i]*test1.a:.3f}',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='lightgrey', alpha=0.6, edgecolor='lightgrey'))
        
        # const beta, Gauss - gamma
        col.text(0.7,0.25, f'$\\beta$ = {test1.R0[i]*test1.a:.3f} \n $ \\gamma = $ Gauss ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='lightgrey', alpha=0.2, edgecolor='lightgrey'))
        
        '''
        # lin beta, zero gamma
        col.text(0.35,0.65, f'$\\beta_1$ = {test1.R0[i]*test1.a:.3f} \n $\\beta_2$ = {0.3*test1.a:.3f}\n $ \\gamma = 0.000$',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='lightgrey', alpha=0.6, edgecolor='lightgrey'))
        col.vlines(x = 60, ymin = 0, ymax = 1, transform=col.get_xaxis_transform(), colors= 'slategrey')
        col.vlines(x = 180, ymin = 0, ymax = 1, transform=col.get_xaxis_transform() , colors= 'darkgrey')
        col.text(0.10,0.28, f'$\\beta_1$ ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        col.text(0.45,0.28, f'$\\beta_l$ ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        col.text(0.80,0.28, f'$\\beta_2$ ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        '''

        '''
        # lin beta, gauss gamma
        #col.text(0.680,0.95, f'$\\beta_1$ = {test1.R0[i]*test1.a:.3f} \n $\\beta_2$ = {0.3*test1.a:.3f}\n $ \\gamma = $ Gauss ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='lightgrey', alpha=0.2, edgecolor='lightgrey'))
        col.vlines(x = 60, ymin = 0, ymax = 1, transform=col.get_xaxis_transform() , colors= 'slategrey')
        col.vlines(x = 180, ymin = 0, ymax = 1, transform=col.get_xaxis_transform() , colors= 'darkgrey')
        col.text(0.10,0.23, f'$\\beta_1$ ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        col.text(0.80,0.23, f'$\\beta_2$ ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        col.text(0.360,0.23, f'$\\beta_l$ ',fontsize=9, transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        '''

        i+= 1

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.995), loc='upper center', ncols=3)
  
    fig.subplots_adjust(wspace=0.36, left=0.1, right=0.97, top=0.82, bottom=0.25)
    plt.savefig('SRT_const_beta_gauss_vacc_300.pdf') 
    plt.show()
