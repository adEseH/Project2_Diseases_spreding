# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:55:13 2023

@author: Anna, Carmen
"""
import numpy as np
import matplotlib.pyplot as plt

#### City 1: Mexico City, City 2: Vancouver, City 3: New York

class Disease:
    def __init__(self,T=250, a=1/6, b= [0.287,0.267,0.283], g = [0.0021,0.0029, 0.0018],
                 W=np.array([[0,0,0],
                            [0,0,0],
                            [0,0,0]]),
                         pop_0=np.array([[19.985E6,4,0],[2.238E6,0,0], [18.309E6,0,0]]), h=1 ):
        
        # initialise gamma_arrays: (standart: constant values)
        if False:
            self.alpha = a
            self.beta_array= np.array([np.repeat(b[0], T),np.repeat(b[1], T),np.repeat(b[2], T)])
            self.gamma_array = g= np.array([np.repeat(g[0], T),np.repeat(g[1], T),np.repeat(g[2], T)])

        else: # No vaccination
            self.alpha = a
            self.beta_array= np.array([np.repeat(b[0], T),np.repeat(b[1], T),np.repeat(b[2], T)])
            self.gamma_array = g= np.array([np.repeat(0, T),np.repeat(0, T),np.repeat(0, T)])

        print('T = ', T, 'beta = ' ,self.beta_array[:,0], 'gamma  = ', self.gamma_array[:,0], 'alpha = ', self.alpha)

        #initialise first values:
        self.beta = self.beta_array[:,0]
        self.gamma = self.gamma_array[:,0]
        
        # Time and population 
        self.t0 = 0
        self.T = T # total time [default: 200 days]
        self.population_0 = pop_0 # inital population [S0, I0, R0] (y0)
        self.N = round(sum(pop_0[0]) +sum(pop_0[1]) +sum(pop_0[2]), 0) # Overall population

        # Transport matrix: [default as in Jakobs example] - TODO
        if False:
            self.W = np.array([[0.0, 5.27324458E-05 ,2.38237046E-04],
                            [9.28720687E-05 ,0.0 ,2.97796307E-05],
                            [8.17828665E-04, 5.80450430E-05 ,0.0]])

        else: # Own traveling
            #calculating probabilities
            W_NM = 0.24*640E3*(pop_0[2,0]/308E6)/pop_0[2,0]
            W_NV = 0.11*0.8*2999.330/pop_0[2,0]
            W_MV = 0.7*0.8*63.5E3/pop_0[0,0]

            W_MN = W_NM*pop_0[2,0]/pop_0[0,0]
            W_VN = W_NV*pop_0[2,0]/pop_0[1,0]
            W_VM = W_MV*pop_0[0,0]/pop_0[1,0]

            self.W = np.array([[0, W_VM, W_NM],[W_MV ,0, W_NV],[W_MN , W_VN,0]])
        
        print("W = ", self.W)

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
        pop0 = np.array([-self.beta[0]*pop[0,0]*pop[0,1]/self.N_i[0] - self.gamma[0]*pop[0,0], self.beta[0]*pop[0,0]*pop[0,1]/self.N_i[0]  - self.alpha*pop[0,1], self.gamma[0]*pop[0,0] + self.alpha*pop[0,1]])
        pop1 = np.array([0,0,0])
        pop2 = np.array([0,0,0])
        return np.array([pop0, pop1, pop2]) 

## SIR model with travel: #Hier Matrix
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

        # ALgorithm stops if overall population changes (TODO: maybe has to be changed)
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
                self.beta = self.beta_array[:,i-1]
                self.gamma = self.gamma_array[:,i-1]
                self.y_RK_SIS[i] = self.RungeKuttaStep(self.x[i-1], self.y_RK_SIS[i-1], model)
                
        elif model == "SIR_model":
            
            y0 = self.population_0 
            self.y_RK_SIR[0] = y0
            
            for i in range(1, (self.n)+1):
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
def plot(xs, ys, labels, colors, linestyles, title, xlabel, ylabel):
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
    RK_la = ("RK4 $S(t)$", "RK4 $I(t)$", "RK4 $R(t)$")
    pEu_la = ("Eu $S(t)$", "Eu $I(t)$", "Eu $R(t)$")
    theo_la = ("exact $S(t)$", "exact $I(t)$", "exact $R(t)$")
    
    
    #test 1:
    test1 = Disease()
    
    ###########################################################################

    if True:
        #1st Jacob's plot
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
    #2nd Jacob's plot
    #plot S(t), I(t) and R(t) on SIR: RK vs days
    if True:
        model = "SIR_model"
        
        test1.RungeKuttaLoop(model)
        
        title = "S(t), I(t) and R(t) on SIR: RK vs days"
        save_title = "SIR"
        
        y_axis = [test1.y_RK_SIR[:,0,0], test1.y_RK_SIR[:,0,1], test1.y_RK_SIR[:,0,2]]
        
        labels = [RK_la[0], RK_la[1], RK_la[2]]
        colors = [S_co, I_co, R_co]
        linestyles = [RK_ls, RK_ls, RK_ls]
        
        x_axis = [test1.x for i in range(len(y_axis))]
        
        plot(x_axis, y_axis, labels, colors, linestyles, title, "Days", "Population")

    ###############################################################################

    if True:
        #3nd Travel
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
#3nd Travel
#plot S(t), I(t) and R(t) on SIR: RK vs days for three cities - Subplots
from matplotlib import rc
rc('text', usetex=True)

txt = [' (a)  Mexico City', '(b)  Vancouver', '(c)  New York City'] ####### Cities
if True:
    model = "SIT_model"
    test1.RungeKuttaLoop(model)

    plt.style.use('rc.mplstyle')
    fig, ax = plt.subplots( ncols=3,figsize=(5, 3))#, layout='constrained')

# (ax0, ax1, ax2)

    #fig.suptitle(f"$S(t)$, $I(t)$ and $R(t)$ on SIT: RK vs days")

    ax[0].set_ylabel("Population")
    i = 0
    for col in ax:
        #col.set_xlabel(r'\begin{center} Days \\\vspace{0.3cm}' + txt[i] + r'\end{center}')
        col.set_xlabel(r'Days')
        col.text(0.5, -0.3, txt[i], transform=col.transAxes, verticalalignment='top', horizontalalignment='center')
        col.grid(True)
        #Plots
        col.plot(test1.x, test1.y_RK_SIT[:,i,0], label = f'$S(t)$', color = S_co) 
        col.plot(test1.x, test1.y_RK_SIT[:,i,1], label = f'$I(t)$', color = I_co)
        col.plot(test1.x, test1.y_RK_SIT[:,i,2], label = f'$R(t)$', color = R_co)
        #col.set_xticks(np.arange(test1.x[0], test1.x[-1], len(test1.x)/4))
        col.set_xlim(test1.x[0], test1.x[-1])
        i+= 1

    #ax[1].legend(bbox_to_anchor=(0.5,1.06), loc='lower center', ncols=3)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.995), loc='upper center', ncols=3)

    #fig.set_tight_layout(True)   
    fig.subplots_adjust(wspace=0.28, left=0.1, right=0.97, top=0.82, bottom=0.25)
    plt.savefig('SRT_3Plots.pdf') 
    plt.show()
