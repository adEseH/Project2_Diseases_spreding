import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


## for one differential equation: 

def dydx(x, y):
    return (6*x - 3*y + 11)


# RUnge Kutta Step

def RungeKuttaStep( x_old, y_old, delta_x):
    # start time, start values, overall time, step size
    k1 = dydx(  x_old, y_old)
    k2 = dydx(x_old + delta_x/2, y_old + k1*delta_x/2)
    k3 = dydx(x_old + delta_x/2, y_old + k2*delta_x/2)
    k4 = dydx(x_old + delta_x, y_old + k3*delta_x/2)

    K = 1/6*(k1 + 2.0*k2 + 2.0* k3 + k4)
    return  y_old + K*delta_x

# Runge Kutta Loop

def RungeKuttaLoop(x0, y0, delta_x, x_dist):  #x_dist such that it is divideable through step size

    N = x_dist/delta_x
    print(N)

    if N != int(N):                     #tested
        print('Take other x_dist')

    N = int(N)

    x = np.linspace(x0, x_dist, N+1)   #tested
    y = np.empty(N+1)                  #tested

    y[0] = y0

    for i in range(1,N+1):
        y[i] = RungeKuttaStep(x[i-1], y[i-1], delta_x)

    return x, y    

#Apply Runge Kutta
x0 = 0
y0 = 4
h = 0.5
x_dist = 2

x,y  = RungeKuttaLoop(x0, y0, h, x_dist)

array = np.array([x, y])


filename = "x-y-values.csv"

with open(Path('Data')/filename, 'w', newline='') as csvfile: 
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the first line
    csvwriter.writerows(array)

with open(Path('Data')/filename, mode = 'r') as file:
    csvFile = csv.reader(file)
    data = list(csvFile)
    data_array = np.array(data, dtype=float)

x_file = np.array(data_array[0, :])
y_file = np.array(data_array[1, :])


print(x)
print(y)
print(x_file)
print(y_file)


# Compare with theory
N = int(x_dist/h)
x_e = np.linspace(x0, x_dist, N+1)   #tested
y_e = np.empty(N+1)                  #tested
y_e[0]  = y0

for i in range(1,len(y_e)): 
    y_e[i] = np.exp(-3*x[i]) + 2*x[i] + 3

'''
### Plot result: 
plt.style.use('rc.mplstyle')
fig, ax = plt.subplots(figsize = (3,4))
ax.set_title('Runge Kutta expample')
ax.plot(x_e,y_e, label = 'Theory')
ax.plot(x,y, label = 'Runge')
ax.plot(x_file,y_file, label = 'From File')    
ax.set_xlabel('time x')
ax.set_ylabel('y')
ax.legend()
plt.show()
'''

n = np.empty((2,3,3))

print(n)
print(n[0])

n[0] = [[1,1,1],[2,2,2],[3,3,3]]

print(n[:,0,:])

