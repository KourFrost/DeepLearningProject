from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


print("With the same starting parameters (precision) the code matched the hand equations with the difference being rounding errors")



xVal = 0 
rate = 0.1 
precision = 0.05
step = 1 
df = lambda x: 2*(x-0.7)*np.exp(-(x-0.7)**2) 

while step > precision:
    xTemp = xVal 
    xVal = xTemp - rate * df(xTemp)
    step = abs(xVal - xTemp) 
    
print("q1 min:", xVal)




xVal = 0 
zVal = 0
rate = 0.1 
precision = 0.08 
step = 1 
dfx = lambda x, z : 4*(x*(0.25 - z)+(x**3)-0.25) 
dfz = lambda x, z : 2*(z-x**2)

while step > precision:
    xTemp = xVal 
    zTemp = zVal 
    xVal = xTemp - rate * dfx(xTemp, zTemp) 
    zVal = zTemp - rate * dfz(xTemp, zTemp)
    step = abs(xVal - xTemp)+abs(zVal -zTemp) 
print("q2 min:", xVal, ", ", zVal)

print("we can get better results matching the Verify Code if we increase the precision")

xVal = 0 
rate = 0.1 
precision = 0.0000001
step = 1 
df = lambda x: 2*(x-0.7)*np.exp(-(x-0.7)**2) 

while step > precision:
    xTemp = xVal 
    xVal = xTemp - rate * df(xTemp)
    step = abs(xVal - xTemp) 
    
print("q1 min:", xVal)




xVal = 0 
zVal = 0
rate = 0.1 
precision = 0.00001 
step = 1 
dfx = lambda x, z : 4*(x*(0.25 - z)+(x**3)-0.25) 
dfz = lambda x, z : 2*(z-x**2)

while step > precision:
    xTemp = xVal 
    zTemp = zVal 
    xVal = xTemp - rate * dfx(xTemp, zTemp) 
    zVal = zTemp - rate * dfz(xTemp, zTemp)
    step = abs(xVal - xTemp)+abs(zVal -zTemp) 
print("q2 min:", xVal, ", ", zVal)




print("/n###  provided Verify code ####")
#3 verify
def f1(x): return -np.exp(-(x - 0.7)**2)
plt.figure(figsize=(10,10))
x = np.linspace(-5, 5, 100)
plt.plot(x,f1(x))
result = optimize.minimize_scalar(f1)
print(result.success) #check if solver was successful
x_min = result.x # print result
print(x_min)


def f2(x): return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

x0 = np.linspace(-5, 5, 100)
x1 = np.linspace(-5, 5, 100)
x = [x0,x1]
f2(x)

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.plot3D(x[0], x[1], f2(x), 'gray')
result = optimize.minimize(f2, x, method="CG") 
# x_min = result.x # print result
# print(x_min)