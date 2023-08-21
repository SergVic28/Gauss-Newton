import pylab
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sympy import diff, symbols


print "Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2"
# Init parameters
maxIteration = 100 # iter
eps = 1e-6 # accuracy

x = -1 # startpoint
y = -1

"""
Gauss-Newton parameters
"""
GN = 0 # iter 0
GNX = [] # massives
GNY = []
GNZ = []
GaussNewtonX = x
GaussNewtonY = y


N = 0
NX = []
NY = []
NZ = []
NewtonX = x
NewtonY = y

"""
Gradient, Hesse
"""
gradient_GN = np.zeros((2, 1))
hesse_GN = np.zeros((2, 2))
jacobi = np.zeros((2, 2))

# Adding initial values to an array
GNX.append(GaussNewtonX)
GNY.append(GaussNewtonY)
NX.append(NewtonX)
NY.append(NewtonY)

x0 = -1
y0 = -1
# Interval
xgrid = np.arange(x0, -x0, 0.1)
ygrid = np.arange(y0, -y0, 0.1)
X, Y = np.meshgrid(xgrid, ygrid)

"""
Rosenbrock function: a = 0.5, b = 0.5
"""
def function(X, Y):
    Z = (0.5 - X)**2 + 0.5*(Y - X**2)**2 # Rosenbrock [-1;1] - xy
    #Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2 # Himmelblow [-5;5] - xy
    #Z = 0.26*(X**2 + Y**2) - 0.48*X*Y # Mathias [-10;10] - xy , always 1 iter
    return Z

def f1(X, Y):
    Z = 1 - X
    return Z

def f2(X, Y):
    Z = Y - X**2
    return Z

# The partial derivative of order 1 (X)
def derivateX1(x, y):
    X, Y = symbols('X Y')
    Z = function(X, Y)
    dZdX = diff(Z, X, 1)
    ansX1 = dZdX.subs({X:x, Y:y})
    return ansX1

# The partial derivative of order 2 (X)
def derivateX2(x, y):
    X, Y = symbols('X Y')
    Z = function(X, Y)
    dZdX = diff(Z, X, 2)
    ansX2 = dZdX.subs({X:x, Y:y})
    return ansX2

# The partial derivative of order 1 (Y)
def derivateY1(x, y):
    X, Y = symbols('X Y')
    Z = function(X, Y)
    dZdY = diff(Z, Y, 1)
    ansY1 = dZdY.subs({X:x, Y:y})
    return ansY1

# The partial derivative of order 2 (Y)
def derivateY2(x, y):
    X, Y = symbols('X Y')
    Z = function(X, Y)
    dZdY = diff(Z, Y, 2)
    ansY2 = dZdY.subs({X:x, Y:y})
    return ansY2

# The second mixed partial derivative
def derivateXY(x, y):
    X, Y = symbols('X Y')
    Z = function(X, Y)
    dZdY = diff(Z, Y, 1)
    dZdYdX = diff(dZdY, X, 1)
    ansXY = dZdYdX.subs({X:x, Y:y})
    return ansXY

def derf1x(x, y):
    X, Y = symbols('X Y')
    Z = f1(X, Y)
    dzdx = diff(Z, X, 1)
    resf1x = dzdx.subs({X:x, Y:y})
    return resf1x

def derf1y(x, y):
    X, Y = symbols('X Y')
    Z = f1(X, Y)
    dzdy = diff(Z, Y, 1)
    resf1y = dzdy.subs({X:x, Y:y})
    return resf1y

def derf2x(x, y):
    X, Y = symbols('X Y')
    Z = f2(X, Y)
    dzdx = diff(Z, X, 1)
    resf2x = dzdx.subs({X:x, Y:y})
    return resf2x

def derf2y(x, y):
    X, Y = symbols('X Y')
    Z = f2(X, Y)
    dzdy = diff(Z, Y, 1)
    resf2y = dzdy.subs({X:x, Y:y})
    return resf2y

# Calculating a function for constructing a surface
Z = function(X, Y)
# Function at the initial points
GaussNewtonZ = function(GaussNewtonX, GaussNewtonY)
GNZ.append(GaussNewtonZ)
NewtonZ = function(NewtonX, NewtonY)
NZ.append(NewtonZ)

while N < maxIteration:
    dZdX = derivateX1(NewtonX, NewtonY)
    dZdY = derivateY1(NewtonX, NewtonY)
    gradient_GN[0] = dZdX
    gradient_GN[1] = dZdY
    dzdx1 = derf1x(NewtonX, NewtonY)
    dzdy1 = derf1y(NewtonX, NewtonY)
    dzdx2 = derf2x(NewtonX, NewtonY)
    dzdy2 = derf2y(NewtonX, NewtonY)
    jacobi[0, 0] = dzdx1
    jacobi[0, 1] = dzdy1
    jacobi[1, 0] = dzdx2
    jacobi[1, 1] = dzdy2
    jacobisLeft = np.dot(jacobi.T, jacobi)
    jacobiLeftInverse = np.linalg.inv(jacobisLeft)
    #print(jacobiLeftInverse)
    jjj = np.dot(jacobiLeftInverse, jacobi.T)
    #print(jjj)
    Q = np.dot(-jacobiLeftInverse, gradient_GN)
    NewtonX = NewtonX + Q[0]
    NewtonY = NewtonY + Q[1]
    NewtonZ = function(NewtonX, NewtonY)
    NX.append(NewtonX)
    NY.append(NewtonY)
    NZ.append(NewtonZ)
    if math.sqrt(dZdX**2 + dZdY**2) < eps: # accuracy
        break
    N += 1
"""
Gauss-Newton method
"""
while GN < maxIteration:
    dZdX = derivateX1(GaussNewtonX, GaussNewtonY) # Calculating the gradient (X, Y coordinates)
    dZdY = derivateY1(GaussNewtonX, GaussNewtonY)
    gradient_GN[0] = dZdX
    gradient_GN[1] = dZdY
    hesse_GN[0, 0] = derivateX2(GaussNewtonX, GaussNewtonY) # Formation of the Hesse matrix
    hesse_GN[1, 1] = derivateY2(GaussNewtonX, GaussNewtonY) # from second partial derivatives
    hesse_GN[0, 1] = derivateXY(GaussNewtonX, GaussNewtonY) # and mixed derivatives
    hesse_GN[1, 0] = derivateXY(GaussNewtonX, GaussNewtonY)
    InverseHesse_GN = np.linalg.inv(hesse_GN) # Inverse matrix
    minus_InverseHesse_GN = -InverseHesse_GN
    S = np.dot(minus_InverseHesse_GN, gradient_GN) # Matrix 2x1
    GaussNewtonX = GaussNewtonX + S[0] # Calculation of new coordinate values
    GaussNewtonY = GaussNewtonY + S[1]
    GaussNewtonZ = function(GaussNewtonX, GaussNewtonY) # Calculating a new function value
    GNX.append(GaussNewtonX)
    GNY.append(GaussNewtonY)
    GNZ.append(GaussNewtonZ)
    if math.sqrt(dZdX**2 + dZdY**2) < eps: # accuracy
        break
    GN += 1

print ("Iteration - ", N, "X = ", NewtonX, "Y = ", NewtonY, "Z = ", NewtonZ)
print ("Iteration - ", GN, "X = ", GaussNewtonX, "Y = ", GaussNewtonY, "Z = ", GaussNewtonZ)

"""
Plotting
"""
fig = pylab.figure()
axes = Axes3D(fig)
axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = cm.jet)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_zlabel('Z Label')
view = (40, 70)
axes.view_init(elev=view[0], azim=view[1])
plt.plot(NX, NY, NZ, 'y')
plt.plot(GNX, GNY, GNZ, 'r', label = "Gauss-Newton")
pylab.legend ()
pylab.show()
