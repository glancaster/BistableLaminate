# - - Bistable Laminate Optimization Code - - 
#   - Graham Lancaster -
#   - grlanca@clemson.edu - 

# - - Importing Libaries - - 
import numpy as np 
from sympy import *
from scipy.optimize import*
import random
import time

# - - Properties - - 

# - - Constant - - 
E1 = 135e9
E2 = 9.5e9
v12 = 0.3
v21 = (E2*v12)/E1
G12 = 5e9
alpha = np.matrix('-2e-6; -3.27e-5; 0')
deltaTemperature = 140 #Knippenberg
plyThickness = 0.00012

# - - Editable - -    #genetic algorithm parameters
thetaLayup0 = 0
thetaLayup1 = 90
lengthX = 0.6096
lengthY = 0.6096

# - - Symbolic Variables - - 
w20,w02,w11,ex00,ex20,ex02,ex11,ey00,ey20,ey02,ey11,u01,u03,v01,v03,x,y= symbols('w20 w02 w11 ex00 ex20 ex02 ex11 ey00 ey20 ey02 ey11 u01 u03 v01 v03 x y', real=True)


# - - Functions - - 
def Q(E1,E2,v12,v21,G12):
  Q11 = E1/(1-v12*v21)
  Q22 = E2/(1-v12*v21)
  Q12 = (v12*E2)/(1-v12*v21)
  Q66 = G12
  Q = np.matrix([[Q11,Q12,0],
                 [Q12,Q22,0],
                 [0,0,Q66]])
  return Q
Q = Q(E1,E2,v12,v21,G12)

# - - Genetic Algorthim - - 
def ga(v):
    # - - Input Genetic Algorithm Parameters - - 
    lengthX = v[0]
    lengthY = v[1]
    thetaLayup = [0,90]

    # - - Intermediate Equations - - 

    #   - In-plane Strains -
    ex0 = ex00 + ex20*x**2 + ex11*x*y + ex02*y**2
    ey0 = ey00 + ey20*x**2 + ey11*x*y + ey02*y**2
    #   - Mid-plane Displacements - 
    w0 = 0.5*(w20*x**2 + w02*y**2 + w11*x*y)
    u0 = integrate(ex0 - 0.5*(diff(w0,x))**2,(x)) + u01*y + u03*y**3
    v0 = integrate(ey0 - 0.5*(diff(w0,y))**2,(y)) + v01*y + v03*y**3
    #   - Mid-plane Strain Vector -
    e0x = diff(u0,x) + 0.5*(diff(w0,x))**2
    e0y = diff(v0,y) + 0.5*(diff(w0,y))**2
    e0xy = 0.5*(diff(u0,y) + diff(v0,x) + diff(w0,x)*diff(w0,y))
    #   - Curvatures - 
    kx = - diff(w0,x,x)
    ky = - diff(w0,y,y)
    kxy = - diff(w0,x,y)
    #   - Strain-Curvature Matrix
    e0k = np.matrix([[e0x],
                     [e0y],
                     [e0xy],
                     [kx],
                     [ky],
                     [kxy]])

    #   - Z Array -
    if len(thetaLayup)%2 == 0:
        halfNPLy = len(thetaLayup)/2
        z = np.arange(-halfNPLy,halfNPLy+1, dtype = 'int' )*plyThickness
    else:
        NPly = len(thetaLayup)
        z = np.arange(-NPly,NPly+1,2)*(plyThickness/2)
        
    #   - Q Bar and Alpha Matrices - 
    QbarArray = []
    AlphaArray = []
    for t in range(len(thetaLayup)):
        m = np.cos(thetaLayup[t]*np.pi/180)
        n = np.sin(thetaLayup[t]*np.pi/180)
        T = np.matrix([[m*m, n*n ,2*m*n] ,
                       [ n*n ,m*m, -2*m*n ], 
                       [-m*n, m*n, m*m-n*n]])
        QbarArray.append(T.T*Q*T)
        AlphaArray.append(T*alpha)
    
    #   - ABBD Matrix and Thermal Force/Moments Matrix -
    A,B,D,N,M = [0,0,0,0,0]
    for k in range(len(thetaLayup)):
        A = A + QbarArray[k]*(z[k+1]-z[k])
        B = B + (1/2)*QbarArray[k]*(z[k+1]**2-z[k]**2)
        D = D + (1/3)*QbarArray[k]*(z[k+1]**3-z[k]**3)
        N = N + QbarArray[k]*AlphaArray[k]*(z[k+1]-z[k])*deltaTemperature
        M = M + (1/2)*QbarArray[k]*AlphaArray[k]*(z[k+1]**2-z[k]**2)*deltaTemperature
    
    ABBD = np.concatenate((np.concatenate((A,B),axis = 1),np.concatenate((B,D),axis = 1)))
    NM = np.concatenate((N,M))

    for i in range(6):
        for j in range(6):
            if ABBD[i,j] < 1e-6:
                ABBD[i,j] = 0
    
    # - - Potential Energy of the Laminate - - 
    Wlam = e0k.T*ABBD*e0k-e0k.T*NM
    Wlam = (1/2)*integrate(Wlam[0,0],(x,-lengthX/2,lengthX/2),(y,-lengthY/2,lengthY/2))

    # - - Setting up NonLinear Solver - - 
    def potentialEnergy(x):
        w20f,w02f,w11f,ex00f,ex20f,ex02f,ex11f,ey00f,ey20f,ey02f,ey11f,u01f,u03f,v01f,v03f = x
        return [diff(Wlam,w20).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,w02).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,w11).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ex00).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ex20).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ex02).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ex11).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ey00).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ey20).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ey02).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,ey11).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,u01).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,u03).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,v01).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)]),
                diff(Wlam,v03).subs([(w20,w20f),(w02,w02f),(w11,w11f),(ex00,ex00f),(ex20,ex20f),(ex02,ex02f),(ex11,ex11f),(ey00,ey00f),(ey20,ey20f),(ey02,ey02f),(ey11,ey11f),(u01,u01f),(u03,u03f),(v01,v01f),(v03,v03f)])]              
    x0 = [1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    start = time.time()
    solution = root(potentialEnergy, x0, method= 'lm') # documentation on this method https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root
    end = time.time()
    timeelasped = (end-start)
    res = solution.x
    res[abs(res) < 1e-6] = 0
    a = res[0]
    b = res[1]

    lx = lengthX/2
    ly = lengthY/2
    state1 = -(1/2)*(a*lx**2 + b*ly**2)
    state2 = -(1/2)*(-b*lx**2 + -a*ly**2)
    totalDeformation = abs(state1)+abs(state2)
    fitnessFunction = totalDeformation

    print("Fitness: ", fitnessFunction, "Solver Time: ", timeelasped)
    return fitnessFunction

#'''
# - - Running - - 
solutions = []
pop = 10
maxiter = 100
maxLength = 0.6096
minLength = 0.3048 
for s in range(pop):
    solutions.append( ((random.uniform(minLength,maxLength)),(random.uniform(minLength,maxLength)))) #check if its doing an integer for theta

for i in range(maxiter):
    
    rankedsolutions = []
    for s in solutions:
        rankedsolutions.append( (ga([s[0],s[1]]),s))
    rankedsolutions.sort()
    rankedsolutions.reverse()
    print(f" --- Gen {i} best solutions --- ")
    print(rankedsolutions[0])

    bestsolutions = rankedsolutions[:5]

    LX = []
    LY = []

    for s in bestsolutions:
        LX.append(s[1][0])
        LY.append(s[1][1])

    newGen = []
    for _ in range(pop):
        e1 = random.choice(LX)*random.uniform(0.9,1.1)
        e2 = random.choice(LY)*random.uniform(0.9,1.1)
        if e1 > maxLength:
            e1 = maxLength
        if e1 < minLength:
            e1 = minLength
        if e2 > maxLength:
            e2 = maxLength
        if e2 < minLength:
            e2 = minLength
        newGen.append((e1,e2))
        
    solutions = newGen
#'''
#ga = ga([lengthX,lengthY,thetaLayup0,thetaLayup1])



# A solver to try - https://github.com/uqfoundation/mystic
