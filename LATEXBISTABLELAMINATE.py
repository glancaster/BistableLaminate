# - - Bistable Laminate Optimization Code - - 
#   - Graham Lancaster -
#   - grlanca@clemson.edu - 

# - - Importing Libaries - - 
import numpy as np 
from sympy import *
from scipy.optimize import*
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# - - Lamina Properties - - 

# - - Constant - - 
E1 = 135e9 #Pa
E2 = 9.5e9 #Pa
v12 = 0.3
v21 = (E2*v12)/E1
G12 = 5e9 #Pa
alpha = np.matrix('-2e-8; -3.27e-5; 0') #C^-1 = K^-1
deltaTemperature = 121 
plyThickness = 0.0002 #m
thetaLayup = [0,90]


# - - MFC Actuator Properties - - 
# - - M2814-P1 - - 
xM2 = 28e-3 #m
yM2 = 14e-3 #m

# - - M5628-P1 - - 
xM5 = 56e-3 #m
yM5 = 28e-3 #m

# - - M8557-P1 - - 
xM8 = 85e-3 #m
yM8 = 57e-3 #m

# - - Constant - - 
E1M = 30.336e9 #Pa
E2M = 15.857e9 #Pa
v12M = 0.31
v21M = 0.16
G12M = 5.515e9 #Pa
thicknessM = 300e-6 #m
V = 1500 #V
x1 = 0.5e-3 #m
d11 = 281  #(ue/(kV/mm))
d12 = -111 #(ue/(kV/mm))
d33 = 460e-12 #m/V
BS33 = 1.48e8 #m/F                     
h13 = -2.72e9 #N/C                         
h23 = -2.72e9 #N/C                                   
Bv = [[1],[-1]]

# - - Symbolic Variables - - 
w20,w02,w11,ex00,ex20,ex02,ex11,ey00,ey20,ey02,ey11,u01,u03,v01,v03,x,y= ...
symbols('w20 w02 w11 ex00 ex20 ex02 ex11 ey00 ey20 ey02 ey11 u01 u03 v01 v03 x y', ...
real=True)


# - - Functions - - 
def Qmat(E1,E2,v12,v21,G12):
  Q11 = E1/(1-v12*v21)
  Q22 = E2/(1-v12*v21)
  Q12 = (v12*E2)/(1-v12*v21)
  Q66 = G12
  Q = np.matrix([[Q11,Q12,0],
                 [Q12,Q22,0],
                 [0,0,Q66]])
  return Q
Q = Qmat(E1,E2,v12,v21,G12)
QM = Qmat(E1,E2,v12,v21,G12)

#   - Z Array -
if len(thetaLayup)%2 == 0:
    halfNPLy = len(thetaLayup)/2
    z = np.arange(-halfNPLy,halfNPLy+1, dtype = 'int' )*plyThickness
else:
    NPly = len(thetaLayup)
    z = np.arange(-NPly,NPly+1,2)*(plyThickness/2)


# - - Intermediate Equations - - 

#   - In-plane Strains -
ex0 = ex00 + ex20*x**2 + ex11*x*y + ex02*y**2
ey0 = ey00 + ey20*x**2 + ey11*x*y + ey02*y**2
#   - Mid-plane Displacements - 
w0 = 0.5*(w20*x**2 + w02*y**2 + w11*x*y)
u0 = integrate(ex0 - 0.5*(diff(w0,x))**2,(x)) + u01*y + u03*y**3
v0 = integrate(ey0 - 0.5*(diff(w0,y))**2,(y)) + v01*x + v03*x**3
#   - Mid-plane Strain Vector -
e0x = diff(u0,x) + 0.5*(diff(w0,x))**2
e0y = diff(v0,y) + 0.5*(diff(w0,y))**2
e0xy = 0.5*(diff(u0,y) + diff(v0,x) + diff(w0,x)*diff(w0,y))
#   - Curvatures - 
kx = w20 #- diff(w0,x,x)
ky = w02 #- diff(w0,y,y)
kxy = w11 #- 2*diff(w0,x,y)
#   - Strain-Curvature Matrix
e0k = np.matrix([[ex0],
                    [ey0],
                    [e0xy],
                    [kx],
                    [ky],
                    [kxy]])


#   - Q Bar and Alpha Matrices - 
QbarArray = []
alphaArray = []
R = [[1,0,0],[0,1,0],[0,0,2]]
for t in range(len(thetaLayup)):
    m = np.cos(thetaLayup[t]*np.pi/180)
    n = np.sin(thetaLayup[t]*np.pi/180)
    T = np.matrix([[m*m, n*n ,2*m*n] ,
                    [ n*n ,m*m, -2*m*n ], 
                    [-m*n, m*n, m*m-n*n]])
    TQT = T.T*Q*R*T
    aT= T*alpha
    for i in range(3):
        for j in range(3):
            if abs(TQT[i,j]) < 2e-5:
                TQT[i,j] = 0
    QbarArray.append(TQT)
    alphaArray.append(aT)

#   - ABBD Matrix and Thermal Force/Moments Matrix -

A,B,D,N,M = [0,0,0,0,0]
for k in range(len(thetaLayup)):
    A = A + QbarArray[k]*(z[k+1]-z[k])
    B = B + (1/2)*QbarArray[k]*(z[k+1]**2-z[k]**2)
    D = D + (1/3)*QbarArray[k]*(z[k+1]**3-z[k]**3)
    N = N + QbarArray[k]*alphaArray[k]*(z[k+1]-z[k])*deltaTemperature
    M = M + (1/2)*QbarArray[k]*alphaArray[k]*(z[k+1]**2-z[k]**2)*deltaTemperature

ABBD = np.concatenate((np.concatenate((A,B),axis = 1),np.concatenate((B,D),axis = 1)))
NM = np.concatenate((N,M))


for i in range(6):
    for j in range(6):
        if abs(ABBD[i,j]) < 1e-6:
            ABBD[i,j] = 0
NM[abs(NM) < 1e-6] = 0

#   -   -   -   -   -   -   -   -   -   -   #
#   - New z array for MFC + Laminate - 
zM = np.append(z,abs(z[0]) + thicknessM)

#   - Stress and Strains for MFC actuator I -
exA = (zM - ((zM[-1]-zM[0])/2))*kx
eyA = (zM - ((zM[-1]-zM[0])/2))*ky
exyA = 0

eA = [[exA],[eyA],[exyA]]


#   - Stress and Strains for MFC actuator II -  
exE = d11*V/x1
eyE = d12*V/x1
exyE = 0

eE = [[exE],[eyE],[exyE]]
#   - Free Strain 


#   - ABBD Matrix and Thermal Force/Moments Matrix MFC -
AM,BM,DM,Nm,MM,Na,Ma= [0,0,0,0,0,0,0]
AM = AM + QM*(zM[-1]-zM[-2])
BM = BM + (1/2)*QM*(zM[-1]**2-zM[-2]**2)
DM = DM + (1/3)*QM*(zM[-1]**3-zM[-2]**3)
Nm = Nm + QM*(zM[-1]-zM[-2])*eE
MM = MM + (1/2)*QM*(zM[-1]**2-zM[-2]**2)*eE#alpha*temp
Na = Na + QM*(zM[-1]-zM[-2])*eA
Ma = Ma + (1/2)*QM*(zM[-1]**2-zM[-2]**2)*eA#alpha*temp   
ABBDM = np.concatenate((np.concatenate((AM,BM),axis = 1),np.concatenate((BM,DM),axis = 1)))
NMm = np.concatenate((Nm,MM))
NMa = np.concatenate((Na,Ma))


for i in range(6):
    for j in range(6):
        if ABBDM[i,j] < 1e-6:
            ABBDM[i,j] = 0


# - - Genetic Algorthim - - 
def ga(v):
    # - - Input Genetic Algorithm Parameters - - 
    lengthX = v[0]
    lengthY = v[1]
    

    # - - Potential Energy of the Laminate - - 
    Wlam = e0k.T*ABBD*e0k-e0k.T*NM
    Wlam = Wlam[0,0].evalf(5)
    Wlam = (1/2)*integrate(Wlam,(x,-lengthX/2,lengthX/2),(y,-lengthY/2,lengthY/2))

    # - - Potential Energy of the MFC - - 
    WM = e0k.T*ABBDM*e0k #-e0k.T*NMa
    WM = WM[0,0].evalf(5)
    WM2 = 4*(1/2)*integrate(WM,(x,0,xM2/2),(y,0,yM2/2))
    WM5 = 4*(1/2)*integrate(WM,(x,0,xM5/2),(y,0,yM5/2))
    WM8 = 4*(1/2)*integrate(WM,(x,0,xM8/2),(y,0,yM8/2))
    # - - Potential Energy of the MFC actuated - - 
    WMact = e0k.T*ABBDM*e0k-e0k.T*NMm
    WMact = WMact[0,0].evalf(5)
    WMact2 = 4*(1/2)*integrate(WMact,(x,0,xM2/2),(y,0,yM2/2))
    WMact5 = 4*(1/2)*integrate(WMact,(x,0,xM5/2),(y,0,yM5/2))
    WMact8 = 4*(1/2)*integrate(WMact,(x,0,xM8/2),(y,0,yM8/2))
    # - - Total Potential Energy of the Laminate + MFC - - 
    W2 = Wlam + WM2
    W5 = Wlam + WM5
    W8 = Wlam + WM8

    # - - Total Potential Energy of the Laminate + MFC actuated- - 
    Wact2 = Wlam + WMact2
    Wact5 = Wlam + WMact5
    Wact8 = Wlam + WMact8

  

    pr = 4 #precison

    
    # - - Setting up NonLinear Solver Laminate - - 
    def potentialEnergylam(x):
        w20f,...,v03f= x
        return [diff(Wlam,w20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,w02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,w11)..subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ex00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ex20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ex02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ex11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ey00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ey20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ey02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,ey11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,u01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,u03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,v01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wlam,v03).subs([(w20,w20f),...,(v03,0)]).evalf(pr)]              


    # - - Setting up NonLinear Solver Laminate + MFC small- - 
    def potentialEnergyLM2(x):
        w20f,...,v03f = x
        return [diff(W2,w20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,w02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,w11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ex00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ex20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ex02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ex11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ey00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ey20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ey02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,ey11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,u01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,u03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,v01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W2,v03).subs([(w20,w20f),...,(v03,0)]).evalf(pr)] 
    # - - Setting up NonLinear Solver Laminate + MFC medium- - 
    def potentialEnergyLM5(x):
        w20f,...,v03f = x
        return [diff(W5,w20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,w02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,w11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ex00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ex20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ex02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ex11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ey00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ey20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ey02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,ey11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,u01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,u03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,v01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W5,v03).subs([(w20,w20f),...,(v03,0)]).evalf(pr)] 
    # - - Setting up NonLinear Solver Laminate + MFC large- - 
    def potentialEnergyLM8(x):
        w20f,...,v03f = x
        return [diff(W8,w20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,w02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,w11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ex00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ex20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ex02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ex11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ey00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ey20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ey02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,ey11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,u01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,u03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,v01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(W8,v03).subs([(w20,w20f),...,(v03,0)]).evalf(pr)] 
    

    # - - Setting up NonLinear Solver Laminate + MFC actuated small - - 
    def potentialEnergyLMact2(x):
        w20f,...,v03f = x
        return [diff(Wact2,w20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,w02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,w11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ex00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ex20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ex02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ex11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ey00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ey20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ey02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,ey11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,u01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,u03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,v01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact2,v03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),] 
    # - - Setting up NonLinear Solver Laminate + MFC actuated medium - - 
    def potentialEnergyLMact5(x):
        w20f,...,v03f = x
        return [diff(Wact5,w20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,w02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,w11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ex00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ex20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ex02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ex11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ey00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ey20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ey02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,ey11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,u01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,u03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,v01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact5,v03).subs([(w20,w20f),...,(v03,0)]).evalf(pr)] 
    # - - Setting up NonLinear Solver Laminate + MFC actuated large - - 
    def potentialEnergyLMact8(x):
        w20f,...,v03f = x
        return [diff(Wact8,w20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,w02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,w11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ex00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ex20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ex02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ex11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ey00).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ey20).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ey02).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,ey11).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,u01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,u03).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,v01).subs([(w20,w20f),...,(v03,0)]).evalf(pr),
                diff(Wact8,v03).subs([(w20,w20f),...,(v03,0)]).evalf(pr)] 
        


        


    start = time.time()

    x0=[-0.8*10,0.001*10,0,-0.00057,3.77e-7,0.000077,0,...
    -0.0002,0.000075,-0.6e-6,0,71,0,-71,0]

    test = root(potentialEnergylam, x0, method= 'hybr',options=...
    {'xtol': 1.49012e-4},tol = 0)

    solutionLM2 = root(potentialEnergyLM2, x0, method= 'lm')
    solutionLM5 = root(potentialEnergyLM5, x0, method= 'lm')
    solutionLM8 = root(potentialEnergyLM8, x0, method= 'lm')
    solutionLMact2 = root(potentialEnergyLMact2, x0, method= 'lm') 
    solutionLMact5 = root(potentialEnergyLMact5, x0, method= 'lm') 
    solutionLMact8 = root(potentialEnergyLMact8, x0, method= 'lm') 

    end = time.time()
    timeelasped = (end-start)

    lam = test.x

    bond2 = solutionLM2.x
    bond5 = solutionLM5.x
    bond8 = solutionLM8.x
    act2 = solutionLMact2.x
    act5 = solutionLMact5.x
    act8 = solutionLMact8.x
 
    
    lam[abs(lam) < 1e-8] = 0
    bond2[abs(bond2) < 1e-8] = 0
    bond5[abs(bond5) < 1e-8] = 0
    bond8[abs(bond8) < 1e-8] = 0
    act2[abs(act2) < 1e-8] = 0
    act5[abs(act5) < 1e-8] = 0
    act8[abs(act8) < 1e-8] = 0
   
    a = lam[0]
    b = lam[1]
    c = 0
    
    abond2 = bond2[0]
    bbond2 = bond2[1]
    abond5 = bond5[0]
    bbond5 = bond5[1]
    abond8 = bond8[0]
    bbond8 = bond8[1]
    aact2 = act2[0]
    bact2 = act2[1]
    aact5 = act5[0]
    bact5 = act5[1]
    aact8 = act8[0]
    bact8 = act8[1]
    
    lx = lengthX/2
    ly = lengthY/2
    state1 = -(1/2)*(a*lx**2 + b*ly**2 + c*lx*ly)
    state2 = -(1/2)*(-b*lx**2 + -a*ly**2+ c*lx*ly)
    totalDeformationLaminate = abs(state1)+abs(state2)

    # - Transverse Strain Laminate- double check 
    [w20f,...,v03f] = lam
    EXY = (0.5*(diff(u0,y) + diff(v0,x) + diff(w0,x)*diff(w0,y)).subs([(w20,w20f),...,(v03,0)])
    EXY = float(abs(EXY.subs([(x,lengthX),(y,lengthY)])))

    e0kk = [[ex0.subs([(w20,w20f),...,(v03,0)])],
            [ey0.subs([(w20,w20f),...,(v03,0)])],
            [e0xy.subs([(w20,w20f),...,(v03,0)])],
            [kx.subs([(w20,w20f),...,(v03,0)])],
            [ky.subs([(w20,w20f),...,(v03,0)])],
            [kxy.subs([(w20,w20f),...,(v03,0)])],

    TransverseStrain = EXY

    
    #   - Potential Energy for 3 MFCs - 
    shape1 = -(1/2)*(abond2*lx**2 + bbond2*ly**2)
    shape2 = -(1/2)*(aact2*lx**2 + bact2*ly**2)
    totalDeformationLaminateMFC2 =  abs(shape1)+abs(shape2)
    shape1 = -(1/2)*(abond5*lx**2 + bbond5*ly**2)
    shape2 = -(1/2)*(aact5*lx**2 + bact5*ly**2)
    totalDeformationLaminateMFC5 =  abs(shape1)+abs(shape2)
    shape1 = -(1/2)*(abond8*lx**2 + bbond8*ly**2)
    shape2 = -(1/2)*(aact8*lx**2 + bact8*ly**2)
    totalDeformationLaminateMFC8 =  abs(shape1)+abs(shape2)
    
    if TransverseStrain > 0.001350: #constrain the transverse strain for the fitness function
       TransverseStrain = 0
    fitnessFunction = totalDeformationLaminateMFC8*TransverseStrain
    #fitnessFunction = totalDeformationLaminate*TransverseStrain
    print("Fitness: ", fitnessFunction, "Solver Time: ", timeelasped, 'lengths: ',...
     (lengthX,lengthY), 'Transverse Strain: ', EXY*1e6)

    return fitnessFunction, EXY*1e6, totalDeformationLaminateMFC2, totalDeformationLaminate,...
     totalDeformationLaminateMFC5, totalDeformationLaminateMFC8, a,b,c

#'''
# - - Running - - 
genresults = [] 
solutions = []
resfitness = []
resX = []
resY = []
transversestrain = []
resLaminate = []
resMFC2 = []
resMFC5 = []
resMFC8 = []
defparams = []
pop = 20
maxiter = 50
maxLength = 0.6096/2  #12 inches
minLength = 0.0762  #3 inches
for s in range(pop):
    solutions.append( [random.uniform(minLength,maxLength),random.uniform(minLength,maxLength)]) 

for i in range(maxiter):
    
    rankedsolutions = []
    for s in solutions:
        lengths = np.array([s[0],s[1]],dtype=float)
        lengths = lengths.astype('float64')
        fit,TS,fitM2,fitL,fitM5,fitM8,wa,wb,wc = ga(lengths)
        rankedsolutions.append([fit,s,TS,fitM2,fitL,fitM5,fitM8,wa,wb,wc])
    
    for j in rankedsolutions:
        if j[0] != 0:
            resfitness.append(j[0])
            resX.append(j[1][0])
            resY.append(j[1][1])
            transversestrain.append(j[2])
            resMFC2.append(j[3])
            resLaminate.append(j[4])
            resMFC5.append(j[5])
            resMFC8.append(j[6])
            defparams.append([j[7],j[8],j[9]])
    rankedsolutions.sort()
    rankedsolutions.reverse()
    print(f" --- Generation {i} best solution --- ")
    print(rankedsolutions[0])
    genresults.append(rankedsolutions[0])

    
    

    if i >= 2:
        currentgen = np.round_(genresults[i][1], decimals = 4)
        lastgen = np.round_(genresults[i-1][1], decimals = 4)

        if currentgen[0] == lastgen[0] and currentgen[1] == lastgen[1]:
            break


    bestsolutions = rankedsolutions[:5] #top 5 

    LX = []
    LY = []

    for s in bestsolutions:
        LX.append(s[1][0])
        LY.append(s[1][1])

    newGen = []
    newGen.append(rankedsolutions[0][1]) #top result always stays in
    for _ in range(pop-1):
        e1 = rankedsolutions[0][1][0]*random.uniform(0.8,1.2)
        e2 = rankedsolutions[0][1][1]*random.uniform(0.8,1.2)
        if e1 > maxLength:
            e1 = maxLength
        if e1 < minLength:
            e1 = minLength
        if e2 > maxLength:
            e2 = maxLength
        if e2 < minLength:
            e2 = minLength
        newGen.append([e1,e2])
    for s in range(len(solutions)):
        solutions[s] = newGen[s]



genX = []
genY = []
genFit = []
for x in genresults:
    genX.append(x[1][0])
    genY.append(x[1][1])
    genFit.append(x[0])



plt.figure(1)
con = plt.tricontour(resX, resY, resfitness,200)
plt.plot(genX,genY,'.')
cbar = plt.colorbar(con)

plt.ylabel('Y Side Length [m]')
plt.xlabel('X Side Length [m]')
cbar.set_label('Fitness')
plt.show()


plt.figure(2)
con2 = plt.tricontour(resX, resY, transversestrain,200)
plt.plot(genX,genY,'.')
cbar2 = plt.colorbar(con2)

plt.ylabel('Y Side Length [m]')
plt.xlabel('X Side Length [m]')
cbar2.set_label('Transverse Strain')
plt.show()

plt.figure(3)
con3 = plt.tricontour(resX, resY, resMFC2,200)
plt.plot(genX,genY,'.')
cbar3 = plt.colorbar(con3)

plt.ylabel('Y Side Length [m]')
plt.xlabel('X Side Length [m]')
cbar3.set_label('Total Deformation Laminate + MFC M2814 [m]')
plt.show()

plt.figure(4)
con3 = plt.tricontour(resX, resY, resMFC5,200)
plt.plot(genX,genY,'.')
cbar3 = plt.colorbar(con3)

plt.ylabel('Y Side Length [m]')
plt.xlabel('X Side Length [m]')
cbar3.set_label('Total Deformation Laminate + MFC M5628 [m]')
plt.show()

plt.figure(5)
con3 = plt.tricontour(resX, resY, resMFC8,200)
plt.plot(genX,genY,'.')
cbar3 = plt.colorbar(con3)

plt.ylabel('Y Side Length [m]')
plt.xlabel('X Side Length [m]')
cbar3.set_label('Total Deformation Laminate + MFC M8557 [m]')
plt.show()


plt.figure(6)
con4 = plt.tricontour(resX, resY, resLaminate,200)
plt.plot(genX,genY,'.')
cbar4 = plt.colorbar(con4)

plt.ylabel('Y Side Length [m]')
plt.xlabel('X Side Length [m]')
cbar4.set_label('Total Deformation Laminate [m]')
plt.show()

top = genresults[-1]
a = top[7]
b = top[8]
c = 0
a2 = b*-1
b2 = a*-1
c2 = 0
xlen = genX[-1]
ylen = genY[-1]
x = np.outer(np.linspace(-xlen/2, xlen/2, 100), np.ones(100))
y = np.outer(np.linspace(-ylen/2, ylen/2, 100), np.ones(100)).T # transpose
w = -(1/2)*(a*x**2 + b*y**2)
w2 = -(1/2)*(a2*x**2 + b2*y**2)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, w,cmap='viridis', edgecolor='none')
ax.plot_surface(x, y, w2,cmap='viridis', edgecolor='none')
 
plt.show()



