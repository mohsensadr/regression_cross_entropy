import sympy as s
from sympy.printing import ccode, fcode
import chaospy as cp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from scipy import integrate
#from scipy.special import *
import os
#from Direct_Newton_Method_1D import *
from Direct_Nelder_Method_1D import *
#import random
#random.seed(30)

def fn(v,m,sig,w,n):
    z = 0
    for i in range(n):
        z += 1.0/s.sqrt(2.0*s.pi*sig[i]**2)*s.exp(- (v-m[i])**2/(2.0*sig[i]**2))
    return w*z


#### 0: setting
dimY = 4## number of moments (Lagrange multipliers)
L=10.0## space of velocity domain is [-L,L]
n_quad=100## number of quadrature points
n_gauss = 4## number of Gaussians used to construct distr.
n_points = 1000## number of data points to generate

#### 1: Quadrature rule
v_quad,w_quad = np.polynomial.legendre.leggauss(n_quad)# find the quadrature points/weights
#v_quad,w_quad = np.polynomial.chebyshev.chebgauss(n_quad)
#v_quad,w_quad = roots_legendre(n)
#v_quad,w_quad =  = np.polynomial.hermite.hermgauss(n_quad)
id =  [0 for i in range(n_quad)]
v_quad = v_quad*L
w_quad = w_quad*L

#### 2: A distribution as sum of Gaussians (symbolic)
from sympy import Array
v = s.symbols('v')#f(v)
m = s.symbols('m1:'+str(n_gauss+1))#mean
sig = s.symbols('sig1:'+str(n_gauss+1), positive=True)#variance
w = 1.0/(1.0*n_gauss)#w is the coefficient for each Gaussian, we considered equal here
f = fn(v, m, sig, w, n_gauss)# the distribution as summation of Gaussian

#### 3: Moments of ditr. (symbolic)
mom = [[] for i in range(dimY+1)]
for i in range(0,dimY+1):
    exp = f*v**i
    mom[i] = s.simplify(exp.integrate((v, -s.oo, s.oo)))

#### 4: find the mean and variance of last Guassian to make sure total mean=0 and variance=1 (symbolic)
m_last = s.solveset(mom[1], m[-1])
sig_last = s.solveset( mom[2]-1, sig[-1],s.Interval(0,s.oo) )

#### 5: Generate data
## space to sample m_1,...,m_{nguass-1} and sig_1,...,sig_{nguass-1}.
sp_m = [-1.0,1.0]
sp_sig = [0.5,2.0]


name_file = "data_set/data_nq"+str(n_quad)+"_ng"+str(n_gauss)+".txt"
fg = open(name_file, "a");
st0 = "#"
for i in range(0, 2 * dimY + n_quad):
    if i == 0:
        st0 += "{:<13}".format("l" + str(i + 1))
    elif i < 1 * dimY:
        st0 += "{:<14}".format("l" + str(i + 1))
    elif i < 2 * dimY:
        st0 += "{:<14}".format("mom" + str(i - dimY + 1))
    else:
        st0 += "{:<14}".format("f" + str(i - 2*dimY + 1))
st0 += "\n"
if os.stat(name_file).st_size == 0:
    fg.write(st0);


for inp in range(n_points):
    done = 0;
    while done == 0:
        ## sample mean and variance of Gaussians
        mm = np.ones(n_gauss)*sp_m[0] + np.random.rand(n_gauss)*(sp_m[1]-sp_m[0])
        ssig = np.ones(n_gauss)*sp_sig[0] + np.random.rand(n_gauss)*(sp_sig[1]-sp_sig[0])

        ## Most likely, we endup dealing with a variation of standard Gaussian
        ## So, let's fix the first Gaussian to be standard
        #mm[0] = 0.0; ssig[0] = 1.0;

        ## Let's fix the last one
        subs_arg_m1 = [ (m[i],mm[i]) for i in range(n_gauss-1)]
        mm[n_gauss-1] = m_last.subs(subs_arg_m1).args[0]
        subs_arg_m1 = [ (m[i],mm[i]) for i in range(n_gauss)]+[ (sig[j],ssig[j]) for j in range(n_gauss-1)]
        ## Make sure the variance of the last Guassian is positive
        ## otherwise resetart
        if len(sig_last.subs(subs_arg_m1).args) > 0:
            done = 1
            ssig[n_gauss-1] = sig_last.subs(subs_arg_m1).args[0]
    ## At this point, we already found a distr. with total mean=0.0 and variance =1.0
    ## compute the moments analytically by substitution of mean and variances of all Gaussians
    subs_arg = [ (m[i],mm[i]) for i in range(n_gauss)]+[ (sig[j],ssig[j]) for j in range(n_gauss)]
    momf = [float(mom[i].subs(subs_arg)) for i in range(dimY+1)]

    ## find the ditr. values at quadrature points
    fv = f.subs(subs_arg)
    ff = s.lambdify(v, fv, "numpy")
    f_quad = ff(v_quad)

    ## solving MED directly
    l0 = np.array([0.0, 0.0, 0.0, 0.0])
    #La, Mo = find_exact_MED(momf[1:dimY + 1], l0, f_quad, v_quad, w_quad)
    La, Mo, done = sample_new_Nl_direct_Nelder(dimY, momf[1:dimY + 1], l0, f_quad, v_quad, w_quad)

    if done == 1:
        print("\n\n---   DONE   ---\n\n")
        print("La=" + str(La)+"Mo="+str(Mo) )
        st = "";
        mo = ['{:.16e}'.format(float(x)) for x in Mo[:]]
        la = ['{:.16e}'.format(float(x)) for x in La[:]]
        f_q = ['{:.16e}'.format(float(x)) for x in f_quad[ :]]
        for j in range(0, dimY):
            st += la[j] + "  "
        for j in range(0, dimY):
            st += mo[j] + "  "
        for j in range(0, n_quad):
            st += f_q[j] + "  "
        st += "\n"
        fg.write(st);
fg.close()


'''
fig, ax = plt.subplots();
plt.plot(vp,fp,color='black')
#[plt.plot(vv,Z(vv,x[:,p][20:24])/intt[p][0]) for p in range(P)]
#ax.set_ylabel(r"$\frac{\exp(-\sum_{i=1}^6 \lambda_i v^i)}{\int_{-10}^{10} \exp({-\sum_{i=1}^6 \lambda_i v^i}) dv}$")
#plt.xlim(-3.0,3.0)
#name = "distrs"
#fig.set_size_inches(size*cm, size*cm)
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
plt.plot(v_quad,f_quad, linewidth=0, marker="s")
plt.plot(v_quad,fMED_quad, linewidth=0, marker="o")
plt.plot(v_quad,id, linewidth=0, marker=".")
plt.show()
'''