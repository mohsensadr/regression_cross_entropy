import numpy as np
from scipy import integrate
import math
import os
from decimal import *

def Z(v,l,dimY):
    getcontext().prec = 30
    f = 0.0;
    for i in range(1,dimY+1):
        f += l[i-1]*v**i
    return np.exp(-f);
def Hi(v,l,dimY,i):
    return Z(v,l,dimY)*v**i

def Hij(v,l,dimY,i,j):
    return Z(v,l,dimY)*v**i*v**j

def Mom(l, dimY, f_quad, v_quad, w_quad):
    q = [0 for x in range(dimY)]
    #D, err = integrate.quad(Z, -1e1, 1e1, args=(l, dimY));
    D = np.sum(Z(v_quad,l,dimY)*f_quad*w_quad)
    for i in range(0, dimY):
        #intt, err = integrate.quad(Hi, -1e1, 1e1, args=(l, dimY, i + 1));
        intt = np.sum(Hi(v_quad, l, dimY, i+1) * f_quad * w_quad)
        q[i] = intt / D;
    return q;

def Mom2(l, dimY, j, f_quad, v_quad, w_quad):
    q = [0 for x in range(dimY)]
    #D, err = integrate.quad(Z, -1e1, 1e1, args=(l, dimY));
    D = np.sum(Z(v_quad, l, dimY) * f_quad * w_quad)
    for i in range(0, dimY):
        #intt, err = integrate.quad(Hij, -1e1, 1e1, args=(l, dimY, i + 1, j + 1));
        intt = np.sum(Hij(v_quad, l, dimY, i + 1, j+1) * f_quad * w_quad)
        q[i] = intt / D;
    return q;

def gradient(l,p,f_quad, v_quad, w_quad):
    ## l is lagrange multipliers \in R^N
    ## p is the vector of given moments \in R^N
    dimY=len(l);
    g = np.zeros(dimY);
    q = Mom(l, dimY,f_quad, v_quad, w_quad)
    for i in range(dimY):
        g[i] = -q[i]+p[i];
    return g

def Hessian(l,f_quad, v_quad, w_quad):
    ## l is lagrange multipliers \in R^N
    ## p is the vector of given moments \in R^N
    dimY=len(l);
    H = np.zeros((dimY,dimY));
    for j in range(dimY):
        H[j][:] = Mom2(l, dimY, j,f_quad, v_quad, w_quad)
    q = Mom(l, dimY,f_quad, v_quad, w_quad)
    for j in range(dimY):
        for i in range(dimY):
            H[j][i] += q[i]*q[j]
    return H


def C(l,p,f_quad, v_quad, w_quad):
    q = Mom(l, len(l),f_quad, v_quad, w_quad)
    #res = 0.0;
    #res = (p[0]-q[0])**2
    #for i in range(1,len(l)):
    #    res += (p[i]-q[i])**2/(p[i])**2
    #return np.sqrt(res);
    c = np.linalg.norm(np.array(p)-np.array(q)) / np.linalg.norm(np.array(p))
    return c


def find_exact_MED(p_ex,ll, f_quad, v_quad, w_quad):
    dimY = len(p_ex)
    Ns=10; s = np.ones(Ns); pp=0.5; c1 = 1e-3; beta = 0.5;
    for i in range(1,Ns):
        s[i] = pp**i;

    done = 0
    l0 = ll
    while done == 0:
        tol = 0.1
        while tol>1e-10:
            H = Hessian(l0,f_quad, v_quad, w_quad)
            g = gradient(l0,p_ex,f_quad, v_quad, w_quad)
            b = -g;
            dl = np.linalg.solve(H, b)

            #for i in range(Ns):
            #    beta = pp**i;
            #    if ( C(l0 + beta*dl,p_ex,f_quad, v_quad, w_quad)< C(l0,p_ex,f_quad, v_quad, w_quad) + c1*beta*np.dot(dl,g)):
            #        break
            beta = 0.5
            for i in range(dimY):
                l0[i] = l0[i] + beta*dl[i];
            #l0 = l0 + beta*dl;
            tol = C(l0,p_ex,f_quad, v_quad, w_quad)
            q = Mom(l0, len(l0),f_quad, v_quad, w_quad)
            print(tol, beta)
            if beta<1e-10 or np.isnan(q[0]) == 1 or np.isinf(q[0]) == 1:
                print(abs(beta*dl))
                break;
        print(l0)
        q = Mom(l0, len(l0),f_quad, v_quad, w_quad)
        if tol<1e-6 and ( np.isnan(q[0]) == 0 and np.isinf(q[0]) == 0):
            print("\n\n---   DONE   ---\n\n")
            done = 1
        else:
            done = 0
    return l0, q

#find_exact_MED([0.0,1.0,-0.166128661675755,3.64176831409943],[0.1,0.5,0.1,0.1])
