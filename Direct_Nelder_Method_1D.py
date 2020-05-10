import numpy as np
from scipy.optimize import minimize
from scipy import integrate
from numpy import linalg as LA

def Z(v,l,dimY):
    f = 0.0;
    for i in range(1,dimY+1):
        f += l[i-1]*v**i
    return np.exp(-f);

def Hi(v,l,dimY,i):
    return Z(v,l,dimY)*v**i

def Mom(l, dimY, dimX, f_quad, v_quad, w_quad):
    q = [0 for x in range(dimX)]
    #D, err = integrate.quad(Z, -1e1, 1e1, args=(l, dimY));
    D = np.sum(Z(v_quad,l,dimY)*f_quad*w_quad)
    for i in range(0, dimX):
        #intt, err = integrate.quad(Hi, -1e1, 1e1, args=(l, dimY, i + 1));
        intt = np.sum(Hi(v_quad, l, dimY, i+1) * f_quad * w_quad)
        q[i] = intt / D
    return q;

def objective(l, Q, dimY, dimX, f_quad, v_quad, w_quad):
    qq = Mom(l, dimY, dimX, f_quad, v_quad, w_quad);
    sum = 0.0;
    sum += ( Q[0]-qq[0])**2
    for i in range(0,len(qq)):
        if i == 0:
            sum += abs((Q[i] - qq[i]) ) ** 2
        else:
           sum += abs( (Q[i]-qq[i])/qq[i] )**2
    return sum

def sample_new_Nl_direct_Nelder(dimY, q, l0,f_quad, v_quad, w_quad):
    dimX = dimY
    done = 0
    iter = 0
    while done == 0:
        print("l0="+str(l0))
        solution = minimize(objective, l0, args=(q, dimY, dimX, f_quad, v_quad, w_quad), method='Nelder-Mead', tol=1e-20,
                                options={"disp": True, 'ftol': 1e-20, "maxiter": 5000})
        l0 = solution.x
        #print("dimX="+str(dimX)+"cost is "+str(objective(l0, q, dimY, dimX, f_quad, v_quad, w_quad))+"and l=" +str(l0))
        tol = objective(l0, q, dimY, dimX, f_quad, v_quad, w_quad)
        iter = iter + 1
        if tol < 1e-12:
            done = 1
        else:
            if iter>10:
                done = -1
    qq = Mom(l0, dimY, dimX, f_quad, v_quad, w_quad)
    return l0, qq, done

