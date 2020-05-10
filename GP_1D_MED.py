import numpy as np
import gpflow as gp
from scipy import integrate
#from integration_max_entropy import moments
#from sampling import samples
from random import randint
from numpy import linalg as LA
from numpy.linalg import inv
import os
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import tensorflow as tf


N1 = 2000;
N2 = 1000;
method = "GPR"

dimY = 4; 
dimNQ = 20;
i0 = dimY+2; i1=2*dimY+dimNQ; i11=0; i2=dimY
dimX =dimY-2+dimNQ; 
'''
        ----    Reading Data      ----
'''
title = "nq20_ng4"
address = "data_set/data_"+title+".txt"
x0 = np.loadtxt(address,skiprows=1, unpack=True);
dim = len(x0[:,0]);## 28 here
'''
        ----    Scale Data      ----
'''
def transf_to(xx,m,v,n):
    for i in range(2,n):
        if abs(v[i])>1e-15:
            xx[i,:] = (xx[i,:] - m[i] ) / v[i] ** 0.5+1.0
    return xx

def transf_back(xx,m,v,n):
    for i in range(2,n):
        if abs(v[i]) > 1e-15:
            xx[i,:] = xx[i,:]* v[i] ** 0.5 + m[i]
    return xx
vvar = []
mm = []
for i in range (dim):
    mm.append(np.mean(x0[i,:]));
    vvar.append(np.var(x0[i, :]));

np.savetxt("data_set/GPR_mean_input.dat",mm[i0:i1],delimiter=',')
np.savetxt('data_set/GPR_variance_input.dat',vvar[i0:i1],delimiter=',')

np.savetxt('data_set/GPR_mean_output.dat',mm[i11:i2],delimiter=',')
np.savetxt('data_set/GPR_variance_output.dat',vvar[i11:i2],delimiter=',')

x0 = transf_to(x0,mm,vvar,dim);
x = np.array(np.transpose(x0))

'''
        ----    Training      ----
'''
ytrain = []; xtrain = [];
for i in range(N1):
    xtrain.append(x[i,i0:i1].copy())
    ytrain.append(x[i,i11:i2].copy())
#xtrain = np.array(xtrain)
#ytrain = np.array(ytrain)
print("Training data is set.")

y2 = np.array(ytrain)
len_sc = []
for i in range(0,dimX):
    len_sc.append(i);

k1 = gp.kernels.RBF(input_dim=dimX, active_dims=len_sc, ARD=True, name="RBF")
k2 = gp.kernels.Linear(input_dim=dimX, active_dims=len_sc, ARD=True)
k3 = gp.kernels.ArcCosine(input_dim=dimX,order=1, active_dims=len_sc, ARD=True)
k4 = gp.kernels.Polynomial(input_dim=dimX,degree=2, active_dims=len_sc, ARD=True)
k5 = gp.kernels.Matern12(input_dim=dimX, active_dims=len_sc, ARD=True)
k6 = gp.kernels.Matern32(input_dim=dimX, active_dims=len_sc, ARD=True)
k7 = gp.kernels.Matern52(input_dim=dimX, active_dims=len_sc, ARD=True)
k8 = gp.kernels.Constant(input_dim=dimX, active_dims=len_sc)

meth = "RBF"
kernel = k1#+k8#gp.kernels.Pow3b2(input_dim=dimX, active_dims=len_sc)

model = gp.models.GPR(xtrain, ytrain, kernel)
f_name = method + "_" + meth + "_wom"+"N1_"+str(N1)

model.likelihood.variance = 1e-10
model.likelihood.variance.trainable = False
#model.kern.lengthscales = np.ones((dimX))#vvarX[2:dimX+2]#[np.var(X[:,2]),np.var(X[:,3]),np.var(X[:,4]),np.var(X[:,5])]
#model.kern.lengthscales = [np.mean(X[:,2]),np.mean(X[:,3]),np.mean(X[:,4]),np.mean(X[:,5])]
#model.kern.lengthscales.trainable = True
#model.mean_function.A.trainable  = True
#model.mean_function.b.trainable  = True
#myfactr = 1e-12
opt = gp.train.ScipyOptimizer(tol=1e-12, options={"eps": 1E-12,"disp":True,'ftol': 1e-12})
#opt = gp.train.ScipyOptimizer(method='SLSQP',tol=1e-12, options={"eps": 1E-12})
#opt = gp.train.ScipyOptimizer(method='SLSQP',options={'maxiter': 5000, 'ftol': 1e-16, 'disp': True, 'eps': 1e-16})
opt.minimize(model, disp=True, maxiter=10000)

point = str(N1)

'''
save the model 
'''
#import pickle
m_name = "GP_models/"+title+"_"+str(N1)+".md"
#if os.stat(m_name).st_size == 0:
saver= gp.saver.Saver()
saver.save(m_name, model)

ystar1,varstar1 = model.predict_y(xtrain)
err1 = abs(ystar1-ytrain)
Relerr1 = abs(ystar1-ytrain)/abs(ystar1)

'''
print(str(model.read_trainables()))
q = np.array([[-1.0,2.2]])
ystar1,varstar1 = model.predict_y(q)
print(varstar1)
'''

'''
        ----    Predict      ----
'''

xtest = []; ytest = [];

for i in range(N1,N1+N2):
    xtest.append(x[i,i0:i1].copy())
    ytest.append(x[i,i11:i2].copy())

ystar2, varstar2 = model.predict_y(xtest)
err2 = abs(ystar2-ytest)
Relerr2 = abs(ystar2-ytest)/abs(np.array(ytest))

'''
        ----    Plots      ----
'''

'''
name_file = "error/Relerror_post"+f_name+".txt"
f = open(name_file, "a");
st = ""
st += point+"  "+str(dimX);
for i in range(dimY):
    st+="  "+'{:.3e}'.format(float(np.mean(err2[:,i])))
for i in range(dimY):
    st+="  "+'{:.3e}'.format(float(np.mean(Relerr2[:,i])))
for i in range(dimY):
    st+="  "+'{:.3e}'.format(float(np.var(Relerr2[:,i])))
for i in range(dimY):
    st+="  "+'{:.3e}'.format(float(np.mean(varstar2[:,i])))
st+="\n"
f.write(st);
'''


'''
name_file = "error/coeff"+f_name+"_"+str(dimX)+".txt"
f = open(name_file, "a");
st = "#### N="
st += str(N1)+":  \n";
st += str(model.read_trainables())
st+="\n"
f.write(st);
'''

'''
if meth == "RBF":
    st+="  "+'{:.6e}'.format(float(model.kern.variance.value))
if meth == "ArcCosine":
    st += "  " + '{:.6e}'.format(float(model.kern.bias_variance.value))
if meth == "Linear":
    for i in range(dimX - 2):
        st += "  " + '{:.6e}'.format(float(model.kern.variance.value[i]))
for i in range(dimX-2):
    if meth == "RBF":
        st+="  "+'{:.6e}'.format(float(model.kern.lengthscales.value[i]))
    elif meth == "ArcCosine":
        st += "  " + '{:.6e}'.format(float(model.kern.weight_variances.value[i]))
for i in range(dimX):
    for j in range(dimY):
        st+="  "+'{:.6e}'.format(float(model.mean_function.A.value[i][j]))
for i in range(dimY):
    st+="  "+'{:.6e}'.format(float(model.mean_function.b.value[0,i]))
st+="\n"
f.write(st);

'''



'''
for i in range(dimY):
    print("E[rel_error_post(Relerr2)_"+str(i)+"]= "+str(np.mean(Relerr2[:,i])))
for i in range(dimY):
    print("E[error_post(err2)_"+str(i)+"]= "+str(np.mean(err2[:,i])))
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
pp = []
for i in range(0,dimY):
    pp.append(i);


fig, ax = plt.subplots();
[plt.plot(varstar1[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("var")
ax.set_yscale("log")
#name = "Figs/post_error"+str(N1)+"_"+meth
plt.legend()
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
#plt.show()

fig, ax = plt.subplots();
[plt.plot(err1[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("error")
ax.set_yscale("log")
#name = "Figs/post_error"+str(N1)+"_"+meth
plt.legend()
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
#plt.show()

fig, ax = plt.subplots();
[plt.plot(Relerr1[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("rel error")
ax.set_yscale("log")
#name = "Figs/post_error"+str(N1)+"_"+meth
plt.legend()
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
#plt.show()


fig, ax = plt.subplots();
[plt.plot(varstar2[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("var")
ax.set_yscale("log")
#name = "Figs/post_error"+str(N1)+"_"+meth
plt.legend()
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
#plt.show()

fig, ax = plt.subplots();
[plt.plot(err2[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("error")
ax.set_yscale("log")
#name = "Figs/post_error"+str(N1)+"_"+meth
plt.legend()
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
#plt.show()


fig, ax = plt.subplots();
[plt.plot(Relerr2[:,p],'.',label='p='+str(p+1)) for p in pp]
ax.set_ylabel("rel error")
ax.set_yscale("log")
#name = "Figs/post_error"+str(N1)+"_"+meth
plt.legend()
#plt.savefig(name+".pdf",format='pdf', bbox_inches="tight", dpi=300);
plt.show()

