import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraypad import pad
from numpy.ma.core import where
import scipy.stats as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from pprint import pprint
from functools import reduce
import pandas as pd
import csv
from pprint import pprint

#Layer 0:
L0 = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L0_res.csv', sep = ',')
L0 = L0.to_numpy()
L0 = L0.flatten()
hist, bin_edges = np.histogram(L0, bins = 100)
# pprint(hist)
# pprint(bin_edges)
bin_centres = (bin_edges[1:]+bin_edges[:-1])/2
# pprint(len(bin_centres))
bin_centres_narrow = []
hist_narrow = []
i = 0
for i in range(0,9):
    if bin_centres[i]>= -1 and bin_centres[i]<= 1:
        bin_centres_narrow.append(bin_centres[i])
        hist_narrow.append(hist[i])
    i +=1
# pprint(bin_centres_narrow)

def gaussian(x,mu,sig,a):
    return a*np.exp(-((x-mu)**2)/(2*sig**2))

u = np.sqrt(hist[np.where(hist!=0)])
pprint(u)
# pprint(hist[np.where(hist!=0)])
p0 = [0,0.2,35] #initial parameters
popt, pcov = curve_fit(gaussian,bin_centres[np.where(hist!=0)],hist[np.where(hist!=0)])
dymin = (hist[np.where(hist!=0)] - gaussian(bin_centres[np.where(hist!=0)],*popt))/u
min_chisq = sum(dymin*dymin)
dof = len(bin_centres[np.where(hist!=0)]) - len(popt)
print('LAYER 0')
print("Chi square: ",min_chisq) 
print("Number of degrees of freedom: " ,dof)
print("Chi square per degree of freedom: ",min_chisq/dof) 
perr = np.sqrt(np.diag(abs(pcov)))
print(popt)
print(perr)
# pprint(pcov)
i = np.linspace(-2,2,500)
plt.plot(i,gaussian(i, *popt), label = 'Fit')
plt.hist(L0,bins = 100)
plt.show()

#Layer 1:
L1 = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L1_res.csv', sep = ',')
L1 = L1.to_numpy()
L1 = L1.flatten()
hist, bin_edges = np.histogram(L1, bins = 100)
# pprint(hist)
# pprint(bin_edges)
bin_centres = (bin_edges[1:]+bin_edges[:-1])/2
# pprint(len(bin_centres))
bin_centres_narrow = []
hist_narrow = []
# i = 0
# for i in range(0,9):
#     if bin_centres[i]>= -1 and bin_centres[i]<= 1:
#         bin_centres_narrow.append(bin_centres[i])
#         hist_narrow.append(hist[i])
#     i +=1
# pprint(bin_centres_narrow)

def gaussian(x,mu,sig,a):
    return a*np.exp(-((x-mu)**2)/(2*sig**2))

u = np.sqrt(hist[np.where(hist!=0)])
pprint(hist[np.where(hist!=0)])
p0 = [0,0.2,35] #initial parameters
popt, pcov = curve_fit(gaussian,bin_centres[np.where(hist!=0)],hist[np.where(hist!=0)])
dymin = (hist[np.where(hist!=0)] - gaussian(bin_centres[np.where(hist!=0)],*popt))/u
min_chisq = sum(dymin*dymin)
dof = len(bin_centres[np.where(hist!=0)]) - len(popt)
print('LAYER 1')
print("Chi square: ",min_chisq) 
print("Number of degrees of freedom: " ,dof)
print("Chi square per degree of freedom: ",min_chisq/dof) 
perr = np.sqrt(np.diag(abs(pcov)))
pprint(popt)
pprint(perr)
# pprint(pcov)
i = np.linspace(-2,2,500)
plt.plot(i,gaussian(i, *popt), label = 'Fit')
plt.hist(L1,bins = 100)
plt.show()


#Layer 2:
L2 = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L2_res.csv', sep = ',')
L2 = L2.to_numpy()
L2 = L2.flatten()
hist, bin_edges = np.histogram(L2, bins = 20)
# pprint(hist)
# pprint(bin_edges)
bin_centres = (bin_edges[1:]+bin_edges[:-1])/2
# pprint(len(bin_centres))
bin_centres_narrow = []
hist_narrow = []
# i = 0
# for i in range(0,9):
#     if bin_centres[i]>= -1 and bin_centres[i]<= 1:
#         bin_centres_narrow.append(bin_centres[i])
#         hist_narrow.append(hist[i])
#     i +=1
# pprint(bin_centres_narrow)

def gaussian(x,mu,sig,a):
    return a*np.exp(-((x-mu)**2)/(2*sig**2))

u = np.sqrt(hist[np.where(hist!=0)])
pprint(hist[np.where(hist!=0)])
p0 = [0,0.2,35] #initial parameters
popt, pcov = curve_fit(gaussian,bin_centres[np.where(hist!=0)],hist[np.where(hist!=0)])
dymin = (hist[np.where(hist!=0)] - gaussian(bin_centres[np.where(hist!=0)],*popt))/u
min_chisq = sum(dymin*dymin)
dof = len(bin_centres[np.where(hist!=0)]) - len(popt)
print('LAYER 2')
print("Chi square: ",min_chisq) 
print("Number of degrees of freedom: " ,dof)
print("Chi square per degree of freedom: ",min_chisq/dof) 
perr = np.sqrt(np.diag(abs(pcov)))
pprint(popt)
pprint(perr)
# pprint(pcov)
i = np.linspace(-0.1,0.1,500)
plt.plot(i,gaussian(i, *popt), label = 'Fit')
plt.hist(L2,bins = 20)
plt.show()

#Layer 3:
L3 = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L3_res.csv', sep = ',')
L3 = L3.to_numpy()
L3 = L3.flatten()
hist, bin_edges = np.histogram(L3, bins = 50)
# pprint(hist)
# pprint(bin_edges)
bin_centres = (bin_edges[1:]+bin_edges[:-1])/2
# pprint(len(bin_centres))
bin_centres_narrow = []
hist_narrow = []
# i = 0
# for i in range(0,9):
#     if bin_centres[i]>= -1 and bin_centres[i]<= 1:
#         bin_centres_narrow.append(bin_centres[i])
#         hist_narrow.append(hist[i])
#     i +=1
# pprint(bin_centres_narrow)

def gaussian(x,mu,sig,a):
    return a*np.exp(-((x-mu)**2)/(2*sig**2))

u = np.sqrt(hist[np.where(hist!=0)])
pprint(hist[np.where(hist!=0)])
p0 = [0,0.2,35] #initial parameters
popt, pcov = curve_fit(gaussian,bin_centres[np.where(hist!=0)],hist[np.where(hist!=0)])
dymin = (hist[np.where(hist!=0)] - gaussian(bin_centres[np.where(hist!=0)],*popt))/u
min_chisq = sum(dymin*dymin)
dof = len(bin_centres[np.where(hist!=0)]) - len(popt)
print('LAYER 3')
print("Chi square: ",min_chisq) 
print("Number of degrees of freedom: " ,dof)
print("Chi square per degree of freedom: ",min_chisq/dof) 
perr = np.sqrt(np.diag(abs(pcov)))
pprint(popt)
pprint(perr)
# pprint(pcov)
i = np.linspace(-0.1,0.1,500)
plt.plot(i,gaussian(i, *popt), label = 'Fit')
plt.hist(L3,bins = 50)
plt.show()

#Layer 4:
L4 = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L4_res.csv', sep = ',')
L4 = L4.to_numpy()
L4 = L4.flatten()
hist, bin_edges = np.histogram(L4, bins = 100)
# pprint(hist)
# pprint(bin_edges)
bin_centres = (bin_edges[1:]+bin_edges[:-1])/2
# pprint(len(bin_centres))
bin_centres_narrow = []
hist_narrow = []
# i = 0
# for i in range(0,9):
#     if bin_centres[i]>= -1 and bin_centres[i]<= 1:
#         bin_centres_narrow.append(bin_centres[i])
#         hist_narrow.append(hist[i])
#     i +=1
# pprint(bin_centres_narrow)

def gaussian(x,mu,sig,a):
    return a*np.exp(-((x-mu)**2)/(2*sig**2))

u = np.sqrt(hist[np.where(hist!=0)])
pprint(hist[np.where(hist!=0)])
p0 = [0,0.2,35] #initial parameters
popt, pcov = curve_fit(gaussian,bin_centres[np.where(hist!=0)],hist[np.where(hist!=0)])
dymin = (hist[np.where(hist!=0)] - gaussian(bin_centres[np.where(hist!=0)],*popt))/u
min_chisq = sum(dymin*dymin)
dof = len(bin_centres[np.where(hist!=0)]) - len(popt)
print('LAYER 4')
print("Chi square: ",min_chisq) 
print("Number of degrees of freedom: " ,dof)
print("Chi square per degree of freedom: ",min_chisq/dof) 
perr = np.sqrt(np.diag(abs(pcov)))
pprint(popt)
pprint(perr)
# pprint(pcov)
i = np.linspace(-2,2,500)
plt.plot(i,gaussian(i, *popt), label = 'Fit')
plt.hist(L4,bins = 100)
plt.show()

#Layer 5:
L5 = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L5_res.csv', sep = ',')
L5 = L5.to_numpy()
L5 = L5.flatten()
hist, bin_edges = np.histogram(L5, bins = 100)
# pprint(hist)
# pprint(bin_edges)
bin_centres = (bin_edges[1:]+bin_edges[:-1])/2
# pprint(len(bin_centres))
bin_centres_narrow = []
hist_narrow = []
# i = 0
# for i in range(0,9):
#     if bin_centres[i]>= -1 and bin_centres[i]<= 1:
#         bin_centres_narrow.append(bin_centres[i])
#         hist_narrow.append(hist[i])
#     i +=1
# pprint(bin_centres_narrow)

def gaussian(x,mu,sig,a):
    return a*np.exp(-((x-mu)**2)/(2*sig**2))

u = np.sqrt(hist[np.where(hist!=0)])
pprint(hist[np.where(hist!=0)])
p0 = [0,0.2,35] #initial parameters
popt, pcov = curve_fit(gaussian,bin_centres[np.where(hist!=0)],hist[np.where(hist!=0)])
dymin = (hist[np.where(hist!=0)] - gaussian(bin_centres[np.where(hist!=0)],*popt))/u
min_chisq = sum(dymin*dymin)
dof = len(bin_centres[np.where(hist!=0)]) - len(popt)
print('LAYER 5')
print("Chi square: ",min_chisq) 
print("Number of degrees of freedom: " ,dof)
print("Chi square per degree of freedom: ",min_chisq/dof) 
perr = np.sqrt(np.diag(abs(pcov)))
pprint(popt)
pprint(perr)
# pprint(pcov)
i = np.linspace(-2,2,500)
plt.plot(i,gaussian(i, *popt), label = 'Fit')
plt.hist(L5,bins = 100)
plt.show()




