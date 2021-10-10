import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraypad import pad
import scipy as sp
from scipy.optimize import curve_fit
from pprint import pprint
from functools import reduce
import pandas as pd
import csv

#Reading in Data:
tracklets = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/Calibrated Simulation Data/tracklets_new_new.csv', sep = ';')
digits_path = pd.read_csv('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/Calibrated Simulation Data/digits.csv')
tracklets = tracklets.drop_duplicates(subset='hcid', ignore_index=True)#can we put this in BIG for loop?
tracklets = tracklets.to_numpy() #data into numpy array
# tracklet_hcid = tracklets.filter(regex='hcid')
# tracklet_slope = tracklets.filter(regex='slope')
# tracklet_dy = tracklets.filter(regex= 'dy')
# tracklet_ypos = tracklets.filter(regex = 'y')

#Can we put this in BIG for loop as well?
sorted_digits = digits_path.sort_values(by = ['hcid','pad'] ,axis = 0) #sorting digits by hcid number and then pad number within
sorted_digits = sorted_digits.drop_duplicates(subset = ['hcid','pad'], keep = 'last')
# pprint(sorted_digits.iloc[0:33,2])
sorted_digits = sorted_digits.to_numpy() #data into numpy array

#Residuals:
L0 = []
L1 = []
L2 = []
L3 = []
L4 = []
L5 = []

iterations = 0

#Nested for loop:
for hcid in tracklets[:,2]:
    tracklet_index = np.where(tracklets[:,2] ==hcid)
    # pprint(tracklet_index)
    tracklet_pad = np.floor(tracklets[tracklet_index[0],10])


    # tracklet_y_pos = tracklets[tracklet_index[0]]
    # pprint(tracklet_pad)
    adc_data = []
    for sorted_hcid in sorted_digits:
        if sorted_hcid[0] == int(hcid):
            # pprint(sorted_hcid)
            adc_data.append(sorted_hcid[2:]) #from pad number to the end
    adc_data = np.asarray(adc_data) #converts from list of numpy arrays to 2D numpy array
    # pprint(adc_data)
    # just_adc_sum = adc_data[:,1]
    # print(type(just_adc_sum))
    # pprint(just_adc_sum)
    # pprint(np.partition(just_adc_sum,-3))
    # pprint(just_adc_sum[np.argpartition(just_adc_sum,-3)])
    # pprint(just_adc_sum[np.argpartition(just_adc_sum,-3,0)[-3:]])
    
    if np.shape(adc_data)[0]==1: #throwing out possible 1 pad tracklets
        continue
    # if np.shape(adc_data)[0]>2: #more than two "rows"
    #     max_start_index = 0
    #     max_sum = sum(just_adc_sum[max_start_index:max_start_index+3])
    #     for i in range(0,np.shape(just_adc_sum)[0]-3):
    #         window = sum(just_adc_sum[i:i+3])
    #         if window > max_sum:
    #             max_start_index = i #find indices of 3 pads with maxiumum adc_sum values
    #     adc_data = adc_data[max_start_index:max_start_index+3]
    # pprint(adc_data)
        # pprint(max_start_index) 
    
    # pprint(range(0,np.shape(adc_data)[0]))
    
    # if np.shape(adc_data)[0]>2: #more than two "rows"
    #     middle_pad_index = 0
    #     for i in range(0,np.shape(adc_data)[0]):
    #         if adc_data[i,0] == int(tracklet_pad):
    #             middle_pad_index = i
    #     adc_data = adc_data[middle_pad_index-1:middle_pad_index+2,:]
    # pprint(adc_data)
    # pprint(adc_data[:,2:32])


    #Making heatmap:
    pad_data = adc_data[:,0]
    # print(pad_data)
    timebin_data = list(range(0, 30))
    # print(len(timebin_data))
    adc_counts = np.abs(adc_data[:,2:32]- 9.5) #subtracting a pedastal
    
    # Plotting heatmap
    im = plt.imshow(adc_counts, origin='lower', aspect='5')
    ax = plt.gca()

    #  Creating colorbar
    cbar = ax.figure.colorbar(im, ax=ax, location='top')
    cbar.ax.set_title("ADC counts", va="top")

    x_ticks = np.arange(adc_counts.shape[1])
    y_ticks = np.arange(adc_counts.shape[0])
    # print(adc_counts.shape[1])

    #We want to show all ticks...
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    #... and label them with the respective list entries.
    ax.set_xticklabels(timebin_data)
    ax.set_yticklabels(pad_data)

    #Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(adc_counts.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(adc_counts.shape[0]+1)-.5, minor=True)
    ax.set_xticks(np.arange(adc_counts.shape[1])-.5, minor=True)
    ax.set_yticks(np.arange(adc_counts.shape[0])-.5, minor=True)
    ax.set_xlabel('time bin')
    ax.set_ylabel('pad')
    ax.tick_params(which="minor", bottom=False, left=False)
    #Check where coordinates are:
    # plt.scatter(0, 0)
    # plt.scatter(0, 1)
    # plt.show()
    # pprint(adc_counts)

    #Clusters:
    #Clusters using equation 6.7:
    #Layer number:
    layer = np.mod(np.floor(hcid/2),6)
    sigma_list = [0.581,0.566,0.556,0.550,0.545,0.524]
    sigma = sigma_list[int(layer)]
    def calcY1(pads, centre_pad_index):
        
        
        # i-1 = otherPadIndeces[0], i+1 = otherPadIndeces[1]
        if len(pad_data)> 2:
            if centre_pad_index>=1 and centre_pad_index!=(len(pad_data)-1):
                other_pad_indeces = (centre_pad_index-1,centre_pad_index+1)
                other_two_product = pads[other_pad_indeces[0]]*pads[other_pad_indeces[1]]
                other_two_quotient = pads[other_pad_indeces[1]]/pads[other_pad_indeces[0]]
                three_pad_eq = 0.5*(np.log(other_two_quotient)/np.log(pads[centre_pad_index]**2/other_two_product))
                return three_pad_eq
            elif centre_pad_index==0:
                two_pad_eq0 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index+1])-(0.5)
                two_pad_eq0 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index+1])-(0.5)
                return two_pad_eq0
            elif centre_pad_index == (len(pad_data)-1):
                two_pad_eq2 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index-1])+(0.5)
                return two_pad_eq2
        elif len(pad_data)==2:
            if centre_pad_index==0:
                two_pad_eq0 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index+1])-(0.5)
                return two_pad_eq0
            elif centre_pad_index==1:
                two_pad_eq2 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index-1])+(0.5)
                return two_pad_eq2
    # Averages of ADC data in each time bin
    y1 = []
    center_pads = []
    for i in range(0, 30):
        current_pads = adc_counts[:, i]
        center_pad_index = int(np.argmax(current_pads))
        center_pads.append(center_pad_index) 
        y1.append(calcY1(current_pads, center_pad_index))

    # pprint(y1)

    # pprint(adc_counts)
    # pprint(heatmap_data.iloc[0:len(pad_data), 0])
    # pprint(np.argmax(heatmap_data.iloc[0:len(pad_data), 0]))
    # pprint(heatmap_data)
    # pprint(y)
    # pprint(len(timebin_data))
    # pprint(adc_counts)
    # pprint(len(y))
    #print(np.array(center_pads)+np.array(y))
    y_values_1 = np.abs(np.array(center_pads)+np.array(y1))
    # pprint(y_values_1)
    plt.scatter(timebin_data, y_values_1, label = "Clusters using eq. 6.7", color= 'red')
    # print(pad_data[2])
    # plt.show()
    

    #Clusters using equation 6.10:
    def calcY2(pads, centre_pad_index,sigma):
        # i-1 = otherPadIndeces[0], i+1 = otherPadIndeces[1]
        if len(pad_data)> 2:
            if centre_pad_index>=1 and centre_pad_index!=(len(pad_data)-1):
                other_pad_indeces = (centre_pad_index-1,centre_pad_index+1)
                w1 = pads[other_pad_indeces[0]]
                w2 = pads[other_pad_indeces[1]]
                three_pad_eq = (1/(w1+w2))*((w1*((-0.5)+(sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index-1]))) + (w2*((0.5)+(sigma**2)*np.log(pads[centre_pad_index+1]/pads[centre_pad_index]))))
                return three_pad_eq
            elif centre_pad_index==0:
                two_pad_eq0 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index+1])-(0.5)
                return two_pad_eq0
            elif centre_pad_index==(len(pad_data)-1):
                two_pad_eq2 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index-1])+(0.5)
                return two_pad_eq2
        elif len(pad_data)==2:
            if centre_pad_index==0:
                two_pad_eq0 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index+1])-(0.5)
                return two_pad_eq0
            elif centre_pad_index==1:
                two_pad_eq2 = (sigma**2)*np.log(pads[centre_pad_index]/pads[centre_pad_index-1])+(0.5)
                return two_pad_eq2
        
    # Averages of ADC data in each time bin
    y2 = []
    center_pads_new = []
    for i in range(0, 30):
        current_pads = adc_counts[:, i]
        center_pad_index = int(np.argmax(current_pads))
        center_pads_new.append(center_pad_index) 
        y2.append(calcY2(current_pads, center_pad_index, sigma))
    y_values_2 = np.abs(np.array(center_pads_new)+np.array(y2))
    plt.scatter(timebin_data, y_values_2, label = "Clusters using eq. 6.10", color= 'limegreen')
    # pprint(y_values_2)
    plt.legend()
    # plt.show()

    #Linear fits to cluster data:
    # Linear function
    def linear(x,m,c):
        return m*x + c
    #Initial parameters:
    m0= 0
    c0= 50
    #Parameter and covariance matrix estimates:
    pars1, cov1 = curve_fit(linear,np.array(timebin_data[5:21]),y_values_1[5:21], p0 = [m0,c0])
    pars2,cov2 = curve_fit(linear, np.array(timebin_data[5:21]),y_values_2[5:21],p0=[m0,c0])

    plt.plot(timebin_data[5:21],linear(np.array(timebin_data[5:21]), *pars1), color='red', label = 'Linear fit for eq. 6.7')
    plt.plot(timebin_data[5:21],linear(np.array(timebin_data[5:21]), *pars2), color='limegreen', label = 'Linear fit for eq. 6.10')
    # plt.show()


    #Tracklets:
    
    tracklet_ypos0 =tracklets[tracklet_index[0],10] - pad_data[0] 
    #-0.5 #shifts data to fit on heatmap
    point1 = [0,tracklet_ypos0]
    plt.scatter(0,tracklet_ypos0)
    tracklet_slopes = 30*tracklets[tracklet_index[0],5] #multiplied by 30 to get accross all pads
    tracklet_ypos1 = tracklet_ypos0 - tracklet_slopes
    point2 = [29, tracklet_ypos1]
    plt.scatter(29,tracklet_ypos1)
    plt.plot([point1[0],point2[0]],[point1[1],point2[1]], label = 'Tracklet', color = 'orange')
    plt.legend()
    plt.show()

    #Residuals
    def residuals(yrec,yfit):
        return yrec-yfit

    res1 = residuals(y_values_1[5:21],linear(np.array(timebin_data[5:21]), *pars1))
    res2 = residuals(y_values_2[5:21],linear(np.array(timebin_data[5:21]), *pars2))
    # pprint(res2)

    # pprint(layer)
    if layer == 0.0:
        L0.append(res2)
    elif layer == 1.0:
        L1.append(res2)
    elif layer == 2.0:
        L2.append(res2)
    elif layer == 3.0:
        L3.append(res2)
    elif layer == 4.0:
        L4.append(res2)
    elif layer == 5.0:
        L5.append(res2)

    # pprint(res2)





    # iterations += 1
    # if iterations == 3:
    #     break

#Saving residuals to .csv files
np.savetxt('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L0_res.csv',L0,delimiter = ',')
np.savetxt('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L1_res.csv',L1,delimiter = ',')
np.savetxt('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L2_res.csv',L2,delimiter = ',')
np.savetxt('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L3_res.csv',L3,delimiter = ',')
np.savetxt('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L4_res.csv',L4,delimiter = ',')
np.savetxt('/Users/Tenille/Desktop/PHY3004W Project/Getting somewhere things/L5_res.csv',L5,delimiter = ',')


    





    #     for adc_sum in adc_data:
    #         index  = np.argpartition(adc_data[adc_sum,0],-3)
        # print(rows_used)




    


    
    
    

    #         pprint[sorted_digits]
    #         break



#hcid = sorted_digits.iloc[0:,i]
# heatmap_data = sorted.filter(regex = '^adc_', axis = 1).head(2)
# pad_data = digits_path.iloc[0:2,2]
# hcid = 24
# timebin_data = list(range(0, 30))