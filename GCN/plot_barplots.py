# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:20:55 2020

@author: Mohammed Amine
"""


import pickle
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#arg_dataset ="RH" 
#arg_dataset ="RH_ASDNC_extracted" 
#arg_dataset ="RH_GSP"
#arg_dataset ="LH" 
#arg_dataset ="LH_ASDNC_extracted" 
arg_dataset ="LH_GSP"

 
fusion_methods = ["SNF_GCN","average_GCN","linear_GCN","MGI_GCN", "netNorm_GCN"]

def get_last_epoch(arg_dataset):
    fusion_methods = ["SNF_GCN","average_GCN","linear_GCN","MGI_GCN", "netNorm_GCN"]
    tracked_dicts = {}
    for i in range(len(fusion_methods)):
        arg_method = fusion_methods[i]
        with open('tracked_dicts_'+arg_dataset+"_"+arg_method, 'rb') as f:
            tracked_dict = pickle.load(f)
        list_cv = []
        for cv in range(len(tracked_dict)):
            tracked_cv={}
            tracked_cv['labels'] = tracked_dict[cv][len(tracked_dict[cv])-1]['labels']
            tracked_cv['preds'] = tracked_dict[cv][len(tracked_dict[cv])-1]['preds']
            list_cv.append(tracked_cv)
        tracked_dicts[arg_method]=list_cv
    return tracked_dicts

def get_metrics(tracked_dicts):
    fusion_methods = ["SNF_GCN","average_GCN","linear_GCN","MGI_GCN", "netNorm_GCN"]
    metrics_dict = {}
    for i in range(len(fusion_methods)):
        metric_method = {}
        acc = []
        sens = []
        spec = []
        for cv in range(len(tracked_dicts[fusion_methods[i]])):
            targets = tracked_dicts[fusion_methods[i]][cv]['labels']
            preds = tracked_dicts[fusion_methods[i]][cv]['preds']
            cm1 = confusion_matrix(targets, preds)
            total1=sum(sum(cm1))
            acc.append(round((cm1[0,0]+cm1[1,1])/total1,2))
            sens.append(cm1[0,0]/(cm1[0,0]+cm1[0,1]))
            spec.append(cm1[1,1]/(cm1[1,0]+cm1[1,1]))
        metrics_acc = {}
        metrics_acc['mean'] = np.mean(acc)
        metrics_acc['std'] = np.std(acc)
        metrics_sens = {}
        metrics_sens['mean'] = np.mean(sens)
        metrics_sens['std'] = np.std(sens)
        metrics_spec = {}
        metrics_spec['mean'] = np.mean(spec)
        metrics_spec['std'] = np.std(spec)
        # accuracy
        metric_method['acc'] = metrics_acc
        # sensitivity 
        metric_method['sens'] = metrics_sens
        # specificity
        metric_method['spec'] = metrics_spec
        
        metrics_dict[fusion_methods[i]] = metric_method
    return metrics_dict
    

tracked_dicts = get_last_epoch(arg_dataset)
metrics_dict = get_metrics(tracked_dicts)

accs_fusion = []
for i in range(len(fusion_methods)):
    a = metrics_dict[fusion_methods[i]]['acc']['mean']
    accs_fusion.append(a)
accs_fusion, fusion_methods = zip(*sorted(zip(accs_fusion, fusion_methods)))



print("finshed")


barWidth = 0.3
 
# Choose the height of the blue bars
bars_acc_mean = [round(metrics_dict[fusion_methods[i]]['acc']['mean'],2) for i in range(len(fusion_methods))]
bars_acc_std = [round(metrics_dict[fusion_methods[i]]['acc']['std'],2) for i in range(len(fusion_methods))]
# Choose the height of the cyan bars
bars_sens_mean = [round(metrics_dict[fusion_methods[i]]['sens']['mean'],2) for i in range(len(fusion_methods))]
bars_sens_std = [round(metrics_dict[fusion_methods[i]]['sens']['std'],2) for i in range(len(fusion_methods))]
# Choose the height of the cyan bars
bars_spec_mean = [round(metrics_dict[fusion_methods[i]]['spec']['mean'],2) for i in range(len(fusion_methods))]
bars_spec_std = [round(metrics_dict[fusion_methods[i]]['spec']['std'],2) for i in range(len(fusion_methods))]
# The x position of bars
r1 = np.arange(len(bars_acc_mean)) * 1.5
r2 = [x + barWidth for x in r1]
r3 = [x + 2*barWidth for x in r1]

plt.figure(figsize=(15,15))
#ax = plt.axes([0.0, 0.0, 1.0, 1.0])

# Create blue bars
bar1 = plt.bar(r1, bars_acc_mean, yerr=bars_acc_std, width = barWidth,  edgecolor = 'black', capsize=7, label='accuracy')
 
# Create cyan bars
bar2 = plt.bar(r2, bars_sens_mean, yerr=bars_sens_std, width = barWidth, edgecolor = 'black', capsize=7, label='sensitivity')

# Create cyan bars
bar3 = plt.bar(r3, bars_spec_mean, yerr=bars_spec_std, width = barWidth, edgecolor = 'black', capsize=7, label='specificity')

# general layout
plt.xticks([r*1.5 + barWidth for r in range(len(bars_acc_mean))], fusion_methods, fontsize=18)
#plt.ylabel('height')
plt.title('LH_Data_Classification Results', fontsize=20)
plt.legend(fontsize=20)
for rect in bar1:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (-10, 5), textcoords="offset points",fontsize=18)

for rect in bar2:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (-10, 5), textcoords="offset points",fontsize=18)

for rect in bar3:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (-10, 5), textcoords="offset points",fontsize=18)

plt.ylim(0, 1.1)  


# Show graphic
plt.show()



    
