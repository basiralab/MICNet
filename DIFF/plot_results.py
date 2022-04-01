# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:08:22 2020

@author: Mohammed Amine
"""
import pickle
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

#tracked_dicts_RH_ASDNC_extracted
with open('tracked_dicts_LH_ASDNC_extracted', 'rb') as f:
    tracked_dicts = pickle.load(f)


# calculate accuracy and generate plots
accs_matrix = np.zeros((200,4))

for cv in range(len(tracked_dicts)):
    for epoch in range(len(tracked_dicts[cv])):
        accs_matrix[epoch,cv] = np.mean(tracked_dicts[cv][epoch]['preds'] == tracked_dicts[cv][epoch]['labels'])
        
accs_vector = np.mean(accs_matrix,axis=1)
accs_scalar = np.mean(accs_vector) 

accs_list = np.ones(200) * accs_scalar

epochs = [(i+1) for i in range(200)]

plt.plot(epochs, accs_vector, label = "line 1")
plt.plot(epochs, accs_list, label = "line 1")

plt.title('Accuracy results')

print("finished")


'''
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars_acc = [0.63,0.57,0.56, 0.66, 0.55]
 
# Choose the height of the cyan bars
bars_sens = [0.5, 0.43, 0.35, 0.49, 0.59]
 
# Choose the height of the cyan bars
bars_spec = [0.75, 0.73, 0.82, 0.81, 0.5]
 
# The x position of bars
r1 = np.arange(len(bars_acc)) * 1.5
r2 = [x + barWidth for x in r1]
r3 = [x + 2*barWidth for x in r1]

plt.figure(figsize=(15,15))

# Create blue bars
bar1 = plt.bar(r1, bars_acc, width = barWidth,  edgecolor = 'black', capsize=7, label='accuracy')
 
# Create cyan bars
bar2 = plt.bar(r2, bars_sens, width = barWidth, edgecolor = 'black', capsize=7, label='sensitivity')

# Create cyan bars
bar3 = plt.bar(r3, bars_spec, width = barWidth, edgecolor = 'black', capsize=7, label='specificity')

# general layout
plt.xticks([r*1.5 + barWidth for r in range(len(bars_acc))], ['MGI', 'Fusion', 'Fusion + Identity matrix', 'Dissimilarity',  'Dissimilarity + Identity matrix'])
#plt.ylabel('height')
plt.title('SVM results')
plt.legend()

for rect in bar1:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (-10, 5), textcoords="offset points")

for rect in bar2:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (-10, 5), textcoords="offset points")

for rect in bar3:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (-10, 5), textcoords="offset points")

plt.ylim(0.0, 1.05)  


# Show graphic
plt.show()

'''




'''
# width of the bars
barWidth = 0.35
# RH ASDNC
# Choose the height of the blue bars
bars_avg_pos = [12.48, 12.53, 12.56, 12.32, 12.49, 12.33]
 
# Choose the height of the cyan bars
bars_avg_neg = [12.49, 12.51, 12.55, 12.39, 12.50, 12.17]

bars_std_pos = [0.04, 0.10, 0.17, 0.08, 0.18, 0.04]
bars_std_neg = [0.08, 0.15, 0.15, 0.13, 0.21, 0.03]


# The x position of bars
r1 = np.arange(len(bars_avg_pos)) * 1.5
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(15,15))

# Create blue bars
bar1 = plt.bar(r1, bars_avg_pos, yerr=bars_std_pos , width = barWidth,  edgecolor = 'black', capsize=7, label='positive')
 
# Create cyan bars
bar2 = plt.bar(r2, bars_avg_neg, yerr=bars_std_neg , width = barWidth, edgecolor = 'black', capsize=7, label='negative')

# general layout
plt.xticks([r*1.5 + 0.5*barWidth for r in range(len(bars_avg_pos))], ['MGI','MGI + Identity matrix' ,'Fusion', 'Fusion + Identity matrix', 'Dissimilarity',  'Dissimilarity + Identity matrix'])
#plt.ylabel('height')
plt.title('within class representativeness ASD-NC (RH)')
plt.legend()

for rect in bar1:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (-35, 5), textcoords="offset points")

for rect in bar2:
    height = rect.get_height()
    plt.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext = (5, 5), textcoords="offset points")

plt.ylim(12, 12.8)  


# Show graphic
plt.show()

'''

