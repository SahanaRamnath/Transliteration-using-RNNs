'''
------------------------------------
PLOT LOSS GRAPH
------------------------------------
'''

import pickle
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--save_dir",help="directory to load data from",required=True)
args=parser.parse_args()

dir_name=args.save_dir

with open(dir_name+'train_loss_list.pkl') as f :
	[epoch_list_train,train_loss_list]=pickle.load(f)

with open(dir_name+'val_loss_list.pkl') as f :
	[epoch_list_val,val_loss_list]=pickle.load(f)

fig1=plt.figure().add_subplot(111)
fig1.plot(epoch_list_train,train_loss_list)
fig1.plot(epoch_list_val,val_loss_list)
plt.legend(('train loss','val loss'),loc='best')
plt.title('Train Loss versus Epoch for different learning rates')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('plots/loss_plot_'+dir_name+'.png')