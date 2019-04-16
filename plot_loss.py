'''
------------------------------------
PLOT LOSS GRAPH
------------------------------------
'''

import pickle
import matplotlib.pyplot as plt
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument("--save_dir",help="directory to load data from",required=True)
args=parser.parse_args()

dir_name=args.save_dir

with open(os.path.join(dir_name,'train_loss_list.pkl')) as f :
	[epoch_list_train,train_loss_list]=pickle.load(f)

with open(os.path.join(dir_name,'val_loss_list.pkl')) as f :
	[epoch_list_val,val_loss_list]=pickle.load(f)

fig1=plt.figure().add_subplot(111)
fig1.plot(epoch_list_train,train_loss_list)
fig1.plot(epoch_list_val,val_loss_list)
plt.legend(('train loss','val loss'),loc='best')
plt.title('Loss versus Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('plots/loss_plot_'+dir_name+'.png')

with open(os.path.join(dir_name,'val_acc_list.pkl')) as f :
	[epoch_list_val,val_acc_list]=pickle.load(f)

fig2=plt.figure().add_subplot(111)
fig2.plot(epoch_list_train,val_acc_list)
fig2.plot(epoch_list_val,val_loss_list)
plt.legend(('val accuracy','val loss'),loc='best')
plt.title('Validation loss and accuracy versus Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.savefig('plots/val_plot_'+dir_name+'.png')