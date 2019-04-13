'''
--------------------
PA3 CODE FOR TESTING
--------------------
'''

import tensorflow as tf
import numpy as np
import pandas as pd

import os
import argparse
import pickle
import codecs
import train




parser=argparse.ArgumentParser()

parser.add_argument("--lr",help="learning rate",default=0.001)
parser.add_argument("--batch_size",help="batch size",default="60")
parser.add_argument("--init",
	help="weight initialization method : xavier,he,normal,uniform",default="xavier")
parser.add_argument("--save_dir",
	help="directory to save weights",required=True)
parser.add_argument("--epochs",help="number of epochs to train",default=30)


parser.add_argument("--train",help="path to train file",
	default=os.path.join('dl2019pa3','train.csv'))
parser.add_argument("--val",help="path to validation file",
	default=os.path.join('dl2019pa3','valid.csv'))
parser.add_argument("--test",help="path to test file",
	default=os.path.join('dl2019pa3','partial_test_400.csv'))
parser.add_argument("--vocab",help="folder containing eng and hindi vocab",
	default="vocab")

parser.add_argument("--keep_prob",help="keep_prob in tf dropout",default=0.9)
parser.add_argument("--bidir",help="1 for bidirectional RNN, 0 for not",default=1)
parser.add_argument("--decode_method",help="0 for greedy, 1 for beam",default=0)
parser.add_argument("--inembed",help="inembed for encoder",default=256)
parser.add_argument("--encsize",help="encsize for encoder",default=512)
parser.add_argument("--decsize",help="decsize for encoder",default=512)
parser.add_argument("--outembed",help="outembed for encoder",default=256)
parser.add_argument("--stack_decoder",help="1 if stacked decoder is to be used",default=1)

args=parser.parse_args()

if not os.path.exists(args.save_dir):
    raise ValueError("Need existing model to infer!")

train=pd.read_csv(args.train)
print('Train data size : ',train.shape)

val=pd.read_csv(args.val)
print('Val data size : ',val.shape)

test=pd.read_csv(args.test)
print('Test data size : ',test.shape)


# hyperparameters from argparse
batch_size=int(args.batch_size)

prev_accuracy=float(-1) # initial val error rate for early stopping

# lists to store losses
train_loss_list=[]
val_loss_list=[]
epoch_list=[]

patience=0
epoch=0

# loading vocab
with codecs.open(os.path.join(args.vocab,'eng.txt'),'r') as f : 
	eng_vocab=[line.rstrip('\n') for line in f]
	len_eng_vocab=len(eng_vocab)
	print('len of english vocab : '+str(len_eng_vocab))

with codecs.open(os.path.join(args.vocab,'hindi.txt'),'r') as f : 
	hindi_vocab=[line.rstrip('\n') for line in f]
	len_hindi_vocab=len(hindi_vocab)
	print('len of hindi vocab : '+str(len_hindi_vocab))

# creating char to index dicts
eng_to_ind={}
ind_to_eng={}
ind=0
for char in eng_vocab : 
	eng_to_ind[char]=ind
	ind_to_eng[ind]=char
	ind=ind+1

hindi_to_ind={}
ind_to_hindi={}
ind=0
for char in hindi_vocab : 
	hindi_to_ind[char]=ind
	ind_to_hindi[ind]=char
	ind=ind+1
print('Made vocab dictionaries')

# creating separate lists
train_ids=train['id'].tolist()
train_eng=train['ENG'].tolist()
train_hindi=train['HIN'].tolist()

max_len_eng=0
max_len_hindi=0 # will be used a max decoder length
for i in range(train.shape[0]) : 
	word_eng=train_eng[i].split(' ')
	if len(word_eng)>max_len_eng : 
		max_len_eng=len(word_eng)
	word_hindi=train_hindi[i].split(' ')
	if len(word_hindi)>max_len_hindi : 
		max_len_hindi=len(word_hindi)
print('train max len eng : ',max_len_eng)
print('train max len hindi : ',max_len_hindi)



val_ids=val['id'].tolist()
val_eng=val['ENG'].tolist()
val_hindi=val['HIN'].tolist()

max_len_eng=0
for i in range(val.shape[0]) : 
	word_eng=val_eng[i].split(' ')
	if len(word_eng)>max_len_eng : 
		max_len_eng=len(word_eng)
print('val max len eng : ',max_len_eng)

# converting to index matrices
val_eng_matrix=np.zeros((val.shape[0],max_len_eng))
val_eng_seqlen=np.zeros(val.shape[0])
val_hindi_matrix=np.zeros((val.shape[0],max_len_hindi))
val_hindi_seqlen=np.zeros(val.shape[0])
val_eng_attn_mask=np.zeros((val.shape[0],max_len_eng))

for i in range(val.shape[0]) : 
	word_eng=val_eng[i].split(' ')
	tmp_char=[]
	for char in word_eng : 
		tmp_char.append(eng_to_ind[char])
	if len(tmp_char)>max_len_eng : 
		val_eng_seqlen[i]=max_len_eng
		val_eng_attn_mask[i][:]=1
		tmp_char=tmp_char[:max_len_eng]
	if len(tmp_char)<max_len_eng : 
		val_eng_seqlen[i]=len(tmp_char)
		val_eng_attn_mask[i][:len(tmp_char)]=1
		tmp_char+=[1]*(max_len_eng-len(tmp_char))
	val_eng_matrix[i]=tmp_char

	word_hindi=val_hindi[i].split(' ')
	tmp_char=[]
	for char in word_hindi : 
		tmp_char.append(hindi_to_ind[char])
	if len(tmp_char)>max_len_hindi : 
		val_hindi_seqlen[i]=max_len_hindi
		tmp_char=tmp_char[:max_len_hindi]
	if len(tmp_char)<max_len_hindi : 
		val_hindi_seqlen[i]=len(tmp_char)
		tmp_char+=[1]*(max_len_hindi-len(tmp_char))
	val_hindi_matrix[i]=tmp_char
print('Val converted characters to indices')


test_ids=test['id'].tolist()
test_eng=test['ENG'].tolist()

max_len_eng=0
for i in range(test.shape[0]) : 
	word_eng=test_eng[i].split(' ')
	if len(word_eng)>max_len_eng : 
		max_len_eng=len(word_eng)
print('test max len eng : ',max_len_eng)

# converting to index matrices
test_eng_matrix=np.zeros((test.shape[0],max_len_eng))
test_eng_seqlen=np.zeros(test.shape[0])
test_eng_attn_mask=np.zeros((test.shape[0],max_len_eng))

for i in range(test.shape[0]) : 
	word_eng=test_eng[i].split(' ')
	tmp_char=[]
	for char in word_eng : 
		if char in eng_to_ind : 
			tmp_char.append(eng_to_ind[char])
		else : 
			print('New character in test : '+char)
			tmp_char.append(eng_to_ind['<unk>'])
	if len(tmp_char)>max_len_eng : 
		test_eng_seqlen[i]=max_len_eng
		test_eng_attn_mask[i][:]=1
		tmp_char=tmp_char[:max_len_eng]
	if len(tmp_char)<max_len_eng : 
		test_eng_seqlen[i]=len(tmp_char)
		test_eng_attn_mask[i][:len(tmp_char)]=1
		tmp_char+=[1]*(max_len_eng-len(tmp_char))
	test_eng_matrix[i]=tmp_char
print('Test converted characters to indices')



with tf.Graph().as_default() : 
	
	# setting seeds
	tf.set_random_seed(12345)
	np.random.seed(1234)

	# session
	sess=tf.Session()
	train_model=train.rnn_model(args,
		len_eng_vocab,len_hindi_vocab,
		max_decoding_steps=max_len_hindi,mode='train')
	print('Train model created!')

	latest_ckpt=tf.train.latest_checkpoint(args.save_dir)

	train_model.saver.restore(sess,latest_ckpt)
	# global_step=train_model.global_step.eval(session=sess)
	global_step=sess.run(train_model.global_step)
	print('Model loaded from saved checkpoint at global step : '+str(global_step))



	val_predicted_hindi_chars=[]
	val_predicted_ids=[]
	if val.shape[0]%batch_size==0 : 
		limit=int(val.shape[0]/batch_size)
	else : 
		limit=int(val.shape[0]/batch_size)+1
	for i in range(limit) : # each epoch
		if i%20==0 : 
			print('In validation loop : '+str(i))
		try : 
			val_ids_temp=val_ids[i*batch_size:(i+1)*batch_size]
			val_eng_temp=val_eng_matrix[i*batch_size:(i+1)*batch_size,:].astype(np.int32)
			val_eng_seqlen_temp=val_eng_seqlen[i*batch_size:(i+1)*batch_size].astype(np.int32)
			val_hindi_temp=val_hindi_matrix[i*batch_size:(i+1)*batch_size,:].astype(np.int32)
			val_hindi_seqlen_temp=val_hindi_seqlen[i*batch_size:(i+1)*batch_size].astype(np.int32)
			val_eng_attn_mask_temp=val_eng_attn_mask[i*batch_size:(i+1)*batch_size,:].astype(np.int32)
		except : 
			val_ids_temp=val_ids[i*batch_size:]
			val_eng_temp=val_eng_matrix[i*batch_size:,:].astype(np.int32)
			val_eng_seqlen_temp=val_eng_seqlen[i*batch_size:].astype(np.int32)
			val_hindi_temp=val_hindi_matrix[i*batch_size:,:].astype(np.int32)
			val_hindi_seqlen_temp=val_hindi_seqlen[i*batch_size:].astype(np.int32)
			val_eng_attn_mask_temp=val_eng_attn_mask[i*batch_size:,:].astype(np.int32)

		[val_ce_loss,predicted_hindi_chars]=train_model.val(sess,val_eng_temp,
			val_eng_seqlen_temp,val_hindi_temp,val_hindi_seqlen_temp,val_eng_attn_mask_temp)


		for j in range(predicted_hindi_chars.shape[0]) : 
			current_pred=predicted_hindi_chars[j,:]
			current_pred_char=[ind_to_hindi[x] for x in current_pred]

			if '<pad>' in current_pred_char : 
				end_index=current_pred_char.index('<pad>')
				current_pred_char=current_pred_char[0:end_index]

			current_pred_char=' '.join(current_pred_char)

			val_predicted_hindi_chars.append(current_pred_char)
			# print val_predicted_hindi_chars					
		val_predicted_ids.extend(val_ids_temp)

	num_correct=0
	for i in range(val.shape[0]) : 
		current_pred=val_predicted_hindi_chars[i]
		if val_hindi[i]==current_pred : 
			num_correct=num_correct+1
	accuracy=float(num_correct)/float(val.shape[0])
	print('Accuracy of loaded model for val is '+str(accuracy))


test_predicted_hindi_chars=[]
test_predicted_ids=[]
if test.shape[0]%batch_size==0 : 
	limit=int(test.shape[0]/batch_size)
else : 
	limit=int(test.shape[0]/batch_size)+1
for i in range(limit) : # each epoch
	if i%20==0 : 
		print('In test loop : '+str(i))
	try : 
		test_ids_temp=test_ids[i*batch_size:(i+1)*batch_size]
		test_eng_temp=test_eng_matrix[i*batch_size:(i+1)*batch_size,:].astype(np.int32)
		test_eng_seqlen_temp=test_eng_seqlen[i*batch_size:(i+1)*batch_size].astype(np.int32)
		test_eng_attn_mask_temp=test_eng_attn_mask[i*batch_size:(i+1)*batch_size,:].astype(np.int32)
	except : 
		test_ids_temp=test_ids[i*batch_size:]
		test_eng_temp=test_eng_matrix[i*batch_size:,:].astype(np.int32)
		test_eng_seqlen_temp=test_eng_seqlen[i*batch_size:].astype(np.int32)
		test_eng_attn_mask_temp=test_eng_attn_mask[i*batch_size:,:].astype(np.int32)

	predicted_hindi_chars=train_model.test(sess,test_eng_temp,test_eng_seqlen_temp,
		test_eng_attn_mask_temp)


	for j in range(predicted_hindi_chars.shape[0]) : 
		current_pred=predicted_hindi_chars[j,:]
		current_pred_char=[ind_to_hindi[x] for x in current_pred]

		if '<pad>' in current_pred_char : 
			end_index=current_pred_char.index('<pad>')
			current_pred_char=current_pred_char[0:end_index]

		current_pred_char=' '.join(current_pred_char)

		test_predicted_hindi_chars.append(current_pred_char)
		# print test_predicted_hindi_chars					
	test_predicted_ids.extend(test_ids_temp)

with codecs.open(os.path.join(args.save_dir,'final_'+args.save_dir+'_'+str(global_step)+'.csv'),'w') as f : 
	f.write('id,HIN')
	f.write('\n')
	for i in range(test.shape[0]) :  
		f.write(str(test_predicted_ids[i]))
		f.write(',')

		current_pred=test_predicted_hindi_chars[i]
		

		f.write(current_pred)
		f.write('\n')

print('Saved test prediction file!')