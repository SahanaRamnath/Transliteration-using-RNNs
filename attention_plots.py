'''
--------------
PLOT ATTENTION
--------------
'''

import numpy as np

import matplotlib.pyplot as plt
import sys
import os

import tensorflow as tf
import codecs
import pandas as pd
import argparse
from matplotlib import rcParams
from matplotlib import font_manager

class rnn_model() : 

	def __init__(self,args,
		len_eng_vocab,len_hindi_vocab,
		max_decoding_steps,mode) : 


		self.mode=mode
		self.args=args
		self.len_eng_vocab=len_eng_vocab
		self.len_hindi_vocab=len_hindi_vocab

		self.max_decoding_steps=max_decoding_steps
		self.inembed=int(self.args.inembed)
		self.encsize=int(self.args.encsize)
		self.outembed=int(self.args.outembed)
		self.decsize=int(self.args.decsize)


		self.create_emb_matrices()

		self.global_step=tf.Variable(0,trainable=False)
		self.create_placeholders()
		
		if self.args.init=='xavier' : 	
			initializer=tf.contrib.layers.xavier_initializer(uniform=True)
		elif self.args.init=='he' : 
			initializer=tf.keras.initializers.he_normal()
		elif self.args.init=='uniform' : 
			initializer=tf.initializers.random_uniform(-0.01,0.01)
		elif self.args.init=='normal' : 
			initializer=tf.initializers.random_normal(0,0.01)
		tf.get_variable_scope().set_initializer(initializer)
	
		self.create_model()
		self.saver=tf.train.Saver(tf.global_variables(),max_to_keep=5) # saver

	def create_placeholders(self) : 

		self.encoder_input_ind=tf.placeholder(shape=[None,None],dtype=tf.int32)
		self.encoder_seqlen=tf.placeholder(shape=[None],dtype=tf.int32)
		self.decoder_output_ind=tf.placeholder(shape=[None,self.max_decoding_steps],
			dtype=tf.int32)
		self.decoder_seqlen=tf.placeholder(shape=[None],dtype=tf.int32)
		self.encoder_attn_mask=tf.placeholder(shape=[None,None],dtype=tf.int32)
		self.keep_prob=tf.placeholder(tf.float32)
		self.is_train=tf.placeholder(tf.bool)

	def train(self,sess,encoder_input_ind,encoder_seqlen,decoder_output_ind,
		decoder_seqlen,encoder_attn_mask,keep_prob=1.0) : 

		keep_prob=float(self.args.keep_prob)
		[ce_loss,global_step,opt,predicted_hindi_chars,dc_ip_1,dc_ip_2]=sess.run(
			[self.ce_loss,self.global_step,self.optimizer,self.predicted_hindi_chars,self.decoder_input_1,self.decoder_input_2],
			feed_dict={self.encoder_input_ind : encoder_input_ind, self.encoder_seqlen : encoder_seqlen, self.decoder_output_ind : decoder_output_ind,self.keep_prob : keep_prob,self.is_train : 1,self.decoder_seqlen : decoder_seqlen,self.encoder_attn_mask : encoder_attn_mask})
		#print '\n\n\n'
		#print dc_ip_1
		#print dc_ip_2

		return [ce_loss,global_step,predicted_hindi_chars,dc_ip_1,dc_ip_2]

	def val(self,sess,encoder_input_ind,encoder_seqlen,decoder_output_ind,
		decoder_seqlen,encoder_attn_mask,keep_prob=1.0,is_train=0) : 

		[ce_loss,predicted_hindi_chars,dc_ip_1,dc_ip_2]=sess.run(
			[self.ce_loss,self.predicted_hindi_chars,self.decoder_input_1,self.decoder_input_2],
			feed_dict={self.encoder_input_ind : encoder_input_ind, self.encoder_seqlen : encoder_seqlen, self.decoder_output_ind : decoder_output_ind,self.keep_prob : keep_prob,self.is_train : 0,self.decoder_seqlen : decoder_seqlen,self.encoder_attn_mask : encoder_attn_mask})
		# print '\n\n\n'
		# print dc_ip_1
		# print dc_ip_2
		return [ce_loss,predicted_hindi_chars]

	def test(self,sess,encoder_input_ind,encoder_seqlen,encoder_attn_mask,
		keep_prob=1.0,is_train=0) : 

		# needed for tf cond, but not used
		decoder_output_ind=np.zeros((encoder_input_ind.shape[0],self.max_decoding_steps))

		predicted_hindi_chars=sess.run(self.predicted_hindi_chars,
			feed_dict={self.encoder_input_ind : encoder_input_ind, self.encoder_seqlen : encoder_seqlen, self.decoder_output_ind : decoder_output_ind,self.keep_prob : keep_prob,self.is_train : 0,self.encoder_attn_mask : encoder_attn_mask})
		return predicted_hindi_chars

	#####################################################################################
	def get_attention(self,sess,encoder_input_ind,encoder_seqlen,encoder_attn_mask,
		keep_prob=1.0,is_train=0) : 

		# needed for tf cond, but not used
		decoder_output_ind=np.zeros((encoder_input_ind.shape[0],self.max_decoding_steps))

		[predicted_hindi_chars,alphas,input_eng_chars]=sess.run(
			[self.predicted_hindi_chars,self.alphas_list,self.encoder_input_ind],
			feed_dict={self.encoder_input_ind : encoder_input_ind, self.encoder_seqlen : encoder_seqlen, self.decoder_output_ind : decoder_output_ind,self.keep_prob : keep_prob,self.is_train : 0,self.encoder_attn_mask : encoder_attn_mask})
		return [predicted_hindi_chars,alphas,input_eng_chars]
	###############################################################################
	def create_emb_matrices(self) : 

		with tf.variable_scope('rnn_model',reuse=tf.AUTO_REUSE) as scope : 
			self.encoder_emb_matrix=tf.get_variable(name='encoder_emb_matrix',
				shape=[self.len_eng_vocab,self.inembed])
			self.decoder_emb_matrix=tf.get_variable(name='decoder_emb_matrix',
				shape=[self.len_hindi_vocab,self.outembed])

	def create_model(self) : 

		with tf.variable_scope('rnn_model',reuse=tf.AUTO_REUSE) as scope1 : 
			# encoder
			encoder_input=tf.nn.embedding_lookup(
				self.encoder_emb_matrix,self.encoder_input_ind)
			with tf.variable_scope('encoder_lstm') as scope : 
				fw_cell=tf.nn.rnn_cell.DropoutWrapper(
					tf.contrib.rnn.BasicLSTMCell(self.encsize,activation=tf.nn.tanh),
					input_keep_prob=self.keep_prob,output_keep_prob=self.keep_prob,
					state_keep_prob=self.keep_prob)
				bw_cell=tf.nn.rnn_cell.DropoutWrapper(
					tf.contrib.rnn.BasicLSTMCell(self.encsize,activation=tf.nn.tanh),
					input_keep_prob=self.keep_prob,output_keep_prob=self.keep_prob,
					state_keep_prob=self.keep_prob)
				encoder_output,encoder_state=tf.nn.bidirectional_dynamic_rnn(
					time_major=False, dtype=tf.float32,scope=scope,
					cell_fw=fw_cell,cell_bw=bw_cell,
					inputs=encoder_input,
					sequence_length=self.encoder_seqlen)

				self.encoder_state=encoder_state#tf.nn.rnn_cell.LSTMStateTuple(encoder_state[0].c,
					#encoder_state[1].c)
				self.encoder_output=tf.concat(encoder_output,-1)
				print('encoder output : ',self.encoder_output.get_shape())

			print('Encoder done!')

			# FFNN to go from encoder final state to decoder initial state
			W_intermediate=tf.get_variable(name='W_intermediate',shape=[2,
				self.encsize,self.decsize])
			decoder_state_fw_c=tf.matmul(self.encoder_state[0].c,W_intermediate[0])
			decoder_state_fw_c=tf.nn.tanh(decoder_state_fw_c)
			decoder_state_bw_c=tf.matmul(self.encoder_state[1].c,W_intermediate[0])
			decoder_state_bw_c=tf.nn.tanh(decoder_state_bw_c)

			decoder_state_fw_h=tf.matmul(self.encoder_state[0].h,W_intermediate[1])
			decoder_state_fw_h=tf.nn.tanh(decoder_state_fw_h)
			decoder_state_bw_h=tf.matmul(self.encoder_state[1].h,W_intermediate[1])
			decoder_state_bw_h=tf.nn.tanh(decoder_state_bw_h)

			# decoder
			print self.args.stack_decoder,type(self.args.stack_decoder)
			if self.args.stack_decoder=='1' : 
				print 'Stacked decoder'
				cell1=tf.contrib.rnn.BasicLSTMCell(self.decsize,activation=tf.nn.tanh)
				cell1=tf.nn.rnn_cell.DropoutWrapper(cell1,input_keep_prob=self.keep_prob,
					output_keep_prob=self.keep_prob,state_keep_prob=self.keep_prob)
				
				cell2=tf.contrib.rnn.BasicLSTMCell(self.decsize,activation=tf.nn.tanh)
				cell2=tf.nn.rnn_cell.DropoutWrapper(cell2,input_keep_prob=self.keep_prob,
					output_keep_prob=self.keep_prob,state_keep_prob=self.keep_prob)
				
				decoder_cell=[cell1,cell2]
				decoder_cell=tf.nn.rnn_cell.MultiRNNCell(decoder_cell)

				decoder_state_fw=tf.contrib.rnn.LSTMStateTuple(decoder_state_fw_c,
					decoder_state_fw_h)
				decoder_state_bw=tf.contrib.rnn.LSTMStateTuple(decoder_state_bw_c,
					decoder_state_bw_h)

				decoder_state=[decoder_state_fw,decoder_state_bw]

			else : 

				cell1=tf.contrib.rnn.BasicLSTMCell(self.decsize,activation=tf.nn.tanh)
				cell1=tf.nn.rnn_cell.DropoutWrapper(cell1,input_keep_prob=self.keep_prob,
					output_keep_prob=self.keep_prob,state_keep_prob=self.keep_prob)

				decoder_cell=cell1
				decoder_state=tf.nn.rnn_cell.LSTMStateTuple(decoder_state_fw_c,
					decoder_state_bw_c)

				
			self.sos_emb=tf.get_variable(name='sos',shape=[1,self.outembed])
			# self.sos_emb=tf.zeros(shape=[1,self.outembed],dtype=tf.float32)
			# self.sos_emb=tf.constant(np.random.normal(0,0.01,size=(1,self.outembed)),
			#	dtype=tf.float32)
			batch_size=tf.size(self.encoder_input_ind[:,0])
			self.sos_emb=tf.tile(self.sos_emb,[batch_size,1])
			print('sos_emb : ',self.sos_emb.get_shape())
			
			W_1=tf.get_variable(shape=[self.decsize,self.len_hindi_vocab],name='W_1')
			# b_1=tf.get_variable(shape=[self.len_hindi_vocab],name='b_1')

			attn_U=tf.get_variable(shape=[1,2*self.encsize,self.outembed],name='attn_U')
			attn_U_1=tf.tile(attn_U,[batch_size,1,1])
			attn_W=tf.get_variable(shape=[2*self.encsize,self.outembed],name='attn_W')
			attn_V=tf.get_variable(shape=[self.outembed,1],name='attn_V')

			ip1=self.encoder_output # batchsize x numchars x 1024
			print self.args.stack_decoder
			if self.args.stack_decoder=='1' : 
				ip2=tf.concat([decoder_state[0].c,decoder_state[1].c],axis=-1) # batchsize x 1024
			if self.args.stack_decoder=='0' : 
				ip2=tf.concat([decoder_state.c,decoder_state.h],axis=-1) # batchsize x 1024


			alphas_list=[]

			e=tf.matmul(ip2,attn_W)
			# print 'e : ',e.get_shape()
			e=tf.tile(tf.expand_dims(e,1),[1,tf.size(ip1[0,:,0]),1])
			# print 'e : ',e.get_shape()
			e=tf.matmul(ip1,attn_U_1)+e # batchsize x numchars x 256
			# print 'e : ',e.get_shape()
			e=tf.nn.tanh(tf.reshape(e,[-1,self.outembed]))
			# print 'e : ',e.get_shape()
			e=tf.matmul(e,attn_V)
			# print 'e : ',e.get_shape()
			e=tf.reshape(e,[batch_size,-1])
			# print 'e : ',e.get_shape()
			alpha=tf.nn.softmax(e,axis=-1) # batchsize x numchars
			alpha=alpha*tf.cast(self.encoder_attn_mask,tf.float32)
			alpha_sum=tf.reduce_sum(alpha,axis=1)+1e-14
			alpha_sum=tf.expand_dims(alpha_sum,1)
			alpha=tf.div(alpha,alpha_sum)
			alphas_list.append(alpha)

			alpha=tf.tile(tf.expand_dims(alpha,2),[1,1,2*self.encsize])
			c_t=tf.multiply(alpha,ip1)
			c_t=tf.reduce_sum(c_t,axis=1) # batchsize x outembed
			print('Done so far!')

			decoder_input=tf.concat([self.sos_emb,c_t],axis=-1)
			
			# print 'e : ',e.get_shape()
			# alpha=tf.nn.softmax(e,axis=-1)

			logits=[]
			predicted_hindi_chars=[]
			
			with tf.variable_scope('decoder_lstm',reuse=tf.AUTO_REUSE) as scope :

				for i in range(self.max_decoding_steps) : 

					#print 'decoder input : ',decoder_input.get_shape()

					new_decoder_output,new_decoder_state=decoder_cell(decoder_input,
						decoder_state,scope=scope)
					
					# to be used for loss
					decoder_pred_logits=tf.matmul(new_decoder_output,W_1)
					#print('decoder pred logits : ',decoder_pred_logits.get_shape())
					# decoder_pred_logits=tf.nn.softmax(decoder_pred_logits,axis=-1)
					logits.append(decoder_pred_logits)
					
					# to be used for inference
					labels_predicted_greedy_1=tf.argmax(decoder_pred_logits,axis=-1)
					#print('labels predicted greedy : ',labels_predicted_greedy_1.get_shape())
					# labels_predicted_greedy=tf.one_hot(labels_predicted_greedy_1,
					# 	depth=self.len_hindi_vocab)
					labels_predicted_greedy=tf.cast(labels_predicted_greedy_1,tf.int32)
					# new_decoder_input=tf.nn.embedding_lookup(self.decoder_emb_matrix,
					# 	labels_predicted_greedy)
					# print 'new decoder input : ',new_decoder_input.get_shape()
					#print 'decoder output[:,i,:] : ',decoder_output[:,i,:].get_shape()
					# print('new_decoder_state : ',new_decoder_state)
					predicted_hindi_chars.append(labels_predicted_greedy_1)

					#  to be used for next loop
					decoder_input_ind=tf.cond(self.is_train,
						lambda : self.decoder_output_ind[:,i], # if true
						lambda : labels_predicted_greedy) # if false
					decoder_input=tf.nn.embedding_lookup(self.decoder_emb_matrix,
						decoder_input_ind)


					# attention
					ip1=self.encoder_output # batchsize x numchars x 1024
					if self.args.stack_decoder=='1' : 
						ip2=tf.concat(
						[new_decoder_state[0].c,new_decoder_state[1].c],
						axis=-1) # batchsize x 1024
					else : 
						ip2=tf.concat(
						[new_decoder_state.c,new_decoder_state.h],
						axis=-1) # batchsize x 1024

					e=tf.matmul(ip2,attn_W)
					e=tf.tile(tf.expand_dims(e,1),[1,tf.size(ip1[0,:,0]),1])
					e=tf.matmul(ip1,attn_U_1)+e # batchsize x numchars x 1024
					e=tf.nn.tanh(tf.reshape(e,[-1,self.outembed]))
					e=tf.matmul(e,attn_V)
					e=tf.reshape(e,[batch_size,-1])
					alpha=tf.nn.softmax(e,axis=-1) # batchsize x numchars
					alpha=alpha*tf.cast(self.encoder_attn_mask,tf.float32)
					
					alpha_sum=tf.reduce_sum(alpha,axis=1)+1e-14
					alpha_sum=tf.expand_dims(alpha_sum,1)
					alpha=tf.div(alpha,alpha_sum)
					alphas_list.append(alpha)

					alpha=tf.tile(tf.expand_dims(alpha,2),[1,1,2*self.encsize])
					c_t=tf.multiply(alpha,ip1)
					c_t=tf.reduce_sum(c_t,axis=1) # batchsize x outembed

					decoder_input=tf.concat([decoder_input,c_t],axis=-1)



					if i==1 : 
						self.decoder_input_1=decoder_input_ind
					if i==2 : 
						self.decoder_input_2=decoder_input_ind
					# if self.mode=='train' : 
					# 	decoder_input=decoder_output[:,i,:]
					# else : 
					# 	decoder_input=new_decoder_input

					decoder_state=new_decoder_state

			print('Decoder done!')
					
			logits=tf.stack(logits)
			logits=tf.transpose(logits,perm=[1,0,2])
			print('logits : ',logits.get_shape())
			print('labels : ',self.decoder_output_ind.get_shape())

			alphas_list=tf.stack(alphas_list) # op_len x batch_size x ip_len
			self.alphas_list=tf.transpose(alphas_list,[1,2,0])

			# loss and optimizer
			self.ce_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
				labels=self.decoder_output_ind)

			max_time=tf.shape(self.decoder_output_ind)[1]
			target_weights=tf.sequence_mask(lengths=self.decoder_seqlen,
				maxlen=max_time,dtype=logits.dtype)
			#target_pad_weights=target_weights*-1+1
			self.ce_loss=tf.reduce_mean(self.ce_loss*target_weights)#+0.25*tf.reduce_mean(self.ce_loss*target_pad_weights)
			self.optimizer=tf.train.AdamOptimizer(float(self.args.lr)).minimize(self.ce_loss,global_step=self.global_step)
			print('Defined optimizer')

			predicted_hindi_chars=tf.stack(predicted_hindi_chars)
			print('predicted_hindi_chars : ',predicted_hindi_chars.get_shape())
			self.predicted_hindi_chars=tf.transpose(predicted_hindi_chars,perm=[1,0])



parser=argparse.ArgumentParser()

parser.add_argument("--lr",help="learning rate",default=0.001)
parser.add_argument("--batch_size",help="batch size",default="10")
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
    raise ValueError("Existing model needed!")

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
	train_model=rnn_model(args,
		len_eng_vocab,len_hindi_vocab,
		max_decoding_steps=max_len_hindi,mode='train')
	print('Train model created!')

	latest_ckpt=tf.train.latest_checkpoint(args.save_dir)

	train_model.saver.restore(sess,latest_ckpt)
	# global_step=train_model.global_step.eval(session=sess)
	global_step=sess.run(train_model.global_step)
	print('Model loaded from saved checkpoint at global step : '+str(global_step))

	print 'Starting val now'

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
test_alphas=[]
test_input_eng_chars=[]
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

	[predicted_hindi_chars,alphas,input_eng_chars]=train_model.get_attention(
		sess,test_eng_temp,test_eng_seqlen_temp,
		test_eng_attn_mask_temp)

	if i==0 : 
		test_alphas=alphas
	else : 
		test_alphas=np.append(test_alphas,alphas,axis=0)

	for j in range(predicted_hindi_chars.shape[0]) : 
		current_pred=predicted_hindi_chars[j,:]
		current_pred_char=[ind_to_hindi[x] for x in current_pred]

		if '<pad>' in current_pred_char : 
			end_index=current_pred_char.index('<pad>')
			current_pred_char=current_pred_char[0:end_index]

		current_pred_char=' '.join(current_pred_char)

		test_predicted_hindi_chars.append(current_pred_char)
	
		current_pred=input_eng_chars[j,:]
		current_pred_char=[ind_to_eng[x] for x in current_pred]

		if '<pad>' in current_pred_char : 
			end_index=current_pred_char.index('<pad>')
			current_pred_char=current_pred_char[0:end_index]

		current_pred_char=' '.join(current_pred_char)

		test_input_eng_chars.append(current_pred_char)


def plot_attention(Ws, X_label, Y_label,i,args):
    '''
    Plots the attention model heatmap
    Args:
        data: attn_matrix with shape [no_hindi_letters, no_eng_letters]
        X_label: list of hindi chars
        Y_label: list of eng chars
    '''
    
    fontP = font_manager.FontProperties(fname = 'Nirmala.ttf')
    #fontP = font_manager.FontProperties(fname = 'AksharUnicode.ttf')
    #Use one of the two .ttf files above.
    #Note that the .ttf file should be in the same place as this code 
    #or specify the correct path to the file
    
    fontP.set_size(16)
    
    #Rest of the plotting code below
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 8))  # set figure size
    heatmap = ax.pcolor(Ws, cmap=plt.cm.Blues, alpha=0.9)

    if X_label != None and Y_label != None:
        #decode fn used below takes care of making labels/Hindi chars unicode
        X_label = [x_label.decode('utf-8') for x_label in X_label]
        Y_label = [y_label.decode('utf-8') for y_label in Y_label]

        xticks = range(0, len(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels('')
        
        xticks1 = [k+0.5 for k in xticks]
        ax.set_xticks(xticks1, minor=True)
        ax.set_xticklabels(X_label, minor=True, fontproperties=fontP)  # labels should be 'unicode'
        #using fontP from above to get Hindi chars

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels('')
        
        yticks1 = [k+0.5 for k in yticks]
        ax.set_yticks(yticks1, minor=True)
        ax.set_yticklabels(Y_label, minor=True,fontproperties=fontP)  # labels should be 'unicode'

        ax.grid(True)
    eng_word=u' '.join(Y_label)
    plt.title(u'Attention Heatmap for '+eng_word)
    plt.savefig(os.path.join(args.save_dir,'attn_map_'+str(i)))





# to_plot=[0,1,2,3,442,497,640,936]
to_plot=range(200,300)
for i in range(test.shape[0]) : 
	
	if i not in to_plot : 
		continue

	print i
	alphas=test_alphas[i,:,:]
	eng_labels=test_input_eng_chars[i].split(' ')
	hindi_labels=test_predicted_hindi_chars[i].split(' ')
	print alphas.shape
	print ' '.join(eng_labels)
	print ' '.join(hindi_labels)
	alphas=alphas[0:len(eng_labels),0:len(hindi_labels)]
	print alphas.shape

	plot_attention(alphas,hindi_labels,eng_labels,i,args)
		