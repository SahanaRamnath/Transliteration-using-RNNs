'''
---------------------------------
PA3 CODE FOR TRAINING AND TESTING
---------------------------------
'''

import tensorflow as tf
import numpy as np
import pandas as pd

import os
import argparse
import pickle
import codecs

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
		self.keep_prob=tf.placeholder(tf.float32)
		self.is_train=tf.placeholder(tf.bool)

	def train(self,sess,encoder_input_ind,encoder_seqlen,decoder_output_ind,
		keep_prob=1.0) : 

		keep_prob=float(self.args.keep_prob)
		[ce_loss,global_step,opt,predicted_hindi_chars]=sess.run(
			[self.ce_loss,self.global_step,self.optimizer,self.predicted_hindi_chars],
			feed_dict={self.encoder_input_ind : encoder_input_ind, self.encoder_seqlen : encoder_seqlen, self.decoder_output_ind : decoder_output_ind,self.keep_prob : keep_prob,self.is_train : 1})
		# print '\n\n\n'
		# print dc_ip_1
		# print dc_ip_2

		return [ce_loss,global_step,predicted_hindi_chars]

	def val(self,sess,encoder_input_ind,encoder_seqlen,decoder_output_ind,
		keep_prob=1.0,is_train=0) : 

		[ce_loss,predicted_hindi_chars,dc_ip_1,dc_ip_2]=sess.run(
			[self.ce_loss,self.predicted_hindi_chars,self.decoder_input_1,self.decoder_input_2],
			feed_dict={self.encoder_input_ind : encoder_input_ind, self.encoder_seqlen : encoder_seqlen, self.decoder_output_ind : decoder_output_ind,self.keep_prob : keep_prob,self.is_train : 0})
		print '\n\n\n'
		# print dc_ip_1
		# print dc_ip_2
		return [ce_loss,predicted_hindi_chars]

	def test(self,sess,encoder_input_ind,encoder_seqlen,
		keep_prob=1.0,is_train=0) : 

		# needed for tf cond, but not used
		decoder_output_ind=np.zeros((encoder_input_ind.shape[0],self.max_decoding_steps))

		predicted_hindi_chars=sess.run(self.predicted_hindi_chars,
			feed_dict={self.encoder_input_ind : encoder_input_ind, self.encoder_seqlen : encoder_seqlen, self.decoder_output_ind : decoder_output_ind,self.keep_prob : keep_prob,self.is_train : 0})
		return predicted_hindi_chars

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
					input_keep_prob=self.keep_prob)
				bw_cell=tf.nn.rnn_cell.DropoutWrapper(
					tf.contrib.rnn.BasicLSTMCell(self.encsize,activation=tf.nn.tanh),
					input_keep_prob=self.keep_prob)
				encoder_output,encoder_state=tf.nn.bidirectional_dynamic_rnn(
					time_major=False, dtype=tf.float32,scope=scope,
					cell_fw=fw_cell,cell_bw=bw_cell,
					inputs=encoder_input,
					sequence_length=self.encoder_seqlen)

				self.encoder_state=encoder_state#tf.nn.rnn_cell.LSTMStateTuple(encoder_state[0].c,
					#encoder_state[1].c)
				self.encoder_output=tf.concat(encoder_output,-1)
				print 'encoder output : ',self.encoder_output.get_shape()

			print('Encoder done!')
			# decoder
			if self.args.stack_decoder==1 : 
				cell1=tf.contrib.rnn.BasicLSTMCell(self.decsize,activation=tf.nn.tanh)
				cell1=tf.nn.rnn_cell.DropoutWrapper(cell1,input_keep_prob=self.keep_prob)
				
				cell2=tf.contrib.rnn.BasicLSTMCell(self.decsize,activation=tf.nn.tanh)
				cell2=tf.nn.rnn_cell.DropoutWrapper(cell2,input_keep_prob=self.keep_prob)
				
				decoder_cell=[cell1,cell2]

				decoder_cell=tf.nn.rnn_cell.MultiRNNCell(decoder_cell)
				decoder_state=[self.encoder_state[0],self.encoder_state[1]]
			else : 

				cell1=tf.contrib.rnn.BasicLSTMCell(self.decsize,activation=tf.nn.tanh)
				cell1=tf.nn.rnn_cell.DropoutWrapper(cell1,input_keep_prob=self.keep_prob)

				decoder_cell=cell1
				decoder_state=self.encoder_state
			self.sos_emb=tf.get_variable(name='sos',shape=[1,self.outembed])
			# self.sos_emb=tf.zeros(shape=[1,self.outembed],dtype=tf.float32)
			# self.sos_emb=tf.constant(np.random.normal(0,0.01,size=(1,self.outembed)),
			#	dtype=tf.float32)
			batch_size=tf.size(self.encoder_input_ind[:,0])
			self.sos_emb=tf.tile(self.sos_emb,[batch_size,1])
			print 'sos_emb : ',self.sos_emb.get_shape()
			decoder_input=self.sos_emb
			
			decoder_output=tf.nn.embedding_lookup(self.decoder_emb_matrix,
				self.decoder_output_ind)
			print 'decoder output : ',decoder_output.get_shape()
			W_1=tf.get_variable(shape=[self.decsize,self.len_hindi_vocab],name='W_1')
			# b_1=tf.get_variable(shape=[self.len_hindi_vocab],name='b_1')

			# attn_U=tf.get_variable(shape=[2*self.encsize,self.encsize],name='attn_U')
			# attn_V=tf.get_variable(shape=[self.decsize,1],name='attn_V')
			# attn_W=tf.get_variable(shape=[self.decsize,self.decsize],name='attn_W')

			# e=tf.matmul(self.encoder_output,attn_U)+tf.matmul(decoder_state,attn_W)
			# e=tf.nn.tanh(e)
			# e=tf.matmul(e,attn_V)
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
					print 'decoder pred logits : ',decoder_pred_logits.get_shape()
					# decoder_pred_logits=tf.nn.softmax(decoder_pred_logits,axis=-1)
					logits.append(decoder_pred_logits)
					
					# to be used for inference
					labels_predicted_greedy_1=tf.argmax(decoder_pred_logits,axis=-1)
					print 'labels predicted greedy : ',labels_predicted_greedy_1.get_shape()
					# labels_predicted_greedy=tf.one_hot(labels_predicted_greedy_1,
					# 	depth=self.len_hindi_vocab)
					labels_predicted_greedy=tf.cast(labels_predicted_greedy_1,tf.int32)
					new_decoder_input=tf.nn.embedding_lookup(self.decoder_emb_matrix,
						labels_predicted_greedy)
					print 'new decoder input : ',new_decoder_input.get_shape()
					print 'decoder output[:,i,:] : ',decoder_output[:,i,:].get_shape()
					print 'new_decoder_state : ',new_decoder_state
					predicted_hindi_chars.append(labels_predicted_greedy_1)

					#  to be used for next loop
					decoder_input=tf.cond(self.is_train,
						lambda : decoder_output[:,i,:], # if true
						lambda : new_decoder_input) # if false
					if i==1 : 
						self.decoder_input_1=decoder_input
					if i==2 : 
						self.decoder_input_2=decoder_input
					# if self.mode=='train' : 
					# 	decoder_input=decoder_output[:,i,:]
					# else : 
					# 	decoder_input=new_decoder_input

					decoder_state=new_decoder_state

			print('Decoder done!')
					
			logits=tf.stack(logits)
			logits=tf.transpose(logits,perm=[1,0,2])
			print 'logits : ',logits.get_shape()
			print 'labels : ',self.decoder_output_ind.get_shape()

			# loss and optimizer
			self.ce_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
				labels=self.decoder_output_ind)
			self.ce_loss=tf.reduce_mean(self.ce_loss)
			self.optimizer=tf.train.AdamOptimizer(float(self.args.lr)).minimize(self.ce_loss,global_step=self.global_step)
			print('Defined optimizer')

			predicted_hindi_chars=tf.stack(predicted_hindi_chars)
			print 'predicted_hindi_chars : ',predicted_hindi_chars.get_shape()
			self.predicted_hindi_chars=tf.transpose(predicted_hindi_chars,perm=[1,0])


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
    os.makedirs(args.save_dir)

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

# converting to index matrices
train_eng_matrix=np.zeros((train.shape[0],max_len_eng))
train_eng_seqlen=np.zeros(train.shape[0])
train_hindi_matrix=np.zeros((train.shape[0],max_len_hindi))
for i in range(train.shape[0]) : 
	word_eng=train_eng[i].split(' ')
	tmp_char=[]
	for char in word_eng : 
		tmp_char.append(eng_to_ind[char])
	if len(tmp_char)>max_len_eng : 
		train_eng_seqlen[i]=max_len_eng
		tmp_char=tmp_char[:max_len_eng]
	if len(tmp_char)<max_len_eng : 
		train_eng_seqlen[i]=len(tmp_char)
		tmp_char+=[1]*(max_len_eng-len(tmp_char))
	train_eng_matrix[i]=tmp_char

	word_hindi=train_hindi[i].split(' ')
	tmp_char=[]
	for char in word_hindi : 
		tmp_char.append(hindi_to_ind[char])
	if len(tmp_char)>max_len_hindi : 
		tmp_char=tmp_char[:max_len_hindi]
	if len(tmp_char)<max_len_hindi : 
		tmp_char+=[1]*(max_len_hindi-len(tmp_char))
	train_hindi_matrix[i]=tmp_char
print('Train converted characters to indices')


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
for i in range(val.shape[0]) : 
	word_eng=val_eng[i].split(' ')
	tmp_char=[]
	for char in word_eng : 
		tmp_char.append(eng_to_ind[char])
	if len(tmp_char)>max_len_eng : 
		val_eng_seqlen[i]=max_len_eng
		tmp_char=tmp_char[:max_len_eng]
	if len(tmp_char)<max_len_eng : 
		val_eng_seqlen[i]=len(tmp_char)
		tmp_char+=[1]*(max_len_eng-len(tmp_char))
	val_eng_matrix[i]=tmp_char

	word_hindi=val_hindi[i].split(' ')
	tmp_char=[]
	for char in word_hindi : 
		tmp_char.append(hindi_to_ind[char])
	if len(tmp_char)>max_len_hindi : 
		tmp_char=tmp_char[:max_len_hindi]
	if len(tmp_char)<max_len_hindi : 
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
		tmp_char=tmp_char[:max_len_eng]
	if len(tmp_char)<max_len_eng : 
		test_eng_seqlen[i]=len(tmp_char)
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
	if latest_ckpt : 
		train_model.saver.restore(sess,latest_ckpt)
		# global_step=train_model.global_step.eval(session=sess)
		global_step=sess.run(train_model.global_step)
		print('Model loaded from saved checkpoint at global step : '+str(global_step))
	else : 
		sess.run(tf.global_variables_initializer())
		print('Model created and initialised')

	while epoch<int(args.epochs) : 

		if train.shape[0]%batch_size==0 : 
			limit=int(train.shape[0]/batch_size)
		else : 
			limit=int(train.shape[0]/batch_size)+1
			print 'limit : ',limit

		for i in range(limit) : # each epoch
			try : 
				train_ids_temp=train_ids[i*batch_size:(i+1)*batch_size]
				train_eng_temp=train_eng_matrix[i*batch_size:(i+1)*batch_size,:].astype(np.int32)
				train_eng_seqlen_temp=train_eng_seqlen[i*batch_size:(i+1)*batch_size].astype(np.int32)
				train_hindi_temp=train_hindi_matrix[i*batch_size:(i+1)*batch_size,:].astype(np.int32)
			except : 
				train_ids_temp=train_ids[i*batch_size:]
				train_eng_temp=train_eng_matrix[i*batch_size:,:].astype(np.int32)
				train_eng_seqlen_temp=train_eng_seqlen[i*batch_size:].astype(np.int32)
				train_hindi_temp=train_hindi_matrix[i*batch_size:,:].astype(np.int32)

			[ce_loss,global_step,predicted_hindi_chars]=train_model.train(sess=sess,encoder_input_ind=train_eng_temp,
				encoder_seqlen=train_eng_seqlen_temp,decoder_output_ind=train_hindi_temp)

			if i%10==0 : 
				num_correct=0
				for j in range(predicted_hindi_chars.shape[0]) : 
					current_pred=predicted_hindi_chars[j,:]
					print 'shape of current pred : ',current_pred.shape
					current_pred_char=[ind_to_hindi[x] for x in current_pred]
					current_pred_char=' '.join(current_pred_char)
					print 'current_pred_char : ',current_pred_char
					print 'label : ',train_hindi[i*batch_size+j]
					if current_pred_char==train_hindi[i*batch_size+j] : 
						num_correct=num_correct+1
				accuracy=float(num_correct)/float(predicted_hindi_chars.shape[0])

				print 'Global Step ',global_step,', i ',i,', loss : ',ce_loss,', accuracy : ',accuracy
			if i==11 : 
				os.sys.exit()
			
		train_loss_list.append(ce_loss)

		# train_model.saver.save(sess,os.path.join(args.save_dir,'rnn-model'),
		# 	global_step=global_step)

		# latest_ckpt=tf.train.latest_checkpoint(args.save_dir)
		# train_model.saver.restore(sess,latest_ckpt)
		# print '\n\n'
		# val_model.saver.restore(sess,latest_ckpt)
		
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
			except : 
				val_ids_temp=val_ids[i*batch_size:]
				val_eng_temp=val_eng_matrix[i*batch_size:,:].astype(np.int32)
				val_eng_seqlen_temp=val_eng_seqlen[i*batch_size:].astype(np.int32)
				val_hindi_temp=val_hindi_matrix[i*batch_size:,:].astype(np.int32)

			[val_ce_loss,predicted_hindi_chars]=train_model.val(sess,val_eng_temp,
				val_eng_seqlen_temp,val_hindi_temp)


			for j in range(predicted_hindi_chars.shape[0]) : 
				current_pred=predicted_hindi_chars[j,:]
				current_pred_char=[ind_to_hindi[x] for x in current_pred]
				current_pred_char=' '.join(current_pred_char)

				val_predicted_hindi_chars.append(current_pred_char)
				# print val_predicted_hindi_chars					
			val_predicted_ids.extend(val_ids_temp)

		val_loss_list.append(val_ce_loss)
		epoch_list.append(epoch)

		num_correct=0
		for i in range(val.shape[0]) : 
			current_pred=val_predicted_hindi_chars[i]
			end_index=current_pred.index('<pad>')
			current_pred=current_pred[0:end_index]
			if val_hindi[i]==current_pred : 
				num_correct=num_correct+1
		accuracy=float(num_correct)/float(val.shape[0])
		print('Accuracy at epoch '+str(epoch)+' is '+str(accuracy))

		patience=patience+1
		if accuracy>prev_accuracy : 
			prev_accuracy=accuracy
			patience=0
			train_model.saver.save(sess,os.path.join(args.save_dir,'rnn-model'),
				global_step=global_step)

		if patience==1 :
			print('Early Stopping with a patience of 5 epochs. Breaking now..') 
			break
		epoch=epoch+1




latest_ckpt=tf.train.latest_checkpoint(args.save_dir)
train_model.saver.restore(sess,latest_ckpt)
global_step=sess.run(train_model.global_step)
print('For testing, model loaded from saved checkpoint at global step : '+str(global_step))
print('Validation accuracy of best model : '+str(prev_accuracy))


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
	except : 
		test_ids_temp=test_ids[i*batch_size:]
		test_eng_temp=test_eng_matrix[i*batch_size:,:].astype(np.int32)
		test_eng_seqlen_temp=test_eng_seqlen[i*batch_size:].astype(np.int32)

	predicted_hindi_chars=train_model.test(sess,test_eng_temp,test_eng_seqlen_temp)


	for j in range(predicted_hindi_chars.shape[0]) : 
		current_pred=predicted_hindi_chars[j,:]
		current_pred_char=[ind_to_hindi[x] for x in current_pred]
		current_pred_char=' '.join(current_pred_char)

		test_predicted_hindi_chars.append(current_pred_char)
		# print test_predicted_hindi_chars					
	test_predicted_ids.extend(test_ids_temp)

with codecs.open(os.path.join(args.save_dir,args.save_dir+'_'+str(global_step)+'.csv'),'w') as f : 
	f.write('id,HIN')
	f.write('\n')
	for i in range(test.shape[0]) :  
		f.write(str(test_predicted_ids[i]))
		f.write(',')

		current_pred=test_predicted_hindi_chars[i]
		end_index=current_pred.index('<pad>')
		current_pred=current_pred[0:end_index]

		f.write(current_pred)
		f.write('\n')

print('Saved test prediction file!')
# saving the loss lists to be used later
with open(os.path.join(args.save_dir,'train_loss_list.pkl'), 'w') as f:
     pickle.dump([epoch_list,train_loss_list], f)
with open(os.path.join(args.save_dir,'val_loss_list.pkl'), 'w') as f:
     pickle.dump([epoch_list,val_loss_list], f)
