'''
----------------------------------------------
CREATE VOCAB FROM TRAIN ENGLISH AND HINDI DATA
----------------------------------------------
'''

import pandas as pd
import os
import collections
import codecs

if not os.path.exists('vocab'):
    os.makedirs('vocab')

train=pd.read_csv('dl2019pa3/train.csv')
ids=train['id'].tolist()
eng_words=train['ENG'].tolist()
hindi_words=train['HIN'].tolist()

# make english vocab
eng_vocab=[]
for i in range(len(eng_words)) : 
	chars=eng_words[i].split(' ')
	eng_vocab.extend(chars)

eng_vocab=list(set(eng_vocab))
with codecs.open(os.path.join('vocab','eng.txt'),'w',encoding='utf8') as f : 
	f.write('<unk>')
	f.write('\n')
	f.write('<pad>')
	f.write('\n')
	for char in eng_vocab : 
		f.write(char)
		f.write('\n')


# make hindi vocab
hindi_vocab=[]
for i in range(len(hindi_words)) : 
	chars=hindi_words[i].split(' ')
	hindi_vocab.extend(chars)

hindi_vocab=list(set(hindi_vocab))
with codecs.open(os.path.join('vocab','hindi.txt'),'w',encoding='utf8') as f : 
	f.write('<unk>')
	f.write('\n')
	f.write('<pad>')
	f.write('\n')
	for char in hindi_vocab : 
		f.write(char)
		f.write('\n')
