#create spell corrector ala https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/spelling-correction/1.bert-base.ipynb
#Author: Husein Zolkepli

!pip install nltk
!pip install bert-tensorflow
!pip install tensorflow --ignore-installed

import random
import nltk
import pandas as pd
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
import tensorflow as tf
import numpy as np
import unicodedata
from bert.tokenization import WordpieceTokenizer, load_vocab, convert_by_vocab

nltk.download('punkt')

!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
!unzip multi_cased_L-12_H-768_A-12.zip

BERT_VOCAB = 'multi_cased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'multi_cased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = 'multi_cased_L-12_H-768_A-12/bert_config.json'

#edit word
def edit_step(word):
    """
    All edits that are one edit away from `word`.
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
  
def edits2(word):
    """
    All edits that are two edits away from `word`.
    """
    return set(e2 for e1 in edit_step(word)
            for e2 in edit_step(e1))
    
def known(words):
    """
    The subset of `words` that appear in the dictionary of WORDS.
    """
    return [w for w in words if w in tokenizer.vocab] #change vocab file?

def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class BasicTokenizer(object):

    def __init__(self, do_lower_case=True, never_split=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text, never_split=None):
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                token = self._run_strip_accents(token)
                split_tokens.extend(self._run_split_on_punc(token))
            else:
                split_tokens.append(token)

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]
    
    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    
def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, 
                                              never_split = ['[CLS]', '[MASK]', '[SEP]'])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

tokenizer = FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)

def get_indices(mask, word):
    splitted = mask.split('**mask**')
    left = tokenizer.tokenize(splitted[0])
    middle = tokenizer.tokenize(word)
    right = tokenizer.tokenize(splitted[1])
    indices = [i for i in range(len(left))]
    for i in range(len(right)):
        indices.append(i + len(middle) + len(left))
    
    indices = indices[1:-1]
    tokenized = tokenizer.tokenize(mask.replace('**mask**',word))
    ids = tokenizer.convert_tokens_to_ids(tokenized)
    ids_left = tokenizer.convert_tokens_to_ids(left)
    ids_right = tokenizer.convert_tokens_to_ids(right)
    indices_word = ids_left + ids_right
    return ids, indices, indices_word[1:-1]

# load model instead
bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
class Model:
    def __init__(
        self,
    ):
        self.X = tf.placeholder(tf.int32, [None, None])
        
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=self.X,
            use_one_hot_embeddings=False)
        
        output_layer = model.get_sequence_output()
        embedding = model.get_embedding_table()
        
        with tf.variable_scope('cls/predictions'):
            with tf.variable_scope('transform'):
                input_tensor = tf.layers.dense(
                    output_layer,
                    units = bert_config.hidden_size,
                    activation = modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer = modeling.create_initializer(
                        bert_config.initializer_range
                    ),
                )
                input_tensor = modeling.layer_norm(input_tensor)
            
            output_bias = tf.get_variable(
            'output_bias',
            shape = [bert_config.vocab_size],
            initializer = tf.zeros_initializer(),
            )
            logits = tf.matmul(input_tensor, embedding, transpose_b = True)
            self.logits = tf.nn.bias_add(logits, output_bias)

#try it on all the test tweets
import pandas as pd

tweets = pd.read_csv('test.csv', header=None)
run = 0
count = 0
correct = 0
print(len(tweets[0]))

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()

sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bert')

for sentence in tweets[0]:
  run+=1
  print(run)
  try:
    sentence = sentence.strip('.')
    sentence = sentence.place(')', ' ')
  except:
    pass
  if sentence == '' or sentence == ' ' or '.com' in sentence or ')' in sentence:
    continue

  tokens = nltk.word_tokenize(sentence)
  rword = random.choice(tokens)

  for n, i in enumerate(tokens):
   if i == rword:
     tokens[n] = '**mask**'

  new_sentence = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(tokens)

  ed1 = edit_step(rword)
  ed2 = edits2(rword)
  edits = list(ed1)+list(ed2)
  possible_states = known(edits) + [rword]
  try:
    possible_states.remove('')
  except:
    pass
  if len(possible_states) > 15 or len(possible_states) < 2: #to prevent memory exhaustion and confirmation bias
    continue

  new_text = '[CLS] ' + new_sentence + ' [SEP]'
  text_mask = new_text.replace(' '+rword+' ', ' **mask** ')

  p_indices = [get_indices(text_mask, iword) for iword in possible_states]
  ids, seq_ids, word_ids = list(zip(*p_indices))

  try: 
    cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'cls')

    saver = tf.train.Saver(var_list = var_lists + cls)
    saver.restore(sess, BERT_INIT_CHKPNT)

    masked_padded = tf.keras.preprocessing.sequence.pad_sequences(ids,padding='post')
    masked_padded.shape

    preds = sess.run(tf.nn.softmax(loaded_model.logits), feed_dict = {loaded_model.X: masked_padded})

  except:
    continue

  count+=1

  scores = []

  for no, ids in enumerate(seq_ids):
      scores.append(np.prod(preds[no, ids, word_ids[no]]))

  prob_scores = np.array(scores) / np.sum(scores)
  probs = list(zip(possible_states, prob_scores))
  probs.sort(key = lambda x: x[1])  
  probs[::-1]
  #print(rword, probs)
  if probs[-1][0] == rword or probs[-2][0] == rword:
    correct+=1
    
print('accuracy:', correct/count)