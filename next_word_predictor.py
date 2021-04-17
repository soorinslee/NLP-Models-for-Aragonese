import csv 
import tensorflow as tf
from tensorflow import keras
import transformers
import random 
import numpy as np
import pandas as pd

class BertPredictor(object):
    def __init__(self):       
      self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
      self.model = transformers.TFBertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
    def predict(self, loaded_model, input, masked_index, vocab):
      left_context = input[0:masked_index]
      right_context = input[masked_index+1:]

      left_string = " ".join(word for word in left_context)
      right_string = " ".join(word for word in right_context)

      context_string = left_string + "[MASK]" + right_string


      input_toks = self.tokenizer.encode(context_string)
      input_mat = np.array(input_toks).reshape((1, -1))
      input_strings = self.tokenizer.convert_ids_to_tokens(input_toks)

      outputs = loaded_model.predict(input_mat)
      predictions = outputs[0]
      best_tokens_filtered = []
      correct = 0
      best_words = np.argsort(predictions[0][masked_index])[::-1][0:300]
      best_tokens = self.tokenizer.convert_ids_to_tokens(best_words)
      for token in best_tokens:
        if token != "[MASK]" and token != "[UNK]" and token != "[SEP]" and token != "[CLS]" :
          best_tokens_filtered.append(token)
      if input[masked_index] in best_tokens_filtered:
        correct +=1
      return best_tokens_filtered, correct


    def train(self):
      train_data = []
      train_data_masked = []
      with open('/content/drive/MyDrive/Aragonese/maskedTrain.csv') as inputfile:
        for row in csv.reader(inputfile):
          if len(row) > 0:
            train_data.append(row[0])
      with open('/content/drive/MyDrive/Aragonese/train.csv') as inputfile2:
        for row in csv.reader(inputfile2):
          if len(row) > 0:
            train_data_masked.append(row[0])
      train_encodings = self.tokenizer(train_data, truncation=True, padding=True)
      train_masked_encodings = self.tokenizer(train_data_masked, truncation=True, padding=True)
      train_short_list = []
      train_masked_short_list= []
      print(len(train_data))
      for i in range(0, len(train_encodings['input_ids'])):
          if len(train_encodings['input_ids'][i]) == len(train_masked_encodings['input_ids'][i]):
            train_short_list.append(i)
            train_masked_short_list.append(i)
      print(len(train_short_list))
      train_new = []
      train_mask_new = []
      for i in range(0, len(train_short_list)):
        train_new.append(train_data[train_short_list[i]])
        train_mask_new.append(train_data_masked[train_masked_short_list[i]])

      train_encodings = self.tokenizer(train_new, truncation=True, padding=True)
      train_masked_encodings = self.tokenizer(train_mask_new, truncation=True, padding=True)
      loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy()
      optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
      self.model.compile(optimizer=optimizer, loss = self.model.compute_loss, metrics=['accuracy'])
      self.model.fit(train_masked_encodings['input_ids'], train_encodings['input_ids'], epochs=3, batch_size=10)
      self.model.save_pretrained("/content/drive/MyDrive/Aragonese/mbert_aragonese")

      loaded_model = transformers.TFBertForMaskedLM.from_pretrained("/content/drive/MyDrive/Aragonese/mbert_aragonese")
      print(loaded_model)

    def test(self, loaded_model):
      # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
      # model.compile(loss = "...", optimizer = "...", metrics = "..", options = run_opts)
      res = open("/content/drive/MyDrive/Aragonese/results.csv", "w")
      writer1 = csv.writer(res)
      df1 = pd.read_csv("/content/drive/MyDrive/Aragonese/vocab.csv", header=None, engine='python', delimiter=None, error_bad_lines=False)
      vocab = df1.values.tolist()
      total_correct = 0
      total = 0
      with open('/content/drive/MyDrive/Aragonese/test.csv') as inputfile:
          for row in csv.reader(inputfile):
            input = row[0]
            max = len(input.split())
            #r=random.randint(0, max-1)
            r = max - 1
            total +=1
            options, correct = self.predict(loaded_model, input.split(), r, vocab)
            if correct == 1:
              total_correct +=1
            #writer1.writerow(["INPUT: ", input])
            #writer1.writerow(["masked token index: ", r])
            #writer1.writerow(["OPTIONS: ", options])
            print("INPUT: ", input)
            print("R: ", r)
            print("OPTIONS: ", options)
            print("--------------------------------")
      print(100 * (total_correct/total))

if __name__ == def main():
	###TRAIN###
	#model = BertPredictor()
	#model.train()
	
	###TEST###
	loaded_model = transformers.TFBertForMaskedLM.from_pretrained("/content/drive/MyDrive/Aragonese/mbert_aragonese")
	model = BertPredictor()
	model.test(loaded_model)

