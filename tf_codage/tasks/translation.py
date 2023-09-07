import tensorflow as tf
import datetime as dt
import pytz
from ..utils import save_metrics_to_file,save_config_to_file,pad_tensor, save_results_to_file
from ..utils import save_tokenizer,load_tokenizer
from ..loss import WeightedBinaryCrossEntropy
from ..evaluate import seq2seq_true_false_positives,seq2seq_accuracy

from .. import MakeDataset
from .. import Transformer, TransformerConfig, KerasTransformer


class TranslationTrainer:
    
    def __init__(self,config): 
        
        self.config = config

        if hasattr(self.config, 'UNK_TOKEN_ID'):
            self.config.UNK_TOKEN_ID = 1
        
    
    def build(self):
        
        md = MakeDataset(self.config)
        md.input_feature = self.config.INPUT_FEATURE
        md.target_feature = self.config.TARGET_FEATURE
        
        self.input_tokenizer = load_tokenizer(self.config.PATH_INPUT_TOKENIZER )
        intput_vocab = self.input_tokenizer.get_vocabulary()
        self.output_tokenizer = load_tokenizer(self.config.PATH_OUTPUT_TOKENIZER )
        output_vocab = self.output_tokenizer.get_vocabulary()
        self.output_index_lookup = dict(zip(range(len(output_vocab)), output_vocab))

        md.tokenizers.update({"input": self.input_tokenizer})
        md.tokenizers.update({"output": self.output_tokenizer})
        
        md.DATASET_NAME = "train"
        self.train_ds = md.get_data("tranformer_train_preprocessing")

        md.DATASET_NAME = "val"
        self.val_ds = md.get_data("tranformer_train_preprocessing")

        md.DATASET_NAME = "test"
        md.DATASET_EPOCHS = 1
        self.test_ds = md.get_data("tranformer_eval_preprocessing")
        
        
        self.config.VOCAB_SIZE = len(intput_vocab)
        
        self.config.OUTPUT_VOCAB_SIZE = len(output_vocab)
        
        
        model_config = TransformerConfig(
                            self.config.NUM_LAYERS,
                            self.config.EMBED_DIM,
                            self.config.NUM_HEADS,
                            self.config.FF_DIM,
                            self.config.OUTPUT_VOCAB_SIZE, 
                            pe_input=self.config.MAX_LEN, 
                            pe_target=self.config.OUTPUT_MAX_LEN,
                            decoder_pad_token_id=0,
                            input_vocab_size=self.config.VOCAB_SIZE)
        
        

        if tf.config.list_physical_devices('GPU'):
          self.strategy = tf.distribute.MirroredStrategy()
          print("GPUs Available for training : ", len(tf.config.list_physical_devices('GPU')))
        else:  # Use the Default Strategy
          self.strategy = tf.distribute.get_strategy()
          print("Training on CPU")

        with self.strategy.scope():
            self.model = Transformer(model_config)



        
    def train(self):         


        with self.strategy.scope():

            self.model.compile(self.config.OPTIMIZER,  
                        loss=self.config.LOSS,  
                        metrics=self.config.METRICS)
            
            if self.config.STEPS_PER_EPOCH == 0:
                STEPS_PER_EPOCH = None
            else:
                STEPS_PER_EPOCH = self.config.STEPS_PER_EPOCH 
                
            self.model.fit(self.train_ds,
                           validation_data=self.val_ds,
                           epochs=self.config.EPOCHS,
                           steps_per_epoch=STEPS_PER_EPOCH,
                           validation_steps=self.config.VALIDATION_STEPS,
                           callbacks=[self.config.CALLBACK])

    def myprint(self,s):
        with open(self.config.PATH_DETAILS_FILE,'a') as f:
             print(s, file=f)
    
    def load_model(self): 

        with self.strategy.scope():
            print(self.config.PATH_MODEL)
            self.model.load_weights(self.config.PATH_MODEL).expect_partial()
        
    def decode_sequence(self,encoded_ids,attention_mask):
        
        encoder_input =  tf.reshape(encoded_ids,(1,self.config.MAX_LEN))
        attention_input=  tf.reshape(attention_mask,(1,self.config.MAX_LEN))
        pred_text = "start"

        for i in range(self.config.OUTPUT_MAX_LEN):
            
            tokenised_target_sentence = self.output_tokenizer(pred_text)
            decoder_input = tf.reshape(tokenised_target_sentence,(1,self.config.OUTPUT_MAX_LEN))
            
            with self.strategy.scope():
                predictions = self.model({"input_ids" : encoder_input,
                                           "attention_mask" : attention_input,
                                           "codes" : decoder_input},training = False)

            sampled_token_index = tf.math.argmax(predictions[0, i, :])
            sampled_token_index_ = tf.expand_dims(sampled_token_index, -1)
            sampled_token = self.output_index_lookup[sampled_token_index.numpy()]

            if i ==0:
                pred_encoded = sampled_token_index_
            else:
                pred_encoded = tf.concat([pred_encoded,sampled_token_index_],axis = -1)

            pred_text += " " + sampled_token

            if sampled_token == "end":
                break

        return pred_encoded, pred_text

    def evaluate_model(self):
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        trues = 0
        nb = 0
        
        PATH_SAVE_RESULTS = self.config.PATH_DATA + self.config.MODEL_NAME + "_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        for nb_batch, batch in enumerate(self.test_ds):
            
            inputs, labels, input_lines, output_lines = batch
            
            predictions =  tf.Variable(tf.zeros(labels.shape,dtype=labels.dtype))
            
            batch_size = self.config.BATCH_SIZE
            
            
            for i in range(batch_size):
                
                if i < inputs['input_ids'].shape[0] :
                    pred_encoded, pred_text = self.decode_sequence(inputs['input_ids'][i],
                                                          inputs['attention_mask'][i])
                    pred_encoded = pad_tensor(pred_encoded,self.config.OUTPUT_MAX_LEN)
                    predictions = predictions[i].assign(pred_encoded)
                
                    save_results_to_file(PATH_SAVE_RESULTS,input_lines[i].numpy(), output_lines[i].numpy(), pred_text)
                
                nb +=1

            tp_batch, fp_batch, fn_batch = seq2seq_true_false_positives(predictions, labels,
                                                                        self.config.FIRST_TOKEN_OUT_TOKENIZER,
                                                                        self.config.UNK_TOKEN_ID)
            trues_batch = seq2seq_accuracy(predictions, labels)

            true_positives += tp_batch
            false_positives += fp_batch
            false_negatives += fn_batch
            trues += trues_batch
            
        if true_positives == 0:
            precision = float(0)
            recall = float(0)
            f_measure = float(0)
        
        else:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f_measure = 2 * precision * recall / (precision + recall)
            precision = precision.numpy()
            recall = recall.numpy()
            f_measure = f_measure.numpy()
        
        accuracy = trues / nb
        accuracy = accuracy.numpy()

        save_metrics_to_file(self.config.PATH_DETAILS_FILE,
                             Date = dt.datetime.now().astimezone(pytz.timezone('Europe/Paris')).strftime("%d/%m/%Y %H:%M:%S"))


        self.model.summary(print_fn=self.myprint)
        
        
        save_metrics_to_file(self.config.PATH_DETAILS_FILE,
                             INPUT_MAX_LEN =self.config.MAX_LEN, 
                             OUTPUT_MAX_LEN = self.config.OUTPUT_MAX_LEN,
                             BATCH_SIZE = self.config.BATCH_SIZE,
                             INPUT_VOCAB_SIZE = self.config.VOCAB_SIZE,
                             OUTPUT_VOCAB_SIZE = self.config.OUTPUT_VOCAB_SIZE,

                             MASK_SELECTION_RATE = self.config.MASK_SELECTION_RATE,
                             DROPOUT_RATE = self.config.DROPOUT_RATE,
                             LR = self.config.LR,
                             
                             EMBED_DIM = self.config.EMBED_DIM,
                             NUM_HEADS = self.config.NUM_HEADS,
                             FF_DIM = self.config.FF_DIM,
                             NUM_LAYERS = self.config.NUM_LAYERS,
                                 
                             FIRST_TOKEN_OUT_TOKENIZER = self.config.FIRST_TOKEN_OUT_TOKENIZER,
                             UNK_TOKEN_ID = self.config.UNK_TOKEN_ID,
                             
                             Precision=precision,
                             Recall=recall,
                             F_measure=f_measure,
                             Accuracy = accuracy)
        
        return precision,recall,f_measure,accuracy
    
# class KerasTranslationTrainer:
    
#     def __init__(self,config): 
        
#         self.config = config

#         if hasattr(self.config, 'UNK_TOKEN_ID'):
#             self.config.UNK_TOKEN_ID = 1
        
    
#     def build(self):
        
#         md = MakeDataset(self.config)
#         md.input_feature = self.config.INPUT_FEATURE
#         md.target_feature = self.config.TARGET_FEATURE
        
#         self.input_tokenizer = load_tokenizer(self.config.PATH_INPUT_TOKENIZER )
#         intput_vocab = self.input_tokenizer.get_vocabulary()
#         self.output_tokenizer = load_tokenizer(self.config.PATH_OUTPUT_TOKENIZER )
#         output_vocab = self.output_tokenizer.get_vocabulary()
#         self.output_index_lookup = dict(zip(range(len(output_vocab)), output_vocab))

#         md.tokenizers.update({"input": self.input_tokenizer})
#         md.tokenizers.update({"output": self.output_tokenizer})
        
#         md.DATASET_NAME = "train"
#         self.train_ds = md.get_data("Keras_tranformer_train_preprocessing")

#         md.DATASET_NAME = "val"
#         self.val_ds = md.get_data("Keras_tranformer_train_preprocessing")

#         md.DATASET_NAME = "test"
#         md.DATASET_EPOCHS = 1
#         self.test_ds = md.get_data("Keras_tranformer_eval_preprocessing")
        
        
#         self.config.VOCAB_SIZE = len(intput_vocab)
        
#         self.config.OUTPUT_VOCAB_SIZE = len(output_vocab)      
        

#         if tf.config.list_physical_devices('GPU'):
#           self.strategy = tf.distribute.MirroredStrategy()
#           print("GPUs Available for training : ", len(tf.config.list_physical_devices('GPU')))
#         else:  # Use the Default Strategy
#           self.strategy = tf.distribute.get_strategy()
#           print("Training on CPU")

#         with self.strategy.scope():
#             self.model = KerasTransformer(self.config)



        
#     def train(self):         


#         with self.strategy.scope():

#             self.model.compile(self.config.OPTIMIZER,  
#                         loss=self.config.LOSS,  
#                         metrics=self.config.METRICS)
            
#             if self.config.STEPS_PER_EPOCH == 0:
#                 STEPS_PER_EPOCH = None
#             else:
#                 STEPS_PER_EPOCH = self.config.STEPS_PER_EPOCH 
                
#             self.model.fit(self.train_ds,
#                            validation_data=self.val_ds,
#                            epochs=self.config.EPOCHS,
#                            steps_per_epoch=STEPS_PER_EPOCH,
#                            validation_steps=self.config.VALIDATION_STEPS,
#                            callbacks=[self.config.CALLBACK])


    
#     def load_model(self): 

#         with self.strategy.scope():
#             print(self.config.PATH_MODEL)
#             self.model.load_weights(self.config.PATH_MODEL).expect_partial()
        
#     def decode_sequence(self,encoded_ids):
        
#         print(encoded_ids)
#         encoder_input =  tf.reshape(encoded_ids,(1,self.config.MAX_LEN))
#         pred_text = "start"

#         for i in range(self.config.OUTPUT_MAX_LEN):
            
#             tokenised_target_sentence = self.output_tokenizer(pred_text)
#             decoder_input = tf.reshape(tokenised_target_sentence,(1,self.config.OUTPUT_MAX_LEN))
            
#             with self.strategy.scope():
#                 predictions = self.model([encoder_input, decoder_input])

#             sampled_token_index = tf.math.argmax(predictions[0, i, :])
#             sampled_token_index_ = tf.expand_dims(sampled_token_index, -1)
#             sampled_token = self.output_index_lookup[sampled_token_index.numpy()]

#             if i ==0:
#                 pred_encoded = sampled_token_index_
#             else:
#                 pred_encoded = tf.concat([pred_encoded,sampled_token_index_],axis = -1)

#             pred_text += " " + sampled_token

#             if sampled_token == "end":
#                 break

#         return pred_encoded, pred_text

#     def evaluate_model(self):
        
#         true_positives = 0
#         false_positives = 0
#         false_negatives = 0
#         trues = 0
#         nb = 0
        
#         PATH_SAVE_RESULTS = self.config.PATH_DATA + self.config.MODEL_NAME + "_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        
#         for nb_batch, batch in enumerate(self.test_ds):
            
#             inputs, labels, input_lines, output_lines = batch
#             predictions =  tf.Variable(tf.zeros(labels.shape,dtype=labels.dtype))
            
#             batch_size = self.config.BATCH_SIZE
            
            
#             for i in range(batch_size):
                
#                  if i < inputs['encoder_inputs'].shape[0] :
#                     pred_encoded, pred_text = self.decode_sequence(inputs['encoder_inputs'][i])
                
#                     pred_encoded = pad_tensor(pred_encoded,self.config.OUTPUT_MAX_LEN)
#                     predictions = predictions[i].assign(pred_encoded)
                    
#                     save_results_to_file(PATH_SAVE_RESULTS,input_lines[i].numpy(), output_lines[i].numpy(), pred_text)
                    
#                     nb +=1

#             tp_batch, fp_batch, fn_batch = seq2seq_true_false_positives(predictions, labels,
#                                                                         self.config.FIRST_TOKEN_OUT_TOKENIZER,
#                                                                         self.config.UNK_TOKEN_ID)
#             trues_batch = seq2seq_accuracy(predictions, labels)

#             true_positives += tp_batch
#             false_positives += fp_batch
#             false_negatives += fn_batch
#             trues += trues_batch
            
#         if true_positives == 0:
#             precision = float(0)
#             recall = float(0)
#             f_measure = float(0)
        
#         else:
#             precision = true_positives / (true_positives + false_positives)
#             recall = true_positives / (true_positives + false_negatives)
#             f_measure = 2 * precision * recall / (precision + recall)
#             precision = precision.numpy()
#             recall = recall.numpy()
#             f_measure = f_measure.numpy()
        
#         accuracy = trues / nb
#         accuracy = accuracy.numpy()

#         save_metrics_to_file(self.config.PATH_DETAILS_FILE,
#                              Date = dt.datetime.now().astimezone(pytz.timezone('Europe/Paris')).strftime("%d/%m/%Y %H:%M:%S"))

#         def myprint(s):
#             with open(self.config.PATH_DETAILS_FILE,'a') as f:
#                 print(s, file=f)

#         self.model.summary(print_fn=myprint)
        
        
#         save_metrics_to_file(self.config.PATH_DETAILS_FILE,
#                              INPUT_MAX_LEN =self.config.MAX_LEN, 
#                              OUTPUT_MAX_LEN = self.config.OUTPUT_MAX_LEN,
#                              BATCH_SIZE = self.config.BATCH_SIZE,
#                              INPUT_VOCAB_SIZE = self.config.VOCAB_SIZE,
#                              OUTPUT_VOCAB_SIZE = self.config.OUTPUT_VOCAB_SIZE,

#                              MASK_SELECTION_RATE = self.config.MASK_SELECTION_RATE,
#                              DROPOUT_RATE = self.config.DROPOUT_RATE,
#                              LR = self.config.LR,
                             
#                              EMBED_DIM = self.config.EMBED_DIM,
#                              NUM_HEADS = self.config.NUM_HEADS,
#                              FF_DIM = self.config.FF_DIM,
#                              NUM_LAYERS = self.config.NUM_LAYERS,
                                 
#                              FIRST_TOKEN_OUT_TOKENIZER = self.config.FIRST_TOKEN_OUT_TOKENIZER,
#                              UNK_TOKEN_ID = self.config.UNK_TOKEN_ID,
                             
#                              Precision=precision,
#                              Recall=recall,
#                              F_measure=f_measure,
#                              Accuracy = accuracy)
        
#         return precision,recall,f_measure,accuracy
    

        
  