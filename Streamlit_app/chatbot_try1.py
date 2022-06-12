# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 13:47:21 2022

@author: faruk
"""

# import module
import streamlit as st
from transformer import transformer
from preprocess import preprocess
import pickle
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
 
tokenizer = pickle.load(open("tokenizer.pkl",'rb'))

def get_model():
    
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    UNITS = 512
    DROPOUT = 0.1
    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2
    
    model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)
    
    model.load_weights("checkpoints/my_checkpoint")
        
    
    return model

model = get_model()

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Maximum sentence length
MAX_LENGTH = 40


#To predict
def function1(sentence_text):
  sentence = preprocess(sentence_text)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)
    
  prediction = tf.squeeze(output, axis=0)

  predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

  #print('Input: {}'.format(sentence_text))
  print('Bot: {}'.format(predicted_sentence))


  return predicted_sentence



# Running it on streamlit

st.title('A transformer based Chatbot')
st.write("This chat bot is trained on cornell movie dataset")

user_input = st.text_input('Input')



if st.button('Send'):
    output = function1(str(user_input))
    st.write(output)

