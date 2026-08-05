# -*- coding: utf-8 -*-

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed, GRU
from tensorflow.keras import initializers, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

    
def auto_encoder_lstm(train_x, train_y, args):
    
    print("LSTM model")
    # Define the encoder
    encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder = LSTM(args.latent_dim,
                    dropout=args.dropout, 
                    recurrent_dropout=args.recurrent_dropout, 
                    return_state=True)

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Define the decoder
    decoder_inputs = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_lstm = LSTM(args.latent_dim, 
                        dropout=args.dropout, 
                        recurrent_dropout=args.recurrent_dropout, 
                        return_sequences=True, 
                        return_state=False)

    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(train_y.shape[2], activation='linear'))
    
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the encoder-decoder model
    model = Model(encoder_inputs, decoder_outputs)
    
    adam_optimizer = Adam(learning_rate=args.learning_rate)
    
    print("MAE Loss function is used")
    model.compile(optimizer=adam_optimizer, loss="mae")
    
    return model


def auto_encoder_gru(train_x, train_y, args):
    print("GRU model")
    # Define the encoder
    encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder = GRU(args.latent_dim,
                    dropout=args.dropout, 
                    recurrent_dropout=args.recurrent_dropout, 
                    return_state=True)

    encoder_outputs, state_h = encoder(encoder_inputs)

    # Define the decoder
    decoder_inputs = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_ = GRU(args.latent_dim, 
                        dropout=args.dropout, 
                        recurrent_dropout=args.recurrent_dropout, 
                        return_sequences=True, 
                        return_state=False)

    decoder_outputs = decoder_(decoder_inputs, initial_state=state_h)
    decoder_dense = TimeDistributed(Dense(train_y.shape[2], activation='linear'))
    
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the encoder-decoder model
    model = Model(encoder_inputs, decoder_outputs)
    
    adam_optimizer = Adam(learning_rate=args.learning_rate)
    
    print("MAE Loss function is used")
    model.compile(optimizer=adam_optimizer, loss="mae")
    
    return model


# class lstm():
#     def __init__(self, train_x, train_y, loss, args):
#         self.train_x = train_x
#         self.train_y = train_y
#         self.args = args
#         self.loss = loss
        
#     def extreme_loss(y_true, y_pred, alpha=3.0, beta=15.0):
#         """
#         Extreme Value Loss Function, focusing on extreme events in predictions.
        
#         Parameters:
#         y_true (tensor): True values.
#         y_pred (tensor): Predicted values.
#         alpha (float): Coefficient for standard deviation to define extremes.
#         beta (float): Weight multiplier for extreme values.
        
#         Returns:
#         loss (tensor): Computed loss value focusing on extreme events.
#         """
#         # Standard deviation and mean based on true values
#         mean, variance = tf.nn.moments(y_true, axes=[0])
#         std_dev = tf.sqrt(variance)

#         # Define extremes: values beyond 'alpha' standard deviations from the mean
#         extreme_mask = tf.abs(y_true - mean) > alpha * std_dev

#         # Calculate absolute errors
#         errors = tf.abs(y_true - y_pred)

#         # Apply different weights to errors based on whether they are extreme
#         weights = tf.where(extreme_mask, beta, 1.0)
#         weighted_errors = weights * errors

#         # Mean error to form the loss
#         loss = tf.reduce_mean(weighted_errors)
#         return loss

#     def auto_encoder(self, train_x, train_y, loss, args):
#         train_x = self.train_x
#         train_y = self.train_y
#         args = self.args
#         loss = self.loss
        
#         # Define the encoder
#         encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
#         encoder = LSTM(args.latent_dim,
#                        dropout=args.dropout, 
#                        recurrent_dropout=args.recurrent_dropout, 
#                        return_state=True)

#         encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#         encoder_states = [state_h, state_c]

#         # Define the decoder
#         decoder_inputs = RepeatVector(train_y.shape[1])(encoder_outputs)
#         decoder_lstm = LSTM(args.latent_dim, 
#                             dropout=args.dropout, 
#                             recurrent_dropout=args.recurrent_dropout, 
#                             return_sequences=True, 
#                             return_state=False)

#         decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
#         decoder_dense = TimeDistributed(Dense(train_y.shape[2], activation='linear'))
        
#         decoder_outputs = decoder_dense(decoder_outputs)
#         # Define the encoder-decoder model
#         model = Model(encoder_inputs, decoder_outputs)
        
#         adam_optimizer = Adam(learning_rate=args.learning_rate)
        
#         model.compile(optimizer=adam_optimizer, loss=loss)
        
#         return model








