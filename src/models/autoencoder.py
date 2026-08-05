#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM and GRU autoencoder models (TensorFlow/Keras).
Registered as "lstm_autoencoder" and "gru_autoencoder" in the model registry.

TensorFlow is imported lazily inside each factory so that this module can be
registered and imported on PyTorch-only machines without error.

Config keys used:
    hidden_dim        — latent / encoder-decoder size
    dropout_rate      — dropout on cell outputs
    recurrent_dropout — dropout on recurrent connections (optional, default 0.0)
    learning_rate     — Adam learning rate
"""

from framework.model_registry import register_tf


@register_tf("lstm_autoencoder")
def build_lstm_autoencoder(train_x, train_y, config):
    """LSTM encoder–decoder registered as 'lstm_autoencoder'."""
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed

    latent_dim        = config["hidden_dim"]
    dropout           = config["dropout_rate"]
    recurrent_dropout = config.get("recurrent_dropout", 0.0)

    encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder        = LSTM(latent_dim,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs  = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_lstm    = LSTM(latent_dim,
                           dropout=dropout,
                           recurrent_dropout=recurrent_dropout,
                           return_sequences=True,
                           return_state=False)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(
        Dense(train_y.shape[2], activation="linear")
    )(decoder_outputs)

    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=config["learning_rate"]), loss="mae")
    return model


@register_tf("gru_autoencoder")
def build_gru_autoencoder(train_x, train_y, config):
    """GRU encoder–decoder registered as 'gru_autoencoder'."""
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GRU, RepeatVector, Dense, TimeDistributed

    latent_dim        = config["hidden_dim"]
    dropout           = config["dropout_rate"]
    recurrent_dropout = config.get("recurrent_dropout", 0.0)

    encoder_inputs  = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder         = GRU(latent_dim,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    decoder_inputs  = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_gru     = GRU(latent_dim,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          return_sequences=True,
                          return_state=False)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_outputs = TimeDistributed(
        Dense(train_y.shape[2], activation="linear")
    )(decoder_outputs)

    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=config["learning_rate"]), loss="mae")
    return model
