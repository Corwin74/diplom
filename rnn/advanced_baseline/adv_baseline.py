import tensorflow as tf
import keras.backend as K
import os
import pandas as pd
import sys
import pickle
import numpy as np

from tensorflow import config
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras import Model

from sklearn.model_selection import train_test_split
from tqdm import tqdm\

from focal_loss import BinaryFocalLoss

# добавим корневую папку, в ней лежат все необходимые полезные функции для обработки данных
sys.path.append('../../')
sys.path.append('../')

from data_generators import batches_generator, transaction_features
from tf_training import train_epoch, eval_model, inference
from training_aux import EarlyStopping



gpus = config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      config.experimental.set_memory_growth(gpu, True)
    logical_gpus = config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

TRAIN_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle/train_transactions_contest/'
TEST_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle/test_transactions_contest/'

TRAIN_TARGET_PATH = '/media/DATA/AlfaBattle/train_target.csv'
PRE_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle//preprocessed_transactions/'
PRE_TEST_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle/preprocessed_test_transactions/'
PICKLE_VAL_BUCKET_PATH = '/media/DATA/AlfaBattle/val_buckets/'
PICKLE_VAL_TRAIN_BUCKET_PATH = '/media/DATA/AlfaBattle/val_train_buckets/'
PICKLE_VAL_TEST_BUCKET_PATH = '/media/DATA/AlfaBattle/val_test_buckets/'
CHECKPOINTS_ADV_PATH = '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/'

path_to_dataset = PICKLE_VAL_BUCKET_PATH
dir_with_datasets = os.listdir(path_to_dataset)
dataset_val = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

path_to_dataset = PICKLE_VAL_TRAIN_BUCKET_PATH
dir_with_datasets = os.listdir(path_to_dataset)
dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

with open('../constants/embedding_projections.pkl', 'rb') as f:
    embedding_projections = pickle.load(f)

def build_transactions_rnn(transactions_cat_features, embedding_projections, product_col_name='product', 
                          rnn_units=128, classifier_units=32, optimizer=None):
    if not optimizer:
        optimizer = keras.optimizers.Adam(lr=1e-3)
        
    inputs = []
    cat_embeds = []
    
    for feature_name in transactions_cat_features:
        inp = L.Input(shape=(None, ), dtype='uint32', name=f'input_{feature_name}')
        inputs.append(inp)
        source_size, projection = embedding_projections[feature_name]
        emb = L.Embedding(source_size+1, projection, trainable=True, mask_zero=False, name=f'embedding_{feature_name}')(inp)
        cat_embeds.append(emb)
    
    # product feature
    inp = L.Input(shape=(1, ), dtype='uint32', name=f'input_product')
    inputs.append(inp)
    source_size, projection = embedding_projections['product']
    product_emb = L.Embedding(source_size+1, projection, trainable=True, mask_zero=False, name=f'embedding_product')(inp)
    product_emb_reshape = L.Reshape((projection, ))(product_emb)
    
    concated_cat_embeds = L.concatenate(cat_embeds)
    dropout_embeds = L.SpatialDropout1D(0.05)(concated_cat_embeds)
    
    sequences = L.Bidirectional(L.GRU(units=rnn_units, return_sequences=True))(dropout_embeds)
    
    pooled_avg_sequences = L.GlobalAveragePooling1D()(sequences)
    pooled_max_sequences = L.GlobalMaxPooling1D()(sequences)
    
    #add dropout=0.3
    concated = L.concatenate([pooled_avg_sequences, pooled_max_sequences, product_emb_reshape])
    
    dropout = L.Dropout(0.5)(concated)

    #change  activation='relu', to sigmoid
    dense_intermediate = L.Dense(classifier_units, activation='relu', 
                                 kernel_regularizer=keras.regularizers.L1L2(1e-7, 1e-5))(dropout)
    
    proba = L.Dense(1, activation='sigmoid')(dense_intermediate)
    
    model = Model(inputs=inputs, outputs=proba)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

path_to_checkpoints = CHECKPOINTS_ADV_PATH
es = EarlyStopping(patience=3, mode='max', verbose=True, save_path=os.path.join(path_to_checkpoints, 'best_checkpoint.pt'), 
                   metric_name='ROC-AUC', save_format='tf')

num_epochs = 20
train_batch_size = 128
val_batch_szie = 128

model = build_transactions_rnn(transaction_features, embedding_projections, classifier_units=128)

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}')
    train_epoch(model, dataset_train, batch_size=train_batch_size, shuffle=True, cur_epoch=epoch, 
                steps_per_epoch=7270) #7270
    
    val_roc_auc = eval_model(model, dataset_val, batch_size=val_batch_szie)
    model.save_weights(os.path.join(path_to_checkpoints, f'epoch_{epoch+1}_val_{val_roc_auc:.3f}.hdf5'))
    
    es(val_roc_auc, model)
    
    train_roc_auc = eval_model(model, dataset_train, batch_size=val_batch_szie)
    print(f'Epoch {epoch+1} completed. Train roc-auc: {train_roc_auc}, Val roc-auc: {val_roc_auc}')
    
    if es.early_stop:
        print('Early stopping reached. Stop training...')
        break


