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
from tqdm import tqdm

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

path_to_test_dataset = PICKLE_VAL_TEST_BUCKET_PATH
dir_with_test_datasets = os.listdir(path_to_test_dataset)
dataset_test = sorted([os.path.join(path_to_test_dataset, x) for x in dir_with_test_datasets])

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


test_frame = pd.read_csv('/media/DATA/AlfaBattle/test_target_contest.csv')
test_preds = []

for weights_name in [
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/1_epoch_6_val_0.797.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/2_epoch_7_val_0.800.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/3_epoch_8_val_0.798.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/4_epoch_6_val_0.796.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/5_epoch_6_val_0.796.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/6_epoch_7_val_0.797.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/7_epoch_5_val_0.796.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/8_epoch_7_val_0.799.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/9_epoch_8_val_0.798.hdf5', 
                  '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/10_epoch_8_val_0.797.hdf5'
                  ]:


    model.load_weights(weights_name)

    test_preds.append(inference(model, dataset_test, batch_size=128))

df = pd.concat(test_preds, axis=1, ignore_index=True)
try:
  df.columns=['index1', 'pred1', 'index2', 'pred2', 'index3', 'pred3',
              'index4', 'pred4', 'index5', 'pred5', 'index6', 'pred6',
              'index7', 'pred7', 'index8', 'pred8', 'index9', 'pred9', 'index10', 'pred10']
except:
  pass

sub = df[['pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6', 'pred7', 'pred8', 'pred9', 'pred10']].mean(axis=1)

submission = pd.DataFrame({
    'app_id' : df.index1.values,
    'score': sub
})

print(submission.head())

submission.to_csv('blend_nn_submission.csv', index=None)
