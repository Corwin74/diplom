import tensorflow as tf
import keras.backend as K
import os
import pandas as pd
import sys
import pickle
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras import Model

from sklearn.model_selection import train_test_split

from dataset_preprocessing_utils import transform_transactions_to_sequences, create_padded_buckets
from data_generators import transaction_features

TRAIN_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle/train_transactions_contest/'
TEST_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle/test_transactions_contest/'
TRAIN_TARGET_PATH = '/media/DATA/AlfaBattle/train_target.csv'
PRE_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle//preprocessed_transactions/'
PRE_TEST_TRANSACTIONS_PATH = '/media/DATA/AlfaBattle/preprocessed_test_transactions/'
PICKLE_VAL_BUCKET_PATH = '/media/DATA/AlfaBattle/val_buckets/'
PICKLE_VAL_TRAIN_BUCKET_PATH = '/media/DATA/AlfaBattle/val_train_buckets/'
PICKLE_VAL_TEST_BUCKET_PATH = '/media/DATA/AlfaBattle/val_test_buckets/'
CHECKPOINTS_ADV_PATH = '/media/DATA/AlfaBattle/checkpoints/tf_advanced_baseline/'

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
    
    #add dropout=0.5
    concated = L.concatenate([pooled_avg_sequences, pooled_max_sequences, product_emb_reshape])
    
    dense_intermediate = L.Dense(classifier_units, activation='relu', 
                                 kernel_regularizer=keras.regularizers.L1L2(1e-7, 1e-5))(concated)
    
    proba = L.Dense(1, activation='sigmoid')(dense_intermediate)
    
    model = Model(inputs=inputs, outputs=proba)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

with open('/home/alex/diplom/constants/buckets_info.pkl', 'rb') as f:
    mapping_seq_len_to_padded_len = pickle.load(f)
    
with open('/home/alex/diplom/constants/dense_features_buckets.pkl', 'rb') as f:
    dense_features_buckets = pickle.load(f)

df = pd.read_csv(sys.argv[1])

for dense_col in ['amnt', 'days_before', 'hour_diff']:
            df[dense_col] = np.digitize(df[dense_col], bins=dense_features_buckets[dense_col])


seq = transform_transactions_to_sequences(df)
seq['sequence_length'] = seq.sequences.apply(lambda x: len(x[1]))
seq['product'] = 1

x = create_padded_buckets(seq, mapping_seq_len_to_padded_len, save_to_file_path=None, has_target=False)

embedding_projections = {'currency': (11, 6),
                        'operation_kind': (7, 5),
                        'card_type': (175, 29),
                        'operation_type': (22, 9),
                        'operation_type_group': (4, 3),
                        'ecommerce_flag': (3, 3),
                        'payment_system': (7, 5),
                        'income_flag': (3, 3),
                        'mcc': (108, 22),
                        'country': (24, 9),
                        'city': (163, 28),
                        'mcc_category': (28, 10),
                        'day_of_week': (7, 5),
                        'hour': (24, 9),
                        'weekofyear': (53, 15),
                        'amnt': (10, 6),
                        'days_before': (23, 9),
                        'hour_diff': (10, 6),
                        'product': (5, 4)}

model = build_transactions_rnn(transaction_features, embedding_projections, classifier_units=128)
model.load_weights(os.path.join(CHECKPOINTS_ADV_PATH, 'epoch_5_val_0.802.hdf5'))
padded_sequences, products = x['padded_sequences'], x['products']
batch_size = 1
bucket, product = padded_sequences[0], products[0]
app_id = x['app_id']
for jdx in range(0, len(bucket), batch_size):
                    batch_sequences = bucket[jdx: jdx + batch_size]
                    batch_products = product[jdx: jdx + batch_size]
                    batch_app_ids = app_id[jdx: jdx + batch_size]
batch_sequences = [batch_sequences[:, i] for i in range(len(transaction_features))]
batch_sequences.append(batch_products)
test_preds_web = model.predict(batch_sequences)
print(test_preds_web[0][0])
