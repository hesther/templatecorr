import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tensorflow as tf
import sklearn
from functools import partial
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense
from rdkit import Chem
from rdkit.Chem import AllChem

def ranks_to_acc(found_at_rank, fid=None):
    def fprint(txt):
#        print(txt)
        if fid is not None:
            fid.write(txt + '\n')
            
    tot = float(len(found_at_rank))
    fprint('{:>8} \t {:>8}'.format('top-n', 'accuracy'))
    accs = []
    for n in [1, 3, 5, 10, 20, 50]:
        accs.append(sum([r <= n for r in found_at_rank]) / tot)
        fprint('{:>8} \t {:>8}'.format(n, accs[-1]))
    return accs

def average_recalls(recalls,fid=None):
    average_recalls={}
    for k in [1, 5, 10, 25, 50, 100]:
        tmp=0
        for recall in recalls:
            tmp+=recall[k]
        tmp/=len(recalls)
        average_recalls[k]=tmp
    if fid is not None:
        fid.write(str(average_recalls))
    return average_recalls

def topk_appl_recall(applicable, k=[1, 5, 10, 25, 50, 100]):
    recall = {}
    for kval in k:
        recall[kval]=sum(applicable[:kval])/kval
    return recall

def loop_over_temps(template, rct):
    from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

    rct = rdchiralReactants(rct)
    
    try:
        rxn = rdchiralReaction(template)
        outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
    except Exception as e:
        outcomes = []
            
    return outcomes

def get_rank_precursor_multi(prob_list,smi,prec_goal,templates,n_cpus):
    sorted_idx=list(np.argsort(prob_list)[::-1])
    ranks=[9999]
    rank=9999
    applicable=[]
    num_prev_prec = 0
    known_prec=set()
    outcomes_all = Parallel(n_jobs=n_cpus, verbose=0)(delayed(loop_over_temps)(templates[idx],smi) for idx in sorted_idx[:100])
    for r,idx in enumerate(sorted_idx[:100]):
        outcomes=outcomes_all[r]
        num_duplicates=sum([item in known_prec for item in outcomes])
        if prec_goal in outcomes:
            ranks = [num_prev_prec+1+i for i in range(len(outcomes)-num_duplicates)]
            rank=r+1
            break
        num_prev_prec += len(outcomes)-num_duplicates
        known_prec.update(outcomes)

        if len(outcomes)!=0:
            applicable.append(1)
        else:
            applicable.append(0)
    recall=topk_appl_recall(applicable)
    
    return ranks,rank,recall

def get_multi_accs(found_at_ranks, fid=None):
    summed_up_accs=np.zeros((6))
    for item in found_at_ranks:
        accs=ranks_to_acc(item)
        summed_up_accs+=np.array(accs)
    accs=summed_up_accs/len(found_at_ranks)
    if fid is not None:
        fid.write(str(accs))
    return accs

def smiles_to_fingerprint(smi, length=2048, radius=2, useFeatures=False, useChirality=True):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise ValueError('Cannot parse {}'.format(smi))
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol=mol, radius=radius, nBits = length, 
        useFeatures=useFeatures, useChirality=useChirality
    )
    return np.array(fp_bit)

def fingerprint_training_dataset(
    smiles, labels, batch_size=256, train=True,
    fp_length=2048, fp_radius=2, fp_use_features=False, fp_use_chirality=True,
    sparse_labels=False, shuffle_buffer=1024, nproc=8, cache=True, precompute=False
):
    smiles_ds = fingerprint_dataset_from_smiles(smiles, fp_length, fp_radius, fp_use_features, fp_use_chirality, nproc, precompute)
    labels_ds = labels_dataset(labels, sparse_labels)
    ds = tf.data.Dataset.zip((smiles_ds, labels_ds))
    ds = ds.shuffle(shuffle_buffer).batch(batch_size)
    if train:
        ds = ds.repeat()
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(buffer_size=batch_size*3)
    return ds

def fingerprint_dataset_from_smiles(smiles, length, radius, useFeatures, useChirality, nproc=8, precompute=False):
    def smiles_tensor_to_fp(smi, length, radius, useFeatures, useChirality):
        smi = smi.numpy().decode('utf-8')
        length = int(length.numpy())
        radius = int(radius.numpy())
        useFeatures = bool(useFeatures.numpy())
        useChirality = bool(useChirality.numpy())
        fp_bit = smiles_to_fingerprint(smi, length, radius, useFeatures, useChirality)
        return np.array(fp_bit)
    def parse_smiles(smi):
        output = tf.py_function(
            smiles_tensor_to_fp, 
            inp=[smi, length, radius, useFeatures, useChirality], 
            Tout=tf.float32
        )
        output.set_shape((length,))
        return output
    if not precompute:
        ds = tf.data.Dataset.from_tensor_slices(smiles)
        ds = ds.map(map_func=parse_smiles, num_parallel_calls=nproc)
    else:
        if nproc!=0:
            fps = Parallel(n_jobs=nproc, verbose=1)(
                delayed(smiles_to_fingerprint)(smi, length, radius, useFeatures, useChirality) for smi in smiles
            )
        else:
            fps = [smiles_to_fingerprint(smi, length, radius, useFeatures, useChirality) for smi in smiles]
        fps = np.array(fps)
        ds = tf.data.Dataset.from_tensor_slices(fps)
    return ds

def labels_dataset(labels, sparse=False):
    if not sparse:
        return tf.data.Dataset.from_tensor_slices(labels)
    coo = labels.tocoo()
    indices = np.array([coo.row, coo.col]).T
    labels = tf.SparseTensor(indices, coo.data, coo.shape)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    labels_ds = labels_ds.map(map_func=tf.sparse.to_dense)
    return labels_ds

def sparse_categorical_crossentropy_from_logits(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def top_k(k=1):
    partial_fn = partial(tf.keras.metrics.sparse_top_k_categorical_accuracy, k=k)
    partial_fn.__name__ = 'top_{}'.format(k)
    return partial_fn
    
def build_model(
    input_shape, output_shape, num_hidden, hidden_size,
    activation='relu', output_activation=None, dropout=0.0, clipnorm=None,
    optimizer=None, learning_rate=0.001, 
    compile_model=True, loss=None, metrics=None
):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    for _ in range(num_hidden):
        model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_shape, activation=output_activation))
    if optimizer is None or optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    if clipnorm is not None:
        optimizer.clipnorm = clipnorm
    if compile_model:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    return model

def relevance(**kwargs):
    loss = sparse_categorical_crossentropy_from_logits
    metrics = [
        top_k(k=1),
        top_k(k=3),
        top_k(k=5),
        top_k(k=10),
        top_k(k=20),
        top_k(k=50),
    ]
    options = {
        'loss': loss,
        'metrics': metrics
    }
    options.update(kwargs)
    return build_model(**options)

def shuffle_arrays(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train_model(num_classes,train_smiles,train_labels,valid_smiles,valid_labels,test_smiles,test_labels,model_name1, model_name2,n_cpus):
    #Hyperparameters:
    fp_length=2048
    fp_radius=2
    weight_classes=True
    num_hidden=1
    hidden_size=2048
    dropout=0.2
    learning_rate=0.001
    activation='relu'
    batch_size=512
    clipnorm=None
    epochs=25
    early_stopping=3
    nproc=n_cpus
    precompute_fps=True

    train_smiles, train_labels = shuffle_arrays(train_smiles, train_labels)
    valid_smiles, valid_labels = shuffle_arrays(valid_smiles, valid_labels)

    train_ds = fingerprint_training_dataset(
        train_smiles, train_labels, batch_size=batch_size, train=True,
        fp_length=fp_length, fp_radius=fp_radius, nproc=nproc, precompute=precompute_fps
    )
    train_steps = np.ceil(len(train_smiles)/batch_size).astype(int)

    valid_ds = fingerprint_training_dataset(
        valid_smiles, valid_labels, batch_size=batch_size, train=False,
        fp_length=fp_length, fp_radius=fp_radius, nproc=nproc, precompute=precompute_fps
    )
    valid_steps = np.ceil(len(valid_smiles)/batch_size).astype(int)

    #Setup model details
    model = relevance(
        input_shape=(fp_length,),
        output_shape=num_classes,
        num_hidden=num_hidden,
        hidden_size=hidden_size,
        dropout=dropout,
        learning_rate=learning_rate,
        activation=activation,
        clipnorm=clipnorm
    )

    if not os.path.exists('models/ml-fixed/training_'+model_name1):
        os.makedirs('models/ml-fixed/training_'+model_name1)
    model_output = 'models/ml-fixed/training_'+model_name1+'/'+model_name2+'-weights.hdf5'
    history_output = 'models/ml-fixed/training_'+model_name1+'/'+model_name2+'-history.json'

    callbacks = []
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=early_stopping,
                restore_best_weights=True
            )
        )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            model_output, monitor='val_loss', save_weights_only=True
        )
    )

    if weight_classes:
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            'balanced', np.unique(train_labels), train_labels
        )
    else:
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            None, np.unique(train_labels), train_labels
        )

    if nproc !=0 :
        multiproc = True
        nproc = nproc
    else:
        multiproc = False
        nproc = None
    
    length = len(np.unique(train_labels))
    class_weight_dict = {i : class_weight[i] for i in range(length)}

    #Train neural network
    history = model.fit(
        train_ds, epochs=epochs, steps_per_epoch=train_steps,
        validation_data=valid_ds, validation_steps=valid_steps,
        callbacks=callbacks, class_weight=class_weight_dict,
        use_multiprocessing=multiproc, workers=nproc
    )

    pd.DataFrame(history.history).to_json(history_output)



    #Predict on test set:
        
    if nproc!=0:
        test_fps = Parallel(n_jobs=nproc, verbose=1)(
            delayed(smiles_to_fingerprint)(smi, length=fp_length, radius=fp_radius) for smi in test_smiles
        )
    else:
        test_fps = [smiles_to_fingerprint(smi, length=fp_length, radius=fp_radius) for smi in test_smiles]
    test_fps = np.array(test_fps)

    acc = model.evaluate(test_fps, test_labels, batch_size=batch_size)
    print(model.metrics_names[1:])
    print(acc[1:])
    with open('models/ml-fixed/evaluation_'+model_name1+'/'+model_name2+'_accuracy_t.out','w') as f:
        print(model.metrics_names[1:],file=f)
        print(acc[1:],file=f)

    return model,batch_size,test_fps
    
    


def ml_fixed_model(system,n_cpus=20):
    for name in ["","canonical_","corrected_","canonical_corrected_"]:
        print("###",name)
        if not os.path.exists('models/ml-fixed/evaluation_'+name+system):
            os.makedirs('models/ml-fixed/evaluation_'+name+system)

        if name in ["","canonical_"]:
            choices= ["default","r0","r1","r2","r3"]
        else:
            choices=["default","r1","r2","r3"]
            
        #Reverse
        for  i in choices:
            num_classes = int(np.load("data/"+system+"num_classes_"+name+"template_"+i+".npy"))
            print("Training model ("+i+" templates, retro) with",num_classes,"classes:")
            data_train=pd.read_csv("data/"+system+"_"+name+"template_"+i+"_train.csv")
            data_val=pd.read_csv("data/"+system+"_"+name+"template_"+i+"_val.csv")
            data_test=pd.read_csv("data/"+system+"_"+name+"template_"+i+"_test.csv")

            train_smiles=data_train['prod_smiles'].values
            train_labels=data_train[name+"template_"+i+"_id"].values
            valid_smiles=data_val['prod_smiles'].values
            valid_labels=data_val[name+"template_"+i+"_id"].values
            test_smiles=data_test['prod_smiles'].values
            test_labels=data_test[name+"template_"+i+"_id"].values

            model,batch_size,test_fps = train_model(num_classes,train_smiles,train_labels,valid_smiles,valid_labels,test_smiles,test_labels,name+system,i+"_retro",n_cpus)

            #Check precursor evaluation:
            data_test=pd.read_csv("data/"+system+"_forward_"+name+"template_"+i+"_test.csv")
            forward_test_smiles=data_test['reac_smiles'].values
            templates=pd.read_csv("data/"+system+"_"+name+"template_"+i+"_unique_templates.txt",header=None)[0].values
            preds=model.predict(test_fps,batch_size=batch_size)
            #Calculate ranks
            found_at_rank = []
            found_at_ranks = []
            recalls=[]
            for j in range(len(test_labels)):
                print(j, end='\r')
                found_ranks,found_rank,recall=get_rank_precursor_multi(preds[j],test_smiles[j],forward_test_smiles[j],templates, n_cpus)
                found_at_rank.append(found_rank)
                found_at_ranks.append(found_ranks)
                recalls.append(recall)
            with open('models/ml-fixed/evaluation_'+name+system+'/'+i+"_retro"+'_accuracy_p.out','w') as fid:
                accs=ranks_to_acc(found_at_rank,fid=fid)
            with open('models/ml-fixed/evaluation_'+name+system+'/'+i+"_retro"+'_accuracy_p_multi.out','w') as fid:
                accs_multi=get_multi_accs(found_at_ranks,fid=fid) 
            with open('models/ml-fixed/evaluation_'+name+system+'/appl__retro_template_'+i+'.out','w') as fid:
                mean_recall=average_recalls(recalls,fid=fid)
            print("Via precursor:")
            print(accs)
            print(mean_recall)
            print("Via precursor (multi):")
            print(accs_multi)
            

        #Forward
        for  i in choices:
            num_classes = int(np.load("data/"+system+"num_classes_forward_"+name+"template_"+i+".npy"))
            print("Training model ("+i+" templates, forward) with",num_classes,"classes:")
            data_train=pd.read_csv("data/"+system+"_forward_"+name+"template_"+i+"_train.csv")
            data_val=pd.read_csv("data/"+system+"_forward_"+name+"template_"+i+"_val.csv")
            data_test=pd.read_csv("data/"+system+"_forward_"+name+"template_"+i+"_test.csv")

            train_smiles=data_train['reac_smiles'].values
            train_labels=data_train['forward_'+name+"template_"+i+"_id"].values
            valid_smiles=data_val['reac_smiles'].values
            valid_labels=data_val['forward_'+name+"template_"+i+"_id"].values
            test_smiles=data_test['reac_smiles'].values
            test_labels=data_test['forward_'+name+"template_"+i+"_id"].values
            
            model,batch_size,test_fps = train_model(num_classes,train_smiles,train_labels,valid_smiles,valid_labels,test_smiles,test_labels,name+system,i+"_forward", n_cpus)

            #Check precursor evaluation:
            data_test=pd.read_csv("data/"+system+"_"+name+"template_"+i+"_test.csv")
            forward_test_smiles=data_test['prod_smiles'].values
            templates=pd.read_csv("data/"+system+"_forward_"+name+"template_"+i+"_unique_templates.txt",header=None)[0].values
            preds=model.predict(test_fps,batch_size=batch_size)
            #Calculate ranks
            found_at_rank = []
            found_at_ranks = []
            recalls=[]
            for j in range(len(test_labels)):
                print(j, end='\r')
                found_ranks,found_rank,recall=get_rank_precursor_multi(preds[j],test_smiles[j],forward_test_smiles[j],templates, n_cpus)
                found_at_rank.append(found_rank)
                found_at_ranks.append(found_ranks)
                recalls.append(recall)
            with open('models/ml-fixed/evaluation_'+name+system+'/'+i+"_retro"+'_accuracy_p.out','w') as fid:
                accs=ranks_to_acc(found_at_rank,fid=fid)
            with open('models/ml-fixed/evaluation_'+name+system+'/'+i+"_retro"+'_accuracy_p_multi.out','w') as fid:
                accs_multi=get_multi_accs(found_at_ranks,fid=fid) 
            with open('models/ml-fixed/evaluation_'+name+system+'/appl__retro_template_'+i+'.out','w') as fid:
                mean_recall=average_recalls(recalls,fid=fid)
            print("Via precursor:")
            print(accs)
            print(mean_recall)
            print("Via precursor (multi):")
            print(accs_multi)
            

if __name__ == '__main__':
    ml_fixed_model('uspto_50k')
    #ml_fixed_model('uspto_460k')
