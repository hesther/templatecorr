"""
 Install chemprop, see https://github.com/chemprop/chemprop
 To train all chemprop models, run the following command on the terminal:

for name in _ _canonical_ _corrected_ _canonical_corrected_; do
    echo $name
    if [[ $name == *"corrected"* ]]; then
        list="default r3 r2 r1"
    else
        list="default r3 r2 r1 r0"
    fi
    echo Iterating through $list
    for i in $list; do
        j=$(cat data/uspto_50k${name}template_${i}_unique_templates.txt | wc -l)
        chemprop_train \
               --data_path data/uspto_50k${name}template_${i}_train.csv \
               --separate_val_path data/uspto_50k${name}template_${i}_val.csv \
               --separate_test_path data/uspto_50k${name}template_${i}_test.csv \
               --dataset_type multiclass \
               --multiclass_num_classes $j \
               --save_dir models/ml-learned/uspto_50k${name}${i} \
               --save_preds
        chemprop_train \
               --data_path data/uspto_50k_forward${name}template_${i}_train.csv \
               --separate_val_path data/uspto_50k_forward${name}template_${i}_val.csv \
               --separate_test_path data/uspto_50k_forward${name}template_${i}_test.csv \
               --dataset_type multiclass \
               --multiclass_num_classes $j \
               --save_dir models/ml-learned/uspto_50k_forward${name}${i} \
               --save_preds
    done
done
"""

import numpy as np
import ast
import pandas as pd
import os     
from joblib import Parallel, delayed

def get_rank(true_index,prob_list):
    sorted_idx=list(np.argsort(prob_list)[::-1])
    rank=sorted_idx.index(true_index)+1
    return rank

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

def loop_over_temps(template, rct):
    from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

    rct = rdchiralReactants(rct)
    
    try:
        rxn = rdchiralReaction(template)
        outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
    except Exception as e:
        outcomes = []
            
    return outcomes

def topk_appl_recall(applicable, k=[1, 5, 10, 25, 50, 100]):
    recall = {}
    for kval in k:
        recall[kval]=sum(applicable[:kval])/kval
    return recall

def get_rank_precursor(prob_list,smi,prec_goal,templates,n_cpus):
    sorted_idx=list(np.argsort(prob_list)[::-1])
    rank=9999
    applicable=[]
    outcomes_all = Parallel(n_jobs=n_cpus, verbose=0)(delayed(loop_over_temps)(templates[idx],smi) for idx in sorted_idx[:100])
    for r,idx in enumerate(sorted_idx[:100]):
        outcomes=outcomes_all[r]
        if prec_goal in outcomes:
            rank=min(rank,r+1)
        if len(outcomes)!=0:
            applicable.append(1)
        else:
            applicable.append(0)
    recall=topk_appl_recall(applicable)
    return rank, recall

def evaluate_mllearned_model(system,n_cpus=20):
    for name in ["","canonical_","corrected_","canonical_corrected_"]:
        print("###",name)

        if not os.path.exists('models/ml-learned/evaluation_'+name+system):
            os.makedirs('models/ml-learned/evaluation_'+name+system)
            
        if name in ["","canonical_"]:
            choices= ["default","r0","r1","r2","r3"]
        else:
            choices=["default","r1","r2","r3"]

        for  i in choices:
            
            data_test=pd.read_csv("data/"+system+"_"+name+"template_"+i+"_test.csv")
            true_labels=data_test[name+"template_"+i+"_id"].values
            test_smiles=data_test['prod_smiles'].values
            
            data_test=pd.read_csv("data/"+system+"_forward_"+name+"template_"+i+"_test.csv")
            forward_test_smiles=data_test['reac_smiles'].values

            #Reverse
            try:
                preds=pd.read_csv("models/ml-learned/"+system+"_"+name+i+"/test_preds.csv")[name+"template_"+i+"_id"].values
            except Exception as e:
                print("Must train Chemprop model first, see header comment in this file.")
                raise e
            templates=pd.read_csv("data/"+system+"_"+name+"template_"+i+"_unique_templates.txt",header=None)[0].values
            found_at_rank = Parallel(n_jobs=n_cpus, verbose=1)(delayed(get_rank)(true_labels[j],ast.literal_eval(preds[j])) for j in range(len(true_labels)))
                                          
            with open('models/ml-learned/evaluation_'+name+system+'/'+i+"_retro"+'_accuracy_t.out','w') as fid:
                ranks_to_acc(found_at_rank,fid=fid)


            #Reverse CheckP
            found_at_rank = []
            recalls=[]
            for j in range(len(test_smiles)):
                found_rank,recall=get_rank_precursor(ast.literal_eval(preds[j]),test_smiles[j],forward_test_smiles[j],templates, n_cpus)
                found_at_rank.append(found_rank)
                recalls.append(recall)
            with open('models/ml-learned/evaluation_'+name+system+'/'+i+"_retro"+'_accuracy_p.out','w') as fid:
                accs=ranks_to_acc(found_at_rank,fid=fid)
            with open('models/ml-learned/evaluation_'+name+system+'/appl__retro_template_'+i+'.out','w') as fid:
                mean_recall=average_recalls(recalls,fid=fid)
            print(accs)
            print(mean_recall)
    
            #Forward
            true_labels=data_test['forward_'+name+"template_"+i+"_id"].values
            try:
                preds=pd.read_csv("models/ml-learned/"+system+"_forward_"+name+i+"/test_preds.csv")['forward_'+name+"template_"+i+"_id"].values
            except Exception as e:
                print("Must train Chemprop model first, see header comment in this file.")
                raise e
            templates=pd.read_csv("data/"+system+"_forward_"+name+"template_"+i+"_unique_templates.txt",header=None)[0].values
            found_at_rank = Parallel(n_jobs=n_cpus, verbose=1)(delayed(get_rank)(true_labels[j],ast.literal_eval(preds[j])) for j in range(len(true_labels)))
  
            with open('models/ml-learned/evaluation_'+name+system+'/'+i+"_forward"+'_accuracy_t.out','w') as fid:
                ranks_to_acc(found_at_rank,fid=fid)           

            #Forward CheckP
            found_at_rank = []
            recalls=[]
            for j in range(len(forward_test_smiles)):
                print(j, end='\r')
                found_rank,recall=get_rank_precursor(ast.literal_eval(preds[j]),forward_test_smiles[j],test_smiles[j],templates, n_cpus)
                found_at_rank.append(found_rank)
                recalls.append(recall)
            with open('models/ml-learned/evaluation_'+name+system+'/'+i+"_forward"+'_accuracy_p.out','w') as fid:
                accs=ranks_to_acc(found_at_rank,fid=fid)
            with open('models/ml-learned/evaluation_'+name+system+'/appl__forward_template_'+i+'.out','w') as fid:
                mean_recall=average_recalls(recalls,fid=fid)
            print(accs)
            print(mean_recall)

if __name__ == '__main__':
    evaluate_mllearned_model('uspto_50k')
