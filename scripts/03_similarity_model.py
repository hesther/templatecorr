import pandas as pd
import numpy as np
import time
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs
from rdchiral.main import rdchiralReactants
from collections import defaultdict
import os
from joblib import Parallel, delayed

# Set up fingerprinting of molecules
getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi),2,useChirality=True, useFeatures=True)

# Set up similarity label
similarity_metric = DataStructs.BulkTanimotoSimilarity
    
def ranks_to_acc(found_at_rank, fid=None):
    """
    Computes top-1-accuracy from list of ranks
   
    :param found_at_rank: List of ranks.
    :param fid: File identifier.
    
    :return accs: top-N-accuracies
    """
    def fprint(txt):
        print(txt)
        if fid is not None:
            fid.write(txt + '\n')
            
    tot = float(len(found_at_rank))
    fprint('{:>8} \t {:>8}'.format('top-n', 'accuracy'))
    accs = []
    for n in [1, 3, 5, 10, 20, 50]:
        accs.append(sum([r <= n for r in found_at_rank]) / tot)
        fprint('{:>8} \t {:>8}'.format(n, accs[-1]))
    return accs

def topk_appl_recall(applicable, k=[1, 5, 10, 25, 50, 100]):
    """
    Computes the fraction of applicable templates up to a ranks in k
    
    :param applicable: List of applicabilities (0 and 1)
    :param k: List of ranks to evaluate at.

    :return average_recalls: Fraction of applicable templates
    """
    recall = {}
    for kval in k:
        recall[kval]=sum(applicable[:kval])/kval
    return recall

def average_recalls(recalls,fid=None):
    """
    Computes average applicability recalls from list of dictionaries

    :param recalls: List of recall dictionaries
    :param fid: File identifier.

    :return average_recalls: Averaged recalls
    """
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

def loop_over_train(template, rcts_ref_fp, rct):
    """
    Parallelizable loop over training datapoints
    
    :param template: Reaction template
    :param rcts_ref_fp: Reference fingerprint for precursors
    :rct: Smiles string on which to apply template

    return outcomes, sim_dict: List of reaction outcomes and fingerprint similarities
    """

    #Imports needed for parallel execution (alternative: put in separate file)
    from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
    import rdkit.Chem as Chem
    import rdkit.Chem.AllChem as AllChem
    from rdkit import DataStructs

    # Set up fingerprinting of molecules
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi),2,useChirality=True, useFeatures=True)

    # Set up similarity label
    similarity_metric = DataStructs.BulkTanimotoSimilarity

    rct = rdchiralReactants(rct)
    
    try:
        rxn = rdchiralReaction(template)
        outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
    except Exception as e:
        outcomes = []

    sim_dict={}
    for precursors in outcomes:
        precursors_fp = getfp(precursors)
        precursors_sim = similarity_metric(precursors_fp, [rcts_ref_fp])[0]
        sim_dict[precursors]= precursors_sim
            
    return outcomes, sim_dict
    
def do_one(column_template,column_left,column_right, datasub,datasub_test_ix, n_cpus, max_prec=100, testlimit = 100):
    """
    Calculates ranks and applicabilities for one test reaction

    :param column_template: Key of column that holds template
    :param column_left: Key of column that holds the molecule to apply template to
    :param column_right: Key of column that holds the correct reaction outcome
    :param datasub: Pandas dataframe of training reactions
    :param datasub_test_ix: Current row of test reaction
    :param n_cpus: Number of cpus to run parallel on
    :param max_prec: Limit up to which to parse through training database
    :param test_limit: Limit up to which to calculate ranks
    
    :return found_rank, recall, found_rank_t: Rank of results via precursor, applicability recall, rank of results via template
    """
    #loads product SMILES into RDChiral object

    #get the fingerprint of the product
    fp = datasub_test_ix[column_left+'_fp']
    
    #calculates similarity metric between fingerprint 
    # and all fingerprints in the database
    sims = similarity_metric (fp, [fp_ for fp_ in datasub[column_left+'_fp']])
    
    #sort the similarity metric in reverse order?
    js = np.argsort(sims,kind='mergesort')[::-1]
    
    #This gets the precursor goal molecule
    prec_goal=datasub_test_ix[column_right+'_smiles']
        
    # Get probability of precursors
    probs = {}

    applicable=[]

    template_for_precs = {}
    template_goal= datasub_test_ix[column_template]

    outcomes_all = Parallel(n_jobs=n_cpus, verbose=0)(delayed(loop_over_train)(datasub[column_template][datasub.index[j]],datasub[column_right+'_fp'][datasub.index[j]],datasub_test_ix[column_left+'_smiles']) for j in js[:max_prec])
    
    for ji,j in enumerate (js[:max_prec]):
        jx = datasub.index[j]
        template = datasub[column_template][jx]
        
        outcomes, sim_dict = outcomes_all[ji]
            
        if len(outcomes)!=0:
            applicable.append(1)
        else:
            applicable.append(0)
            
        for precursors in sorted(outcomes):
            precursors_sim = sim_dict[precursors]
            if precursors in probs:
                probs[precursors] = max(probs[precursors], precursors_sim * sims[j])
                template_for_precs[precursors].append(template)
            else:
                probs[precursors] = precursors_sim * sims[j]
                template_for_precs[precursors] = [template]

    #Rank via precursor
    found_rank = 9999
    for r, (prec, prob) in enumerate(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:testlimit]):
        if prec == prec_goal:
            found_rank = r + 1
            break
    recall=topk_appl_recall(applicable)

    #Rank via template
    found_rank_t = 9999
    for r, (prec, prob) in enumerate(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:testlimit]):
        if template_goal in template_for_precs[prec]:
            found_rank_t = r + 1
            break
        
    return found_rank, recall, found_rank_t

def calculate_accuracies(datasub,datasub_test,column_template,column_left,column_right,name, n_cpus):
    """
    Calculates top-N-accuracies for a specified dataset.
    :param datasub: Pandas dataframe of training reactions
    :param datasub_test: Pandas dataframe of test reactions
    :param column_template: Key of column that holds template
    :param column_left: Key of column that holds the molecule to apply template to
    :param column_right: Key of column that holds the correct reaction outcome
    :param name: Name to save files under
    :param n_cpus: Number of cpus to run parallel on
    """
    found_at_rank = []
    found_at_rank_t = []
    recalls=[]

    for ii, ix in enumerate(datasub_test.index):
        print(ii, end='\r')
        found_rank, recall, found_rank_t = do_one(column_template,column_left,column_right,datasub,datasub_test.loc[ix], n_cpus)
        found_at_rank.append(found_rank)
        found_at_rank_t.append(found_rank_t)
        recalls.append(recall)

    if column_left=='prod':
        direction="_retro"
    else:
        direction="_forward"
    print("Via precursor, full test set (",len(found_at_rank)," reactions)")
    with open("models/sim/evaluation_"+name+"/"+column_template+direction+"_p.out", 'w') as fid:
        ranks_to_acc(found_at_rank,fid=fid)
    print("Via template, full test set (",len(found_at_rank_t)," reactions)")    
    with open("models/sim/evaluation_"+name+"/"+column_template+direction+"_t.out", 'w') as fid:
        ranks_to_acc(found_at_rank_t,fid=fid)
    print("Applicabilities")
    with open("models/sim/evaluation_"+name+"/appl_"+direction+"_"+column_template+".out", 'w') as fid:
        mean_recall=average_recalls(recalls,fid=fid)
    print(mean_recall)


def similarity_model(system, n_cpus=20):
    """
    Computes a similarity model for a specified system.

    :param system: Name of system, e.g. uspto_50k
    :param n_cpus: Number of cpus to run parallel on
    """
    datasub_test=pd.read_pickle("data/"+system+"_test.pkl")
    datasub=pd.read_pickle("data/"+system+"_train.pkl")
    
    for name in ["","canonical_","corrected_","canonical_corrected_"]:
        print("###",name)
        if not os.path.exists('models/sim/evaluation_'+name+system):
            os.makedirs('models/sim/evaluation_'+name+system)

        if name in ["","canonical_"]:
            choices= ["default","r0","r1","r2","r3"]
        else:
            choices=["default","r1","r2","r3"]
            
        #Retro
        for i in choices:
            print(i+" templates, retro direction:")
            calculate_accuracies(datasub,datasub_test,name+"template_"+i,"prod","reac",name+system, n_cpus)

        #Forward:
        for i in choices:
            print(i+" templates, forward direction:")
            calculate_accuracies(datasub,datasub_test,"forward_"+name+"template_"+i,"reac","prod",name+system, n_cpus)

if __name__ == '__main__':
    similarity_model("uspto_50k")
