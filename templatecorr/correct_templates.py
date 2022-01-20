import rdkit.Chem as Chem
import pandas as pd
import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool, TimeoutError
from .extract_templates import canonicalize_mol, get_templates, get_templates_temprel
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def template_substructmatch(template1, template2, remove_brackets=True):
    """
    Computes whether template1 and template2 are substructures of each other.
    
    :param template1: Reaction smarts string
    :param template2: Reaction smarts string
    :param remove_brackets: Boolean whether templates contain brackets to make the right side unimolecular.

    :return: Tuple of substructure matches
    """

    template1_r, _ , template1_p = template1.split(">")
    template2_r, _ , template2_p = template2.split(">")
    if remove_brackets:
        if template1_r[0] == "(" and template1_r[-1] == ")":
            template1_r=template1_r[1:-1]
        if template2_r[0] == "(" and template2_r[-1] == ")":
            template2_r=template2_r[1:-1]
            
    template1_r_mol = Chem.MolFromSmarts(template1_r)
    template1_p_mol = Chem.MolFromSmarts(template1_p)
    template2_r_mol = Chem.MolFromSmarts(template2_r)
    template2_p_mol = Chem.MolFromSmarts(template2_p)

    r12 = template1_r_mol.HasSubstructMatch(template2_r_mol, useChirality=True, useQueryQueryMatches=True)
    p12 = template1_p_mol.HasSubstructMatch(template2_p_mol, useChirality=True, useQueryQueryMatches=True)
    match12 = r12 and p12
        
    r21 = template2_r_mol.HasSubstructMatch(template1_r_mol, useChirality=True, useQueryQueryMatches=True)
    p21 = template2_p_mol.HasSubstructMatch(template1_p_mol, useChirality=True, useQueryQueryMatches=True)
    match21 = r21 and p21

    if match12 and not match21:
        return (True, False)
    elif match21 and not match12:
        return (False, True)
    elif match12 and match21:
        return (True, True)
    else:
        return (False, False)

def correct_templates(template_list):
    """
    Function to correct a list of templates
    :param template_list: List of reaction templates
    :return: Corrected list of reaction templates
    """
    unique_template_list=sorted(list(set(template_list)))
    
    name_parent={}
    name_child={}
    include_list=[]
    duplicates={}   

    if len(unique_template_list)==1:
        include_list=unique_template_list
    else:
        for template in unique_template_list:
            if len(include_list)==0:
                include_list.append(template)
                continue
            used=False
            current_include_list=include_list.copy()
            for included_template in current_include_list:
                results=template_substructmatch(template,included_template)
                if results[1] and not results[0]:
                    #Need to change include_list and parent dictionary
                    if template not in include_list:
                        include_list.append(template)
                    include_list.remove(included_template)
                    name_parent[included_template]=template
                    if template not in name_child.keys():
                        name_child[template]=[included_template]
                    else:
                        name_child[template].append(included_template)
                    if included_template in name_child.keys():
                        for child in name_child[included_template]:
                            name_parent[child]=template
                            name_child[template].append(child)
                    used=True
                elif results[0] and not results[1]:
                    #Found a child of an already included template
                    name_parent[template]=included_template
                    
                    if included_template not in name_child.keys():
                        name_child[included_template]=[template]
                    else:
                        name_child[included_template].append(template)
                    used=True
                    break
                elif results[0] and results[1]:
                    if template in duplicates.keys():
                        raise ValueError("error in correction routine")
                    duplicates[template]=included_template
                    used=True
                    break
            if not used:
                include_list.append(template)
                    
    templates=[]
    for template in template_list:
        if template in duplicates.keys():
            template=duplicates[template]
        if template not in include_list:
            template=name_parent[template]
        templates.append(template)
    return templates

def correct_all_templates(data,column_name1,column_name2, n_cpus):
    """
    Computes a corrected set of templates for templates of different specificity levels 

    :param data: Pandas dataframe
    :param column_name1: Name of column with more general templates
    :param column_name2: Name of column with more specific templates
    :return: List of new templates in order of templates in dataframe
    """
    unique_templates=sorted(list(set(data[column_name1].values)))
    large_unique_templates=sorted(list(set(data[column_name2].values)))
    data["new_t"] = None
    print("...Unique templates in column",column_name1,":",len(unique_templates))
    print("...Unique templates in column",column_name2,":",len(large_unique_templates))
    print("...Correcting templates in column",column_name2)

    results = Parallel(n_jobs=n_cpus, verbose=1)(delayed(correct_loop)(data[data[column_name1]==template].copy(), column_name2, template) for template in unique_templates)
    for result in results:
        idxs, templates = result
        ctr=0
        for idx in idxs:
            data.at[idx,"new_t"]=templates[ctr]
            ctr+=1  
    new_unique_templates=set(data["new_t"].values)
    print("...Unique corrected templates in column",column_name2,":",len(new_unique_templates))
    print("")
    return list(data["new_t"].values)

def correct_loop(df,column_name2, template):
    """
    Calls correct_templates function for a set of templates where data[column_name1]==template

    :param df: Pandas dataframe.
    :param column_name2: Name of column with more specific templates
    :param template: Template

    :return: Indices of dataframe, corrected templates
    """
    templates=correct_templates(df[column_name2])
    return df.index, templates

def parallel_extract(df, nol, r, reaction_column, add_brackets=False, nproc = 20, timeout=3):
    pool = Pool(processes=nproc)
    async_results = [pool.apply_async(get_templates, (data[reaction_column][idx], data["canonical_reac_smiles"][idx], nol, r, add_brackets)) for idx in df.index]
    templates = []
    for res in tqdm.tqdm(async_results):
        try:
            templates.append(res.get(timeout))
        except TimeoutError:
            templates.append(None)
    return templates

def parallel_extract_temprel(df, nol, r, nproc = 20, timeout=3):
    pool = Pool(processes=nproc)
    async_results = [pool.apply_async(get_templates_temprel, (reaction, nol, r)) for reaction in df.to_dict(orient='records')]
    templates = []
    for res in tqdm.tqdm(async_results):
        try:
            templates.append(res.get(timeout))
        except TimeoutError:
            templates.append({'dimer_only':None, 'intra_only':None, 'necessary_reagent':None, 'reaction_id':None, 'reaction_smarts':None})
    return templates

def templates_from_file(path, reaction_column = "rxn_smiles", name="template", nproc=20, drop_extra_cols = True, data_format='csv', save=True):
    print("Reading file...")
    if data_format in ['json', 'json.gz']:
        data = pd.read_json(path+"."+data_format)
    elif data_format == 'csv':
        data = pd.read_csv(path+"."+data_format)
    elif data_format in ['pkl', 'pickle']:
        data = pd.read_pickle(path+"."+data_format)
    elif data_format == 'hdf5':
        data = pd.read_hdf(path+"."+data_format, "table")
    else:
        raise ValueError("Invalid option for data_format")
        
    print("Preprocessing reactants...")
    data["canonical_reac_smiles"] = Parallel(n_jobs=nproc, verbose=1)(delayed(canonicalize_mol)(rxn_smi,0) for rxn_smi in data[reaction_column])

    print("Extracting templates (Radius 1 with special groups)...")
    data[name] = parallel_extract(data, False, 1, reaction_column, nproc=nproc)

    print("Extracting templates (Radius 1 without special groups)...")
    data[name+"_r1"] = parallel_extract(data, True, 1, reaction_column, nproc=nproc)

    print("Extracting templates (Radius 0 without special groups)...")
    data[name+"_r0"] = parallel_extract(data, True, 0, reaction_column, nproc=nproc)

    data = data.dropna(subset=[name,name+"_r0",name+"_r1"])

    print("Hierarchically correcting templates...")
    data[name+"_r1"] = correct_all_templates(data,name+"_r0",name+"_r1", nproc)
    data[name] = correct_all_templates(data,name+"_r1",name, nproc)

    if drop_extra_cols:
        data = data.drop(columns=["canonical_reac_smiles", name+"_r0",name+"_r1"])

    if save:
        if data_format == 'csv':
            data.to_csv(path+"_corrected."+data_format, index=False)
        elif data_format == 'json':
            data.to_json(path+"_corrected."+data_format, orient='record')
        elif data_format == 'json.gz':
            data.to_json(path+"_corrected."+data_format, orient='record', compression='gzip')
        elif data_format in ['pkl', 'pickle']:
            data.to_pickle(path+"_corrected."+data_format)
        elif data_format == 'hdf5':
            data.to_hdf(path+"_corrected."+data_format, key='table')
        print("Wrote dataframe to", path+"_corrected."+data_format)
            
    return data

def templates_from_df(df, nproc = 20, reaction_column = "reaction_smiles", name="reaction_smarts"):
    data=df.copy()
    
    print("Extracting templates (Radius 1 with special groups)...")
    templates = parallel_extract_temprel(df, False, 1, nproc)
    templates = pd.DataFrame(filter(lambda x: x, templates))
    data[['dimer_only', 'intra_only', 'necessary_reagent', 'reaction_id', 'reaction_smarts']] = templates[['dimer_only', 'intra_only', 'necessary_reagent', 'reaction_id', 'reaction_smarts']].values
    
    print("Extracting templates (Radius 1 without special groups)...")
    templates = parallel_extract_temprel(df, True, 1, nproc) 
    templates = pd.DataFrame(filter(lambda x: x, templates))
    data[name+"_r1"] = templates['reaction_smarts'].values                         
    
    print("Extracting templates (Radius 0 without special groups)...")
    templates = parallel_extract_temprel(df, True, 0, nproc)
    templates = pd.DataFrame(filter(lambda x: x, templates))
    data[name+"_r0"] = templates['reaction_smarts'].values
    
    not_na_mask = data[name].notna() & data[name+"_r0"].notna() & data[name+"_r1"].notna()

    print("Hierarchically correcting templates...")
    data.loc[not_na_mask,name+"_r1"] = correct_all_templates(data[not_na_mask],name+"_r0",name+"_r1", nproc)
    data.loc[not_na_mask,name] = correct_all_templates(data[not_na_mask],name+"_r1",name, nproc)

    return data
