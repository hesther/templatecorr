import os
import re
import pandas as pd
import hashlib
from joblib import Parallel, delayed
from temprel.templates.validate import validate_template
from .correct_templates import templates_from_df
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def create_hash(pd_row):
    return hashlib.md5(pd_row.to_json().encode()).hexdigest()

def unmap(smarts):
    return re.sub(r':[0-9]+]', ']', smarts)

def count_products(smiles):
    return 1 + smiles.count('.')

try:
    from temprel.templates.filter import filter_by_bond_edits
    from temprel.rdkit import fix_spectators
    
    def templates_from_reactions(df, output_prefix=None, nproc=8, timeout=3, filter_fn=filter_by_bond_edits, filter_kwargs={}):
        """Version with filter (proprietary gitlab version)"""
    
        assert output_prefix is not None, 'Please specify a directory for the output files'

        if not os.path.exists(output_prefix):
            os.makedirs(output_prefix)

        out_json = output_prefix + 'templates.df.json.gz'

        assert type(df) is pd.DataFrame
        assert 'reaction_smiles' in df.columns
        df['reaction_smiles'] = Parallel(n_jobs=nproc, verbose=1)(
            delayed(fix_spectators)(rsmi) for rsmi in df['reaction_smiles']
        )
        rxn_split = df['reaction_smiles'].str.split('>', expand=True)
        df[['reactants', 'spectators', 'products']] = rxn_split.rename(
            columns={
                0: 'reactants',
                1: 'spectator',
                2: 'products'
            }
        )
        if '_id' not in df.columns:
            df['_id'] = df.apply(create_hash, axis=1)
        num_products = df['products'].apply(count_products)
        df = df[num_products==1]
        templates = templates_from_df(df, nproc)
        keep_cols = ['dimer_only', 'intra_only', 'necessary_reagent', 'reaction_id', 'reaction_smarts']
        df = df.merge(templates[keep_cols], left_on='_id', right_on='reaction_id')
        df = df.dropna(subset=['reaction_smarts'])
        valid = Parallel(n_jobs=nproc, verbose=1)(
            delayed(validate_template)(template) for template in df.to_dict(orient='records')
        )
        df = df[valid]

        df = filter_fn(df, nproc=nproc, **filter_kwargs)

        df['unmapped_template'] = df['reaction_smarts'].apply(unmap)
        unique_templates = pd.DataFrame(
            df['unmapped_template'].unique(), columns=['unmapped_template']
        ).reset_index()
        df = df.merge(unique_templates, on='unmapped_template')
        df = df.drop_duplicates(subset=['products', 'index'])
        df['count'] = df.groupby('index')['index'].transform('count')
        if out_json[-2] == 'gz':
            df.to_json(out_json, compression='gzip')
        else:
            df.to_json(out_json)
        return df

except ImportError:
    
    from temprel.rdkit import remove_spectating_reactants
    
    def templates_from_reactions(df, output_prefix=None, nproc=8, timeout=3):
        """Version without filter (open-source gitlab version)"""
    
        assert output_prefix is not None, 'Please specify a directory for the output files'

        if not os.path.exists(output_prefix):
            os.makedirs(output_prefix)

        out_json = output_prefix + 'templates.df.json.gz'

        assert type(df) is pd.DataFrame
        assert 'reaction_smiles' in df.columns
        df['reaction_smiles'] = Parallel(n_jobs=nproc, verbose=1)(
        delayed(remove_spectating_reactants)(rsmi) for rsmi in df['reaction_smiles']
        )
        rxn_split = df['reaction_smiles'].str.split('>', expand=True)
        df[['reactants', 'spectators', 'products']] = rxn_split.rename(
            columns={
                0: 'reactants',
                1: 'spectator',
                2: 'products'
            }
        )
        if '_id' not in df.columns:
            df['_id'] = df.apply(create_hash, axis=1)
        num_products = df['products'].apply(count_products)
        df = df[num_products==1]
        templates = templates_from_df(df, nproc)
        keep_cols = ['dimer_only', 'intra_only', 'necessary_reagent', 'reaction_id', 'reaction_smarts']
        df = df.merge(templates[keep_cols], left_on='_id', right_on='reaction_id')
        df = df.dropna(subset=['reaction_smarts'])
        valid = Parallel(n_jobs=nproc, verbose=1)(
            delayed(validate_template)(template) for template in df.to_dict(orient='records')
        )
        df = df[valid]

        df['unmapped_template'] = df['reaction_smarts'].apply(unmap)
        unique_templates = pd.DataFrame(
            df['unmapped_template'].unique(), columns=['unmapped_template']
        ).reset_index()
        df = df.merge(unique_templates, on='unmapped_template')
        df = df.drop_duplicates(subset=['products', 'index'])
        df['count'] = df.groupby('index')['index'].transform('count')
        if out_json[-2] == 'gz':
            df.to_json(out_json, compression='gzip')
        else:
            df.to_json(out_json)
        return df
