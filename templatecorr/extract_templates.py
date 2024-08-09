import rdkit.Chem as Chem
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

def canonicalize_mol(rxn_smi,split):
    """
    Canonicalizes reactant or product molecule from a smiles string
    
    :param rxn_smi: Reaction smiles string
    :param split: 0 for reactants, -1 for products

    :return: Canonicalized reactant or product smiles strings
    """

    smi=rxn_smi.split(">")[split]
    mol=Chem.MolFromSmiles(smi)
    [a.ClearProp("molAtomMapNumber") for a in mol.GetAtoms()]
    smi = Chem.MolToSmiles(mol, True)
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), True)
    return smi

def get_templates(rxn_smi, prec, no_special_groups, radius, add_brackets=True):
    """
    Extracts a template at a specified level of specificity for a reaction smiles.

    :param rxn_smi: Reaction smiles string
    :param prec: Canonical smiles string of precursor
    :param no_special_groups: Boolean whether to omit special groups in template extraction
    :param radius: Integer at which radius to extract templates
    :param add_brackets: Whether to add brackets to make template pseudo-unimolecular

    :return: Template
    """    
    #Extract:
    try:
        rxn_split = rxn_smi.split(">")
        reaction={"_id":0,"reactants":rxn_split[0],"spectator":rxn_split[1],"products":rxn_split[2]}
        template = extract_from_reaction(reaction,no_special_groups=no_special_groups,radius=radius)["reaction_smarts"]
        if add_brackets:
            template = "(" + template.replace(">>", ")>>")
    except TypeError as e:
        print("An Exception was thrown. This likely originates from a wrong rdchiral installation, try installing rdchiral_cpp==1.1.2")
        raise e
    except:
        template = None  
    #Validate:
    if template != None:
        rct = rdchiralReactants(rxn_smi.split(">")[-1])
        try:
            rxn = rdchiralReaction(template)
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except:
            outcomes =[]
        if not prec in outcomes:
            template=None
    return template

def get_templates_temprel(reaction, no_special_groups, radius):
    """
    Extracts a template at a specified level of specificity for a reaction smiles.

    :param rxn_smi: Reaction smiles string
    :param no_special_groups: Boolean whether to omit special groups in template extraction
    :param radius: Integer at which radius to extract templates

    :return: Template
    """    
    try:
        return extract_from_reaction(reaction,no_special_groups=no_special_groups,radius=radius)
    except Exception as e:
        return {
            'reaction_id': reaction['_id'],
            'error': str(e)
        }


def switch_direction(template, brackets=True):
    """Computes reversed templates.

    :param template: Reaction template
    :param brackets: Boolean whether template contains brackets to make the right side unimolecular.

    :return: Reversed template
    """
    if brackets:
        left_side=template.split(">")[0][1:-1]
        right_side=template.split(">")[-1]
        reverse_template="("+right_side+")>>"+left_side
    else:
        left_side=template.split(">")[0]
        right_side=template.split(">")[-1]
        reverse_template=right_side+">>"+left_side
    return reverse_template
