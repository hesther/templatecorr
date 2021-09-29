import rdkit.Chem as Chem

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

