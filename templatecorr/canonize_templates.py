try:
    from rdchiral import canonicalize_transform  # Must be rdchiral_cpp drop-in

    def canonicalize_template(template, brackets=True):
        """
        Canonicalizes templates.
        
        :param template: Reaction template
        :param brackets: Boolean whether template contains brackets to make the right side unimolecular.
        
        :return: Canonicalized template
        """
        
        template_r, _ , template_p = template.split(">")
        if brackets:
            template_r=template_r[1:-1]
            template = template_r + ">>" + template_p
        template = canonicalize_transform(template)
        if brackets:
            template = "(" + template.replace(">>", ")>>")
        return template
except ImportError:

    def canonicalize_template(template, brackets=True):
        """Mock implementation if canonicalize_transform cannot be imported."""
        print("WARNING: Canonicalization code could not be loaded. Make sure to use the rdchiral C++ version (conda install -c conda-forge -c ljn917 rdchiral_cpp), only available for Unix, currently. Returning the non-canonicalized templates")
        return template
