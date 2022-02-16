from templatecorr import  templates_from_file
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract and hierarchically correct templates')
    parser.add_argument('--path', dest='path')
    parser.add_argument('--reaction_column', dest='reaction_column', default='rxn_smiles')
    parser.add_argument('--name', dest='name', default='template')
    parser.add_argument('--nproc', dest='nproc', type=int, default=20)
    parser.add_argument('--keep_extra_cols', dest='keep_extra_cols', action='store_true', default=False)
    parser.add_argument('--data_format', dest='data_format', default='csv')
    
    return parser.parse_args()
        
if __name__ == "__main__":
    args = parse_arguments()
    templates_from_file(path=args.path,
                        reaction_column = args.reaction_column,
                        name=args.name,
                        nproc=args.nproc,
                        drop_extra_cols = not args.keep_extra_cols,
                        data_format=args.data_format,
                        save=True)
