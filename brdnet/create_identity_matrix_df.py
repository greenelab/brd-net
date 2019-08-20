'''Based on a Z file from PLIER, create a dataframe with the same information
that keeps all genes instead of performing dimensionality reduction'''
import argparse

import numpy
import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plier_file', help="A file output by run_plier.R, typically "
                                           "named something like 'plier_10_Z.tsv'")
    parser.add_argument('out_file', help='The path to save the resulting tsv to',
                        default='../data/plier_out/plier_all_Z.tsv')
    args = parser.parse_args()

    plier_df = pandas.read_csv(args.plier_file, sep='\t')

    identity_matrix = numpy.identity(len(plier_df.index))

    out_df = pandas.DataFrame(identity_matrix, index=plier_df.index)

    out_df.to_csv(args.out_file, sep='\t', header=True, index=True, index_label=False)
