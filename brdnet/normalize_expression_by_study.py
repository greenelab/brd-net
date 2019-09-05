import argparse

import numpy
import pandas

import utils


def save_normalized_data(normalized_df, healthy_columns,
                         healthy_original_file_path, disease_original_file_path):
    '''Save the normalized expression data as tsv files

    Arguments
    ---------
    normalized_df: pandas.DataFrame
        A pandas dataframe containing normalized gene expression data
    healthy_columns: pandas.Series
        The columns of normalized_df which contain healthy gene expression data
    healthy_original_file_path: str or Path
        The path to the file containing healthy gene expression data
    disease_original_file_path: str or Path
        The path to the file containin unhealthy gene expression data

    Returns
    -------
    None
    '''

    healthy_out_path = healthy_original_file_path.rstrip('.tsv') + '_normalized.tsv'
    disease_out_path = disease_original_file_path.rstrip('.tsv') + '_normalized.tsv'

    healthy_df = normalized_df[healthy_columns]
    # I don't know of an easier way to select all columns not in healthy_columns
    disease_df = normalized_df[normalized_df.columns.difference(healthy_columns)]

    healthy_df.to_csv(healthy_out_path, sep='\t', index_label=False)
    disease_df.to_csv(disease_out_path, sep='\t', index_label=False)


def normalize_study(df, study):
    ''' Set the mean expression of all samples in the given study to zero and the standard
    deviation to one for all genes

    Arguments
    ---------
    df: pandas.DataFrame
        The dataframe containing healthy and disease gene expression data. The expected format for
        the column names is <study_id>.<run_id>
    study: str
        The string containing the study id of samples to normalize

    Returns
    -------
    normalized_study_df: pandas.DataFrame
        A dataframe containing normalized gene expression information for the given study
    '''
    study_columns = df.columns[df.columns.str.contains(study)]
    current_df = df[study_columns]

    current_df = current_df.subtract(current_df.mean(axis=1), axis=0)

    # If there is only one sample in the study, the standard deviation won't exist
    if len(study_columns) == 1:
        return current_df

    # Using population std instead of sample makes downstream testing easier and doesn't
    # change anything
    stds = current_df.std(axis=1, ddof=0)

    # If the row has a NaN or zero standard deviation, don't try to normalize it
    stds = stds.replace(to_replace=numpy.nan, value=1)
    stds = stds.replace(to_replace=0, value=1)

    normalized_study_df = current_df.divide(stds, axis=0)

    return normalized_study_df


def normalize_by_study(df):
    ''' For each gene in each study in df, set the mean expression to zero and the
    standard deviation to one

    Arguments
    ---------
    df: pandas.DataFrame
        The dataframe containing healthy and disease gene expression data. The expected format for
        the column names is <study_id>.<run_id>

    Returns
    -------
    normalized_df: pandas.DataFrame
        A dataframe containing the normalized gene expression data
    '''

    studies = list(set(utils.get_study_list(df)))

    normalized_df = None
    for study in studies:
        normalized_study_df = normalize_study(df, study)

        if normalized_df is None:
            normalized_df = normalized_study_df
        else:
            normalized_df = normalized_df.merge(normalized_study_df, left_index=True,
                                                right_index=True)

    return normalized_df


def get_expression_df(healthy_file_path, disease_file_path):
    '''Combine the data found in healthy and disease files, and return the combined dataframe

    Arguments
    ---------
    healthy_file_path: str or Path
        The path to the file containing healthy gene expression data
    disease_file_path: str or Path
        The path to the file containin unhealthy gene expression data

    Returns
    -------
    combined_df: pandas.DataFrame
        A dataframe containing the gene expression data from both files
    healthy_columns: pandas.Series
        The list of columns in combined_df that contain healthy gene expression data
    '''
    healthy_df = pandas.read_csv(healthy_file_path, sep='\t')
    disease_df = pandas.read_csv(disease_file_path, sep='\t')

    combined_df = healthy_df.merge(disease_df, left_index=True, right_index=True)

    return combined_df, healthy_df.columns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('healthy_data', help='The path to a tsv file containing healthy gene '
                                             'expression data')
    parser.add_argument('disease_data', help='The path to a tsv file containing unhealthy gene '
                                             'expression data')

    args = parser.parse_args()

    combined_df, healthy_columns = get_expression_df(args.healthy_data, args.disease_data)

    normalized_df = normalize_by_study(combined_df)

    save_normalized_data(normalized_df, healthy_columns, args.healthy_data, args.disease_data)
