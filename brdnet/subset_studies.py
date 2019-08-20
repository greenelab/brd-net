'''This script uses metasra to select a subset of the studies from a dataframe
created by download_categorized_data.ipynb'''
import argparse
import collections

import pandas
import requests


def study_has_all_terms(study, required_terms):
    '''Check whether the given study has all the ontology terms passed in

    Arguments
    ---------
    study: str
        The SRA study id corresponding to a given study (e.g. SRP012345)
    required_terms: tuple of str
        The terms to look for in the study

    Returns
    -------
    bool
        True if all the study matches all terms, False otherwise
    '''
    query_string = 'http://metasra.biostat.wisc.edu/api/v01/samples.json?study={}'
    query_string = query_string.format(study)

    data = requests.get(query_string).json()

    # Collect all term ids associated with the current study
    current_study_terms = data['terms']
    term_set = set()
    for term in current_study_terms:
        ids = term['dterm']['ids']
        for term_id in ids:
            term_set.add(term_id)

    for term in required_terms:
        if term not in term_set:
            return False
    return True


def subset_df(df, terms):
    '''Use MetaSRA to filter the dataframe so that it only contains studies matching all the
    terms passed in as arguments

    Arguments
    ---------
    df: pandas.DataFrame
        The dataframe to be subsetted
    terms: list of strs
        The ontology ids to be used in filtering the dataframe. For example: 'UBERON:0000955' would
        be passed in if the user only wanted samples from brains

    Returns
    -------
    subset_df: pandas.DataFrame
        The subset of df that matches all the terms
    '''
    samples = df.columns
    # Get the study id for each sample
    studies = [sample.strip().split('.')[0] for sample in samples]
    # Keep only one instance of each study id
    studies = list(set(studies))

    studies_to_keep = []
    for study in studies:
        if study_has_all_terms(study, terms):
            studies_to_keep.append(study)

    # Keep only samples from studies in studies_to_keep
    filtered_studies = get_df_subset_by_study(df, studies_to_keep)

    return filtered_studies


def get_study_counts(sample_list):
    '''Get the names of all studies in the list and the number of samples each one has

    Arguments
    ---------
    sample_list: list of strs
        A list of samples in the 'SRPxxxxx.SRRxxxxxxx' format

    Returns
    -------
    Counter
        A Counter object mapping studies to sample counts
    '''
    studies = [sample.strip().split('.')[0] for sample in sample_list]
    return collections.Counter(studies)


def balance_study_counters(plier_study_counter, classifier_study_counter, fraction):
    done_balancing = False

    plier_sample_count = sum(plier_study_counter.values())
    classifier_sample_count = sum(classifier_study_counter.values())

    total_samples = plier_sample_count + classifier_sample_count
    plier_target = int(total_samples * fraction)
    classifier_target = total_samples - plier_target

    # Go back and forth moving studies from one set to the other if it decreases the total error
    while not done_balancing:
        old_plier_count = plier_sample_count

        error = abs(plier_target - plier_sample_count)

        # We have to cast these as lists because python hates it when you modify a dictionary
        # as you are iterating over it
        plier_study_list = list(plier_study_counter.keys())
        for study in plier_study_list:
            # If we already have too few samples from plier, don't look to move any over
            plier_sample_count = sum(plier_study_counter.values())
            error = abs(plier_target - plier_sample_count)
            if plier_sample_count < plier_target:
                break

            if abs((plier_sample_count - plier_study_counter[study]) - plier_target) < error:
                # Move study from plier_study_counter to classifier_study_counter
                classifier_study_counter[study] = plier_study_counter[study]
                del plier_study_counter[study]

        classifier_study_list = list(classifier_study_counter.keys())
        for study in classifier_study_list:
            # If we already have too few samples from classifier, don't look to move any over
            classifier_sample_count = sum(classifier_study_counter.values())
            plier_sample_count = sum(plier_study_counter.values())
            error = abs(plier_target - plier_sample_count)
            if classifier_sample_count < classifier_target:
                break

            error_if_moved = abs((classifier_sample_count - classifier_study_counter[study]) -
                                 classifier_target)
            if error_if_moved < error:
                # Move study from plier_study_counter to classifier_study_counter
                plier_study_counter[study] = classifier_study_counter[study]
                del classifier_study_counter[study]

        # If we didn't move any samples between the two sets then we're done balancing
        if old_plier_count == plier_sample_count:
            done_balancing = True

    plier_study_list = plier_study_counter.keys()
    classifier_study_list = classifier_study_counter.keys()

    return plier_study_list, classifier_study_list


def get_df_subset_by_study(df, studies_to_keep):
    '''Create a dataframe containing only the samples that are from a study in studies_to_keep

    Arguments
    ---------
    df: pandas.DataFrame
        The dataframe to subset
    studies_to_keep: list of str
        The list of SRA study ids (e.g. SRP012345) to keep

    Returns
    -------
    filtered_samples: pandas.DataFrame
        The subset of df to keep
    '''
    # TODO studies_to_keep is zero, find out why
    studies_regex = '|'.join(studies_to_keep)
    filtered_samples = df.filter(regex=studies_regex)
    return filtered_samples


def rebalance(plier_healthy, plier_disease, classifier_healthy, classifier_disease, fraction=.2):
    '''Ensure that the dataframe to be put into PLIER and the dataframe to be used in
    the classifier have the the correction ratio of samples

    Arguments
    ---------
    plier_healthy: pandas.DataFrame
        The dataframe containing healthy samples to train PLIER on
    plier_disease: pandas.DataFrame
        The dataframe containing unhealthy gene expression samples to train PLIER on
    classifier_healthy: pandas.DataFrame
        The dataframe containing healthy samples to train a classifier on
    classifier_disease: pandas.DataFrame
        The dataframe containing unhealthy gene expression samples to train a classifier on
    fraction: float
        The fraction of the total number of samples that should be in the plier_df

    Returns
    -------
    new_plier_healthy: pandas.DataFrame
        The newly rebalanced version of the plier_healthy dataframe
    new_plier_disease: pandas.DataFrame
        The newly rebalanced version of the plier_disease dataframe
    new_classifier_healthy: pandas.DataFrame
        The newly rebalanced version of the classifier_healthy dataframe
    new_classifier_disease: pandas.DataFrame
        The newly rebalanced version of the classifier_disease dataframe
    '''
    # Pass studies back and forth if the change will even out the fraction more

    plier_healthy_samples = plier_healthy.columns
    plier_disease_samples = plier_disease.columns
    classifier_healthy_samples = classifier_healthy.columns
    classifier_disease_samples = classifier_disease.columns

    plier_study_counter = get_study_counts(plier_healthy_samples)
    disease_studies = [sample.strip().split('.')[0] for sample in plier_disease_samples]
    plier_study_counter.update(disease_studies)

    classifier_study_counter = get_study_counts(classifier_healthy_samples)
    disease_studies = [sample.strip().split('.')[0] for sample in classifier_disease_samples]
    classifier_study_counter.update(disease_studies)

    plier_study_list, classifier_study_list = balance_study_counters(plier_study_counter,
                                                                     classifier_study_counter,
                                                                     fraction)

    # Instead of moving studies between dataframes each time we a study is moved in
    # balance_study_counter, we can do the whole thing at once by throwing all the studies into
    # a single dataframe and creating two subsets.
    all_healthy_df = plier_healthy.join(classifier_healthy)
    all_disease_df = plier_disease.join(classifier_disease)

    new_plier_healthy_df = get_df_subset_by_study(all_healthy_df, plier_study_list)
    new_plier_disease_df = get_df_subset_by_study(all_disease_df, plier_study_list)
    new_classifier_healthy_df = get_df_subset_by_study(all_healthy_df, classifier_study_list)
    new_classifier_disease_df = get_df_subset_by_study(all_disease_df, classifier_study_list)

    return (new_plier_healthy_df, new_plier_disease_df,
            new_classifier_healthy_df, new_classifier_disease_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plier_healthy', help='The path to the file "plier_healthy.tsv" generated '
                                              'by download_categorized_data.ipynb')
    parser.add_argument('plier_disease', help='The path to the file "plier_disease.tsv" generated '
                                              'by download_categorized_data.ipynb')
    parser.add_argument('classifier_healthy', help='The path to the file "classifier_healthy.tsv" '
                                                   'generated by download_categorized_data.ipynb')
    parser.add_argument('classifier_disease', help='The path to the file "classifier_disease.tsv" '
                                                   'generated by download_categorized_data.ipynb')
    parser.add_argument('plier_base', help='The file to print the subsetted plier dataframes to. '
                                           'For example "./plier_blood" will save the dataframes '
                                           'to the current directory as "plier_blood_healthy" and '
                                           'plier_blood_unhealthy', default='plier_subset')
    parser.add_argument('classifier_base', help='The file to print the subsetted classifier '
                                                'dataframes to. For example "./classifier_blood" '
                                                'will save the dataframes to the current '
                                                'directory as "classifier_blood_healthy" and '
                                                '"classifier_blood_unhealthy"',
                                                default='classifier_subset')
    parser.add_argument('term_ids', help='The ontology IDs required to keep a sample. For '
                                         'example, if you want to keep only blood samples, pass '
                                         'in UBERON:0000178', nargs='+')
    args = parser.parse_args()

    plier_healthy_df = pandas.read_csv(args.plier_healthy, sep='\t')
    plier_disease_df = pandas.read_csv(args.plier_disease, sep='\t')
    classifier_healthy_df = pandas.read_csv(args.classifier_healthy, sep='\t')
    classifier_disease_df = pandas.read_csv(args.classifier_disease, sep='\t')

    subset_plier_healthy_df = subset_df(plier_healthy_df, args.term_ids)
    subset_plier_disease_df = subset_df(plier_disease_df, args.term_ids)
    subset_classifier_healthy_df = subset_df(classifier_healthy_df, args.term_ids)
    subset_classifier_disease_df = subset_df(classifier_disease_df, args.term_ids)

    subset_plier_healthy_df, subset_plier_disease_df, \
    subset_classifier_healthy_df, subset_classifier_disease_df = rebalance(subset_plier_healthy_df,
                                                                           subset_plier_disease_df,
                                                                      subset_classifier_healthy_df,
                                                                      subset_classifier_disease_df)

    plier_healthy_path = args.plier_base + '_healthy.tsv'
    plier_disease_path = args.plier_base + '_disease.tsv'
    classifier_healthy_path = args.classifier_base + '_healthy.tsv'
    classifier_disease_path = args.classifier_base + '_disease.tsv'

    subset_plier_healthy_df.to_csv(plier_healthy_path, sep='\t')
    subset_plier_disease_df.to_csv(plier_disease_path, sep='\t')
    subset_classifier_healthy_df.to_csv(classifier_healthy_path, sep='\t')
    subset_classifier_disease_df.to_csv(classifier_disease_path, sep='\t')
