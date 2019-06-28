'''A script to categorize case and control data using metaSRA and Entrez

Author: Ben Heil
Date Created: 6/21/19

Notes
-----
Uses metaSRA version 1.5
'''


import argparse
import json
import sys
from time import sleep
import xml.etree.ElementTree as ET

import requests


def strip_unicode(string):
    '''Remove unicode characters like nonbreaking spaces that bother parsers'''

    return ''.join([char for char in string if 0 < ord(char) < 127])


def print_study_options(study):
    '''Print the user's options for processing a study

    Arguments
    ---------
    study: str
        The name of the current study

    '''
    sys.stderr.write('{}\n'.format('-' * 80))

    # MetaSRA doens't like unicode quotes for some reason, so we'll strip unicode characters
    study_name = strip_unicode(study['title'])
    sys.stderr.write('Study Name:\t{}\n'.format(study_name))
    sys.stderr.write('Options:\nvalid\t\t(This study looks like it has human gene expression data)\n')
    sys.stderr.write('invalid\t\t(This study does not appear to have human gene expression data)\n')
    sys.stderr.write('exit\t\t(Quit annotating and save progress)\n')
    sys.stderr.write('{}\n'.format('-' * 80))


def merge_study_groups(study):
    '''Combine all study groups into a single group

    metaSRA splits samples from a study into multiple groups based on their metadata.
    This function merges all the samples from a study back into one group

    Arguments
    ---------
    study: dict
        The json-derived object containing information about a study

    Returns
    -------
    all_groups: list
        A list of dictionaries containing all experiments for a given study

    '''
    study_groups = study['sampleGroups']

    all_groups = []
    for group in study_groups:
        for experiment in group['samples']:
            all_groups.append(experiment)
    return all_groups


def print_rule_prompt():
    ''' Print the prompt giving the user their options for creating a rule'''
    sys.stderr.write('{}\n'.format('-' * 80))
    sys.stderr.write('Rule options:\n')
    sys.stderr.write('case:\t\tAll samples containing this string are cases\n')
    sys.stderr.write('control:\tAll samples containing this string are controls\n')
    sys.stderr.write('invalid:\tAll samples containing this string are invalid (not RNA, not human, etc)\n')
    sys.stderr.write('study_info:\tPrint information about the study so it can be looked over in SRA\n')
    sys.stderr.write('done:\t\tStop writing rules for the study. All samples not assigned to case or control '
                     'will be saved as invalid\n')
    sys.stderr.write('restart:\tClear all previously written rules for this study and start over\n')
    sys.stderr.write('{}\n'.format('-' * 80))


def get_study_info_as_xml(groups):
    ''' Use a list of sample ids to get information about each sample from the SRA

    Arguments
    ---------
    groups - dict
        The MetaSRA info about all the samples for a given study

    Returns
    -------
    root: xml.etree.ElementTree
        An XML object containing the sample information returned by Entrez

    '''
    ids = []
    for sample in groups:
        # Get the SRX (experiment) id
        ids.append(sample['experiments'][0]['id'])

    ids = ','.join(ids)

    # Get information about all the samples in the experiment
    sys.stderr.write('Querying Entrez...\n')

    attempts = 0
    while attempts < 10:
        try:
            result = requests.post('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', data={'db': 'sra', 'id': ids})

            # Throw an exception if we get an HTTP error
            result.raise_for_status()

            # Parse the XML returned from Entrez
            sys.stderr.write('Parsing XML...\n')
            root = ET.fromstring(strip_unicode(result.text))

            return root
        except requests.exceptions.HTTPError:
            attempts += 1
            sys.stderr.write('Downloading dataset failed {} times\n'.format(attempts))
            sleep(2 * attempts)
    sys.stderr.write('Too many download failures, exiting...\n')
    sys.exit(1)


def apply_rule(rule, root, category):
    ''' Add a category to sample denoting whether they are cases or controls based on a rule

    Arguments
    ---------
    rule: str
        A string to match against a sample's name, denoting that it belongs in the given category
    root: xml.etree.ElementTree
        An XML object containing the sample information returned by Entrez in get_study_info_as_xml
    category: str
        The label to be added to the matching BioSample objects in the tree as a category

    Returns
    -------
    root: xml.etree.ElementTree
        The tree passed in, updated according to the rule

    Notes
    -----
    apply_rule matches the rule against the the title string by determining whether rule is a
        (case_insensitive) substring of the title. For the example title 'RNA from Whole Blood'
        the rules 'RNA', 'whole blood', and 'RNA from Whole Blood' would match, but 'blood RNA'
        would not.

    '''
    samples = root.findall('.//EXPERIMENT_PACKAGE')

    for sample in samples:
        title = sample.find('.//SAMPLE/TITLE').text.lower()

        if rule.lower() in title:
            sample.set('category', category)

    return root


def set_all_samples_invalid(root):
    '''Set the category attribute for all samples in the tree to 'invalid'

    Arguments
    ---------
    root: xml.etree.ElementTree
        An XML object containing the sample information returned by Entrez in get_study_info_as_xml

    Returns
    -------
    root: xml.etree.ElementTree
        The same XML object, but with the category attribute for each BioSample set to invalid

    '''
    samples = root.findall('.//EXPERIMENT_PACKAGE')

    for sample in samples:
        sample.set('category', 'invalid')

    return root


def categorize_samples(root):
    '''Prompt the user for rules that divide the data into cases, controls, and other entries

    Arguments
    --------
    root: xml.etree.ElementTree
        An XML object containing the sample information returned by Entrez in get_study_info_as_xml

    Returns
    -------
    root: xml.etree.ElementTree
        The tree passed in, but with category attributes added to the BioSample objects
    rules: list
        A list of tuples containing the rules used to split up the data
    '''

    # The samples are all invalid by default
    root = set_all_samples_invalid(root)

    rules = []
    done_making_rules = False

    while not done_making_rules:
        print_sample_header(root)
        print_rule_prompt()

        valid_option = False
        while not valid_option:
            option = input('Input [case/control/invalid/study_info/done/restart]: ').strip().lower()
            if option == 'done':
                done_making_rules = True
                valid_option = True
            elif option == 'case':
                rule = input('Input string denoting case samples: ').strip().lower()
                rules.append(('case', rule))

                root = apply_rule(rule, root, 'case')

                valid_option = True
            elif option == 'control':
                rule = input('Input string denoting control samples: ').strip().lower()
                rules.append(('control', rule))

                root = apply_rule(rule, root, 'control')

                valid_option = True
            elif option == 'invalid':
                rule = input('Input string denoting invalid samples: ').strip().lower()
                rules.append(('invalid', rule))

                root = apply_rule(rule, root, 'invalid')

                valid_option = True
            elif option == 'restart':
                set_all_samples_invalid(root)
                rules = []
                valid_option = True
            elif option == 'study_info':
                study = root.find('.//EXPERIMENT_PACKAGE/STUDY')
                study_id = study.get('accession')
                sys.stderr.write('Study Accession: {}\n'.format(study_id))

    return root, rules


def process_valid_study(study):
    '''Ask the user for a rule to split a study into case and control samples

    Save case and control samples for a given study to case_dict and control_dict based on
    a user generated rule

    Arguments
    ---------
    study: dict
        The json-derived object containing information about a study

    Returns
    -------
    root: xml.etree.ElementTree
        The tree passed in, but with category attributes added to the BioSample objects
    rules: list
        A list of tuples containing the rules used to split up the data

    '''

    # MetaSRA splits samples into different groups; recombine them
    groups = merge_study_groups(study)

    # Get the metadata for the samples as XML
    root = get_study_info_as_xml(groups)

    root, rules = categorize_samples(root)

    return root, rules


def print_sample_names(root, category):
    ''' Print out the study titles for all Biosample objects in the list

    Arguments
    ---------
    root: xml.etree.ElementTree
        An XML object containing the sample information returned by Entrez in get_study_info_as_xml
    category: str
        The category of sample to be printed. This string corresponds to the category attribute in
        BioSample elements of root.

    '''
    samples = root.findall('.//EXPERIMENT_PACKAGE')

    for sample in samples:
        if sample.get('category') == category:
            name = sample.find('.//SAMPLE/TITLE')

            # If we can't find the name, the sample is categorized as invalid
            if name is not None:
                sys.stderr.write(name.text + '\n')


def print_sample_header(root):
    '''Print which samples have been assigned to which categories'''
    sys.stderr.write('{}\n'.format('-' * 80))
    sys.stderr.write('Currently unassigned or invalid samples:\n')
    sys.stderr.write('{}\n'.format('-' * 80))
    print_sample_names(root, 'invalid')
    sys.stderr.write('\n\n')

    sys.stderr.write('{}\n'.format('-' * 80))
    sys.stderr.write('Cases:\n')
    sys.stderr.write('{}\n'.format('-' * 80))
    print_sample_names(root, 'case')
    sys.stderr.write('\n\n')

    sys.stderr.write('{}\n'.format('-' * 80))
    sys.stderr.write('Controls:\n')
    sys.stderr.write('{}\n'.format('-' * 80))
    print_sample_names(root, 'control')
    sys.stderr.write('\n\n')


def add_skip_to_tree(root, skip):
    '''Store the information about the number of studies read so far into the tree

    Arguments
    ---------
    root: xml.etree.ElementTree
        A tree containing all the studies categorized so far
    skip: int
        The number of studies to skip when resuming

    Returns
    -------
    root: xml.etree.ElementTree
        The tree passed in, with skip added to the
    '''
    root.set('skip', str(skip))

    return root


def save_progress(root, rules, results_out, rules_out, skip):
    '''Save the program's output to a file

    Arguments
    ---------
    root: xml.etree.ElementTree
        A tree containing all the studies categorized so far
    rules: dict
        A dictionary containing the set of rules used to categorize each study
    results_out: str
        The file to save the resulting xml tree to
    rules_out: str
        The file to save the rules used to categorize the data to
    skip: int
        The number of studies to skip when resuming

    Notes
    -----
    The tree will be saved to the disk as <out_file>.xml, and the rules will be saved
    as <out_file>_rules.json

    '''
    root = add_skip_to_tree(root, skip)
    tree = ET.ElementTree(root)
    tree.write(results_out)

    with open(rules_out, 'w') as rules_file:
        json.dump(rules, rules_file)


def process_samples(args):
    '''Query metaSRA and separate samples based on case/control status

    Arguments
    ---------
    args: namespace
        Args contains all the arguments passed in from the command line

    '''
    sys.stderr.write('Querying metasra...\n')

    previously_seen_studies = set()

    root = ET.Element("ProgramRun")
    rule_dict = {}
    if args.previous_xml is not None:
        tree = ET.parse(args.previous_xml)
        root = tree.getroot()

        studies = root.findall('.//EXPERIMENT_PACKAGE_SET')
        for study in studies:
            previously_seen_studies.add(study.get('study_id'))
    if args.previous_rules is not None:
        with open(args.previous_rules) as rule_file:
            rule_dict = json.load(rule_file)

    skip = 0
    done_reading_studies = False
    while not done_reading_studies:
        data = requests.get('http://metasra.biostat.wisc.edu/api/v01/samples.json?'
                            'and=UBERON:0001062&sampletype=tissue&skip={}&limit=3'.format(skip)).json()

        if 'error' in data:
            sys.stderr.write('Error" ' + data['error'] + '\n')
            sys.exit(1)

        studies = data['studies']

        study_count = len(studies)
        if study_count == 0:
            done_reading_studies = True

        for study_object in studies:
            study = study_object['study']

            study_id = study['id']

            # Skip studies whose samples were categorized in a previous run
            if study_id in previously_seen_studies:
                continue

            processed = False

            while not processed:
                try:
                    print_study_options(study)
                    option = ''
                    while len(option) == 0:
                        option = input('Input [valid/invalid/exit]: ').strip().lower()

                        if option == 'valid':
                            subtree, rules = process_valid_study(study_object)
                            subtree.set('study_id', study_id)

                            root.append(subtree)
                            rule_dict[study_id] = rules
                            processed = True
                        # If an entire study is irrelevant for some reason, we can just continue on
                        elif option == 'invalid':
                            processed = True
                        elif option == 'exit':
                            save_progress(root, rule_dict, args.results_out, args.rules_out, skip)
                            sys.exit('Save successful')
                        else:
                            sys.stderr.write('\n{} is not a valid option. Please try again\n'.format(option))
                except BaseException as e:
                    # If something fails, save the progress that we had so far
                    sys.stderr.write('\n\nCaught following exception, saving progress before exiting:\n')
                    sys.stderr.write(str(e) + '\n')

                    save_progress(root, rule_dict, args.results_out, args.rules_out, skip)
                    raise

        skip += study_count

    sys.stderr.write('Annotation complete, saving results...\n')
    save_progress(root, rule_dict, args.results_out, args.rules_out, skip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A script to categorize case and control data using metaSRA and Entrez')

    parser.add_argument('--results_out', help='The xml file to save the categorized data to',
                        required=True)
    parser.add_argument('--rules_out', help='The json file to which to save the rules that were used'
                        'to categorize the samples', required=True)
    parser.add_argument('--previous_xml', help='The xml file output from a previous run')
    parser.add_argument('--previous_rules', help='The json file containing the rules already '
                                                 'generated by the program')

    args = parser.parse_args()

    process_samples(args)
