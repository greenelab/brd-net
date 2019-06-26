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
import urllib.request
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
    print('-' * 80)

    # MetaSRA doens't like unicode quotes for some reason, so we'll strip unicode characters
    study_name = strip_unicode(study['title'])
    print('Study Name:\t{}'.format(study_name))
    print('Options:\nvalid\t\t(This study looks like it has human gene expression data)')
    print('invalid\t\t(This study does not appear to have human gene expression data)')
    print('exit\t\t(Quit annotating and save progress)')
    print('-' * 80)


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
    print('-' * 80)
    print('Rule options:')
    print('case:\t\tAll samples containing this string are cases')
    print('control:\tAll samples containing this string are controls')
    print('invalid:\tAll samples containing this string are invalid (not RNA, not human, etc)')
    print('study_info:\tPrint information about the study so it can be looked over in SRA')
    print('done:\t\tStop writing rules for the study. All samples not assigned to case or control '
          'will be saved as invalid')
    print('restart:\tClear all previously written rules for this study and start over')
    print('-' * 80)


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
    print('Querying Entrez...')

    attempts = 0
    while attempts < 10:
        try:
            result = requests.post('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', data={'db': 'sra', 'id': ids})

            # Throw an exception if we get an HTTP error
            result.raise_for_status()

            # Parse the XML returned from Entrez
            print('Parsing XML...')
            out = open('test.txt', 'w')
            out.write(strip_unicode(result.text))
            root = ET.fromstring(strip_unicode(result.text))

            return root
        except requests.exceptions.HTTPError:
            attempts += 1
            print('Downloading dataset failed {} times'.format(attempts))
            sleep(1)
    print('Too many download failures, exiting...')
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
                print('Study Accession: {}'.format(study_id))

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
                print(name.text)


def print_sample_header(root):
    '''Print which samples have been assigned to which categories'''
    print('-' * 80)
    print('Currently unassigned or invalid samples:')
    print('-' * 80)
    print_sample_names(root, 'invalid')
    print('\n')

    print('-' * 80)
    print('Cases:')
    print('-' * 80)
    print_sample_names(root, 'case')
    print('\n')

    print('-' * 80)
    print('Controls:')
    print('-' * 80)
    print_sample_names(root, 'control')
    print('\n')


def save_progress(root, rules, results_out, rules_out):
    '''Save the program's output to a file

    Arguments
    ---------
    tree: xml.etree.ElementTree
        A tree containing all the studies categorized so far
    rules: dict
        A dictionary containing the set of rules used to categorize each study
    out_file: str
        The path to save the files to

    Notes
    -----
    The tree will be saved to the disk as <out_file>.xml, and the rules will be saved
    as <out_file>_rules.json

    '''
    tree = ET.ElementTree(root)
    tree.write(results_out)

    rules_json = json.dumps(rules)
    rules_file = open(rules_out, 'w')
    json.dump(rules_json, rules_file)


def process_samples(args):
    '''Query metaSRA and separate samples based on case/control status
    
    Arguments
    ---------
    args: namespace
        Args contains all the arguments passed in from the command line

    '''
    # TODO run on all samples using pagination (UBERON:0001062)

    print('Querying metasra...')

    rule_dict = {}

    previously_seen_studies = set()

    root = ET.Element("ProgramRun")
    if args.previous_data is not None:
        tree = ET.parse(args.previous_data)
        root = tree.getroot()

        studies = root.findall('.//EXPERIMENT_PACKAGE_SET')
        for study in studies:
            previously_seen_studies.add(study.get('study_id'))

    data = None
    with urllib.request.urlopen('http://metasra.biostat.wisc.edu/api/v01/'
        'samples.json?and=UBERON:0000178&sampletype=tissue') as url:
        data = json.loads(url.read().decode())

    studies = data['studies']

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
                        save_progress(root, rule_dict, args.results_out, args.rules_out)
                        sys.exit(0)
                    else:
                        print('\n{} is not a valid option. Please try again'.format(option))
            except BaseException as e:
                # If something fails, save the progress that we had so far
                print('\n\nCaught following exception, saving progress before exiting:')
                print(e)
                

                save_progress(root, rule_dict, args.results_out, args.rules_out)
                raise

    print('Annotation complete, saving results...')
    save_progress(root, rule_dict, args.results_out, args.rules_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A script to categorize case and control data using metaSRA and Entrez')

    parser.add_argument('--results_out', help='The xml file to save the categorized data to',
                        required=True)
    parser.add_argument('--rules_out', help='The json file to which to save the rules that were used'
                        'to categorize the samples', required=True)
    parser.add_argument('--previous_data', help='The xml file output from a previous run')

    args = parser.parse_args()

    process_samples(args)
