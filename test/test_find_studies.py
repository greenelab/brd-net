from unittest import TestCase
import xml.etree.ElementTree as ET

import pytest

from brdnet import find_studies


group1 = {'samples': ['a']}
group2 = [{'samples': ['c', 'd']}, {'samples': 'e'}]

study1_in = {'sampleGroups': [group1]}
study1_expected = ['a']
study2_in = {'sampleGroups': [{'samples': []}]}
study2_expected = []
study3_in = {'sampleGroups': group2}
study3_expected = ['c', 'd', 'e']
@pytest.mark.parametrize('test_input,expected', [(study1_in, study1_expected),
                                                 (study2_in, study2_expected),
                                                 (study3_in, study3_expected),
                                                ])
def test_merge_study_groups(test_input, expected):
    merged_input = find_studies.merge_study_groups(test_input)

    case = TestCase()
    case.assertCountEqual(merged_input, expected)


case = '<WRAPPER><EXPERIMENT_PACKAGE category="invalid"><SAMPLE><TITLE>Tumor</TITLE></SAMPLE></EXPERIMENT_PACKAGE></WRAPPER>'
control = '<WRAPPER><EXPERIMENT_PACKAGE category="invalid"><SAMPLE><TITLE>Non-Tumor</TITLE></SAMPLE></EXPERIMENT_PACKAGE></WRAPPER>'
@pytest.mark.parametrize('test_xml,test_category,test_rule,expected', [(case, 'case', 'tumor', 'case'),
                                                        (case, 'case', 'non-tumor', 'invalid'),
                                                        (control, 'control', 'non-tumor', 'control'),
                                                        (control, 'invalid', 'non-tumor', 'invalid'),
                                                        ])
def test_apply_rule(test_xml, test_category, test_rule, expected):
    element = ET.fromstring(test_xml)
    result = find_studies.apply_rule(test_rule, element, test_category)

    print(ET.dump(result))
    applied_category = result.find('.//EXPERIMENT_PACKAGE').get('category')

    assert applied_category == expected
