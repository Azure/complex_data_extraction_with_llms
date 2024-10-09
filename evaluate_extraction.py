# %%
import json
import pandas as pd
import unidecode
import re
import ast
import json
from pydantic import BaseModel
from typing import List, Union
from typing import Optional,List
from typing import Dict, List
import re
from dateutil import parser
import argparse

DATE_FORMAT = "%d %B, %Y"

# %%
# Normalization methods

def normalize_date(date_str):
    isdayfirst = False
    if "/" in date_str:
        isdayfirst = True
    try:
        date = parser.parse(date_str, dayfirst=isdayfirst).strftime(DATE_FORMAT) if date_str else ''
    except:
        date = ''

    return date.strip()

def normalize_jurisdiction(jurisdiction_str):
    jurisdiction_str = jurisdiction_str.split(', ')[0].lower()
    return jurisdiction_str.lower().replace('\u2013', '-').replace(' county', '').replace(' city', '').replace('school district', '').replace("*","").replace("city of ", "").strip()

'''
def normalize_rate(rate_str):
    if isinstance(rate_str, float):
        rate_str = str(rate_str)
    return rate_str.replace('%', '').replace('percent', '').lstrip('0').rstrip('0').lower().strip()
'''
def normalize_rate(rate_str):
    if isinstance(rate_str, float):
        rate_str = str(rate_str)
    rate_str = rate_str.replace('%', '').replace('percent', '').lower().strip()
    rate_str = rate_str.lstrip('0')
    if '.' in rate_str:
        rate_str = rate_str.rstrip('0').rstrip('.')
    return rate_str

def normalize_tax_type(tax_type_str):
    tax_type = tax_type_str.lower().replace(' tax', '').replace(' rate', '').replace('municipality ', '').replace(
        ' on consumer utilities', '')
    return tax_type.lower().strip()

def normalize_template(template):
    for entity in template.keys():
        if template[entity] is None:
            template[entity] = "nan"
        if entity == 'effective_from' or entity == 'expire_date':
            template[entity] = normalize_date(template[entity] if entity in template.keys() else '')
        elif entity == 'jurisdiction' or entity == 'parent_city' or entity == 'parent_county' or entity == 'parent_state':
            template[entity] = normalize_jurisdiction(template[entity] if entity in template.keys() else '')
        elif entity == 'target':
            template[entity] = normalize_tax_type(template[entity] if entity in template.keys() else '')
        elif entity == 'new_rate':
            template[entity] = normalize_rate(template[entity] if entity in template.keys() else '')
        elif entity=='scores':
            continue
        else:
            template[entity] = template[entity].lower() if entity in template.keys() else ''
    return template

#---------------------------------------------------------------

# For handling truncated outputs, collecting the valid templates
class TemplateReformater:

    @staticmethod
    def distinguish_the_templates(s: str):
        """
        this function is useful especially for models like flan-t5 where the output template doesn't have curly brackets.
        prediction is a flat list of key values.
        eg: ["jurisdiction": "new york", "new rate": "1.0", ..., "effective date": "", "jurisdiction": "mexico" ...]
        """
        s = re.sub(",\"jurisdiction\"", "}, {\"jurisdiction\"", s)
        s = re.sub("\[\"jurisdiction\"", "[{\"jurisdiction\"", s)
        s = re.sub("\]", "}]", s)
        return s


    @staticmethod
    def reformat(template: str, normalize: bool,with_score=False):
        """
        0. Check closed brackets
        1. Extract each template from input list
        2. Remove incomplete template, and remove duplicate ones
        3. Concatenate all templates into a list
        4. Return list
        """
        # check if json is closed or not
        # template = ast.literal_eval(template)
        template = template.strip()
        template = template.replace('\\"', '"')
        template = template.replace('\"', '"')
        template = re.sub("(,\s*)+", ",", template)
        compact_template = " ".join(i for i in template.split())
        #print(compact_template)
        #print("\n\n")

        # 0
        if compact_template.endswith("]"):
            # start_index, end_index = compact_template.index('[')+1, compact_template.index(']')-1
            compact_template = compact_template[1:-1] # remove [ and ]
            #print("first if")
        else:
            # start_index = compact_template.index('[') + 1
            compact_template = compact_template[1:]
            #print("else")

        if not compact_template.endswith('}'):
            compact_template = compact_template + "}"

        compact_template = re.sub(r'(?<!})\s*,\s*{', '}\g<0>', compact_template)

        # 1 split based on pattern: } followed with a "," 
        all_templates = re.split("(?<=})\s*,\s*", compact_template)
        all_templates = [t for t in all_templates if t]
        # print(all_templates)
        # print(len(all_templates))
        # print("\n\n")
        
        # 2, 3, 4
        valid_jsons = []
        invalid_jsons = []
        seen_jsons = set()
        for t in all_templates:
            #try:
            t = re.sub(r"(?<!\\)'", r'"', t)
            #print(t)
            #print("\n\n")
            parsed_json = json.loads(t.replace("\t", " "))
            # dict (mutable) cannot be used as items in a set
            json_string = json.dumps(parsed_json) 
            if json_string not in seen_jsons:
                seen_jsons.add(json_string)
                valid_jsons.append(json.loads(json_string)) # a list of dict, not a list of json strings
            # except json.JSONDecodeError:
            #     invalid_jsons.append(t)
        normalized_jsons = []
        for template in valid_jsons:
            for entity in template.keys():
                if type(template[entity]) ==  str:
                    if (template[entity].lower() in ["nan", "none"]):
                        template[entity] = ""
            if normalize:
                normalized_jsons.append(normalize_template(template))
            else:
                normalized_jsons.append(template)
        return normalized_jsons, invalid_jsons

    @staticmethod
    def cleanup_templates(output: Union[str, List[str]], normalize: bool):
        final_output = {"templates": [], "other": []}
        unique_templates = []
        if isinstance(output, str):
            output = [output]
           # print('YESS')
        #print(output)
        #print("\n\n")
        for template in output:
            valid_templates, invalid_templates = TemplateReformater.reformat(template, normalize)
            final_output["templates"].extend(valid_templates)
            final_output["other"].extend(invalid_templates)
        for template in final_output["templates"]:
            if template not in unique_templates:
                unique_templates.append(template)
        final_output["templates"] = unique_templates
        #print(final_output["templates"])
        return final_output
def extract_specific_entities(js, entities):
    result = []
    for x in js:
        entity_values = []
        for entity in entities:
            entity_values.append(x[entity].lower() if entity in x.keys() else '')
        result.append(entity_values)
    return result

def extract_entities(js):
    result = []
    for x in js:
        #print(x)
        #print("\n\n")
        effective_from = x['effective_from'] if 'effective_from' in x.keys() else ''
        effective_to = x['expire_date'] if 'expire_date' in x.keys() else ''
        jurisdiction = x['jurisdiction'] if 'jurisdiction' in x.keys() else ''
        taxed_for = x['target'] if 'target' in x.keys() else ''
        rate_value = x['new_rate'] if 'new_rate' in x.keys() else ''
        polity_type = x['polity_type'] if 'polity_type' in x.keys() else ''
        parent_city = x['parent_city'] if 'parent_city' in x.keys() else ''
        parent_county = x['parent_county'] if 'parent_county' in x.keys() else ''
        parent_state = x['parent_state'] if 'parent_state' in x.keys() else ''
        r = [effective_from.lower(), 
             effective_to.lower(), 
             jurisdiction.lower(), 
             taxed_for.lower(), 
             rate_value.lower(),
             polity_type.lower(), 
             parent_city.lower(),  
             parent_state.lower(),
             parent_county.lower(),
             ]
        result.extend([r])
    #print(result)
    #print("\n\n")
    return result

# Scores extraction scripts
def get_document_wise_stats(row):

    strict_annotated_in_predicted = 0
    strict_annotated_not_in_predicted = 0
    strict_predicted_in_annotated = 0
    strict_predicted_not_in_annotated = 0

    #print(row['predicted_entities'])
    predicted_entities = extract_entities(row['predicted_entities'])
    
    #print(predicted_entities)
    #print("\n\n")
    annotated_entities = extract_entities(row['annotated_entities'])
    #print(annotated_entities)
    matches = []
    for strict_predicted in predicted_entities:
        if strict_predicted in annotated_entities:
            strict_predicted_in_annotated += 1
            matches.append(strict_predicted)
        else:
            strict_predicted_not_in_annotated += 1

    for strict_annotated in annotated_entities:
        if strict_annotated in predicted_entities:
            strict_annotated_in_predicted += 1
        else:
            strict_annotated_not_in_predicted += 1

    row['strict_annotated_in_predicted'] = strict_annotated_in_predicted
    row['strict_annotated_not_in_predicted'] = strict_annotated_not_in_predicted
    row['strict_predicted_in_annotated'] = strict_predicted_in_annotated
    row['strict_predicted_not_in_annotated'] = strict_predicted_not_in_annotated
    #print(matches)
    return row

def get_document_entities_wise_stats(row, entities):
    strict_annotated_in_predicted = 0
    strict_annotated_not_in_predicted = 0
    strict_predicted_in_annotated = 0
    strict_predicted_not_in_annotated = 0

    predicted_entities = extract_specific_entities(row['predicted_entities'], entities)
    annotated_entities = extract_specific_entities(row['annotated_entities'], entities)

    matches = []
    for strict_predicted in predicted_entities:
        if strict_predicted in annotated_entities:
            strict_predicted_in_annotated += 1
            matches.append(strict_predicted)
        else:
            strict_predicted_not_in_annotated += 1

    for strict_annotated in annotated_entities:
        if strict_annotated in predicted_entities:
            strict_annotated_in_predicted += 1
        else:
            strict_annotated_not_in_predicted += 1

    row['strict_annotated_in_predicted'] = strict_annotated_in_predicted
    row['strict_annotated_not_in_predicted'] = strict_annotated_not_in_predicted
    row['strict_predicted_in_annotated'] = strict_predicted_in_annotated
    row['strict_predicted_not_in_annotated'] = strict_predicted_not_in_annotated
    return row

def get_metrics(input_data):
    input_df_sum = input_data.sum()

    strict_annotated_in_predicted_count = input_df_sum['strict_annotated_in_predicted']
    strict_annotated_not_in_predicted_count = input_df_sum['strict_annotated_not_in_predicted']
    strict_predicted_in_annotated_count = input_df_sum['strict_predicted_in_annotated']
    strict_predicted_not_in_annotated_count = input_df_sum['strict_predicted_not_in_annotated']

    try:
        strict_precision = strict_predicted_in_annotated_count / (strict_predicted_in_annotated_count + strict_predicted_not_in_annotated_count)
    except:
        strict_precision = 0
    try:
        strict_recall = strict_annotated_in_predicted_count / (strict_annotated_in_predicted_count + strict_annotated_not_in_predicted_count)
    except:
        strict_recall = 0

    strict_precision = round(strict_precision * 100, 2)
    strict_recall = round(strict_recall * 100, 2)
    try:
        f1_score = round(2 * ((strict_precision * strict_recall) / (strict_precision + strict_recall)), 2)
    except:
        f1_score = 0

    print("Current Precision: " + str(strict_precision))
    print("Current Recall: " + str(strict_recall))
    print("Current F1-score: " + str(f1_score))
    return strict_precision, strict_recall, f1_score



def main(model_predictions_path):
    df = pd.read_csv(model_predictions_path, sep="\t")
    print(len(df))

    # output is in the json representation
    df['output'] = df['output'].apply(lambda x: str(x).replace(u'\xa0', u''))
    df['output'] = df['output'].str.split().str.join(' ')

    df['annotated_entities'] = df['output'].apply(lambda x: TemplateReformater.cleanup_templates(x, True)['templates'])
    df['predicted_entities'] = df['predictions'].apply(lambda x: TemplateReformater.cleanup_templates(x, True)['templates'])

    input_df = df.apply(lambda x: get_document_wise_stats(x), axis=1)
    #print(input_df)
    print('Strict Metrics')
    get_metrics(input_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_predictions_path", help="Path to the model predictions file")
    args = parser.parse_args()
    main(args.model_predictions_path)

# %%
