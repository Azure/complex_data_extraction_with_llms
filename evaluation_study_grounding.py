import json
import pandas as pd
import unidecode
import re
import ast
from pydantic import BaseModel
from typing import List, Union, Optional, Dict
from dateutil import parser
import argparse
import matplotlib.pyplot as plt

DATE_FORMAT = "%d %B, %Y"


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
## Handle new schema (with grounding and confidence)
def extract_schema(json_list):
    schema_list = []

    for item in json_list:
        for i in range(len(item)):

            schema = {
                "jurisdiction": item[i]['jurisdiction']['item'],
                "target": item[i]['target']['item'],
                "new_rate": normalize_rate(item[i]['new_rate']['item']),
                "expire_date": item[i]['expire_date']['item'],
                "effective_from": normalize_date(item[i]['effective_from']['item']),
                "polity_type": item[i]['polity_type']['item'],
                "parent_city": item[i]['parent_city']['item'],
                "parent_county": item[i]['parent_county']['item'],
                "parent_state": item[i]['parent_state']['item']
            }
            schema_list.append(schema)
    return json.dumps(schema_list)

#--------------------------------------------------------------
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
    def reformat(template: str, normalize: bool, with_score=False):
        """
        0. Check closed brackets
        1. Extract each template from input list
        2. Remove incomplete template, and remove duplicate ones
        3. Concatenate all templates into a list
        4. Return list
        """
        # check if json is closed or not
        template = template.strip()
        template = template.replace('\\"', '"')
        template = template.replace('\"', '"')
        template = re.sub("(,\s*)+", ",", template)
        compact_template = " ".join(i for i in template.split())

        # 0
        if compact_template.endswith("]"):
            compact_template = compact_template[1:-1] # remove [ and ]
        else:
            compact_template = compact_template[1:]

        if not compact_template.endswith('}'):
            compact_template = compact_template + "}"

        compact_template = re.sub(r'(?<!})\s*,\s*{', '}\g<0>', compact_template)

        # 1 split based on pattern: } followed with a "," 
        all_templates = re.split("(?<=})\s*,\s*", compact_template)
        all_templates = [t for t in all_templates if t]

        # 2, 3, 4
        valid_jsons = []
        invalid_jsons = []
        seen_jsons = set()
        for t in all_templates:
            t = re.sub(r"(?<!\\)'", r'"', t)
            parsed_json = json.loads(t.replace("\t", " "))
            # dict (mutable) cannot be used as items in a set
            json_string = json.dumps(parsed_json) 
            if json_string not in seen_jsons:
                seen_jsons.add(json_string)
                valid_jsons.append(json.loads(json_string)) # a list of dict, not a list of json strings
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

        for template in output:
            valid_templates, invalid_templates = TemplateReformater.reformat(template, normalize)
            final_output["templates"].extend(valid_templates)
            final_output["other"].extend(invalid_templates)
        for template in final_output["templates"]:
            if template not in unique_templates:
                unique_templates.append(template)
        final_output["templates"] = unique_templates

        return final_output

def extract_entities(js, entities):
    result = []
    for x in js:
        entity_values = []
        for entity in entities:
            entity_values.append(x[entity].lower() if entity in x.keys() else '')
        result.append(entity_values)
    return result

# Scores extraction scripts
def get_document_wise_stats(row, entities):
    strict_annotated_in_predicted = 0
    strict_annotated_not_in_predicted = 0
    strict_predicted_in_annotated = 0
    strict_predicted_not_in_annotated = 0

    predicted_entities = extract_entities(row['predicted_entities'], entities)
    annotated_entities = extract_entities(row['annotated_entities'], entities)

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

    return strict_precision, strict_recall, f1_score

def ablation_study(df, entities_sequence):
    metrics = {'point': [], 'precision': [], 'recall': [], 'f1_score': []}
    current_entities = []

    for entity in entities_sequence:
        current_entities.append(entity)
        input_df = df.apply(lambda x: get_document_wise_stats(x, current_entities), axis=1)
        precision, recall, f1_score = get_metrics(input_df)
        metrics['point'].append(f"+ {entity}")
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1_score)
        print(f"Added entity {entity}: Precision={precision}, Recall={recall}, F1-Score={f1_score}")

    return metrics
def entity_study(df, entities_sequence):
    metrics = {'point': [], 'precision': [], 'recall': [], 'f1_score': []}
    

    for entity in entities_sequence:
        input_df = df.apply(lambda x: get_document_wise_stats(x, [entity]), axis=1)
        precision, recall, f1_score = get_metrics(input_df)
        metrics['point'].append(f"{entity}")
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1_score)
        print(f"Entity {entity}: Precision={precision}, Recall={recall}, F1-Score={f1_score}")

    return metrics
# Create the argument parser
parser = argparse.ArgumentParser(description='Ablation Study CLI')

# Add the model_predictions_path argument
parser.add_argument('model_predictions_path', type=str, help='Path to the model predictions file')

# Parse the arguments
args = parser.parse_args()

# Get the model_predictions_path from the command line argument
model_predictions_path = args.model_predictions_path

# Rest of the code
df = pd.read_csv(model_predictions_path, sep="\t")
print(len(df))

# output is in the json representation
df['output'] = df['output'].apply(lambda x: str(x).replace(u'\xa0', u''))
df['output'] = df['output'].str.split().str.join(' ')

df['annotated_entities'] = df['output'].apply(lambda x: TemplateReformater.cleanup_templates(x, True)['templates'])
df['predictions_filtered'] = df['predictions'].apply(lambda x: extract_schema(json.loads(x)))
df['predicted_entities'] = df['predictions_filtered'].apply(lambda x: TemplateReformater.cleanup_templates(x, True)['templates'])
#entities_sequence = ['jurisdiction', 'effective_from', 'effective_to', 'polity_type', 'parent_state', 'parent_county', 'target', 'new_rate']
entities_sequence = ['jurisdiction', 'effective_from', 'effective_to', 'new_rate', 'target', 'polity_type', 'parent_state', 'parent_county', 'parent_city']
metrics = ablation_study(df, entities_sequence)

# Plotting the metrics
points = metrics['point']
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1_score']

plt.figure(figsize=(10, 6))
x = range(len(points))

plt.bar(x, precision, width=0.2, label='Precision', align='center')
plt.bar(x, recall, width=0.2, label='Recall', align='edge')
plt.bar(x, f1_score, width=0.2, label='F1-Score', align='center')

plt.xlabel('Points')
plt.ylabel('Metrics')
plt.title('Ablation Study Metrics')
plt.xticks(x, points, rotation='vertical')
plt.legend()
plt.tight_layout()
plt.show()

metrics = entity_study(df, entities_sequence)

# Plotting the metrics
points = metrics['point']
precision = metrics['precision']
recall = metrics['recall']
f1_score = metrics['f1_score']

plt.figure(figsize=(10, 6))
x = range(len(points))

plt.bar(x, precision, width=0.2, label='Precision', align='center')
plt.bar(x, recall, width=0.2, label='Recall', align='edge')
plt.bar(x, f1_score, width=0.2, label='F1-Score', align='center')

plt.xlabel('Points')
plt.ylabel('Metrics')
plt.title('Entity Study Metrics')
plt.xticks(x, points, rotation='vertical')
plt.legend()
plt.tight_layout()
plt.show()