#############################
## IMPORTING PACKAGES 
###########################
import json
import pandas as pd
import unidecode
import re
import ast
from pydantic import BaseModel
from typing import List, Union, Optional, Dict
from dateutil import parser as ps
import argparse
import matplotlib.pyplot as plt
from pprint import pprint

##--------------------------------------------------------------------------------
## Formating Functions
def format_str_entity(entity):
    return entity.lower().strip()

def format_date(date):
    try:
        ## Parse the date string into a datetime object
        date_obj = ps.parse(date)
        # Format the datetime object into the desired format
        formatted_date = date_obj.strftime("%d %B, %Y")
        return formatted_date
    except Exception as e:
        if date:
            return f"Error: {e}"
        else:
            return ''

def format_rate(rate):
    if isinstance(rate, float):
        rate_str = str(rate)
    rate_str = rate.replace('%', '').replace('percent', '').lower().strip()
    if '.' in rate_str:
        rate_str = rate_str.rstrip('0').rstrip('.').lstrip('0')
    return rate_str

def format_record(record):
    for entity in record.keys():
        if record[entity] is None:
            record[entity] = "nan"
        if entity == 'effective_from' or entity == 'expire_date':
            record[entity] = format_date(record[entity] if entity in record.keys() else '')
        elif entity == 'jurisdiction' or entity == 'parent_city' or entity == 'parent_county' or entity == 'parent_state':
            record[entity] = format_str_entity(record[entity] if entity in record.keys() else '')
        elif entity == 'target':
            record[entity] = format_str_entity(record[entity] if entity in record.keys() else '')
        elif entity == 'new_rate':
            record[entity] = format_rate(record[entity] if entity in record.keys() else '')
    return record

##---------------------------------------------------------------
## Handle new schema (with grounding)
def extract_schema(json_list):
    schema_list = []
    for item in json_list:
        for i in range(len(item)):
            schema = {
                "jurisdiction": item[i]['jurisdiction']['item'],
                "target": item[i]['target']['item'],
                "new_rate": item[i]['new_rate']['item'],
                "expire_date": item[i]['expire_date']['item'],
                "effective_from": item[i]['effective_from']['item'],
                "polity_type": item[i]['polity_type']['item'],
                "parent_city": item[i]['parent_city']['item'],
                "parent_county": item[i]['parent_county']['item'],
                "parent_state": item[i]['parent_state']['item']
            }
            schema_list.append(format_record(schema))
    return json.dumps(schema_list)

##---------------------------------------------------------------------------------
## Helper Functions
def extract_entities(js, entities):
    result = []
    for x in js:
        entity_values = []
        for entity in entities:
            entity_values.append(x[entity].lower() if entity in x.keys() else '')
        result.append(entity_values)
    print(result)
    return result

def ablation_study_metrics(df, entities_sequence):
    metrics = {'point': [], 'precision': [], 'recall': [], 'f1_score': []}
    current_entities = []

    for entity in entities_sequence:
        current_entities.append(entity)
        input_df = df.apply(lambda x: evaluation_per_document(x, current_entities), axis=1)
        precision, recall, f1_score = calculate_metrics(input_df)
        metrics['point'].append(f"+ {entity}")
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1_score)
        print(f"Added entity {entity}: Precision={precision}, Recall={recall}, F1-Score={f1_score}")
    return metrics


def entity_study_metrics(df, entities_sequence):
    metrics = {'point': [], 'precision': [], 'recall': [], 'f1_score': []}
    
    for entity in entities_sequence:
        input_df = df.apply(lambda x: evaluation_per_document(x, [entity]), axis=1)
        precision, recall, f1_score = calculate_metrics(input_df)
        metrics['point'].append(f"{entity}")
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1_score)
        print(f"Entity {entity}: Precision={precision}, Recall={recall}, F1-Score={f1_score}")
    return metrics

##--------------------------------------------------------------------------------
## Metrics Calcilation
def evaluation_per_document(record, entity_list):
    actual_in_predicted = 0
    actual_not_in_predicted = 0
    predicted_in_actual = 0
    predicted_not_in_actual = 0

    predicted_items = extract_entities(json.loads(record['output']), entity_list)
    actual_items = extract_entities(json.loads(record['predictions_filtered']), entity_list)

    matched_items = []
    for predicted_item in predicted_items:
        if predicted_item in actual_items:
            predicted_in_actual += 1
            matched_items.append(predicted_item)
        else:
            predicted_not_in_actual += 1

    for actual_item in actual_items:
        if actual_item in predicted_items:
            actual_in_predicted += 1
        else:
            actual_not_in_predicted += 1

    record['actual_in_predicted'] = actual_in_predicted
    record['actual_not_in_predicted'] = actual_not_in_predicted
    record['predicted_in_actual'] = predicted_in_actual
    record['predicted_not_in_actual'] = predicted_not_in_actual
    return record

def calculate_metrics(df):
    data_sum = df.sum()

    actual_in_predicted = data_sum['actual_in_predicted']
    actual_not_in_predicted = data_sum['actual_not_in_predicted']
    predicted_in_actual = data_sum['predicted_in_actual']
    predicted_not_in_actual = data_sum['predicted_not_in_actual']

    try:
        precision = predicted_in_actual / (predicted_in_actual + predicted_not_in_actual)
    except:
        precision = 0
    try:
        recall = actual_in_predicted / (actual_in_predicted + actual_not_in_predicted)
    except:
        recall = 0

    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)
    try:
        f1_score = round(2 * ((precision * recall) / (precision + recall)), 2)
    except:
        f1_score = 0

    return precision, recall, f1_score

##---------------------------------------------------------------------------------------------------------
## Visualization
def visualize_evaluation_study(metrics):
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

    metrics = entity_study_metrics(df, entities_sequence)

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

##---------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    ## Create the argument parser
    parser = argparse.ArgumentParser(description='Ablation Study CLI')

    ## Add the model_predictions_path argument
    parser.add_argument('model_predictions_path', type=str, help='Path to the model predictions file')

    ## Parse the arguments
    args = parser.parse_args()

    ## Get the model_predictions_path from the command line argument
    model_predictions_path = args.model_predictions_path

    ## Preparing the GT Vs. Extracted records
    df = pd.read_csv(model_predictions_path, sep="\t")
    print(len(df))

    # output is in the json representation
    df['output'] = df['output'].apply(lambda x: str(x).replace(u'\xa0', u''))
    df['output'] = df['output'].str.split().str.join(' ')

    df['predictions_filtered'] = df['predictions'].apply(lambda x: extract_schema(json.loads(x)))
    entities_sequence = ['jurisdiction', 'effective_from', 'effective_to', 'new_rate', 'target', 'polity_type', 'parent_state', 'parent_county', 'parent_city']
    metrics = ablation_study_metrics(df, entities_sequence)
    visualize_evaluation_study(metrics)
