import json
import pandas as pd


# Helper functions
def read_input(input_raw, format=True):
    ''' Read and format data received as input '''
    input_records = json.loads(input_raw)
    if not format:
        return input_records

    records = []
    for input_record in input_records:
        metadata = format_input_metadata(input_record)
        data = format_input_data(input_record, metadata)
        records.append({
            'metadata': metadata,
            'data': data
        })

    return records


def format_input_metadata(input_record):
    return {k: v for k, v in input_record.items() if k != 'data'}


def format_input_data(input_record, metadata):
    data = pd.read_json(input_record['data'])
    time_column_name = metadata['time_column_name']
    data[time_column_name] = pd.to_datetime(data[time_column_name])
    return data
