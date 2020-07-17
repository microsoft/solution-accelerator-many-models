import json
import importlib  # To import pandas if needed


# Input and output sample formats

input_sample = [
    {
        "store": "Store1000", "brand": "dominicks", "model_type": "lr",
        "forecast_horizon": 5, "date_freq": "W-THU",
        "data": {
            "dates": ["2020-04-30", "2020-05-07", "2020-05-14"],
            "values": [11450, 12235, 14713]
        }
    },
    {
        "store": "Store1010", "brand": "minute.maid", "model_type": "lr",
        "forecast_horizon": 5, "date_freq": "W-THU",
        "data": {
            "dates": ["2020-04-30", "2020-05-07", "2020-05-14"],
            "values": [25692, 32976, 28610]
        }
    }
]

output_sample = [
    {
        "store": "Store1000", "brand": "dominicks", "model_type": "lr",
        "forecast_horizon": 5, "date_freq": "W-THU",
        "forecast": {
            "dates": ["2020-05-21", "2020-05-28", "2020-06-04", "2020-06-11", "2020-06-18"],
            "values": [14572, 9834, 10512, 12854, 11046]
        }
    },
    {
        "store": "Store1010", "brand": "minute.maid", "model_type": "lr",
        "forecast_horizon": 5, "date_freq": "W-THU",
        "forecast": {
            "dates": ["2020-05-21", "2020-05-28", "2020-06-04", "2020-06-11", "2020-06-18"],
            "values": [24182, 31863, 27539, 26910, 22600]
        }
    }
]


# Helper functions

def read_input(input_raw, format=True):
    ''' Read and format data received as input '''
    input_records = json.loads(input_raw)

    if format:
        input_records = [{
            'metadata': format_input_metadata(input_record),
            'data': format_input_data(input_record)
        } for input_record in input_records]

    return input_records


def format_input_metadata(input_record):
    return {k:v for k,v in input_record.items() if k != 'data'}


def format_input_data(input_record):
    pd = importlib.import_module('pandas')
    data = pd.DataFrame(input_record['data'])
    data['dates'] = pd.to_datetime(data.dates, format='%Y-%m-%d')
    return data


def format_output_record(metadata, dates, values):
    ''' Format data to be sent as one record within the output list '''
    output_record = {
        **metadata,
        "forecast": {
            "dates": [d.strftime('%Y-%d-%m') for d in dates],
            "values": values.tolist()
        }
    }
    return output_record
