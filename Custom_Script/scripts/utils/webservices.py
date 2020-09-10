import json
import importlib  # To import pandas if needed


# Input and output sample formats

input_sample = [
    {
        "id": {
            "Store": "1000",
            "Brand": "dominicks"
        },
        "model_type": "lr",
        "forecast_start": "2020-05-21", "forecast_freq": "W-THU", "forecast_horizon": 5,
        "data": {
            "historical": {
                "Quantity": [11450, 12235, 14713]
            },
            "future": {
                "Price": [2.4, 2.5, 3, 3],
                "Advert": [0, 1, 1, 1]
            }
        }
    },
    {
        "id": {
            "Store": "1010",
            "Brand": "minute.maid"
        },
        "model_type": "lr",
        "forecast_start": "2020-05-21", "forecast_freq": "W-THU", "forecast_horizon": 5,
        "data": {
            "historical": {
                "Quantity": [25692, 32976, 28610]
            },
            "future": {
                "Price": [1.5, 1.5, 3.1, 3.2, 3.5],
                "Advert": [0, 0, 1, 1, 1]
            }
        }
    }
]

output_sample = [
    {
        "id": {
            "Store": "1000",
            "Brand": "dominicks"
        },
        "model_type": "lr",
        "forecast_start": "2020-05-21", "forecast_freq": "W-THU", "forecast_horizon": 5,
        "forecast": {
            "timestamps": ["2020-05-21", "2020-05-28", "2020-06-04", "2020-06-11", "2020-06-18"],
            "values": [14572, 9834, 10512, 12854, 11046]
        }
    },
    {
        "id": {
            "Store": "1010",
            "Brand": "minute.maid"
        },
        "model_type": "lr",
        "forecast_start": "2020-05-21", "forecast_freq": "W-THU", "forecast_horizon": 5,
        "forecast": {
            "dattimestampses": ["2020-05-21", "2020-05-28", "2020-06-04", "2020-06-11", "2020-06-18"],
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
    data_historical = pd.DataFrame(input_record['data'].get('historical'))
    data_future = pd.DataFrame(input_record['data'].get('future'))
    return data_historical, data_future


def format_output_record(metadata, timestamps, values):
    ''' Format data to be sent as one record within the output list '''
    all_dates = all(ts.time().isoformat() == '00:00:00' for ts in timestamps)
    ts_format = '%Y-%d-%m' if all_dates else '%Y-%d-%m %H:%M:%S'
    output_record = {
        **metadata,
        "forecast": {
            "timestamps": timestamps.strftime(ts_format).tolist(),
            "values": values.tolist()
        }
    }
    return output_record
