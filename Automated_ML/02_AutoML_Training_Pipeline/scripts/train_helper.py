import argparse
import os
import pickle


class MetadataFileHandler:

    # Metadata file names
    ARGS_FILE_NAME = "args.pkl"
    AUTOML_SETTINGS_FILE_NAME = "automl_settings.pkl"
    LOGS_FILE_NAME = "logs.pkl"
    RUN_DTO_FILE_NAME = "run_dto.pkl"

    def __init__(self, data_dir):
        # Directory where metadata files live
        self.data_dir = data_dir

        # Full paths to metadata files
        self._args_file_path = os.path.join(self.data_dir, self.ARGS_FILE_NAME)
        self._automl_settings_file_path = os.path.join(self.data_dir, self.AUTOML_SETTINGS_FILE_NAME)
        self._logs_file_path = os.path.join(self.data_dir, self.LOGS_FILE_NAME)
        self._run_dto_file_name = os.path.join(self.data_dir, self.RUN_DTO_FILE_NAME)

    def delete_logs_file_if_exists(self):
        if not os.path.exists(self._logs_file_path):
            return
        os.remove(self._logs_file_path)

    def load_automl_settings(self):
        return self.load_obj_from_disk(self._automl_settings_file_path)

    def load_args(self):
        return self.load_obj_from_disk(self._args_file_path)

    def load_logs(self):
        return self.load_obj_from_disk(self._logs_file_path)

    def load_run_dto(self):
        return self.load_obj_from_disk(self._run_dto_file_name)

    def write_args_to_disk(self, args):
        self.serialize_obj_to_disk(args, self._args_file_path)

    def write_automl_settings_to_disk(self, automl_settings):
        self.serialize_obj_to_disk(automl_settings, self._automl_settings_file_path)

    def write_logs_to_disk(self, logs):
        self.serialize_obj_to_disk(logs, self._logs_file_path)

    def write_run_dto_to_disk(self, run_dto):
        self.serialize_obj_to_disk(run_dto, self._run_dto_file_name)

    @classmethod
    def load_obj_from_disk(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def serialize_obj_to_disk(cls, obj, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)


class TrainUtil:

    @staticmethod
    def str2bool(v):

        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
