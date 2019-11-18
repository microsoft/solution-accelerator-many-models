from azureml.core.model import Model, Dataset
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id")
parser.add_argument("--model_path")
parser.add_argument("--ds_name")
args = parser.parse_args()

print("Argument 1(model_id): %s" % args.model_id)
print("Argument 2(model_path): %s" % args.model_path)
print("Argument 3(ds_name): %s" % args.ds_name)

run = Run.get_context()
ws = None
if type(run)==_OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

    
Model.register(workspace=ws,
               model_path=args.model_path,
               model_name=args.model_id)
