from azureml._restclient.models.run_type_v2 import RunTypeV2
from azureml._restclient.models.create_run_dto import CreateRunDto
from azureml._restclient.experiment_client import ExperimentClient


def set_telemetry_scenario(run, scenario_type):
    try:
        # Update step run with the right traits to denote it is training or inference
        run_id = run.id
        create_run_dto = CreateRunDto(run_id=run_id, run_type_v2=RunTypeV2(traits=[scenario_type]))

        experiment_client = ExperimentClient(run.experiment.workspace.service_context,
                                             run.experiment.name,
                                             experiment_id=run.experiment.id)
        experiment_client.create_run(run_id=run_id, create_run_dto=create_run_dto)

    except Exception as e:
        print('exception happened during updating telemetry {}'.format(e))
        pass
