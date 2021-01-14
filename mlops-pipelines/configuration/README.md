# MLOps Pipelines Configuration

This folder contains all the configuration files that are used inside the MLOps pipelines.

To customize the names of the AML artifacts, check `many-models-variables.yml`(many-models-variables.yml).

The rest of the parameters depend on the training method you are using: Automated ML or Custom Script. If you navigate to the corresponding folder you will find the following files:

- `script_settings.json`: settings for the training/forecasting scripts in the many models [scripts/](../../scripts/) folder.
- `train-parallelrunstep-config.yml`: configuration for the training ParallelRunStep.
- `forecast-parallelrunstep-config.yml`: configuration for the forecasting ParallelRunStep.
- `model-deployment-config-aks.yml`: configuration for the model webservice in AKS.
- `model-deployment-config-aci.yml`: configuration for the model webservice in ACI.
- `routing-deployment-config-aks.yml`: configuration for the routing webservice in AKS.
- `routing-deployment-config-aci.yml`: configuration for the routing webservice in ACI.

## Customizing script settings

The training and forecasting scripts many models (located in the corresponding [scripts/](../../scripts/) folder) receive a settings file as a parameter that contain all the relevant settings related to the dataset format and the training configuration.

This file is called `script_settings.json`, and the contents are different depending on the training method you are using: Automated ML or Custom Script.

## Customizing ParallelRunStep config

All parameters defined in the `-parallelrunstep-config.yml` files are passed directly to the [`ParallelRunConfig`](https://docs.microsoft.com/python/api/azureml-pipeline-steps/azureml.pipeline.steps.parallelrunconfig) build during the creation of the AML Pipeline.

There you can specify parameters like:

- `mini_batch_size`
- `run_invocation_timeout`
- `error_threshold`
- `process_count_per_node`
- `node_count`

## Customizing container configuration

All four `-deployment-config-` files follow the same pattern. There are three main sections in the YAML file:

- `computeType`: can be `AKS` or `ACI`.
- `containerResourceRequirements`: parameters that will be used to configure the compute.
- `environmentVariables`: additional environment variables that will be set in the container. These can be used to enable concurrency in the container. Read [the next section](#enabling-concurrency-in-container) for more information about this.

All parameters defined under `containerResourceRequirements` are passed directly to the corresponding `deploy_configuration()` method of the azureml SDK.

- AKS: [`AksWebservice.deploy_configuration()`](https://docs.microsoft.com/python/api/azureml-core/azureml.core.webservice.akswebservice?view=azure-ml-py#deploy-configuration-autoscale-enabled-none--autoscale-min-replicas-none--autoscale-max-replicas-none--autoscale-refresh-seconds-none--autoscale-target-utilization-none--collect-model-data-none--auth-enabled-none--cpu-cores-none--memory-gb-none--enable-app-insights-none--scoring-timeout-ms-none--replica-max-concurrent-requests-none--max-request-wait-time-none--num-replicas-none--primary-key-none--secondary-key-none--tags-none--properties-none--description-none--gpu-cores-none--period-seconds-none--initial-delay-seconds-none--timeout-seconds-none--success-threshold-none--failure-threshold-none--namespace-none--token-auth-enabled-none--compute-target-name-none--cpu-cores-limit-none--memory-gb-limit-none-)

- ACI: [`AciWebservice.deploy_configuration()`](https://docs.microsoft.com/python/api/azureml-core/azureml.core.webservice.aciwebservice?view=azure-ml-py#deploy-configuration-cpu-cores-none--memory-gb-none--tags-none--properties-none--description-none--location-none--auth-enabled-none--ssl-enabled-none--enable-app-insights-none--ssl-cert-pem-file-none--ssl-key-pem-file-none--ssl-cname-none--dns-name-label-none--primary-key-none--secondary-key-none--collect-model-data-none--cmk-vault-base-url-none--cmk-key-name-none--cmk-key-version-none--vnet-name-none--subnet-name-none-)

For example:

- `cpu_cores`
- `memory_gb`
- `autoscale_enabled`
- `autoscale_min_replicas`
- `autoscale_max_replicas`
- `scoring_timeout_ms`
- `max_request_wait_time`

## Enabling concurrency in container

As stated in the [documentation](https://docs.microsoft.com/python/api/azureml-core/azureml.core.webservice.akswebservice?view=azure-ml-py#deploy-configuration-autoscale-enabled-none--autoscale-min-replicas-none--autoscale-max-replicas-none--autoscale-refresh-seconds-none--autoscale-target-utilization-none--collect-model-data-none--auth-enabled-none--cpu-cores-none--memory-gb-none--enable-app-insights-none--scoring-timeout-ms-none--replica-max-concurrent-requests-none--max-request-wait-time-none--num-replicas-none--primary-key-none--secondary-key-none--tags-none--properties-none--description-none--gpu-cores-none--period-seconds-none--initial-delay-seconds-none--timeout-seconds-none--success-threshold-none--failure-threshold-none--namespace-none--token-auth-enabled-none--compute-target-name-none--cpu-cores-limit-none--memory-gb-limit-none-), the `replica_max_concurrent_requests` parameter should be left unchanged to the default value of 1.

However, if you want to enable concurrency in the containers you can follow this instructions. This can be useful to remove possible bottlenecks in the routing webservice for heavy workloads. **Please notice this is a preview feature, use it at your discretion**.

1. Under `containerResourceRequirements`, set `replica_max_concurrent_requests` to the number of worker processes you want to enable in the container.

2. Add the following environment variables to the `environmentVariables` section in the config file:

- `WORKER_COUNT`: set to the same as `replica_max_concurrent_requests`. This is the number of worker processes (gunicorn+flask) that will be spun up.
- `MKL_NUM_THREADS`, `OMP_NUM_THREADS`: number of threads per worker. These parameters are designed to protect neighbour containers, as by default they are set to the number of CPU cores on the machine. You need to check what is relevant for your ML framework.
- `WORKER_PRELOAD`: `true` flag can be set to enable shared memory (might cause issues with some models, specially Tensorflow/Keras).

If you have a use case that could benefit from this and need any further guidance please contact Azure Machine Learning Product Group.
