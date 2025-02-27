from prefect import flow
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta


@flow
def myflow():
    print("hello")


DeploymentSpec(
    flow=myflow,
    name="model_training-dev",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    tags=["dev"]
)

DeploymentSpec(
    flow=myflow,
    name="model_training-prod",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    tags=["prod"]
)
