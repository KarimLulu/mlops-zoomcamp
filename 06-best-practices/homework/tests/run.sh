#!/usr/bin/env bash

cd "$(dirname "$0")"

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then 
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="batch-model-duration:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} ..
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

export AWS_DEFAULT_REGION=eu-west-2
export YEAR=2021
export MONTH=01
export S3_BUCKET_NAME=nyc-duration
export S3_ENDPOINT_URL=http://localhost:4566
export INPUT_FILE_PATTERN="s3://${S3_BUCKET_NAME}/in/${YEAR}-${MONTH}.parquet"
export OUTPUT_FILE_PATTERN="s3://${S3_BUCKET_NAME}/out/${YEAR}-${MONTH}.parquet"

docker-compose up bucket -d

sleep 3

aws --endpoint-url=${S3_ENDPOINT_URL} s3 mb s3://${S3_BUCKET_NAME}

pipenv run python upload_data.py

aws --endpoint-url=${S3_ENDPOINT_URL} s3 ls ${S3_BUCKET_NAME}/in/

docker-compose up backend

pipenv run python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down