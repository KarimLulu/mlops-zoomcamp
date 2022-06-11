REMOTE_URI="501037195096.dkr.ecr.us-east-1.amazonaws.com/duration-model"
REMOTE_TAG="latest"
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}

LOCAL_IMAGE="stream-model-duration:v1"
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}
