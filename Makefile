
DOCKER_RUN := @docker run --rm \
	-v ./data:/workspace/dlio/data \
	-v ./dlio_benchmark:/workspace/dlio/dlio_benchmark \
	--env-file env-container \
	-it dlio

.PHONY: build bash gen train mirror
build:
	@docker build -t dlio .
	@mkdir -p data/default/train
	@mkdir -p data/default/valid

bash:
	${DOCKER_RUN} bash

# Unsets the s3 config on workload.storage until we can generate and save direct to S3
gen:
	${DOCKER_RUN} dlio_benchmark workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False ~workload.storage

train:
	${DOCKER_RUN} dlio_benchmark workload=unet3d ++workload.workflow.generate_data=False ++workload.workflow.train=True

mirror:
	@mc mb -p local/unetbucket
	mc mirror --overwrite --remove ./data local/unetbucket/data
