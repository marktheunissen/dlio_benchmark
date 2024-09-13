
# Running a training

Start with building container:

    make build

Generate data using container:

    make gen

Data is now available in the host. Sync it to the S3 bucket.

    make mirror

Do training:

    make train

# Debugging

In the yaml config file, set Debug: True for debug level logging

# Notes on hacks

## fork

`RDMAV_FORK_SAFE=1` is set in env vars, due to error:

    A process has executed an operation involving a call
    to the fork() system call to create a child process.

    As a result, the libfabric EFA provider is operating in
    a condition that could result in memory corruption or
    other system errors.

- https://github.com/mlcommons/storage/issues/44
- https://github.com/mlcommons/storage/issues/62

Tried changing `multiprocessing_context: spawn` in the yaml but there were further errors that indicated lack of support in DLIO

## s3 torch connector

Tried to install via setup.py, got:

    error: Couldn't find a setup script in /tmp/easy_install-jbjegl26/s3torchconnector-1.2.5.tar.gz

Instead just pulling in directly in Dockerfile
