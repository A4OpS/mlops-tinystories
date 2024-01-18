#!/bin/bash

# Pull data using DVC
dvc pull
echo "Finished pulling data"

# Execute the main command that is passed to this entrypoint
exec "$@"

# dvc push
# this training technically has to push the generated models to the remote bucket as well. #TODO