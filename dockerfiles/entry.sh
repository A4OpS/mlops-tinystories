#!/bin/bash

# Pull data using DVC
dvc pull
echo "Finished pulling data"

# Execute the main command that is passed to this entrypoint
exec "$@"
