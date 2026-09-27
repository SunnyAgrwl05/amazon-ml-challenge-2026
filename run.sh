#!/bin/sh
set -e
python -m pip install -r code/business_entity_resolution/requirements.txt
python -m code.business_entity_resolution.src.train --base .
python -m code.business_entity_resolution.src.infer --base .
echo "Run the official challenge validator before uploading."
