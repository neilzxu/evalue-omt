#!/bin/bash

# save conda to environment.yml
conda env export --no-builds --from-history | grep -v "prefix" > environment.yml

# save pip to requirements.txt
pipdeptree --warn silence | grep -E '^\w+' > requirements.txt
