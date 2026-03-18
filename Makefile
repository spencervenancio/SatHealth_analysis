#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = SatHealth
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	

#################################################################################
# PROJECT RULES                                                                 #
################################################################################


## Make dataset
.PHONY: data
data:
	conda run -n SatHealth python src/data/dataset.py


