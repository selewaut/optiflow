========
optiflow
========


Inventory optimziation toolset

# Summary of project
This project attempts to solve the problem of inventory optimization. It implements classical approaches for simple single echelon systems such as EOQ models with its different variants (stocahstic, production, etc), then multi-echelon models, and ultimately Reinformance Learning based models for more complex systems. Most classical models are textbook classical inventory managenemt theory models.
# Feedback
Please feel free to submit any feedback or bug reports as issues on the project repository.


# Get started
## Create development environment
1. Install `tox` for project dependencies
- `pip intall tox`
2. Create development environment
- `tox d venv`
3. Activate development environment.
- `source venv/bin/activate`

After this you should be able to run the project with the `requirments_dev.txt` deps installed.