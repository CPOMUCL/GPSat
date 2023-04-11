from PyOptimalInterpolation.models.base_model import BaseGPRModel

try:
    from PyOptimalInterpolation.models.gpflow_models import GPflowGPRModel, GPflowSGPRModel, GPflowSVGPModel
except Exception as e:
    print(f"Exception:\n{e}\noccurred while trying to import: GPflowGPRModel, GPflowSGPRModel")
    print("Could not load GPflow models. Check if GPflow 2 is installed")

try:
    from PyOptimalInterpolation.models.sklearn_models import sklearnGPRModel
except Exception as e:
    print(f"Exception:\n{e}\noccurred while trying to import: sklearnGPRModel")
    print("Could not load sklearn model. Check if scikit-learn is installed")

try:
    from PyOptimalInterpolation.models.vff_model import GPflowVFFModel
except Exception as e:
    print(f"Exception:\n{e}\noccurred while trying to import: GPflowVFFModel")
    print("Could not load VFF model. Check if GPflow 2 is installed")

try:
    from PyOptimalInterpolation.models.asvgp_model import GPflowASVGPModel
except Exception as e:
    print(f"Exception:\n{e}\noccurred while trying to import: GPflowASVGPModel")

