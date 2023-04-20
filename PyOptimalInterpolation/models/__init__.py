from PyOptimalInterpolation.models.base_model import BaseGPRModel

def get_model(name):
    # IMPORT MODELS THROUGH THIS FUNCTION
    model = None
    try:
        from PyOptimalInterpolation.models.gpflow_models import GPflowGPRModel, GPflowSGPRModel, GPflowSVGPModel
    except Exception as e:
        print(f"Exception:\n{e}\noccurred while trying to import: GPflowGPRModel, GPflowSGPRModel")
        print("Could not load GPflow models. Check if GPflow 2 is installed")

    if name == "GPflowGPRModel":
        from PyOptimalInterpolation.models.gpflow_models import GPflowGPRModel as model
    elif name == 'GPflowSGPRModel':
        from PyOptimalInterpolation.models.gpflow_models import GPflowSGPRModel as model
    elif name == 'GPflowSVGPModel':
        from PyOptimalInterpolation.models.gpflow_models import GPflowSVGPModel as model
    elif name == "sklearnGPRModel":
        from PyOptimalInterpolation.models.sklearn_models import sklearnGPRModel as model
    elif name == "GPflowVFFModel":
        from PyOptimalInterpolation.models.vff_model import GPflowVFFModel as model
    elif name == "GPflowASVGPModel":
        from PyOptimalInterpolation.models.asvgp_model import GPflowASVGPModel as model
    else:
        raise NotImplementedError(f"model with name: '{name}' is not implemented")

    assert model is not None

    return model
