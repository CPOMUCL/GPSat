from PyOptimalInterpolation.models.base_model import BaseGPRModel

def get_model(name):
    # IMPORT MODELS THROUGH THIS FUNCTION
    model = None

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
    elif name == "PurePythonGPR":
        from PyOptimalInterpolation.models.pure_python_gpr import PurePythonGPR as model
    else:
        raise NotImplementedError(f"model with name: '{name}' is not implemented")

    assert model is not None

    return model
