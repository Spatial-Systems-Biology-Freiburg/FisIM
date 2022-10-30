import json
from pydantic.json import custom_pydantic_encoder
from dataclasses import asdict
import numpy as np

from FisInMa.model import FisherResults


def _get_encoder(fsr: FisherResults):
    encoders = {
        # TODO we need to make an additional entry here
        np.ndarray: lambda x: x.tolist(),
        np.int32: lambda x: str(x),
        fsr.ode_fun.__class__.__mro__[-2]: lambda x: x.__name__,
    }
    
    # Define the encoder as a modification of the pydantic encoder
    return lambda obj: custom_pydantic_encoder(encoders, obj)
    

def json_dumps(fsr: FisherResults, **args):
    # Special encoders for any object we might come across
    if not "default" in args:
        args["default"] = _get_encoder(fsr)
    if not "indent" in args.keys():
        args["indent"] = 4
    # Return the json output as string
    return json.dumps(fsr, indent=4, **args)


def json_dump(fsr: FisherResults, out, **args):
    # Special encoders for any object we might come across
    if not "default" in args.keys():
        args["default"] = _get_encoder(fsr)
    if not "indent" in args.keys():
        args["indent"] = 4
    # Return the json output as string
    
    if type(out) is str:
        with open(out, "w") as fp:
            json.dump(fsr, fp, **args)
    else:
        json.dump(fsr, out, **args)