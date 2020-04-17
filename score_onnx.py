import json, shutil, os
import onnxruntime
import pandas as pd
import numpy as np
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame(data={
    "SensorCH1": [5.1],
    "SensorCH2": [5.1],
    "SensorCH3": [5.1],
    "SensorCH4": [5.1],
    "SensorCH5": [5.1],
    "SensorCH6": [5.1],
    "SensorCH7": [5.1],
    "SensorCH8": [5.1],
    "SensorCH9": [5.1],
    "SensorCH10": [5.1],
    "SensorCH11": [5.1],
    "SensorCH12": [5.1],
    "SensorCH13": [5.1],
    "SensorCH14": [5.1],
    "SensorCH15": [5.1],
    "SensorCH16": [5.1],
})

output_sample = np.array([0.1])


def init():
    global session, output_name
    model_path = Model.get_model_path(model_name) 

    session = onnxruntime.InferenceSession(model_path)
    output = session.get_outputs()[0] 
    inputs = session.get_inputs()
    
@input_schema('input_data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(input_data):
    try:
        input_data= {i.name: v for i, v in zip(inputs, input_sample.values.reshape(len(inputs),1,1).astype(np.float32))}}
        result = session.run([output.name], inputs)
        result = np.argmax(np.array(result).squeeze(), axis=0)

        print('[INFO] Results was ' + json.dumps(preds))
        return preds

    except Exception as e:
        result_dict = {"error": str(e)}
    
    return result_dict