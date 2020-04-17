import json, shutil, os
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
    # One-time initialization of PySpark and predictive model
    import pyspark
    from azureml.core.model import Model
    from pyspark.ml import PipelineModel
 
    global trainedModel
    global spark
    
    #Creating an spark session
    spark = pyspark.sql.SparkSession.builder.appName("gbt_methaneconc_ppm").getOrCreate()
    model_name = "gbt_methaneconc_ppm"
    #Loading the model
    #My model is stored in Azure Machine Learning Services. If not your case, replace accordingly
    model_path = Model.get_model_path(model_name) 
    model_unpacked = "./" + model_name

    #Unpacking archive
    shutil.unpack_archive(model_path, model_unpacked)

    #Creating the PipelineModel object from path
    trainedModel = PipelineModel.load(model_unpacked)


@input_schema('input_data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(input_data):
    if isinstance(trainedModel, Exception):
        #Loading rutine failed to load the model
        return json.dumps({{"trainedModel":str(trainedModel)}})
      
    try:
    
        #Getting the spark context
        sc = spark.sparkContext

        #Converting Pandas to Dataframe (Spark)
        input_df = spark.createDataFrame(input_data)
    
        # Compute prediction
        predictions = trainedModel.transform(input_df).collect()
 
        #Get each scored result
        preds = [x['prediction'] for x in predictions]
        
        print('[INFO] Results was ' + json.dumps(preds))
        return preds
    except Exception as e:
        print('[ERR] Exception happened: ' + str(e))
        result = 'Input ' + str(input_data) + '. Exception was: ' + str(e)
        return result
