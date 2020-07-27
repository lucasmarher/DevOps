#Recoger configuracion del entorno y permisos 
from azureml.core import Workspace
ws = Workspace.from_config()

#Obtener información del experimento y los modelos publicados
from azureml.core import Experiment
experiment = Experiment(workspace=ws, name="diabetes-experiment")

#Selección del mejor modelo en base a métrica RMSE
minimum_rmse_runid = None
minimum_rmse = None

for run in experiment.get_runs():
    run_metrics = run.get_metrics()
    run_details = run.get_details()
    # each logged metric becomes a key in this returned dict
    run_rmse = run_metrics["rmse"]
    run_id = run_details["runId"]
    
    if minimum_rmse is None:
        minimum_rmse = run_rmse
        minimum_rmse_runid = run_id
    else:
        if run_rmse < minimum_rmse:
            minimum_rmse = run_rmse
            minimum_rmse_runid = run_id

#Sobre el mejor modelo, obtener una referencia del mismo
from azureml.core import Run
best_run = Run(experiment=experiment, run_id=minimum_rmse_runid)
print(best_run.get_file_names())

#Descargar el modelo
model_file_name = best_run.get_file_names()[0]
best_run.download_file(name=model_file_name)

#Cargamos el modelo en python
from sklearn.externals import joblib
clf = joblib.load(model_file_name)

#Uso del clasificador para nuevos datos (referencia)
#y_predecido = clf.predict(X) 
#Comparar x con y y calcular rmse
#rmse = math.sqrt(mean_squared_error(y_true=y_realdelosdatos, y_pred=y_predecido))
