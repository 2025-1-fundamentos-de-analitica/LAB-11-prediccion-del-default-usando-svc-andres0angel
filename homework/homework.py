import os
import json
import pickle
import gzip
import glob
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)


def limpiar_directorio(destino):
    if os.path.exists(destino):
        for f in glob.glob(f"{destino}/*"):
            os.remove(f)
        os.rmdir(destino)
    os.makedirs(destino)


def leer_datasets():
    df_train = pd.read_csv("./files/input/train_data.csv.zip", compression="zip")
    df_test = pd.read_csv("./files/input/test_data.csv.zip", compression="zip")
    return df_train, df_test


def limpiar_dataframe(tabla):
    datos = tabla.copy()
    datos.rename(columns={"default payment next month": "default"}, inplace=True)
    datos.drop(columns=["ID"], inplace=True)
    datos = datos[datos["MARRIAGE"] != 0]
    datos = datos[datos["EDUCATION"] != 0]
    datos["EDUCATION"] = datos["EDUCATION"].apply(lambda v: 4 if v >= 4 else v)
    return datos.dropna()


def dividir_atributos(tabla):
    x = tabla.drop(columns=["default"])
    y = tabla["default"]
    return x, y


def construir_modelo(df_x):
    cat_atrib = ["SEX", "EDUCATION", "MARRIAGE"]
    num_atrib = list(set(df_x.columns) - set(cat_atrib))

    preprocesamiento = ColumnTransformer(
        transformers=[
            ('categoricos', OneHotEncoder(handle_unknown='ignore'), cat_atrib),
            ('numericos', StandardScaler(), num_atrib),
        ],
        remainder='passthrough'
    )

    pipeline_modelo = Pipeline([
        ('pre', preprocesamiento),
        ('reduccion', PCA()),
        ('seleccion', SelectKBest(score_func=f_classif)),
        ('svc', SVC(kernel="rbf", random_state=12345, max_iter=-1)),
    ])
    return pipeline_modelo


def configurar_gridsearch(pipeline, df_x):
    parametros = {
        "reduccion__n_components": [20, df_x.shape[1] - 2],
        "seleccion__k": [12],
        "svc__kernel": ["rbf"],
        "svc__gamma": [0.1],
    }

    cv = StratifiedKFold(n_splits=10)
    score = make_scorer(balanced_accuracy_score)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring=score,
        cv=cv,
        n_jobs=-1
    )
    return grid


def exportar_modelo(ruta_modelo, modelo):
    limpiar_directorio("files/models/")
    with gzip.open(ruta_modelo, "wb") as f_out:
        pickle.dump(modelo, f_out)


def calcular_metricas(nombre, y_real, y_pred):
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, zero_division=0),
    }


def construir_confusion(nombre, y_real, y_pred):
    matriz = confusion_matrix(y_real, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(matriz[0][0]), "predicted_1": int(matriz[0][1])},
        "true_1": {"predicted_0": int(matriz[1][0]), "predicted_1": int(matriz[1][1])},
    }


def ejecutar():
    limpiar_directorio("files/output/")

    datos_train, datos_test = leer_datasets()
    datos_train = limpiar_dataframe(datos_train)
    datos_test = limpiar_dataframe(datos_test)

    x_train, y_train = dividir_atributos(datos_train)
    x_test, y_test = dividir_atributos(datos_test)

    modelo = construir_modelo(x_train)
    modelo_ajustado = configurar_gridsearch(modelo, x_train)
    modelo_ajustado.fit(x_train, y_train)

    exportar_modelo(os.path.join("files/models/", "model.pkl.gz"), modelo_ajustado)

    y_pred_train = modelo_ajustado.predict(x_train)
    y_pred_test = modelo_ajustado.predict(x_test)

    salida = [
        calcular_metricas("train", y_train, y_pred_train),
        calcular_metricas("test", y_test, y_pred_test),
        construir_confusion("train", y_train, y_pred_train),
        construir_confusion("test", y_test, y_pred_test),
    ]

    with open("files/output/metrics.json", "w", encoding="utf-8") as archivo_json:
        for entrada in salida:
            archivo_json.write(json.dumps(entrada) + "\n")


if __name__ == "__main__":
    ejecutar()
