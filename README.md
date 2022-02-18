# # Segmentacion-de-documentos-digitales

A continuación se presenta una guia de como entrenar los modelos presentes en este repositorio asi como al importancia de los diferentes ficheros que lo componen y como ejecutarlos.


# Ficheros
En esta sección se describen los ficheros de mayor importancia que componen el repositorio.

|        Fichero/Carpeta        | Función                          |
|----------------|-------------------------------|
|`Main.py` | Ejecutar este fichero empezará el proceso de entrenamiento, validación y test de un modelo, también guardará un checkpoint por cada Epoch de entrenamiento del modelo y registrará los resultados obtenidos con la herramienta de SummaryWriter.
|`train_validation_test.py` |Contiene todos los bucles de entrenamiento, validación y test, ya sean estas ejecutadas de forma secuencial, paralelizada o aplicando ténicas como Adaptative Learning Rate.       |
|`test_tobaco800.py`          | Lanza una prueba con el modelo especificado en el dataset de tobacco800 y registra los resultados usando SUmmaryWritter.            |
|`Models`          |En esta carpeta se encuentran todos los modelos |
|` H5DFDataset` | En esta carpeta se encuentra la clase dataset que se usa para cargar los ficheros H5DF que componen el dataset|
|`Main_conf_files`|En esta carpeta se encuentran todos los JSON que contienen las diferentes configuraciones para cada prueba|
|`CreateDataset`|En esta carpeta se encuentran todos los scripts necesarios para crear un dataset formado por ficheros H5DF a partir de imagenes .png y ficheros .txt|

# Entrenar modelo
Para la entrenamiento de un modelo los pasos ha seguir han de ser los siguientes:
1. Crear un JSON que determine como ha de ser el entrenamiento del modelo. 
2. Ejecutar el script Main.py de la forma `python Main.py Config.json` 

El JSON que determinará la ejecución debe contener las siguientes palabras claves:
 ```yaml
{
"Parallel": 1 para indicar que se ejecutará de forma paralelizada 0 
para indicar ejecución secuencial,

"Version": 1 para indicar que se quiere hacer uso de los modelos que
no implementan capas Densas 2 para indicar que se quiere hacer uso
de los modelos que haces uso de capas densas,

"BertCalls": Numero de veces que se va a llamar a BERT,

"AdLR_SLTR": 1 para aplicar Adaptative Learning Rate y Slanded 
Triangular Schedule y 0 para no aplicar,

"num_pages": Numero de páginas a introducir al modelo,

"name_efficient": nombre del modelo de efficientnet que se quiere
usar siendo este des de efficientnet-b0 a efficientnet-b7 (el codigo intenta cargar los pesos de efficientnet por lo que será necesario que estos se encuentren presentes en una carpeta llamada Weights),

"num_GPUs": Número de GPUs que se van a usar para entrenar el modelo,

"LR_BERT": Learning Rate que se aplica a BERT,

"LR_EFFICIENT": Learning Rate que se aplica a EfficientNet,

"LR_BASE": Learning Rate que se aplica a todas las capas que no 
pertenezcan ni a BERT ni a EfficientNet,

"directory_h5df_files": Directorio donde se encuentran las carpetas 
train, validation y test que contienen ficheros h5df de BigTobacco,

"directory_tobacco800": Directorio donde se encuentran las carpetas 
train, validation y test que contienen ficheros h5df de Tobacco800,

"BATCH": Batch que se desea usar,

"workers": Workers usados en la carga del dataset,

"EPOCH": Número de Epochs,

"decay_factor": Decay factor aplicado al LR de BERT (la formula
aplicada es LR_BERT*decay_factor^(k) entendiendo k como la 
profundidad en capas)

"BETAS": Betas aplicadas al optimizador,

"num_features": Numero de neuronas densas para procesar la salida
obtenida de EfficientNet (solo usado en modelos que hacen uso de capa 
Densa),

"feature_concatenation": Numero de neuronas densas para procesar
la concatenación de las llamadas a BERT y EfficientNet(solo usado 
en modelos que hacen uso de capa Densa).
}
```


# Testear modelo con Tobacco800
Para hacer el test de un modelo con Tobaco800 los pasos ha seguir han de ser los siguientes:

1.  Seleccionar el mismo JSON que se ha usado para entrenar el modelo llamando al fichero Main.py.
2.  Ejecutar el script  test_tobacco800.py  de la forma  `python test_tobacco800.py [args]`. Los argumentos que se pueden introducir en la llamada son los siguientes:
	* `tobacco800_conf`: Indicar el fichero JSON usado para el entrenamiento (NECESARIO) [string].
	* `select_epoch`: Indicar la epoch de la cual se desea cargar el checkpoint, de no estar presente cogera la epoch con mejor resultado [integer].
	* `filtered`: Indica si el checkpoint que se quiere cargar es de un modelo obtenido de entrenar con BigTobacco después de filtrar los datos, por defecto esta en falso [bool].
	* `fine_tune`: Indica si se quiere usar una porcion de Tobacco800 para entrenar y validar el modelo, por defecto esta en falseo [bool].
	* `full_train`: Indica si se quiere coger un modelo y sin cargar ningun checkpoint hacer un entrenamiento, validación y test en Tobacco800. Los parametros usados serían escogidos a partir del JSON, por defecto esta en falseo [bool].


# Crear un dataset
En caso de que fuera necesario los scripts dentro de la carpeta llamada CreateDataset sirven para crear un dataset dadas imagenes y ficheros .txt distribuidas en diversas carpetas.

Para crear un dataset los scripts se deben ejecutar en el siguiente orden:
1. creation_sublist.py: Para ejecutar este fichero se debe llamar de la forma `python create_sublist.py [tobacco800]`. La llamada registrará todos los documentos, entendiendose como estos todas las imagenes dentro de una unica carpeta, y lo dividirá en 4 listas separadas, esto se hace para más tarde tratar estas 4 listas de forma paralela, una vez creadas las 4 listas se guardaran en 4 ficheros .txt que mas tarde se podrán leer. El argumento tobacco800 cuando se indica como 1, estamos indicando que no queremos crear 4 sublistas, sino una unica lista, esto sirve si el dataset no es muy grande.
2. CreationOCR_parallel.py: Para ejecutar este fichero se debe llamar de la forma `python create_sublist.py [agrs]` y los argumentos de entrada son los siguientes:
	* iden: Indica la lista que se debe leer, este valor puede oscilar de 0 a 3 [string] (NECESARIO).
	* BigTobacco: Indicamos si estamos creando ficheros para BigTobacco [bool].
	* Tobacco800: Indicamos si estamos creando ficheros para Tobacco800 [bool].
3. create_H5DF.py: Para ejecutar este fichero se debe llamar de la forma `python createHDF5.py [agrs]` y los argumentos de entrada son los siguientes:
	* `filtering`: Indica si debe o no filtrar los datos de entrenamiento [bool].
	* `visualize_data`: Indica si crea o no plots mostrando la distribución de los datos de entrenamiento [bool].
	* `create_json_information`: Indica si debe o no crear un JSON con toda la información de los documentos respecto al balanceo de clases [bool].
	* `mode`: Indica si aplica modo "Tail" o "Head" en el momento de filtrar los datos. Es necesario solo si se indica `filtering`.
	* `tobacco800`: Indica si se estan creando ficheros H5DF relacionados con tobaco800 [bool]. 
	* `splitTobacco800`: Indica si debe o no dividir los datos del dataset de tobacco800 entre entrenamiento validación y test [bool].
	* `trainT800`: Indica porcion de entrenamiento de tobacco800 en caso de dividir el dataset [bool].
	* `valT800`: Indica porcion de validación de tobacco800 en caso de dividir el dataset [bool].
	* `testT800`: Indica porcion de test de tobacco800 en caso de dividir el dataset [bool].
