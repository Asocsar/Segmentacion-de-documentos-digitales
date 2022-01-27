# Segmentación de documentos digitales

Grado en Ingenieria Informática
Especialidad de Computaci ́on
Mejora de la segmentaci ́on de
documentos digitales usando nuevas
arquitecturas de redes neuronales
Daniel Cano Carascosa
Director: Jordi Torres Vi ̃nals
Codirector: Juan Luis Dominguez
Resumen
En los ́ultimos a ̃nos se han podido ver grandes avances en el ́area de las redes neuronales en
diferentes ́areas que engloban tanto lo visual como detecci ́on de objetos en im ́agenes, como an ́alisis
o generaci ́on de textos, siendo as ́ı capaces de generar historias cre ́ıbles a tiempo real.

El objetivo de este proyecto es hacer uso de ambos tipos de arquitecturas, tanto visuales como
textuales y aplicarlas en el problema de segmentaci ́on de documentos donde dada una lista de
p ́aginas, la red tendr ́a que dividir la lista de p ́aginas para decidir cuando comienza y acaba un
documento, el cual puede estar formado por una o m ́as p ́aginas.

Para lograr esta tarea se seleccionan datasets p ́ublicos ya conocidos y se analizan, encontrando
y corrigiendo fallos que pueden afectar a nuestro proyecto. Gracias al uso y uni ́on de redes dedicadas
im ́agenes y texto, podremos ver como los modelos propuestos no solo igualan a propuestas previas
que se consideran estado del arte en esta ́area, sino que en algunos casos obtienen resultados
superiores a los mostrados en investigaciones anteriores.

Resum
En els darrers anys s’han pogut veure grans aven ̧cos en l’area de les xarxes neuronals en diferentsarees que engloben tant allo visual com detecci ́o d’objectes en imatges, com analisi o generaci ́o de
textos, i aix ́ı s ́on capa ̧cos de generar hist`ories cre ̈ıbles a temps real.

L’objectiu d’aquest projecte ́es fer ́us d’ambd ́os tipus d’arquitectures, tant visuals com textuals
i aplicar-les al problema de segmentaci ́o de documents on donada una llista de pagines, la xarxa haura de dividir la llista de pagines per decidir quan comen ̧ca i acaba un document, el qual pot estar format per una o m ́es pagines.

Per aconseguir aquesta tasca se seleccionen datasets p ́ublics ja coneguts i s’analitzen, trobant
i corregint errors que poden afectar el nostre projecte. Gracies a l’ ́us i uni ́o de xarxes dedicades imatges i text, podrem veure com els models proposats no nom ́es igualen propostes previes que es
consideren estat de l’art en aquesta `area, sin ́o que en alguns casos obtenen resultats superiors als
mostrats en investigacions anteriors.

Abstract
In recent years, great advances have been seen in the area of neural networks in different areas
that encompass both the visual and the detection of objects in images, such as analysis or text
generation, thus being able to generate credible stories in real time.

The objective of this project is to make use of both types of architectures, both visual and
textual, and apply them to the document segmentation problem where given a list of pages, the

network will have to divide the list of pages to decide when a document begins and ends, which can
consist of one or more pages.

To achieve this task, already known public datasets are selected and analyzed, finding and
correcting faults that may affect our project. Thanks to the use and union of dedicated images and
text networks, we will be able to see how the proposed models not only match previous proposals
that are considered state of the art in this area, but also in some cases obtain superior results than
those shown in previous research.

Agradecimiento
Realizar un proyecto de esta envergadura supone un esfuerzo supone una inversi ́on de tiempo muy
notable, por lo que en primer lugar quiero agradecer a mi familia, la cual me ha soportado estos 4
meses en los que no he estado con ellos debido a que estaba ocupado con este proyecto, pero que
aun as ́ı me han dado todo su apoyo en todo momento

Tambi ́en me gustar ́ıa agradecer al grupo BSC [1] y a mi director y codirector de proyecto, ya
que son gracias a ellos dos que ha sido posible llevar a cabo este proyecto, ya que en momentos
donde nos encontr ́abamos con un reto a enfrentar siempre me han sido una gu ́ıa, bien d ́andome
consejos, conocimientos o experiencias que de otra forma hubiera o bien hubiera tardado demasiado
en adquirir y por ello el proyecto no se hubiera finalizado a tiempo o bien no hubiera sido capaz de
adquirir en la ventana de tiempo que ten ́ıamos disponible.

Sin el soporte y paciencia, ya que a veces puedo tener momentos en los que me cuesta com-
prender alg ́un concepto, de mis directores y familiares, este proyecto no hubiera sido posible, por
ello les expreso mis m ́as sinceros agradecimientos.

́Indice de Contenidos
1 Introducci ́on
1.1 Contexto
1.2 Identificaci ́on del problema
1.3 Agentes implicados
1.4 Alcance
1.5 Requerimientos
2 Metodologia.
2.1 Conceptos
2.2 Soluciones existentes
2.3 Evaluaci ́on de modelos
3 Datos.
3.1 Obtenci ́on de los datos
3.2 Generaci ́on de Texto
3.3 Generar ficheros H5DF
3.4 Selecci ́on de dataset para realizar un benchmark
3.5 Problemas en el balanceo de clases
4 Modelos
4.1 Decisi ́on de Modelos para texto e imagen
4.2 Uni ́on de modelos de texto e imagen
4.3 Entrenamiento
4.4 Implementacion MultiGPU
4.5 Recreaci ́on del modelo estado del arte
5 Resultados.
5.1 An ́alisis de Resultados BigTobacco
5.2 An ́alisis de Resultados Tobacco800
6 Gesti ́on del proyecto
6.1 Obst ́aculos y riesgos previstos
6.2 Seguimiento realizado del proyecto
6.3 Herramientas de control empleadas
6.4 Planificaci ́on temporal
6.5 Gesti ́on econ ́omica
6.6 Sostenibilidad
7 Conclusi ́on
7.1 Conclusiones personales
7.2 Objetivos cumplidos
7.3 Trabajo futuro
1 Modelos propuestos en este proyecto. ́Indice de tablas
mencionadas 2 Experimentos con los primeros modelos propuestos aplicando y sin aplicar las t ́ecnicas
3 Tiempo de ejecuci ́on para el entrenamiento de un modelo
4 Experimentos con dos separaciones diferentes haciendo uso de VGG16
5 Modelos que representan el estado del arte actual.
6 Resultados experimentos de test en BigTobacco
7 Resultados experimentos de test en Tobacco800
8 Tareas con sus tiempos y cargos encargados de cada tarea.
9 Costes de personal dependiendo de su rol.
10 Coste asociado a cada etapa del proyecto seg ́un el personal encargado
11 Presupuesto del hardware utilizado.
12 Contingencia del 20% por cada tipo de gasto.
13 Riesgos para los posibles imprevistos y costes asociados
14 Presupuesto final del proyecto
1 Segmentaci ́on del Dataset en Entrenamiento, validaci ́on y Test. ́Indice de figuras
2 Matriz de confusi ́on.
3 Distribuciones de clases en BigTobacco y en Tobacco800
4 Distribuciones de clases en BigTobacco validaci ́on y Test
5 Numero de documentos segun el numero de p ́aginas en la secci ́on de BigTobacco Train.
6 Filtrado de un m ́aximo de 58 p ́aginas y reducci ́on progresiva de documentos.
7 Descartados todos los documentos de m ́as de 10 p ́aginas
8 Esquema visual de EfficientNet
9 Modulo de texto haciendo uso de Distil BERT con 1 documento
10 Modulo de texto haciendo uso de Distil BERT con 2 documentos
11 Modulo de texto e imagen unidos
12 Gr ́afica de la pl ́anificaci ́on temporal del proyecto.
1 Introducci ́on
Hace aproximadamente 15 a ̃nos las redes neuronales no jugaban un papel muy importante o desta-
cado en ning ́un sector m ́as all ́a del de la investigaci ́on, aunque como todos hemos podido ver hoy
en d ́ıa ya no es el caso, ya que actualmente hemos visto un resurgir de estas t ́ecnicas las cuales
han pasado de no tener un papel muy importante a tener o bien un papel fundamental e incluso
indispensables en sectores como reconocimiento de objetos en im ́agenes [2], detector de poses de
personas dada una imagen [3] o traductores [4].

Como la mayor ́ıa de nosotros hemos podido observar, los grandes avances de los ́ultimos a ̃nos
han sido en el ́ambito de las redes relacionadas con im ́agenes, ya que son estas las que visualmente
m ́as impactan y por lo tanto las m ́as conocidas. No obstante en los ́ultimos 2 a ̃nos tambi ́en se
han estado observando importantes avances en el ́ambito de las redes que tratan con secuencias
temporales, es decir datos que guardan relaci ́on en el tiempo, algunos ejemplos podr ́ıan ser los
valores de la bolsa para el oro, un texto, ya que cada palabra depende temporalmente de la anterior
o sonidos como una conversaci ́on donde el tono que vocalizamos a cada momento depende del
anterior para formar un significado conjunto como por ejemplo una palabra.

En este TFG se pretende desarrollar un modelo que haciendo uso de los ́ultimos avances en
an ́alisis de im ́agenes y texto con redes neuronales, dada una secuencia de N im ́agenes, donde cada
imagen ser ́a una p ́agina escaneada, decidir cuando termina un documento y cuando empieza el
siguiente.

1.1 Contexto
Dentro de la facultad de inform ́atica de Barcelona existen diferentes menciones. Este trabajo de
fin de grado pertenece a la menci ́on de computaci ́on, concretamente al de Inteligencia Artificial y
se ha realizado con la ayuda y cooperaci ́on del BSC [1].

El proyecto parte de la necesidad de diferentes organismos en la actualidad a gestionar una gran
cantidad de documentos de forma diaria, como lo pueden ser por ejemplo bancos o universidades.

Una vez completado el proyecto, cualquier organizaci ́on o particular ser ́a capaz de dividir en
documentos un conjunto de p ́aginas, seg ́un la relaci ́on entre estas, lo cual permitir ́a automatizar
este proceso.

1.2 Identificaci ́on del problema
En este trabajo se busca encontrar cuando termina y cuando acaba un documento, tarea que para
nosotros puede parecer simple, pero cuando te encuentras en una entidad bancaria, por ejemplo,
y te llegan cerca de cientos de miles im ́agenes de documentos sin dividir cada d ́ıa, entonces nos
encontramos con un problema bastante importante de log ́ıstica. Por lo que es en estas situaciones
donde es de vital importancia generar un sistema autom ́atico que se encargue de este proceso. No

obstante para resolver este problema se deben lidiar tanto con la imagen de los documentos como
con el texto de los mismos, ya que de no hacerlo se pueden crear situaciones donde el rendimiento
obtenido disminuya.

Por ejemplo en el contexto donde dado una p ́agina de un documento solo tenemos acceso a la
imagen de esta, la red no ser ́a capaz de identificar el contexto del texto asociado.

Por otro lado, si solo le informamos a la red de la informaci ́on visual, es posible que alguna
p ́agina de un documento solo contenga una gr ́afica, pero el modelo al no tener acceso a esta infor-
maci ́on podr ́ıa entender que es la ́ultima p ́agina del documento.

Por lo que nuestro modelo en cuesti ́on har ́a uso de ambas fuentes de informaci ́on para intentar
resolver este problema.

1.3 Agentes implicados
La herramienta que se va a desarrollar en este proyecto ofrece un gran beneficio a grandes enti-
dades como bancos o cualquier organismo que tenga alg ́un tipo de tr ́amite que implique diversos
documentos, ya que estos en m ́ultiples ocasiones son bien o escaneados como un solo documento
o enviados por los usuarios a trav ́es de sistemas inform ́aticos como un solo fichero, por lo que en
frente de esta situaci ́on donde se pueden recibir millones de documentos mensuales es necesaria una
herramienta automatizada.

1.4 Alcance
A continuaci ́on se definen los objetivos, sub-objetivos, los requerimientos funcionales y no fun-
cionales y riesgos posibles debido a que como el tiempo disponible para realizar el proyecto es
limitado tenemos que tener todos estos factores presentes a la hora de distribuir nuestro tiempo.

1.4.1 Objetivos

Los objetivos principales de este proyecto son los siguientes:

Ser capaz de construir un modelo que haciendo uso de Transformers y redes convolucionales
combinadas sea capaz de resolver el problema con un m ́ınimo de un 70% deaccuracy.
Superar o igualar resultados de los otros trabajos de investigaci ́on presentados, demostrando
as ́ı que nuestro enfoque es o bien una alternativa v ́alida y por lo tanto una alternativa con
potencial para ser ampliada o bien una mejor alternativa que demuestre que la combinaci ́on
de redes convolucionales y Transformers supera con creces los anteriores m ́etodos propuestos.
Usar m ́etodos de paralelizaci ́on para reducir tiempos de entrenamiento para que se pueda
obtener la red deseada en un tiempo m ́ınimo.
1.5 Requerimientos
1.5.1 Requerimientos funcionales

A continuaci ́on se describen los requisitos funcionales para llevar a cabo este proyecto y tambi ́en
para su posterior uso en caso de que este se quisiera usar en una empresa.

Compatibilidad con cualquier dispositivo inform ́atico actual para que su ejecuci ́on sea posible
en cualquier m ́aquina, siempre que esta disponga de una GPU.
1.5.2 Requerimientos no funcionales

A continuaci ́on se describen los requisitos no funcionales, es decir aquellos requerimientos que no
tienen relaci ́on directa con el funcionamiento del sistema, pero que se deben de tener en cuenta.

Rapidez: Como todos sabemos hoy en d ́ıa las aplicaciones que usamos tienen unos tiempos
de respuesta muy peque ̃nos, ya que hay situaciones en donde esta velocidad es necesaria. En
este proyecto se entiende que puede no parecer de vital importancia procesar r ́apidamente las
p ́aginas, pero si tard ́aramos 100ms para procesar cada par de p ́aginas, en una situaci ́on donde
tenemos 1.000.000 de p ́aginas, una situaci ́on no muy dif ́ıcil de imaginar como ya he propuesto
en otros apartados, estar ́ıamos hablando de 70 d ́ıas para procesar toda esta informaci ́on. Por
lo que el apartado de rapidez no se debe ignorar.
Usabilidad: El modelo debe ser sencillo de usar para que alguien que no tenga conocimientos
avanzados en esta ́area sea capaz de utilizar el modelo dentro de un ́ambito de empresa.
Reusabilidad: Dado que en esta ́area existe una t ́ecnica llamadatransfer learning, donde el
modelo a partir de conocimiento ya adquirido aprende nuevas tareas, este modelo podr ́ıa ser
entrenado documentos bancarios y posteriormente ser entrenado en alguna otra tarea de las
propuestas al inicio de este documento, como divisi ́on de cap ́ıtulos dentro de un libro, ya
que como se ha visto en otras investigacionestransfer learningnos permite obtener mejores
resultados de los que obtendr ́ıamos con la misma red entrenando des de cero [5].
2 Metodologia.
2.1 Conceptos
El ́area del Deep Learning es muy extensa y compleja, donde los m ́as m ́ınimos detalles pueden
marcar la diferencia no solo en rendimiento, sino en resultados, es por ello que en esta secci ́on,
aunque se definan los conceptos necesarios para comprender este proyecto solo estaremos definiendo
una peque ̃na parte de esta ́area tan extensa.

A continuaci ́on se definen los conceptos clave para el proyecto, por lo que a partir de este
momento usar ́e libremente los siguientes t ́erminos en m ́ultiples ocasiones.

Tensor

Un tensor se puede entender como objetos que almacenan valores num ́ericos y que pueden tener
distintas dimensiones, permitiendo as ́ı expresar conceptos como orden temporal entre otros. Por lo
que un tensor de 0 Dimensiones es un escalar, 1 Dimensi ́on es un vector, de 2D una matriz y de 3D
un cubo.

Hiperpar ́ametro

Se llama hiperpar ́ametro a todos los par ́ametros que influyen en el entrenamiento del modelo y, por
lo tanto, en su rendimiento final y que son establecidos antes del comienzo del mismo.

Capa Densa

Una capa densa [6] es una capa de red neuronal donde todas las neuronas que forman esta capa
est ́an densamente conectadas con todas las neuronas de la capa anterior, lo que significa que cada
neurona de la capa densa recibe informaci ́on de todas las neuronas de la capa anterior.

Capa convolucional

Las capa convolucionales son el pilar de las redes convolucionales, y estas est ́an compuestas por uno
o m ́as filtros llamados habitualmente Kernels, cuyos par ́ametros son modificados en el proceso de
entrenamiento. Todos los filtros realizan una convoluci ́on con la imagen, haciendo que el tama ̃no
de la imagen original sea reducida y el n ́umero de operaciones necesarias para la siguiente capa.

Red Convolucional

Una red convolucional es un tipo de red neuronal formada por capas convolucionales la cual se
especializa en procesar im ́agenes [7].

Existen excepciones donde se pueden usar capas convolucionales que no se encargan de procesar
im ́agenes si no texto [8]. No obstante como en este caso no aplicaremos esta aproximaci ́on no
tendremos en cuenta estos casos.

Red Recurrente

Una red recurrente [9] est ́a formada por neuronas conectadas en forma de grafo dirigido, siendo as ́ı
capaces de mostrar un comportamiento adaptable a dependencias temporales en datos, permitiendo
que sean usadas en tareas como reconocimiento de voz o de an ́alisis texto.

Transformers

Los transformers[10] son una arquitectura de Deep Learning que aplican el mecanismo de atenci ́on
a los datos de entrada, el cual consiste en ponderar cada dato de entrada seg ́un su importancia y
relevancia respecto los otros datos de entrada.

Estas redes no solo han demostrado un alto rendimiento tratando con im ́agenes [11] sino
tambi ́en con textos [12].

Vector Latente

En muchas situaciones me referir ́e a la informaci ́on que se encuentra entre la entrada y la salida
de una red neuronal, es decir aquella informaci ́on que solo la red entiende y nosotros no podr ́ıamos
entender, como vector latente.

Un vector latente se puede entender como una secuencia de n ́umeros flotantes que codifican,
en el caso de una imagen, el tipo de imagen, cuantas personas hay en la imagen, si la imagen
tiene tonos claros, si tiene una situaci ́on de peligro o cualquier informaci ́on que la red considere
importante guardar.

Modelo Pre-entrenado

Un modelo pre-entrenado es aquel que ya se ha sometido a un proceso de aprendizaje y que, por lo
tanto, ha aprendido a abstraer y procesar informaci ́on de su entrada siempre que est ́e relacionada
con lo que aprendi ́o en el entrenamiento, ya que en caso de un modelo que aprenda a analizar textos
en ingl ́es solo tendr ́a conocimiento de este lenguaje por lo que toda entrada deber ́a estar en ingl ́es.

Transfer learning

Se denominatransfer learninga la capacidad de que un modelo ya pre-entrenado, sea capaz de
aprender una nueva tarea, haciendo uso de los conocimientos previos obtenidos. Por lo que si se
seleccionara un modelo entrenado en ingl ́es y se le entrenara con portugu ́es, este aprender ́ıa m ́as
r ́apido, ya que podr ́ıa hacer uso tanto de gram ́atica como de estructura sint ́actica que aprendi ́o de
los textos en ingl ́es.

Loss function

LaLoss functiono tambi ́en conocida como la funci ́on de error, es la funci ́on que determina para
cada salida de nuestra modelo, el error que se ha cometido respecto el valor ideal deseado.

Learning Rate

Se defineLearning Rateel par ́ametro del algoritmo de optimizaci ́on llamadoGradient Descent[13],
que determina el tama ̃no del paso en cada iteraci ́on hacia el punto m ́ınimo en la funci ́on deLoss.
Dado que este par ́ametro influye en el tama ̃no del paso, se debe tener en cuenta que unLearning Rate
muy peque ̃no tendr ́a el efecto de un proceso de entrenamiento lento, mientras que uno demasiado
elevado podr ́ıa evitar alcanzar el punto m ́ınimo debido a que si la longitud de estos pasos fueran
demasiado grandes, podr ́ıa ocurrir que nunca convergiera en el punto m ́ınimo.

Batch

Batchse puede entender como cuantos datos van a ser suministrados a la imagen en una sola
llamada, por lo que cuanto mayor sea este valor, mayor ser ́a la velocidad de entrenamiento.

Epoch

Epochse puede entender como el n ́umero de veces que nuestro modelo aprender ́a de todo nuestro
conjunto de datos de entrenamiento y el n ́umero de veces que el modelo se evaluar ́a con los datos
de validaci ́on.

2.2 Soluciones existentes
Por supuesto como ya he mencionado este problema no solo existe actualmente, sino que supone
un gran reto para much ́ısimas empresas que generan miles o millones de documentos de forma
diaria y de forma constante, por lo que como se puede suponer ya se han hecho investigaciones e
implementaciones que intentan solucionar este problema de la mejor forma posible.

A continuaci ́on expondr ́e los ́ultimos trabajos publicados que miran de resolver este problema
en cuesti ́on y los m ́etodos que utilizan.

2.2.1 Leveraging effectiveness and efficiency in Page Stream DeepSegmentation

El paper que actualmente es el estado del arte [14] hace ́unicamente uso de im ́agenes con redes
convolucionales, siendo la red seleccionada VGG16 [15].

Para lograr esta tarea los autores del paper emplean dos t ́ecnicas. La primera es donde
́unicamente hacen empleo de dos p ́aginas siendo estas consecutivas y decidiendo si pertenecen o
no al mismo documento, y otro m ́etodo donde hacen uso de 3 p ́aginas, y nuevamente deciden si
pertenecen o no al mismo documento. Por lo que en su implementaci ́on como VGG16 solo puede
procesar una imagen a la vez llaman al modelo tres veces, extraen la informaci ́on del modelo previa
a las ́ultimas capas Densas y finalmente la concatenan y la pasan por una ́ultima capa Densa.

Los resultados que obtienen son bastante prometedores, llegando a obtener hasta un 92% de
accuracyen un dataset de benchmark llamado Tobacco800 y 91.1% en un dataset propio que ellos
mismos han generado manualmente.

2.2.2 Page Stream Segmentation with Convolutional Neural Nets Combining Textual
and Visual Features

Ya ha habido previamente equipos que han intentado esta misma tarea haciendo uso ́unicamente
de redes convolucionales, pero procesando imagen y texto [16]. Su estrategia fue la de, dada una
situaci ́on donde tenemos N p ́aginas, coger las p ́aginask− 1 yk 0 , y procesar las im ́agenes y el texto
por separado, pero ambos elementos con redes convolucionales.

En primer lugar, se entrenaban dos redes, una intentando separar los documentos ́unicamente
con im ́agenes y la otra ́unicamente con texto, haciendo que las dos redes, aunque no tuvieran
resultados muy prometedores, estuvieran pre entrenadas.

La forma de entrenar estas redes serian coger, en el caso de la red que recibe las im ́agenes
k− 1 yk 0 , introducirlas por la red convolucional, obtener dos vectores latentes, que posteriormente
concatenar ́ıa e introducir ́ıa a una red Densa para obtener el resultado, y lo mismo para el texto.

Y posteriormente y una vez entrenadas por separado, se agrupar ́ıan por lo que obtendr ́ıamos 4
vectores latentes que contendr ́ıan informaci ́on sobre estas p ́aginas tanto en imagen como en texto,
las cuales se introducen por una capa densa para obtener finalmente un n ́umero que se situara
entre el 0 y el 1 para darnos un porcentaje de como de fiable cree el modelo que las dos p ́aginas en
cuesti ́on pertenecen a dos documentos diferentes.

Los resultados que obtiene este paper son muy cercanos al estado del arte, logrando unaccuracy
de 91% y un score de Kappa de 81.6% en Tobacco800.

2.2.3 Use of language models for document stream segmentation

Por supuesto al igual que hay quien ha hecho uso ́unicamente de convolucionales tambi ́en hay quien
ha intentado abordar el problema haciendo uso de redes recurrentes [17], las cuales como ya hemos
mencionado, son capaces de abstraer propiedades temporales entre los datos, aunque son menos
potentes que los Transformers como ya se ha demostrado previamente en diferentes investigaciones.

Los resultados que obtienen, pese a no estar dentro del estado del arte se encuentran bastante
cerca, teniendo unaccuracydel 90% en Tobacco800 y unaccuracyde 96% con un dataset llamado
READ-CORPUS, el cual tal y como ellos mencionan es privado.

2.3 Evaluaci ́on de modelos
Para evaluar nuestro modelo necesitamos de tres elementos.

El primero debe ser unos datos de entrenamiento, con los que entrenar a nuestro modelo.
Este conjunto de datos lo llamaremos datos de entrenamiento.
El segundo debe ser unos datos de evaluaci ́on, donde ver como de bien funciona nuestro
modelo, ya que si en estos datos nuestro modelo no funciona correctamente esto querr ́a decir
que no ha aprendido de los datos de entrenamiento, sino que los ha memorizado. Este conjunto
de datos lo llamaremos datos de validaci ́on.
En tercer lugar, tenemos los datos de test los cuales nos sirven para emular como funcionar ́ıa el
modelo en un entorno real. Ya que si solo nos bas ́aramos en los datos de validaci ́on estar ́ıamos
suponiendo que los datos de validaci ́on representan el mundo real, es decir desconocido, por
lo que este tercer grupo no solo tiene como objetivo evaluar al modelo, sino evitar nuestra
influencia sobre el resultado.
Un ejemplo visual y f ́acil de entender se puede visualizar en la figura 1, donde con ́unicamente
un dataset generamos estos tres grupos independientes.

Figure 1: Segmentaci ́on del Dataset en Entrenamiento, validaci ́on y Test.
Adem ́as de esto, por nuestra parte tambi ́en aleatorizamos los datos dentro de cada grupo, es
decir eliminamos todo orden posible, para as ́ı evitar que el mismo orden de los datos genere alg ́un
beneficio oculto en el entrenamiento, esta aleatorizaci ́on se aplica ́unicamente en las subsecciones
generadas de entrenamiento, validaci ́on y test.

No se aplica en el conjunto global, antes de hacer la divisi ́on, debido a que en la propia web
donde se suministra el dataset que se va a usar, se adjunta un csv indicando los datos que van a ir a la

secci ́on de entrenamiento, validaci ́on y test. Esta divisi ́on se ha respetado, ya que como los mismos
autores del dataset explican esta divisi ́on est ́a pensada para que en entrenamiento, validaci ́on y test,
el n ́umero de documentos de que pertenezcan a una misma clase, sea proporcionalmente el mismo a
los de los otros subconjuntos. Por lo que si en entrenamiento un 10% de los documentos pertenecen
a ciencia, en validaci ́on y entrenamiento tambi ́en habr ́a un 10% de documentos cient ́ıficos respecto
el total.

Finalmente, para medir el rendimiento de nuestros modelos, se toman 3 medidas diferentes a
usar, aunque para entender estos conceptos de una forma m ́as f ́acil nos basaremos en la figura 2:

Figure 2: Matriz de confusi ́on.
1.Accuracy: Laaccuracymedir ́a el porcentaje de aciertos respecto el total, pero esta m ́etrica
presenta un problema en el problema con el que estamos tratando, ya que en un caso donde
tenemos 2 documentos, de 10 p ́aginas cada uno, si nuestro modelo decidiera que hay dos
documentos de 5 y 15 p ́aginas, elaccuracyser ́ıa del 85%, ya que de todas las veces que se le
ha pedido identificar si dos p ́aginas son diferentes documentos o no, ha tenido ́exito en un 85%
de las ocasiones, por lo que pese a lo que nosotros considerar ́ıamos un error de importancia el
modelo considera que ha tenido una gran tasa de acierto. La f ́ormula para obtener este valor
seriaAccuracy=T P+F PT P++F NT N+T N
F1 Score: Para lidiar en parte con lo que hemos observado que puede suceder con elAccuracy
tambi ́en hemos implementado F1 Score, aunque para entenderlo antes se deben entender dos
conceptos:
- Precisi ́on: Cuando hablamos de la precisi ́on de un modelo estamos respondiendo a la
pregunta ”¿De todos las p ́aginas etiquetadas como inicios de documento, cuantas de estas
son realmente inicio de documentos?” Por lo que cogiendo como ejemplo el anterior caso
con 2 documentos de 10 p ́aginas obtendr ́ıamos una precisi ́on de 0, ya que ninguna de las
p ́aginas marcadas como inicio de documento realmente lo es. La f ́ormula para obtener
este valor esP recision= T PT P+F P. El problema de esta m ́etrica por si sola es que
ignora los falsos negativos, es decir no tiene en cuenta cuantas veces fallamos diciendo
que dos p ́aginas pertenecen al mismo documento, solo cuando decimos que pertenecen a
diferentes documentos esta m ́etrica entra en juego.
Recall: Esta m ́etrica responde a la pregunta ”¿De todas las p ́aginas que eran comienzo
de documento, cuantas de ellas hemos etiquetado?”. Con esta m ́etrica se busca detectar
la sensibilidad de nuestro modelo, es decir que tan bien es capaz de detectar. No obstante
esta m ́etrica por sola tiene el problema que en un caso de etiquetar todo como nuevo
documento tendr ́ıamos un Recall del 100%, ya que no tiene en cuenta la precisi ́on sino
el n ́umero de falsos negativos que nuestro modelo ha dado. La f ́ormula para calcularlo
esRecall=T PT P+F N.
Por lo que F1 Score se basa en coger estos dos conceptos y combinarlos, haciendo que los
puntos d ́ebiles de una se vean compensados por la otra y la forma que tiene de unirlas es
mediante la f ́ormulaF 1 Score=^2 (∗Recall(Recall+∗P recisionP recision)).
Kappa: Esta m ́etrica mide de manera cuantitativa la cantidad de veces que un observador
aleatorio y nuestro modelo, coincidir ́an en que p ́aginas pertenecen al inicio de un nuevo doc-
umento. Que dicho con en otras palabras, Kappa nos dir ́a cuantitativamente la probabilidad,
de que dos sujetos, o modelos, piensen lo mismo delante de unos datos, es decir que haya
ambos puntos de vista se pongan de acuerdo. Esto nos sirve para, en caso de que viniera un
tercer observador (otro modelo), este deber ́ıa de ”estar de acuerdo” con nuestro modelo.
Por lo que con estos datos presentes, en todos los experimentos se har ́a una ronda de entre-
namiento y una de validaci ́on, donde obtendr ́a elAccuracy, F1 Score y Kappa, y se seleccionar ́a el
modelo con m ́as F1 Score, en caso de empate el modelo con unAccuracym ́as elevado y en caso de
empate el modelo con Kappa m ́as elevado. Una vez seleccionado el modelo con los par ́ametros m ́as
prometedores, se realizar ́a una prueba de Test, que es la que finalmente presentaremos en la tabla
de resultados.

3 Datos.
3.1 Obtenci ́on de los datos
El dataset utilizado para entrenar a la red es BigTobacco (tambi ́en llamado RVL-CDIP) el cual es
un dataset p ́ublico [18], que contiene 400,000 documentos. Estos documentos se clasifican en 16
clases diferentes y sus dimensiones no superan los 1000 pixeles. Hay 25,000 documentos para cada
clase por lo que el dataset est ́a balanceado

En la propia p ́agina se ofrece unos ficheros .csv que nos distribuyen los datos para as ́ı obtener
320.000 documentos de entrenamiento, 40.000 de validaci ́on y 40.000 de test respetando el n ́umero
de diferentes clases de documentos en cada secci ́on.

3.2 Generaci ́on de Texto
Para generar los datos se ha hecho uso de una herramienta llamada pytesseract[19]. Esta librer ́ıa
nos permite analizar una imagen, y extraer el texto que se encuentra en esta, ya que en los datos
que disponemos no contienen el texto de las im ́agenes de los documentos.

Para el entrenamiento de los modelos se har ́a empleo de los procesadoresIBM Power9 8335-
GTH @ 2.4GHz, pero para poder hacer uso de esta librer ́ıa de la forma m ́as eficiente se ha tenido
que usar procesadoresAMD EPYC 7742 @ 2.250GHz, los cuales tienen incorporadas operaciones
vectoriales, que la librer ́ıa de pytesseract usa para generar el texto. Gracias a esto el tiempo para
generar el texto es 23 d ́ıas, que es un speed-up del 2.6 respectos los procesadores de IBM.

Para acelerar a ́un m ́as el proceso se aplica una estrategia de paralelizaci ́on, donde se lanzar ́an
32 procesos, y cada proceso ser ́a responsable de 12.500 documentos. Por lo que para finalmente
generar todos los datos, ser ́an necesarios 3 d ́ıas y 12 horas. No obstante como el entorno en el que
trabajamos una tarea solo puede estar en ejecuci ́on durante 48h, se decide dividir en 4 tareas.

Por lo que al finalmente se lanzar ́an 4 tareas, donde cada tarea procesar ́a 100.000 documentos
y cada proceso generar ́a el texto de 3.125 documentos, lo que nos deja con una estimada de tiempo
de 21h 30min por cada tarea, que si observamos no es el speed up ideal, pero tras una detallada
observaci ́on se ha observado que la gesti ́on y sincronizaci ́on de 32 procesos supone una gran carga
en tiempo para el entorno en el que nos encontramos.

Para conservar la informaci ́on cada proceso conservar ́a el texto extra ́ıdo en ficheros JSON que
posteriormente ser ́an usados para la generaci ́on de ficheros H5DF.

3.3 Generar ficheros H5DF
Para conservar los datos y consultarlos de forma eficiente se ha decidido utilizar ficheros en formato
H5DF. El formato H5DF es un formato de archivo de c ́odigo abierto que admite datos grandes,

complejos y heterog ́eneos. HDF5 utiliza una estructura similar a un ́arbol N-ario, que le permite
organizar los datos dentro del archivo de muchas maneras estructuradas diferentes, de manera
eficiente, evitando la fragmentaci ́on de memoria y permitiendo que los datos se puedan acceder
eficientemente en todo momento gracias a su estructura, lo cual es esencial para que la lectura de
los datos no resulte ser un cuello de botella en el entrenamiento.

Haciendo uso de los ficheros CSV proporcionados, aplicamos la divisi ́on de los documentos en
las categor ́ıas detrain, validaci ́on ytest, donde un documento se compone de 1 a N im ́agenes y sus
respectivos OCRs. Seguidamente, se generan laslabelspara cada p ́agina de cada documento, las
cuales solo tendr ́an dos valores, siendo estos 0 y 1, donde 0 indicar ́a que la p ́agina no es la primera
p ́agina de un documento, y 1 indicar ́a que s ́ı que es la primera p ́agina de un documento. Debido
a que el dataset BigTobacco nos ofrece los documentos ordenados, podemos asignar sin problemas
estaslabelsa cada p ́agina.

A la hora de generar los ficheros H5DF, se plantearon dos estrategias, ya que se deb ́ıa tener en
cuenta que los datos almacenados en estos ficheros ser ́an cargados en RAM, por lo que se debe elegir
un m ́etodo que no sobrecargue estos ficheros con informaci ́on repetida, pero tampoco provocar que
la manipulaci ́on de los datos, de ser necesario, resulte en un cuello de botella.

La primera de ellas es que la estructura fuera la de[imagenk,imagenk+1,ocrk,ocrk+1,
max(labelk, labelk+1)]. Con esta implementaci ́on se pretend ́ıa que cada entrada fuera com-
puesta por dos p ́aginas, por lo que cada entrada de los ficheros ser ́ıa directamente introducida
al modelo.
La segunda de ellas es que cada elemento tuviera la estructura[imagenk, ocrk, labelk].
Este enfoque nos obliga a acceder a la siguiente entrada para poder generar un ejemplo de
entrada para nuestro modelo, aunque reducimos significativamente el n ́umero de ejemplos que
podemos introducir por fichero.
Despu ́es de pruebas en tiempos se observ ́o que la mejor opci ́on era la segunda, ya que esta no
supone un tiempo elevado de carga en memoria RAM y la uni ́on de datos manual no resulta en un
cuello de botella.

Debido a este estudio del tama ̃no de los ficheros y de su impacto en la memoria RAM, se ha
decidido no generan un ́unico fichero para la secci ́on de train, sino generar m ́ultiples ficheros de
tama ̃no m ́as reducido, lo que nos permite cargar los datos en memoria RAM sin problemas, siendo
3000 el n ́umero de p ́aginas contenidas en cada fichero. Aunque debido a esta divisi ́on se ha tenido
que a ̃nadir al final de cada fichero H5DF el primer ejemplo del siguiente fichero, para conservar la
continuidad sin necesidad de abrir 2 ficheros simult ́aneamente.

3.4 Selecci ́on de dataset para realizar un benchmark
Una vez ya tenemos el dataset principal que vamos a usar para medir nuestro rendimiento, nece-
sitamos de otro dataset de uso com ́un que se haya usado en otros proyectos de investigaci ́on que

nos permita obtener m ́etricas deaccuracy, F1 ScoreyKappay as ́ı por nuestra parte, podernos
comparar con los resultados obtenidos de otras fuentes con el fin de saber si nuestros resultados
han sido mejores o peores que otros trabajos previos.

Para poder realizar esta prueba en nuestro utilizaremos Tobacco800, que ya ha sido amplia-
mente utilizado como dataset de benchmark no solo en tareas como segmentaci ́on de documentos
sino tambi ́en en otras tareas como clasificaci ́on de documentos, por lo que es bastante conocido. Este
dataset consta de 1290 p ́aginas, las cuales conforman 742 documentos y en nuestro caso estaremos
utilizando todo el dataset como ́unicamente un dataset de pruebas, en ning ́un momento se entrenar ́a
en este dataset. El motivo de esta decisi ́on es cuando se ha estudiado los otros proyectos[14] se
ha observado que en ninguno deja constancia de que divisi ́on de entrenamiento, validaci ́on y test
que se ha empleado, y no ́unicamente eso sino que como en uno de estos se estudia[14] la divisi ́on
de las clases afecta en gran medida a los resultados obtenidos, adem ́as de que hay riesgo dedata
leak, que significa que es hay p ́aginas de documentos tan similares que en un caso donde dividimos
el documento en las secciones detrain, validaci ́onytestsi dos p ́aginas casi id ́enticas terminaran
una en la secci ́on detrainy la otra en la secci ́on detest, por ejemplo, nuestro modelo se podr ́ıa
ver beneficiado debido a esta similitud, debido a que hay diferentes p ́aginas en este dataset que son
pr ́acticamente id ́enticas y se encuentran en diferentes conjuntos que deber ́ıan ser independientes,
por lo que para evitar esta situaci ́on nosotros no segmentaremos en ning ́un momento el dataset ni
entrenaremos en ́el.

3.5 Problemas en el balanceo de clases
3.5.1 Origen del problema

Una vez realizado seleccionado el dataset que vamos a utilizar como benchmark, procedemos a
analizar este dataset para ver su distribuci ́on de clases y compararlo con la distribuci ́on de clases de
la porci ́on de entrenamiento de RVL-CDIP (BigTobacco), ya que como exclusivamente es la secci ́on
de entrenamiento la que influye en el entrenamiento de nuestros modelos es esta la ́unica que puede
jugar un papel destacable en como se desarrolla el aprendizaje de los modelos, puesto que nuestra
intenci ́on es visualizar si hay alg ́un desbalance de clases que pueda favorecer a nuestros modelos
en frente de otros no porque nuestros modelos sean m ́as precisos, y los resultados los podemos
visualizar en la figura 3.

Lo que estamos visualizando es como en Tobacco las clases se encuentran balanceadas, por lo
que no hay ning ́un problema, en cambio, cuando miramos BigTobacco nos encontramos con un serio
problema, el cual es que cerca del 77% de las p ́aginas pertenecen a un mismo documento, por lo que
si nosotros gener ́asemos un modelo que diera todo el rato como salida ceros, tendr ́ıamos un 77%
de precisi ́on en la secci ́on de entrenamiento, aunque esto no ser ́ıa un problema si las secciones de
validaci ́on y test mantienen un balance correcto, ya que nos percatar ́ıamos de esto, por lo que a causa
de esta situaci ́on procedemos a analizar los conjuntos de validaci ́on y de test de BigTobacco, por
el hecho de que de presentarse el mismo desbalance, podr ́ıamos estar arriesg ́andonos a que nuestro
modelo estuviera vi ́endose beneficiado de este factor a la hora de obtener m ́etricas deaccuracy,F
ScoreyKappa.

Nuevo
Documento
Mismo
Documento
0
50
100
22. 67
77. 33
Percentage
BigTobacco Train
Nuevo
Documento
Mismo
Documento
0
50
100
58. 29
41. 61
Percentage
Tobacco800
Figure 3: Distribuciones de clases en BigTobacco y en Tobacco800
Nuevo
Documento
Mismo
Documento
0
50
100
21. 9
78. 1
Percentage
BigTobacco Validaci ́on
Nuevo
Documento
Mismo
Documento
0
50
100
23. 55
76. 45
Percentage
BigTobacco Test
Figure 4: Distribuciones de clases en BigTobacco validaci ́on y Test
Como se puede observar en la figura 4, tanto la secci ́on de validaci ́on como de test, presentan la
misma distribuci ́on de clases que la secci ́on de entrenamiento, lo cual es un problema porque existe
la posibilidad de que nuestros entrenamientos est ́en siendo sesgados, por lo que en vista de esto,
tenemos que tomar la decisi ́on de aplicar un filtrado a nuestro dataset para corregir este desbalanceo
de clases, aunque este filtrado solo se aplicar ́a a la secci ́on de entrenamiento, pues es de esta de la
cual el modelo aprende y, por lo tanto, de la cual se puede ver influenciado

3.5.2 Filtrado de datos de entrenamiento

En primer lugar, para observar la situaci ́on en la que nos encontramos generamos una gr ́afica que
nos agrupe los documentos en grupos, siendo estos grupos definidos por el n ́umero de p ́aginas que
contiene cada documento.

El resultado de esta visualizaci ́on se puede observar en la figura 5, donde el extremo izquierdo
hay el n ́umero de documentos que se conforman de una ́unica p ́agina y a la derecha el n ́umero
de documentos que se conforman de 1942 p ́aginas, adem ́as debido al extremo desbalance en esta
gr ́afica se ha tenido que escalar el eje que nos da informaci ́on sobre el n ́umero de documentos a una

Figure 5: Numero de documentos segun el numero de p ́aginas en la secci ́on de
BigTobacco Train.
escala logar ́ıtmica para que este sea visible.

Entonces una vez tenemos delante nuestra situaci ́on hay dos alternativas:
En primer lugar podemos acortar el n ́umero de documentos seleccionados de forma exponen-
cial a medida que estos tienen m ́as p ́aginas (por lo que si hubiera 4 documentos de 2 p ́aginas y
16 de 3 p ́aginas solo seleccionar ́ıamos aleatoriamente 2 documentos de 2 p ́aginas y 4 documen-
tos de 3 p ́aginas), por lo que estar ́ıamos acortando el n ́umero de ocurrencias de p ́aginas en un
mismo documento, y al mismo tiempo descartamos todos los documentos que contengan m ́as
de K p ́aginas, ya que de no hacerlo debido a la incre ́ıble cantidad de documentos de 1 sola
p ́agina se vuelve muy complejo encontrar un equilibrio. El resultado de esta aproximaci ́on se
puede observar en la figura 6.
En segundo lugar podemos descartar todos los documentos que contengan m ́as de K p ́aginas,
siendo K un n ́umero entre 1 y 1942, siendo esta la opci ́on m ́as sencilla. El resultado de aplicar
este enfoque se puede ver en la figura 7.
Despu ́es de diversas pruebas encontramos que si escog ́ıamos la primera opci ́on, deb ́ıamos limitar
el n ́umero de p ́aginas por documento a 58, mientras que si escog ́ıamos la segunda el n ́umero de
p ́aginas deb ́ıa ser limitado a 10, ya que al no reducir el n ́umero de documentos hay muchas m ́as
ocurrencias de ”Mismo documento”.

Despu ́es de un an ́alisis se decidi ́o optar por la segunda opci ́on, puesto que esta, a pesar de
Figure 6: Filtrado de un m ́aximo de 58 p ́aginas y reducci ́on progresiva de
documentos.

Figure 7: Descartados todos los documentos de m ́as de 10 p ́aginas
destruir todos los documentos de m ́as de 10 p ́aginas, conservaba el 92% de los documentos respecto
el total, mientras que el primer m ́etodo apenas conservaba el 70%, debido al notable desbalance en
las ocurrencias de documentos con 1 ́unica p ́agina. No obstante aun en caso de perder solo un 8%
de los documentos respecto el total, la perdida en cantidad de p ́aginas resulta del 59%.

4 Modelos
4.1 Decisi ́on de Modelos para texto e imagen
Una vez ya tenemos todos los datos preparados, es momento de seleccionar la arquitectura de los
modelos, para esto por nuestra parte optamos como decisiones base que el modelo que analizara
el OCR fuera Distil BERT [20], ya que Distil BERT es un modelo que pertenece a la familia
de los Transformers, que ́ultimamente est ́an demostrando un gran rendimiento tanto en imagen
como texto [11][12] por lo que decidimos poner a prueba cuanto rendimiento podr ́ıamos obtener de
ellos y que la parte que analizara la imagen fuera EfficientNet[21], ya que no solo ha mostrado un
gran rendimiento en tareas relacionadas con im ́agenes, sino que es una arquitectura muy eficiente
haciendo uso de la m ́ınima cantidad necesaria de par ́ametros.

4.1.1 Modelo para imagen

EfficientNet es una familia de redes convolucionales la cual se basa en la m ́axima eficiencia re-
duciendo al m ́aximo el tama ̃no del modelo, optimizando la velocidad de entrenamiento. Existen
diferentes tipos de EfficientNet comenzando por la B0, que es la m ́as peque ̃na y la menos potente,
pero m ́as veloz, hasta la B6 que es la m ́as grande de todas y por ende la m ́as potente, aunque la
m ́as lenta de entrenar. En todos los modelos que mencionemos que hagan empleo de EfficientNet,
se har ́a empleo de la EfficientNetB0 en caso de que no se diga lo contrario.

Como se puede observar en la figura 8 la familia de redes EfficientNet toma por entrada una
imagen de entrada, y solo una, la cual en nuestro caso es RGB, y la procesa por las diferentes capas
convolucionales que conforman a la red. Finalmente la ́ultima capa convolucional es procesada por
una capa Densa de 256 neuronas, habiendo sido este n ́umero decidido previo a los experimentos.
Aunque al tratarse de una red optimizada en todo momento para conservar elaccuracyy tener un
tiempo necesario para entrenar m ́ınimo, la red est ́a optimizada para tratar con im ́agenes de 224x224
pixeles, por lo que en todo momento las im ́agenes son redimensionadas a esta medida. Por lo que
finalmente obtenemos un tensor de 256 neuronas que contienen la informaci ́on visual de la imagen.

Figure 8: Esquema visual de EfficientNet
4.1.2 Modelo para texto

Distil BERT es una versi ́on destilada de BERT[22]. Un modelo destilado se obtiene con el proceso
de transferir conocimiento, tambi ́en conocido comodistilling[23], de un modelo, como es en nuestro
caso BERT, a un modelo con una arquitectura m ́as reducida. El modelo original, BERT, tendr ́a m ́as
capacidad de almacenar conocimiento, pero modelos como BERT que son muy extensos y que han
sido pre-entrenados con diversas tareas, es posible que no se utilice todo su potencial dependiendo
de la tarea, por lo que una versi ́on reducida ya puede ser v ́alida. Por lo que el proceso de destilaci ́on,
permite transferir el conocimiento evitando, en algunos casos, que disminuya elaccuracy. Adem ́as,
al ser m ́as peque ̃nos estos se pueden implementar en hardware menos potente o bien combinar con
otros modelos sin ocupar tanto espacio.

Por lo que usaremos Distil Bert para procesar las relaciones entre palabras de la p ́agina o
p ́aginas introducidas y nos dar ́a informaci ́on sobre el texto del documento.

Para lograr esta tarea, en primer lugar haremos uso de un Tokenizador. Un Tokenizador es
una funci ́on, que dado una entrada en forma de texto, mapea a un identificador num ́erico. Este
mapeado puede ser simple, como coger todas las palabras del diccionario, asignarles un identificador
y as ́ı poder mapear cada palabra a un n ́umero o bien puede ser m ́as complejo, donde lo que se
mapean no son directamente palabras sino secciones de palabras. Por ejemplo la palabra, amigo y
amable comparten la combinaci ́on ”am” al principio, por lo que se podr ́ıa hacer un mapeo de esta
combinaci ́on a un identificador num ́erico, lo cual permite que no se deba de hacer un mapeo por
cada palabra existente.

El Tokenizador seleccionado ha sido utilizado en el pre-entrenamiento de distil BERT, por lo
que nos podemos aprovechar de esta herramienta para traducir las palabras de entrada que nosotros
entendemos, a identificadores que nuestro modelo es capaz de entender.

En segundo lugar, debemos verificar la longitud de estas secuencias de n ́umeros generados, ya
que distil BERT fue originalmente entrenada para ser capaz de procesar frases de una longitud
m ́axima de 512, siendo siempre la primera posici ́on un token especial llamado [CLS]. Este token
no proviene del texto y tiene como objetivo asimilar el contexto general de todo nuestro texto des
de un marco global, adem ́as este token fue empleado en el pre-entrenamiento de la red, por lo que
nos podemos beneficiar del hecho que la red ya tiene el conocimiento de como extraer el contexto
general de la p ́agina.

En caso de exceder este l ́ımite, nuestro modelo podr ́ıa seguir procesando la informaci ́on, pero
o bien no ser ́ıa capaz de asimilarla o bien no estar ́ıamos aprovechando al m ́aximo la capacidad de
nuestra red, ya que al cambiar la longitud presentada en su pre-entrenamiento estar ́ıamos obligando
al modelo a re-aprender la relaci ́on m ́axima entre las palabras, por lo que en caso de exceder este
n ́umero nos quedaremos con las primeras 511 posiciones del texto y le a ̃nadiremos el token especial
[CLS] tal y como se puede ver en la figura 11.

Distil BERT tambi ́en es capaz de procesar 2 o m ́as p ́aginas de forma simult ́anea, pero en caso
de introducir 2 p ́aginas o m ́as un token especial llamado [SEP] deber ́a ser introducido entre los
tokens de cada p ́agina para indicar al modelo la existencia de uno o m ́as textos independientes, as ́ı
mismo, en caso de que se introduzcan 2 o m ́as p ́aginas, se limitar ́a el n ́umero m ́aximo de tokens

por p ́agina dependiendo del n ́umero de p ́aginas existente, por lo que si tenemos dos p ́aginas de
entrada y tenemos un m ́aximo de 512 tokens, restando el token [CLS] y [SEP] nos quedan 510
posiciones, por lo que se limitar ́a a un m ́aximo de 255 tokens por documento, y en caso de que
queden posiciones vac ́ıas, ya bien porque se introduzca un documento de ́unicamente 4 palabras o
bien porque al introducir 2 p ́aginas una contenga 255 tokens y la siguiente 200 tokens, el resto de
posiciones ir ́an cubiertas por el ́ultimo tipo de token especial llamado [PAD] el cual le indica a la
red que en esa posici ́on no hay informaci ́on. Un ejemplo de esto se puede observar en la figura 10.

Figure 9: Modulo de texto haciendo uso de Distil BERT con 1 documento
Figure 10: Modulo de texto haciendo uso de Distil BERT con 2 documentos
Una vez las p ́aginas sean introducidas, ya sea en el caso de una sola p ́agina o bien de m ́ultiples,
se seleccionar ́a la salida obtenida de la entrada [CLS] que como se ha mencionado contiene la
informaci ́on global del documento, lo que se podr ́ıa interpretar como el contexto. Por lo que
finalmente obtenemos un tensor de 768 posiciones que contiene el contexto general del texto.

4.2 Uni ́on de modelos de texto e imagen
Una vez definidos los modelos solo nos queda hacer la uni ́on de estos, no obstante debido a que
estos pueden ser tratados como m ́odulos independientes, siendo EfficientNet capaz de procesar una
́unica p ́agina, y distil BERT capaz de procesar de 1 a N p ́aginas, se toman tres decisiones:

Se plantear ́an modelos que hagan uso de dos p ́aginas y tres p ́aginas, ya que como se ha podido
ver en el estudio del estado del arte para esta tarea, el uso de 3 p ́aginas nos ofrece una ventaja
adicional, y a causa de que para nosotros la diferencia entre 2 y 3 p ́aginas es el n ́umero de
llamadas a EfficientNet y distil BERT, la estructura del modelo no cambia.
Se probar ́an diferentes formas de utilizar distil BERT, puesto que en el caso de 2 p ́aginas
bien podemos o llamar a distil BERT una vez y concatenar los tokens a la entrada, o bien
podemos pasar las p ́aginas de forma independiente, obtener 2 tensores de 786 y concatenarlos
con las llamadas de EfficientNet. Por lo que en el caso de 2 p ́aginas se har ́an experimentos
llamando a distil BERT una o dos veces, y en el caso de 3 p ́aginas se llamar ́a 1 vez, haciendo
la uni ́on de los 3 documentos, 2 veces, haciendo la uni ́on de la manera (Kn− 1 ,Kn) y (Kn,
Kn+1) siendo K el documento y n la posici ́on o bien haciendo 3 llamadas independientes.
En vista de los resultados de VGG16, haremos dos pruebas con una ́unica llamada a distil
BERT, es decir concatenando todos los tokens en la entrada, y haciendo uso de EfficientNetB2.
El objetivo de estas pruebas es observar el impacto que tiene la red convolucional en el
rendimiento global del modelo.
Por lo que ya con los modelos y el c ́omo hacer empleo de ellos solo resta hacer la uni ́on de estos,
para ello se concatenan los tensores obtenidos de todas las llamadas en un ́unico tensor resultante, el
cual es procesado por una ́ultima capa Densa formada por una neurona con una activaci ́on sigmoide.
La activaci ́on sigmoide nos servir ́a para ser capaces de obtener como resultado en la salida un valor
num ́erico que tendr ́a un rango des del 0 hasta el 1, siendo este interpretado como una probabilidad
de que las dos p ́aginas en cuesti ́on formen parte de dos documentos diferentes.

Los modelos propuestos en esta secci ́on, pueden ser visualizados en la tabla 1:
Experimentos
Modelo Imagen P ́aginas
Llamadas
Distil BERT
EfficientNetB0 2 1
EfficientNetB0 2 2
EfficientNetB0 3 1
EfficientNetB0 3 2
EfficientNetB0 3 3
EfficientNetB2 2 1
EfficientNetB2 3 1
Table 1: Modelos propuestos en este proyecto.
Figure 11: Modulo de texto e imagen unidos
4.3 Entrenamiento
Debido a la naturaleza de las redes neuronales, existe la posibilidad que dos entrenamientos difer-
entes nos proporcionen dos modelos con un rendimiento diferente, siendo estos entrenados con los
mismos datos, por lo que se han realizado como m ́ınimo 3 pruebas de cada experimento para poder
obtener informaci ́on sobre la variabilidad de nuestros resultados.

4.3.1 Hardware

Para el entrenamiento de los modelos el hardware utilizado ha sido el siguiente:

4xGPUs Tesla-V100
2 x IBM Power9 8335-GTH @ 2.4GHz
512GB de memoria RAM distribuida en 16 dimms x 32GB @ 2666MHz
2 x SSD 1.9TB as local storage
4.3.2 Hiperpar ́ametros

Para entrenar a nuestro modelo se ha hecho uso de la funci ́on delossllamadaBinary Cross Entropy
compara cada una de las probabilidades pronosticadas con el resultado real de la clase, que puede
ser 0 o 1. Luego calcula la puntuaci ́on que penaliza las probabilidades en funci ́on de la distancia
desde el valor esperado.

El optimizador usado ha sidoAdam[24]. Al igual que ellearning ratese puede entender como
el tama ̃no del paso, el optimizador se puede entender como el c ́omo damos este paso. Adam es un
algoritmo de optimizaci ́on de reemplazo para el descenso de gradiente estoc ́astico[25], que combina
propiedades de otros dos algoritmos llamados AdaGrad[26] y RMSProp[27].

El n ́umero deEpochsse ha establecido en 3, el batch size utilizado ha sido de 9 y el n ́umero
de workers para cargar nuestro dataset han sido 16, en cuanto alLearning Rateaplicado al modelo
convolucional ha sido de 2∗ 10 −^2 y a distil BERT se han aplicado dos t ́ecnicas llamadasLearning
Rate Adaptative y Slanded triangular Learning Rate.

Las t ́ecnicaslearning rate adaptative y slanded triangular learning ratese observaron cuando
se hizo un estudio de distil BERT y de como mejorar su rendimiento, ya que se hab ́ıa observado que
en Transformers es posible que un entrenamiento resulte en un modelo que no ha logrado aprender
los conceptos deseados, pero que tampoco sea capaz de realizar tareas que ya hab ́ıa aprendido
con anterioridad, este fen ́omeno es denominadoCatastrophic Forgetting [28]. Por lo que para
enfrentarnos a este problema aplicamosLearning Rate Adaptative[29], el cual consiste en que a
cada capa se le establece unLearning Ratediferente, siendo menor las capas m ́as profundas y mayor
las m ́as externas, para que el modelo sea capaz de aprender pero no olvidar. Por lo observado en
nuestra investigaci ́on este error solo afecta a los Transformers, por lo que esta estrategia deLearning
Rate Adaptativesolo ha sido aplicada al m ́odulo de texto.

ConLearning Rate Adaptativese combin ́o una t ́ecnica llamada WarmUp la cual consist ́ıa en
incrementar el LR en las primeras Epochs, y despu ́es lentamente reducirlo, por lo que en las primeras
fases elLearning Ratees muy bajo, por lo que el modelo aprende poco a poco, y a medida que las
Epochs avanzan, este va aumentando para que el modelo empieza a aprender, seguida de una fase
de disminuci ́on del Learning Rate en la que el modelo aprende los ́ultimos detalles de los datos.
Esta t ́ecnica es conocida con el nombre de Warmup[22] y aplicandoSlanded triangular Learning
Ratea nuestro LR, podemos lograr el mismo efecto.

Se realiz ́o una prueba para visualizar la mejora causada por la estrategia delLearning Ratey
Slanded triangular Learning Rateen distil BERT, la cual se puede visualizar en la tabla 2.

LLamadas a Distil BERT
Accuracy
Con ADLR y SLTR Sin ADLR y SLTR
1 0.85 0.3759
2 0.77 0.54
Table 2: Experimentos con los primeros modelos propuestos aplicando y sin aplicar
las t ́ecnicas mencionadas
4.4 Implementacion MultiGPU
4.4.1 Escalabilidad

En primer lugar se han hecho diversas pruebas de entrenamiento con una ́unica GPU y progresiva-
mente se han ido a ̃nadiendo GPUs, para ver como escalaban los modelos a medida que se a ̃nad ́ıan
GPUs para determinar si merec ́ıa o no la pena aplicar este enfoque. Los resultados obtenidos son
fruto de procesar el 10% de los datos de entrenamiento y haciendo uso de una ́unica llamada a distil
BERT con dos p ́aginas. Los datos se pueden observar en la tabla 3.

GPUs Tiempo cuando el n ́umero de llamadas a Distil BERT es Una
1 7h 2min
2 3h 8min
4 2h
Table 3: Tiempo de ejecuci ́on para el entrenamiento de un modelo
Por lo que los experimentos se han tenido que realizar todos haciendo empleo de 4 GPUs,
debido a que el entorno en el que nos encontramos no es posible ejecutar una tarea por m ́as de 48h
seguidas. A causa de esta limitacion, se ha implementado el c ́odigo necesario para un entrenamiento
de una ́unica GPU, pero no se ha podido utilizar.

4.4.2 Estrategias

Existen diferentes estrategias y enfoques a la hora de entrenar de forma paralela un modelo de
inteligencia artificial, cada una con sus ventajas y desventajas. A continuaci ́on se presentan dos
tecnicas muy conocidas, as ́ı como sus puntos fuertes y d ́ebiles:

1.Data parallelism: este m ́etodo consiste en copiar el modelo en cada GPU, hacer el entre-
namiento independiente en cada GPU, con datos independientes en cada GPU (ya que quere-
mos que cada modelo de cada GPU vea datos diferentes), y al final de cada batch, despu ́es de
calcular el error y, por lo tanto, actualizar los pesos, hacer una media de estos. Por lo que al
final aumentamos la velocidad de entrenamiento, pero debido a la sincronizaci ́on necesaria, al
hecho de que se hace una media de los pesos, y as ́ı como el factor de tener que copiar tantas
veces el modelo como gr ́aficas, puede provocar que sea m ́as lento a la hora de entrenar de lo
que considerar ́ıamos ideal, por lo que el speed-up obtenido no siempre ser ́a el ideal.
2.Model parallelism: consiste en dividir el modelo entre diferentes GPUs, por ejemplo en un
escenario donde solo tenemos 2 GPUs, la estrategia ser ́ıa enviar el m ́odulo de imagen a laGP U 1
y el modelo de texto a laGP U 2. El efecto de esto es que podemos aumentar substancialmente
el batch, aunque el efecto m ́as notable es que podemos crear modelos mucho m ́as complejos
y potentes que antes, ya que al dividir el modelo por diferentes GPUs, este puede crecer en
tama ̃no.
En el caso actual utilizaremos la t ́ecnica dedata parallelism, puesto que en nuestro caso nos
interesa aumentar la velocidad de entrenamiento debido a las 48 horas l ́ımite que disponemos de
entrenamiento.

4.4.3 Implementaci ́on

En cuanto a la implementaci ́on de este bucle de entrenamiento paralelo que hace uso de multiples
GPUs, se han usado diversas librer ́ıas para facilitar la implementaci ́on del mismo. En primer lugar,
se ha usado la librer ́ıa de multiprocesado de Pytorch [30] la cual permite llamar a una funci ́on de
forma paralela en diferentes hilos de ejecuci ́on independientes, y puesto que vamos a hacer uso de
4 GPUs se ha hecho uso de esta llamada indicando que genere 4 hilos independientes.

Seguidamente, se ha hecho empleo de la librer ́ıa distributed dentro de Pytorch [31], la cual
nos permite establecer el entorno, es decir un medio por el cual los diferentes hilos se comunicaran
en todo momento para informar entre ellos del estado de entrenamiento y, por lo tanto, de llevar
a cabo las acciones necesarias, as ́ı como el backend en el que se entrenaran las GPUs, lo que nos
permite acceso a una serie de llamadas as ́ıncronas las cuales permiten a los hilos independientes
generar puntos de sincronizaci ́on o de parada seg ́un el entrenamiento lo requiera. Junto a esta im-
plementaci ́on se hace uso de una librer ́ıa llamada DistributedDataParallel [32] la cual se usar ́a para
copiar el modelo a las distintas gr ́aficas para que estas puedan entrenarlo de forma independiente.

Finalmente, para decidir con que datos se entrenar ́an los diferentes modelos en las diferentes
gr ́aficas lo que hemos hecho ha sido dividir los 470 ficheros con estructura h5df generados previa-
mente en 4 segmentos independientes, por lo que los 4 modelos no entrenan en ningun momento
con los mismos datos.

4.5 Recreaci ́on del modelo estado del arte
Para ser capaces de validar nuestros resultados debemos comparar estos con el estado actual del
arte, siendo este VGG16 [15].

El ́unico problema que encontramos es que por su parte ellos entrenan, validan y hacen un
test en dos datasets diferentes, uno de ellos lo nombran AI.Lab.Splitter, al cual no tenemos acceso

porque no es p ́ublico. Y el segundo es tobacco800, el problema de este es que tal y como ellos
mismos demuestran el c ́omo se haga la divisi ́on de train, validaci ́on y test tiene un gran impacto en
los resultados obtenidos, y adem ́as ellos por su parte mencionan el porcentaje de validaci ́on respecto
al de entrenamiento, pero no mencionan el porcentaje de test, por lo que no tenemos forma de imitar
sus experimentos, por lo que decidimos implementar el mismo modelo que ellos proponen, con el
mismo procedimiento sobre los datos para as ́ı por nuestra parte primero comprobar que los datos
son de confianza y segundo poder hacer nuestras propias pruebas sobre BigTobacco para as ́ı tener
un mejor contraste sobre nuestra mejora sobre VGG16.

Una vez creado el modelo de VGG16 lanzamos una peque ̃na prueba con VGG16 sobre To-
bacco800 para primero confirmar el funcionamiento de la red, y segundo verificar resultados simi-
lares a los presentados en su paper. Los resultados son visibles en la tabla 4.

10% train, 20%val, 70% test 30% train, 20%val, 50% test
VGG16 Acc 0.913, F1 0.944, Kappa 0.755 Acc 0.92, F1 0.87, Kappa 0.90
Table 4: Experimentos con dos separaciones diferentes haciendo uso de VGG16
Por lo que procedemos a implementar el pipeline completo para poder implementarlo con
BigTobacco, aunque nos encontramos con dos factores destacables:

El modelo que ellos usan consta de una VGG16, que al igual que nuestros modelos extrae un
tensor de dimensiones (batch, nf eatures) el cual conecta a una capa Densa. Pero ellos han
congelado los pesos del modelo VGG16.
Ellos implementan un proceso a las im ́agenes llamado Otsu [33] el cual tiene como objetivo
mejorar el rendimiento de la red aplicando unthresholda las im ́agenes para dividir el contenido
y el fondo de las p ́aginas, haciendo que todos los pixeles superiores al threshold opten por un
valor de 255 y todos los inferiores por un valor de 0.
En vista de estos dos puntos por nuestra parte tomamos dos decisiones.
La primera siendo revertir el efecto de los pesos congelados en VGG16, ya que al entrenar con
BigTobacco si VGG16 est ́a con los pesos congelados no ser ́a capaz de aprender toda la informaci ́on
de un dataset tan grande.

La segunda es aplicar las t ́ecnicas usadas por ellos a nuestro pipeline para que el proceso sea lo
m ́as fiel posible, ya que nuestra intenci ́on es poder comparar lo mejor posible nuestros resultados.

Experimentos
Modelo Imagen P ́aginas
VGG16 2
VGG16 3
Table 5: Modelos que representan el estado del arte actual.
5 Resultados.
En las tablas 6 y 7 se pueden visualizar los resultados obtenidos tanto en BigTobacco como en
Tobacco800. Los resultados han sido obtenidos de seleccionar la Epoch con la mejoraccuracyen
la secci ́on de validaci ́on, y seleccionar laaccuracy, F1 Score y Kappa del resultado de test para esa
misma Epoch. Adem ́as, debido a que estamos ejecutando los entrenamientos con 4 GPUs, y como
ya se ha comentado por la naturaleza del mismo existe una posible inestabilidad, hemos ejecutado
cada entrenamiento 3 veces y despu ́es hemos hecho la media de los resultados, para obtener un
resultado representativo, adem ́as para representar la estabilidad de los resultados se ha calculado
la desviaci ́on est ́andar de la media calculada.
BigTobacco Dataset
Entrenamiento sin
filtrar datos
Entrenamiento con
datos filtrados
Modelo Imagen Paginas
Llamadas
Distil BERT Accuracy F1 Score Kappa Accuracy F1 Score Kappa
EfficientNetB0 2 1 0. 966 ± 0. 027 0. 924 ± 0. 046 0. 886 ± 0. 068 0. 956 ± 0. 044 0. 903 ± 0. 108 0. 881 ± 0. 126
EfficientNetB0 2 2 0. 944 ± 0. 040 0. 895 ± 0. 076 0. 846 ± 0. 112 0. 951 ± 0. 047 0. 889 ± 0. 119 0. 829 ± 0. 190
EfficientNetB0 3 1 0. 961 ± 0. 021 0.965±0.010 0.943±0.016 0. 940 ± 0. 056 0. 915 ± 0. 056 0. 874 ± 0. 096
EfficientNetB0 3 2 0.977±0.006 0. 956 ± 0. 006 0. 929 ± 0. 009 0. 938 ± 0. 056 0. 902 ± 0. 072 0. 845 ± 0. 111
EfficientNetB0 3 3 0. 957 ± 0. 013 0. 927 ± 0. 021 0. 882 ± 0. 032 0. 913 ± 0. 088 0. 866 ± 0. 109 0. 786 ± 0. 170
EfficientNetB2 2 1 0. 972 ± 0. 001 0. 955 ± 0. 001 0. 904 ± 0. 035 0.987±0.001 0.981±0.002 0. 956 ± 0. 003
EfficientNetB2 3 1 0. 958 ± 0. 003 0. 929 ± 0. 007 0. 888 ± 0. 005 0. 974 ± 0. 004 0. 959 ± 0. 014 0. 920 ± 0. 021
VGG16 2 0 0. 940 ± 0. 001 0. 927 ± 0. 025 0. 883 ± 0. 003 0. 968 ± 0. 006 0. 938 ± 0. 016 0. 932 ± 0. 015
VGG16 3 0 0. 935 ± 0. 002 0. 934 ± 0. 009 0. 893 ± 0. 014 0. 982 ± 0. 003 0. 980 ± 0. 002 0.974±0.003

Table 6: Resultados experimentos de test en BigTobacco
Tobacco800 Dataset
Entrenamiento sin
filtrar datos
Entrenamiento con
datos filtrados
Modelo Imagen Paginas
Llamadas
Distil BERT Accuracy F1 Score Kappa Accuracy F1 Score Kappa
EfficientNetB0 2 1 0. 918 ± 0. 005 0. 948 ± 0. 003 0.757±0.011 0. 909 ± 0. 007 0. 942 ± 0. 005 0.730±0.012
EfficientNetB0 2 2 0. 880 ± 0. 001 0. 924 ± 0. 002 0. 644 ± 0. 013 0. 869 ± 0. 002 0. 916 ± 0. 002 0. 624 ± 0. 003
EfficientNetB0 3 1 0. 921 ± 0. 007 0. 953 ± 0. 004 0. 694 ± 0. 023 0. 909 ± 0. 014 0. 946 ± 0. 008 0. 641 ± 0. 053
EfficientNetB0 3 2 0.921±0.008 0.953±0.005 0. 696 ± 0. 034 0. 870 ± 0. 032 0. 921 ± 0. 020 0. 535 ± 0. 092
EfficientNetB0 3 3 0. 887 ± 0. 007 0. 935 ± 0. 004 0. 502 ± 0. 035 0. 862 ± 0. 023 0. 918 ± 0. 013 0. 462 ± 0. 093
EfficientNetB2 2 1 0. 908 ± 0. 006 0. 942 ± 0. 004 0. 717 ± 0. 014 0. 878 ± 0. 014 0. 922 ± 0. 012 0. 643 ± 0. 011
EfficientNetB2 3 1 0. 893 ± 0. 030 0. 935 ± 0. 021 0. 630 ± 0. 052 0.911±0.001 0.948±0.001 0. 655 ± 0. 010
VGG16 2 0 0. 910 ± 0. 002 0. 941 ± 0. 001 0. 744 ± 0. 001 0. 890 ± 0. 015 0. 927 ± 0. 012 0. 700 ± 0. 020
VGG16 3 0 0. 891 ± 0. 004 0. 935 ± 0. 003 0. 581 ± 0. 010 0. 892 ± 0. 007 0. 936 ± 0. 004 0. 580 ± 0. 020

Table 7: Resultados experimentos de test en Tobacco800
5.1 An ́alisis de Resultados BigTobacco
Como se puede observar en la tabla 6 en todos los par ́ametros estudiados los valores m ́aximos
siempre los alcanzan diferentes modelos propuestos por nosotros, superando as ́ı el estado del arte
que ha existido ahora siendo este VGG16. Adem ́as, nuestro modelo propuesto contiene por parte
de distil BERT 66 millones de par ́ametros y por parte de EfficientB2 9.2 millones de par ́ametros,
que compar ́andolo con VGG16, la cual tiene 130 millones de par ́ametros, podemos ver como hemos
reducido en 54 millones el n ́umero de par ́ametros del modelo, reduciendo el espacio que ocupa este
y teniendo margen para aumentar la potencia de EfficientNet.

Si observamos las tendencias de estos podremos visualizar como en nuestros modelos propuestos
cuantas m ́as llamadas a distil BERT, menor elF1 Scorelogrado, dejando claro que distil BERT
se beneficia de una ́unica llamada donde pueda tener acceso a toda la informaci ́on textual, aunque
este efecto es visible no perjudica en gran medida el rendimiento general del modelo. Lo observado
tambi ́en se aplica alaccuracyy elkappaque como se puede observar este es ́el m ́as variabilidad
tiene de todos los par ́ametros.

Cuando analizamos los resultados de VGG16, podemos ver que esta obtiene unos resultados
deaccuracymuy elevados, pero unas puntuaciones enF1 Scorem ́as reducidas, excepto en el uso de
3 p ́aginas con datos filtrados como se puede apreciar, por lo que es posible que est ́e sucediendo lo
mismo que se planteaba en el ejemplo cuando se explicaba el concepto deaccuracyy su problema
principal para esta tarea, aunque se puede observar comoaccuracyyF1 Scoresiguen la misma
tendencia en todos los modelos mostrados.

Por otro lado, comparamos los resultados con el dataset filtrado respecto el que no se ha filtrado
se puede observar como tanto elaccuracy,F1 scoreykappase ven reducidos en la secci ́on donde
el dataset ha sido filtrado, mostrando como los modelos estaban sacando ventaja del desbalance
de clases mostrado en secciones anteriores y as ́ı obteniendo unos resultados mejores que los reales,
aunque como se puede ver en la tabla hay casos donde no ha sido como se menciona y aun existiendo
este factor la diferencia de resultados en algunas pruebas es m ́ınima.

Por otra parte, si comparamos los modelos que hacen empleo de EfficientNetB2 con Efficient-
NetB0 para imagen podemos ver una clara subida de la precisi ́on obtenida, esta se puede observar en
el modelo entrenado con 3 p ́aginas y 1 llamada a distil BERT cuando el dataset ha sido balanceado,
lo que nos indica claramente que el papel de la convolucional sigue teniendo un rol fundamental
aun cuando esta se combina con una red especializada en el an ́alisis textual.

En cuanto a la estabilidad de los resultados, se puede observar como los modelos entrenados
en datos filtrados, son en lo general m ́as estables que los modelos entrenados en datos que no se
han filtrado, aunque teniendo en cuenta la distribuci ́on de los datos cuando no se han filtrado, es
bastante probable que el modelo se est ́e beneficiando de este debalance no solo en resultados, sino
en estabilidad tambi ́en. Aunque al igual que suced ́ıa con los resultados obtenidos, la diferencia de
estabilidad en los resultados no es muy pronunciada, aunque si visible.

Los mejores modelos obtenidos ser ́ıan o bien EfficientNetB0 con 3 p ́aginas y 1 llamada a distil
BERT para los modelos entrenados con datos sin filtrar o EfficientNetB2 con 2 p ́aginas y 1 llamada
a distil BERT siendo este el mejor de los modelos obtenidos que se han entrenado con los datos

filtrados.

5.2 An ́alisis de Resultados Tobacco800
En la tabla 7 podemos observar como los resultados son bastante m ́as incoherentes, esto se puede ver
claramente cuando comparamos los modelos que hacen empleo de EfficientNetB0 y EfficientNetB2,
1 sola llamada a distil BERT y 2 p ́aginas, obteniendo peores resultados con un modelo claramente
m ́as potente. Esto no es de extra ̃nar debido a que como hemos mencionado previamente a pesar
de utilizar todo al dataset como un test, y, por lo tanto, evitar el problema de que dependiendo de
como dividi ́eramos el dataset los resultados se ver ́ıan gravemente afectados, como ya se presenta en
trabajos previos, sino que adem ́as tiene un problema de posible DataLeak, donde hay p ́aginas que
son tan similares que es posible que el modelo se beneficie o perjudique de esta similitud. Por lo
que las tendencias mostradas en la tabla de resultados anterior, aqu ́ı se siguen viendo pero menos
marcadas.

Aunque si analizamos nuestros resultados con la secci ́on de datos filtrados y sin filtrar, en este
caso podemos observar, nuevamente con excepciones, como los modelos entrenados con los datos sin
filtrar son mejores que los datos filtrados, aunque viendo como el modelo VGG16 con datos filtrados
obtiene resultados notablemente inferiores a los otros modelos en esta misma categor ́ıa, nos hace
pensar que los resultados obtenidos por parte del proyecto de investigaci ́on que actualmente se
encuentra en el estado del arte, se ha visto influenciado por caracter ́ısticas comoDataLeaky el
efecto de particionar el dataset, que tiene Tobacco800.

Por otra parte, al igual que en la tabla anterior, el par ́ametro Kappa es el que presenta una
mayor variabilidad, obteniendo resultados que van des de un 50% a resultados que alcanzan el 75%,
aunque estos son notablemente menores que los observados en BigTobacco.

Si nos centramos en la estabilidad de los resultados, es aqu ́ı donde vemos que todos los resulta-
dos, independientemente de si se han entrenado o no con datos filtrados, muestran una desviaci ́on
muy leve, lo que no nos permite poder comparar los resultados de forma m ́as directa, aunque
teniendo en cuenta los problemas que este dataset presenta.

Si quisi ́eramos seleccionar los mejores modelos obtenidos estos ser ́ıan EfficientNetB0, con 3
p ́aginas y 2 llamadas para los modelos entrenados con datos sin filtrar y EfficientNetB0, con 3
p ́aginas y 1 llamada para los modelos entrenados con los datos filtrados, ya que si bien hay modelos
que alcanzan unF1 Scorem ́as elevado como ya hemos explicado nos basamos en F1 Score como
m ́etrica m ́as importante.

6 Gesti ́on del proyecto
6.1 Obst ́aculos y riesgos previstos
Por supuesto al igual que en todo proyecto existen riesgos y posibles obst ́aculos que uno se puede
encontrar por el camino, y en este caso no es diferente, por el hecho de que al haber muchos
factores en juego no es muy dif ́ıcil que haya alg ́un contratiempo en alg ́un momento del proyecto,
a continuaci ́on menciono los principales obst ́aculos y riesgos que m ́as potencial tienen de causar
alg ́un problema.

Teniendo en cuenta que ya ha habido otros equipos que han intentado una aproximaci ́on a
este problema, probablemente la obtenci ́on de los datos no sea un problema, no obstante
eso no niega el hecho de que de forma muy segura los datos no est ́en preparados para ser
introducidos directamente a la red neuronal, por lo que seguramente sea necesario hacer un
pre procesado intensivo a los datos.
Una vez tengamos todos los datos y el modelo, estos deben de ser introducidos en la memoria
de la GPU que usemos, la cual es limitada, por lo que en caso de que los modelos sean demasi-
ado complejos, se tendr ́an que buscar alternativas m ́as simples, aunque esta simplificaci ́on si
no es llevada a cabo correctamente, podr ́ıa dejarnos en una situaci ́on donde el modelo no es
capaz de aprender la tarea que tiene que llevar a cabo.
Aunque el modelo y los datos sean escogidos correctamente, es probable que otros factores
como el propio proceso de entrenamiento del modelo acabe en fracaso, no por la imple-
mentaci ́on por mi parte, sino por la propia naturaleza del ́ambito en el que nos encontramos.
6.2 Seguimiento realizado del proyecto
Para la realizaci ́on de este proyecto, debido a que las pruebas pueden llevar diversas horas o incluso
d ́ıas, se tendr ́a que optar por un m ́etodo ́agil y flexible, donde pueda a ̃nadir cambios, comprobarlos
y decidir si utilizo o no estos cambios.

Por lo que la intenci ́on es concretar una reuni ́on semanal con el director del proyecto de forma
semanal para poder tomar decisiones r ́apidamente en cualquier situaci ́on con el m ́ınimo tiempo de
latencia posible.

6.3 Herramientas de control empleadas
Para facilitar el seguimiento de las tareas se har ́a uso de Slack [34]. Se trata de una herramienta
online, donde se pueden enviar mensajes de manera instant ́anea para mantener el contacto en todo
momento, as ́ı como para hacer un control de las tareas pendientes y los resultados obtenidos.

En cuanto al control de versiones se har ́a empleo de la plataforma Github [35], ya que esta
permitir ́a llevar un control del c ́odigo y tener una copia de seguridad en todo momento para evitar
la perdida de informaci ́on.

Y finalmente para planificar el tiempo necesario a emplear en cada tarea har ́e uso de Gantter
[36], porque gracias al uso de un Gantt se podr ́a tener una primera idea de la panificaci ́on temporal
del proyecto.

6.4 Planificaci ́on temporal
A continuaci ́on se mostrar ́a la planificaci ́on temporal elaborada teniendo en cuenta el tiempo
disponible y las tareas a realizar.

El proyecto empieza el 13 de septiembre de 2021 y la finalizaci ́on est ́a prevista para el 27 de
enero de 2022. Por lo que el estimado de d ́ıas del proyecto es de 133 d ́ıas con un total de 6h diarias
dedicadas tendr ́ıamos 798 horas de dedicaci ́on totales.

6.4.1 Descripci ́on de tareas.

A continuaci ́on se describen las tareas a realizar y en que consiste cada una de ellas. Aunque para
un mejor entendimiento y organizaci ́on del documento las tareas han sido distribuidas en grupos
para as ́ı poder organizarlas f ́acilmente.

(TP) Trabajo Previo

En esta secci ́on se encuentran las tareas que he ejecutado previas al inicio de GEP y del TFG como
tal, las cuales, pese a ser superficiales, son una primera aproximaci ́on al problema.

TP.0 Investigaci ́on estado del arte: En este apartado he explorado las ́ultimas aplicaciones
hasta la fecha con el objetivo de tener un primer conocimiento de por donde empezar a
elaborar mi plan de ruta y tambi ́en para saber si la tarea es en primer lugar posible.
TP.1 Preparaci ́on entorno de trabajo: En este campo concretamente donde cada d ́ıa hay
novedades las librer ́ıas se ven constantemente actualizadas d ́ıa a d ́ıa, y, por lo tanto, cada d ́ıa
se generan nuevas incompatibilidades lo cual puede ser bastante problem ́atico.
(GP) Gesti ́on del proyecto

Esta secci ́on est ́a dedicada a tareas m ́as relacionadas con la gesti ́on del proyecto, ya que esta gesti ́on
es necesaria si quiero llevar el proyecto al d ́ıa y actualizado. Por lo que como se puede intuir, gran
parte de GEP est ́a contenido dentro de esta secci ́on.

GP.0 Reuniones: Como es habitual en estos proyectos para el correcto seguimiento de la
evoluci ́on del trabajo, se concertar ́an reuniones semanales con el director del TFG para que
este tenga siempre la informaci ́on disponible en todo momento y tambi ́en para corregir el
rumbo en caso de que fuese necesario.
GP.1 Documentaci ́on: Al igual que las reuniones se ir ́an produciendo hasta la finalizaci ́on de
este proyecto lo mismo ocurre con la documentaci ́on la cual se ir ́a actualizando semanalmente
para as ́ı tener los documentos actualizados en todo momento.
GP.2 Alcance: Se estudia el alcance que tendr ́a el proyecto, es decir que objetivos tendr ́a,
que se pretende hacer y los medios necesarios. Por supuesto para esto se tiene que haber
investigado sobre el estado del arte para tener un punto de partida.
GP.3 Planificaci ́on: Como en todo proyecto la planificaci ́on es esencial para que este sea
organizado y estructurado, por lo que este tramo ser ́a invertido en la estructuraci ́on del
mismo.
GP.4 Presupuesto: Quiz ́as uno de los aspectos m ́as importantes en el d ́ıa a d ́ıa de todo
proyecto, en este intervalo de tiempo se estudiar ́a el presupuesto necesario para llevar a cabo
este proyecto.
GP.5 Sostenibilidad: Quiz ́as uno de los apartados m ́as ignorados, pero m ́as relevantes de cara
a un futuro cada vez m ́as inmediato aqu ́ı se estudiar ́a el impacto medioambiental que este
proyecto puede tener.
GP.6 Presentaci ́on: En estos d ́ıas se preparar ́a una presentaci ́on oral para que un jurado
gestionado por la FIB pueda evaluar el trabajo hecho por el alumno.
(PT) Preparaci ́on del dataset

Esta secci ́on se centra en la preparaci ́on del dataset o conjunto de datos, ya que si bien la con-
strucci ́on del modelo es una etapa crucial, puesto que es el momento en el que estamos construyendo
el n ́ucleo del proyecto, la construcci ́on del dataset es bien igual o m ́as importante.

PT.0 Analisis de la informaci ́on: Aqu ́ı se estudiar ́an los datos de los datasets disponibles de
forma p ́ublica para, bas ́andonos en la informaci ́on que cada dataset nos provea, decidir cu ́al
parece ser m ́as prometedor y, por lo tanto, cu ́al deber ́ıamos usar.
PT.1 Prueba Mejora calidad de las im ́agenes: En esta secci ́on se estudiar ́a si es posible hacer
un pre procesado a las im ́agenes con el objetivo de que estas se vean con mejor calidad y
mayor detalle.
PT.2 Creaci ́on OCR: Se crea el OCR (o texto) para cada imagen en caso de que no se encuentre
disponible.
PT.3 Prueba ficheros H5DF en rendimiento y memoria: Pruebas sobre la mejor manera de
estructurar los datos dentro de estos ficheros H5DF.
PT.4 Creaci ́on ficheros H5DF: Generamos los ficheros H5DF, los cuales los emplearemos para
cargar los datos durante el entrenamiento (se usa este tipo de formato, ya que son altamente
eficientes).
(CR) Creaci ́on del modelo

En esta secci ́on se analiza los pasos en la creaci ́on del modelo que entrenaremos.

CR.0 Elecci ́on de modelo para imagen: An ́alisis y b ́usqueda de las arquitecturas m ́as utilizadas
para este tipo de problemas en relaci ́on con la imagen.
CR.1 Elecci ́on de modelo para texto: An ́alisis y b ́usqueda de las arquitecturas m ́as empleadas
para este tipo de problemas en relaci ́on con el texto.
CR.2 Estudio arquitectura del modelo: Se estudia como unir los modelos en un ́unico modelo
optimizando lo m ́aximo posible el tama ̃no de la arquitectura y laaccuracy.
CR.3 Uni ́on de modelos y primeros entrenamientos de prueba: Una vez unidos los modelos se
entrena el conjunto para ver los resultados.
(MGPU) Implementaci ́on Multi GPU

En esta secci ́on se implementar ́a una versi ́on paralelizada del modelo.

MGPU.0 Adaptaci ́on del bucle de entrenamiento: Adaptar el bucle de entrenamiento para
que se pueda hacer de forma paralela
MGPU.1 Entrenamiento: Entrenamiento del modelo y visualizaci ́on de resultados.
(IMA) Implementacion pipeline y modelo Estado del Arte

En esta secci ́on se implementar ́a una versi ́on paralelizada del modelo que representa el estado del
arte actual para poder comparar los resultados obtenidos de este modelo con nuestras propuestas.

AD.0 Adaptaci ́on del problema: Aplicar los cambios necesarios para aplicar el modelo y el
pipeline.
AD.1 Modificaci ́on del modelo: Crear el modelo propuesto en el estado del arte.
AD.2 Entrenamiento: Entrenar el modelo para ver resultados.
6.4.2 Recursos

6.4.3 Recursos Humanos

En este proyecto se encuentran se pueden encontrar 3 roles diferentes, el de jefe de proyecto,
investigador y programador, las tareas de cada uno de estos es la siguiente:

Jefe de proyecto: Encargado de dirigir el rumbo del proyecto en todo momento y de controlar
la calidad de este, as ́ı como las fechas establecidas (Jf).
Investigador: Encargado de obtener informaci ́on sobre el estado del arte y todo lo relevante
a las ́ultimas t ́ecnicas usadas en los ́ultimos a ̃nos para poder aplicarlas en este proyecto (In).
Programador: Encargado de programar toda la parte de c ́odigo con tal de que todo funcione
correctamente (Pr).
6.4.4 Recursos materiales

A continuaci ́on se listan los recursos que ser ́an necesarios en cada etapa, siendo los recursos
disponibles y necesarios los siguientes:

Ordenador de altas prestaciones con las siguientes prestaciones:
4xGPUs Tesla-V100
2 x IBM Power9 8335-GTH @ 2.4GHz
512GB de memoria RAM distribuida en 16 dimms x 32GB @ 2666MHz
2 x SSD 1.9TB as local storage
6.4.5 Gesti ́on de riesgos

Al tratarse de una implementaci ́on que hasta d ́ıa de hoy no se ha hecho, es posible que surjan diversos
errores durante el desarrollo del proyecto, a continuaci ́on se describen los posibles obst ́aculos que
podr ́ıa ir enfrentando a lo largo del desarrollo y sus soluciones.

Incompatibilidad de librer ́ıas: Si bien es cierto que parte del trabajo previo ha sido la config-
uraci ́on del entorno, es posible que en alg ́un momento del desarrollo necesite de alguna otra
herramienta, la cual sea incompatible. En caso de que esto ocurra se deber ́an no solo actualizar
las librer ́ıas, sino entrenar los modelos nuevamente, ya que es posible que debido al cambio
de versi ́on haya cambios sutiles que no podamos visualizar sin una b ́usqueda exhaustiva.
Entrenamientos fallidos: Es posible que debido al ́ambito en el que nos estamos moviendo
algunos entrenamientos resulten exitosos, mientras que otros fracasen, por lo que de ocurrir
este error se deber ́ıa de volver a entrenar a la red para confirmar si esta es nuestra situaci ́on.
Arquitectura o hyperparametros err ́oneos: Es perfectamente posible que bien debido a la ar-
quitectura o a los hyperparametros el entrenamiento fracase, este hecho no se podr ́a confirmar
hasta el momento en el que construyamos la arquitectura, por lo que en caso de que ocurra
se deber ́an buscar r ́apidamente una soluci ́on.
Dataset con errores: Si el dataset escogido contiene errores no detectados inicialmente, como
un mal balanceo de ejemplos o cualquier otro detalle que pase desapercibido se deber ́a o bien
corregir estos errores o bien cambiar de dataset.
ID Tarea
Tiempo
(horas) Dependencia Recursos Roles
TP Trabajo Previo 177
TP.0 Investigaci ́on estado del arte 33 Port ́atil In
TP.1 Preparaci ́on entorno del trabajo 141 Port ́atil Pr
GP Gesti ́on de Proyectos 798
GP.0 Reuniones 775 Port ́atil Jf,In,Pr
GP.1 Documentaci ́on 775 Port ́atil In
GP.2 Alcance 37 TP.0 Port ́atil In
GP.3 Planificaci ́on 40 GP.2 Port ́atil, Grannter [36] Jf
GP.4 Presupuesto 19 GP.3 Port ́atil Jf
GP.5 Sostenibilidad 15 GP.4 Port ́atil Jf
GP.6 Presentaci ́on 21 GP.1 Port ́atil Jf
PT Preparaci ́on dataset 228
PT.0 Analisis de la informaci ́on 30 TP.1 Port ́atil In
PT.1 Pruebas Mejora Calidad de Im ́agenes 30 PT.0 1xGPU Tesla-V100
+ 1 x IBM Power9 8335-GTH @ 2.4GHz
Pr
PT.2 Creaci ́on OCR 60 PT.1 1 x IBM Power9 8335-GTH @ 2.4GHz Pr
PT.3 Prueba ficheros H5DF en
rendimiento y memoria
48 PT.2 1 x IBM Power9 8335-GTH @ 2.4GHz Pr
PT.4 Creaci ́on ficheros H5DF 60 PT.3 1 x IBM Power9 8335-GTH @ 2.4GHz Pr
CR Creaci ́on del modelo 132
CR.0 Elecci ́on de modelo para imagen 30 PT.4 Port ́atil In
CR.1 Elecci ́on de modelo para texto 30 CR.0 Port ́atil In
CR.2 Estudio arquitectura del modelo 36 CR.1 Port ́atil In
CR.3 Uni ́on de modelos y
primeros entrenamientos de prueba
36 CR.2 1xGPU Tesla-V100
+ 1 x IBM Power9 8335-GTH @ 2.4GHz
Pr
MGPU Implementaci ́on Multi GPU 72
MGPU.0 Adaptaci ́on del bucle de entrenamiento 36 CR.3 Port ́atil In

MGPU.1 Entrenamiento 36 MGPU.0 4xGPU Tesla-V100

2 x IBM Power9 8335-GTH @ 2.4GHz
Pr
IMA
Implementacion pipeline y modelo
Estado del Arte^138
IMA.0 Adaptaci ́on del problema 72 MGPU.1 Port ́atil In
IMA.1 Modificaci ́on del modelo 30 IMA.0 Port ́atil In
IMA.2 Entrenamiento 36 IMA.1
1xGPU Tesla-V100
+ 1 x IBM Power9 8335-GTH @ 2.4GHz Pr
Total 798
Table 8: Tareas con sus tiempos y cargos encargados de cada tarea.
6.5 Gesti ́on econ ́omica
6.5.1 Presupuesto

Coste de Personal

Bas ́andonos en la planificaci ́on se definen los costes del personal, teniendo en cuenta los roles desig-
nados en anteriormente en el apartado de recursos humanos, siendo estos: jefe de proyecto, investi-
gador y programador. En la tabla 9 se pueden observar los costes por hora seg ́un las observaciones
hechas en la empresa de reclutamientoHays[37].

Rol Coste por hora
Jefe de equipo 30 €
Investigador 20 €
Programador 16 €
Table 9: Costes de personal dependiendo de su rol.
En la siguiente secci ́on se detallan los costes de cada etapa seg ́un los roles implicados en cada
etapa, as ́ı como el coste total del proyecto el cual se estima multiplicando el coste por 1,3. En total
el coste del proyecto es de 33.872€.

ID Tarea Tiempo
(horas)
Coste Coste + SS Roles
TP Trabajo Previo 177 2.916€ 3.791€
TP.0 Investigaci ́on estado del arte 33 660 € 858 € In
TP.1 Preparaci ́on entorno del trabajo 141 2.256€ 2.933€ Pr
GP Gesti ́on de Proyectos 798 6.090€ 7.917€
GP.0 Reuniones 30 660 € 858 € Jf,In,Pr
GP.1 Documentaci ́on 70 1470 € 1911 € Jf, In
GP.2 Alcance 37 1110 € 1443 € Jf
GP.3 Planificaci ́on 40 1200 € 1560 € Jf
GP.4 Presupuesto 19 570 € 741 € Jf
GP.5 Sostenibilidad 15 450 € 585 € Jf
GP.6 Presentaci ́on 21 630 € 819 € Jf
PT Preparaci ́on dataset 228 3.768€ 4.898€
PT.0 Analisis de la informaci ́on 30 600 € 780 € In
PT.1 Pruebas Mejora Calidad de Im ́agenes 30 480 € 624 € Pr
PT.2 Creaci ́on OCR 60 960 € 1.248€ Pr
PT.3
Prueba ficheros H5DF en
rendimiento y memoria^48768 €^998 € Pr
PT.4 Creaci ́on ficheros H5DF 60 960 € 1.248€ Pr
CR Creaci ́on del modelo 132 2.496€ 3.245€
CR.0 Elecci ́on de modelo para imagen 30 600 € 780 € In
CR.1 Elecci ́on de modelo para texto 30 600 € 780 € In
CR.2 Estudio arquitectura del modelo 36 720 € 936 € In
CR.3
Uni ́on de modelos y
primeros entrenamientos de prueba^36576 €^749 € Pr
MGPU Implementaci ́on Multi GPU 72 1.296€ 1.685€
MGPU.0 Adaptaci ́on del bucle de entrenamiento 36 720 € 936 € In
MGPU.1 Entrenamiento 36 576 € 749 € Pr

IMA Implementacion pipeline y modelo
Estado del Arte
138 2.616€ 3.401€
IMA.0 Adaptaci ́on del problema 72 1.440€ 1.872€ In
IMA.1 Modificaci ́on del modelo 30 600 € 780 € In
IMA.2 Entrenamiento 36 576 € 749 € Pr
Total 19.182€ 24.937€

Table 10: Coste asociado a cada etapa del proyecto seg ́un el personal encargado
Costes gen ́ericos

Debido a la situaci ́on actual con Covid, actualmente el trabajo se est ́a ejerciendo de forma telem ́atica,
con posibilidad de ir presencialmente de lunes a viernes a las oficinas. Como ambos espacios se
encuentran en Barcelona se ha hecho una estimaci ́on del coste del espacio decoworkingsiendo este
un total 220€, el cual incluye electricidad, agua, una mesa individual y acceso todos los d ́ıas de la
semana. Por lo que a los 5 meses de finalizar el proyecto ser ́ıan 1100€.

El software utilizado no tiene ning ́un coste aplicado al ser este p ́ublico, as ́ı como el dataset el
cual tambi ́en es de dominio p ́ublico y sin coste.

En la tabla 10 se presentan los costes en cada tr ́amite seg ́un el personal encargado de cada
tarea, de esta manera como su coste aplicando la seguridad social.

Adem ́as del coste de los trabajadores por cada tramo en el proyecto tambi ́en se debe tener
en cuenta el coste del hardware usado, ya que este ser ́a necesario para llevar a cabo la tarea. As ́ı
mismo se ha calculado la amortizaci ́on de cada pieza de hardware haciendo un estimado de la
esperanza de vida de cada uno, teniendo en cuenta que al a ̃no hay 220 d ́ıas h ́abiles de 8 horas
diarias de trabajo, siendoAmortizaci ́on=V idaHorasutil ́ ∗dde ́ıasusohabiles∗Coste∗horasdispositivolaboralesla f ́ormula utilizada
para saber la amortizaci ́on. En la tabla 11 se pueden observar el precio de los materiales necesarios,
las unidades, una estimaci ́on de la vida ́util de cada uno y las horas de uso que se indican en la
planificaci ́on laboral.

Hardware Precio Unidades Vida ́util Horas Amortizaci ́on
Portatil 700 € 1 4 a ̃nos 530h 52 €
Ordenador de
altas prestaciones
29.889€^11 4 a ̃nos 504h 2139 €
Table 11: Presupuesto del hardware utilizado.
Contingencia

Como en todo proyecto, siempre puede haber imprevistos u obst ́aculos por el camino que dificulten
la completaci ́on de una tarea y, por lo tanto, un gasto inesperado, por lo que es aconsejable hacer
un plan de contingencia dejando un margen por si surgiera cualquier imprevisto.

Como la probabilidad de encontrar alg ́un problema es alta, establecer ́e un 20% de margen por
si surgiera alg ́un imprevisto. En la tabla 12 se puede observar el total de contingencia para cada
tipo de gasto.

(^1) Para obtener el coste del Ordenador en altas prestaciones me he basado en el servicio de AWS[38] donde he
escogido un equipo con las mismas prestaciones necesarias que el usado para hacer este proyecto.

Tipo Coste Contingencia
Espacio 1.100€ 220 €
Hardware 30.589€ 6.117€
Personal 24.937€ 4.987€
Total 56.626€ 11.324€
Table 12: Contingencia del 20% por cada tipo de gasto.
Imprevistos

A continuaci ́on se mencionan los imprevistos que podr ́ıa haber durante el desarrollo del proyecto,
los cuales ya se comentaron en la planificaci ́on temporal. En esta secci ́on se estimar ́a su coste en
t ́erminos de dinero y tiempo para cada uno de ellos. En la tabla 13 se puede observar con detalle
las probabilidades y costos asociados a cada riesgo.

Incompatibilidad de librer ́ıas: En caso de encontrarnos en esta situaci ́on, se deber ́an volver a
ejecutar, solo en los casos m ́as extremos, pero no imposibles, todos los experimentos para as ́ı
confirmar que el cambio de versiones de librer ́ıa no afecta al rendimiento de los modelos, por
lo que se deber ́an a ̃nadir 72 horas de entrenamiento, los costes de hardware ser ́ıan 172€y los
costes del programador serian 384€. El riesgo de este caso es del 40%, ya que pese a que hay
mucha gente que hace uso de estas librer ́ıas al ser actualizadas debido a los avances diarios
siempre pueden surgir imprevistos
Entrenamientos fallidos: En este caso se deber ́an volver a hacer como m ́ınimo un entre-
namiento m ́as, para confirmar si se trata de la arquitectura del modelo o de un fallo en el
proceso de entrenar el propio modelo. Los costes en horas serian de 48h, los costes de hard-
ware serian 115€y los costes en personal ser ́ıan 256€. El riesgo de este caso es del 10%
porque si bien puede ocurrir no suele ser el caso y se puede detectar r ́apidamente.
Arquitectura o hyperparametros err ́oneos: Este caso ser ́ıa el mismo que el anterior, pero esta
vez ser ́ıa a causa del propio programador, los costes en tiempo y dinero serian los mismos. El
riesgo en este caso es del 30%, puesto que como se est ́a aproximando el problema de una forma
donde se combinan redes que nunca antes se ha probado es posible que surjan complicaciones
en el proceso.
Dataset con errores: En este caso se deber ́ıa de contribuir 32h de revisi ́on del dataset y si
este no fuera viable debido a su gran cantidad de errores se deber ́ıan invertir 100h en volver
a buscar y tratar un dataset lo m ́as r ́apidamente posible. Por lo que teniendo en cuenta
que estas tareas son llevadas a cabo por un programador implicar ́ıa un coste de 512€en el
primer caso donde el dataset fuera utilizable y un coste de 1.600€en caso de que se tuviera
que volver a hacer todo el procedimiento. El riesgo de este caso es del 5% porque la mayor ́ıa
de datasets p ́ublicos son conocidos y ya han sido verificados por la comunidad.
Fallo en el ordenador de altas prestaciones: En este equipo se dispone de una copia de seguri-
dad por lo que el fallo de este dispositivo supondr ́ıa 5h de tiempo perdido para restaurar el
entorno y un coste de personal de 80€. El riesgo de este caso es del 2% porque al tratarse de
un equipo de altas prestaciones este ha pasado unas pruebas muy rigurosas de calidad.
Fallo del port ́atil: En el caso del port ́atil no se dispone de datos relevantes, pero el reemplazo
de este supondr ́ıa una perdida de tiempo de aproximadamente 48h y un coste de 500€. El
riesgo de este caso es del 5%.
Imprevisto Coste Riesgo Coste total
Incompatibilidad de librer ́ıas 556 € 40% 222.4€
Entrenamientos fallidos 371 € 10% 37.1€
Arquitectura o hyperparametros err ́oneos 371 € 10% 37.1€
Dataset con errores 512 € 5% 25.6€
Fallo en el ordenador de altas prestaciones 80 € 2% 1.6€
Fallo del port ́atil 500 € 5% 25 €
Total 2.390€ 348,8€
Table 13: Riesgos para los posibles imprevistos y costes asociados
Coste Total

Con todo lo que hemos visto el coste total del proyecto ascender ́ıa a 94.336€, en la tabla 14 se
pueden observar los detalles de cada coste.

Tipo Coste
Personal 24.937€
Espacio 1.100€
Hardware 56.626€
Contingencia 11.324€
Imprevistos 349 €
Total 94.336€
Table 14: Presupuesto final del proyecto
6.5.2 Control de gesti ́on

Una vez definido el presupuesto se definen mecanismos de control para controlar que no haya
perdidas en tiempos o dinero perdido al final del proyecto. De esta forma se podr ́a encontrar
r ́apidamente cualquier error.

Desviaci ́on coste personal por tarea: (coste estimado coste real) horas reales
Desviaci ́on realizaci ́on tareas: (horas estimadas horas reales) coste real
Desviaci ́on total en la realizaci ́on de tareas: coste estimado total coste real total
Desviaci ́on total de recursos (software, hardware, espacio o personal): coste estimado total
coste real total
Desviaci ́on total coste de imprevistos: coste estimado imprevistos coste real imprevistos
Desviaci ́on total de horas: horas estimadas horas reales
6.5.3 Imprevistos encontrados

Los imprevistos que nos hemos visto obligados a resolver y que han afectado a nuestra planificaci ́on
inicial ha sido la necesidad de volver a recrear el dataset debido a los fallos encontrados en este y
el re-entrenamiento de los modelos en este dataset, aunque esta situaci ́on ya se ha tenido en cuenta
en el apartado de imprevistos y ya se le ha asignado un presupuesto, por lo que la planificaci ́on
general no se ha visto afectada.

6.6 Sostenibilidad
A continuaci ́on se hace un estudio sobre la sostenibilidad en tres diferentes apartados, siendo es-
tos: econ ́omico, ambiental y social. Aunque previamente se har ́a una autoevaluaci ́on sobre los
conocimientos del autor de este trabajo sobre esta ́area.

6.6.1 Autoevaluaci ́on

A lo largo del Grado en ingenier ́ıa inform ́atica muchas veces nos han no solo recordado y recalcado,
sino tambi ́en ense ̃nado como de importante es la sostenibilidad de un proyecto y como de ineficientes
somos hoy en d ́ıa en cuanto a gesti ́on de recursos, ya que debido a esta cultura que tenemos de ”tirar
y usar” generamos una gran cantidad de residuos de los cuales la mayor ́ıa ni somos conscientes,
pese a que somos nosotros quienes generamos gran parte de estos.

Por mi parte, des de un punto de vista personal siempre he pensado que la sostenibilidad de
un proyecto es relevante, quiero decir si se va a hacer algo mejor hacerlo bien des del principio no?
Aunque por supuesto es m ́as f ́acil decirlo que hacerlo, ya que si bien pueden parecer palabras muy
f ́aciles de utilizar nunca he cre ́ıdo seriamente en estos t ́erminos y lo que implican.

Por lo que mientras hacia la encuesta proporcionada por los profesores me he dado cuenta de
la poca importancia, si no inexistente, que le daba a la sostenibilidad social y econ ́omica, y no
solo eso, sino tambi ́en el poco conocimiento que ten ́ıa sobre estas ́areas, considerando siempre en la
ambiental, la cual suele ser la m ́as conocida de todas, pese a que el apartado econ ́omico no se puede
obviar, ya que no importa lo noble que sea un proyecto si este no se puede mantener, entonces no
tiene sentido iniciarlo, al igual que si este no beneficia a la sociedad, no tiene sentido que sea llevado
a cabo.

Por lo que en este proyecto, se ha reflexionado y tenido en cuenta los ́ambitos m ́as conocidos
de la sostenibilidad como lo puede ser la dimensi ́on ambiental, as ́ı como otras ́areas que no siempre
se les da la importancia que merecen pese a que son igual de importantes, como lo son la econ ́omica
y la social.

6.6.2 Dimensi ́on Econ ́omica

En cuanto a la visi ́on econ ́omica, estamos ante un caso en el que al tratarse de las ́ultimas t ́ecnicas
enDeep Learning, los requisitos de los equipos necesarios para entrenar estas redes no es peque ̃no.

No obstante debido a la misma naturaleza de esta ́area, una vez entrenado nuestro modelo este
ya no tendr ́ıa la necesidad de volver a ser entrenado nuevamente, por lo que el coste de inferencia de
un modelo es despreciable, ya que estamos hablando de que un m ́ovil ya tiene la potencia necesaria
para hacer esta tarea.

Por lo que si bien la entrada tiene un alto coste, el posterior mantenimiento econ ́omico y
sostenibilidad econ ́omica se reducen pr ́acticamente a 0, mientras que se hace uso del modelo, y,

por lo tanto, se obtienen beneficios. Aunque no solo eso, una vez entrenado, se puede utilizar
posteriormente como base para pre entrenar para otras tareas reduciendo significativamente el
tiempo necesario para hacerlo y as ́ı reduciendo el coste econ ́omico de futuros proyectos.

6.6.3 Dimensi ́on Ambiental

Desgraciadamente el impacto que puede llegar a tener este proyecto en el medioambiente puede
llegar a ser notable, no porque el modelo sea directamente perjudicial o vaya en contra del mismo,
sino porque para entrenar cualquier modelo, debido a las prestaciones m ́ınimas del equipo, este
consumir ́a bastante energ ́ıa, lo cual se traduce en contaminaci ́on casi directa.

Por supuesto se intenta mitigar esto mirando de que las fuentes de donde se consume esta
energ ́ıa sean fuentes de energ ́ıa verde respetuosamente con el medio ambiente, tambi ́en se han
tomado medida para evitar en todo momento repetir experimentos innecesarios, ya que si bien
puede haber necesidad de repetir experimentos cuanto menos ocurra mejor.

Aun de esta forma, debido a que si queremos obtener, o como m ́ınimo intentar obtener, resul-
tados cercanos o superiores al estado del arte no tenemos otra opci ́on que recurrir a este tipo de
recursos.

6.6.4 Dimensi ́on Social

Los beneficios sociales, como ya he comentado previamente, son inmensos. Ya que para comenzar
atacamos un problema real y existente no solo en empresas sino en organismos p ́ublicos.

Aparte de eso como se ha comentado repetidas veces una vez entrenado el modelo, este puede
ser usado por cualquier persona en cualquier lugar sin ning ́un tipo de problema, aportando un
beneficio directo para esta persona. O bien se puede utilizar para el entrenamiento en otra ́area,
que haciendo empleo deltransfer learningpodr ́ıan obtener mejores resultados y verse beneficiados.

Por lo que en este proyecto el beneficio social es mucho mayor que en otros, ya que es un
beneficio directo para un p ́ublico muy amplio y donde no se necesitan m ́as recursos que un ordenador
dom ́estico para poder ponerlo a prueba.

7 Conclusi ́on
7.1 Conclusiones personales
En este proyecto se han planteado diversos modelos de deep learning que han hecho uso de con-
volucionales y Transformers, algo nunca antes hecho, para la tarea de segmentaci ́on de documentos
logrando resultados que superan al estado del arte actual.

Adem ́as, se ha hecho un estudio exhaustivo sobre los datasets seleccionados, gracias al cual se
ha detectado un problema, por lo que se ha ideado una soluci ́on para solucionarlo permitiendo as ́ı
entrenar nuestros modelos con y sin esta correcci ́on en el dataset para visualizar el impacto en el
rendimiento.

El c ́odigo utilizado se puede encontrar en el siguiente enlace:https://github.com/Asocsar/
Segmentacion-de-documentos-digitales.

7.2 Objetivos cumplidos
Todos los objetivos propuestos en un inicio han sido alcanzados con ́exito, ya que los modelos
construidos han sido capaces de sobrepasar no solo el 70% deaccuracyque inicialmente se plante ́o
como un m ́ınimo a lograr, sino que hemos sido capaces de obtener en diversas propuestas de modelo
un rendimiento superior al obtenido en trabajos de investigaci ́on previos que hasta el d ́ıa de hoy se
consideran el estado del arte en esta tarea de segmentaci ́on.

Y finalmente, hemos sido capaces de aplicar un m ́etodo de entrenamiento en paralelo que nos
ha permitido acortar los tiempos lo suficiente como para poder ser capaces de entrenar nuestros
modelos en menos de 48h.

7.3 Trabajo futuro
En estos 4 meses hemos invertido cientos de horas en desarrollar un modelo que sobrepase o iguale
al estado del arte actual, no solo para obtener resultados, sino tambi ́en para hacer una exploraci ́on
sobre el rendimiento de los Transformers en este tipo de tareas, demostrando resultados impresio-
nantes como ya hemos visto, por lo que en vista de esto nuestro trabajo est ́a lejos de terminar, en
vista del rendimiento de Transformers con convolucionales nos queda investigar el rendimiento de
hacer uso ́unicamente de Transformers. Despu ́es de un an ́alisis m ́as reciente se ha determinado que
el mejor candidato es LayoutLMV2, un modelo creado por Microsoft, por lo que en un futuro lle-
varemos a cabo las pruebas con este modelo para ver si es posible no tratar con redes convolucionales
y obtener un rendimiento equivalente en el problema de segmentaci ́on de documentos.

PDF Generated On: 10/04/2021, 16:44:16
Page 1 / 1
Implementación pipeline y modelo Estado del Arte
Implementación pipeline y modelo Estado del Arte

Figure 12: Gr ́afica de la pl ́anificaci ́on temporal del proyecto.
46
References
[1] “Barcelona supercomputing center.”https://www.bsc.es/, 2021. Visitado (27-09-2021).
[2] I. P. Z. S.-t. X. Zhong-Qiu Zhao, Member and F. Xindong Wu, “Object detec-
tion with deep learning: A review.” https://arxiv.org/pdf/1807.05511.pdf&usg=
ALkJrhhpApwNJOmg83O8p2Ua76PNh6tR8A, 2019. Visitado (21-09-2021).
[3] Y. L. Arjun Jain, Jonathan Tompson and C. Bregler, “Modeep: A deep learning framework
using motion features for human pose estimation.”https://arxiv.org/pdf/1409.7963.pdf,
Visitado (22-09-2021).
[4] R. T. Siddhant Srivastava, Anupam Shukla, “Machine translation : From statistical to modern
deep-learning practices.”https://arxiv.org/pdf/1812.04238.pdf, 2018. Visitado (22-09-
2021).
[5] T. K. W. Z.-C. Y. Chuanqi Tan, Fuchun Sun and C. Liu, “A survey on deep transfer learning.”
https://arxiv.org/pdf/1808.01974.pdf, 2018. Visitado (24-09-2021).
[6] “Capa densa.”https://towardsdatascience.com/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6.
[7] “Convolutional neural networks, explained.” https://towardsdatascience.
com/convolutional-neural-networks-explained-9cc5188c4939#:~:text=A%
20Convolutional%20Neural%20Network%2C%20also,topology%2C%20such%20as%20an%
20image.&text=Each%20neuron%20works%20in%20its,cover%20the%20entire%20visual%
20field.
[8] S. K. Mark HUGHE, Irene LI and T. SUZUMURA, “Medical text classification using convo-
lutional neural networks.”https://arxiv.org/pdf/1704.06841.pdf, 2017. Visitado (20-09-
2021).
[9] A. Sherstinsky, “Fundamentals of recurrent neural network (rnn) and long short-term memory
(lstm) network.”https://arxiv.org/pdf/1808.03314.pdf, 2021. Visitado (21-09-2021).
[10] N. P. J. U.-L. J. A. N. G. K. Ashish Vaswani, Noam Shazeer, “Attention is all you need.”
https://arxiv.org/pdf/1706.03762.pdf, 2017. Visitado (25-10-2021).

[11] A. K. D. W.-X. Z. T. U. M. D. M. M. G. H. S. G. J. U. N. H. Alexey Dosovitskiy, Lucas Beyer,
“An image is worth 16x16 words: Transformers for image recognition at scale.”https://
arxiv.org/pdf/2010.11929.pdf, 2021. Visitado (26-10-2021).

[12] N. R. M. S. J. K. P. D. A. N. P. S. G. S. A. A. S. A. A. H.-V. G. K. T. H. R. C. A. R.
D. M. Z. J. W. C. W. C. H. M. C. E. S. M. L. S. G. B. C. J. C. C. B. S. M. A. R. I.
S. D. A. Tom B. Brown, Benjamin Mann, “Language models are few-shot learners.”https:
//arxiv.org/pdf/2005.14165v4.pdf, 2020. Visitado (26-10-2021).

[13] “Understand the impact of learning rate on neural net-
work performance.” https://machinelearningmastery.com/
understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/.

[14] J. A. S. L. Fabricio Ataides Braz, Nilton Correia da Silva, “Leveraging effectiveness and
efficiency in page stream deep segmentation.”https://www.sciencedirect.com/science/
article/pii/S0952197621002426, 2020. Visitado (24-09-2021).

[15] A. Z. Karen Simonyan, “Very deep convolutional networks for large-scale image recognition.”
https://arxiv.org/pdf/1409.1556.pdf, 2015. Visitado (24-09-2021).

[16] G. H. Gregor Wiedemann, “Page stream segmentation with convolutional neural nets combin-
ing textual and visual features.”https://arxiv.org/pdf/1710.03006.pdf, 2019. Visitado
(23-09-2021).

[17] A. B. Chems Neche, Yolande Bela ̈ıd, “Use of language models for document stream segmenta-
tion.”https://hal.inria.fr/hal-02975046/document, 2020. Visitado (20-10-2021).

[18] A. U. Adam W. Harley and K. G. Derpanis, “The rvl-cdip dataset.”https://www.cs.cmu.
edu/~aharley/rvl-cdip/, 2017. Visitado (21-09-2021).

[19] S. Hoffstaetter, “Python tesseract.”https://github.com/madmaze/pytesseract, 2019. Vis-
itado (23-09-2021).

[20] J. C. T. W. Victor SANH, Lysandre DEBUT, “Distilbert, a distilled version of bert: smaller,
faster, cheaper and lighter.”https://arxiv.org/pdf/1910.01108.pdf, 2020. Visitado (25-
10-2021).

[21] M. Tan and Q. V. Le, “Efficientnet: Rethinking model scaling for convolutional neural net-
works.”https://arxiv.org/pdf/1905.11946v5.pdf, 2020. Visitado (23-10-2021).

[22] J. D. M.-W. C. K. L. K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for
language understanding.”https://arxiv.org/pdf/1810.04805.pdf, 2019. Visitado (24-10-
2021).

[23] J. D. Geoffrey Hinton, Oriol Vinyals, “Distilling the knowledge in a neural network.”https:
//arxiv.org/pdf/1503.02531.pdf, 2015. Visitado (25-10-2021).

[24] J. L. B. Diederik P. Kingma, “Adam: A method for stochastic optimization.”https://arxiv.
org/pdf/1412.6980.pdf, 2017. Visitado (04-10-2021).

[25] S. M. Herbert Robins, “A stochastic approzimation method.” https://
projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/
issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full, 1951.
Visitado (04-10-2021).

[26] Y. S. John Duchi, Elad Hazan, “Adaptive subgradient methods for online learning and
stochastic optimization.” https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.
pdf, 2011. Visitado (04-10-2021).

[27] A. Graves, “Generating sequences with recurrent neural networks.”https://arxiv.org/pdf/
1308.0850.pdf, 2014. Visitado (04-10-2021).

[28] D. X. A. C. Y. B. Ian J. Goodfellow, Mehdi Mirza, “An empirical investigation of catas-
trophic forgetting in gradient-based neural networks.”https://arxiv.org/pdf/1312.6211.
pdf, 2015. Visitado (26-10-2021).

[29] B. G. Sharath Sreenivas, Swetha Mandava and C. Forster, “Pretraining bert
with layer-wise adaptive learning rates.” https://developer.nvidia.com/blog/
pretraining-bert-with-layer-wise-adaptive-learning-rates/, 2019. Visitado
(21-10-2021).

[30] “Multiprocessing package - torch.multiprocessing.” https://pytorch.org/docs/stable/
multiprocessing.html.

[31] “Pytorch distributed.”https://pytorch.org/tutorials/beginner/dist_overview.html.

[32] “Getting started with distributed data parallel.” https://pytorch.org/tutorials/
intermediate/ddp_tutorial.html.

[33] L. J. E. S. Xiangyang Xu, Shengzhou Xu, “Characteristic analysis of otsu thresh-
old and its applications.” https://www.sciencedirect.com/science/article/abs/pii/
S0167865511000365, 2011. Visitado (24-09-2021).

[34] “Slack.”https://slack.com/. Visitado (24-09-2021).

[35] “Github.”https://github.com/, 2021. Visitado (27-09-2021).

[36] “Gantter.”https://www.gantter.com/, 2021. Visitado (27-09-2021).

[37] “Hays.”https://www.hays.es/, 2021. Visitado (04-10-2021).

[38] “Aws.”https://calculator.aws/#/, 2021. Visitado (06-10-2021).
