# Ideas costeras

## Info municipal base

Protección civil en Veracruz consideró que sería útil para autoridades y público en general, contar con recursos documentales accesibles para comprender los procesos que convergen en la zona costera. Así surgió  esta colección sobre [municipios costeros](https://www.flipsnack.com/deblith/municipios-costeros-de-veracruz/full-view.html).


## Ciudad esponja: Veracruz

Veracruz es una ciudad asentada sobre dunas y sobre humedales. [Esos ecosistemas originarios funcionaban
como una esponja](https://libros.inecol.mx/index.php/libros/catalog/book/634), regulando la dinámica hidráulica del ciclo natural del agua en la región. Con el desarrollo urbano, o bien ni se comprendió esto o se ha olvidado que este comportamiento natural ocurría. Por las razones que sean, la urbanización ha rellenado los humedales, aplanado las
dunas y construido sobre estos ecosistemas. El resultado ha sido un amenaza a la calidad de vida de los habitantes asentados en estos lugares. Sin embargo, es enteramente posible "incorporar a la naturaleza" para tener un buen manejo del agua y por tanto una mejor calidad de vida.



## Restauración de Manglares

En este [artículo de Crónica](https://www.cronica.com.mx/academia/hay-quince-centimetros-vida-muerte-manglar.html
), Jorge López-Portillo *et al.* explican la gran diferencia que hace una pequeña altura en el establecimiento y supervivencia de los manglares. Esta es parte de una experiencia que se ha venido desarrollando en torno a la restauración de manglares degradados de largo aliento. La [CONABIO ha sido parte de esto en algúnos momentos](https://www.inecol.mx/index.php/divulgacion/ciencia-hoy/navegando-por-los-manglares-desde-el-sillon). Lo interesante es que el proceso lo van acogiendo las comunidades locales en formas muy interesantes. [Ese artículo todavía está por escribirse](https://www.facebook.com/watch/?v=1256192325897357).

## Proyecto Integridad Costera

SECIHTI: *Estimación de la integridad ecosistémica de las costas arenosas mexicanas a través de técnicas de aprendizaje de máquina*

El proyecto se propone generar estimaciones de la condición en la que se encuentran los ecosistemas costeros, especialmente los arenosos, del país con una representación ráster y resolución geográfica de pixeles de 3 segundos (aproximadamente 100 m). También se está considerando tener una valoración de mayor resolución para la zona costera de Quintana Roo, en donde se propone lograr una resolución geográfica con pixeles 0.3 segundos. Se ha desarrollado un enfoque de automatización mediante flujos de trabajo basados en [Snakemake](https://snakemake.readthedocs.io/), que opera con base en una colección de scripts desarrollados en Python y datos de tipo tabular, vectorial y ráster (según el tipo de fuente y tema). El proceso limpia y armoniza los datos hasta geenrar una colección de variables congruentes. A partir de esos datos se aplica modelación con redes Bayesianas que producen la *máquina inferencial* con la que se estima la "condición de los ecosistemas" a través de un índice de integridad ecosistémica. Esta aproximación ya la hemos ensayado como viable para ser incorporada en el marco de contabilidad ecosistémica que ha venido desarrollando la ONU en el marco del SEEA. México fue parte del [piloto mundial NCAVES](https://www.inegi.org.mx/contenidos/investigacion/cem/doc/docNCAVES.pdf) en donde se puso a prueba este concepto.

El enfoque se enmarca en lo que está emergiendo como “Inteligencia artificial interpretable” (IAI por las siglas en inglés, Mihaljevic et al 2021). Una aproximación del tipo IAI para el caso de las cuentas de ecosistemas permitiría, con base en datos y un enfoque sistémico, un mayor entendimiento de las relaciones entre las acciones de origen antrópico y su repercusión en los ecosistemas del país y en los servicios que estos proporcionan. En este proyecto, se propone evaluar como prueba de concepto, la condición de las costas arenosas (playas y dunas) de todo el país, por medio del cálculo de un índice de integridad de los ecosistemas de costas arenosas (IIECA) bajo el paradigma de IAI.