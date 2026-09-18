# Bone Fracture Project

Proyecto práctico desarrollado para la asignatura de Fundamentos de Análisis y Cálculo Numérico de la Universidad del Valle.

El objetivo del proyecto es aplicar conceptos de redes neuronales utilizando
PyTorch para realizar clasificación de imágenes.

## Dataset seleccionado

**Bone Fracture Multi-Region X-ray Data**

El conjunto de datos contiene imágenes de rayos X de diferentes regiones del
cuerpo clasificadas en dos categorías:

- Fracture
- No Fracture

El dataset contiene aproximadamente 9.246 imágenes.

Fuente:

https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

## Objetivo

Construir y entrenar una red neuronal capaz de clasificar imágenes de rayos X según la presencia o ausencia de una fractura.

Udacity:

https://github.com/udacity/DL_PyTorch

Las primeras etapas del proyecto consisten en comprender el funcionamiento de PyTorch utilizando MNIST y posteriormente adaptar los conceptos al dataset de
fracturas.

## Estructura del proyecto

```text
proyecto_fracturas/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── mnist/
│   └── fractures/
│
├── src/
│
├── reports/
│
├── presentation/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Instalación y entorno local

Consulta [la guía de instalación](README_SETUP.md) para crear el entorno Python 3.12, instalar las versiones del grupo y ejecutar Jupyter.
