# Demo GCN Grupo 6

![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.7-orange.svg) ![PyG](https://img.shields.io/badge/torch--geometric-%3E%3D1.6-brightgreen.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Descripción

Este repositorio contiene una demo de un **Graph Convolutional Network (GCN)** implementado con PyTorch Geometric.  
Utiliza el dataset de ejemplo _Karate Club_ (34 nodos, 4 comunidades) y ofrece una versión genérica que puede adaptarse a grafos más grandes (por ejemplo Cora).  
Incluye:

- Visualización estática del grafo
- Información detallada de nodos, aristas y máscaras de entrenamiento
- Entrenamiento de un modelo GCN con métricas de pérdida y exactitud
- Animación de la evolución de las predicciones a lo largo de los epochs

## Requisitos

- Python 3.7 o superior
- [PyTorch](https://pytorch.org/) ≥ 1.7
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) ≥ 1.6
- NetworkX
- Matplotlib (con backend `TkAgg`)
- FFmpeg (para guardar animaciones como video)

## Instalación

1. **Clonar el repositorio**:

   ```bash
   git clone https://github.com/ELAlejandroo/demo-GCN-06-10am.git
   cd demo-GCN-06-10am
   ```

2. **Crear y activar el entorno con Conda**

   ```bash
   conda create -n demo-gcn-06-10am python=3.10 -y
   conda activate demo-gcn-06-10am
   ```

3. **Instalar dependencias**

   ```bash
   conda install -c pytorch pytorch -y
   conda install -c conda-forge torch-geometric networkx matplotlib ffmpeg
   ```

## Uso

```bash
python demo_GCN_grupo06_10am.py
```

**Durante la ejecución se te preguntará:**

- ¿Ver gráfico del dataset? (s/n)

- ¿Mostrar info completa del dataset? (s/n)

- Pulsa Enter para iniciar el entrenamiento.

- Tras el entrenamiento, se generará una animación:

- ¿Guardar video? (s/n) → gcn_evolution.mp4

- ¿Visualizar animación? (s/n)

## Otra opción de dataset: usar Cora

Para cambiar al dataset Cora, en el script comenta la línea de KarateClub() y descomenta:

```bash
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='data/Cora', name='Cora')
```
