import torch # PyTorch core library for tensors and automatic differentiation
import numpy as np # Numerical computing library
import networkx as nx # For graph creation and visualization
import matplotlib # Core matplotlib functionality
matplotlib.use('TkAgg') # Use the TkAgg backend for interactive plots
import matplotlib.pyplot as plt # Matplotlib plotting interface
from matplotlib import animation # Matplotlib animation module
from matplotlib.animation import FFMpegWriter # Writer to save animations as video files

from torch.nn import Linear # Simple linear layer from PyTorch
from torch_geometric.datasets import KarateClub # Example graph dataset
from torch_geometric.nn import GCNConv # Graph convolutional layer
from torch_geometric.utils import to_dense_adj, to_networkx # Utils to convert between PyG and NetworkX

import random
random.seed(0)

# Fijar semilla para PyTorch y NumPy para resultados consistentes
torch.manual_seed(0)
np.random.seed(0)

# ********** Cargar dataset: **********

# Opcion 1: Karate Club (grafo de ejemplo con 34 nodos y dos comunidades conocidas)
# Cargar dataset
dataset = KarateClub()

"""
# Opcion 2: +2700 Nodos
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='data/Cora', name='Cora')
"""

# *************************************

data = dataset[0] # Obtenemos la primera (y única) muestra del dataset

# Convertir el objeto PyG a un grafo de NetworkX para visualización
graph_full = to_networkx(data, to_undirected=True)
# Generar layout inicial para dibujar el grafo completo
pos_full   = nx.spring_layout(graph_full, seed=0)

# Parámetros de submuestreo para grafos muy grandes (no aplicable a KarateClub)
# Es para que el codigo funcione para cualquier grafo
max_nodes = 50 #Variable que define cuanto es el maximo de nodos que se va a mostrar
if graph_full.number_of_nodes() > max_nodes:
    sampled_nodes = random.sample(list(graph_full.nodes()), max_nodes)
    graph_sub = graph_full.subgraph(sampled_nodes).copy()
    pos_sub = {n: pos_full[n] for n in sampled_nodes}
else:
    sampled_nodes = None

# Función para visualizar información básica del dataset y grafo generado
def print_information():
    print('---------------------------------------------------------------------')
    print(f'dataset: {dataset}')
    print('---------------------------------------------------------------------')
          
    print(f'number of graphs: {len(dataset)}')
    print(f'number of features: {dataset.num_features}')
    print(f'number of classes: {dataset.num_classes}\n\n')
    
    # * Print first element (unico grafo) *
    print("*** Informacion del grafo ***\n")
    print(f'Graph: {data}\n\n')
    
    # * Print node feature matrix *
    print("*** Node feature matrix: ***\n")
    print(f'x = {data.x.shape}')
    print(data.x, '\n\n')
    
    # * Print edge index *
    print("*** Edge Index - Lista de coordenadas(COO): ***\n")
    print(f'edge_index = {data.edge_index.shape}')
    print(data.edge_index, '\n\n')
    
    # * Print adjacency matrix *
    print("*** Adjacency matrix: ***\n")
    A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
    print(f'A = {A.shape}')
    print(A, '\n\n')
    
    # * Print ground-truth labels (comunidad a la que pertenece cada nodo) *
    print("*** Comunidad/Categoria a la que pertenece cada nodo: ***\n")
    print(f'y = {data.y.shape}')
    print(data.y, '\n\n')
    
    # * Print train mask *
    print("*** Node Train Mask: ***\n")
    print(f'train_mask = {data.train_mask.shape}')
    print(data.train_mask, '\n\n')
    
    # * Print utility functions *
    print("*** Itility Functions: ***\n")
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}\n\n')

# Funcion para mostrar el plot del grafo original
def plot_original_graph():
    G, pos = get_graph_and_pos()
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_size=800,
        node_color=[data.y[n] for n in G.nodes()],
        cmap='hsv',
        vmin=-2,
        vmax=3,
        width=0.8,
        edge_color='grey',
        font_size=14
    )
    plt.show()

# Función para obtener grafo y posición según tamaño
def get_graph_and_pos():
    if sampled_nodes is None:
        return graph_full, pos_full
    return graph_sub, pos_sub

# Función para construir el modelo GCN
def build_model():
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn = GCNConv(dataset.num_features, 3)
            self.out = Linear(3, dataset.num_classes)

        def forward(self, x, edge_index):
            h = self.gcn(x, edge_index).relu()
            z = self.out(h)
            return h, z

    return GCN()

# Función de precisión
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

# Función de entrenamiento y evaluación del modelo
def train_model(model, epochs=201, lr=0.02):
    """
    Entrena el modelo GCN sobre el dataset y recolecta métricas.
    Devuelve embeddings, lista de pérdidas, lista de accuracies y salidas por epoch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    embeddings, losses, accuracies, outputs = [], [], [], []

    for epoch in range(epochs):
        optimizer.zero_grad()
        h, z = model(data.x, data.edge_index)
        loss = criterion(z, data.y)
        acc = accuracy(z.argmax(dim=1), data.y)
        loss.backward()
        optimizer.step()

        embeddings.append(h.detach())
        losses.append(loss.item())
        accuracies.append(acc.item())
        outputs.append(z.argmax(dim=1).detach().cpu().numpy())

        if epoch % 5 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

    return embeddings, losses, accuracies, outputs

# Función para animar la evolución de los embeddings del modelo
def animate_and_save(outputs, losses, accuracies):
    plt.rcParams['animation.bitrate'] = 3000
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')

    def animate(i):
        plt.clf()
        G, pos = get_graph_and_pos()
        node_colors = [outputs[i][n] if n < len(outputs[i]) else 0 for n in G.nodes()]
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_size=800,
            node_color=node_colors,
            cmap='hsv',
            vmin=-2,
            vmax=3,
            width=0.8,
            edge_color='grey',
            font_size=14
        )
        plt.title(
            f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
            fontsize=18, pad=20
        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        #frames=len(outputs)*5,
        np.arange(0, 200, 5),
        interval=1000,
        repeat=False
    )
    guardar = input('Desea guardar la animacion en un video? (s/n): ')
    if guardar.lower() == 's':
        anim.save('gcn_evolution.mp4', writer='ffmpeg', fps=1)
        print('Vídeo guardado en gcn_evolution.mp4')
        
    ver = input('Desea ver la animacion? (s/n): ')
    if ver.lower() == 's':
        plt.show()

# Función principal del script
def main():
    verPlot = input('Desea ver el plot del dataset? (s/n): ')
    if verPlot.lower() == 's':
        plot_original_graph()
        
    verInformacion = input('Desea ver toda la informacion del dataset? (s/n): ')
    if verInformacion.lower() == 's':
        print_information()
        
    input("Pulsa Enter para iniciar el entrenamiento")
    print()

    model = build_model()

    embeddings, losses, accuracies, outputs = train_model(model)
    animate_and_save(outputs, losses, accuracies)

if __name__ == '__main__':
    main()
