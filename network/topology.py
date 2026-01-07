import networkx as nx
import matplotlib.pyplot as plt

def draw_topology(drone_names):
    G = nx.Graph()
    for d in drone_names:
        G.add_node(d)

    for i in range(len(drone_names)):
        for j in range(i + 1, len(drone_names)):
            G.add_edge(drone_names[i], drone_names[j])

    nx.draw(G, with_labels=True)
    plt.title("UAV Swarm Network Topology")
    plt.show()
