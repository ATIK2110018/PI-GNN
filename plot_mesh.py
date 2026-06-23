import torch
from graph.build_graph import load_hecras_geometry, build_pyg_graph
from viz.visualize import plot_mesh_bc_obs

print("Loading Graph...")
geo = load_hecras_geometry("data/hecras/sacramento_2/Geometries/Geometry.h5")
graph = build_pyg_graph(geo, "data/hecras/sacramento_2/Terrains/Terrain.sacramento_bathy.tif")

print("Plotting Mesh...")
plot_mesh_bc_obs(graph, obs_points={"BTC": (586420, 4368120)}, btc_cell_idx=17661, save_path="runs/pignn/mesh_bc_obs.png")
print("Done!")
