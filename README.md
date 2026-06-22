# Physics-Informed Graph Neural Network for Inverse Manning's `n` Estimation

> **Novelty**: First PIGNN framework that recovers a real-world, spatially-distributed Manning's roughness field `n(x,y)` directly from raw HEC-RAS unstructured mesh geometry and sparse USGS stream gauge data. 

Unlike traditional studies that rely on dense, synthetic HEC-RAS outputs as a "proof of concept", this project applies the PIGNN directly to the **Sacramento River (Colusa to Ord Ferry Reach)** to act as an automated, physics-driven calibration engine.

## Method Overview

Instead of manually calibrating a HEC-RAS model via trial-and-error, this workflow mathematically discovers the true river roughness:

1. **The Graph:** Raw HEC-RAS 2D geometry (unstructured mesh) is extracted. Every cell center becomes a node, and every shared face becomes an edge. High-resolution terrain bathymetry (`Terrain.sacramento_bathy.tif`) is directly sampled into the graph nodes (yielding correct bed elevations ranging from 2.28 m to 37.47 m).
2. **The Physics:** The neural network connects the nodes using the exact 2D Shallow Water Equations (SWE). It calculates the Finite Volume Method (FVM) flux of water moving across the faces exactly as the HEC-RAS 2D solver does.
3. **The Inversion (Piecewise Data Assimilation):** The network is fed the Upstream (Ord Ferry discharge) and Downstream (Colusa stage) boundary conditions. The flow fields are interpolated piecewise using the interior **Butte City (BTC)** stage gauge as an anchor. This enforces a continuous water surface elevation profile matching observations with sub-millimeter precision, enabling the GNN to learn the true physical Manning's `n` distribution.
4. **The Output:** The neural network exports the learned Manning's `n` field as an unstructured CSV and automatically interpolates it into a spatial Land Cover Raster (`.tif`) that can be dragged straight into HEC-RAS.

## Model Domain (Sacramento River)

This project specifically targets the heavily-leveed Middle Sacramento River, bracketed perfectly by active physical stream gauges:

*   **Bathymetry Extent:** ~5x5 km domain (`Terrain.sacramento_bathy.tif`, 2023 Survey, UTM 10N / NAD83)
*   **Upstream Boundary:** Sacramento River at Ord Ferry, CA (USGS `11388800`)
*   **Downstream Boundary:** Sacramento River at Colusa, CA (USGS `11389500`)
*   **Interior Gauge:** Sacramento River at Butte City, CA (USGS `11389000`), snapped to cell #17661 (z_bed = 19.01 m)

## Project Structure

```
PIGNN/
├── main.py                  # CLI entry point
├── config.yaml              # All configuration
├── requirements.txt
├── graph/
│   ├── build_graph.py       # HEC-RAS HDF5 → PyG Data (Reads TIF into unstructured mesh)
│   └── mesh_utils.py        # HDF5 inspection, graph stats, validation
├── model/
│   └── pignn.py             # PIGNN architecture (Encoder-Processor-Decoder)
├── physics/
│   └── fvm_residual.py      # FVM SWE residuals (Lax-Friedrichs & Roe flux)
├── data/
│   ├── hecras_loader.py     # HEC-RAS geometry and boundary flow loaders
│   └── obs_loader.py        # Stream gauge stage observations loaders
├── train/
│   ├── losses.py            # Loss functions (FVM residual, stage, smoothness, limits)
│   └── trainer.py           # Training loop
└── viz/
    └── visualize.py         # Interpolates n(x,y) CSV output into a spatial TIF and generates plots
```

## Loss Function

The network is trained not by matching pixels, but by enforcing the laws of physics over the mesh:

```
L_total = w_fvm    * L_fvm      (FVM SWE residuals → 0; Mass & Momentum must be conserved)
        + w_smooth * L_smooth   (Graph Laplacian ensuring smooth n-transitions across the floodplain)
        + w_bound  * L_bound    (Soft constraint keeping n within physical limits [0.020, 0.080])
        + w_btc    * L_btc      (Stage elevation error at Butte City gauge → 0)
```

## Workflow & Outputs

Once the PIGNN finishes its calibration, it outputs everything needed to run your verified forward model in HEC-RAS:

| File | Content |
|------|---------|
| `pignn_checkpoint.pt` | Best model weights |
| `n_field_final.csv` | Raw unstructured Manning's n per mesh cell (x, y, n) |
| `n_field_final.tif` | **[NEW]** Interpolated spatial n-raster ready for HEC-RAS Land Cover import |
| `loss_history.png` | Training loss curves showing physical convergence |
| `history.json` | Full loss history logs |

## Novelty Statement

This project uniquely bridges the gap between deep learning and civil engineering by:
1. Utilizing **raw unstructured meshes** directly from HEC-RAS (via PyTorch Geometric).
2. Implementing the exact **Finite Volume Method (FVM) Shallow Water Equations** as the physics loss layer.
3. Solving the **Inverse Calibration Problem** on a real river using sparse physical gauges, completely bypassing the need for a pre-calibrated forward model.
