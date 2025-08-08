This repository contains run scripts for the paper "Augmented Lagrangian Solvers for Poroelasticity with Fracture Contact Mechanics".
The code is structured as follows:

- Runscipts for reproducing figures 3-6 and 8 in the article are located in appropriately named files.
- The IRM solver is defined in `implicit_return_map.py`, while the GNM-RM solver is defined in `newton_return_map.py`. The GNM solver is included in PorePy by default.
- Material and solution parameters are defined in `parameters.py`
- The model_setup files contain all the pieces necessary to set up the physical models. The parts common to both the two- and three-dimensional case are collected in `model_setup_common.py`, while the other two files contain the parts unique to each individual case.
- The remaining files contain various utility methods for running simulations, visualization etc. 

This code has been run using PorePy commit `993208680a4382b0a96aecba1fe93bee1348eb02`.
