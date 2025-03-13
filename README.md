# MDM2-Group-09: Synchronisation of metronomes

Details of the project can be found in [Project.pdf](Project.pdf).

## Kuramoto Model

Code regarding the Kuramoto model can be found in the `Kuramoto Model` folder.

- `Animation of Kuramoto model.py` produces an animation comparing the effects of the coupling parameter $K$
- `Kuramoto_Antiphase.py` contains all the code used to investigate & visualise antiphase in the model

## ODE Model

Code regarding the ODE model can be found in the `ODE Model` folder.

* `ODE_Model.ipynb` contains all the code used to investigate & visualise the ODE model, as well as the predictor training.
    - The base model code is modified from `Deprecated/Lugo-Cardenas model 2.py`
* `4 metro.ipynb` contains code that can simulate the system with 4 metronomes (not in presentation)

- The `Deprecated` subfolder contains previous simulation results, as well as a some work done on the PD control (not in presentation).
- The `Figures` subfolder contains the visual results of the ODE model.
- The `masses.ipynb`  for mass variation analysis and heatmap generation to minimise sync time.

## Previous Study

External resources studied can be found in the `Previous Study` folder.

* `Metronome-Sync` contains files to imitate (two) metronomes syncing by discrete time calculations. [By Dave McCulloch (2020)](https://github.com/dfivesystems/Metronome-Sync.git).
* `metronomes` contains files to model (two) metronomes on a moving cart by solving two coupled SODEs. [By Paul Gribble (2012)](https://github.com/paulgribble/metronomes.git).
* `Previous Study.pdf` details the two models methods and equations.

- `References` contains the papers referenced in `Project.pdf`.
