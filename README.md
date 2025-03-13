# MDM2-Group-09: Synchronisation of metronomes

Details of the project can be found in [Project.pdf](Project.pdf).

## Kuramoto Model

Code regarding the Kuramoto model can be found in the `Kuramoto Model` folder.

- `Animation of Kuramoto model.py` produces an animation comparing the effects of the coupling parameter $K$
- `Kuramoto_Antiphase.py` contains all the code used to investigate & visualise antiphase in the model
- `Antiphase_Kuramoto_Demo.py` produces an animation that shows nodes cannot all travel in the same clockwise direction

## ODE Model

Code regarding the ODE model can be found in the corresponding subfolders in the `ODE Model` folder.

### Presentation

All files referenced are found in `/ODE Model/Presentation/`

* `ODE_Model.ipynb` contains all the code used to investigate & visualise the ODE model, as well as the predictor training, with the supporting data in `energy_to_sync3.csv` & `predictor_results2.csv`
    - The base model code is modified from `Deprecated/Lugo-Cardenas model 2.py`

- The `Deprecated` subfolder contains previous simulation results, as well as a some work done on the PD control (not in presentation)
- The `Figures` subfolder contains the visual results of the ODE model

### Technical Note

All files referenced are found in `/ODE Model/Technical Note/`

* `4 metro.ipynb` contains code that can simulate the system with 4 metronomes, alongside a corresponding derivation in `4 metronomes mathematical model.pdf`
* `masses.ipynb` contains code to explore the effects of changing pendulum & cart masses
* `NC_PD_FLC.ipynb` contain the code for comparing the control methods

## Previous Study

External resources studied can be found in the `Previous Study` folder.

* `Metronome-Sync` contains files to imitate (two) metronomes syncing by discrete time calculations. [By Dave McCulloch (2020)](https://github.com/dfivesystems/Metronome-Sync.git)
* `metronomes` contains files to model (two) metronomes on a moving cart by solving two coupled SODEs. [By Paul Gribble (2012)](https://github.com/paulgribble/metronomes.git)
* `Previous Study.pdf` details the two models methods and equations

- `References` contains the papers referenced in `Project.pdf`
