# MDM2-Group-09: Synchronisation of metronomes

MDM2 Project 2 - Group 09

Synchronisation of metronomes

## Files

* `Metronome-Sync` contains files to imitate (two) metronomes syncing by discrete time calculations. [By Dave McCulloch (2020)](https://github.com/dfivesystems/Metronome-Sync.git).
* `metronomes` contains files to model (two) metronomes on a moving cart by solving two coupled SODEs. [By Paul Gribble (2012)](https://github.com/paulgribble/metronomes.git).
* `Previous Study.pdf` details the two models methods and equations.

- `References` contains the papers referenced in `Project.pdf`.
- `old` contains the first versions of the metronome simulation code, as well as the first datasets originally generated before the energy formula was reworked.

* `LC Model FLC Tests.ipynb` contains all the work done for the ODE model and the feedback linearisation, including the synchronicity finding, energy calculations and the ML predictor model.

## Notes on energy

* This system has the oscillators reach max $PE$ and $KE$ at the same time (at the equilibrium point)
    * This results in oscillating total energy of the system
    * The missing energy in the system is *probably* the energy that is in the metronomes' actuators that keep the metronomes oscillating - since they are inverted pendulums, some energy is needed to turn them the other way and keep them inverted

* Energy has been calculated now accounting for some sideways translational energy of the metronomes as well, but this results in some negative values for input energies
    * This is because the `input_energy` is not really the input energy, rather it is the difference in the system energy of the system compared to one with no control.
    * Some systems then sync and reach oscillations of lower amplitudes, resulting in lower total system energies and thus a lower "input_energy" metric
        * This should really be changed for another metric of "input energy" to be more accurate

* **Despite whatever the numerical values are, the expected trends are still shown, with systems with more "input energy" (more control) having better sync times than systems with lower control, as well as a diminishing return of input energy to sync time, where no matter how much energy is input, the system cannot be synced any faster**
    * These are the expected results and these results are reached, despite the *likely incorrect* numerical values found by the system.