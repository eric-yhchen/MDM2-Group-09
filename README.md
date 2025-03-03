# MDM2-Group-09

MDM2 Project 2 - Group 09
Synchronisation of metronomes with feedback

## Files

* `Metronome-Sync` contains files to imitate (two )metronomes syncing by discrete time calculations. [By Dave McCulloch (2020)](https://github.com/dfivesystems/Metronome-Sync.git).
* `metronomes` contains files to model (two) metronomes on a moving cart by solving two coupled SODEs. [By Paul Gribble (2012)](https://github.com/paulgribble/metronomes.git).
* `Previous Study.pdf` details the two models methods and equations.
* `References` contains the papers referenced in `Project.pdf`.

## More

* ~~Come up with some measure of synchronicity -~~ consider velocities as well?
* ~~Parameter sweep the simulation for time to sync~~
* ~~Calculate energy in the system for NC and FLC - should be oscillating~~
* ~~Calculate the energy put in as the integral of this?~~
* See how to vary the input energy and the results of this -> needs checking

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