"""
The classes here define the most high level abstractions of the fundamental classes used for control and environment interaction.
1. `Controller`: The class that maps high level actions to low level actions on the emulator.
2. `Environment`: The class that implements the Gym API, combining an `Emulator` and a `Controller` to provide observations, rewards, and episode termination logic.


"""