# TinnitusEEG

Toolbox for EEG analysis for finding markers of Tinnitus

Authors: Louis Korczowski & Robin Guillard
Owernship: Robin Guillard EIRL (all rights reserved), previously owned by Zeta Technology 

- Code is shared as is for public research purpose and shouldn't be reproduced without the direct consent of the authors.
- Use/reproduction by commercial companies is striclty prohibited (contact us if you want to use/understand that code).
- EEG Data were shared to us for research purpose, we don't own the data

# Goals

- Be able to diffentiate EEG pattern of Tinnitus subjects from non-Tinnitus subjects (spatial, temporal and frequency patterns)
- Maximize accuracy, robustness and "efficiency" of Tinnitus versus non-Tinnitus EEG classification using Riemmanian Geometry

# Results
- ROC AUC about 0.8 using only 2 seconds of EEG epochs (Tinnitus versus non-Tinnitus) using simple RG classif
- increasing the number of epochs/length of epochs didn't increase significatively the AUC


# Setting things up

You can read the [Tutorial](TUTO_SET_GITHUB_SSH.md) if you want to clone this repo and start working. In this tutorial, you'll see how to:
- set github and local git
- set your ssh to push and pull from github
- make a local branch and track it on github
- get a branch from github
- merge branches
