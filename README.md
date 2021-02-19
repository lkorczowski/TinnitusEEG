# TinnitusEEG

Toolbox for EEG analysis for finding markers of Tinnitus

Author: Louis Korczowski

Ownership: Robin Guillard EIRL (all rights reserved), previously owned by Zeta Technology 

- Code is shared as is for public research purpose and shouldn't be reproduced without the direct consent of the authors.
- Use/reproduction by commercial companies is strictly prohibited (contact us if you want to use/understand that code).
- Some of the EEG Data were shared to us for research purpose, we don't own them (Thanks to S. Vanneste, D. De Ridder and M. Congedo)

# Goals

- Be able to differentiate EEG pattern of Tinnitus subjects from non-Tinnitus subjects (spatial, temporal and frequency patterns)
- Maximize accuracy, robustness and "efficiency" of Tinnitus versus non-Tinnitus EEG classification using Riemannian Geometry
- Find EEG markers of residual inhibition

# Results
- ROC AUC about 0.926 using only 2 seconds of EEG epochs (Tinnitus versus non-Tinnitus) using simple RG classif
- increasing the number of epochs/length of epochs didn't increase significantly the AUC

# Bibliography 
- Vanneste et al. 2010 The neural correlates of tinnitus-related distress https://pubmed.ncbi.nlm.nih.gov/20417285/
- De Ridder et al. 2011 The Distressed Brain: A Group Blind Source Separation Analysis on Tinnitus https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0024273
