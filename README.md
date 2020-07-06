# SIRD-vaccination
Individual-level SIRD contact model for studying vaccination prioritization (implemented in Python). Notably:
* Individuals are modeled explicitly, including family units
* Contacts between individuals are modeled explicitly (using sparse computations)
* Vaccination of individuals is modeled explicitly
* Symptom level is a factor of an individual's inherent vulnerability (mortality)
* Probability of death related to symptom level

Dependencies:
* Numpy
* Scipy
* PyTorch
* Matplotlib (for plotting)

## Conclusions
Findings are similar to, but not equivalent to, the CDC vaccination prioritization guidelines for influenza: https://www.cdc.gov/flu/pandemic-resources/national-strategy/planning-guidance/index.html.
