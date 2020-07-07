# SIRD-vaccination
Individual-level SIRD contact model for studying vaccination prioritization (implemented in Python). Notably:
* Individuals are modeled explicitly, including family units
* Contacts between individuals are modeled explicitly (using sparse computations)
* Vaccination of individuals is modeled explicitly
* Symptom level modeled (including asymptomatic carriers)
* Symptom level is a factor of an individual's inherent vulnerability (mortality)
* Probability of death related to symptom level

Dependencies:
* Numpy
* Scipy
* PyTorch
* Matplotlib (for plotting)

## Methods
Using Monte Carlo sampling of an individual-level SIRD model, we find a vaccination order based on contact rate, susceptibility (likelihood of infection given exposure), mortality (parameter modulating probability of symptom severity), infectivity (likelihod of transmission), symptom level, and home size, as well as the maximum value of each variable within an individual's household.

## Results

<img src="/images/bars.png" width="400">

<img src="/images/weights.png" width="400">

## Conclusions
Overall, findings are similar to, but not equivalent to, the [CDC vaccination prioritization guidelines for influenza](https://www.cdc.gov/flu/pandemic-resources/national-strategy/planning-guidance/index.html).

<img src="/images/prioritytree.png" width="400">

For more details, see [SIRD-vaccination-paper.pdf](https://github.com/johannes-lee/SIRD-vaccination/blob/master/SIRD-vaccination-paper.pdf)
