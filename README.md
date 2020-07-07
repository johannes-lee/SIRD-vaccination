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
Using Monte Carlo sampling of an individual-level SIRD model, we find a vaccination order based on contact rate, susceptibility (likelihood of infection given exposure), mortality (parameter modulating probability of symptom severity), infectivity (likelihod of transmission), symptom level, and home size, as well as the maximum value of each variable within an individual's household. The vaccination ordering is represented by a function from 11 dimensions to 1 dimensions, and is optimized for a linear function and a two-layer neural network. Optimization is performed using a least squares approach, or with the REINFORCE algorithm (Williams, 1992). 

At each day, the λ individuals with highest priority indicated by the vaccination ordering function are vaccinated, with the objective being to find a vaccination ordering function which minimizes the total number of deaths.

## Results
The least squares (linear) solution yields the lowest average deaths over 1000 trials, with a 0.388% of total population reduction of deaths compared to random vaccination (0.388% of the US's 327 million people is 1.27 million deaths). 

<img src="/images/bars.png" width="400">

Looking at the weights of each variable of the optimal least squares vaccination ordering function, we see that the most vulnerable, i.e. those with the highest mortality variable, are prioritized, while those with symptoms are deprioritized. Outside contact rate also plays a strong role in vaccination priority.

<img src="/images/weights.png" width="400">

## Conclusions
Overall, findings are similar to, but not equivalent to, the [CDC vaccination prioritization guidelines for influenza](https://www.cdc.gov/flu/pandemic-resources/national-strategy/planning-guidance/index.html). This allows us to formulate a (simple) COVID-19-specific flowchart which separates individuals into several vaccination priority groups:

<img src="/images/prioritytree.png" width="400">

For more details, see [SIRD-vaccination-paper.pdf](https://github.com/johannes-lee/SIRD-vaccination/blob/master/SIRD-vaccination-paper.pdf)
