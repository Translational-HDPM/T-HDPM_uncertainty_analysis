This folder contains a script version of the functions in the Jupyter Notebooks in the `team-3` branch, along with some additional functions to make V-plots and misclassification scatter plots from Monte Carlo simulations.

- `classifier.py` contains functions that calculate classifier score, run Monte Carlo simulations for a given noise percentage and determine a threshold given a specified value of sensitivity or specificity.
-  `threshold.py` carries functions to conduct simulations and make misclassification plots and v-plots.
-  `main.py` defines a pipeline from ingestion of data to Monte Carlo simulations to generate plots based on two main scenarios:
    - changing threshold independently and experimenting at different levels of noise.
    - specifying a metric (either sensitivity or specificity), fixing the threshold based on this metric and conducting experiments at different noise (relative standard deviation) levels.   

Please refer to comments within the code for more detailed documentation.

### Dependencies

`joblib`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`, `statsmodels`
