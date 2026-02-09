# DualOptim: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers

## Dataset Overview

The full dataset consists of **4,000 QA pairs**.  
We define three forget sets corresponding to different forgetting ratios:

- **1% forget set (`forget01`)**: 40 samples  
- **5% forget set (`forget05`)**: 200 samples  
- **10% forget set (`forget10`)**: 400 samples  

For each forget set, the remaining samples form the **retain set**.


### Example: Constructing `forget01`

- The file `forget01.json` contains candidate samples for the 1% forget set.
- To construct the complete `forget01` forget set (40 samples), select entries with `task_id = 1, 2, ..., 10` from `forget01.json` and get 40 samples as the forget set.
- The remaining samples of the full dataset constitute the retain set.


## Data Location

Dataset files are located under:  
