"""
Script to compile and aggregate results from the Random, EA, TPEBO, TPEC RF experiments.
Produces "all_scores.csv".
Requires "filtered_random_scores.csv". 
"""

import pandas as pd
import os
import numpy as np
import json


# Path structure: './rfresults/{TASK ID}/result_{METHOD}_{RUN}.jsonl
