"""
Script to compile and aggregate results from the EA, TPEBO, and TPEC RF experiments.
Produces "agg_other_res.csv"
"""

import pandas as pd
import os
import numpy as np
import json


# Path structure: './rfresults/{TASK ID}/result_{METHOD}_{RUN}.jsonl
