#!/usr/bin/env python3

import pandas as pd
import numpy as np


if __name__ == "__main__":
    # Read the file containing a list of json files
    logs_file = "logs.json"
    df = pd.read_json(logs_file, lines=True)

    # Now iterate over all possible jobs
    for job in np.unique(df["job"]):
        job_msg_total = "".join(df[df["job"]==job]["msg"])

        with open("docker_" + job.lower().strip().replace(r' ', '-').replace('/', '-'), "w") as out_file:
            out_file.write(job_msg_total)
