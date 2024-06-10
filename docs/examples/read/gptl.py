#!/usr/bin/env python

import hatchet as ht


if __name__ == "__main__":
    # Path to GPTL database directory.
    dir_location = "../../../hatchet/tests/data/gptl/multi-node/"
    # Use hatchet's ``from_gptl`` API to read in the GPTL database.
    # The result is stored into Hatchet's GraphFrame.
    gf = ht.GraphFrame.from_gptl(dir_location)

    # Printout the DataFrame component of the GraphFrame.
    print(gf.dataframe)

    # Printout the graph component of the GraphFrame.
    # Use "time (inc)" as the metric column to be displayed
    print(gf.tree(metric_column="time (inc)"))


    # Path to GPTL log file.
    file_location = "../../../hatchet/tests/data/gptl/log_cpu"
    # Use hatchet's ``from_gptl`` API to read in the GPTL log file.
    # The result is stored into Hatchet's GraphFrame.
    gf_from_file = ht.GraphFrame.from_gptl(file_location)

    # Printout the DataFrame component of the GraphFrame.
    print(gf_from_file.dataframe)

    # Printout the graph component of the GraphFrame.
    # Use "time (inc)" as the metric column to be displayed
    print(gf_from_file.tree(metric_column="time (inc)"))
