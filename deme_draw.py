"""
Template code to draw demographic history plots
Author: Sara Mathieson, Jacky Siu Pui Chung
Date: 09/08/23
"""

import demes
import demesdraw
import simulation
import param_set
import math
import msprime
import matplotlib.pyplot as plt
import argparse
import os

POP1 = "GN"
POP2 = "BF"

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Draw demographic history plots")
    parser.add_argument("--model", choices=["mig", "no_mig"], help="Demographic model to use: 'mig' or 'no_mig'")
    
    parser.add_argument("--NI", type=float, required=True, help="Initial ancestral population size")
    parser.add_argument("--TG", type=float, required=True, help="Time when the ancestral population begins to change in size (generations)")
    parser.add_argument("--NF", type=float, required=True, help="Final ancestral population size, immediately prior to the split")
    parser.add_argument("--TS", type=float, required=True, help="Time of the split (generations)")
    parser.add_argument("--NI1", type=float, required=True, help="Initial size of population 1")
    parser.add_argument("--NI2", type=float, required=True, help="Initial size of population 2")
    parser.add_argument("--NF1", type=float, required=True, help="Final size of population 1")
    parser.add_argument("--NF2", type=float, required=True, help="Final size of population 2")
    parser.add_argument("--MG", type=float, default=0, help="Migration rate (only used if model is 'mig')")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to store figure and YAML file")

    args = parser.parse_args()

    try:
        gen_per_year = 11

        # Adjust time parameters based on generations per year
        TG = args.TG * gen_per_year
        TS = args.TS * gen_per_year

        # Compute growth rates from the start/end sizes and times
        g1 = -(1/TS) * math.log(args.NI1/args.NF1)
        g2 = -(1/TS) * math.log(args.NI2/args.NF2)
        g = -(1/(TG-TS)) * math.log(args.NI/args.NF)

        demography = msprime.Demography()
        demography.add_population(name=POP1, initial_size=args.NF1)
        demography.add_population(name=POP2, initial_size=args.NF2)
        demography.add_population(name="ANC", initial_size=args.NF, initially_active=False)

        demography.add_population_parameters_change(time=0, growth_rate=g1, population=POP1)
        demography.add_population_parameters_change(time=0, growth_rate=g2, population=POP2)
        
        if args.model == "mig":
            MG = args.MG / 2 / args.NF
            demography.set_symmetric_migration_rate(populations=[POP1, POP2], rate=MG)
        
        demography.add_population_split(time=TS, derived=[POP1, POP2], ancestral="ANC")
        demography.add_population_parameters_change(time=TS, growth_rate=g, population="ANC")
        demography.add_population_parameters_change(time=TG, growth_rate=0, population="ANC")
        
        graph = msprime.Demography.to_demes(demography)
        
        log_time = demesdraw.utils.log_time_heuristic(graph)
        log_size = demesdraw.utils.log_size_heuristic(graph)
        fig, ax = plt.subplots()
        
        demesdraw.tubes(
            graph,
            ax=ax,
            log_time=log_time,
            title=f"dadi_joint_{args.model}_posterior"
        )

        # Create output directory if it doesn't exist
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        
        plt.savefig(os.path.join(args.out_dir, f"dadi_joint_posterior_{args.model}.pdf"))
        demes.dump(graph, os.path.join(args.out_dir, f"dadi_joint_posterior_{args.model}.yaml"))
        
        print(f"Demographic history plot and YAML file for model '{args.model}' saved successfully in '{args.out_dir}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
