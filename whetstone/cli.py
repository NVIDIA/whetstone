# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from datetime import datetime

from whetstone.core import *
from whetstone.modules import *


console = Console()

@hydra.main(version_base=None, config_path="../conf", config_name="base")
def whetstone(cfg : DictConfig) -> None:
    job: Job = ModuleRegistry.instantiate(cfg, Job)

    if os.path.exists(job.state_location):
        with open(job.state_location, "r") as f:
            try:
                ModuleRegistry.rehydrate_state(json.load(f))
            except Exception as e:
                pwd = os.getcwd()
                folder = os.path.join(pwd, job.state_location)
                raise Exception(f"""Failed to rehydrate state. 
You may have changed the config between runs. If so, you can either run with the saved config: 
{folder}/.hydra/config.yaml 

or delete the folder {folder} to start fresh.""") from e


    # Signal handlers to save state on SIGINT and SIGTERM
    import signal
    import sys

    def save_state_on_signal(signum, frame):
        nonlocal job
        job.save_state()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_state_on_signal)
    signal.signal(signal.SIGTERM, save_state_on_signal)

    from tqdm import tqdm

    previous_best_score = float('inf')
    recent_samples = [] # Max 10 most recent samples

    def sample_callback(samples: List[Sample]):
        nonlocal recent_samples
        recent_samples.extend(samples)
        recent_samples = recent_samples[-10:]

    previous_bests = set()
    
    try:
        for iteration in tqdm(job.run(sample_callback), total=job.iterations, desc="Iterations",
                          initial=len(job.state.iterations)):
            # Get the overall best input and score
            best_input = job.target.corpus.best_input
            best_score = job.target.corpus.best_score
            
            # Create a pretty table
            table = Table(box=box.ROUNDED)
            table.add_column("Type", style="cyan")
            table.add_column("Input", style="white")
            table.add_column("Score", style="yellow")
            table.add_column("Status", style="green")
            
            # Add recent samples
            for sample in recent_samples:
                status = ""
                if sample.score < previous_best_score or sample.input in previous_bests:
                    status = "🎯 New Best!"
                    previous_best_score = sample.score
                    previous_bests.add(sample.input)
                table.add_row(
                    "Recent",
                    str(sample.input),
                    f"{sample.score:.6f}",
                    status
                )
            
            # Add best sample
            if best_input is not None:
                table.add_row(
                    "Best",
                    str(best_input),
                    f"{best_score:.6f}",
                    "🏆",
                    style="bold green"
                )
            
            # Clear previous output and show new table
            console.clear()
            console.print(table)

            # Save state
            if len(job.state.iterations) % job.save_interval == 0:
                job.save_state()

    except Exception as e:
        console.print(f"Error: {e}")
        job.save_state()
        raise e
    
    # End of run, printcomprehensive statistics
    console.clear()
    
    # Create a statistics table
    stats_table = Table(title="Optimization Run Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan", justify="right")
    stats_table.add_column("Value", style="yellow")
    
    # Calculate run duration
    run_duration = datetime.now() - job.state.started_at
    
    # Add statistics rows
    stats_table.add_row("Total Iterations", str(len(job.state.iterations)))
    stats_table.add_row("Total Samples", str(job.state.total_samples))
    stats_table.add_row("Run Duration", str(run_duration))
    stats_table.add_row("Best Score", f"{job.target.corpus.best_score:.6f}")
    stats_table.add_row("Best Input", str(job.target.corpus.best_input))
    stats_table.add_row("Corpus Size", str(len(job.target.corpus)))
    
    # Print the statistics
    console.print("\n")
    console.print(Text("🎯 Optimization Complete!", style="bold green"))
    console.print("\n")
    console.print(stats_table)
    
    job.save_state()

if __name__ == "__main__":
    whetstone()