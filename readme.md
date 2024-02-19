# Stationing Simulator for MCTS

See [here](https://zenodo.org/records/10594255) for the Zenodo entry for this Artifact.
## Requirements:
Programs: Either Docker or Python 3.11

Machine used for testing:
* CPU: AMD Ryzen Threadripper 1950X 16-Core Processor 3.7MHz
* RAM: 94GB total, 300G swap
* GPU: NVIDIA TITAN Xp 12GB x4
* OS: Ubuntu 18.04.5 LTS

## Workspace
### Unix Machines
1. Clone the repo:
    ``` bash
    git clone https://github.com/jptalusan/ICCPS2024_AE_Stationing.git
    ```
2. Setup the repo:
    ```bash
    cd ICCPS2024_AE_Stationing
    wget --no-check-certificate "https://zenodo.org/records/10594255/files/ARTIFACT_FILES.tar.gz?download=1" -O ARTIFACT_FILES.tar.gz
    tar -xzvf ARTIFACT_FILES.tar.gz
    ```
3. Adjusting the experiment if desired:
* "early_end": Can be set to false, to run through the entire day.
* "method": "mcts" or "baseline"
    * If MCTS, "iter_limit" count can be adjusted while "pool_thread_count" should be at least 1.
* "noise_level": "", 1, 5, 10
* "overload_start_depots": Dictionary of where the substitute buses will start from.
4. Running the experiment:

    Run experiment using Python:
    ```bash
    cd code_root/experiments/TEST
    python run_mcts_no_inject.py -c configs/test
    ```
    Running using Docker:
    ```bash
    docker build -t iccps2024_stationing .
    docker run -v $PWD/code_root:/usr/src/app/code_root iccps2024_stationing
    ```
* Note: If running using Docker Desktop on a Mac, you might need to allow file sharing on the current git repo directory.

### Windows Machines
1. Clone the repo:
    ``` bash
    git clone https://github.com/jptalusan/ICCPS2024_AE_Stationing.git
    ```
2. Setup the repo:
    ```bash
    cd ICCPS2024_AE_Stationing
    curl "https://zenodo.org/records/10594255/files/ARTIFACT_FILES.tar.gz?download=1" -o ARTIFACT_FILES.tar.gz
    tar -xzvf ARTIFACT_FILES.tar.gz
    ```
3. Adjusting the experiment if desired:
* "early_end": Can be set to false, to run through the entire day.
* "method": "mcts" or "baseline"
    * If MCTS, "iter_limit" count can be adjusted while "pool_thread_count" should be at least 1.
* "noise_level": "", 1, 5, 10
* "overload_start_depots": Dictionary of where the substitute buses will start from.
4. Running the experiment:

    Run experiment using Python:
    ```bash
    cd code_root\experiments\TEST
    python run_mcts_no_inject.py -c configs/test
    ```
    Running using Docker:
    ```
    docker build -t iccps2024_stationing .
    docker run -v "%cd%"/code_root:/usr/src/app/code_root iccps2024_stationing
    ```

## Verify the results
Plots will appear in `code_root/experiments/TEST/plots` they will be labeled by figure number and correspond to the figures in the Appendix.

```bash
less -S code_root/experiments/TEST/logs/20210211_test/stream.log
less -S code_root/experiments/TEST/results/20210211_test/results.csv
less -S code_root/experiments/TEST/results/20210211_test/stops_results.csv
less -S code_root/experiments/TEST/results/20210211_test/buses_results.csv
```
> Note the datetime in the results are in UTC time. The first one contains the raw logs detailing the bus movement and passenger pickups and dropoffs. The second one is a summary containing 3 distinct CSVs and a summary of the results at the bottom.

## Extra Information
ARTIFACT_FILES.tar.gz contains the processed data and input files. They are extracted to the following folders:
* Config file -> code_root/experiments/TEST/configs
* prepped_disruptions.parquet -> code_root/experiments/TEST/
* Input data -> code_root/scenarios/REAL_WORLD
* Input data -> code_root/scenarios/TEST_WORLD
    * These are the inputs from the transit agency (test plan and bus plans).
    * Includes the occupancy predictions.
* code_root/scenarios/common/sampled_travel_times_dict.pkl
* code_root/scenarios/common/stops_tt_dd_node_dict.pkl
* code_root/scenarios/common/stops_node_matching_dict.pkl

To get all of this from the base code for use in the evaluation:
1. Navigate to the root `mta_simulator_redo`
2. Run:
```
    tar -cvzf ARTIFACT_FILES.tar.gz -C . \
    code_root/scenarios/common/sampled_travel_times_dict.pkl \
    code_root/scenarios/common/stops_node_matching_dict.pkl \
    code_root/scenarios/common/stops_tt_dd_node_dict.pkl \
    code_root/scenarios/REAL_WORLD \
    code_root/scenarios/TEST_WORLD \
    data_analysis/data/best_solutions.parquet \
    code_root/experiments/TEST/prepped_disruptions.parquet
```