# Knowledge Graph Reasoning Experiments

This repository contains Python scripts and shell scripts for running experiments on knowledge graph reasoning models. The main experiment script (`experiment.py`) provides a wide range of functionality and configuration options for training and evaluating these models. The repository also includes several helper scripts that define important components used in the experiments.

## Project Structure

The project is organized into the following structure:

- `configs/`: Contains configuration scripts for different experiments.
- `data/`: Contains downloaded and processed data for WD-singer and FB15K-237-10 datasets.
- `src/`: Contains all the source code for the project.
 - `emb/`: Contains code related to embedding models.
 - `error_analysis/`: Contains jupyetr notebook for result analysis.
 - `rl/`: Contains code for backbone of algorithims implementation on reinforcement learning.
 - `utils/`: Contains utility functions and modules.
- `results/`: Contains sample output for reasoning paths in .txt format.
- `requirements.txt`: Specifies the required libraries and their versions.

## Key Scripts (all of them are under src/ directory)


### `experiment.py`

The `experiment.py` script serves as the main entry point for running experiments. It includes the following key components:

- `process_data()`: Processes the knowledge graph data.
- `initialize_model_directory(args, random_seed=None)`: Initializes the model directory based on the provided arguments and optional random seed.
- `construct_model(args)`: Constructs the neural network model based on the specified arguments.
- `train(lf)`: Trains the model using the provided training data.
- `inference(lf)`: Performs inference on the test set and computes evaluation metrics.
- `run_ablation_studies(args)`: Runs ablation study experiments.
- `export_to_embedding_projector(lf)`: Exports the model embeddings to the Tensorflow Embedding Projector format.
- `export_reward_shaping_parameters(lf)`: Exports the knowledge graph embeddings and fact network parameters for reward shaping models.
- `export_fuzzy_facts(lf)`: Exports the facts recovered by the embedding-based method.
- `export_error_cases(lf)`: Exports the error cases of a model.
- `compute_fact_scores(lf)`: Computes the fact scores using the embedding-based method.
- `run_experiment(args)`: Runs the experiment based on the provided arguments.

The script also includes various command-line arguments to control the experiment settings, such as the data directory, model architecture, hyperparameters, and evaluation options.

### `knowledge_graph.py`

The `knowledge_graph.py` script defines the `KnowledgeGraph` class, which represents a knowledge graph and provides functionality for loading and manipulating the graph data. It includes methods for loading entity and relation embeddings, computing entity and relation embeddings, performing graph traversal, and handling edge masks.

### `emb/fact_network.py`

The `emb/fact_network.py` script defines the `FactNetwork` class, which represents a fact network used for knowledge graph reasoning. It includes modules for entity and relation embeddings, as well as scoring functions for evaluating the plausibility of facts.

### `rl/graph_search/pn.py`

The `rl/graph_search/pn.py` script defines the `GraphSearchPolicy` class, which represents a graph search policy network used for reasoning in a knowledge graph. It handles action spaces, path initialization and updating, transition computation, and incorporates techniques such as action dropout and action space bucketing.

### `rl/graph_search/pg.py`

The `rl/graph_search/pg.py` script defines the `PolicyGradient` class, which implements a policy gradient algorithm for reinforcement learning in a knowledge graph setting. It includes components for computing the policy gradient loss, performing rollouts, sampling actions, and recording path traces.

## Running Experiments

To run experiments, you can use the provided shell scripts that automate the process of executing the `experiment.py` script with different configurations. Here's a brief overview of each shell script:

- `experiment.sh`: This script is used for preprocessing the knowledge graph data. It sets the appropriate flags based on the configuration file and executes the `experiment.py` script with the `--process_data` flag.
- `experiment-emb.sh`: This script is used for pre-training the embedding models (e.g., TransE, ComplEx, DistMult). It sets the necessary flags and hyperparameters for the selected embedding model and executes the `experiment.py` script with the `--train` flag.
- `experiment-rs.sh`: This script is used to run the full pipeline with reward shaping. It sets the required flags and hyperparameters for the reward shaping model and executes the `experiment.py` script with the `--train`, `--inference`, and other relevant flags.

To run an experiment, execute the desired shell script with the appropriate command-line arguments. For example:

```bash
./experiment.sh config/fb15k-237-10.sh
./experiment-emb.sh config/fb15k-237-10-emb.sh
./experiment-rs.sh config/fb15k-237-10-rs.sh