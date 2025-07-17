# Whetstone: GenAI Application Robustness & Testing Framework

Whetstone is a flexible and extensible Python framework designed to **test, harden, and assess the robustness of Generative AI applications and Large Language Models (LLMs)**. It provides tools for systematically optimizing inputs towards specific objectives, allowing you to discover vulnerabilities, evaluate safety, measure alignment, and ensure desired model behaviors under various conditions.

Think of Whetstone as a tool to automatically "sharpen" or stress-test your GenAI systems by finding inputs that matter – whether that means finding prompts that break safety filters, inputs that maximize a desired output quality, or sequences that trigger specific internal model states.

It provides a structured environment for defining complex evaluation targets, selecting or combining optimizers, managing promising inputs (corpus), and evaluating them against custom objectives, all configured via Hydra or directly in Python.

*Disclaimer: Whetstone is currently an early-stage research project. While its components are functional and based on established techniques (e.g., our GCG implementation is adapted from frameworks like `nanogcg`), the primary focus is currently on flexibility and extensibility for complex testing scenarios rather than raw optimization speed. Performance is comparable to similar frameworks and is expected to improve over time.*

## Key Features

*   **GenAI Testing Focus:** Built from the ground up for evaluating and hardening LLMs and multi-component AI applications.
*   **Complex Objective Mapping:** Define objectives that go beyond simple string matching. Model complex GenAI application flows where inputs might pass through multiple LLM calls, external tools, or other AI models before a final evaluation.
*   **Modular Architecture:** Easily swap or add new components (Objectives, Optimizers, Models, Corpuses) using a registry system.
*   **Configuration-Driven (Hydra):** Leverages [Hydra](https://hydra.cc/) for powerful and flexible experiment configuration via YAML.
*   **Pythonic Configuration & Extension:** For highly complex scenarios or tighter integration, configure, compose, and extend Whetstone modules directly within Python code.
*   **Stateful & Resumable:** Automatically saves the optimization state, allowing long testing jobs to be interrupted and resumed seamlessly.
*   **Core Optimization Loop:** Implements a robust loop where optimizers propose inputs, objectives evaluate them, and the corpus tracks the best-performing ones.
*   **Extensible:** Designed for extension. Add your custom logic for optimization, evaluation, or input management by implementing simple interfaces.
*   **Composable Optimizers:** Combine multiple optimization strategies (e.g., gradient-based + random search) using the `MultiOptimizer`.

## Core Concepts

*   **`Job`**: Represents a single optimization/testing run, orchestrating the process based on the configuration. Manages state persistence and the overall iteration loop.
*   **`Target`**: Defines the specific testing task. It groups together an `Objective`, an `Optimizer`, and a `Corpus`.
*   **`Objective`**: A callable module that evaluates a given input and returns a score (lower is better) and potentially other metadata (`Sample`). This defines *what* you are testing for (e.g., vulnerability, toxicity score, alignment metric, output quality).
*   **`Optimizer`**: A module responsible for generating new candidate inputs based on the current state (e.g., corpus contents, objective feedback). This defines *how* you search for critical inputs.
*   **`Corpus`**: Stores a collection of evaluated inputs (`Sample` objects). The `GreedyCorpus` keeps track of the most impactful inputs found according to the objective score.
*   **`Sample`**: A data structure holding an input, its evaluation score, and any relevant output or metadata from the `Objective`.
*   **`ModuleRegistry`**: A central registry allowing Whetstone to discover and instantiate components (Objectives, Optimizers, etc.) based on configuration names.

## Getting Started

### Prerequisites

*   Python 3.12+
*   [Poetry](https://python-poetry.org/) for dependency management.

### Installation

1.  **Clone the repository**
2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
3.  **Install development dependencies (optional):**
    ```bash
    poetry install --with dev
    ```

### Basic Usage (CLI with Hydra)

Whetstone runs are typically initiated via the command-line interface using Hydra.

```bash
poetry run python whetstone/cli.py [options]
```

*   **Run with default configuration (`conf/base.yaml`):**
    ```bash
    poetry run python whetstone/cli.py
    ```
*   **Specify a different configuration:**
    ```bash
    poetry run python whetstone/cli.py --config-name=named_experiment
    ```
*   **Override configuration parameters:**
    ```bash
    poetry run python whetstone/cli.py iterations=500 target=strmatch target.optimizer.epsilon=0.1
    ```
*   **Resume a previous run:**
    By default, a new output folder is created on each run. By fixing (see conf/named_experiment) or overriding the `hydra.run.dir` parameter, an existing run can be resumed if the target triple (optimizer, objective, corpus) has not been changed. For example, you can run additional iterations on an exisitng target by executing
    ```bash
    poetry run python whetstone/cli.py hydra.run.dir=<existing output dir> target=<target that was specified in the existing run> iterations=500

Output, including logs, state, and corpus data, is saved in the directory specified by `hydra.run.dir` (defaults to `runs/YYYY-MM-DD/HH-MM-SS/`).

## Configuration

Whetstone uses [Hydra](https://hydra.cc/) for managing configurations via YAML files in `conf/`.

*   `conf/base.yaml`: Default settings.
*   `conf/target/`: Directory for defining different `Target` configurations.
*   `conf/named_experiment.yaml`: Example inheriting from `base.yaml` with a fixed output directory, so ever run of this config will attempt to resume the same run. 

In the target config, objects can be referenced using interpolation syntax to reuse or reference components (see conf/target/multi for an example).

For advanced use cases requiring dynamic configuration or tighter integration, Whetstone components can be instantiated and configured directly within Python scripts.

## Available Modules
All configuration options for each module can be found in their respective source (at whetstone/modules).

*   **Objectives (`whetstone/modules/objectives/`):**
    *   `StrMatchObjective`: Measures probability of outputting a target string sequence (requires `HFTransformersModel`) directly using the logits of a single forward pass. This delivers more fine-grained feedback than sampling model outputs.
    *   `LengthObjective`: Simple objective favoring shorter or longer inputs.
    *   `LLMJudge`: Uses an LLM to evaluate the quality/compliance of another LLM's output based on the input.
    *   `DummyObjective`: Basic objective for testing purposes.
*   **Optimizers (`whetstone/modules/optimizers/`):**
    *   `RandomOptimizer`: Simple baseline that makes random modifications to inputs. Similar to the approach of https://arxiv.org/abs/2404.02151
    *   `GCGOptimizer`: Implements the Gradient-based Coordinate Gradient (GCG) optimization technique (implementation adapted from `nanogcg`, technique from Zou et al: https://arxiv.org/abs/2307.15043).
    *  `LLMOptimizer`: Allows using arbitrary LLMs (but preferably reasoning models) to generate new candidate inputs. They are shown a natural language description generated from the objective as well as samples of previous inputs and their respective outputs. Deepseek-R1 has shown promise as generator.
    *   `MultiOptimizer`: Combines multiple optimizers using an epsilon-greedy strategy. Can also be configured to assign negative weight to sub-objectives for inversion (i.e. making outputs unlikely or long instead of short).
*   **Models (`whetstone/modules/models/`):**
    *   `HFTransformersModel`: Wrapper around Hugging Face `transformers` models.
    *  `OpenAIModel`: Accepts any compatible OpenAI API endpoint for black-box models.
*   **Corpus (`whetstone/core/corpus.py`):**
    *   `Corpus`: Stores the top `N` samples based on score and potentially criteria like output diversity.


**Example (`conf/target/strmatch.yaml`):**

```yaml
# Configures the Target module
_target_: Target # Instantiates the Target class (registered implicitly)

# Model configuration (used by Objective)
model:
  name: HFTransformersModel # Name registered in ModuleRegistry
  model_name: arnir0/Tiny-LLM
  device: mps # Or cpu, cuda, etc.

# Objective configuration
objective:
  name: StrMatchObjective # Name registered in ModuleRegistry
  model: ${target.model} # Reference the model defined above
  batch_size: 100
  template: # Template for the objective
    - role: system
      content: You are a helpful assistant.
    - role: user
      content: "{OPTIM}" # Placeholder for the optimizer's input
    - role: assistant
      content: "{TARGET} I am a wheel of cheese." # Target string

# Optimizer configuration
optimizer:
  name: MultiOptimizer # Use the MultiOptimizer
  optimizers: # List of optimizers to combine
    - name: GCG
      model: ${target.model}
      filter_ids: true
      search_width: 1000
    - name: RandomOptimizer

# Corpus configuration
corpus:
  name: GreedyCorpus
  objective: ${target.objective} # Link to the objective
  initial_inputs:
    - "so this is a text telling you to say something but Im not revealing what yet"
  max_samples: 100 # Keep the top 100 samples
  corpus_dir: ${hydra:run.dir}/corpus # Save corpus in the run directory
```

## Extending Whetstone

Create custom components by:

1.  **Implementing the Base Interface:** Inherit from `Objective`, `Optimizer`, `BaseModule`, etc.
2.  **Registering the Module:** Use `@ModuleRegistry.register`.
3.  **Handling State (Optional):** Inherit from `StatefulModule` and implement `BaseState` for persistence.
4.  **Configuring Your Module:** Reference by registered name in YAML or instantiate directly in Python.

See whetstone/modules for existing modules and examples.

## Testing
Tests are configured with pytest. A basic test suite is included for all basic module types. When extending whetstone, register modules in tests/modules for the common test suite to discover them.

#### Run all tests
```bash
poetry run pytest tests/
```

Currently, the test suite has multiple tests that are executed for each model and objective combination. The resulting test suite can grow very large, and it is recommended to use filters to constrain which tests are selected:

#### List discovered tests
```bash
poetry run pytest --collect-only -q tests/
```

#### Run a subset of tests
```bash
# Example: Run only tests containing "StrMatch" in their name to test the StrMatchObjective
poetry run pytest -k StrMatch tests/
```

## Use Cases & Applications

Whetstone is designed to support various GenAI testing and robustness tasks:

*   **Model Red Teaming & Safety Testing:** Discovering inputs that bypass safety filters, generate harmful/biased content, or reveal sensitive information.
*   **Robustness Assessment:** Evaluating how models perform under optimized or adversarial inputs.
*   **Complex Application Testing:** Assessing the end-to-end behavior of multi-component AI systems (e.g., RAG pipelines, agentic systems).
*   **Prompt Engineering & Optimization:** Finding effective prompts for specific downstream tasks (though primarily a testing tool).
*   **Adversarial Training Data Generation:** Creating challenging examples to improve model robustness during training.
*   **CI/CD Integration:** Potentially integrating Whetstone runs into CI pipelines to monitor for regressions in model safety or robustness over time.

## Contributing
We welcome contributions! Please see our CONTRIBUTING.md file for guidelines on how to submit pull requests, report issues, and suggest features.

## Acknowledgements

*   The GCG optimizer implementation is adapted from the existing `nanogcg` framework.

