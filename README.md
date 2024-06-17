# Markov Model Simulation for Multiple Myeloma Treatments

This project simulates the progression of multiple myeloma through different sequences of treatments using a Markov model. The simulation evaluates the probabilities of different states (Remission, Stable Disease, Progression, Relapse, Death) over time and saves the resulting plots as image files.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have Python 3.x installed on your machine.
- You have `pip` installed for managing Python packages.

## Installation

1. Open a terminal on your Mac.

2. Install the required Python libraries (`numpy` and `matplotlib`) using the following commands:

    ```bash
    pip3 install numpy matplotlib
    ```

    If you encounter permission errors, use `sudo`:

    ```bash
    sudo pip3 install numpy matplotlib
    ```

## Usage

1. **Clone the Repository or Download the Script**:

    - If you are using a Git repository, clone it using:
      ```bash
      git clone <repository_url>
      cd <repository_directory>
      ```

    - If you are not using a Git repository, simply save the provided script to a file named `markov_model_optimal_treatment_save.py`.

2. **Run the Script**:

    - Navigate to the directory containing the script:
      ```bash
      cd path/to/your/directory
      ```

    - Run the script using Python 3:
      ```bash
      python3 markov_model_optimal_treatment_save.py
      ```

3. **View the Results**:

    - The script will generate and save plots showing the probabilities of being in each state over time for different sequences of treatments.
    - The plots will be saved as image files (`sequence_1.png`, `sequence_2.png`, `sequence_3.png`) in the same directory as the script.

## Project Description

The project uses a Markov model to simulate the progression of multiple myeloma under different treatment sequences. Each treatment has its own transition probabilities, and the model accounts for increasing relapse rates over time. The simulation helps identify the most effective sequences of treatments based on the probabilities of different states.

### Key Components

- **Transition Matrices**: Define the probabilities of moving between different states for each treatment.
- **Markov Model Simulation**: Simulates the progression through different treatments and evaluates the outcomes.
- **Plotting Results**: Generates and saves plots showing the state probabilities over time.

## Example Code

Here is a snippet of the main code for reference:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define states
states = ["Remission", "Stable Disease", "Progression", "Relapse", "Death"]
num_states = len(states)

# Define base probabilities for each treatment (these would be derived from your data)
base_probabilities_list = [
    {"remission": 0.7, "stable": 0.2, "progression": 0.1, "relapse": 0.05, "death": 0.05},
    {"remission": 0.6, "stable": 0.25, "progression": 0.1, "relapse": 0.05, "death": 0.05},
    {"remission": 0.65, "stable": 0.2, "progression": 0.1, "relapse": 0.05, "death": 0.05},
    {"remission": 0.55, "stable": 0.3, "progression": 0.1, "relapse": 0.05, "death": 0.05},
    {"remission": 0.5, "stable": 0.3, "progression": 0.1, "relapse": 0.05, "death": 0.05}
]

# Function to create transition matrices with increasing relapse rates
def create_transition_matrix(base_probabilities, attrition_factor):
    relapse_factor = base_probabilities["relapse"] + attrition_factor
    return np.array([
        [base_probabilities["remission"] * (1 - attrition_factor), 0.2, 0.1, relapse_factor, 0.05],  # From Remission
        [0.0, base_probabilities["stable"] * (1 - attrition_factor), 0.3, relapse_factor, 0.1],       # From Stable Disease
        [0.0, 0.0, base_probabilities["progression"] * (1 - attrition_factor), relapse_factor, 0.1],  # From Progression
        [0.0, 0.0, 0.0, relapse_factor, 0.5],                                                         # From Relapse
        [0.0, 0.0, 0.0, 0.0, 1.0]                                                                    # From Death
    ])

# Create transition matrices for each treatment
attrition_rate = 0.10
transition_matrices_list = [
    create_transition_matrix(base_probabilities, attrition_rate * i)
    for i, base_probabilities in enumerate(base_probabilities_list)
]

# Initial state distribution (starting in Remission)
initial_distribution = np.array([1, 0, 0, 0, 0])

# Number of cycles to simulate
num_cycles = 20

# Function to simulate the Markov model for a given sequence of treatments
def simulate_markov_sequence(transition_matrices_list, initial_distribution, num_cycles):
    num_treatments = len(transition_matrices_list)
    state_distributions = np.zeros((num_cycles, num_states))
    state_distributions[0] = initial_distribution
    
    for cycle in range(1, num_cycles):
        treatment_index = min(cycle // (num_cycles // num_treatments), num_treatments - 1)
        state_distributions[cycle] = np.dot(state_distributions[cycle-1], transition_matrices_list[treatment_index])
    
    return state_distributions

# Define sequences to test (this is a simplification, more sophisticated optimization is needed for real use)
sequences = [
    [0, 1, 2, 3, 4],  # Sequence 1: Treatment 1 -> Treatment 2 -> Treatment 3 -> Treatment 4 -> Treatment 5
    [4, 3, 2, 1, 0],  # Sequence 2: Treatment 5 -> Treatment 4 -> Treatment 3 -> Treatment 2 -> Treatment 1
    [0, 2, 4, 1, 3]   # Sequence 3: Mixed sequence
]

# Simulate and save each sequence plot
for seq_index, sequence in enumerate(sequences):
    transition_matrices_sequence = [transition_matrices_list[i] for i in sequence]
    state_distributions = simulate_markov_sequence(transition_matrices_sequence, initial_distribution, num_cycles)
    
    plt.figure(figsize=(12, 6))
    for i, state in enumerate(states):
        plt.plot(state_distributions[:, i], label=state)
    plt.xlabel('Cycle')
    plt.ylabel('Probability')
    plt.title(f'Sequence {seq_index + 1}: {sequence}')
    plt.legend()
    
    # Save the plot
    plt.savefig(f'sequence_{seq_index + 1}.png')
    plt.close()

print("Plots have been saved successfully.")
