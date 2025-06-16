# API Recommendation System

An intelligent system for recommending APIs based on tags, descriptions, and semantic analysis, built on a Heterogeneous Graph Neural Network (HeteroGNN) architecture.

## Overview

This system recommends relevant APIs based on user input tags and descriptions by leveraging graph-based machine learning techniques. It features explainable recommendations, tag validation, and debugging capabilities to help users understand why specific APIs are recommended.

## System Architecture

The system consists of several interconnected components:

### Components

- **Graph Generation (`graph_gen.py`)**: Constructs a heterogeneous graph with mashups and APIs as nodes, using tag and description-based features.
- **Model (`model.py`)**: Contains the HeteroGNN model that processes the graph data for recommendations.
- **Trainer (`trainer.py`)**: Handles model training with data balancing and evaluation procedures.
- **Tuner (`tuner.py`)**: Provides hyperparameter tuning capabilities for optimizing model performance.
- **Input Processing (`input.py`)**: Manages user input processing, tag validation, and recommendation generation.
- **Command Line Interface (`cli.py`)**: Provides a user-friendly command-line interface for interacting with the system.
- **Demo (`demo.py`)**: Contains demonstration scripts to showcase system capabilities.

### Data Flow

1. Raw data from CSV files is processed into a heterogeneous graph by `graph_gen.py`.
2. The graph is used to train the HeteroGNN model through `trainer.py`.
3. User queries (tags and descriptions) are processed by `input.py`.
4. Recommendations are generated based on node embeddings and semantic similarity.
5. Results are presented with explanations through the CLI or interactive session.

## Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- scikit-learn
- pandas
- numpy
- tqdm

### Setup

1. Clone the repository
2. Install required dependencies:

```bash
pip install torch torch_geometric pandas numpy scikit-learn tqdm matplotlib seaborn
```

3. Ensure the dataset files are in the correct location:
   - `csv/api_nodes.csv`: API information
   - `csv/mashup_nodes.csv`: Mashup information
   - `csv/mashup_api_edges.csv`: Edges between mashups and APIs

## Usage

### Command Line Interface

The system can be used through the CLI with various options:

```bash
# Basic recommendation with tags
python cli.py --tags "social,mapping,location" --top-k 5

# Recommendation with tags and description
python cli.py --tags "ecommerce,financial" --description "Payment processing system" --top-k 3

# Interactive mode
python cli.py --interactive

# Run demo
python cli.py --demo

# Quick demo
python cli.py --quick-demo

# Test tag validation
python cli.py --validate "social,fakecategory,mapping"
```

### Interactive Mode

In interactive mode:

1. Enter tags separated by commas (e.g., "social,mapping,search")
2. Optionally provide a description
3. Specify the number of recommendations to show
4. Choose whether to enable explainability features

### Examples

```bash
# Get recommendations for social mapping applications
python cli.py --tags "social,mapping,location" --description "Social mapping application with location features" --top-k 5

# Run comprehensive demo
python cli.py --demo
```

## Features

### Tag Validation and Suggestions

The system validates input tags against a known vocabulary and provides:

- Recognition of valid and invalid tags
- Coverage percentage calculation
- Suggestions for similar tags when invalid tags are entered

### Explainable Recommendations

Each API recommendation includes:

- Explanation of why the API was recommended
- Tag overlap analysis
- Quality indicators (Excellent/Good/Fair/Weak match)
- Coverage percentage

### Debugging Capabilities

Debugging features include:

- Detailed tag analysis
- Embedding similarity visualization
- Tag-based clustering analysis
- Detailed recommendation scoring

## Training Your Own Model

To train or tune the model with your own parameters:

```bash
# Generate the graph dataset
python graph_gen.py

# Train the model
python trainer.py

# Perform hyperparameter tuning
python tuner.py
```

### Hyperparameter Tuning

Tuning explores multiple configurations including:

- Learning rates
- Hidden dimensions
- Number of layers
- Dropout rates
- Weight decay values
- Attention heads
- Edge sampling strategies

Results are saved in the `hetero_gnn_results` directory with plots and performance metrics.

## File Structure

- `model.py`: HeteroGNN model definition
- `trainer.py`: Model training procedures
- `tuner.py`: Hyperparameter tuning
- `graph_gen.py`: Graph generation from raw data
- `input.py`: Input processing and recommendation generation
- `cli.py`: Command-line interface
- `demo.py`: Demonstration scripts
- `dataset.pt`: Processed graph dataset
- `model.pt`: Trained model weights
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer for text processing
- `tag_analysis.pkl`: Tag analysis data
- `csv/`: Directory containing raw data files
- `hetero_gnn_results/`: Directory containing tuning results

## Model Architecture

The HeteroGNN model:

- Uses heterogeneous graph convolution with attention mechanisms
- Processes different node types (mashups and APIs)
- Features multi-layer architecture with residual connections
- Applies batch normalization and dropout for regularization
- Employs cosine similarity for recommendation scoring

## Performance and Metrics

The model is evaluated using several metrics:

- AUC (Area Under the ROC Curve)
- AP (Average Precision)
- Accuracy, Precision, and Recall
- Tag coverage and semantic similarity

## Contributing

Contributions are welcome! Please feel free to submit pull requests.
