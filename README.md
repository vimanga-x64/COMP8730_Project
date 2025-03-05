# Automated Interlinear Glossing Pipeline

This project implements an enhanced pipeline for automated interlinear glossing. The system uses a Transformer-based character encoder with relative positional encodings, an unsupervised morpheme segmentation module with adaptive thresholding and a full forward–backward algorithm (with utility masking), a translation encoder, and a Transformer decoder with cross-attention to generate gloss sequences from source words and their translations.

## Project Architecture

- **Encoder.py**  
  Implements a Transformer-based character encoder with relative positional encodings.  
  **Input:** One-hot encoded source characters (shape: `(batch_size, seq_len, input_size)`) and sequence lengths.  
  **Output:** Contextualized embeddings of shape `(batch_size, seq_len, embed_dim)`, which serve as input to the segmentation module.

- **MorphemeSegmenter.py**  
  Implements an unsupervised segmentation module that computes segmentation probabilities for each character, predicts an adaptive threshold from rich encoder statistics (max, mean, variance), and applies a forward–backward algorithm with utility masks (via `make_mask_2d` and `make_mask_3d`) to produce a binary segmentation mask and auxiliary outputs (predicted morpheme count and raw segmentation probabilities).  
  **Input:** Encoder outputs and valid sequence lengths (and optionally target morpheme counts).  
  **Output:** Binary segmentation mask, morpheme count, adaptive threshold, and segmentation probabilities.

- **GlossingDecoder.py**  
  Implements a Transformer decoder that generates gloss tokens. It uses cross-attention over a memory that is formed by concatenating aggregated morpheme representations (derived via an aggregation function from the segmentation mask) with a translation representation.  
  **Input:** Target gloss token indices (for teacher forcing during training) and a memory tensor (aggregated segments + translation representation).  
  **Output:** Logits over the gloss vocabulary for each token.

- **Utilities.py**  
  Contains helper functions for the pipeline:
  - **Masking Functions:** `make_mask`, `make_mask_2d`, and `make_mask_3d` create boolean masks to handle variable-length sequences.
  - **Aggregation:** `aggregate_segments` aggregates contiguous character encoder outputs (based on the segmentation mask) into fixed-size morpheme-level representations.
  - **Pooling:** `max_pool_2d` is used for auxiliary pooling operations if needed.

- **GlossingModel.py**  
  Integrates the above modules into a full end-to-end system (implemented as a PyTorch Lightning module). The pipeline takes one-hot encoded source features, source lengths, target gloss tokens, and translation tokens; it processes these through the encoder, segmentation module, and decoder to generate gloss predictions.  
  **Output:** The decoder produces logits over the gloss vocabulary, along with auxiliary segmentation outputs (morpheme count, adaptive threshold, and segmentation probabilities).

- **main.py**  
  Contains the training script that:
  - Loads the dataset (from `Dummy_Dataset.csv`),
  - Builds vocabularies for source characters, gloss tokens, and translation tokens,
  - Creates a PyTorch DataLoader,
  - Trains the integrated glossing model using PyTorch Lightning, and
  - Saves a model checkpoint (e.g., `glossing_model.ckpt`).

- **Sample.py**  
  Provides a sample prediction script that:
  - Loads the dataset (to retrieve vocabularies and a sample input),
  - Loads the trained model checkpoint,
  - Converts the source input into the required one-hot format,
  - Runs the integrated model to generate gloss predictions, and
  - Outputs the predicted gloss (truncated at the first occurrence of the `</s>` token).

## Installation

This project is best run using conda 
however if you do not have conda already installed and configured
we recommend you use python to run the start up scripts immediately and get the preliminary results.

We recommend the use of conda due to GPU access,
for these purposes to just run the sample train and evaluate scripts a 
CPU will suffice.
Follow these steps:

- Use Python versions between 3.11.0 to 3.12.7

**PYTHON VIEW**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gfkaceli/COMP8730_Project.git
   cd COMP8730_Project
2. **Create Environment and Install Dependencies**
   - A. Create Virtual Environment:
      ```bash
      python -m venv venv
   - B. Activate Virtual Environment:
     - For Windows:
        ```bash
        venv\Scripts\activate
     - For Linux/MacOS:
       ```bash
        source venv/bin/activate
   - C. To Install Dependencies Do:
       ```bash
        pip install -r requirements.txt
   
4. **Run the Training Script**
    ```bash
   python main.py

5. **Run the Inference Script**
    ```bash
   python Sample.py

**CONDA VIEW**

- Conda version utilized is 24.11.3

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gfkaceli/COMP8730_Project.git
   cd COMP8730_Project
   
2. **Create Environment and Install Dependencies**
   ```bash
   conda create -n glossing_env python=3.12.7 pip # python version 3.11.0 to 3.12.7 should work so feel free to change
   conda activate glossing_env
   pip install -r requirements.txt

3. **Run the Training Script**
    ```bash
   python main.py

4. **Run the Inference Script**
    ```bash
   python Sample.py
   
## Notes

- After training and `lightning_logs` directory 
  will be created. In this directory you will see
  the training metrics, parameters and checkpoints
  for the model, also the final model checkpoint
  will be saved as `glossing_model.ckpt`
- `glossing_model.ckpt` will be loaded into Sample.py
   to run inference, make sure the model parameters during
   training and inference line up, otherwise the model will
   not process the input.
   

# 
