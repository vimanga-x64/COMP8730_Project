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
  Contains the training script which:
  - Loads the glossing data (via `data.py`),
  - Builds the necessary vocabularies,
  - Creates DataLoader objects,
  - Trains the integrated glossing model using PyTorch Lightning, and
  - Saves the model checkpoint.
  - Runs inference and returns metrics

command-line arguments (batch size, epochs, language, etc.) can be parsed using `argparse` to configure training dynamically. 
Additionally, the script prints out the predictions alongside the true gloss for evaluation.

- **data.py**
  - Loads the datasets by using the DataLoader Object
  **Input:** Takes a data file
  **Output:** Outputs a DataLoader object for train, validation and test sets

- **metrics.py**
  Contains helper functions for our metric calculations:
  - **Word Level Gloss Accuracy**
  - **Morpheme Level Gloss Accuracy**


- **old_main.py**  
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
   
3. **Make Models Directory (if you do not have it already)**
    ```bash
   mkdir models

4. **Run the Training and Inference Scripts**
    ```bash
   python main.py --language Gitksan --batch 7 --epochs 20
   python main.py --language Lezgi --batch 128 --numheads 32 --epochs 30
   python main.py --language Natugu --dropout 0.1354 --batch 128 --numheads 64 --epochs 20
   python main.py --language Tsez --batch 128 --epochs 25


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

3. **Make Models Directory (if you do not have it already)**
    ```bash
       mkdir models

4. **Run the Training and Inference Script**
    ```bash
   python main.py --language Gitksan --batch 7 --epochs 20
   python main.py --language Lezgi --batch 128 --numheads 32 --epochs 30
   python main.py --language Natugu --dropout 0.1354 --batch 128 --numheads 64 --epochs 20
   python main.py --language Tsez --batch 128 --epochs 25

   
## Notes

- After training and `lightning_logs` directory 
  will be created. In this directory you will see
  the training metrics, parameters and checkpoints
  for the model, also the final model checkpoint
  will be saved as `models/glossing_model_{lang}.ckpt`
- Training and Prediction is done in main.py feel free to 
  tune or adjust the hyperparameters to your liking
- the possible hyperparameters to tune are as follows:
  - **--batch:**  the size of the batch
  - **--layers:** the number of layers
  - **--dropout:** the dropout rate per layer
  - **--lr:** the learning rate
  - **--embdim:** the embedding dimensions
  - **--ffdim:** the feed forward dimensions
  - **--numheads:** the number of attention heads
  - **--epochs:** the number of epochs

  
## Contact

Any questions or concerns reach out to **kaceli@uwindsor.ca** or **umange@uwindsor.ca**
