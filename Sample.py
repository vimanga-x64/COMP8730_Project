
from GlossingModel import GlossingPipeline
import torch
import torch.nn.functional as F
from main import GlossingDataset

#########################################
# 3. Inference Function
#########################################
def predict_gloss(model, dataset, source_text, translation_text, max_len=20):
    # Convert input text to tensors
    src_tensor = dataset.text_to_tensor(source_text, dataset.src_vocab,max_len, char_level=True).unsqueeze(0)  # (1, seq_len)
    trans_tensor = dataset.text_to_tensor(translation_text, dataset.trans_vocab,max_len, char_level=False).unsqueeze(
        0)  # (1, seq_len)

    # Convert source tensor to one-hot encoding (required for encoder)
    src_tensor = F.one_hot(src_tensor, num_classes=len(dataset.src_vocab)).float()

    # Start with an empty output sequence
    tgt_tensor = torch.zeros((1, 1), dtype=torch.long).to(model.device)  # (1, 1) placeholder

    generated_tokens = []

    for _ in range(max_len):  # Generate up to max_len tokens
        gloss_logits, _, _, _ = model(src_tensor, torch.tensor([src_tensor.shape[1]]), tgt_tensor, trans_tensor)

        # Getting the next token
        next_token = torch.argmax(gloss_logits[:, -1, :], dim=-1).item()

        if next_token == dataset.gloss_vocab["<pad>"]:  # Stop if padding token is predicted
            break

        generated_tokens.append(next_token)

        # Append new token to tgt_tensor
        tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]], dtype=torch.long).to(model.device)], dim=1)

    # Convert generated indices to gloss text
    gloss_text = dataset.tensor_to_text(torch.tensor(generated_tokens), dataset.gloss_vocab)
    return gloss_text[:20]


torch.manual_seed(42)

dataset = GlossingDataset("data/Dummy_Dataset.csv")

trained_model = GlossingPipeline.load_from_checkpoint("glossing_model.ckpt")

# Test Run
print("The predicted gloss for language 'inopi-a' with translation 'a wine shortage' is:")
print(predict_gloss(trained_model, dataset, "inopi-a", "a wine shortage"))