import torch

def generate_simple(model, input_ids, max_length, context_size):
    for _ in range(max_length):
        input_ids_crop = input_ids[:, -context_size:]
        with torch.no_grad():
            logits  = model(input_ids_crop)
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim = -1)
        next_token = torch.argmax(next_token_probs, dim = -1, keepdim = True)
        input_ids = torch.cat((input_ids, next_token), dim = 1)
    return input_ids