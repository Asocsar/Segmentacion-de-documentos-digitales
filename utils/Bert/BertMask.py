import torch

def get_features(input_ids, tokenizer, device=None):
    attention_mask = []

    # Iterate over batch
    for input_ids_example in input_ids:
        # Convert tensor to a 1D list
        input_ids_example = input_ids_example.squeeze().tolist()
        # Set example to whole input when batch size is 1
        if input_ids.shape[0] == 1:
            input_ids_example = input_ids.squeeze().tolist()
        # Get padding information
        padding_token_id = tokenizer.encode('[PAD]')[0]
        padding_length = input_ids_example.count(padding_token_id)
        text_length = len(input_ids_example) - padding_length


        # Get input mask -> 1 for real tokens, 0 for padding tokens
        attention_mask_example = ([1] * text_length) + ([0] * padding_length)


        assert len(attention_mask_example) == len(input_ids_example)
        attention_mask.append(attention_mask_example)

    if device is None:
        attention_mask = torch.tensor(data=attention_mask).cuda()
    else:  
        attention_mask = torch.tensor(data=attention_mask, device=device)
    return attention_mask