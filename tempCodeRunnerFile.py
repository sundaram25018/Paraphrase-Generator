
@torch.no_grad()
def paraphrase_chunk(text, max_input_length=512, max_output_length=512, top_k=50, top_p=0.90):