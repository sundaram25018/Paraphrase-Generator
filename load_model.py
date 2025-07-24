from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Vamsi/T5_Paraphrase_Paws"
save_path = "./local_models/t5_paraphrase"  # change to your desired path

# Download and save model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)