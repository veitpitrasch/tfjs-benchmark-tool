from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Lade GPT-2 + Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Speichere das Modell im TensorFlow SavedModel-Format (wichtig!)
tf.saved_model.save(model, "./gpt2-savedmodel")
