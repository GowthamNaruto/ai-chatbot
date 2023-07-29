""" Create a virtual environment to manage your Python packages. This can help isolate your project's dependencies and prevent conflicts with other projects or the system-wide Python installation."""
# Python -m venv myenu
# Activate the virtual environment
# myenv\Scripts\activate

import keras_nlp

# using KerasNLP with Keras Core
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")

# Promt the user to input some words to generate text
inputs = input("Write something: ")

# Generate words from the inputs at the length of 100 words
Response = gpt2_lm.generate(inputs, max_length=200)
print(f"Response: {Response}")
