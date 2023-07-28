import os
import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Setting and Hyperparameters
# Data
BATCH_SIZE = 64
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 450

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # Limits parameters in model

# Training
EPOCHS = 6

# Inference
NUM_TOKEN_TO_GENERATE = 80

# Load the data
keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

# Load simplebooks-92 train set and filter out short lines.
raw_train_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# Load simplebooks-92 validation set and filter out short lines.
raw_val_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
)

# Train the Tokenizer
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

# Load tokenizer
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

# Tokenize data
start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    lables = outputs
    return features, lables


# Tokenize and split into train and label sequences.
train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)

# Build the model
inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
# Embedding
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)
# Transformer docoders
for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)  # Giving one arugument only skip cross-attention
# Ouptut
outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])
model.summary()

# Training
model.fit(train_ds, validation_data=val_ds, verbose=2, epochs=EPOCHS)

# Inference
# The "packer" layers adds the [BOS] token for us
prompt_tokens = start_packer(tokenizer([""]))
prompt_tokens


def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    # Ignore hidden states for now; only needed for contrastive search
    hidden_states = None
    return logits, hidden_states, cache


# Greedy search
sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,  # Start sampling immediately after the [BOS] token
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")

# # Beam search
# sampler = keras_nlp.samplers.BeamSampler(num_beams=10)
# output_tokens = sampler(
#     next=next,
#     prompt=prompt_tokens,
#     index=1,
# )
# txt = tokenizer.detokenize(output_tokens)
# print(f"Beam search generated text: \n{txt}\n")

# # Randon search
# sampler = keras_nlp.samplers.RandonSampler()
# output_tokens = sampler(
#     next=next,
#     prompt=prompt_tokens,
#     index=1,
# )
# txt = tokenizer.detokenize(output_tokens)
# print(f"Randon search generated text: \n{txt}\n")

# # Top-K search
# sampler = keras_nlp.samplers.TopKsampler(k=10)
# output_tokens = sampler(
#     next=next,
#     prompt=prompt_tokens,
#     index=1,
# )
# txt = tokenizer.detokenize(output_tokens)
# print(f"Top-k search generated text: \n{txt}\n")

# # Top-P search
# sampler = keras_nlp.samplers.TopPSampler(p=0.5)
# output_tokens = sampler(
#     next=next,
#     prompt=prompt_tokens,
#     index=1,
# )
# txt = tokenizer.detokenize(output_tokens)
# print(f"Top-p search generated text: \n{txt}\n")

# # Using callbacks for text generation


class TopKTextGenerator(keras.callback.Callback):
    """A callback to generate text from a trained model using Top-k"""

    def __init__(self, k):
        self.sampler = keras_nlp.samplers.TopKSampler(k)

    def on_epoch_end(self, epoch, logs=None):
        output_tokens = self.sampler(
            next=next,
            prompt=prompt_tokens,
            index=1.
        )
        txt = tokenizer.detokenize(output_tokens)
        print(f"Top-K search generated text: \n{txt}\n")


text_generation_callback = TopKTextGenerator(k=10)
# Dummy training loop to demonstrate callback
model.fit(train_ds.take(1), verbose=2, epoch=2,
          callback=[text_generation_callback])
