# =========================
# 2.1 Dependencies
# =========================
import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
import mitdeeplearning as mdl
import numpy as np
import os
from scipy.io.wavfile import write
from tqdm import tqdm
from IPython import display as ipythondisplay

# Insert your Comet API key here
COMET_API_KEY = ""  # <-- ENTER YOUR COMET API KEY

# Ensure GPU is available
assert torch.cuda.is_available(), "Please enable GPU in runtime settings."
assert COMET_API_KEY != "", "Please insert your Comet API Key"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2.2 Dataset
# =========================
songs = mdl.lab1.load_training_data()
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
print(f"There are {len(vocab)} unique characters in the dataset")

# =========================
# 2.3 & 2.4 Vectorize dataset & define mappings
# =========================
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(string):
    return np.array([char2idx[c] for c in string], dtype=np.int32)

vectorized_songs = vectorize_string(songs_joined)
print(f"{repr(songs_joined[:10])} ---- characters mapped to int ----> {vectorized_songs[:10]}")

# =========================
# 2.3 Create training examples
# =========================
def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]

    x_batch = torch.tensor(input_batch, dtype=torch.long)
    y_batch = torch.tensor(output_batch, dtype=torch.long)
    return x_batch, y_batch

# =========================
# 2.4 Define LSTM model
# =========================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.fc(out)
        return out if not return_state else (out, state)

# =========================
# 2.5 Loss function
# =========================
cross_entropy = nn.CrossEntropyLoss()

def compute_loss(labels, logits):
    batched_labels = labels.view(-1)
    batched_logits = logits.view(-1, logits.size(-1))
    loss = cross_entropy(batched_logits, batched_labels)
    return loss

# =========================
# 2.6 Hyperparameters
# =========================
params = dict(
    num_training_iterations=3000,
    batch_size=8,
    seq_length=100,
    learning_rate=5e-3,
    embedding_dim=256,
    hidden_size=1024
)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

# =========================
# Comet experiment
# =========================
def create_experiment():
    if 'experiment' in locals():
        experiment.end()
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name="6S191_Lab1_Part2"
    )
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()
    return experiment

# =========================
# 2.5 Training step
# =========================
model = LSTMModel(len(vocab), params['embedding_dim'], params['hidden_size']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

def train_step(x, y):
    model.train()
    optimizer.zero_grad()
    y_hat = model(x)
    loss = compute_loss(y, y_hat)
    loss.backward()
    optimizer.step()
    return loss

# =========================
# 2.5 Training loop
# =========================
history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
experiment = create_experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear()
for iter in tqdm(range(params["num_training_iterations"])):
    x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    loss = train_step(x_batch, y_batch)

    experiment.log_metric("loss", loss.item(), step=iter)
    history.append(loss.item())
    plotter.plot(history)

    if iter % 100 == 0:
        torch.save(model.state_dict(), checkpoint_prefix)

torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()

# =========================
# 2.6 Generate text
# =========================
def generate_text(model, start_string="X", generation_length=1000):
    input_idx = torch.tensor([[char2idx[c] for c in start_string]], dtype=torch.long).to(device)
    state = model.init_hidden(input_idx.size(0), device)
    text_generated = []

    tqdm._instances.clear()
    for i in tqdm(range(generation_length)):
        predictions, state = model(input_idx, state, return_state=True)
        predictions = predictions[:, -1, :]  # last timestep
        input_idx = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1)
        text_generated.append(idx2char[input_idx.item()])

    return start_string + ''.join(text_generated)

generated_text = generate_text(model, start_string="X", generation_length=1000)

# =========================
# 2.6 Play back generated songs
# =========================
generated_songs = mdl.lab1.extract_song_snippet(generated_text)
for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)
    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)
        numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
        wav_file_path = f"output_{i}.wav"
        write(wav_file_path, 88200, numeric_data)
        experiment.log_asset(wav_file_path)

experiment.end()
