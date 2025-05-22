
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


device = torch.device("cuda")
device

# In[5]:


rng = np.random.default_rng(73512)

# ## Dane syntetyczne

# In[6]:


indices = np.arange(0, 50, 0.02)
synthetic_data = (
    np.sin(indices * 3) + indices / 10 + (indices / 10) ** 2 + np.sin(indices * 10)
)  # / np.exp(indices / 20)

# In[ ]:


plt.figure(figsize=[20, 10])
plt.plot(synthetic_data)
plt.show()

# In[8]:


min_value = synthetic_data.min()
max_value = synthetic_data.max()

# In[9]:


data_seq = []
data_targets = []
sequence_len = 150
for i in range(len(synthetic_data) - sequence_len - 1):
    data_seq.append(torch.from_numpy(synthetic_data[i : i + sequence_len]))
    data_targets.append(synthetic_data[i + sequence_len + 1])

# In[10]:


data = (torch.stack(data_seq).float() - min_value) / max_value
data_targets = (torch.Tensor(data_targets).float() - min_value) / max_value
train_indices = rng.random(len(data_seq)) > 0.3
test_indices = ~train_indices
train_set = torch.utils.data.TensorDataset(
    data[train_indices], data_targets[train_indices]
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_data, test_targets = data[test_indices], data_targets[test_indices]

# In[ ]:


class SimpleRegressor(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


model = SimpleRegressor(sequence_len, 5, 1).to(device)
model

# In[12]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

# In[ ]:


model.train()

# Training loop
for epoch in range(101):
    for x, targets in train_loader:
        x = x.to(device)
        targets = targets.to(device)

        preds = model(x)
        preds = preds.squeeze(dim=1)

        optimizer.zero_grad()
        loss = loss_fun(preds, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[ ]:


with torch.no_grad():
    test_preds = model(test_data.to(device))
    print(torch.abs(test_preds.squeeze() - test_targets.to(device)).mean())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(test_targets.numpy(), label="original")
plt.plot(test_preds.cpu().numpy(), label="predictions")
plt.legend()
plt.show()

# ### Co jest nie tak?

# In[17]:


train_split = int(len(data) * 0.7)
train_set = torch.utils.data.TensorDataset(
    data[:train_split], data_targets[:train_split]
)
train_loader = DataLoader(train_set, batch_size=32, drop_last=True)
test_data, test_targets = data[train_split:], data_targets[train_split:]

# In[ ]:


model = SimpleRegressor(sequence_len, 5, 1).to(device)
model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

# Training loop
for epoch in range(101):
    for x, targets in train_loader:
        x = x.to(device)
        targets = targets.to(device)

        preds = model(x)
        preds = preds.squeeze(dim=1)

        optimizer.zero_grad()
        loss = loss_fun(preds, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[ ]:


with torch.no_grad():
    test_preds = model(test_data.to(device))
    print(torch.abs(test_preds.squeeze() - test_targets.to(device)).mean())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(test_targets.numpy(), label="original")
plt.plot(test_preds.cpu().numpy(), label="predictions_short_term")
plt.legend()
plt.show()


# In[ ]:


class RecurrentRegressor(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.act_fn = nn.Tanh()
        self.linear_xh = nn.Linear(num_inputs, num_hidden)
        self.linear_hh = nn.Linear(num_hidden, num_hidden)
        self.linear_hy = nn.Linear(num_hidden, num_outputs)

    def forward(self, x, state):
        h = self.act_fn(self.linear_hh(state) + self.linear_xh(x))
        y = self.linear_hy(h)
        return y, h


HIDDEN_SIZE = 5
model = RecurrentRegressor(1, HIDDEN_SIZE, 1).to(device)
model

# ### Dokończ pętlę uczącą poniżej:

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

for epoch in range(101):
    for x, targets in train_loader:
        x = x.to(device)
        targets = targets.to(device)
        loss_total = 0
        state = torch.zeros(len(x), HIDDEN_SIZE).to(device)
        optimizer.zero_grad()
        for i in range(x.size(1)):
            x_one = x[:, i].unsqueeze(1)
            if i < sequence_len - 1:
                target = x[:, i + 1]
            else:
                target = targets
            preds, state = model(x_one, state)
            preds = preds.squeeze(dim=1)
            loss = loss_fun(preds, target)
            loss_total += loss
        loss_total.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[ ]:


with torch.no_grad():
    state = torch.zeros(len(test_data), HIDDEN_SIZE).to(device)
    test_preds = []
    for i in range(test_data.size(1)):
        x_one = test_data[:, i].unsqueeze(1).to(device)
        preds, state = model(x_one, state)
    test_preds.append(preds)
    print(torch.abs((torch.cat(test_preds).squeeze() - test_targets.to(device))).mean())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(test_targets, label="original")
plt.plot(torch.cat(test_preds).squeeze().cpu().numpy(), label="predictions")
plt.legend()
plt.show()

# ## Sieci rekurencyjne w Torchu
#
# https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html

# In[ ]:


BATCH_SIZE = 2
INPUT_SIZE = 3
NUM_LAYERS = 1
rnn = nn.RNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    batch_first=False,
)  # batch_first=False is default!!
rnn

# In[30]:


with torch.no_grad():
    sequence_len = 5
    x = torch.randn(sequence_len, BATCH_SIZE, INPUT_SIZE)
    h0 = torch.randn(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
    output, hn = rnn(x, h0)

# In[ ]:


h0.shape

# In[ ]:


output.shape  # seq_len, batch_size, hidden_size
# output features (h_t) from the last layer of the RNN, for each t

# In[ ]:


hn.shape  # num_layers, batch_size, hidden_size
# the final hidden state for each element in the batch

# In[ ]:


class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden

    def forward(self, x, hidden):
        x = torch.transpose(x, 0, 1)
        all_outputs, hidden = self.rnn(x, hidden)
        out = all_outputs[-1]  # We are interested only in the last output
        x = self.fc(out)
        return x, hidden


model = RNNRegressor(1, HIDDEN_SIZE, NUM_LAYERS, 1).to(device)
model

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

for epoch in range(101):
    for x, targets in train_loader:
        x = x.to(device).unsqueeze(2)
        targets = targets.to(device)
        hidden = model.init_hidden(x.size(0)).to(device)
        preds, last_hidden = model(x, hidden)
        preds = preds.squeeze(1)
        optimizer.zero_grad()
        loss = loss_fun(preds, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[ ]:


with torch.no_grad():
    hidden = model.init_hidden(len(test_data)).to(device)
    test_preds, _ = model(test_data.to(device).unsqueeze(2), hidden)
    print(torch.abs(test_preds.squeeze() - test_targets.to(device)).mean())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(test_targets, label="original")
plt.plot(test_preds.squeeze().cpu().numpy(), label="predictions")
plt.legend()
plt.show()

# ## Long Shor Term Memory Networks (LSTM): Czy możemy jakoś rozdzielić krótką i długą pamięć?
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/

# ![LSTM](https://cdn-images-1.medium.com/max/1000/1*Ht2-sUJHi65wDwnR276k3A.png)

# In[41]:


sequence_length = 5
batch_size = 2
lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
lstm_input = torch.randn(sequence_length, batch_size, 1)
hidden_0 = torch.randn(NUM_LAYERS, batch_size, HIDDEN_SIZE)
cell_state_0 = torch.randn(NUM_LAYERS, batch_size, HIDDEN_SIZE)
output, (hn, cn) = lstm(lstm_input, (hidden_0, cell_state_0))

# In[ ]:


print(output.shape)  # seq_len, batch_size, hidden_size
# containing the output features (h_t) from the last layer of the LSTM, for each t

# In[ ]:


print(hn.shape)  # num_layers, batch_size, hidden_size
# containing the final hidden state for each element in the sequence

# In[ ]:


print(cn.shape)  # num_layers, batch_size, hidden_size
# containing the final cell state for each element in the sequence


# In[ ]:


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden):
        x = torch.transpose(x, 0, 1)
        all_outputs, hidden = self.lstm(x, hidden)
        out = all_outputs[-1]  # We are interested only in the last output
        x = self.fc(out)
        return x, hidden


model = LSTMRegressor(1, HIDDEN_SIZE, 2, 1).to(device)
model

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

for epoch in range(101):
    for x, targets in train_loader:
        x = x.to(device).unsqueeze(2)
        targets = targets.to(device)
        hidden, state = model.init_hidden(x.size(0))
        hidden, state = hidden.to(device), state.to(device)
        preds, last_hidden = model(x, (hidden, state))
        preds = preds.squeeze(1)
        optimizer.zero_grad()
        loss = loss_fun(preds, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[ ]:


with torch.no_grad():
    hidden, state = model.init_hidden(test_data.size(0))
    hidden, state = hidden.to(device), state.to(device)
    test_preds, _ = model(test_data.to(device).unsqueeze(2), (hidden, state))
    print(torch.abs(test_preds.squeeze() - test_targets.to(device)).mean().item())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(test_targets, label="original")
plt.plot(test_preds.squeeze().cpu().numpy(), label="predictions")
plt.legend()
plt.show()

# ### Mini zadanie: Jak wyglądałyby predykcje w oparciu o poprzednie predykcje?

# In[54]:


with torch.no_grad():
    hidden, state = model.init_hidden(1)
    hidden, state = hidden.to(device), state.to(device)
    hidden = (hidden, state)
    x_one = test_targets[0:1].unsqueeze(0).to(device)
    preds = []
    for i in range(len(test_targets)):
        x_one, hidden = model(x_one.unsqueeze(0), hidden)
        preds.append(x_one.item())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(test_targets, label="original")
plt.plot(test_preds.squeeze().cpu().numpy(), label="predictions_short_term")
plt.plot(preds, label="predictions_long_term")
plt.legend()
plt.show()

# # Predykcja Sequence to sequence

# In[ ]:


!wget --no-check-certificate https://www.galera.ii.pw.edu.pl/~kdeja/data/all_stocks_5yr.csv

# In[56]:


stock_price = pd.read_csv("all_stocks_5yr.csv")

# In[57]:


mastercard_stock = stock_price[stock_price.Name == "MA"].open.values
visa_stock = stock_price[stock_price.Name == "V"].open.values

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(mastercard_stock, label="mastercard")
plt.plot(visa_stock, label="visa")
plt.legend()
plt.show()

# In[59]:


m_min_value = mastercard_stock.min()
m_max_value = mastercard_stock.max()
v_min_value = visa_stock.min()
v_max_value = visa_stock.max()

# In[60]:


data_seq = []
data_targets = []
sequence_len = 50
for i in range(len(mastercard_stock) - sequence_len):
    data_seq.append(torch.from_numpy(mastercard_stock[i : i + sequence_len]))
    data_targets.append(torch.from_numpy(visa_stock[i : i + sequence_len]))

data = (torch.stack(data_seq).float() - m_min_value) / m_max_value
data_targets = (torch.stack(data_targets).float() - v_min_value) / v_max_value

train_split = int(len(data) * 0.7)
train_set = torch.utils.data.TensorDataset(
    data[:train_split], data_targets[:train_split]
)
train_loader = DataLoader(train_set, batch_size=32, drop_last=True)
train_set = torch.utils.data.TensorDataset(
    data[:train_split], data_targets[:train_split]
)
train_loader = DataLoader(train_set, batch_size=32, drop_last=True)
test_data, test_targets = data[train_split:], data_targets[train_split:]

# In[ ]:


class LSTM_Seq_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.proj_size = out_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            proj_size=out_size,
        )

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.proj_size)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden):
        x = torch.transpose(x, 0, 1)
        all_outputs, hidden = self.lstm(x, hidden)
        all_outputs = torch.transpose(all_outputs, 0, 1)
        return all_outputs, hidden


model = LSTM_Seq_Regressor(1, 50, 2, 1).to(device)
model

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

# Training loop
for epoch in range(101):
    for x, targets in train_loader:
        x = x.to(device).unsqueeze(2)
        targets = targets.to(device)
        hidden, state = model.init_hidden(x.size(0))
        hidden, state = hidden.to(device), state.to(device)
        preds, last_hidden = model(x, (hidden, state))
        preds = preds.squeeze(2)
        optimizer.zero_grad()
        loss = loss_fun(preds, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[69]:


with torch.no_grad():
    selected_test_targets = []
    preds = []
    for i in range(0, len(test_targets), sequence_len):
        hidden, state = model.init_hidden(1)
        hidden, state = hidden.to(device), state.to(device)
        selected_test_targets.append(test_targets[i])
        pred, _ = model(
            test_data[i].to(device).unsqueeze(0).unsqueeze(2), (hidden, state)
        )
        preds.append(pred.squeeze())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(torch.cat(selected_test_targets).cpu().numpy(), label="original")
plt.plot(torch.cat(preds).cpu().numpy(), label="predicted")
plt.legend()
plt.show()

# ### Co się stało, skąd takie dziwne predykcje?

# In[71]:


with torch.no_grad():
    selected_test_targets = []
    preds = []
    hidden, state = model.init_hidden(1)
    hidden, state = hidden.to(device), state.to(device)
    for i in range(0, len(test_targets), sequence_len):
        selected_test_targets.append(test_targets[i])
        pred, hidden_out = model(
            test_data[i].to(device).unsqueeze(0).unsqueeze(2), (hidden, state)
        )
        hidden, state = hidden_out
        preds.append(pred.squeeze())

# In[ ]:


plt.figure(figsize=[10, 5])
plt.plot(torch.cat(selected_test_targets).cpu().numpy(), label="original")
plt.plot(torch.cat(preds).cpu().numpy(), label="predicted")
plt.legend()
plt.show()

# # Klasyfikacja serii

# In[73]:


libras = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data",
    header=None,
)

# In[ ]:


libras.head()

# In[75]:


classes = libras[90].values
data = libras.values[:, :-1]

# In[ ]:


np.unique(classes).shape

# In[77]:


data = torch.from_numpy(data).float()
data_targets = torch.from_numpy(classes).long()

train_indices = rng.random(len(data)) > 0.3
test_indices = ~train_indices
train_set = torch.utils.data.TensorDataset(
    data[train_indices], data_targets[train_indices]
)
train_loader = DataLoader(train_set, batch_size=32)
test_data, test_targets = data[test_indices], data_targets[test_indices]

# ### Napisz klasyfikator

# In[ ]:




# In[ ]:


class LSTMRegressor(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, out_size, bidirectional=False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if bidirectional:
            self.bidirectional = 2
        else:
            self.bidirectional = 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * 90 * self.bidirectional, out_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(
            self.num_layers * self.bidirectional, batch_size, self.hidden_size
        )
        state = torch.zeros(
            self.num_layers * self.bidirectional, batch_size, self.hidden_size
        )
        return hidden, state

    def forward(self, x, hidden):
        x = torch.transpose(x, 0, 1)
        all_outputs, hidden = self.lstm(x, hidden)
        all_outputs = torch.transpose(all_outputs, 0, 1)
        out = torch.flatten(all_outputs, 1)
        x = self.fc(out)
        return x, hidden


model = LSTMRegressor(1, 5, 2, 16, bidirectional=True).to(device)
model

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.CrossEntropyLoss()

# Training loop
for epoch in range(101):
    for x, targets in train_loader:
        x = x.to(device).unsqueeze(2)
        #         x = x.unsqueeze(2)
        targets = targets.to(device)
        hidden, state = model.init_hidden(x.size(0))
        hidden, state = hidden.to(device), state.to(device)
        preds, _ = model(x, (hidden, state))
        preds = preds.squeeze(1)
        optimizer.zero_grad()
        loss = loss_fun(preds, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[ ]:


with torch.no_grad():
    hidden, state = model.init_hidden(len(test_data))
    hidden, state = hidden.to(device), state.to(device)
    preds, _ = model(test_data.to(device).unsqueeze(2), (hidden, state))
print(
    f"Accuracy: {(torch.argmax(preds, 1).cpu() == test_targets).sum().item() / len(test_targets):.3}"
)

# # Dane o różnej długości

# In[88]:


from torch.utils.data import Dataset


class VariableLenDataset(Dataset):
    def __init__(self, in_data, target):
        self.data = [(x, y) for x, y in zip(in_data, target)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data, target = self.data[idx]
        return in_data, target

# In[89]:


min_gen_val = 10
max_gen_val = 1001
samples = 1000
max_gen_len = 32

data = []
targets = []
max_val = -1
for _ in range(samples):
    seq_len = rng.integers(low=1, high=max_gen_len, size=1)
    data_in = rng.integers(low=min_gen_val, high=max_gen_val, size=seq_len)
    data_sum = np.array([data_in[: i + 1].sum() for i in range(len(data_in))])
    data.append(torch.from_numpy(data_in))
    targets.append(torch.from_numpy(data_sum))
    max_val = data_sum[-1] if data_sum[-1] > max_val else max_val

# In[ ]:


[len(x) for x in data[:10]]

# In[91]:


train_indices = int(len(data) * 0.7)
data = [(x / max_val).float() for x in data]
targets = [(x / max_val).float() for x in targets]
train_set = VariableLenDataset(data[:train_indices], targets[:train_indices])
test_set = VariableLenDataset(data[train_indices:], targets[train_indices:])

# In[92]:


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

pad = 0


def pad_collate(batch, pad_value=0):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=pad_value)

    return xx_pad, yy_pad, x_lens, y_lens

# In[93]:


train_loader = DataLoader(
    train_set, batch_size=50, shuffle=True, collate_fn=pad_collate
)
test_loader = DataLoader(
    test_set, batch_size=50, shuffle=False, drop_last=False, collate_fn=pad_collate
)

# In[ ]:


next(iter(train_loader))

# In[ ]:


class LSTM_Seq_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.proj_size = out_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            proj_size=out_size,
        )

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.proj_size)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden):
        # x = torch.transpose(x, 0, 1)
        all_outputs, hidden = self.lstm(x, hidden)
        # all_outputs = torch.transpose(all_outputs, 0, 1)
        return all_outputs, hidden


model = LSTM_Seq_Regressor(1, 200, 1, 1).to(device)
model

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.MSELoss()

# Training loop
for epoch in range(101):
    for x, targets, x_len, target_len in train_loader:
        x = x.to(device).unsqueeze(2)
        targets = targets.to(device)
        hidden, state = model.init_hidden(x.size(0))
        hidden, state = hidden.to(device), state.to(device)

        x = torch.transpose(x, 0, 1)
        preds, _ = model(x, (hidden, state))
        preds = torch.transpose(preds, 0, 1)

        #         x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        #         preds_packed, _ = model(x_packed, (hidden, state))
        #         preds, pred_len = pad_packed_sequence(preds_packed, batch_first=True, padding_value=pad)

        preds = preds.squeeze(2)
        optimizer.zero_grad()
        mask = targets != pad
        loss = loss_fun(preds[mask], targets[mask])
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

# In[ ]:


with torch.no_grad():
    for x, targets, x_len, target_len in test_loader:
        x = x.to(device).unsqueeze(2)
        targets = targets.to(device)
        hidden, state = model.init_hidden(x.shape[0])
        hidden, state = hidden.to(device), state.to(device)

        #         x = torch.transpose(x, 0, 1)
        #         preds, _ = model(x, (hidden, state))
        #         preds = torch.transpose(preds, 0, 1)
        x_packed = pack_padded_sequence(
            x, x_len, batch_first=True, enforce_sorted=False
        )
        preds_packed, _ = model(x_packed, (hidden, state))
        preds, pred_len = pad_packed_sequence(
            preds_packed, batch_first=True, padding_value=pad
        )

        preds = preds.squeeze(2)
        mask_tgt = targets != pad

# Jeśli chcemy wykorzystać ostatnie stany ukryte komórki, np. do klasyfikacji, naley wykorzystać `pack_padded_sequence`.
# Dane nadal pozostaną w odpowiednim formacie, ale RNN nie będzie uwzględniał paddingu (jeśli tego nie zrobimy, RNN zwróci stany ukryty odpowiadający paddingowi, co jest niepozadane).
