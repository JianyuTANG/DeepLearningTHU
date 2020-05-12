import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, ninput):
        super().__init__()
        self.context_encoder = nn.RNN(
            input_size=ninput,
            hidden_size=ninput // 2,
            num_layers=1,
            bidirectional=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(ninput * 2, ninput),
            nn.Tanh(),
            nn.Linear(ninput, 1),
            nn.Tanh(),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, s, x):
        h, _ = self.context_encoder(x)
        e = []
        for i in range(h.size(0)):
            e.append(self.linear(torch.cat((s, h[i]), 1)))
        e = torch.cat(e, 1)
        e = self.softmax(e).t().contiguous().view(h.size(0), h.size(1), 1)
        return (e * h).sum(0)


class RNNwithAttention(nn.Module):
    def __init__(self, ninput, nhid, nlayers, dropout=0.5, device="cpu"):
        super().__init__()
        self.attention_model = TemporalAttention(ninput)
        self.layers = [nn.LSTMCell(ninput, nhid).to(device)] + [nn.LSTMCell(nhid, nhid).to(device) for i in range(nlayers - 1)]
        self.dropouts = [nn.Dropout(dropout).to(device) for i in range(nlayers - 1)]
        self.nlayers = nlayers

    def forward(self, x):
        hid = [None for i in range(self.nlayers)]
        res = []
        for i in range(x.size(0)):
            input = x[i]
            if i != 0:
                hid[0] = (hid[0][0] + self.attention_model(hid[0][0], x), hid[0][1])
            for j in range(self.nlayers):
                hid[j] = self.layers[j](input, hid[j])
                input = hid[j][0]
                if j != self.nlayers - 1:
                    input = self.dropouts[j](hid[j][0])
            res.append(input)
        return torch.stack(res, 0)


class LSTMwithAttention(nn.Module):
    def __init__(self, ninput, nhid, nlayers, dropout=0.5, device="cpu"):
        super().__init__()
        self.attention_model = TemporalAttention(ninput)
        # self.contex_encoder = nn.RNN(
        #     input_size=ninput,
        #     hidden_size=ninput // 2,
        #     num_layers=1,
        #     bidirectional=True,
        # )
        self.lstm = nn.LSTM(
            input_size=ninput,
            hidden_size=nhid,
            num_layers=nlayers,
            bidirectional=False,
            dropout=0.5,
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        # h, _ = self.contex_encoder(x)
        res = []
        for i in range(output.size(0)):
            res.append(self.attention_model(output[i], x[ : i + 1]) + output[i])
        return torch.stack(res, 0)


class LMModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, ninput, nhid, nlayers, device, attention=True):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, ninput)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you RNN model here. You can add additional parameters to the function.

        if attention:
            self.rnn = LSTMwithAttention(
                ninput=ninput,
                nhid=nhid,
                nlayers=nlayers,
                dropout=0.5,
                device=device,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=ninput,
                hidden_size=nhid,
                num_layers=nlayers,
                bidirectional=False,
                dropout=0.5,
            )

        self.attention = attention

        ########################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        embeddings = self.drop(self.encoder(input))

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes

        if self.attention:
            output = self.rnn(embeddings)
        else:
            output, hidden = self.rnn(embeddings)

        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        # return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        return decoded.view(-1, decoded.size(1)), 0



if __name__ == "__main__":
    # test attention
    model = TemporalAttention(10)
    s = torch.rand(20, 10)
    x = torch.rand(30, 20, 10)
    att = model(s, x)
    print(att.shape)

    # test rnn_with_attention
    x = torch.rand(30, 20, 150)
    rnn = RNNwithAttention(150, 150, 4, 0.5)
    output = rnn(x)
    print(output.shape)

    # test lstm_with_attention
    for i in range(100):
        x = torch.rand(30, 20, 150)
        rnn = LSTMwithAttention(150, 150, 4, 0.5)
        output = rnn(x)
        print(output.shape)
