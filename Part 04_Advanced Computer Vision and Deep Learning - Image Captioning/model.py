import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        dropout = 0.3
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=dropout, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Check for a GPU
        self.train_on_gpu = torch.cuda.is_available()
        if not self.train_on_gpu:
            print('No GPU found. Please use a GPU to train your neural network.')
    
    def forward(self, features, captions):
        # embeddings and lstm_out
        embeds = self.embedding(captions[:,:-1])
        
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)
        lstm_out, _ = self.lstm(inputs)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # return last sigmoid output and hidden state
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.fc(self.dropout(lstm_out))
            outputs = outputs.squeeze(1)
            wordid  = outputs.argmax(dim=1)
            captions.append(wordid.item())
            
            # prepare input for next iteration
            inputs = self.embedding(wordid.unsqueeze(0))
        return captions