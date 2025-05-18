import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Existing LSTM class
class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=6,
            hidden_size=90,
            num_layers=4,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )  
        self.softmax = torch.nn.Softmax(dim=1) 
        self.fc = torch.nn.Linear(90, 2)

        # Load the model with map_location set to the appropriate device
        model = torch.load('results/pretrained-models/zv_lstm_model.tar', map_location=device)
        my_dict = self.state_dict()
        for key, value in my_dict.items():
            my_dict[key] = model[key]
        self.load_state_dict(my_dict)
        self.eval()

    def forward(self, x, h=None, mode="train"):
        x = torch.FloatTensor(x).view((1, -1, 6))
        if h is None:
            h_n = x.data.new(4, x.size(0), 90).normal_(0, 0.1)
            h_c = x.data.new(4, x.size(0), 90).normal_(0, 0.1)
        else:
            h_n, h_c = h
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(x, (h_n, h_c))
        output = self.softmax(self.fc(r_out[0, :, :]))
        zv_lstm = torch.max(output.cpu().data, 1)[1].numpy()
        prob = torch.max(output.cpu().data, 1)[0].numpy()
        zv_lstm[np.where(prob <= 0.0)] = 0
        return zv_lstm


class BiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=5, output_size=2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Load the trained model
        model_path = 'results/pretrained-models/zv_bilstm_model.pth'
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.eval()

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.batch_norm(lstm_out[:, -1, :])
        out = self.fc(lstm_out)
        return out

    def compute_zv_lrt(self, data, batch_size=32, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        data = self.preprocess_data(data).to(device)
        predictions = self.predict_in_batches(data, batch_size, device)
        return np.array(predictions)  # Convert the list to a NumPy array

    def preprocess_data(self, data):
        # Reshape the data to match the expected input shape for the LSTM
        return torch.FloatTensor(data).view((data.shape[0], -1, 6))

    def predict_in_batches(self, data, batch_size, device):
        self.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, data.size(0), batch_size):
                batch_data = data[i:i+batch_size].to(device)
                outputs = self.forward(batch_data)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions

    # def preprocess_data(self, data):
    #     # Add any necessary preprocessing steps here
    #     # For example, normalization, reshaping, etc.
    #     return torch.FloatTensor(data).view((1, -1, 6))

    # def predict_in_batches(self, data, batch_size, device):
    #     self.eval()
    #     predictions = []
    #     with torch.no_grad():
    #         for i in range(0, len(data), batch_size):
    #             batch_data = data[i:i+batch_size].to(device)
    #             outputs = self.forward(batch_data)
    #             _, predicted = torch.max(outputs, 1)
    #             predictions.extend(predicted.cpu().numpy())
    #     return predictions

# Example usage
# model = BiLSTM()
# data = ...  # Load and preprocess your data
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_size = 32
# predicted_labels = model.compute_zv_lrt(data, batch_size, device)