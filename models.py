import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class LSTM_FFN_branch(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_n_layers, bidirectional, 
                 ffn_dims):

        super().__init__()

        self.lstm_ffn_dim = (lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim) + ffn_dims[-1]
        self.ffn_layers = nn.ModuleList()

        # 1. lstm layer
        self.lstm = nn.LSTM(lstm_input_dim, 
                            lstm_hidden_dim, 
                            num_layers=lstm_n_layers, 
                            bidirectional=bidirectional, 
                            batch_first=True) # Input/output tensors are (batch, seq, feature)
        # BN layer for stabalization
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim)
        
        # 2. ffn layer
        for i in range(len(ffn_dims) -1):
            ffn_input_dim = ffn_dims[i]
            ffn_hidden_dim = ffn_dims[i+1]
            ffn_block = nn.Sequential(
                nn.Linear(ffn_input_dim, ffn_hidden_dim),
                nn.BatchNorm1d(ffn_hidden_dim),    # BN layer for stabalization
                nn.ReLU())
            self.ffn_layers.append(ffn_block)
        
        
    def forward(self, pitch_hnrs, pitchhnr_lengths, global_features):
      
        # 1. Pack sequence
        ### Compute actual data and ignore the padded values
        packed_input = rnn_utils.pack_padded_sequence(pitch_hnrs, pitchhnr_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # 2. Pass packed sequence through LSTM
        ### packed_output: Hidden states for every time step.
        ### hidden: The final hidden state (summary) of the entire sequence.
        ### cell: The final cell state (long-term memory) of the entire sequence.
        packed_output, (lstm_hidden, cell) = self.lstm(packed_input)
        
        # 3. Concatenate the final forward and backward hidden states (if bidirectional)
        if self.lstm.bidirectional:
            lstm_hidden = torch.cat((lstm_hidden[-2,:,:], lstm_hidden[-1,:,:]), dim=1)
        else:
            lstm_hidden = lstm_hidden[-1,:,:]
        lstm_hidden = self.bn_lstm(lstm_hidden)

        # 4. Pass global features (jitter and shimmer) through the FFN
        for layer in self.ffn_layers:
            global_features = layer(global_features)
        ffn_output = global_features

        # 5. Concatenate the outputs from lstm and fnn
        combined_output = torch.cat((lstm_hidden,ffn_output), dim=1)

        return combined_output
    

class CNN_branch(nn.Module):
    def __init__(self, cnn_channels, conv_kernel, pool_kernel, cnn_padding):

        super().__init__()

        self.cnn_dim = cnn_channels[-1]

        self.conv_layers = nn.ModuleList()

        for i in range(len(cnn_channels)-2):
            cnn_in = cnn_channels[i]
            cnn_out = cnn_channels[i+1]
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=cnn_in, out_channels=cnn_out, kernel_size=conv_kernel, padding=cnn_padding),
                nn.BatchNorm2d(cnn_out),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel))
            self.conv_layers.append(conv_block)

        # final layer of CNN
        final_in = cnn_channels[-2]
        final_out = cnn_channels[-1]

        conv_final = nn.Sequential(
            nn.Conv2d(in_channels=final_in, out_channels=final_out, kernel_size=conv_kernel, padding=cnn_padding),
            nn.BatchNorm2d(final_out),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))  # Output size: [batch, 64, 1, 1]
        )
        self.conv_layers.append(conv_final)
        
    def forward(self, mfccs):

        # expected shape (batch_size, in_channel, height, width) -> unsqeeze
        mfccs = mfccs.unsqueeze(1)

        for layer in self.conv_layers:
            mfccs = layer(mfccs)
        cnn_out = mfccs.view(mfccs.size(0), -1)
        
        return cnn_out
    



class BranchAttention(nn.Module):
    def __init__(self, branch1_dim, branch2_dim):
        super().__init__()
        self.attention_net = nn.Linear(branch1_dim + branch2_dim, 2)

        with torch.no_grad():
            self.attention_net.bias.fill_(0)

    def forward(self, branch1_out, branch2_out):
        # 1. concatenate the raw outputs from both branches
        combined_out = torch.cat((branch1_out, branch2_out), dim=1)
        
        # 2. Predict the score for each branch's importance
        attention_scores = self.attention_net(combined_out)
        
        # 3. turn scores into weights that sum to 1 (e.g., [0.7, 0.3])
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 4. Get the individual weight for each branch
        # .unsqueeze(1) is needed to make the dimensions compatible for multiplication
        branch1_weight = attention_weights[:, 0].unsqueeze(1)
        branch2_weight = attention_weights[:, 1].unsqueeze(1)
        
        # 5. Scale each branch's output by its learned weight
        branch1_weighted = branch1_out * branch1_weight
        branch2_weighted = branch2_out * branch2_weight
        
        # 6. Concatenate the *weighted* features to pass to the final classifier
        weighted_combined_features = torch.cat((branch1_weighted, branch2_weighted), dim=1)
        
        # Return the combined features and the weights for inspection
        return weighted_combined_features, attention_weights
    
    

class SpoofEnsemble(nn.Module):
    def __init__(self, lstm_ffn_branch, cnn_branch, output_dim, dropout):

        super().__init__()

        self.lstm_ffn_branch = lstm_ffn_branch
        self.cnn_branch = cnn_branch

        lstm_ffn_dim = lstm_ffn_branch.lstm_ffn_dim
        cnn_dim = cnn_branch.cnn_dim
        self.fc = nn.Linear(lstm_ffn_dim + cnn_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pitch_hnrs, pitchhnr_lengths, global_features, mfccs):
      
        lstm_ffn_out = self.lstm_ffn_branch(pitch_hnrs, pitchhnr_lengths, global_features)
        
        # Get the output from the second branch
        cnn_out = self.cnn_branch(mfccs)
        
        # Concatenate all features
        combined_features = torch.cat((lstm_ffn_out, cnn_out), dim=1)

        # Apply dropout
        combined_features = self.dropout(combined_features)
        
        # Final classification
        output = self.fc(combined_features)
        
        return output
    


# --- for LSTM_FFN training alone --- 
class LSTM_FFN_classifer(nn.Module):
    def __init__(self, lstm_ffn_out, output_dim, dropout):
        super().__init__()

        self.lstm_ffn_layer = lstm_ffn_out
        self.fc = nn.Linear(lstm_ffn_out.lstm_ffn_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, pitch_hnrs, pitchhnr_lengths, global_features, _a):

        lstm_ffn_out = self.lstm_ffn_layer(pitch_hnrs, pitchhnr_lengths, global_features)
        lstm_ffn_out = self.dropout(lstm_ffn_out)
        output = self.fc(lstm_ffn_out)

        return output


# for CNN training alone
class CNN_classifer(nn.Module):
    def __init__(self, cnn_out, output_dim, dropout):
        super().__init__()

        self.cnn_layer = cnn_out
        self.fc = nn.Linear(cnn_out.cnn_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, _a, _b, _c, mfccs):

        cnn_out = self.cnn_layer(mfccs)
        cnn_out = self.dropout(cnn_out)
        output = self.fc(cnn_out)

        return output
    

class SpoofEnsemble_attention(nn.Module):
    def __init__(self, lstm_ffn_branch, cnn_branch, output_dim, dropout):

        super().__init__()

        self.lstm_ffn_branch = lstm_ffn_branch
        self.cnn_branch = cnn_branch

        lstm_ffn_dim = lstm_ffn_branch.lstm_ffn_dim
        cnn_dim = cnn_branch.cnn_dim

        # Instantiate the attention module
        self.attention = BranchAttention(lstm_ffn_dim, cnn_dim)
        
        # This attribute will store the weights from the last forward pass
        # for later analysis and interpretation.
        self.attention_weights = None

        self.fc = nn.Linear(lstm_ffn_dim + cnn_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pitch_hnrs, pitchhnr_lengths, global_features, mfccs):
      
        lstm_ffn_out = self.lstm_ffn_branch(pitch_hnrs, pitchhnr_lengths, global_features)
        
        # Get the output from the second branch
        cnn_out = self.cnn_branch(mfccs)
        
        # Pass the raw outputs through the attention mechanism
        combined_features, self.attention_weights = self.attention(lstm_ffn_out, cnn_out)
        
        # Apply dropout
        combined_features = self.dropout(combined_features)
        
        # Final classification
        output = self.fc(combined_features)
        
        return output