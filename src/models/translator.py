import torch
import torch.nn as nn

class InstructionFollowingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size):
        super(InstructionFollowingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + input_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, instruction, state):
        embedded_instr = self.embedding(instruction)
        lstm_out, _ = self.lstm(embedded_instr)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((lstm_out, state), dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return output
