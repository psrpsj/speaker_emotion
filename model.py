import torch
import torch.nn as nn

from transformers import (
    BertModel,
    BertForSequenceClassification,
)

from transformers.modeling_outputs import SequenceClassifierOutput


class BERTwithLSTM(BertForSequenceClassification):
    def __init__(self, model, config, *args, **kwargs):
        super(BERTwithLSTM, self).__init__(config=config)

        self.bert = BertModel.from_pretrained(model)
        self.lstm = nn.LSTM(
            config.hidden_size, 256, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(256 * 2, config.num_labels)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        lstm_output, (h, c) = self.lstm(output[0])
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        linear_output = self.linear(hidden.view(-1, 256 * 2))
        x = self.tanh(linear_output)
        x = self.dropout(x)
        outputs = SequenceClassifierOutput(logits=x)
        return outputs
