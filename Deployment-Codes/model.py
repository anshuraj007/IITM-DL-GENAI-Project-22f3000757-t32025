# import torch
# import torch.nn as nn
# from transformers import DebertaV2Model

# class EmotionClassifier(nn.Module):
#     def __init__(self, model_name="microsoft/deberta-v3-large", num_labels=5):
#         super().__init__()
#         self.deberta = DebertaV2Model.from_pretrained(model_name)
#         hidden_size = self.deberta.config.hidden_size

#         self.dropout = nn.Dropout(0.3)
#         self.norm = nn.LayerNorm(hidden_size)
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.deberta(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )

#         cls_hidden_state = outputs.last_hidden_state[:, 0, :]
#         x = self.norm(cls_hidden_state)
#         x = self.dropout(x)
#         logits = self.classifier(x)
#         return logits


import os
import torch
import torch.nn as nn
from transformers import DebertaV2Model

# -------------------- DISABLE ALL HF DOWNLOADS + LOGS --------------------
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import transformers
transformers.utils.logging.set_verbosity_error()
# -------------------------------------------------------------------------


class EmotionClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-large", num_labels=5):
        super().__init__()

        # -------------------- LOAD MODEL *ONLY* FROM LOCAL CACHE --------------------
        self.deberta = DebertaV2Model.from_pretrained(
            model_name,
            local_files_only=True    # ← prevents downloading
        )
        # -----------------------------------------------------------------------------

        hidden_size = self.deberta.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        x = self.norm(cls_hidden_state)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
