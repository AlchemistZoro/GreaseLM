from transformers.modeling_roberta import RobertaModel,RobertaEmbeddings
import torch
from torch import nn

class RobertaPoolModel(RobertaModel):
    def __init__(self):
        super().__init__()
        self.embeddings = RobertaEmbeddings()

class RobertaPoolEmbeddings(RobertaEmbeddings):

    def __init__(self, config):
        super().__init__(config = config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, pool_mask=None,past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # print(inputs_embeds.shape)
        # print(input_ids.shape)
        # print(pool_mask.shape)
        inputs_embeds = self.pool_input_embeds(inputs_embeds,pool_mask)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def pool_input_embeds(self,inputs_embeds,pool_mask):
        # print(pool_mask)
        # print(inputs_embeds)
        for bs in range(pool_mask.shape[0]):
            row_pool = pool_mask[bs]
            start,end = 0,0
            for i in range(pool_mask.shape[1]-1):
                if row_pool[i]==0 and row_pool[i+1]==1:
                    start = i+1
                elif row_pool[i]==1 and i==0:
                    start = i
                elif row_pool[i+1] ==1 and i == pool_mask.shape[1]-2:
                    end = i+1
                elif row_pool[i]==1 and row_pool[i+1]==0:
                    end = i
                
                if end != 0:
                    inputs_embeds[bs][start:end+1] = torch.mean(inputs_embeds[bs][start:end+1],dim=0,keepdim=True)
                    # print(start,end+1)
                    start,end = 0,0 
                    
        # print(inputs_embeds)
        return inputs_embeds
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
 
def pool_input_embeds(inputs_embeds,pool_mask):
    # print(pool_mask)
    # print(inputs_embeds)
    for bs in range(pool_mask.shape[0]):
        row_pool = pool_mask[bs]
        start,end = 0,0
        for i in range(pool_mask.shape[1]-1):
            if row_pool[i]==0 and row_pool[i+1]==1:
                start = i+1
            elif row_pool[i]==1 and i==0:
                start = i
            elif row_pool[i+1] ==1 and i == pool_mask.shape[1]-2:
                end = i+1
            elif row_pool[i]==1 and row_pool[i+1]==0:
                end = i
            
            if end != 0:
                inputs_embeds[bs][start:end+1] = torch.mean(inputs_embeds[bs][start:end+1],dim=0,keepdim=True)
                # print(start,end+1)
                start,end = 0,0 
                
    # print(inputs_embeds)
    return inputs_embeds  



if __name__ == "__main__":
    bs = 2
    seq_len = 5
    inputs_embeds = torch.randint(1,10,[bs, seq_len,3], dtype=torch.float)

    # pool_mask 
    pool_mask = torch.zeros([bs, seq_len], dtype=torch.long)
    pool_mask[0][0:2] = 1
    pool_mask[0][3:5] = 1
    pool_mask[-1][2:5] = 1
    
    pool_input_embeds(inputs_embeds,pool_mask)