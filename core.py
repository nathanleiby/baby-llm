import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout: float, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert (
            d_out % num_heads == 0
        ), f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # b is number inputs in a batch
        b, num_tokens, d_in = x.shape

        # after multiplying by weights, the shape of these is:
        # (b, num_tokens, d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # view() aka "reshape"
        # We can think of this as breaking up the single matrix into multiple, one per head (num_heads)
        # recall that `d_out = num_heads * head_dim`
        # after running view(), the new dimensions are: (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose results in shape
        # before: (b, num_tokens, num_heads, head_dim)
        #                 ->
        # after:  (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 1st, we called .T
        # 2nd, we called .transpose(1,2) to handle batches (idx=0)
        # now, we call   .transpose(2,3) to handle batches (idx=0) with multiple heads (idx=1)
        attn_scores = queries @ keys.transpose(2, 3)

        # apply mask for causal attention
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # normalize
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

        # dropout
        attn_weights = self.dropout(attn_weights)

        # compute context vector
        #
        # we transpose to convert from:
        # before: (b, num_heads, num_tokens, head_dim)
        #     ->
        # after: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # combine the heads
        # before: (b, num_heads, num_tokens, head_dim)
        #     ->
        # after: (b, num_tokens, d_out)
        #                        d_out = num_tokens * head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # combine the heads through a linear layer
        # this is considered optional... why?
        # TODO: appendix B for more details
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 1 is mult identity
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 0 is add identity

    def forward(self, x):
        # (1) normalize
        mean = x.mean(dim=-1, keepdim=True)
        # NOTE: unbiased=False mirrors GPT2 implementation,
        # which was in Tensorflow where that's the default
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # NOTE: We include eps as a small constant to prevent division by zero
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # (2) shift and scale
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


# TODO: Refactor so it takes actual args intead of cfg
# This would be consistent with our other modules
# and keeps untyped cfg dict at outermost interface, only
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layer_norm_1 = LayerNorm(cfg["emb_dim"])
        self.mha = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.dropout_1 = nn.Dropout(cfg["drop_rate"])

        self.layer_norm_2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.dropout_2 = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut1 = x
        x = self.layer_norm_1(x)
        x = self.mha(x)
        x = self.dropout_1(x)
        x = x + shortcut1

        shortcut2 = x
        x = self.layer_norm_1(x)
        x = self.ff(x)
        x = self.dropout_2(x)
        x = x + shortcut2

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        ## (1) Input -> Embeddings ##
        # token embedding
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # positional embedding
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        ## (2) Transformers ##
        # 12x transformer blocks
        self.tf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        ## (3) Embeddings -> Output ##
        # layer norm
        self.final_layer_norm = LayerNorm(cfg["emb_dim"])

        # linear output layer (convert back to vocab)
        self.linear_output = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=cfg["qkv_bias"]
        )

    def forward(self, x):
        batch_size, context_length = x.shape

        # `tok_emb` should be initialized randomly, so we just need to do the lookup
        tok_embedding = self.token_emb(x)
        # `pos_emb` should be initialized based on ordered position
        # TODO: what does it mean that we call this during the forward() step?
        #       intuitively, I think of pos_emb as initialized at start and then updated at each step
        pos_embedding = self.pos_emb(torch.arange(context_length, device=x.device))
        x = tok_embedding + pos_embedding
        x = self.drop_emb(x)

        x = self.tf_blocks(x)

        x = self.final_layer_norm(x)
        logits = self.linear_output(x)

        return logits


## Text Generation ##


def generate_text_simple(model, token_ids, max_new_tokens, context_size):
    for i in range(max_new_tokens):
        # all samples in batch .. most-recent `<= context_size` tokens
        x = token_ids[:, -context_size:]
        with torch.no_grad():  # during inference, we don't need backprop
            logits = model(x)

        token = next_token_id(logits)
        token_ids = torch.cat((token_ids, token), dim=1)

    return token_ids


def next_token_id(batch_logits):
    """Given batched model output,
    return next token's id"""
    # for each sample in batch, select logits for last token
    last_token_logits = batch_logits[:, -1, :]

    # softmax to get probabilities
    probas = torch.softmax(last_token_logits, dim=-1)

    # argmax to choose most likely probability
    token_id = torch.argmax(probas, dim=-1, keepdim=True)
    return token_id


## Data Loader ##
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
