import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging

# 设置日志记录
logger = logging.getLogger(__name__)

from .base import InteractionLayer


class FSAttention(InteractionLayer):
    def __init__(self, input_dim=128, emb_size=128, dropout_rate=0.1,
                 n_layer=2, intermediate_size=512, num_attention_heads=4,
                 attention_probs_dropout=0.1, hidden_dropout_rate=0.1, device="cpu", emb_max_pos_size=545):
        super(FSAttention, self).__init__()
        # 记录初始参数
        logger.info(f"Initializing FSAttention with emb_max_pos_size={emb_max_pos_size}")

        # 修改：使用emb_max_pos_size初始化位置嵌入表
        self.emb = Embeddings(input_dim, emb_size, emb_max_pos_size, dropout_rate)
        self.encoder = Encoder_MultipleLayers(n_layer,
                                              emb_size,
                                              intermediate_size,
                                              num_attention_heads,
                                              attention_probs_dropout,
                                              hidden_dropout_rate)
        self.output_shape = emb_size
        self.norm = nn.LayerNorm(emb_size)
        self.device = device
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(True),

            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate))

        self.output_shape = emb_size
        self.device = device
        # 添加：最大碎片数和口袋数
        self.max_fragments = 11
        self.max_pockets = 30
        self.emb_size = emb_size
        self.emb_max_pos_size = emb_max_pos_size  # 保存最大位置大小

    def get_output_shape(self):
        return 1

    def forward(self, *inputs):
        # 1. 解包输入
        pocket, pos_index = inputs[0]  # [B, max_pockets, emb]
        fragments, fragments_index = inputs[1]  # [B, max_fragments, emb]

        # 2. 拼接特征
        stacked_input = torch.cat([fragments, pocket], dim=1)  # [B, max_frag+max_pock, emb]
        full_pos_index = torch.cat([fragments_index, pos_index], dim=1)  # [B, max_frag+max_pock]

        # 3. 创建注意力掩码
        attention_mask = (full_pos_index != -1).float()
        ex_e_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0

        # 4. 处理输入序列
        stacked_input = self.norm(stacked_input)
        full_pos_index = full_pos_index.clamp(0, self.emb_max_pos_size - 1)
        emb = self.emb(stacked_input, full_pos_index)

        # 5. 通过Transformer编码器
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())

        # 6. 拆分特征并聚合
        valid_mask = (full_pos_index != -1).float()
        fragments_encoded = encoded_layers[:, :self.max_fragments, :]
        pockets_encoded = encoded_layers[:, self.max_fragments:, :]

        # 碎片特征聚合（掩码平均）
        frag_mask = valid_mask[:, :self.max_fragments].unsqueeze(-1)
        frag_valid_num = torch.sum(frag_mask, dim=1).clamp(min=1e-6)
        frag_mean = torch.sum(fragments_encoded * frag_mask, dim=1) / frag_valid_num

        # 口袋特征聚合（掩码平均）
        pock_mask = valid_mask[:, self.max_fragments:].unsqueeze(-1)
        pock_valid_num = torch.sum(pock_mask, dim=1).clamp(min=1e-6)
        pock_mean = torch.sum(pockets_encoded * pock_mask, dim=1) / pock_valid_num

        # 7. 拼接特征并预测
        combined = torch.cat([frag_mean, pock_mean], dim=1)
        output = self.decoder(combined)  # [B, 1]
        return output


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon
        self.hidden_size = hidden_size

    def forward(self, x):
        # 添加输入检查
        if x.numel() == 0:
            return x

        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN or Inf found in LayerNorm input!")
            # 用0替换无效值
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        # 检查输入维度
        if x.size(-1) != self.hidden_size:
            logger.warning(f"Input size mismatch: expected hidden_size={self.hidden_size}, got {x.size(-1)}")

        # 添加安全计算
        try:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.gamma * x + self.beta
        except Exception as e:
            logger.error(f"Error in LayerNorm: {e}")
            logger.error(f"Input shape: {x.shape}, dtype: {x.dtype}")
            logger.error(f"Input min: {x.min().item()}, max: {x.max().item()}")
            logger.error(f"Input contains NaN: {torch.isnan(x).any().item()}, Inf: {torch.isinf(x).any().item()}")
            # 返回安全的输出
            return torch.zeros_like(x)


class Embeddings(nn.Module):
    """Construct the embeddings from pre-embedded protein/target and position embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        # 修改：使用max_position_size初始化位置嵌入表
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_position_size = max_position_size
        logger.info(f"Initialized Embeddings with max_position_size={max_position_size}")

    def forward(self, embedded_input_ids, position_ids):
        """
        embedded_input_ids: Tensor of shape [batch, seq_len, hidden_size] (already embedded)
        position_ids: Tensor of shape [batch, seq_len]
        """
        # 确保position_ids是long类型
        position_ids = position_ids.long()

        # 检查位置索引范围
        if (position_ids < 0).any() or (position_ids >= self.max_position_size).any():
            logger.warning(f"Invalid position_ids: min={position_ids.min().item()}, max={position_ids.max().item()}")
            # 将无效位置索引设为0
            position_ids = torch.clamp(position_ids, 0, self.max_position_size - 1)

        # 生成位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        # 检查输入形状
        if embedded_input_ids.shape != position_embeddings.shape:
            logger.error(
                f"Shape mismatch: embedded_input_ids {embedded_input_ids.shape}, position_embeddings {position_embeddings.shape}")
            # 尝试调整大小
            if embedded_input_ids.size(0) == position_embeddings.size(0) and embedded_input_ids.size(
                    1) == position_embeddings.size(1):
                position_embeddings = position_embeddings[:, :, :embedded_input_ids.size(2)]
            else:
                logger.error("Cannot resolve shape mismatch, using zero embeddings")
                position_embeddings = torch.zeros_like(embedded_input_ids)

        # 修改：移除对embedded_input_ids的long转换
        embeddings = embedded_input_ids + position_embeddings

        # 检查加法结果
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            logger.error("NaN or Inf in embeddings after adding position embeddings")
            # 用0替换无效值
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e4, neginf=-1e4)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # 检查输入
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            logger.error("NaN or Inf in SelfAttention input")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # 添加输入检查
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            logger.error("NaN or Inf in encoder input")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

            # 检查输出
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                logger.error("NaN or Inf after encoder layer")
                hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)

        return hidden_states

