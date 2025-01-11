import torch
import torch.nn as nn

class TransformerWithTimeInfo(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, embed_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()

        # 线性变换层
        self.encoder_embedding = torch.nn.Linear(in_features=in_dim, out_features=embed_dim)
        self.decoder_embedding = torch.nn.Linear(in_features=out_dim, out_features=embed_dim)
        self.output_layer = torch.nn.Linear(in_features=embed_dim, out_features=out_dim)

        # Transformer 模型
        self.transformer = torch.nn.Transformer(
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            d_model=embed_dim,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """前向传播函数"""
        # 输入嵌入
        src = self.encoder_embedding(src)

        # 目标嵌入
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt = self.decoder_embedding(tgt)

        # Transformer前向传播
        pred = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # 输出层
        pred = self.output_layer(pred)

        return pred
