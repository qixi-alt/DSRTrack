import torch
import torch.nn as nn
import sys
import os


class AnchorGuidedCrossModalRepair(nn.Module):
    """
    Anchor-Guided Cross-Modal Repair.


    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        hidden_ratio: float = 1.0,
        repair_margin: float = 3.0,
        use_anchor_as_kv: bool = True,
        aux_loss_weight: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.repair_margin = repair_margin
        self.use_anchor_as_kv = use_anchor_as_kv





        self.aux_loss_weight = aux_loss_weight

        hidden_dim = max(embed_dim // 2, int(embed_dim * hidden_ratio))

        # ------------------------------------------------------------
        # 1. Normalization
        # ------------------------------------------------------------
        self.norm_q_v = nn.LayerNorm(embed_dim)
        self.norm_q_i = nn.LayerNorm(embed_dim)

        self.norm_kv_v = nn.LayerNorm(embed_dim)
        self.norm_kv_i = nn.LayerNorm(embed_dim)

        # ------------------------------------------------------------
        # 2. Bidirectional cross-modal reconstruction
        # ------------------------------------------------------------
        # visible <- infrared / anchor infrared
        self.attn_v_from_i = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )

        # infrared <- visible / anchor visible
        self.attn_i_from_v = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )

        # ------------------------------------------------------------
        # 3. Health estimation
        # ------------------------------------------------------------




        quality_in_dim = embed_dim * 4

        self.norm_quality_v = nn.LayerNorm(quality_in_dim)
        self.norm_quality_i = nn.LayerNorm(quality_in_dim)

        self.health_scorer_v = nn.Sequential(
            nn.Linear(quality_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.health_scorer_i = nn.Sequential(
            nn.Linear(quality_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ------------------------------------------------------------
        # 4. Conservative initialization
        # ------------------------------------------------------------




        nn.init.constant_(self.health_scorer_v[-1].weight, 0.0)
        nn.init.constant_(self.health_scorer_v[-1].bias, 1.5)

        nn.init.constant_(self.health_scorer_i[-1].weight, 0.0)
        nn.init.constant_(self.health_scorer_i[-1].bias, 1.5)

    def forward(
        self,
        d_v: torch.Tensor,
        d_i: torch.Tensor,
        z_v: torch.Tensor,
        z_i: torch.Tensor,
        foreground_mask: torch.Tensor = None,
        contam_mask_v: torch.Tensor = None,
        contam_mask_i: torch.Tensor = None,
    ):
        """
        Args:
            d_v: [B, N, C], visible online-template tokens.
            d_i: [B, N, C], infrared online-template tokens.
            z_v: [B, N, C], visible initial-template tokens.
            z_i: [B, N, C], infrared initial-template tokens.
            foreground_mask: [B, N] or [B, H, W], 1 means foreground token.
            contam_mask_v: [B, N] or None, only used for statistics.
            contam_mask_i: [B, N] or None, only used for statistics.

        Returns:
            d_fixed_v: [B, N, C]
            d_fixed_i: [B, N, C]
            stats: dict
        """

        B, N, C = d_v.shape
        device = d_v.device
        dtype = d_v.dtype

        # ------------------------------------------------------------
        # 1. Foreground mask
        # ------------------------------------------------------------
        if foreground_mask is None:
            fg = torch.ones(B, N, 1, device=device, dtype=dtype)
            fg_bool = torch.ones(B, N, device=device, dtype=torch.bool)
        else:
            if foreground_mask.dim() == 3:
                foreground_mask = foreground_mask.flatten(1)

            fg_bool = foreground_mask.to(device=device).bool()
            fg = fg_bool.float().unsqueeze(-1).to(dtype=dtype)

        # ------------------------------------------------------------
        # 2. Cross-modal reconstruction with anchor guidance
        # ------------------------------------------------------------
        q_v = self.norm_q_v(d_v)
        q_i = self.norm_q_i(d_i)

        if self.use_anchor_as_kv:


            # 1) infrared online template d_i
            # 2) infrared initial template z_i
            kv_i = torch.cat([d_i, z_i], dim=1)
            kv_i_norm = self.norm_kv_i(kv_i)

            recon_v, _ = self.attn_v_from_i(
                query=q_v,
                key=kv_i_norm,
                value=kv_i,
            )



            # 1) visible online template d_v
            # 2) visible initial template z_v
            kv_v = torch.cat([d_v, z_v], dim=1)
            kv_v_norm = self.norm_kv_v(kv_v)

            recon_i, _ = self.attn_i_from_v(
                query=q_i,
                key=kv_v_norm,
                value=kv_v,
            )

        else:
            kv_i_norm = self.norm_kv_i(d_i)

            recon_v, _ = self.attn_v_from_i(
                query=q_v,
                key=kv_i_norm,
                value=d_i,
            )

            kv_v_norm = self.norm_kv_v(d_v)

            recon_i, _ = self.attn_i_from_v(
                query=q_i,
                key=kv_v_norm,
                value=d_v,
            )

        # ------------------------------------------------------------
        # 3. Health cue construction
        # ------------------------------------------------------------
        cross_diff_v = torch.abs(d_v - recon_v)
        cross_diff_i = torch.abs(d_i - recon_i)

        anchor_diff_v = torch.abs(d_v - z_v)
        anchor_diff_i = torch.abs(d_i - z_i)

        quality_input_v = torch.cat(
            [d_v, recon_v, cross_diff_v, anchor_diff_v],
            dim=-1,
        )

        quality_input_i = torch.cat(
            [d_i, recon_i, cross_diff_i, anchor_diff_i],
            dim=-1,
        )

        health_logit_v = self.health_scorer_v(
            self.norm_quality_v(quality_input_v)
        )

        health_logit_i = self.health_scorer_i(
            self.norm_quality_i(quality_input_i)
        )

        # ------------------------------------------------------------
        # 4. Logit competition repair gate
        # ------------------------------------------------------------




        repair_logit_v = health_logit_i.detach() - health_logit_v - self.repair_margin





        repair_logit_i = health_logit_v.detach() - health_logit_i - self.repair_margin

        repair_w_v = torch.sigmoid(repair_logit_v) * fg
        repair_w_i = torch.sigmoid(repair_logit_i) * fg

        # ------------------------------------------------------------
        # 5. Soft residual repair
        # ------------------------------------------------------------
        d_fixed_v = d_v + repair_w_v * (recon_v - d_v)
        d_fixed_i = d_i + repair_w_i * (recon_i - d_i)

        # ------------------------------------------------------------
        # 6. Statistics
        # ------------------------------------------------------------
        stats = {
            "repair_w_v": repair_w_v.mean().detach(),
            "repair_w_i": repair_w_i.mean().detach(),
            "health_v": health_logit_v.mean().detach(),
            "health_i": health_logit_i.mean().detach(),
        }



        if contam_mask_v is not None:
            contam_mask_v = contam_mask_v.to(device=device).bool()

            if contam_mask_v.dim() == 3:
                contam_mask_v = contam_mask_v.flatten(1)

            valid_v = contam_mask_v & fg_bool

            if valid_v.any():
                stats["repair_w_v_contam"] = repair_w_v.squeeze(-1)[valid_v].mean().detach()
            else:
                stats["repair_w_v_contam"] = torch.tensor(0.0, device=device)

        if contam_mask_i is not None:
            contam_mask_i = contam_mask_i.to(device=device).bool()

            if contam_mask_i.dim() == 3:
                contam_mask_i = contam_mask_i.flatten(1)

            valid_i = contam_mask_i & fg_bool

            if valid_i.any():
                stats["repair_w_i_contam"] = repair_w_i.squeeze(-1)[valid_i].mean().detach()
            else:
                stats["repair_w_i_contam"] = torch.tensor(0.0, device=device)

        return d_fixed_v, d_fixed_i, stats