from collections import defaultdict

import numpy as np
import torch


def compute_rs_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    beta: float = 2.0,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2soft = {}

    rewards = scores

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2soft[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                tmp_scores = torch.tensor(id2score[idx], dtype=torch.float32)
                log_len = torch.log(torch.tensor(len(tmp_scores), dtype=torch.float32))  # compute log constant
                if beta != 0:
                    id2soft[idx] = (1 / beta) * (torch.logsumexp(tmp_scores * beta, dim=0) - log_len)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            pass_at_1_scores[i] = rewards[i] - id2mean[index[i]]
            if beta == 0:
                scores[i] = pass_at_1_scores[i]
            else:
                scores[i] = rewards[i] - id2soft[index[i]]
                scores[i] = (torch.exp(beta * scores[i]) - 1) / beta

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return scores, scores