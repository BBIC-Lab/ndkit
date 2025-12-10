import torch
import torch.nn as nn
from .layers.mlp import MlpBlock
from torch.distributions.normal import Normal
from .registry import register_model

@register_model("StateMoE")
class Model(nn.Module):
    """ A simple implementation """
    def __init__(self, cfg):
        super().__init__()
        input_size = getattr(cfg, "input_size")
        output_size = getattr(cfg, "output_size")
        encoder_size = getattr(cfg, "encoder_size")
        encoder_num_layers = getattr(cfg, "encoder_num_layers")
        num_experts = getattr(cfg, "num_experts")
        expert_sizes = getattr(cfg, "expert_sizes")
        dropout = getattr(cfg, "dropout", 0.0)
        k = getattr(cfg, "k", 1)
        noisy_gating = getattr(cfg, "noisy_gating", True)
        w_imp = getattr(cfg, "w_imp", 0.0)
        w_load = getattr(cfg, "w_load", 0.0)

        assert(k <= num_experts)
        
        self.encoder = nn.LSTM(
            input_size,
            encoder_size,
            num_layers=encoder_num_layers,
            dropout=dropout if encoder_num_layers > 1 else 0,
            batch_first=True
        )
        self.router = LinearRNN(encoder_size, num_experts)
        self.w_noise = nn.Parameter(torch.zeros(encoder_size, num_experts), requires_grad=True)
        self.experts = nn.ModuleList(
            [MlpBlock(encoder_size,
                      output_size, 
                      hidden_sizes=expert_sizes,
                      dropout=dropout)
             for _ in range(num_experts)])
        
        self.noisy_gating = noisy_gating
        self.k = k
        self.num_experts = num_experts
        self.w_imp = w_imp
        self.w_load = w_load

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

  
    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_logits, noisy_logits, noise_stddev, noisy_top_logits):
        batch = clean_logits.size(0)
        m = noisy_top_logits.size(1)
        top_values_flat = noisy_top_logits.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_logits.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_logits, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_logits - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_logits - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, h_t, g_t, noise_epsilon = 1e-2):
        clean_logits = g_t
        if self.noisy_gating and self.training:
            raw_noise_stddev = h_t @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(self.num_experts, dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        importance = gates.sum(0)
        if self.noisy_gating and self.k < self.num_experts and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, importance, load

    def forward(self, x, dispatch=False, return_detail=False):
        h, _ = self.encoder(x) # 提取表征
        h_T = h[:, -1, :]
        g_T = self.router(h)

        w_T, importance, load = self.noisy_top_k_gating(h_T, g_T)
        loss = self.w_imp * self.cv_squared(importance) + self.w_load * self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, w_T)
        expert_inputs = dispatcher.dispatch(h_T)
        w_T = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y_pred = dispatcher.combine(expert_outputs)

        if not return_detail:
            return y_pred, loss
        else:
            return (y_pred, loss), (w_T, importance, load, expert_outputs)


class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Linear(input_size, hidden_size, bias=True)  
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)  
    
    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(x.device)  
        for t in range(x.size(1)): 
            h = self.Wxh(x[:, t, :]) + self.Whh(h)
        return h
    

class SparseDispatcher(object):

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
