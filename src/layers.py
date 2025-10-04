import functools
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import utils

class RandomFourierFeaturesGaussianProcess(torch.nn.Module):
    def __init__(self, in_features, out_features, learnable_lengthscale=False, learnable_outputscale=False, lengthscale=20.0, outputscale=1.0, rank=1024):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.learnable_lengthscale = learnable_lengthscale
        self.learnable_outputscale = learnable_outputscale
        
        if self.learnable_lengthscale:
            self.raw_lengthscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(lengthscale, dtype=torch.float32)))
        else:
            self.register_buffer("raw_lengthscale", utils.inv_softplus(lengthscale, dtype=torch.float32))
        
        if self.learnable_outputscale:
            self.raw_outputscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float32)))
        else:
            self.register_buffer("raw_outputscale", utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float32)))
                    
        self.rank = rank
        self.register_buffer("feature_weight", torch.randn(self.rank, self.in_features))
        self.register_buffer("feature_bias", 2 * torch.pi * torch.rand(self.rank))
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=self.out_features, bias=False)

    def featurize(self, h):
        return self.outputscale * (2/self.rank)**0.5 * torch.cos(torch.nn.functional.linear(h, (1/self.lengthscale) * self.feature_weight, self.feature_bias))
        
    def forward(self, h):
        features = self.featurize(h)
        logits = self.linear(features)
        return logits
                                                    
    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.raw_outputscale)
    
class CNBlock(torch.nn.Module):
    def __init__(self, dim, layer_scale, stochastic_depth_prob, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(torch.nn.LayerNorm, eps=1e-6)

        self.layer_scale = torch.nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False)
        self.layer_scale.weight.data.fill_(layer_scale)
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            torchvision.ops.misc.Permute([0, 2, 3, 1]),
            norm_layer(dim),
            torch.nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            torchvision.ops.misc.Permute([0, 3, 1, 2]),
        )
        self.stochastic_depth = torchvision.ops.StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale(self.block(input))
        result = self.stochastic_depth(result)
        result += input
        return result
    
class VariationalLinear(torch.nn.Module):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.raw_sigma = raw_sigma
        self.use_posterior = use_posterior
                
    def forward(self, x):
        if self.training or self.use_posterior:
            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            return torch.nn.functional.linear(
                x, 
                variational_weight, 
                variational_bias,
            )
        return self.layer(x)
    
    def _variational_params(self):
        params = self._flatten()
        eps = torch.randn_like(params)
        sigma = torch.nn.functional.softplus(self.raw_sigma)
        return params + sigma * eps
            
    def _flatten(self):
        return torch.cat([param.view(-1) for param in [self.layer.weight, self.layer.bias] if param is not None])
    
    def _unflatten(self, params):
        out = []
        for param in [self.layer.weight, self.layer.bias]:
            out.append(None if param is None else params[:param.numel()].view_as(param))
            if param is not None:
                params = params[param.numel():]
        return out
     
class VariationalConv2d(VariationalLinear):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__(layer, raw_sigma, use_posterior)
        
    def forward(self, x):
        if self.training or self.use_posterior:
            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            return torch.nn.functional.conv2d(
                x,
                variational_weight,
                variational_bias,
                self.layer.stride,
                self.layer.padding,
                self.layer.dilation,
                self.layer.groups
            )
        
        return self.layer(x)

class VariationalBatchNorm2d(VariationalLinear):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__(layer, raw_sigma, use_posterior)
        
    def forward(self, x):
        if self.training or self.use_posterior:

            if self.layer.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.layer.momentum

            if self.layer.training and self.layer.track_running_stats:
                if self.layer.num_batches_tracked is not None:
                    self.layer.num_batches_tracked.add_(1)
                    if self.layer.momentum is None:
                        exponential_average_factor = 1.0 / float(self.layer.num_batches_tracked)
                    else:
                        exponential_average_factor = self.layer.momentum

            if self.layer.training:
                bn_training = True
            else:
                bn_training = (self.layer.running_mean is None) and (self.layer.running_var is None)

            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            return torch.nn.functional.batch_norm(
                x, 
                self.layer.running_mean if not self.layer.training or self.layer.track_running_stats else None, 
                self.layer.running_var if not self.layer.training or self.layer.track_running_stats else None, 
                variational_weight,
                variational_bias,
                bn_training, 
                exponential_average_factor, 
                self.layer.eps, 
            )
        
        return self.layer(x)

class VariationalLayerNorm(VariationalLinear):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__(layer, raw_sigma, use_posterior)
    
    def forward(self, x):
        if self.training or self.use_posterior:
            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            return torch.nn.functional.layer_norm(
                x,
                self.layer.normalized_shape,
                variational_weight,
                variational_bias,
                self.layer.eps
            )
            
        return self.layer(x)
    
class VariationalLayerNorm2d(VariationalLinear):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__(layer, raw_sigma, use_posterior)
    
    def forward(self, x):
        if self.training or self.use_posterior:
            x = x.permute(0, 2, 3, 1)
            variational_params = self._variational_params()
            variational_weight, variational_bias = self._unflatten(variational_params)
            x = torch.nn.functional.layer_norm(
                x,
                self.layer.normalized_shape,
                variational_weight,
                variational_bias,
                self.layer.eps
            )
            x = x.permute(0, 3, 1, 2)
            return x
            
        return self.layer(x)
    
class VariationalCNBlock(torch.nn.Module):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.raw_sigma = raw_sigma
        self.use_posterior = use_posterior
        
    def forward(self, x):
        if self.training or self.use_posterior:
            variational_params = self._variational_params()
            variational_layer_scale, = self._unflatten(variational_params)
            result = variational_layer_scale * self.layer.block(x)
            result = self.layer.stochastic_depth(result)
            result += x
            return result
    
        return self.layer(x)
       
    def _variational_params(self):
        params = self._flatten()
        eps = torch.randn_like(params)
        sigma = torch.nn.functional.softplus(self.raw_sigma)
        return params + sigma * eps
            
    def _flatten(self):
        return torch.cat([param.view(-1) for param in [self.layer.layer_scale] if param is not None])
    
    def _unflatten(self, params):
        out = []
        for param in [self.layer.layer_scale]:
            out.append(None if param is None else params[:param.numel()].view_as(param))
            if param is not None:
                params = params[param.numel():]
        return out
        
class VariationalMultiheadAttention(torch.nn.Module):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.raw_sigma = raw_sigma
        self.use_posterior = use_posterior

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        if self.training or self.use_posterior:
            why_not_fast_path = ""
            if (
                (attn_mask is not None and torch.is_floating_point(attn_mask))
                or (key_padding_mask is not None)
                and torch.is_floating_point(key_padding_mask)
            ):
                why_not_fast_path = "floating-point masks are not supported for fast path."

            is_batched = query.dim() == 3

            key_padding_mask = torch.nn.functional._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=torch.nn.functional._none_or_dtype(attn_mask),
                other_name="attn_mask",
                target_type=query.dtype,
            )

            attn_mask = torch.nn.functional._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

            if not is_fastpath_enabled:
                why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
            elif not is_batched:
                why_not_fast_path = (
                    f"input not batched; expected query.dim() of 3 but got {query.dim()}"
                )
            elif query is not key or key is not value:
                why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
            elif self.layer.in_proj_bias is not None and query.dtype != self.layer.in_proj_bias.dtype:
                why_not_fast_path = f"dtypes of query ({query.dtype}) and self.layer.in_proj_bias ({self.layer.in_proj_bias.dtype}) don't match"
            elif self.layer.in_proj_weight is None:
                why_not_fast_path = "in_proj_weight was None"
            elif query.dtype != self.layer.in_proj_weight.dtype:
                why_not_fast_path = f"dtypes of query ({query.dtype}) and self.layer.in_proj_weight ({self.layer.in_proj_weight.dtype}) don't match"
            elif self.layer.training:
                why_not_fast_path = "training is enabled"
            elif (self.layer.num_heads % 2) != 0:
                why_not_fast_path = "self.layer.num_heads is not even"
            elif not self.layer.batch_first:
                why_not_fast_path = "batch_first was not True"
            elif self.layer.bias_k is not None:
                why_not_fast_path = "self.layer.bias_k was not None"
            elif self.layer.bias_v is not None:
                why_not_fast_path = "self.layer.bias_v was not None"
            elif self.layer.add_zero_attn:
                why_not_fast_path = "add_zero_attn was enabled"
            elif not self.layer._qkv_same_embed_dim:
                why_not_fast_path = "_qkv_same_embed_dim was not True"
            elif query.is_nested and (
                key_padding_mask is not None or attn_mask is not None
            ):
                why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                     is not supported with NestedTensor input"
            elif torch.is_autocast_enabled():
                why_not_fast_path = "autocast is enabled"

            if not why_not_fast_path:
                variational_params = self._variational_params()
                variational_in_proj_weight, variational_in_proj_bias, variational_out_proj_weight, variational_out_proj_bias, variational_bias_k, variational_bias_v, variational_q_proj_weight, variational_k_proj_weight, variational_v_proj_weight = _unflatten(self, variational_params)
                tensor_args = (
                    query,
                    key,
                    value,
                    variational_in_proj_weight,
                    variational_in_proj_bias,
                    variational_out_proj_weight,
                    variational_out_proj_bias,
                )
                if torch.overrides.has_torch_function(tensor_args):
                    why_not_fast_path = "some Tensor argument has_torch_function"
                elif _is_make_fx_tracing():
                    why_not_fast_path = "we are running make_fx tracing"
                elif not all(_check_arg_device(x) for x in tensor_args):
                    why_not_fast_path = (
                        "some Tensor argument's device is neither one of "
                        f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                    )
                elif torch.is_grad_enabled() and any(
                    _arg_requires_grad(x) for x in tensor_args
                ):
                    why_not_fast_path = (
                        "grad is enabled and at least one of query or the "
                        "input/output projection weights or biases requires_grad"
                    )
                if not why_not_fast_path:
                    merged_mask, mask_type = self.layer.merge_masks(
                        attn_mask, key_padding_mask, query
                    )

                    if self.layer.in_proj_bias is not None and self.layer.in_proj_weight is not None:
                        variational_params = self._variational_params()
                        variational_in_proj_weight, variational_in_proj_bias, variational_out_proj_weight, variational_out_proj_bias, variational_bias_k, variational_bias_v, variational_q_proj_weight, variational_k_proj_weight, variational_v_proj_weight = _unflatten(self, variational_params)
                        return torch._native_multi_head_attention(
                            query,
                            key,
                            value,
                            self.layer.embed_dim,
                            self.layer.num_heads,
                            variational_in_proj_weight,
                            variational_in_proj_bias,
                            variational_out_proj_weight,
                            variational_out_proj_bias,
                            merged_mask,
                            need_weights,
                            average_attn_weights,
                            mask_type,
                        )

            any_nested = query.is_nested or key.is_nested or value.is_nested
            assert not any_nested, (
                "MultiheadAttention does not support NestedTensor outside of its fast path. "
                + f"The fast path was not hit because {why_not_fast_path}"
            )

            if self.layer.batch_first and is_batched:
                if key is value:
                    if query is key:
                        query = key = value = query.transpose(1, 0)
                    else:
                        query, key = (x.transpose(1, 0) for x in (query, key))
                        value = key
                else:
                    query, key, value = (x.transpose(1, 0) for x in (query, key, value))

            if not self.layer._qkv_same_embed_dim:
                variational_params = self._variational_params()
                variational_in_proj_weight, variational_in_proj_bias, variational_out_proj_weight, variational_out_proj_bias, variational_bias_k, variational_bias_v, variational_q_proj_weight, variational_k_proj_weight, variational_v_proj_weight = _unflatten(self, variational_params)
                attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.layer.embed_dim,
                    self.layer.num_heads,
                    variational_in_proj_weight,
                    variational_in_proj_bias,
                    variational_bias_k,
                    variational_bias_v,
                    self.layer.add_zero_attn,
                    self.layer.dropout,
                    variational_out_proj_weight,
                    variational_out_proj_bias,
                    training=self.layer.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=variational_q_proj_weight,
                    k_proj_weight=variational_k_proj_weight,
                    v_proj_weight=variational_v_proj_weight,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal,
                )
            else:
                variational_params = self._variational_params()
                variational_in_proj_weight, variational_in_proj_bias, variational_out_proj_weight, variational_out_proj_bias, variational_bias_k, variational_bias_v, variational_q_proj_weight, variational_k_proj_weight, variational_v_proj_weight = _unflatten(self, variational_params)
                attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.layer.embed_dim,
                    self.layer.num_heads,
                    variational_in_proj_weight,
                    variational_in_proj_bias,
                    variational_bias_k,
                    variational_bias_v,
                    self.layer.add_zero_attn,
                    self.layer.dropout,
                    variational_out_proj_weight,
                    variational_out_proj_bias,
                    training=self.layer.training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal,
                )
            if self.layer.batch_first and is_batched:
                return attn_output.transpose(1, 0), attn_output_weights
            else:
                return attn_output, attn_output_weights
            
        return self.layer(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)
    
    def _variational_params(self):
        params = self._flatten()
        eps = torch.randn_like(params)
        sigma = torch.nn.functional.softplus(self.raw_sigma)
        return params + sigma * eps
    
    def _flatten(self):
        return torch.cat([param.view(-1) for param in [self.layer.in_proj_weight, self.layer.in_proj_bias, self.layer.out_proj_weight, self.layer.out_proj_bias, self.layer.bias_k, self.layer.bias_v, self.layer.q_proj_weight, self.layer.k_proj_weight, self.layer.v_proj_weight] if param is not None])
    
    def _unflatten(self, params):
        out = []
        for param in [self.layer.in_proj_weight, self.layer.in_proj_bias, self.layer.out_proj_weight, self.layer.out_proj_bias, self.layer.bias_k, self.layer.bias_v, self.layer.q_proj_weight, self.layer.k_proj_weight, self.layer.v_proj_weight]:
            out.append(None if param is None else params[:param.numel()].view_as(param))
            if param is not None:
                params = params[param.numel():]
        return out

class VariationalEmbedding(torch.nn.Module):
    def __init__(self, layer, raw_sigma=None, use_posterior=False):
        super().__init__()
        self.layer = layer
        self.raw_sigma = raw_sigma
        self.use_posterior = use_posterior

    def forward(self, x):
        if self.training or self.use_posterior:
            variational_params = self._variational_params()
            variational_weight = self._unflatten(variational_params)
            return torch.nn.functional.embedding(
                x,
                variational_weight,
                self.layer.padding_idx,
                self.layer.max_norm,
                self.layer.norm_type,
                self.layer.scale_grad_by_freq,
                self.layer.sparse,
            )
        
        return self.layer(x)
    
    def _variational_params(self):
        params = self._flatten()
        eps = torch.randn_like(params)
        sigma = torch.nn.functional.softplus(self.raw_sigma)
        return params + sigma * eps
    
    def _flatten(self):
        return torch.cat([param.view(-1) for param in [self.layer.weight] if param is not None])
    
    def _unflatten(self, params):
        out = []
        for param in [self.layer.weight]:
            out.append(None if param is None else params[:param.numel()].view_as(param))
            if param is not None:
                params = params[param.numel():]
        return out
        