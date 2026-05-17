import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import math

class AdapterBank(nn.Module):
    """
    Self-Reflective Mixture of Adapters (LoRA Bank).
    Dynamically routes representations through specific experts based on router weights.
    
    Shapes:
        lora_A: [num_adapters, in_features, rank]
        lora_B: [num_adapters, rank, out_features]
    """
    def __init__(self, in_features: int, out_features: int, num_adapters: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.num_adapters = num_adapters
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(num_adapters, in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Applies the dynamically routed adapters to the input tensor.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, in_features]
            routing_weights: Tensor of shape [batch_size, num_adapters]
            
        Returns:
            Tensor of shape [batch_size, seq_len, out_features]
        """
        x = x.to(self.lora_A.dtype)
        routing_weights = routing_weights.to(self.lora_A.dtype)
        
        # Compute the weighted dynamic adapter matrices for this batch
        W_A_dynamic = torch.einsum('ba,air->bir', routing_weights, self.lora_A)
        W_B_dynamic = torch.einsum('ba,aro->bro', routing_weights, self.lora_B)
        
        # Apply the adapter
        lora_hidden = torch.bmm(x, W_A_dynamic)
        lora_out = torch.bmm(lora_hidden, W_B_dynamic)
        
        return lora_out * self.scaling


class Router(nn.Module):
    """
    Differentiable Routing Network.
    Calculates routing logits to select the optimal adapter based on current context and memory state.
    """
    def __init__(self, memory_dim: int, hidden_dim: int, num_adapters: int):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(memory_dim + hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_adapters)
        )

    def forward(self, memory_state: torch.Tensor, current_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_state: Tensor of shape [batch_size, memory_dim] from the external memory graph (DSM)
            current_hidden: Tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            logits: Tensor of shape [batch_size, num_adapters]
        """
        # We use the final token as the representative context for routing
        context = current_hidden[:, -1, :] 
        
        # Fuse memory and current context
        fused_state = torch.cat([memory_state, context], dim=-1)
        fused_state = fused_state.to(self.proj[0].weight.dtype)
        
        logits = self.proj(fused_state)
        return logits


class SRMoAModel(nn.Module):
    """
    Self-Reflective Mixture of Adapters (SR-MoA) Global Architecture.
    Combines a frozen pre-trained LLM core with a differentiable routing network and adapter bank.
    """
    def __init__(self, base_model_name: str, memory_dim: int = 1024, num_adapters: int = 16, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self._last_past_key_values = None
        
        # 1. Load Base Model (Frozen Core)
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=True)
        except OSError:
            print("[SR-MoA] Local cache miss. Downloading base model weights...")
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Freeze base parameters (Critical for SR-TTT stability)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        hidden_dim = self.base_model.config.hidden_size
        
        # 2. Initialize Routing and Adapter Layers
        self.router = Router(memory_dim=memory_dim, hidden_dim=hidden_dim, num_adapters=num_adapters).float()
        self.adapter_bank = AdapterBank(in_features=hidden_dim, out_features=hidden_dim, num_adapters=num_adapters).float()

    def forward(self, input_ids: torch.Tensor, memory_state: torch.Tensor, attention_mask=None, past_key_values=None):
        """
        Standard forward pass through the frozen core and the dynamic SR-MoA layers.
        """
        # Process through the frozen core
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        
        self._last_past_key_values = outputs.past_key_values
        hidden_states = outputs.hidden_states[-1]
        
        # Calculate routing probabilities
        router_logits = self.router(memory_state.float(), hidden_states.float())
        routing_weights = F.softmax(router_logits / self.temperature, dim=-1)
        
        # Apply the chosen adapter pathways
        dynamic_delta = self.adapter_bank(hidden_states.float(), routing_weights)
        
        dtype = hidden_states.dtype
        modulated_hidden_states = hidden_states + dynamic_delta.to(dtype)
        
        # Final Language Modeling Head prediction
        logits = self.base_model.lm_head(modulated_hidden_states)
        
        return logits, routing_weights, router_logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, memory_state: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.5, pad_token_id: int = None, attention_mask: torch.Tensor = None):
        """
        Optimized generation loop utilizing KV-caching.
        O(1) complexity per token.
        """
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is None:
                logits, _, _ = self(
                    generated, 
                    memory_state, 
                    attention_mask=attention_mask,
                    past_key_values=None
                )
            else:
                logits, _, _ = self(
                    generated[:, -1:], 
                    memory_state, 
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )
            
            past_key_values = self._last_past_key_values
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                ], dim=-1)
                
            if pad_token_id is not None and (next_token == pad_token_id).all():
                break
                
        return generated
