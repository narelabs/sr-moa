import torch
import torch.nn.functional as F

class SRMoATrainer:
    """
    Trainer for Self-Reflective Test-Time Training (SR-TTT).
    Manages gradient calculation and parameter updates for the SR-MoA architecture.
    Updates only the Router and Adapter parameters, leaving the base LLM frozen.
    """
    def __init__(self, learning_rate: float = 5e-4, router_loss_weight: float = 1.5):
        self.learning_rate = learning_rate
        self.router_loss_weight = router_loss_weight

    def compute_gradients(self, model, tokenizer, rule_text: str, target_adapter_idx: int):
        """
        Translates a symbolic correction rule into neural gradient signals.
        Computes both Causal LM loss (for memory embedding) and Router CrossEntropy loss (for routing mapping).
        
        Args:
            model: The SRMoAModel instance
            tokenizer: The HuggingFace tokenizer
            rule_text (str): The text of the correction rule to be memorized
            target_adapter_idx (int): The index of the adapter targeted for adaptation
            
        Returns:
            loss (Tensor): The composite weighted loss
            routing_weights (Tensor): Probabilities output by the router
        """
        # Tokenize rule text
        inputs = tokenizer(rule_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        input_ids = inputs.input_ids

        # Build memory state targeting the specific adapter
        memory_state = torch.zeros(1, 128, device="cuda")
        memory_state[0, target_adapter_idx % 128] = 5.0
        memory_state = memory_state.to(next(model.parameters()).dtype)

        # Forward pass
        logits, routing_weights, router_logits = model(
            input_ids, memory_state, attention_mask=inputs.attention_mask
        )

        # 1. Causal LM Loss (incorporates rule into target adapter weights)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_lm = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # 2. Router Loss (trains the router to activate target adapter for this context)
        target_tensor = torch.tensor([target_adapter_idx % 16], dtype=torch.long, device="cuda")
        loss_router = F.cross_entropy(router_logits, target_tensor)

        # Combined loss
        loss = loss_lm + self.router_loss_weight * loss_router

        return loss, routing_weights

    def step(self, model, loss) -> int:
        """
        Executes a single backpropagation step.
        Updates only parameters that have requires_grad=True (Router and active Adapters).
        
        Args:
            model: The SRMoAModel instance
            loss (Tensor): The computed loss to backpropagate
            
        Returns:
            int: Number of parameters updated
        """
        model.zero_grad()
        loss.backward()

        # Isolate trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # Perform manual gradient descent step
        with torch.no_grad():
            for p in trainable_params:
                if p.grad is not None and not torch.isnan(p.grad).any():
                    p.data -= self.learning_rate * p.grad

        return len(trainable_params)
