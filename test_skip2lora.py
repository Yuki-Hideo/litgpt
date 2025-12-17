#!/usr/bin/env python
# Test script for Skip2-LoRA integration in LitGPT

import torch
from litgpt.lora import Config, GPT, Skip2LoRA
from litgpt.config import Config as BaseConfig

def test_skip2lora_class():
    """Test Skip2LoRA class initialization and forward pass"""
    print("Testing Skip2LoRA class...")
    
    skip2lora = Skip2LoRA(
        in_features=768,
        out_features=768,
        r=32,
        lora_alpha=16,
        lora_dropout=0.05
    )
    
    # Test forward pass
    x = torch.randn(2, 10, 768)  # (batch_size, seq_len, embd)
    output = skip2lora(x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"  ✓ Skip2LoRA forward pass: input shape {x.shape} -> output shape {output.shape}")


def test_skip2lora_disabled():
    """Test Skip2LoRA with r=0 (disabled)"""
    print("Testing Skip2LoRA disabled (r=0)...")
    
    skip2lora = Skip2LoRA(
        in_features=768,
        out_features=768,
        r=0,
        lora_alpha=16,
        lora_dropout=0.05
    )
    
    x = torch.randn(2, 10, 768)
    output = skip2lora(x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert torch.allclose(output, torch.zeros_like(output)), "Output should be all zeros when r=0"
    print(f"  ✓ Skip2LoRA disabled: produces zero tensor")


def test_config_skip2lora_params():
    """Test Config class with Skip2-LoRA parameters"""
    print("Testing Config with Skip2-LoRA parameters...")
    
    config = Config(
        name="test",
        block_size=512,
        vocab_size=50000,
        padded_vocab_size=50256,
        n_layer=32,
        n_head=32,
        n_embd=2048,
        skip2lora_enabled=True,
        skip2lora_block_indices=(0, 1, 2, 3),
        skip2lora_output_layer="lm_head",
        lora_r=32,
        lora_alpha=16,
    )
    
    assert config.skip2lora_enabled == True
    assert config.skip2lora_block_indices == (0, 1, 2, 3)
    assert config.skip2lora_output_layer == "lm_head"
    print("  ✓ Config Skip2-LoRA parameters loaded correctly")


def test_gpt_with_skip2lora():
    """Test GPT model with Skip2-LoRA"""
    print("Testing GPT model with Skip2-LoRA...")
    
    config = Config(
        name="test",
        block_size=128,
        vocab_size=50000,
        padded_vocab_size=50256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        head_size=32,
        skip2lora_enabled=True,
        skip2lora_block_indices=(0, 1),  # Apply Skip2-LoRA to first 2 blocks
        skip2lora_output_layer="lm_head",
        lora_r=16,
        lora_alpha=16,
    )
    
    model = GPT(config)
    
    # Check that blocks have skip2lora layers
    assert model.transformer.h[0].skip2lora is not None, "Block 0 should have skip2lora"
    assert model.transformer.h[1].skip2lora is not None, "Block 1 should have skip2lora"
    assert model.transformer.h[2].skip2lora is None, "Block 2 should NOT have skip2lora"
    assert model.transformer.h[3].skip2lora is None, "Block 3 should NOT have skip2lora"
    
    print("  ✓ Block skip2lora assignment correct")
    
    # Test forward pass
    idx = torch.randint(0, config.padded_vocab_size, (2, 32))  # (batch_size, seq_len)
    try:
        logits = model(idx)
        assert logits.shape == (2, 32, config.padded_vocab_size), f"Expected shape (2, 32, {config.padded_vocab_size}), got {logits.shape}"
        print(f"  ✓ Forward pass successful: output shape {logits.shape}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        raise


def test_skip2lora_disabled_model():
    """Test GPT model with Skip2-LoRA disabled"""
    print("Testing GPT model with Skip2-LoRA disabled...")
    
    config = Config(
        name="test",
        block_size=128,
        vocab_size=50000,
        padded_vocab_size=50256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        head_size=32,
        skip2lora_enabled=False,
        skip2lora_block_indices=(),
        skip2lora_output_layer="lm_head",
        lora_r=0,  # Disable LoRA
    )
    
    model = GPT(config)
    
    # Check that no blocks have skip2lora layers
    for i in range(config.n_layer):
        assert model.transformer.h[i].skip2lora is None, f"Block {i} should NOT have skip2lora when disabled"
    
    print("  ✓ All blocks correctly have no skip2lora when disabled")
    
    # Test forward pass
    idx = torch.randint(0, config.padded_vocab_size, (2, 32))
    logits = model(idx)
    assert logits.shape == (2, 32, config.padded_vocab_size)
    print(f"  ✓ Forward pass successful: output shape {logits.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Skip2-LoRA Integration Tests")
    print("=" * 60)
    
    try:
        test_skip2lora_class()
        test_skip2lora_disabled()
        test_config_skip2lora_params()
        test_gpt_with_skip2lora()
        test_skip2lora_disabled_model()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
