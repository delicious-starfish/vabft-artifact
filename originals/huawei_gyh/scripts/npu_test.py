#!/usr/bin/env python3
"""
Bit-level precision test - Full comparison
Run this script on both platforms and compare results
"""

import numpy as np

# Try to import torch_npu for Ascend
try:
    import torch_npu
    has_npu = True
except ImportError:
    has_npu = False

import torch

def get_device():
    """Get the available accelerator device"""
    if has_npu and torch.npu.is_available():
        return torch.device("npu"), "NPU"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    else:
        return torch.device("cpu"), "CPU"

def view_bits(x):
    """View float32 as uint32 to compare exact bits"""
    return np.array(x.flatten().cpu().numpy()).view(np.uint32)

print("=" * 70)
print("Bit-Level Precision Test - Full Report")
print("=" * 70)

device, device_type = get_device()
print(f"\nDevice: {device_type}")
if device_type == "CUDA":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
elif device_type == "NPU":
    print(f"  NPU: {torch.npu.get_device_name(0)}")
print(f"  PyTorch: {torch.__version__}")
if has_npu:
    print(f"  torch_npu: {torch_npu.__version__}")

# ============================================================================
# Test 1: Basic Arithmetic on Device
# ============================================================================
print("\n" + "=" * 70)
print("Test 1: Basic Arithmetic (+ - * /)")
print("=" * 70)

test_cases = [
    ("Add: 0.1 + 0.2", 0.1, 0.2, lambda a, b: a + b),
    ("Add: 1.5 + 2.25", 1.5, 2.25, lambda a, b: a + b),
    ("Sub: 5.0 - 1.25", 5.0, 1.25, lambda a, b: a - b),
    ("Mul: 0.1 * 0.1", 0.1, 0.1, lambda a, b: a * b),
    ("Mul: 3.5 * 2.5", 3.5, 2.5, lambda a, b: a * b),
    ("Div: 7.0 / 3.0", 7.0, 3.0, lambda a, b: a / b),
    ("Div: 1.0 / 3.0", 1.0, 3.0, lambda a, b: a / b),
    ("Div: 10.0 / 7.0", 10.0, 7.0, lambda a, b: a / b),
]

basic_bits = {}
print("\n[Results on Device]")
for name, a_val, b_val, op in test_cases:
    a = torch.tensor(a_val, dtype=torch.float32, device=device)
    b = torch.tensor(b_val, dtype=torch.float32, device=device)
    c = op(a, b)
    bits_c = view_bits(c)[0]
    basic_bits[name] = hex(bits_c)
    print(f"{name:20s} | bits: {hex(bits_c):10s} | value: {c.item():.15f}")

# ============================================================================
# Test 2: Element-wise Vector Operations
# ============================================================================
print("\n" + "=" * 70)
print("Test 2: Element-wise Vector Operations (First 8 elements)")
print("=" * 70)

vec_a = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=torch.float32, device=device)
vec_b = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=torch.float32, device=device)

vec_add = vec_a + vec_b
vec_mul = vec_a * vec_b
vec_div = vec_a / vec_b

print(f"{'Idx':<4} {'a':<12} {'b':<12} {'a+b':<12} {'a*b':<12} {'a/b':<12}")
print("-" * 70)
for i in range(8):
    print(f"{i:<4} {hex(view_bits(vec_a)[i]):<10} {hex(view_bits(vec_b)[i]):<10} "
          f"{hex(view_bits(vec_add)[i]):<10} {hex(view_bits(vec_mul)[i]):<10} {hex(view_bits(vec_div)[i]):<10}")

# ============================================================================
# Test 3: Matrix Multiplication (Small)
# ============================================================================
print("\n" + "=" * 70)
print("Test 3: Matrix Multiplication (2x2)")
print("=" * 70)

A = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32, device=device)
B = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32, device=device)
C = torch.mm(A, B)

bits_C = view_bits(C)
print(f"A @ B bits: {[hex(b) for b in bits_C]}")

# ============================================================================
# Test 4: Reduction (Sum of 100 * 0.1)
# ============================================================================
print("\n" + "=" * 70)
print("Test 4: Reduction Operations")
print("=" * 70)

red_vec = torch.ones(100, dtype=torch.float32, device=device) * 0.1
red_sum = torch.sum(red_vec)
bits_sum = hex(view_bits(red_sum)[0])
print(f"Sum of 100 * 0.1: value = {red_sum.item():.15f} | bits = {bits_sum}")

# Mean
red_mean = torch.mean(red_vec)
bits_mean = hex(view_bits(red_mean)[0])
print(f"Mean of 100 * 0.1: value = {red_mean.item():.15f} | bits = {bits_mean}")

# ============================================================================
# Test 5: Mathematical Functions
# ============================================================================
print("\n" + "=" * 70)
print("Test 5: Mathematical Functions")
print("=" * 70)

math_tests = [
    ("sqrt(0.5)", lambda x: torch.sqrt(x), 0.5),
    ("exp(1.0)", lambda x: torch.exp(x), 1.0),
    ("exp(2.0)", lambda x: torch.exp(x), 2.0),
    ("log(2.0)", lambda x: torch.log(x), 2.0),
    ("sin(0)", lambda x: torch.sin(x), 0.0),
    ("sin(1.57)", lambda x: torch.sin(x), 1.57),
    ("cos(0)", lambda x: torch.cos(x), 0.0),
    ("pow(2.0, 3)", lambda x: torch.pow(x, 3), 2.0),
]

math_bits = {}
for name, func, val in math_tests:
    x = torch.tensor(val, dtype=torch.float32, device=device)
    y = func(x)
    bits_y = hex(view_bits(y)[0])
    math_bits[name] = bits_y
    print(f"{name:15s} | bits: {bits_y:10s} | value: {y.item():.15f}")

# ============================================================================
# Test 6: JIT Kernel
# ============================================================================
print("\n" + "=" * 70)
print("Test 6: JIT Compiled Kernel")
print("=" * 70)

@torch.jit.script
def add_mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x + y) * x

a = torch.tensor(2.5, dtype=torch.float32, device=device)
b = torch.tensor(1.5, dtype=torch.float32, device=device)
c = add_mul_kernel(a, b)
bits_jit = hex(view_bits(c)[0])
print(f"(a + b) * a where a=2.5, b=1.5: bits = {bits_jit}, value = {c.item():.15f}")

# ============================================================================
# Test 7: Special Values
# ============================================================================
print("\n" + "=" * 70)
print("Test 7: Special Values")
print("=" * 70)

try:
    inf_bits = hex(view_bits(torch.tensor(1.0, device=device) / torch.tensor(0.0, device=device))[0])
    neg_inf_bits = hex(view_bits(torch.tensor(-1.0, device=device) / torch.tensor(0.0, device=device))[0])
    nan_bits = hex(view_bits(torch.tensor(0.0, device=device) / torch.tensor(0.0, device=device))[0])
    pos_zero_bits = hex(view_bits(torch.tensor(0.0, device=device))[0])
    neg_zero_bits = hex(view_bits(-torch.tensor(0.0, device=device))[0])

    print(f"+Inf:  {inf_bits}")
    print(f"-Inf:  {neg_inf_bits}")
    print(f"NaN:   {nan_bits}")
    print(f"+0.0:  {pos_zero_bits}")
    print(f"-0.0:  {neg_zero_bits}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Test 8: Low Precision (float16, bfloat16)
# ============================================================================
print("\n" + "=" * 70)
print("Test 8: Low Precision Types")
print("=" * 70)

# float16
a_f16 = torch.tensor(1.5, dtype=torch.float16, device=device)
b_f16 = torch.tensor(2.25, dtype=torch.float16, device=device)
c_f16 = a_f16 + b_f16
c_f16_as_f32 = c_f16.float()
f16_bits = hex(view_bits(c_f16_as_f32)[0])
print(f"float16 (1.5 + 2.25): bits = {f16_bits}, value = {c_f16_as_f32.item():.15f}")

# bfloat16
try:
    a_bf16 = torch.tensor(1.5, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.tensor(2.25, dtype=torch.bfloat16, device=device)
    c_bf16 = a_bf16 + b_bf16
    c_bf16_as_f32 = c_bf16.float()
    bf16_bits = hex(view_bits(c_bf16_as_f32)[0])
    print(f"bfloat16 (1.5 + 2.25): bits = {bf16_bits}, value = {c_bf16_as_f32.item():.15f}")
except Exception as e:
    print(f"bfloat16 error: {e}")

# ============================================================================
# Summary output (for easy comparison)
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY - Bit Values for Comparison")
print("=" * 70)

print("\n[Basic Arithmetic]")
for name, bits in basic_bits.items():
    print(f"  {name}: {bits}")

print("\n[Matrix Multiplication (2x2)]")
print(f"  Result bits: {[hex(b) for b in bits_C]}")

print("\n[Reduction]")
print(f"  Sum of 100 * 0.1: {bits_sum}")
print(f"  Mean of 100 * 0.1: {bits_mean}")

print("\n[Math Functions]")
for name, bits in math_bits.items():
    print(f"  {name}: {bits}")

print("\n[JIT Kernel]")
print(f"  (2.5 + 1.5) * 2.5: {bits_jit}")

print("\n[Special Values]")
print(f"  +Inf: {inf_bits if 'inf_bits' in locals() else 'N/A'}")
print(f"  NaN: {nan_bits if 'nan_bits' in locals() else 'N/A'}")

print("\n" + "=" * 70)
