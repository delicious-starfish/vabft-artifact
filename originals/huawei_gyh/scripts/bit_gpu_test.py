#!/usr/bin/env python3
"""
Bit-level precision test on GPU/NPU
Tests basic arithmetic and kernel-level operations
"""

import torch
import numpy as np

def get_device():
    """Get the available accelerator device - prefer accelerator over CPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device("npu"), "NPU"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    else:
        return torch.device("cpu"), "CPU"

def view_bits(x):
    """View float32 as uint32 to compare exact bits"""
    return np.array(x.flatten().cpu().numpy()).view(np.uint32)

print("=" * 70)
print("Bit-Level Precision Test on GPU/NPU")
print("=" * 70)

device, device_type = get_device()
print(f"\nDevice: {device_type}")
if device_type == "CUDA":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
elif device_type == "NPU":
    print(f"  NPU: {torch.npu.get_device_name(0)}")
print(f"  PyTorch: {torch.__version__}")

# ============================================================================
# Test 1: Basic Arithmetic on Device
# ============================================================================
print("\n" + "=" * 70)
print("Test 1: Basic Arithmetic (+ - * /) on Device")
print("=" * 70)

test_cases = [
    ("Add: 0.1 + 0.2", 0.1, 0.2, lambda a, b: a + b),
    ("Add: 1.5 + 2.25", 1.5, 2.25, lambda a, b: a + b),
    ("Sub: 5.0 - 1.25", 5.0, 1.25, lambda a, b: a - b),
    ("Sub: 1.0 - 0.9", 1.0, 0.9, lambda a, b: a - b),
    ("Mul: 0.1 * 0.1", 0.1, 0.1, lambda a, b: a * b),
    ("Mul: 3.5 * 2.5", 3.5, 2.5, lambda a, b: a * b),
    ("Mul: 1.1 * 1.1", 1.1, 1.1, lambda a, b: a * b),
    ("Div: 7.0 / 3.0", 7.0, 3.0, lambda a, b: a / b),
    ("Div: 1.0 / 3.0", 1.0, 3.0, lambda a, b: a / b),
    ("Div: 10.0 / 7.0", 10.0, 7.0, lambda a, b: a / b),
    ("Add: -1.5 + 2.5", -1.5, 2.5, lambda a, b: a + b),
    ("Mul: -2.5 * 3.0", -2.5, 3.0, lambda a, b: a * b),
    ("Div: -1.0 / 2.0", -1.0, 2.0, lambda a, b: a / b),
]

basic_results = []

print("\n[Results on Device]")
print("-" * 70)

for name, a_val, b_val, op in test_cases:
    a = torch.tensor(a_val, dtype=torch.float32, device=device)
    b = torch.tensor(b_val, dtype=torch.float32, device=device)
    c = op(a, b)

    bits_c = view_bits(c)

    print(f"{name:20s} | bits: {hex(bits_c[0]):10s} | value: {c.item():.15f}")
    basic_results.append((name, hex(bits_c[0]), c.item()))

# ============================================================================
# Test 2: FMA (Fused Multiply-Add)
# ============================================================================
print("\n" + "=" * 70)
print("Test 2: FMA (Fused Multiply-Add)")
print("=" * 70)

a = torch.tensor(2.5, dtype=torch.float32, device=device)
b = torch.tensor(3.5, dtype=torch.float32, device=device)
c_val = torch.tensor(1.0, dtype=torch.float32, device=device)

# Separate operations
d_separate = (a * b) + c_val

# FMA operation (if available)
try:
    d_fma = torch.fmadd(a, b, c_val)
    bits_fma = view_bits(d_fma)
    has_fmadd = True
except (AttributeError, TypeError):
    d_fma = None
    has_fmadd = False

bits_separate = view_bits(d_separate)

print(f"a = {a.item():.15f}, b = {b.item():.15f}, c = {c_val.item():.15f}")
print(f"(a * b) + c  = {d_separate.item():.15f} | bits: {hex(bits_separate[0])}")
if has_fmadd:
    print(f"fma(a,b,c)   = {d_fma.item():.15f} | bits: {hex(bits_fma[0])}")
    print(f"Same bits? {bits_separate[0] == bits_fma[0]}")
else:
    print(f"fmadd not available in this PyTorch version")

# ============================================================================
# Test 3: Element-wise Operations (Vector)
# ============================================================================
print("\n" + "=" * 70)
print("Test 3: Element-wise Operations (Vector of 16 elements)")
print("=" * 70)

# Create test vectors
vec_a = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                      4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
                     dtype=torch.float32, device=device)
vec_b = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                     dtype=torch.float32, device=device)

vec_add = vec_a + vec_b
vec_mul = vec_a * vec_b
vec_div = vec_a / vec_b

bits_a = view_bits(vec_a)
bits_b = view_bits(vec_b)
bits_add = view_bits(vec_add)
bits_mul = view_bits(vec_mul)
bits_div = view_bits(vec_div)

print("\nFirst 8 elements of vector operations:")
print("-" * 70)
print(f"{'Index':<6} {'a':<12} {'b':<12} {'a+b':<12} {'a*b':<12} {'a/b':<12}")
print("-" * 70)

for i in range(8):
    print(f"{i:<6} {hex(bits_a[i]):<10} {hex(bits_b[i]):<10} "
          f"{hex(bits_add[i]):<10} {hex(bits_mul[i]):<10} {hex(bits_div[i]):<10}")

# ============================================================================
# Test 4: Reduction Operations (Sum)
# ============================================================================
print("\n" + "=" * 70)
print("Test 4: Reduction Operations (Sum, Mean, Prod)")
print("=" * 70)

# Test sum of a simple sequence
red_vec = torch.ones(100, dtype=torch.float32, device=device) * 0.1
red_sum = torch.sum(red_vec)

bits_sum = view_bits(red_sum)
print(f"Sum of 100 * 0.1 = {red_sum.item():.15f} | bits: {hex(bits_sum[0])}")

# Test sum of large array
large_vec = torch.randn(10000, dtype=torch.float32, device=device)
large_sum = torch.sum(large_vec)

bits_large_sum = view_bits(large_sum)
print(f"Sum of 10000 random values = {large_sum.item():.15f} | bits: {hex(bits_large_sum[0])}")

# Test mean
red_mean = torch.mean(red_vec)
bits_mean = view_bits(red_mean)
print(f"Mean of 100 * 0.1 = {red_mean.item():.15f} | bits: {hex(bits_mean[0])}")

# ============================================================================
# Test 5: Matrix Multiplication (single element for bit comparison)
# ============================================================================
print("\n" + "=" * 70)
print("Test 5: Matrix Multiplication (Small Scale)")
print("=" * 70)

# Simple 2x2 matrices for exact bit comparison
A = torch.tensor([[1.5, 2.5],
                  [3.5, 4.5]], dtype=torch.float32, device=device)
B = torch.tensor([[0.5, 1.5],
                  [2.5, 3.5]], dtype=torch.float32, device=device)

C = torch.mm(A, B)

print(f"A = [[1.5, 2.5],")
print(f"     [3.5, 4.5]]")
print(f"B = [[0.5, 1.5],")
print(f"     [2.5, 3.5]]")
print(f"\nA @ B = [[{C[0,0].item():.10f}, {C[0,1].item():.10f}],")
print(f"        [{C[1,0].item():.10f}, {C[1,1].item():.10f}]]")

bits_C = view_bits(C)
print(f"\nBit pattern (row-major):")
for i in range(4):
    idx = i // 2 * 2 + i % 2
    print(f"  C[{i//2}, {i%2}] = {hex(bits_C[idx])}")

# ============================================================================
# Test 6: Tensor Operations (sqrt, exp, sin, etc.)
# ============================================================================
print("\n" + "=" * 70)
print("Test 6: Mathematical Functions")
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

math_results = []
for name, func, val in math_tests:
    x = torch.tensor(val, dtype=torch.float32, device=device)
    y = func(x)
    bits_y = view_bits(y)
    math_results.append((name, hex(bits_y[0]), y.item()))
    print(f"{name:15s} | bits: {hex(bits_y[0]):10s} | value: {y.item():.15f}")

# ============================================================================
# Test 7: Comparison Operations
# ============================================================================
print("\n" + "=" * 70)
print("Test 7: Comparison and Bitwise Operations")
print("=" * 70)

a = torch.tensor(1.5, dtype=torch.float32, device=device)
b = torch.tensor(1.5, dtype=torch.float32, device=device)

# Check equality
eq_result = torch.equal(a, b)
print(f"a = 1.5, b = 1.5")
print(f"torch.equal(a, b) = {eq_result}")

# Check bit-level equality
bits_a = view_bits(a)
bits_b = view_bits(b)
print(f"a bits: {hex(bits_a[0])}, b bits: {hex(bits_b[0])}")
print(f"Bit-level equal: {bits_a[0] == bits_b[0]}")

# ============================================================================
# Test 8: In-place operations
# ============================================================================
print("\n" + "=" * 70)
print("Test 8: In-place Operations")
print("=" * 70)

a = torch.tensor(1.5, dtype=torch.float32, device=device)
b = torch.tensor(2.5, dtype=torch.float32, device=device)

a.add_(b)
bits_a = view_bits(a)
print(f"a = 1.5, b = 2.5")
print(f"a.add_(b) = {a.item():.15f} | bits: {hex(bits_a[0])}")

a = torch.tensor(5.0, dtype=torch.float32, device=device)
a.mul_(b)
bits_a = view_bits(a)
print(f"a = 5.0, b = 2.5")
print(f"a.mul_(b) = {a.item():.15f} | bits: {hex(bits_a[0])}")

# ============================================================================
# Test 9: Special values on device
# ============================================================================
print("\n" + "=" * 70)
print("Test 9: Special Values (Inf, NaN, Zero) on Device")
print("=" * 70)

try:
    inf_result = torch.tensor(1.0, dtype=torch.float32, device=device) / torch.tensor(0.0, dtype=torch.float32, device=device)
    bits_inf = view_bits(inf_result)
    print(f"1.0 / 0.0  = {inf_result.item():.5f}  | bits: {hex(bits_inf[0])}")

    neg_inf_result = torch.tensor(-1.0, dtype=torch.float32, device=device) / torch.tensor(0.0, dtype=torch.float32, device=device)
    bits_neg_inf = view_bits(neg_inf_result)
    print(f"-1.0 / 0.0 = {neg_inf_result.item():.5f} | bits: {hex(bits_neg_inf[0])}")

    nan_result = torch.tensor(0.0, dtype=torch.float32, device=device) / torch.tensor(0.0, dtype=torch.float32, device=device)
    bits_nan = view_bits(nan_result)
    print(f"0.0 / 0.0  = nan           | bits: {hex(bits_nan[0])}")

    neg_zero = -torch.tensor(0.0, dtype=torch.float32, device=device)
    pos_zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    bits_neg_zero = view_bits(neg_zero)
    bits_pos_zero = view_bits(pos_zero)
    print(f"+0.0       | bits: {hex(bits_pos_zero[0])}")
    print(f"-0.0       | bits: {hex(bits_neg_zero[0])}")
except Exception as e:
    print(f"Error testing special values: {e}")

# ============================================================================
# Test 10: JIT compiled kernel (TorchScript)
# ============================================================================
print("\n" + "=" * 70)
print("Test 10: JIT Compiled Kernel (TorchScript)")
print("=" * 70)

@torch.jit.script
def add_mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x + y) * x

a = torch.tensor(2.5, dtype=torch.float32, device=device)
b = torch.tensor(1.5, dtype=torch.float32, device=device)
c = add_mul_kernel(a, b)

bits_c = view_bits(c)
print(f"(a + b) * a where a=2.5, b=1.5")
print(f"JIT result = {c.item():.15f} | bits: {hex(bits_c[0])}")

# ============================================================================
# Test 11: Low precision (float16, bfloat16)
# ============================================================================
print("\n" + "=" * 70)
print("Test 11: Low Precision Types (float16, bfloat16)")
print("=" * 70)

# Test float16
a_f16 = torch.tensor(1.5, dtype=torch.float16, device=device)
b_f16 = torch.tensor(2.25, dtype=torch.float16, device=device)
c_f16 = a_f16 + b_f16

print(f"float16: 1.5 + 2.25 = {c_f16.item():.5f}")

# Convert to float32 to see the bits
c_f16_as_f32 = c_f16.float()
bits_f16_result = view_bits(c_f16_as_f32)
print(f"  As float32: {c_f16_as_f32.item():.15f} | bits: {hex(bits_f16_result[0])}")

# Test bfloat16
try:
    a_bf16 = torch.tensor(1.5, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.tensor(2.25, dtype=torch.bfloat16, device=device)
    c_bf16 = a_bf16 + b_bf16

    print(f"bfloat16: 1.5 + 2.25 = {c_bf16.item():.5f}")

    c_bf16_as_f32 = c_bf16.float()
    bits_bf16_result = view_bits(c_bf16_as_f32)
    print(f"  As float32: {c_bf16_as_f32.item():.15f} | bits: {hex(bits_bf16_result[0])}")
except Exception as e:
    print(f"bfloat16 not supported or error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Device Type: {device_type}")
print(f"All basic arithmetic operations completed successfully")
print("=" * 70)
