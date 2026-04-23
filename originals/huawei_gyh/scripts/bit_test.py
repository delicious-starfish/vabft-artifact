#!/usr/bin/env python3
"""
Bit-level precision test for Ascend 910B2 vs NVIDIA H100
Tests if + - * / operations produce IDENTICAL bit patterns
"""

import torch
import numpy as np

def view_bits(x):
    """View float32 as uint32 to compare exact bits"""
    return np.array(x.flatten().cpu().numpy()).view(np.uint32)

def test_basic_operations_bit_exact():
    """Test if basic arithmetic produces bit-identical results"""
    results = {
        'addition': [],
        'subtraction': [],
        'multiplication': [],
        'division': []
    }

    print("=" * 60)
    print("Bit-Level Precision Test")
    print("=" * 60)

    # Test 1: Simple addition
    print("\n[1] Addition: a + b")
    a = torch.tensor(1.5, dtype=torch.float32)
    b = torch.tensor(2.25, dtype=torch.float32)
    c = a + b

    bits_a = view_bits(a)
    bits_b = view_bits(b)
    bits_c = view_bits(c)

    print(f"  a = {a.item():.10f} | bits: {hex(bits_a[0])}")
    print(f"  b = {b.item():.10f} | bits: {hex(bits_b[0])}")
    print(f"  c = {c.item():.10f} | bits: {hex(bits_c[0])}")
    print(f"  Expected (float64): {(1.5 + 2.25):.10f}")

    results['addition'].append({
        'a': (float(a), hex(bits_a[0])),
        'b': (float(b), hex(bits_b[0])),
        'result': (float(c), hex(bits_c[0])),
        'expected': 1.5 + 2.25
    })

    # Test 2: Subtraction
    print("\n[2] Subtraction: a - b")
    a = torch.tensor(5.0, dtype=torch.float32)
    b = torch.tensor(1.25, dtype=torch.float32)
    c = a - b

    bits_a = view_bits(a)
    bits_b = view_bits(b)
    bits_c = view_bits(c)

    print(f"  a = {a.item():.10f} | bits: {hex(bits_a[0])}")
    print(f"  b = {b.item():.10f} | bits: {hex(bits_b[0])}")
    print(f"  c = {c.item():.10f} | bits: {hex(bits_c[0])}")
    print(f"  Expected (float64): {(5.0 - 1.25):.10f}")

    results['subtraction'].append({
        'a': (float(a), hex(bits_a[0])),
        'b': (float(b), hex(bits_b[0])),
        'result': (float(c), hex(bits_c[0])),
        'expected': 5.0 - 1.25
    })

    # Test 3: Multiplication
    print("\n[3] Multiplication: a * b")
    a = torch.tensor(3.5, dtype=torch.float32)
    b = torch.tensor(2.5, dtype=torch.float32)
    c = a * b

    bits_a = view_bits(a)
    bits_b = view_bits(b)
    bits_c = view_bits(c)

    print(f"  a = {a.item():.10f} | bits: {hex(bits_a[0])}")
    print(f"  b = {b.item():.10f} | bits: {hex(bits_b[0])}")
    print(f"  c = {c.item():.10f} | bits: {hex(bits_c[0])}")
    print(f"  Expected (float64): {(3.5 * 2.5):.10f}")

    results['multiplication'].append({
        'a': (float(a), hex(bits_a[0])),
        'b': (float(b), hex(bits_b[0])),
        'result': (float(c), hex(bits_c[0])),
        'expected': 3.5 * 2.5
    })

    # Test 4: Division
    print("\n[4] Division: a / b")
    a = torch.tensor(7.0, dtype=torch.float32)
    b = torch.tensor(3.0, dtype=torch.float32)
    c = a / b

    bits_a = view_bits(a)
    bits_b = view_bits(b)
    bits_c = view_bits(c)

    print(f"  a = {a.item():.10f} | bits: {hex(bits_a[0])}")
    print(f"  b = {b.item():.10f} | bits: {hex(bits_b[0])}")
    print(f"  c = {c.item():.10f} | bits: {hex(bits_c[0])}")
    print(f"  Expected (float64): {(7.0 / 3.0):.15f}")

    results['division'].append({
        'a': (float(a), hex(bits_a[0])),
        'b': (float(b), hex(bits_b[0])),
        'result': (float(c), hex(bits_c[0])),
        'expected': 7.0 / 3.0
    })

    # Test 5: More comprehensive tests
    print("\n" + "=" * 60)
    print("[5] Comprehensive Bit-Exact Test Suite")
    print("=" * 60)

    test_cases = [
        ("Add: 0.1 + 0.2", 0.1, 0.2, lambda a, b: a + b),
        ("Add: 1.0 + 1.0", 1.0, 1.0, lambda a, b: a + b),
        ("Sub: 1.0 - 0.9", 1.0, 0.9, lambda a, b: a - b),
        ("Mul: 0.1 * 0.1", 0.1, 0.1, lambda a, b: a * b),
        ("Mul: 1.1 * 1.1", 1.1, 1.1, lambda a, b: a * b),
        ("Div: 1.0 / 3.0", 1.0, 3.0, lambda a, b: a / b),
        ("Div: 10.0 / 7.0", 10.0, 7.0, lambda a, b: a / b),
        ("Add: -1.5 + 2.5", -1.5, 2.5, lambda a, b: a + b),
        ("Mul: -2.5 * 3.0", -2.5, 3.0, lambda a, b: a * b),
        ("Div: -1.0 / 2.0", -1.0, 2.0, lambda a, b: a / b),
    ]

    all_results = []

    for name, a_val, b_val, op in test_cases:
        a = torch.tensor(a_val, dtype=torch.float32)
        b = torch.tensor(b_val, dtype=torch.float32)
        c = op(a, b)

        bits_a = view_bits(a)
        bits_b = view_bits(b)
        bits_c = view_bits(c)

        # float64 reference
        ref = op(torch.tensor(a_val, dtype=torch.float64),
                 torch.tensor(b_val, dtype=torch.float64))

        print(f"\n{name}")
        print(f"  a = {a.item():.15f} | bits: {hex(bits_a[0])}")
        print(f"  b = {b.item():.15f} | bits: {hex(bits_b[0])}")
        print(f"  result = {c.item():.15f} | bits: {hex(bits_c[0])}")
        print(f"  float64 ref = {ref.item():.15f}")

        all_results.append({
            'name': name,
            'a_bits': hex(bits_a[0]),
            'b_bits': hex(bits_b[0]),
            'result_bits': hex(bits_c[0]),
            'result_value': float(c),
            'float64_ref': float(ref)
        })

    # Test 6: Fused multiply-add
    print("\n" + "=" * 60)
    print("[6] FMA (Fused Multiply-Add): a * b + c")
    print("=" * 60)

    a, b, c = torch.tensor(2.5, dtype=torch.float32), torch.tensor(3.5, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)

    # Separate operations
    d_separate = (a * b) + c

    # FMA operation (if available)
    try:
        d_fma = torch.fmadd(a, b, c)
        bits_fma = view_bits(d_fma)
        has_fmadd = True
    except AttributeError:
        d_fma = None
        has_fmadd = False

    bits_separate = view_bits(d_separate)

    print(f"  a = {a.item():.15f}")
    print(f"  b = {b.item():.15f}")
    print(f"  c = {c.item():.15f}")
    print(f"\n  (a * b) + c = {d_separate.item():.15f} | bits: {hex(bits_separate[0])}")
    if has_fmadd:
        print(f"  fma(a, b, c) = {d_fma.item():.15f} | bits: {hex(bits_fma[0])}")
        print(f"  Same bits? {bits_separate[0] == bits_fma[0]}")
    else:
        print(f"  fmadd not available in this PyTorch version")

    # Test 7: Special values
    print("\n" + "=" * 60)
    print("[7] Special Values (Inf, NaN, Zero)")
    print("=" * 60)

    # Division by zero
    inf_result = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(0.0, dtype=torch.float32)
    bits_inf = view_bits(inf_result)
    print(f"  1.0 / 0.0 = {inf_result.item()} | bits: {hex(bits_inf[0])}")

    # 0 / 0 = NaN
    nan_result = torch.tensor(0.0, dtype=torch.float32) / torch.tensor(0.0, dtype=torch.float32)
    bits_nan = view_bits(nan_result)
    print(f"  0.0 / 0.0 = {nan_result.item()} | bits: {hex(bits_nan[0])}")

    # Negative zero
    neg_zero = -torch.tensor(0.0, dtype=torch.float32)
    bits_neg_zero = view_bits(neg_zero)
    pos_zero = torch.tensor(0.0, dtype=torch.float32)
    bits_pos_zero = view_bits(pos_zero)
    print(f"  +0.0 bits: {hex(bits_pos_zero[0])}")
    print(f"  -0.0 bits: {hex(bits_neg_zero[0])}")

    # Platform info
    print("\n" + "=" * 60)
    print("Platform Information")
    print("=" * 60)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device_str()}")

    return results, all_results

def device_str():
    """Get current device string"""
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        return f"NPU ({torch.npu.get_device_name(0)})"
    else:
        return "CPU"

if __name__ == "__main__":
    results, all_results = test_basic_operations_bit_exact()
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
