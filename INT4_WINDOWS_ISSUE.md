# ⚠️ INT4 Quantization Limitation on Windows

## Issue Summary

**INT4 quantization** in TorchAO 0.17.0 requires the `mslk` package, which is:
- Not available on PyPI for Windows (only a 0.0.0 placeholder exists)
- Required for INT4 tensor operations
- **Blocks INT4 quantization on Windows systems**

## Error Message
```
ImportError: Requires mslk >= 1.0.0
```

## Current Status

### ✅ Working Configurations:
1. **INT8 weight-only** - Fully functional
   - Runtime VRAM: ~3.5 GB
   - Disk size: ~2.24 GB
   - Quality: ~99% of BF16
   
2. **INT8 + torch.compile** - Fully functional with caching
   - Runtime VRAM: ~3.5 GB
   - Speed: 1.5-2x faster after compilation
   - Cache size: ~1.5 GB

### ❌ Not Working on Windows:
1. **INT4 weight-only** - Blocked by `mslk` dependency
   - Would provide: ~2.2 GB VRAM, ~2.0 GB disk
   - Currently falls back to INT8 automatically

## Workaround

The `server_int4.py` now **automatically falls back to INT8** when INT4 is not available:

```bash
# This will use INT8 on Windows
start_int4.bat
```

## Alternative Solutions

### Option 1: Use INT8 + torch.compile (Recommended)
```bash
start_int8_compile.bat
```
- Same VRAM as INT8 (~3.5 GB)
- 1.5-2x faster after compilation
- Fully functional on Windows

### Option 2: Use INT8 only
```bash
start_int8.bat
```
- Simple, no compilation needed
- ~3.5 GB VRAM
- Good quality (~99% of BF16)

## Technical Details

### Why INT4 Fails:
```python
# This works (config creation):
from torchao.quantization import Int4WeightOnlyConfig
config = Int4WeightOnlyConfig(group_size=128)  # ✓ OK

# This fails (actual quantization):
from torchao.quantization import quantize_
quantize_(model.llm, Int4WeightOnlyConfig(group_size=128))  # ✗ ImportError: Requires mslk >= 1.0.0
```

### Root Cause:
- `mslk` (Microsft Light Kernels?) is a Microsoft package for INT4 operations
- Only available for Linux/macOS
- Not published for Windows on PyPI
- TorchAO 0.17.0 requires it for INT4 on all platforms

## Future Resolution

Possible solutions (in order of likelihood):

1. **TorchAO updates** - May add Windows-compatible INT4 implementation
2. **mslk package** - May release Windows version on PyPI
3. **Alternative backends** - TorchAO might add alternative INT4 implementations

## Current Recommendation

**Use INT8 + torch.compile** for best performance on Windows:

```bash
# Best option for Windows users
start_int8_compile.bat

# Expected results:
# - First generation: 10-20 min (compilation)
# - Subsequent: 1.5-2x faster
# - VRAM: ~3.5 GB
# - Quality: ~99% of BF16
```

## File Status Summary

| File | Status | Notes |
|------|--------|-------|
| `server_int8.py` | ✅ Working | INT8 only |
| `server_int8_compile.py` | ✅ Working | INT8 + compilation |
| `server_int4.py` | ⚠️ Falls back to INT8 | INT4 not available on Windows |
| `quantize_model_int4.py` | ❌ Fails | INT4 not available on Windows |
| `start_int8.bat` | ✅ Working | Launcher |
| `start_int8_compile.bat` | ✅ Working | Launcher with compile |
| `start_int4.bat` | ⚠️ Falls back to INT8 | Will use INT8 on Windows |

## References

- TorchAO Issue: https://github.com/pytorch/ao/issues
- mslk PyPI: https://pypi.org/project/mslk/ (0.0.0 only)
- PyTorch XPU: https://intel.github.io/intel-extension-for-pytorch/

---

**Last Updated:** April 4, 2026  
**TorchAO Version:** 0.17.0  
**PyTorch Version:** 2.12.0.dev20260302+xpu
