# Make a prefill-model GGUF for --prefill-model: a quantized GGUF with selected weights (default: ffn_gate and
# ffn_up) replaced by their fp16 copies from an F16 file. Every other tensor and all metadata are copied byte for
# byte from the quantized file. Make the F16 file as the exact dequantization of the same quantized model:
#   llama-quantize --allow-requantize model-Q4_0.gguf model-deq-F16.gguf F16
# usage: make-prefill-gguf.py <q4_0.gguf> <deq-f16.gguf> <out.gguf> [tensor-name-regex]
# needs gguf-py (pip install gguf, or set LLAMA_SRC to a llama.cpp tree)
import os, re, sys
if os.environ.get('LLAMA_SRC'):
    sys.path.insert(0, os.path.join(os.environ['LLAMA_SRC'], 'gguf-py'))
import gguf
src, f16, out = sys.argv[1:4]
pat = re.compile(sys.argv[4] if len(sys.argv) > 4 else r'^blk\.\d+\.ffn_(gate|up)\.weight$')
rq = gguf.GGUFReader(src, 'r'); rf = gguf.GGUFReader(f16, 'r')
ft = {t.name: t for t in rf.tensors}
arch = rq.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = gguf.GGUFWriter(out, arch=arch, endianess=rq.endianess)
for field in rq.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]; st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(field.name, field.contents(), vt, sub_type=st)
sel = []
for t in rq.tensors:
    if pat.match(t.name):
        g = ft[t.name]
        assert g.tensor_type == gguf.GGMLQuantizationType.F16, (t.name, g.tensor_type)
        sel.append(g)
    else:
        sel.append(t)
n_rep = sum(1 for a, b in zip(rq.tensors, sel) if a is not b)
print(f'replacing {n_rep} tensors with fp16')
for t in sel:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in sel:
    w.write_tensor_data(t.data, tensor_endianess=rq.endianess)
w.close()
print('done', out)
