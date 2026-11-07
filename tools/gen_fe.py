"""Generate the Fe source of the `model` ingot from a Tensile model file.

Usage:
    python tools/gen_fe.py [--model ref/out/model.json] [--out ingots/model/src]
                           [--vectors ref/out] [--samples 3]

Writes one `layerK.fe` per layer (packed weights, bias, the overflow proof, the typed layer
function and the single-neuron functions a contract calls), `lib.fe` (the network as a chain of
layers), `commitment.fe` (the Merkle root of the neuron tree, from tools/gen_tree.py, with a few
leaves and proofs for the tests) and, with --vectors, `reference.fe` (the first samples of the
reference test vectors as public constants, and unit tests on them).

The output is a pure function of the input files: no timestamps, fixed order, LF line endings, and
laid out simply: `tools/gen_model.sh` runs this script and then `fe fmt`, which gives the files
their final layout. Standard library only, so it runs under any Python 3.

The generator checks what it needs for its own work: value ranges and array lengths (it packs the
values) and that every shift leaves a scale of zero or more (it writes the scales into the types).
It does NOT check that the layers fit together, or that the recorded worst case is right: every
layer is written from its own fields, and the Fe compiler rejects a network whose layers do not
chain, whose scales disagree with the kernel, or whose accumulators could overflow.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gen_tree import commitment  # noqa: E402

# Layers with at most this many weights also carry their int8 matrix, and the compiler packs it and
# compares it with the packed literal. One compile-time evaluation is limited to 1,000,000 steps:
# measured on Fe 26.5.1, the comparison fits for 9,408 weights and not for 12,544
# (docs/fe-notes.md).
PACK_CHECK_MAX_WEIGHTS = 8192

INDENT = "    "


class ModelError(ValueError):
    pass


def pack_words(values):
    """Pack int8 values 32 per word: value i is byte i % 32 of word i // 32, most significant
    byte first, two's complement, the last word padded with zero bytes."""
    words = []
    for start in range(0, len(values), 32):
        chunk = bytes(v & 0xFF for v in values[start:start + 32])
        words.append("0x" + chunk.ljust(32, b"\0").hex())
    return words


def check_layer(index, layer):
    n_in, n_out = layer["in"], layer["out"]
    weights, bias = layer["weights"], layer["bias"]
    if len(weights) != n_out or any(len(row) != n_in for row in weights):
        raise ModelError("layer %d: weights are not %d rows of %d" % (index, n_out, n_in))
    if len(bias) != n_out:
        raise ModelError("layer %d: bias does not have %d values" % (index, n_out))
    if any(not -127 <= w <= 127 for row in weights for w in row):
        raise ModelError("layer %d: a weight is outside -127..127" % index)
    if any(not -2**31 <= b < 2**31 for b in bias):
        raise ModelError("layer %d: a bias is not an int32" % index)


def array(items):
    return "[" + ", ".join(items) + "]"


def rows(items):
    """An array with one item per line (the rows of a matrix)."""
    return "[\n" + "".join(INDENT + item + ",\n" for item in items) + "]"


def layer_source(model, index, layer, f_in):
    """Source of layerK.fe. `f_in` is the number of fractional bits of the layer's input."""
    n_in, n_out, fw = layer["in"], layer["out"], layer["weight_frac_bits"]
    shift = layer["shift"]
    f_acc = f_in + fw
    if shift is not None and not 0 <= shift <= min(f_acc, 31):
        raise ModelError("layer %d: shift %d is outside 0..%d, the fractional bits of its "
                         "accumulator (an int32)" % (index, shift, min(f_acc, 31)))
    weights, bias = layer["weights"], layer["bias"]
    pack_check = n_in * n_out <= PACK_CHECK_MAX_WEIGHTS

    in_type = "Activations<%d, %d>" % (n_in, f_in)
    row_type = "Row<%d, %d>" % (n_in, fw)
    acc_type = "Q<%d>" % f_acc
    words_type = "[u256; %d]" % ((n_in + 31) // 32)
    if shift is None:
        out_type = "Accumulators<%d, %d>" % (n_out, f_acc)
        imports = ["Accumulators", "Activations"]
        body = "dense(weights: WEIGHTS, x: x, bias: BIAS)"
        doc = ("/// Layer %d, the last one: its accumulators are the logits, with %d fractional bits."
               % (index, f_acc))
        neuron_type = "i32"
        neuron_body = "tensile::neuron(row, x, bias).raw()"
        neuron_doc = "the logit"
    else:
        out_type = "Activations<%d, %d>" % (n_out, f_acc - shift)
        imports = ["Activations"]
        body = "dense(weights: WEIGHTS, x: x, bias: BIAS).requantise<%d>()" % shift
        doc = ("/// Layer %d: accumulators with %d fractional bits, requantised by %d bits and clamped\n"
               "/// to 0..127." % (index, f_acc, shift))
        neuron_type = "i8"
        neuron_body = "tensile::neuron(row, x, bias).shifted<%d>().clamped()" % shift
        neuron_doc = "the activation"
    imports += ["Bias", "I32_MAX", "Q", "Row", "Weights", "dense", "worst_case"]

    packed_rows = ["Row::from_words(%s)" % array(pack_words(row)) for row in weights]
    out = [
        "//! GENERATED by tools/gen_fe.py from the model \"%s\". Do not edit." % model["name"],
        "//!",
        "//! Layer %d: %d inputs, %d outputs, weights with %d fractional bits." % (index, n_in, n_out, fw),
        "use tensile::{%s}" % ", ".join(sorted(imports, key=str.lower)),
        "",
        "/// The weights, 32 per word (see `tensile::Packed`).",
        "pub const WEIGHTS: Weights<%d, %d, %d> = Weights::new(%s)" % (
            n_out, n_in, fw, rows(packed_rows)),
        "",
        "/// The biases, at the scale of a weight times an input.",
        "pub const BIAS: Bias<%d, %d> = Bias::from_raw(%s)" % (
            n_out, f_acc, array([str(b) for b in bias])),
        "",
        "/// The largest absolute accumulator over every input in 0..127, as recorded in the model",
        "/// file.",
        "pub const WORST_CASE: u256 = %d" % layer["worst_case_abs_acc"],
        "",
        "// The overflow proof: the compiler recomputes the bound from the weights and biases above,",
        "// requires the recorded value, and requires that it fits an int32.",
        "static_assert(worst_case(weights: WEIGHTS, bias: BIAS) == WORST_CASE)",
        "static_assert(WORST_CASE <= I32_MAX)",
        "",
    ]
    if pack_check:
        int8_rows = [array([str(w) for w in row]) for row in weights]
        out += [
            "/// The same weights as plain int8 values.",
            "const INT8: [[i8; %d]; %d] = %s" % (n_in, n_out, rows(int8_rows)),
            "",
            "// The compiler packs `INT8` itself and requires the packed literal above.",
            "static_assert(WEIGHTS.is_packed_from(INT8))",
            "",
        ]
    out += [
        doc,
        "pub fn apply(x: %s) -> %s {" % (in_type, out_type),
        INDENT + body,
        "}",
        "",
        "/// Output `j` of this layer alone, %s of one neuron, from row `j` of the weights and" % neuron_doc,
        "/// bias `j`. It is `tensile::neuron`, the function `apply` runs once per row.",
        "pub fn neuron(row: %s, x: %s, bias: %s) -> %s {" % (row_type, in_type, acc_type, neuron_type),
        INDENT + neuron_body,
        "}",
        "",
        "/// `neuron` on packed words, the form a contract receives (`docs/spec.md` section 7). Fails",
        "/// if the row or the input is malformed: a padding byte that is not zero, or an input",
        "/// outside 0..127.",
        "pub fn neuron_from_words(row: %s, x: %s, bias: i32) -> %s {" % (words_type, words_type, neuron_type),
        INDENT + "neuron(row: Row::from_words(row), x: Activations::from_words(x), bias: Q::from_raw(bias))",
        "}",
        "",
        "/// Row `j` of the weights.",
        "pub const fn row(_ j: usize) -> %s {" % row_type,
        INDENT + "WEIGHTS.row(j)",
        "}",
        "",
        "/// Bias `j`.",
        "pub const fn bias(_ j: usize) -> %s {" % acc_type,
        INDENT + "BIAS.at(j)",
        "}",
        "",
    ]
    return "\n".join(out), in_type, out_type, (None if shift is None else f_acc - shift)


def lib_source(model, signatures):
    """Source of lib.fe. `signatures` holds (input type, output type) per layer."""
    count = len(signatures)
    sizes = " -> ".join([str(model["layers"][0]["in"])] + [str(lay["out"]) for lay in model["layers"]])
    out = [
        "//! GENERATED by tools/gen_fe.py from the model \"%s\". Do not edit." % model["name"],
        "//!",
        "//! The network %s as a chain of typed layers. Each `layerK` module states the type of" % sizes,
        "//! its own input and output, so layers whose sizes or scales do not fit together do not",
        "//! compile.",
        "use tensile::{Accumulators, Activations, Blind, Probe}",
        "",
        "/// The input of the network.",
        "pub type Input = %s" % signatures[0][0],
        "",
        "/// The logits: the accumulators of the last layer.",
        "pub type Logits = %s" % signatures[-1][1],
        "",
        "/// The forward pass. The probe is shown the output of every hidden layer, in order.",
        "pub fn forward(x: Input) -> Logits uses (probe: mut Probe) {",
    ]
    previous = "x"
    for k in range(1, count):
        out.append(INDENT + "let hidden%d = layer%d::apply(x: %s)" % (k, k, previous))
        out.append(INDENT + "probe.hidden(layer: %d, hidden%d)" % (k, k))
        previous = "hidden%d" % k
    out += [
        INDENT + "layer%d::apply(x: %s)" % (count, previous),
        "}",
        "",
        "/// The logits, with nothing observing the hidden layers.",
        "pub fn logits(x: Input) -> Logits {",
        INDENT + "var blind = Blind {}",
        INDENT + "with (Probe = blind) {",
        INDENT * 2 + "forward(x: x)",
        INDENT + "}",
        "}",
        "",
        "/// The predicted class: the index of the largest logit, the lowest index winning a tie.",
        "pub fn classify(x: Input) -> usize {",
        INDENT + "logits(x: x).argmax()",
        "}",
        "",
    ]
    return "\n".join(out)


def reference_source(model, signatures, vectors_dir, samples):
    """Source of reference.fe: unit tests from the first `samples` reference test vectors."""
    n_in = model["layers"][0]["in"]
    count = len(signatures)
    with open(os.path.join(vectors_dir, "inputs.bin"), "rb") as f:
        inputs = f.read(n_in * samples)
    with open(os.path.join(vectors_dir, "trace.jsonl"), "r", encoding="utf-8") as f:
        traces = [json.loads(f.readline()) for _ in range(samples)]
    if len(inputs) != n_in * samples:
        raise ModelError("inputs.bin holds fewer than %d samples" % samples)

    sizes = [layer["out"] for layer in model["layers"]]
    out = [
        "//! GENERATED by tools/gen_fe.py from the model \"%s\". Do not edit." % model["name"],
        "//!",
        "//! The first %d reference test vectors (`ref/out/inputs.bin`, `ref/out/trace.jsonl`): constants" % samples,
        "//! for the tests of other ingots, and unit tests in which every hidden activation, every logit",
        "//! and the class must equal the NumPy reference, layer by layer and neuron by neuron.",
        "use ingot::{Input, Logits, classify, %s}" % ", ".join("layer%d" % k for k in range(1, count + 1)),
        "use tensile::{Accumulators, Activations}",
        "",
        "/// Number of reference samples in this module: `X0`, `X1` and so on.",
        "pub const SAMPLES: usize = %d" % samples,
        "",
    ]
    for n, trace in enumerate(traces):
        if trace["i"] != n:
            raise ModelError("trace.jsonl line %d is sample %d" % (n, trace["i"]))
        x = [b if b < 128 else b - 256 for b in inputs[n * n_in:(n + 1) * n_in]]
        out.append("/// The input of reference sample %d." % n)
        out.append("pub const X%d: Input = Activations::from_words(%s)" % (n, array(pack_words(x))))
        for k, hidden in enumerate(trace["hidden"], start=1):
            out.append("/// The output of layer %d for sample %d." % (k, n))
            out.append("pub const HIDDEN%d_%d: %s = Activations::pack(%s)" % (
                k, n, signatures[k - 1][1], array([str(v) for v in hidden])))
        out.append("/// The logits of sample %d." % n)
        out.append("pub const LOGITS%d: Logits = Accumulators::from_raw(%s)" % (
            n, array([str(v) for v in trace["logits"]])))
        out.append("/// The class of sample %d." % n)
        out.append("pub const CLASS%d: usize = %d" % (n, trace["class"]))
        out += ["", "#[test]", "fn sample_%d_matches_the_reference() {" % n]
        previous = "X%d" % n
        for k in range(1, count):
            out.append(INDENT + "assert!(layer%d::apply(x: %s) == HIDDEN%d_%d)" % (k, previous, k, n))
            previous = "HIDDEN%d_%d" % (k, n)
        out.append(INDENT + "assert!(layer%d::apply(x: %s) == LOGITS%d)" % (count, previous, n))
        out.append(INDENT + "assert!(classify(x: X%d) == CLASS%d)" % (n, n))
        out += ["}", "", "#[test]", "fn sample_%d_matches_neuron_by_neuron() {" % n]
        previous = "X%d" % n
        for k in range(1, count + 1):
            if k < count:
                expected = "HIDDEN%d_%d.get(j)" % (k, n)
            else:
                expected = "LOGITS%d.at(j).raw()" % n
            out += [
                INDENT + "for j in 0 .. %d {" % sizes[k - 1],
                INDENT * 2 + "let got = layer%d::neuron_from_words(row: layer%d::row(j).words(), x: %s.words(), "
                "bias: layer%d::bias(j).raw())" % (k, k, previous, k),
                INDENT * 2 + "assert!(got == %s)" % expected,
                INDENT + "}",
            ]
            previous = "HIDDEN%d_%d" % (k, n)
        out += ["}", ""]
    return "\n".join(out)


def commitment_source(model):
    """Source of commitment.fe: the Merkle root of the neuron tree (docs/spec-dispute.md section
    2) and, for the first and last neuron of every layer, the leaf and its proof, so that a test
    can recompute the leaf from the layer's row and bias and verify it against the root."""
    tree = commitment(model)
    by_position = {(x["layer"], x["neuron"]): x for x in tree["leaves"]}
    out = [
        "//! GENERATED by tools/gen_fe.py from the model \"%s\". Do not edit." % model["name"],
        "//!",
        "//! The model commitment: the root of the Merkle tree with one leaf per neuron",
        "//! (`docs/spec-dispute.md` section 2, built by `tools/gen_tree.py`), and the leaf and proof of",
        "//! the first and last neuron of every layer for the tests that recompute them.",
        "",
        "/// The root of the neuron tree: what the arbiter is deployed with.",
        "pub const MODEL_ROOT: u256 = %s" % tree["root"],
        "",
        "/// Number of leaves (neurons) in the tree.",
        "pub const LEAF_COUNT: usize = %d" % tree["leaf_count"],
        "",
    ]
    for index, layer in enumerate(model["layers"], start=1):
        for j in sorted({0, layer["out"] - 1}):
            entry = by_position[(index, j)]
            out += [
                "/// The leaf of layer %d, neuron %d." % (index, j),
                "pub const LEAF_%d_%d: u256 = %s" % (index, j, entry["leaf"]),
                "/// The proof of that leaf: siblings from the leaf's level up to the child of the root.",
                "pub const PROOF_%d_%d: [u256; %d] = %s" % (
                    index, j, len(entry["proof"]), array(entry["proof"])),
                "",
            ]
    return "\n".join(out)


def generate(model, vectors_dir=None, samples=3):
    """Return {file name: source} for the model ingot's src directory."""
    files = {"commitment.fe": commitment_source(model)}
    signatures = []
    f_in = model["input_frac_bits"]
    for index, layer in enumerate(model["layers"], start=1):
        check_layer(index, layer)
        if (layer["shift"] is None) != (index == len(model["layers"])):
            raise ModelError("layer %d: only the last layer has a null shift" % index)
        source, in_type, out_type, f_out = layer_source(model, index, layer, f_in)
        files["layer%d.fe" % index] = source
        signatures.append((in_type, out_type))
        f_in = f_out
    files["lib.fe"] = lib_source(model, signatures)
    if vectors_dir is not None and samples > 0:
        files["reference.fe"] = reference_source(model, signatures, vectors_dir, samples)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.path.join(ROOT, "ref", "out", "model.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "ingots", "model", "src"))
    parser.add_argument("--vectors", default=None,
                        help="directory with inputs.bin and trace.jsonl, to generate reference.fe")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        model = json.load(f)
    try:
        files = generate(model, args.vectors, args.samples)
    except ModelError as error:
        print("gen_fe: %s" % error, file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    for name in sorted(os.listdir(args.out)):
        if name.endswith(".fe") and name not in files:
            os.remove(os.path.join(args.out, name))
    for name, source in sorted(files.items()):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(source)
        print("wrote %s (%d bytes)" % (os.path.join(args.out, name), len(source)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
