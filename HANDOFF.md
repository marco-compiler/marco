# Native Modelica frontend: status and handoff

Work in progress on a native frontend that flattens Modelica models without
OpenModelica. Nothing below was committed as of 2026-11-08.

## Pipeline

```
.mo --(Parser/Modelica)--> AST --(Lowering/Modelica)--> modelica dialect (class tree)
    --modelica-instantiate--> instance tree
    --modelica-expand-connections--> connection equations
    --modelica-flatten--> flat model
    --convert-modelica-to-bmodelica--> bmodelica (existing backend)
```

The driver uses it with `--modelica-frontend=native` (default: `omc`).
`-emit-mlir-modelica` now emits the new dialect after flattening.
`-emit-mlir-bmodelica` emits bmodelica, which is what `-emit-mlir-modelica` used
to do; the tests were renamed accordingly.

## Where things are

| Component | Headers | Sources | Tests |
|---|---|---|---|
| AST | `include/marco/AST/Modelica/` | `lib/AST/Modelica/` | |
| Parser | `include/marco/Parser/Modelica/` | `lib/Parser/Modelica/` | `unittest/Parser/Modelica/` |
| Lowering | `include/marco/Codegen/Lowering/Modelica/` | `lib/Codegen/Lowering/Modelica/` | `unittest/Codegen/Lowering/Modelica/` |
| Dialect | `include/marco/Dialect/Modelica/IR/` | `lib/Dialect/Modelica/IR/` | `test/Dialect/Modelica/IR/` |
| Passes | `include/marco/Dialect/Modelica/Transforms/` | `lib/Dialect/Modelica/Transforms/` | `test/Dialect/Modelica/Transforms/` |
| Conversion | `include/marco/Codegen/Conversion/ModelicaToBaseModelica/` | `lib/Codegen/Conversion/ModelicaToBaseModelica/` | `test/Codegen/Conversion/ModelicaToBaseModelica/` |

Driver and frontend wiring: `lib/Frontend/` (`CompilerInvocation.cpp`,
`FrontendActions.cpp`), `tool/marco-opt/marco-opt.cpp`. End-to-end tests:
`test/Driver/`, `test/Frontend/Actions/`, `test/Simulation/ida/native-frontend.mo`.

## LLVM fork changes (also uncommitted)

The driver flags live in the LLVM fork (`llvm-project`, branch
`mscuttari/shared`), not in this repository. Modified files:
`clang/include/clang/Driver/Options.td` (`-emit-mlir-bmodelica`,
`--modelica-frontend=`), `clang/include/clang/Driver/Types.def`,
`clang/lib/Driver/Types.cpp`, `clang/lib/Driver/Driver.cpp`,
`clang/lib/Driver/ToolChains/Marco.cpp`,
`clang/include/clang/Basic/DiagnosticDriverKinds.td`.

Commit them in the fork as well, then rebuild and reinstall LLVM. Until then,
`lib/Frontend/CompilerInvocation.cpp` doesn't compile (`OPT_emit_mlir_bmodelica`,
`OPT_modelica_frontend`), so `marco` and `marco-opt` don't link.
To run the tests before the rebuild, temporarily drop the `OPT_emit_mlir_bmodelica`
case and replace `args.getLastArg(options::OPT_modelica_frontend)` with
`nullptr`, then revert.

## Design decisions on the modelica dialect

- **Indexing is zero-based**: subscripts and the dimension operands of `size`
  and `cat`. The lowerer converts from the one-based source: literals and range
  bounds are shifted, anything else gets `- 1`, `end` lowers to the size.
  Boolean and enumeration indices select the position of the value
  (`false` before `true`, literals in declaration order); `enum_to_integer`
  keeps the one-based ordinal, as it's a value.
- **Type inference**: every op computing a value lists
  `InferTypeOpAdaptorWithIsCompatible`. Its `inferReturnTypes` at most calls a
  shared function in `TypeInference.cpp`; there is no common base op for it.
- **One `modelica.ref`** with a `SymbolRefAttr` name: a simple name (`@x`) is
  looked up from the enclosing scope, a composite one
  (`@Modelica::@Constants::@pi`, global if rooted at `@"."`) starts with a
  class. Use `hasSimpleName()` / `getSimpleName()`; connect-equations,
  assignments and the passes after instantiation require simple names.
- **One op per language construct**: `zeros`, `ones`, `fill` stay separate;
  so do `for_equation`/`for`, `if_equation`/`if`, `when_equation`/`when`.
  The if-expression is `modelica.if_expr`.
- **Bmodelica conversion** rebuilds integer index arithmetic directly on
  `index` values, because the bmodelica access analysis doesn't look through
  casts. Unused ops are erased at the end, not during the conversion.

## Conventions

- All ops are in `Ops.td`, grouped by `//===---===//` section comments.
- TableGen layout: multi-line `(ins`/`(outs`/`(region`/`!con(` lists have
  one item per line and the closing `);` on its own line. Descriptions are
  wrapped at 80 columns, never splitting an inline code span.
- Examples in descriptions only when they aren't trivial.
- Headers only under `include/`; include guards follow the path, with no `LIB`.
- Never cite section, chapter or version numbers of the language specification
  in code, TableGen or tests.
- The existing code isn't clang-format clean: format only the changed lines
  (`clang-format --lines=...`).
- Don't commit `MLS.pdf`.
- Lit needs a Python with the `lit` module (here `/usr/bin/python3 bin/llvm-lit`
  from `build/`).

## Status

With the temporary `CompilerInvocation.cpp` stub: lit 645/653, unit tests
412/412. The 8 lit failures are the tests using the new driver flags
(`test/Driver/emit-mlir-*`, `test/Frontend/Actions/emit-mlir-*`,
`native-frontend.mo`). Forcing the native frontend, `native-frontend.mo` (RC
ladder with arrays of instances and `connect` inside `for`) matched its
expected output.

## Open items

- Rebuild LLVM with the fork changes and rerun the 8 driver tests.
- Decide whether to keep `Connections.root`, `Connections.potentialRoot` and
  `Connections.uniqueRoot` as separate ops.
- Rerun the sweep over the Modelica Standard Library examples (flatten and
  convert counts). The earlier harness is gone, and the `marco` driver stops on
  unsupported library classes (e.g. `Clocked`), so a harness that flattens one
  model at a time is needed.
- Known gaps: members of record values in Flatten ("records are not supported
  yet"), clocked operators.
