# Native Modelica frontend: status and handoff

Work in progress on a native frontend that flattens Modelica models. Nothing
below was committed as of 2026-11-08.

## Pipeline

```
.mo --(Parser/Modelica)--> AST --(Lowering/Modelica)--> modelica dialect (class tree)
    --modelica-instantiation--> instance tree
    --modelica-connection-expansion--> connection equations
    --modelica-flattening--> flat model
    --convert-modelica-to-bmodelica--> bmodelica (existing backend)
```

`marco` chooses the frontend from the input type: `.mo` files go to the native
frontend, `.bmo` files to the Base Modelica parser. Marco doesn't run any other
compiler: Base Modelica code produced by other tools is passed as `.bmo` files.

- `-emit-mlir-modelica` emits the class tree right after the lowering of the
  classes (no `--model` needed).
- `-emit-mlir-bmodelica` emits the flat model in the Base Modelica dialect.
- `CodeGenAction::generateMLIR` builds the starting module (the class tree for
  `.mo` files). `buildMLIRFlatteningPipeline` adds the instantiation,
  connection expansion and flattening passes, and the callers add the
  conversion to Base Modelica, all in the same pass manager as the rest of
  the compilation.

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

The driver flags live in the LLVM fork (`llvm-project`, branch `marco-llvm`),
not in this repository. Modified files:
`clang/include/clang/Driver/Options.td` (`-emit-mlir-bmodelica`; the options
running an external Modelica compiler, `-emit-base-modelica` and marco's
`-emit-mlir` removed, the latter back to the original alias of `-emit-fir`),
`clang/include/clang/Driver/Types.def` (`mlir-bmodelica`, `.bmo.mlir`),
`clang/lib/Driver/Types.cpp`, `clang/lib/Driver/Driver.cpp`,
`clang/lib/Driver/ToolChains/Marco.cpp`, `clang/lib/Driver/ToolChain.cpp`,
`clang/include/clang/Basic/DiagnosticDriverKinds.td`.

LLVM was rebuilt and reinstalled in `install/llvm-project/rel-with-deb-info`
with these changes. Commit them in the fork as well.

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
- **One op per class restriction**: `modelica.class`, `model`, `block`,
  `record`, `connector`, `type`, `package`, `function`, `operator`, sharing
  `ClassInterface` and the `Modelica_ClassBaseOp` TableGen base. `operator
  record`, `expandable connector` and `operator function` are unit attributes.
  The same `modelica.model` / `modelica.function` ops are the result of the
  instantiation (top-level, containing variables and instances); the
  instantiated root is always a `modelica.model`, whatever its restriction.
  `modelica.instance` marks connector instances with `connector` /
  `expandable connector`. There is no class-kind enum.
- **One op per language construct**: `zeros`, `ones`, `fill` stay separate;
  so do `for_equation`/`for`, `if_equation`/`if`, `when_equation`/`when`.
  The if-expression is `modelica.if_expr`.
- **Connection expansion on index sets**: `modelica-connection-expansion`
  builds a graph whose vertices are `(variable, side)` pairs (inside or
  outside of the connector), each with the index set of the variable (scalars
  are rank 1 `{0}`). The edges are the connect-equations, as an equation
  index set (the for-equation ranges) plus an access function per side.
  Connection sets are a piecewise representative map over the vertices.
  Equations are emitted as for-equations or reductions over index sets, with
  no per-element work for offset, constant and permutation accesses, except
  for the self-relations below. Per-point fallbacks, all marked with TODOs:
  - strided accesses (`a[2 * k]`) and self translations by more than one
    element (`x[k] ~ x[k + 2]`), which need ranges with a step;
  - self permutations (`x[i, j] ~ x[j, i]`) and self translations along more
    than one dimension (`x[i, j] ~ x[i + 1, j + 1]`), whose sets are not
    boxes;
  - the comparison of the inside and outside sides of the same variable,
    when their maps to the representatives use different dimensions;
  - for-equation ranges that are not a `modelica.range` or have a step other
    than 1;
  - for-equation ranges and slices whose bounds depend on the iterators of
    the enclosing for-equations (the iterators are enumerated).

  The per-point work grows linearly with the number of elements (the unit
  tests count it), but the index sets made of many ranges are slow to build
  (see the open items). Unit tests are in `unittest/Dialect/Modelica/`.
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

lit 670/702 with 4 expected failures, unit tests 444/444 (`DialectModelicaTest`
30, `ModelingTest` 134, `ParserBaseModelicaTest` 177, `ParserModelicaTest` 60,
`LoweringModelicaTest` 25, `LexerTest` 7, `VariableFilterTest` 11). The 28 lit
failures are simulation tests that compile but abort in `MPI_Comm_dup` at
runtime: the installed runtime predates SUNDIALS 7 support, and needs to be
rebuilt from the current `marco-runtime`. In an earlier run with a working
runtime, `native-frontend.mo` (RC ladder with arrays of instances and
`connect` inside `for`) matched its expected output.

## Open items

- The simulation tests exist both as `.bmo` (Base Modelica parser) and `.mo`
  (native frontend), with the same expected output. The `.mo` copies of
  `record-construction`, `record-inline-function`, `record-subscripts` and
  `der-function-call` are XFAIL, as they use features the native frontend
  doesn't support yet: record constructors, record arguments of functions,
  members of arrays of records, derivative definitions of functions.

- Rebuild the runtime and rerun the simulation tests, now going through the
  native frontend by default.
- Decide whether to keep `Connections.root`, `Connections.potentialRoot` and
  `Connections.uniqueRoot` as separate ops.
- Rerun the sweep over the Modelica Standard Library examples (flatten and
  convert counts). The earlier harness is gone, and the `marco` driver stops on
  unsupported library classes (e.g. `Clocked`), so a harness that flattens one
  model at a time is needed.
- Known gaps: members of record values in the flattening pass ("records are not supported
  yet"), clocked operators.
- Ranges with a step in `marco::modeling`, to keep the strided cases of the
  connection expansion compact (now per-point fallbacks).
- Index sets made of many ranges in `marco::modeling`: `IndexSet::insert`
  scans all the ranges, `getCompactRanges` is quadratic in their number, and
  in Debug builds each update checks the whole tree. Building them in bulk
  (`IndexSet(ArrayRef<MultidimensionalRange>)`, also used by `intersect`)
  breaks the fan-out invariant of the R-tree for some numbers of ranges (e.g.
  200), failing a Debug assertion. The connection expansion avoids most of
  these operations on its per-point paths.
- Non-affine indices in connect-equations: they need the evaluation of
  Modelica expressions.
- The `marco::modeling` Debug assertions make million-element index sets slow:
  the scaling lit test uses 10^5 elements, and the `selfChain` unit test takes
  about 30 s in Debug.
