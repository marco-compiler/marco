# Array-aware Connection Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the `modelica-connection-expansion` pass so that connection sets and connection equations are computed on index sets and access functions, without enumerating array elements.

**Architecture:** An IR-independent core (`ConnectionGraph`, `ConnectionEquations`) computes, from vertices (connector variables with their index sets) and edges (connect-equations as equation index sets plus two access functions), a piecewise representative map and then equation families. The pass builds the graph from the instantiated model, runs the core, and emits the families as for-equations. The core is unit-tested with gtest; the pass with lit.

**Tech Stack:** C++17, MLIR, `marco::modeling` (`IndexSet`, `MultidimensionalRange`, `Point`, `AccessFunction`), gtest, lit/FileCheck.

**Spec:** `docs/superpowers/specs/2026-11-15-array-aware-connection-expansion-design.md`

## Global Constraints

- Code in namespace `mlir::modelica::connection` (helpers) and `mlir::modelica` (pass); no code in `marco::modeling`.
- No new external dependencies.
- Indices are zero-based in the Modelica dialect.
- Accesses that are offsets, constants or permutations of the equation indices (built by `AccessFunction::build` as `AccessFunctionRotoTranslation` / constant kinds) must never cause per-element work.
- Other affine accesses and self-translations with `|c| > 1` are handled correctly by per-point processing, with the comment `// TODO: keep compact once ranges with a step are supported.`
- Non-affine accesses: error `"non-affine indices in connect-equations are not supported yet"` with `// TODO: enumerate the indices once Modelica expressions can be evaluated.`
- Connect-equations nested in anything but for-equations: error `"connect-equations can only be nested in for-equations after the instantiation"`.
- Stream variables are not vertices; constants and parameters are not part of connection sets (as today).
- Inside flow terms are added, outside flow terms subtracted (as today).
- Do not commit: the user commits. Leave changes in the working tree.
- Follow `HANDOFF.md` conventions: clang-format only changed lines, headers under `include/`, no section numbers of the language specification.

## Review Focus

- A connector variable connected both as inside and as outside: the two vertices of the same variable end up in one set; no `x = x` equation may be emitted, and the flow sum has `+x - x` terms. (Task 4 test `insideAndOutsideSameVariable`.)
- A connect-equation repeated twice, or two for-equations connecting the same pairs: no duplicate or redundant equations. (Task 2 test `duplicateEdge`.)
- Equation indices with an unused dimension (e.g. a 2-D loop connecting `a[i]` to `b[i]`): treated as one connection per `i`, not per `(i, j)`. (Task 2 test `unusedEquationDimension`.)
- Empty for-equation ranges (e.g. `for k in 1:0`): no vertices or edges created, no equations. (Task 6 lit test `empty-range.mlir`.)
- Scalar connect-equations mixed with array ones on the same variables (`connect(a[1].p, g)` plus `for k: connect(a[k].n, a[k+1].p)`). (Task 2 test `junctionScalarAndArray`.)

---

### Task 1: Connection graph and union of bijective edges

**Files:**
- Create: `include/marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.h`
- Create: `lib/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.cpp`
- Modify: `lib/Dialect/Modelica/Transforms/CMakeLists.txt` (add the source)
- Create: `unittest/Dialect/CMakeLists.txt`, `unittest/Dialect/Modelica/CMakeLists.txt` (target `DialectModelicaTest`, linking `MLIRModelicaTransforms`), `unittest/Dialect/Modelica/ConnectionGraph.cpp`
- Modify: `unittest/CMakeLists.txt` (`add_subdirectory(Dialect)`)

**Interfaces:**
- Produces:

```cpp
namespace mlir::modelica::connection {
using VertexId = unsigned;

struct Vertex {
  unsigned variable;            // Opaque id of the primitive variable.
  bool outside;
  bool isFlow;
  marco::modeling::IndexSet indices;
};

struct Edge {
  marco::modeling::IndexSet equationIndices;
  VertexId lhs, rhs;
  std::unique_ptr<marco::modeling::AccessFunction> lhsAccess, rhsAccess;
};

/// Elements of a vertex sharing the way their representative is computed.
/// A root piece contains only representatives (toRoot is the identity and
/// root is the vertex itself); a non-root piece contains none.
struct Piece {
  marco::modeling::IndexSet domain;
  VertexId root;
  std::unique_ptr<marco::modeling::AccessFunction> toRoot;
  bool isRoot;
};

class ConnectionGraph {
public:
  explicit ConnectionGraph(mlir::MLIRContext *context);
  VertexId addVertex(Vertex vertex);
  void addEdge(Edge edge);
  size_t getNumVertices() const;
  const Vertex &getVertex(VertexId id) const;
  void computeRepresentatives();
  llvm::ArrayRef<Piece> getPieces(VertexId id) const;
};
} // namespace mlir::modelica::connection
```

- [ ] **Step 1: Write the failing test `ConnectionGraph.ladder`** — the spec example, zero-based. Vertices `rn = {0,1,2}`, `rp = {0,1,2}`, `g` scalar (rank-0 index set with one point); edges: A `k ∈ [0,2)`, `rn[k]` ~ `rp[k+1]`; B `rn[2]` ~ `g`; C `rp[0]` ~ `g`. Assert, after `computeRepresentatives()`, that every element of each set reaches the same `(root, index)` (evaluate `toRoot->map(Point)` per piece), that the three sets are `{rn0, rp1}`, `{rn1, rp2}`, `{rn2, g, rp0}`, and that the total number of pieces is ≤ 6.
- [ ] **Step 2: Run `ninja DialectModelicaTest && bin/DialectModelicaTest --gtest_filter=ConnectionGraph.*`** — expected: compile failure (missing header).
- [ ] **Step 3: Implement the class.** Initial state: one root piece per vertex (identity over `indices`). `computeRepresentatives()` processes the edges in insertion order with the union below; equality of `(root, toRoot)` short-circuits.

```
union(edge):
  for each piece Pa of lhs, Pb of rhs:
    D = lhsAccess.inverseMap(Pa.domain, E) ∩ rhsAccess.inverseMap(Pb.domain, E)
    if D empty: continue
    L = combine(lhsAccess, Pa.toRoot), rootL = Pa.root        // eq index -> root index
    R = combine(rhsAccess, Pb.toRoot), rootR = Pb.root
    if rootL == rootR and L == R on D: continue
    if rootL == rootR: selfUnion(rootL, L, R, D); continue     // Task 3
    child = side with larger root id, unless only the other side's map is
            invertible on D (then that other side is the child)
    if neither map invertible on D: pointwiseUnion(D)           // Task 2
    else rePoint(childRoot, childMap(D), parentRoot, parentMap ∘ childMap⁻¹)

rePoint(childRoot, C, parentRoot, F):   // roots C of childRoot now map via F
  for each piece Q (any vertex) with Q.root == childRoot:
    S = Q.toRoot.inverseMap(C, Q.domain)
    if S empty: continue
    split Q into Q - S (unchanged) and S with root = parentRoot,
    toRoot = combine(Q.toRoot, F), isRoot = false
```

  Keep an index from root vertex to the pieces pointing to it so `rePoint` doesn't scan all vertices. Merge adjacent pieces with equal `(root, toRoot, isRoot)` after each edge to keep their number small.
- [ ] **Step 4: Run the test** — expected: PASS.

### Task 2: Junctions, duplicates, non-invertible edges

**Files:**
- Modify: `lib/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.cpp`
- Test: `unittest/Dialect/Modelica/ConnectionGraph.cpp`

**Interfaces:**
- Consumes: Task 1 `ConnectionGraph`.
- Produces: private `pointwiseUnion(const IndexSet &D, const Edge &edge)` (unions each point separately, with the strided TODO comment).

- [ ] **Step 1: Write failing tests**
  - `junctionScalarAndArray`: `a = {0..9}`, `b = {0..9}`, `g` scalar; edges `a[0]` ~ `g` and, for `k ∈ [0,10)`, `a[k]` ~ `b[k]`. Assert sets `{a0, b0, g}` and `{ak, bk}` for `k ≥ 1`, and pieces ≤ 6.
  - `duplicateEdge`: the same edge `a[k]` ~ `b[k]` added twice; assert pieces equal to the single-edge case.
  - `unusedEquationDimension`: equation indices `[0,3) × [0,4)`, accesses `a[i]` and `b[i]` (dimension `j` unused); assert sets `{ai, bi}`.
  - `strided`: `a = {0..9}`, `b = {0..4}`, `k ∈ [0,5)`, `a[2k]` ~ `b[k]`; assert sets `{a(2k), bk}` and singletons for odd `a`.
- [ ] **Step 2: Run** — expected: FAIL on at least `strided` / `unusedEquationDimension`.
- [ ] **Step 3: Implement** — drop unused equation dimensions before the union (project `E` and the accesses); use `pointwiseUnion` when neither map is invertible on `D`.
- [ ] **Step 4: Run** — expected: PASS.

### Task 3: Self-unions and closure of self-translations

**Files:**
- Modify: `lib/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionGraph.cpp`
- Test: `unittest/Dialect/Modelica/ConnectionGraph.cpp`

**Interfaces:**
- Consumes: Task 1 union, `rePoint`, Task 2 `pointwiseUnion`.
- Produces: private `selfUnion(VertexId root, const AccessFunction &L, const AccessFunction &R, const IndexSet &D)`.

- [ ] **Step 1: Write failing tests**
  - `selfChain`: `x = {0..999999}`, `k ∈ [0, 999999)`, `x[k]` ~ `x[k+1]`. Assert one set (all elements map to `x[0]`), pieces ≤ 2, and the test runs under 1 s (`EXPECT_LT` on a `std::chrono` measurement).
  - `selfChain2D`: `x = [0,4) × [0,5)`, `i ∈ [0,4), j ∈ [0,4)`, `x[i,j]` ~ `x[i,j+1]`; assert sets are the rows (`x[i,·]` → `x[i,0]`).
  - `selfChainStride2`: `x = {0..9}`, `k ∈ [0,8)`, `x[k]` ~ `x[k+2]`; assert two sets (even → `x0`, odd → `x1`).
  - `cycleNoRedundancy`: `x[k]` ~ `x[k+1]` for `k ∈ [0,4)` plus `x[4]` ~ `x[0]`; assert one set.
- [ ] **Step 2: Run** — expected: FAIL.
- [ ] **Step 3: Implement `selfUnion`.** If `T = R ∘ L⁻¹` is a translation by `c = ±e_d` (one unit component): `G = L.map(D)`; split `G ∪ T(G)` into maximal runs along `d` (boxes of `G` with identical ranges in the other dimensions whose `d`-segments `[lo, hi + 1]` touch or overlap merge; split boxes on a common grid of the other dimensions first). For each run with start `s` along `d`: the slice `coord_d == s` stays a root piece; `rePoint(root, run − slice, root, F)` with `F` setting coordinate `d` to `s` and keeping the others. Otherwise (other `c`, non-translation): `pointwiseUnion` with the strided TODO.
- [ ] **Step 4: Run** — expected: PASS.

### Task 4: Equation families

**Files:**
- Create: `include/marco/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionEquations.h`
- Create: `lib/Dialect/Modelica/Transforms/ConnectionExpansion/ConnectionEquations.cpp`
- Test: `unittest/Dialect/Modelica/ConnectionEquations.cpp`

**Interfaces:**
- Consumes: Task 1 `ConnectionGraph`, `Piece`.
- Produces:

```cpp
namespace mlir::modelica::connection {
/// vertex[i] = root[toRoot(i)] for each i in indices.
struct PotentialFamily {
  marco::modeling::IndexSet indices;
  VertexId vertex, root;
  std::unique_ptr<marco::modeling::AccessFunction> toRoot;
};

/// Term vertex[fromRoot(r)] of the sum of the set whose root is r.
struct FlowTerm {
  VertexId vertex;
  std::unique_ptr<marco::modeling::AccessFunction> fromRoot;
};

/// Sum over the iterators of vertex[access(r, iterators)].
struct FlowReduction {
  VertexId vertex;
  marco::modeling::MultidimensionalRange iterators;
  std::unique_ptr<marco::modeling::AccessFunction> access;
};

/// sum(terms) + sum(reductions) = 0 for each root index r in rootIndices.
struct FlowFamily {
  marco::modeling::IndexSet rootIndices;
  VertexId root;
  llvm::SmallVector<FlowTerm> terms;
  llvm::SmallVector<FlowReduction> reductions;
};

struct ConnectionEquations {
  llvm::SmallVector<PotentialFamily> potentials;
  llvm::SmallVector<FlowFamily> flows;
};

ConnectionEquations computeEquations(const ConnectionGraph &graph);
} // namespace mlir::modelica::connection
```

- [ ] **Step 1: Write failing tests** (graphs from Task 1–3 tests, plus):
  - `ladderEquations`: potentials cover exactly `rp{1,2}`, `rn{2}`, `rp{0}` (3 elements) with the expected roots; flows: one family over `rn{0,1}` with terms `rn[r]`, `rp[r+1]`, one over `g` with terms `g`, `rn[2]`, `rp[0]`.
  - `insideAndOutsideSameVariable`: vertices `(v=7, inside)` and `(v=7, outside)` with the same index set connected element-wise; assert no potential family for them, and one flow family with both terms (signs are applied at emission).
  - `unconnectedFlow`: flow vertex `{0..2}` with no edges → one flow family over `{0..2}` with the single identity term.
  - `selfChainReduction`: from `selfChain` with `x` flow → one flow family over `{0}` with one `FlowReduction` iterating `x` over `[0, 1000000)`.
- [ ] **Step 2: Run** — expected: compile failure.
- [ ] **Step 3: Implement `computeEquations`.** Potentials: every non-root piece of a potential vertex yields a family, skipping elements whose root is the same variable at the same index (inside/outside pairs). Flows: per root vertex, overlay the images of the pieces pointing to it over its root pieces; each region of the overlay is a family; invertible `toRoot` pieces give `FlowTerm`s with `fromRoot = toRoot⁻¹`; pieces whose `toRoot` sets some dimensions to constants and maps the others invertibly give `FlowReduction`s over the ranges of the constant dimensions; any other piece is split into points (strided TODO).
- [ ] **Step 4: Run** — expected: PASS.

### Task 5: Building the graph from the model

**Files:**
- Rewrite: `lib/Dialect/Modelica/Transforms/ConnectionExpansion.cpp` (graph construction part; keep `collectVariables`, `getLeaves`, connector-resolution checks and their error messages)
- Test: lit, Task 6

**Interfaces:**
- Consumes: Task 1 `Vertex`, `Edge`, `ConnectionGraph`.
- Produces (file-local): `std::optional<mlir::AffineExpr> getAffineIndex(mlir::Value index, llvm::ArrayRef<mlir::Value> iterators, Evaluator &evaluator)` — constants (anything the evaluator computes without the iterators), iterators (as dimensions), `add`, `sub`, `neg`, `mul` with a constant operand; otherwise `std::nullopt`.

- [ ] **Step 1: Implement.**
  - Vertices: one inside vertex per flow variable of the model (so unconnected flows get `= 0`), plus one per `(variable, side)` reached by a connect-equation; index set = full shape (enclosing instance dims + own dims). Skip stream variables; skip constants and parameters in edges.
  - Equation indices of a connect-equation: the ranges of its enclosing for-equations (evaluated; a range with step 1 becomes a `Range`, anything else becomes points with the strided TODO), followed by one dimension per colon / slice subscript and per own dimension of the connected leaf (same sizes required on both sides, existing error messages).
  - Accesses: `AccessFunction::build(AffineMap)` from the `getAffineIndex` results along the connector chain; slice subscripts that evaluate to step-1 vectors become `dim + offset`.
  - Errors: the two messages from Global Constraints, with their TODOs.
- [ ] **Step 2: Build** — `ninja MLIRModelicaTransforms` — expected: success (emission still old until Task 6; keep the old emission compiling or stub the pass to fail with `"not implemented"` only within this task).

### Task 6: Emitting the equations and replacing the pass

**Files:**
- Rewrite: `lib/Dialect/Modelica/Transforms/ConnectionExpansion.cpp` (emission; delete `SetElement`, `EquationRange`, `EquationSignature`, the union-find and the regrouping)
- Modify: `include/marco/Dialect/Modelica/Transforms/Passes.td` (pass description: no element-wise processing)
- Modify tests: `test/Dialect/Modelica/Transforms/ConnectionExpansion/{chain,grid,signals}.mlir`
- Create tests: `test/Dialect/Modelica/Transforms/ConnectionExpansion/{junction,cycle,self-chain,inside-outside,unconnected,nesting-error,non-affine-error,empty-range,scaling}.mlir`

**Interfaces:**
- Consumes: Task 4 `computeEquations`, Task 5 graph construction.

- [ ] **Step 1: Write the lit tests.** Each starts with `// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s` (errors: `--verify-diagnostics`). `scaling.mlir`: `TwoPin r[1000000]` chained by `connect(r[k].n, r[k+1].p)` in a for-equation; CHECK the potential and flow families as `modelica.for_equation`s with `tensor<999999x!modelica.int>` ranges, and that the only equations outside for-equations are the ones of the two chain ends. Timing is asserted by the `selfChain` unit test, not by lit.
- [ ] **Step 2: Run** `PYTHONPATH=…/llvm/utils/lit /usr/bin/python3 bin/llvm-lit -v ../../test/Dialect/Modelica/Transforms/ConnectionExpansion` — expected: new tests FAIL.
- [ ] **Step 3: Implement emission.** For each family and each `MultidimensionalRange` of its index set: nested `ForEquationOp`s with `RangeOp` from `begin` to `end - 1` (as the current code builds them), an `EquationOp`, references built with the existing `emitReference` and indices from the family's access function affine map (emit `ConstantOp`, `AddOp`, `MulOp` for its results). Flow families: `sum = 0`, outside vertices subtracted; `FlowReduction` → `ReductionOp` `sum` over one 1-D `RangeOp` per iterator dimension. Erase the connect-equations and the emptied for-equations as today.
- [ ] **Step 4: Update** the CHECK lines of the existing three tests to the new (equivalent) equations.
- [ ] **Step 5: Run** the directory, then the whole suite — expected: all ConnectionExpansion tests PASS; the full suite reports the same 28 MPI-runtime simulation failures as before and no others.

### Task 7: Documentation

**Files:**
- Modify: `HANDOFF.md` (design decisions: connection expansion on index sets; open items: steps in ranges, non-affine TODO)

- [ ] **Step 1: Update** `HANDOFF.md`.
- [ ] **Step 2: Run** the unit tests (`bin/DialectModelicaTest`) and lit once more — expected: PASS (same 28 MPI failures).
