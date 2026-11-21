# Array-aware connection expansion

## Intent

The `modelica-connection-expansion` pass replaces the connect-equations of an
instantiated model with the equations they imply. Today it expands every
connect-equation into one entry per scalar element, builds the connection sets
with a union-find over those scalars, and then rebuilds for-equations by
recognizing affine patterns among the generated scalar equations. Its cost and
memory grow with the number of scalar elements, which defeats the purpose of
MARCO: models with arrays of millions of instances must be handled without
ever enumerating their elements.

The pass is to be rewritten around a set-based graph, in which arrays are
first-class: vertices and edges describe whole arrays through index sets and
access functions, and the connection sets and the equations are computed
symbolically. The cost must depend on the number of connect-equations and on
the complexity of their index patterns, not on the size of the arrays.

## Goals

- Compute the connection sets and generate the connection equations without
  enumerating the elements of the arrays, for connect-equations whose
  accesses are offsets, constants and permutations of the loop variables
  (e.g. `k`, `k + 1`, `[j, i]`, `3`), that is the accesses that map ranges
  to ranges.
- Handle the other affine accesses (e.g. `2 * k`) correctly. Their index
  sets are not ranges with step 1, so `IndexSet` stores them point by point
  and their cost depends on the number of elements they involve. A TODO
  marks that ranges with a step in `marco::modeling` will make them compact:
  this is an optimization that comes later.
- Handle any connection topology correctly: scalar connections, families of
  connections in (possibly nested) for-equations, junctions of several
  connectors, cycles, and arrays connected to themselves at shifted indices
  (e.g. `connect(x[k], x[k + 1])`).
- Generate equations equivalent to the ones implied by the scalar expansion of
  the connections, emitted as for-equations over index ranges.
- Demonstrate the scaling with a test on a model with arrays of about a
  million instances.

## Non-goals

- Connect-equations with non-affine accesses (e.g.
  `connect(a[if k < 3 then 1 else 2], b[k])`). The pass reports that they are
  not supported yet, with a TODO in the code. Supporting them requires
  enumerating their indices, which in turn requires evaluating arbitrary
  Modelica expressions: this will come with a future interpreter / JIT of
  Modelica, which is out of scope here.
- Stream variables: as today, they are not equated or summed by the
  connections, as their semantics is given by the `inStream` /
  `actualStream` operators.
- Overdetermined connectors and the `Connections.*` operators, and expandable
  connectors: not handled by today's pass either.
- A general-purpose set-based graph library: the code lives with the pass.

## Anti-goals

- Any step whose cost depends on the number of array elements for
  connections whose accesses are offsets, constants and permutations of the
  loop variables, including hidden ones (e.g. a fixed-point iteration whose
  number of iterations grows with the array size).
- Equations that differ in meaning from the scalar expansion: missing,
  redundant or conflicting equations.

## Constraints

- The code lives in the `mlir::modelica` namespace, with the pass in
  `lib/Dialect/Modelica/Transforms/` (its helpers may go in a subdirectory,
  as the instantiation does with `Instantiation/`). It is not a component of
  `marco::modeling`.
- It builds on the existing `marco::modeling` classes: `IndexSet`,
  `MultidimensionalRange`, `Point` and `AccessFunction` (with `map`,
  `inverseMap`, `combine`, `isInvertible` and `inverse`). An access function
  maps equation indices to variable indices.
- No new external dependencies (e.g. no ISL).
- Indices are zero-based in the Modelica dialect.
- The pass runs after the instantiation, on `modelica.model` operations:
  instances and variables have static shapes, component references are
  absolute, and the ranges of the for-equations containing connect-equations
  are evaluable constants.

## Design

### Example

The rest of this section refers to this model:

```modelica
connector Pin  Real v; flow Real i;  end Pin;
model TwoPin   Pin p, n;             end TwoPin;

model Ladder
  TwoPin r[3];
  Pin g;
equation
  for k in 1:2 loop
    connect(r[k].n, r[k + 1].p);   // (A)
  end for;
  connect(r[3].n, g);               // (B)
  connect(r[1].p, g);               // (C)
end Ladder;
```

Element-wise, its connection sets are `{r[1].n, r[2].p}`,
`{r[2].n, r[3].p}` (from A) and `{r[3].n, g, r[1].p}` (from B and C, joined
through `g`).

### Vertices

A vertex is a primitive variable of the connectors referenced by the
connect-equations, together with the side (inside or outside) of the
connector it belongs to, as the same variable may be connected both as an
inside and as an outside connector, and the two are distinct elements of the
connection sets (the `lhsOutside` / `rhsOutside` flags of `modelica.connect`).

The index set of a vertex covers all its elements: the dimensions of the
enclosing arrays of instances followed by the dimensions of the variable.
In the example, `r[1].n.v`, `r[2].n.v` and `r[3].n.v` are the three elements
of the vertex `r.n.v`, whose index set is `{0, 1, 2}`; the vertices are
`r.p.v`, `r.p.i`, `r.n.v`, `r.n.i` (3 elements each), `g.v` and `g.i` (one
element each).

Stream variables don't become vertices.

### Edges

Each connect-equation produces one edge for each pair of corresponding
primitive variables of the two connectors. An edge is an equation family:

- its equation indices are the index set given by the ranges of the enclosing
  for-equations (a single point for a connect-equation not nested in
  for-equations);
- each side has an access function from the equation indices to the indices
  of its vertex.

In the example (one-based indices for readability):

| Edge | Equation indices | Left access    | Right access     |
|------|------------------|----------------|------------------|
| (A)  | `k ∈ {1, 2}`     | `r.n.v[k]`     | `r.p.v[k + 1]`   |
| (B)  | one equation     | `r.n.v[3]`     | `g.v`            |
| (C)  | one equation     | `r.p.v[1]`     | `g.v`            |

and the same for the `i` variables.

Connect-equations nested in operations other than for-equations can't occur
after the instantiation (if-equations containing connect-equations have
parameter conditions and are resolved by it, and connect-equations can't
appear in when-equations). The pass keeps checking it, with a message saying
so.

### Connection sets

Each connection set is identified by its representative: its smallest
element, according to a fixed total order of the elements (e.g. by vertex,
then by index). Following the set-based graph approach of Kofman, Fernández
and Zimmermann, the pass computes for every element the representative of its
set, as a fixed point: initially every element is its own representative,
and the smaller representative is propagated across the edges until nothing
changes. All the operations work on index sets and access functions, never on
single elements.

The result is a piecewise map: for each vertex, a small number of pieces, each
made of an index set and an access function giving the representatives of
its elements. In the example:

| Vertex  | Indices    | Representative   |
|---------|------------|------------------|
| `r.n.v` | `{1, 2}`   | `r.n.v[k]`       |
| `r.n.v` | `{3}`      | `g.v`            |
| `r.p.v` | `{2, 3}`   | `r.n.v[k - 1]`   |
| `r.p.v` | `{1}`      | `g.v`            |
| `g.v`   | (scalar)   | `g.v`            |

With 1000 resistors the table keeps the same five rows: only the index sets
change.

Arrays connected to themselves at shifted indices (`connect(x[k], x[k + 1])`)
put many elements in the same set; propagating the representative one step
at a time would take a number of iterations proportional to the array size.
The fixed point must therefore close such self-referencing maps in a single
step: for a translation by `c` within a range, the representative is
computable in closed form. With `|c| = 1` the whole range has the same
representative and stays compact; with `|c| > 1` the sets are the residue
classes modulo `c`, which are not ranges with step 1 and are thus stored
point by point (same TODO as the strided accesses).

### Equations

All the equations are emitted as for-equations over the ranges of the index
sets of the pieces, or as plain equations for single points.

- Potential variables: every element that is not its own representative is
  equated to its representative. Each set of `n` elements thus produces
  `n - 1` equations, without redundancy even in presence of cycles, as the
  equations follow the representatives and not the edges. In the example:

  ```modelica
  for k in 2:3 loop r[k].p.v = r[k - 1].n.v; end for;
  r[3].n.v = g.v;
  r[1].p.v = g.v;
  ```

- Flow variables: one zero-sum equation per set, its members being the
  elements sharing the same representative (found by inverting the
  representative map). Inside connectors are summed positively and outside
  connectors negatively, as today. When the members of the sets of a piece
  are a fixed number of terms, the sum is written term by term:

  ```modelica
  for k in 1:2 loop r[k].n.i + r[k + 1].p.i = 0; end for;
  g.i + r[3].n.i + r[1].p.i = 0;
  ```

  When the number of members depends on the indices (e.g. the whole of `x`
  for `connect(x[k], x[k + 1])`), the sum is a `modelica.reduction` over the
  index set of the members.

- Unconnected flow variables: the elements of the flow vertices that are not
  reached by any edge (the vertex index set minus the images of the edges'
  access functions) are set to zero.

The connect-equations are then erased, as today.

### Errors

- Non-affine accesses: "not supported yet" error on the connect-equation, with
  a TODO referring to the future Modelica expression evaluation.
- Connect-equations nested in operations other than for-equations: error
  explaining that only for-equations can enclose connect-equations after the
  instantiation.
- Mismatching connectors and other malformed input: as today.

## Testing

- The existing tests in `test/Dialect/Modelica/Transforms/ConnectionExpansion/`
  remain as correctness tests; their expected output is updated to the new
  equations, which must be equivalent to the current ones.
- New lit tests covering: scalar connections, junctions, cycles, nested
  for-equations, multi-dimensional arrays, arrays connected to themselves
  (variable-size sets), inside / outside connectors, unconnected flows, and
  the non-affine error.
- A scaling test on a model with arrays of about a million instances, which
  must complete quickly and must produce for-equations over ranges, with no
  per-element equations.
- The end-to-end simulation tests using the native frontend (e.g.
  `test/Simulation/ida/native-frontend.mo`) keep producing the same results.

## Left to the builder

- The data structures of the piecewise maps and of the graph.
- The order of the elements used to choose the representatives, as long as it
  is total and deterministic.
- The technique used to close self-referencing affine maps in the fixed point.
- How the pieces are merged to keep their number small, and how the index
  sets of the pieces are split into ranges for the emitted for-equations.
- The exact content of the scaling test model and its time limit.
- Whether for-equations with implicit ranges can still contain
  connect-equations after the instantiation; if they can, their ranges are
  deduced from the accessed arrays.
- How a variable-size sum over an index set made of several ranges is
  emitted (e.g. as the sum of one reduction per range), including any support
  needed in the conversion to Base Modelica.
