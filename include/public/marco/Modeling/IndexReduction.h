#ifndef MARCO_MODELING_INDEXREDUCTION_H
#define MARCO_MODELING_INDEXREDUCTION_H

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "index-reduction"
#endif

#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/LocalMatchingSolutionsImpl.h"
#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/Support/Debug.h"

namespace marco::modeling {

namespace internal::indexReduction {

/// A variable vertex in the graph.
template <typename VariableData>
class VariableVertex final : public Dumpable {
public:
  using Id = typename VariableData::Id;

  Id getId() const { return id; }
  const IndexSet &getIndices() const { return indices; }

  void hideIndices(const IndexSet &indices) { visibleIndices -= indices; }
  const IndexSet &getVisibleIndices() const { return visibleIndices; }

  explicit VariableVertex(const Id &id, const IndexSet &indices)
      : id(id), indices(indices), visibleIndices(indices) {}

  void dump(llvm::raw_ostream &os) const override {
    if (visibleIndices != indices) {
      os << "Variable(id: " << getId() << ", indices: " << getIndices()
         << ", visible indices: " << getVisibleIndices() << ")";
      return;
    }
    os << "Variable(id: " << getId() << ", indices: " << getIndices() << ")";
  }

private:
  /// The id of the variable.
  Id id;
  /// The set of all indices for the variable.
  IndexSet indices;
  /// The set of visible indices for the variable.
  /// Indices are hidden when the variable is differentiated.
  IndexSet visibleIndices;
};

/// An equation vertex in the graph.
template <typename EquationData>
class EquationVertex final : public Dumpable {
public:
  using Id = typename EquationData::Id;

  Id getId() const { return id; }

  const IndexSet &getIndices() const { return validIndices; }

  const IndexSet &getAllIndices() const { return allIndices; }

  void validateIndices(const IndexSet &indices) {
    assert(allIndices.contains(indices) && "Invalid indices for equation");
    validIndices += indices;
  }

  EquationVertex(const Id &id, const IndexSet &indices,
                 const IndexSet &validIndices)
      : id(id), allIndices(indices), validIndices(validIndices) {
    assert(allIndices.contains(validIndices) &&
           "Valid indices must be a subset of equation indices");
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "Equation(id: " << getId() << ", indices: " << getIndices() << ")";
  }

private:
  /// The ID of the equation.
  Id id;

  /// The set of all indices for the equation.
  IndexSet allIndices;

  /// The set of valid indices for the equation.
  ///
  /// For any initial graph this will be the same as allIndices.
  /// When differentiating an equation along a subset of its indices
  /// validIndices < allIndices.
  IndexSet validIndices;
};

/// An edge from an equation to a variable in the graph.
class Edge final : public Dumpable {
public:
  Edge(const MCIM &mapping)
      : incidenceMatrix(std::make_shared<MCIM>(mapping)),
        mappings(std::make_shared<std::vector<MCIM>>()) {}

  /// List of one-to-one mappings between the indices of the two vertices.
  const std::vector<MCIM> &getMappings() const {
    if (mappings->empty()) {
      *mappings = incidenceMatrix->splitGroups();
    }
    assert(llvm::all_of(*mappings, [](const MCIM &m) {
      return isValidLocalMatchingSolution(m);
    }));
    return *mappings;
  }

  const MCIM &getIncidenceMatrix() const { return *incidenceMatrix; }

  void dump(llvm::raw_ostream &os) const override {
    os << "incidence matrix:\n" << *incidenceMatrix;
  }

  static IndexSet equationIndices(const IndexSet &variableIndices,
                                  const MCIM &incidenceMatrix) {
    return incidenceMatrix.filterColumns(variableIndices).flattenColumns();
  }

  static IndexSet variableIndices(const IndexSet &equationIndices,
                                  const MCIM &incidenceMatrix) {
    return incidenceMatrix.filterRows(equationIndices).flattenRows();
  }

private:
  std::shared_ptr<MCIM> incidenceMatrix;
  std::shared_ptr<std::vector<MCIM>> mappings;
};

/// A variable and a subset of its indices.
template <typename VariableId>
struct VariableSubset {
  VariableId id;
  IndexSet indices;
};

/// An equation and a subset of its indices.
template <typename EquationId>
struct EquationSubset {
  EquationId id;
  IndexSet indices;
};

/// Visitation state for a single `augmentPath`-execution.
template <typename VariableId, typename EquationId>
class Coloring final : public Dumpable {
public:
  void color(const EquationId &equation, const IndexSet &indices) {
    if (indices.empty()) {
      return;
    }
    if (equationColoring.contains(equation)) {
      equationColoring[equation] += indices;
    } else {
      equationColoring.insert_or_assign(equation, indices);
    }
  }

  IndexSet getColoredIndices(const EquationId &equation) const {
    if (auto it = equationColoring.find(equation);
        it != equationColoring.end()) {
      return it->second;
    }
    return IndexSet();
  }

  void color(const VariableId &variable, const IndexSet &indices) {
    if (indices.empty()) {
      return;
    }
    if (variableColoring.contains(variable)) {
      variableColoring[variable] += indices;
    } else {
      variableColoring.insert_or_assign(variable, indices);
    }
  }

  void uncolor(const VariableId &variable, const IndexSet &indices) {
    variableColoring[variable] -= indices;
  }

  IndexSet getColoredIndices(const VariableId &variable) const {
    if (auto it = variableColoring.find(variable);
        it != variableColoring.end()) {
      return it->second;
    }
    return IndexSet();
  }

  auto coloredVariables() const {
    return llvm::make_range(variableColoring.begin(), variableColoring.end());
  }

  auto coloredEquations() const {
    return llvm::make_range(equationColoring.begin(), equationColoring.end());
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "Coloring:\n";
    for (const auto &[id, indices] : variableColoring) {
      os << "  Variable id: " << id << ", indices: " << indices << "\n";
    }
    for (const auto &[id, indices] : equationColoring) {
      os << "  Equation id: " << id << ", indices: " << indices << "\n";
    }
  }

private:
  llvm::DenseMap<VariableId, IndexSet> variableColoring;
  llvm::DenseMap<EquationId, IndexSet> equationColoring;
};


template <typename VariableId, typename EquationId>
class Assignments final : public Dumpable {
public:
  struct AssignmentComponent {
    AssignmentComponent(EquationId e, const IndexSet &indicesV,
                        const internal::MCIM &evMapping)
        : equationId(e), indicesV(indicesV), m(evMapping) {
      assert(internal::isValidLocalMatchingSolution(evMapping) &&
             "Assignment mapping must be one-to-one");
    }
    /// The equation that the variable is assigned to.
    EquationId equationId;
    /// The indices of the variable that are assigned.
    IndexSet indicesV;
    /// The (one-to-one) mapping between the variable and equation.
    internal::MCIM m;
  };

  /// Get the current assignment for a variable.
  llvm::SmallVector<AssignmentComponent> &getAssignment(const VariableId id) {
    if (assignmentsMap.contains(id)) {
      return assignmentsMap[id];
    }
    return assignmentsMap.insert({id, {}}).first->second;
  }

  void assign(const VariableId variableId,
              AssignmentComponent &&assignmentComponent) {
    llvm::SmallVector<AssignmentComponent> &assignment =
        getAssignment(variableId);
    assert(llvm::none_of(assignment,
                         [&](const AssignmentComponent &existingComponent) {
                           return existingComponent.indicesV.overlaps(
                               assignmentComponent.indicesV);
                         }) &&
           "Cannot assign already assigned indices");
    assignment.push_back(std::move(assignmentComponent));
  }

  /// Get all assigned variable indices from the assignment.
  IndexSet allAssignedVariableIndices(const VariableId id) {
    IndexSet result;
    for (const AssignmentComponent &component : getAssignment(id)) {
      result += component.indicesV;
    }
    return result;
  }

  /// Remove empty assignments.
  void removeEmptyAssignments(const VariableId id) {
    llvm::erase_if(getAssignment(id),
                   [](const AssignmentComponent &assignmentComponent) {
                     return assignmentComponent.indicesV.empty();
                   });
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "Assignments:\n";
    for (const auto &[id, assignments] : assignmentsMap) {
      if (assignments.empty()) {
        continue;
      }
      os << "  Variable id: " << id << "\n";
      for (const auto &assignment : assignments) {
        os << "    Assignment: " << assignment.equationId << ", "
           << assignment.indicesV << "\n";
      }
    }
  }

private:
  llvm::DenseMap<VariableId, llvm::SmallVector<AssignmentComponent>>
      assignmentsMap;
};
} // namespace internal::indexReduction

template <typename VariableData, typename EquationData>
class IndexReductionGraph final : public internal::Dumpable {
public:
  using VariableVertex = internal::indexReduction::VariableVertex<VariableData>;
  using VariableId = typename VariableVertex::Id;
  using EquationVertex = internal::indexReduction::EquationVertex<EquationData>;
  using EquationId = typename EquationVertex::Id;

  struct PantelidesResult {
    llvm::DenseMap<EquationId, llvm::SmallVector<IndexSet>> equationDerivatives;
    llvm::DenseMap<VariableId, llvm::SmallVector<IndexSet>> variableDerivatives;
  };

private:
  using Vertex = std::variant<VariableVertex, EquationVertex>;
  using Edge = internal::indexReduction::Edge;
  using Graph = internal::UndirectedGraph<Vertex, Edge>;
  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

  using VariableSubset = internal::indexReduction::VariableSubset<VariableId>;
  using EquationSubset = internal::indexReduction::EquationSubset<EquationId>;
  using Coloring = internal::indexReduction::Coloring<VariableId, EquationId>;

  using Assignments =
      internal::indexReduction::Assignments<VariableId, EquationId>;

  template <typename VertexType>
  VertexType &getVertex(const VertexDescriptor descriptor) {
    Vertex &vertex = graph[descriptor];
    assert(std::holds_alternative<VertexType>(vertex) &&
           "Invalid vertex type for descriptor");
    return std::get<VertexType>(vertex);
  }

  template <typename VertexType>
  const VertexType &getVertex(const VertexDescriptor descriptor) const {
    const Vertex &vertex = graph[descriptor];
    assert(std::holds_alternative<VertexType>(vertex) &&
           "Invalid vertex type for descriptor");
    return std::get<VertexType>(vertex);
  }

  template <typename VertexType>
  auto getDescriptorRange() const {
    std::function filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<VertexType>(vertex);
    };
    return llvm::make_range(graph.verticesBegin(filter),
                            graph.verticesEnd(filter));
  }

  /// Returns a range of the incident edges of [vertex].
  auto getEdgesRange(const VertexDescriptor &vertex) const {
    return llvm::make_range(graph.outgoingEdgesBegin(vertex),
                            graph.outgoingEdgesEnd(vertex));
  }

  VertexDescriptor getDescriptorFromId(const VariableId id) const {
    auto it = variablesMap.find(id);
    assert(it != variablesMap.end() && "Variable not found");
    return it->second;
  }

  VertexDescriptor getDescriptorFromId(const EquationId id) const {
    auto it = equationsMap.find(id);
    assert(it != equationsMap.end() && "Equation not found");
    return it->second;
  }

  /// Get the derivative of a variable if it has one, along with the derived
  /// indices.
  std::optional<VariableSubset>
  getVariableDerivative(const VariableId id) const {
    if (auto it = variableAssociations.find(id);
        it != variableAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  void setVariableDerivative(const VariableId id,
                             const VariableSubset &derivative) {
    variableAssociations.insert_or_assign(id, derivative);
  }

  /// Get the derivative of an equation if it has one.
  std::optional<EquationSubset>
  getEquationDerivative(const EquationId id) const {
    if (auto it = equationAssociations.find(id);
        it != equationAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  void setEquationDerivative(const EquationId id,
                             const EquationSubset &derivative) {
    equationAssociations.insert_or_assign(id, derivative);
  }

  void addVariable(const VariableData &variableData) {
    VariableVertex variable(variableData.getId(), variableData.getIndices());
    VariableId id = variable.getId();
    assert(!variablesMap.contains(id) && "Variable id already in the graph");
    VertexDescriptor variableDescriptor = graph.addVertex(std::move(variable));
    variablesMap.insert({id, variableDescriptor});
  }

  /// Hide the (indices of) variables that have derivatives (of those indices).
  void hideDerivedVariables() {
    for (VertexDescriptor descriptor : getDescriptorRange<VariableVertex>()) {
      VariableVertex &variable = getVertex<VariableVertex>(descriptor);
      auto derivative = getVariableDerivative(variable.getId());
      if (derivative &&
          variable.getVisibleIndices().overlaps(derivative->indices)) {
        variable.hideIndices(derivative->indices);
      }
    }
  }

  /// Attempt to assign indices of v to the e.
  ///
  /// Traverses the edge `evDescriptor` from e to v, and assigns
  /// any reachable, unassigned variable indices to e, updating
  /// the assignment state accordingly.
  ///
  /// Returns the subset of `indicesE` that could not be assigned.
  IndexSet tryAssignEdge(const EdgeDescriptor &evDescriptor, IndexSet indicesE,
                         Assignments &assignments) {
    const auto &e = getVertex<EquationVertex>(evDescriptor.from);
    const auto &v = getVertex<VariableVertex>(evDescriptor.to);

    IndexSet unassignedV = v.getVisibleIndices() -
                           assignments.allAssignedVariableIndices(v.getId());

    for (const auto &m : graph[evDescriptor].getMappings()) {
      const IndexSet &assignableV =
          Edge::variableIndices(indicesE, m).intersect(unassignedV);
      // No indices can be assigned to e.
      if (assignableV.empty()) {
        continue;
      }
      // assignableV is not empty -> assign those indices to i.
      assignments.assign(v.getId(), {e.getId(), assignableV, m});
      unassignedV -= assignableV;
      indicesE -= Edge::equationIndices(assignableV, m);
      if (indicesE.empty()) {
        return IndexSet();
      }
    }

    return indicesE;
  }

  /// Step (2) of the augmenting path algorithm.
  ///
  /// Look for unassigned variables reachable from the given indices of the
  /// equation. Returns the indices where an assignment could not be made.
  IndexSet tryAssignAdjacent(const EquationId eId, IndexSet indicesE,
                             Assignments &assignments) {
    for (EdgeDescriptor evDescriptor :
         getEdgesRange(getDescriptorFromId(eId))) {
      indicesE = tryAssignEdge(evDescriptor, std::move(indicesE), assignments);
      if (indicesE.empty()) {
        break;
      }
    }

    return indicesE;
  }

  /// Attempt to augment the assignment of variable v, at indices `indicesV`.
  ///
  /// Reclaims previously assigned indices of v by recursively traversing
  /// existing assignments and reassigning them elsewhere via augmenting paths.
  /// The assignment and coloring state are updated accordingly.
  ///
  /// Returns the subset of `indicesV` that were successfully reassigned.
  IndexSet tryAugmentVariable(const VariableId v, const IndexSet &indicesV,
                              Assignments &assignments, Coloring &coloring) {
    IndexSet toAssignV;
    for (auto &assignmentComponent : assignments.getAssignment(v)) {
      // The indices of the variable that can be reassigned.
      const IndexSet &candidatesV =
          assignmentComponent.indicesV.intersect(indicesV - toAssignV);
      if (candidatesV.empty()) {
        continue;
      }

      // Traverse the edge from v to e2 (the equation that the indices are
      // currently assigned to), to get the currently assigned indices of
      // e2.
      const IndexSet &candidatesE2 =
          Edge::equationIndices(candidatesV, assignmentComponent.m);

      // Recursively continue the search for an augmenting path from the
      // candidate indices.
      const IndexSet &augmentedE2 =
          candidatesE2 - augmentPath(assignmentComponent.equationId,
                                     candidatesE2, assignments, coloring);
      // Traverse the edge from e2 to v to get the indices of v to assign.
      const IndexSet &augmentedV =
          Edge::variableIndices(augmentedE2, assignmentComponent.m);
      toAssignV += augmentedV;
      // Remove the indices from the old assignment component.
      assignmentComponent.indicesV -= augmentedV;
      // Uncolor the indices of v that were augmented.
      coloring.uncolor(v, augmentedV);
      if (toAssignV == indicesV) {
        break;
      }
    }
    return toAssignV;
  }

  /// Step (3) of the augmenting path algorithm.
  ///
  /// Perform a (recursive) DFS step to build an augmenting path emenating from
  /// equation i, at indices `equationIndices`.
  ///
  /// All variable indices adjacent to `equationIndices` are assumed to be
  /// assigned.
  ///
  /// Returns the indices where no path was found.
  IndexSet tryAugmentAdjacent(const EquationId e, IndexSet indicesE,
                              Assignments &assignments, Coloring &coloring) {
    // Look at each variable, v, accessed by e, at indices that are uncolored.
    for (EdgeDescriptor evDescriptor : getEdgesRange(getDescriptorFromId(e))) {
      VariableVertex &v = getVertex<VariableVertex>(evDescriptor.to);

      // For each one-to-one mapping from e to v, search for augmenting paths.
      for (const auto &mEV : graph[evDescriptor].getMappings()) {
        IndexSet indicesV = Edge::variableIndices(indicesE, mEV)
                                .intersect(v.getVisibleIndices()) -
                            coloring.getColoredIndices(v.getId());
        if (indicesV.empty()) {
          continue;
        }

        // Color the accessed indices of v.
        coloring.color(v.getId(), indicesV);

        // Search for an augmenting path from the indices of v.
        indicesV =
            tryAugmentVariable(v.getId(), indicesV, assignments, coloring);
        if (indicesV.empty()) {
          continue;
        }

        // Create the new assignment.
        assignments.assign(v.getId(), {e, indicesV, mEV});
        // Remove any assignments that have now been made empty.
        assignments.removeEmptyAssignments(v.getId());
        indicesE -= Edge::equationIndices(indicesV, mEV);
        if (indicesE.empty()) {
          return indicesE;
        }
      }
    }

    return indicesE;
  }

  /// Augment path
  ///
  /// - e: the id of the equation vertex to start from
  /// - equationIndices: the indices of the equation vertex to use
  /// - assignments: the current assignments
  ///
  /// Returns the indices for which an augmenting path was not found.
  ///
  /// If the indices are not empty, the colored vertices make up a
  /// structurally singular subset of the system.
  IndexSet augmentPath(const EquationId e, IndexSet indicesE,
                       Assignments &assignments, Coloring &coloring) {
    assert(!coloring.getColoredIndices(e).overlaps(indicesE) &&
           "Equation already colored at indices");
    assert(getVertex<EquationVertex>(getDescriptorFromId(e))
               .getIndices()
               .contains(indicesE) &&
           "Indices must be valid for the equation");

    // Assign free adjacent variables to the e.
    indicesE = tryAssignAdjacent(e, std::move(indicesE), assignments);
    if (indicesE.empty()) {
      return indicesE;
    }

    // Try to create augmenting paths from the indices without an assignment.
    indicesE =
        tryAugmentAdjacent(e, std::move(indicesE), assignments, coloring);

    // Color the indices that were not successfully assigned.
    coloring.color(e, indicesE);
    return indicesE;
  }

  /// Differentiate colored nodes
  ///
  /// - coloring: the variables and equations to differentiate, according to
  /// their coloring state
  /// - variableAssignments: the current variable assignments
  void differentiateNodes(const Coloring &coloring, Assignments &assignments) {
    // Differentiate colored variables
    for (const auto &[v, coloredIndices] : coloring.coloredVariables()) {
      if (auto dv = getVariableDerivative(v)) {
        // The variable is an array that was already differentiated along
        // other indices. Therefore update the derived indices.
        assert(!dv->indices.overlaps(coloredIndices) &&
               "Previously derived indices should not be revisited.");
        dv->indices += coloredIndices;
        setVariableDerivative(v, *dv);
      } else {
        // The variable is a scalar or an array that is not yet
        // differentiated. Therefore, create a new variable, and establish
        // the derivative relationship.
        const VariableData &dvData = differentiateVariable(v);
        addVariable(dvData);
        setVariableDerivative(v,
                              VariableSubset{dvData.getId(), coloredIndices});
      }
    }

    // Differentiate colored equations
    for (const auto &[e, coloredIndices] : coloring.coloredEquations()) {
      // Check if the equation was already differentiated.
      if (auto existingDe = getEquationDerivative(e)) {
        // An equation derivative already exists. Therefore, validate any newly
        // derived indices and update the association.
        assert(!existingDe->indices.overlaps(coloredIndices) &&
               "Previously derived indices should not be revisited.");
        setEquationDerivative(
            e, {existingDe->id, existingDe->indices + coloredIndices});
        getVertex<EquationVertex>(getDescriptorFromId(existingDe->id))
            .validateIndices(coloredIndices);
      } else {
        // Create and add an equation node representing the derivative of e.
        const EquationData &deData = differentiateEquation(e);
        VertexDescriptor deDescriptor = graph.addVertex(EquationVertex(
            deData.getId(), deData.getIndices(), coloredIndices));
        EquationVertex &de = getVertex<EquationVertex>(deDescriptor);
        equationsMap[de.getId()] = deDescriptor;

        // Add edges to the variables accessed by the original equation,
        // and their derivatives.
        llvm::DenseMap<VariableId, internal::MCIM> toAdd;
        for (EdgeDescriptor edgeDescriptor :
             getEdgesRange(getDescriptorFromId(e))) {
          const auto &v = getVertex<VariableVertex>(edgeDescriptor.to).getId();
          const auto &evMapping = graph[edgeDescriptor].getIncidenceMatrix();

          auto [vMapping, inseted] = toAdd.try_emplace(v, evMapping);
          if (!inseted) {
            vMapping->second += evMapping;
          }

          auto dv = getVariableDerivative(v);
          assert(dv && "All adjacent variables should be derived.");
          auto [dvMapping, inserted] = toAdd.try_emplace(dv->id, evMapping);
          if (!inserted) {
            dvMapping->second += evMapping;
          }
        }
        // Add edges to the graph.
        for (auto &[v, mapping] : toAdd) {
          graph.addEdge(deDescriptor, getDescriptorFromId(v), Edge(mapping));
        }

        setEquationDerivative(e, {de.getId(), coloredIndices});
      }
    }

    // Assign derivatives of colored variables to the
    // derivatives of their assigned equations.
    for (const auto &[v, coloredIndices] : coloring.coloredVariables()) {
      IndexSet unassignedIndicesV = coloredIndices;
      auto dv = getVariableDerivative(v);
      assert(dv && "Variable derivative not found");

      for (const auto &assignmentComponent : assignments.getAssignment(v)) {
        // The indices of dv are a subset of the indices of v.
        // Therefore we have to check that the indices of an assignment
        // are valid for dv before we use it.
        const IndexSet &assignableV =
            unassignedIndicesV.intersect(assignmentComponent.indicesV);
        if (assignableV.empty()) {
          continue;
        }
        unassignedIndicesV -= assignableV;
        // Because the v has been differentiated, all adjacent equations
        // must have been differentiated as well.
        auto de = getEquationDerivative(assignmentComponent.equationId);
        assert(de && "Equation derivative not found");
        assignments.assign(dv->id,
                           {de->id, assignableV, assignmentComponent.m});
      }
      assert(unassignedIndicesV.empty() && "Did not assign all indices.");
    }
  }

  /// Collect the derivative-chains for the original equations and variables.
  PantelidesResult buildPantelidesResult() {
    PantelidesResult result;
    // Collect the derivation-chain for each original equation
    for (EquationId equationId : initialEquations) {
      llvm::SmallVector<IndexSet> derivativeChain;
      auto derivative = getEquationDerivative(equationId);
      while (derivative) {
        derivativeChain.push_back(derivative->indices);
        derivative = getEquationDerivative(derivative->id);
      }
      if (!derivativeChain.empty()) {
        result.equationDerivatives.try_emplace(equationId,
                                               std::move(derivativeChain));
      }
    }

    // Collect the derivation-chain for each original variable.
    // The initial variables may contain both a variable and its derivative.
    // In those cases only the former derivative chain is kept, as the latter
    // chain is just a subset.
    llvm::DenseSet<VariableId> seenDerivatives;
    for (VariableId variableId : initialVariables) {
      llvm::SmallVector<IndexSet> derivativeChain;
      std::optional<VariableSubset> derivative =
          getVariableDerivative(variableId);
      while (derivative) {
        derivativeChain.push_back(derivative->indices);
        seenDerivatives.insert(derivative->id);
        derivative = getVariableDerivative(derivative->id);
      }
      if (!derivativeChain.empty()) {
        result.variableDerivatives.try_emplace(variableId,
                                               std::move(derivativeChain));
      }
    }
    // Remove the redundant derivative chains.
    for (const auto &seenId : seenDerivatives) {
      result.variableDerivatives.erase(seenId);
    }

    return result;
  }

public:
  IndexReductionGraph(const std::function<const VariableData &(VariableId)>
                          &differentiateVariable,
                      const std::function<const EquationData &(EquationId)>
                          &differentiateEquation)
      : differentiateVariable(differentiateVariable),
        differentiateEquation(differentiateEquation) {}

  /// Data for adding equations to the graph.
  /// Consists of an equation paired with the list of variable/access function
  /// pairs
  using EquationWithAccesses =
      std::pair<EquationData *,
                llvm::SmallVector<
                    std::pair<VariableId, std::unique_ptr<AccessFunction>>>>;
  /// Data for adding variable derivatives to the graph.
  /// Consists of a variable id, the id of the derivative, and the indices for
  /// which the variable is derived.
  using VariableDerivative = std::tuple<VariableId, VariableId, IndexSet>;

  /// Initialize the graph with the given variables, variable derivatives, and
  /// equations.
  void
  initialize(const llvm::SmallVector<VariableData *> &variables,
             const llvm::SmallVector<VariableDerivative> &variableDerivatives,
             const llvm::SmallVector<EquationWithAccesses> &equations) {
    for (const VariableData *variable : variables) {
      addVariable(*variable);
      initialVariables.push_back(variable->getId());
    }

    for (const auto &[variableId, derivativeId, derivedIndices] :
         variableDerivatives) {
      setVariableDerivative(variableId,
                            VariableSubset{derivativeId, derivedIndices});
    }

    for (const auto &[equation, accesses] : equations) {
      EquationVertex eq(equation->getId(), equation->getIndices(),
                        equation->getIndices());
      EquationId id = eq.getId();
      assert(!equationsMap.contains(id) && "Equation id already in graph");
      VertexDescriptor equationDescriptor = graph.addVertex(std::move(eq));
      equationsMap[id] = equationDescriptor;
      initialEquations.push_back(id);

      const IndexSet &equationRanges =
          getVertex<EquationVertex>(equationDescriptor).getIndices();

      // Build the mappings for each accessed variable.
      llvm::DenseMap<VariableId, internal::MCIM> mappings;
      for (const auto &[variableId, accessFunction] : accesses) {
        internal::MCIM mapping(
            equationRanges,
            getVertex<VariableVertex>(getDescriptorFromId(variableId))
                .getIndices());
        mapping.apply(*accessFunction);

        auto [existingMapping, inserted] =
            mappings.try_emplace(variableId, mapping);
        if (!inserted) {
          existingMapping->second += mapping;
        }
      }

      // Add one edge per accessed variable.
      for (auto &[variableId, mapping] : mappings) {
        graph.addEdge(equationDescriptor, getDescriptorFromId(variableId),
                      Edge(mapping));
      }
    }

    // Hide indices of initially derived variables.
    hideDerivedVariables();
  }

  /// Apply the array-aware pantelides algorithm to the graph.
  ///
  /// Returns the chain of derivations to be performed for each equation and
  /// variable.
  PantelidesResult pantelides() {
    Assignments assignments;
    for (EquationId e : initialEquations) {
      LLVM_DEBUG(llvm::dbgs() << "Starting from equation: " << e << "\n");
      IndexSet indicesE =
          getVertex<EquationVertex>(getDescriptorFromId(e)).getIndices();

      while (true) {
        LLVM_DEBUG({
          llvm::dbgs() << "---\nAugmenting path for equation: " << e
                       << ", indices: " << indicesE << "\n";
          assignments.dump(llvm::dbgs());
        });
        Coloring coloring;
        indicesE = augmentPath(e, std::move(indicesE), assignments, coloring);
        if (indicesE.empty()) {
          break;
        }

        LLVM_DEBUG({
          dump(llvm::dbgs());
          llvm::dbgs() << "Augmenting path not found\n";
          coloring.dump(llvm::dbgs());
          llvm::dbgs() << "---\n";
        });

        differentiateNodes(coloring, assignments);
        hideDerivedVariables();
        e = getEquationDerivative(e)->id;
      }
    }

    return buildPantelidesResult();
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "Index Reduction Graph:\n";

    os << " Variables:\n";
    for (const auto &descriptor : getDescriptorRange<VariableVertex>()) {
      const auto &v = getVertex<VariableVertex>(descriptor);
      os << "  ", v.dump(os);
      if (auto dv = getVariableDerivative(v.getId())) {
        os << " -der-> " << dv->id << " @ " << dv->indices;
      }
      os << "\n";
    }

    os << " Equations:\n";
    for (const auto &descriptor : getDescriptorRange<EquationVertex>()) {
      const auto &e = getVertex<EquationVertex>(descriptor);
      os << "  ", e.dump(os);
      if (auto de = getEquationDerivative(e.getId())) {
        os << " -der-> " << de->id << " @ " << de->indices;
      }
      os << "\n";
      for (const auto &edgeDescriptor : getEdgesRange(descriptor)) {
        os << "   " << getVertex<VariableVertex>(edgeDescriptor.to).getId()
           << " - ";
        graph[edgeDescriptor].dump(os);
      }
    }
    os << "---\n";
  };

private:
  Graph graph;

  std::function<const VariableData &(const VariableId)> differentiateVariable;
  std::function<const EquationData &(const EquationId)> differentiateEquation;

  llvm::DenseMap<VariableId, VertexDescriptor> variablesMap;
  llvm::DenseMap<EquationId, VertexDescriptor> equationsMap;

  /// THe initial variables and equations added to the graph.
  llvm::SmallVector<VariableId> initialVariables;
  llvm::SmallVector<EquationId> initialEquations;

  /// Associates a variable with its derivative.
  /// v -> (v', indices of v)
  llvm::DenseMap<VariableId, VariableSubset> variableAssociations;

  /// Associates an equation with its derivative.
  /// e -> (e', indices of e)
  llvm::DenseMap<EquationId, EquationSubset> equationAssociations;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_INDEXREDUCTION_H
