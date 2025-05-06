#ifndef MARCO_MODELING_INDEXREDUCTION_H
#define MARCO_MODELING_INDEXREDUCTION_H

#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/LocalMatchingSolutionsImpl.h"
#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace marco::modeling {

using mlir::bmodelica::bridge::EquationBridge;
using mlir::bmodelica::bridge::VariableBridge;

namespace internal::indexReduction {

/// A variable vertex in the graph.
class VariableVertex final : public Dumpable {
public:
  using Id = VariableBridge::Id;

  Id getId() const { return id; }
  const IndexSet &getIndices() const { return indices; }

  void hideIndices(const IndexSet &indices) { visibleIndices -= indices; }
  const IndexSet &getVisibleIndices() const { return visibleIndices; }

  explicit VariableVertex(const VariableBridge &bridge)
      : id(bridge.getId()), indices(bridge.getIndices()),
        visibleIndices(indices) {}

  void dump(llvm::raw_ostream &os) const override {
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
class EquationVertex final : public Dumpable {
public:
  using Id = EquationBridge::Id;

  Id getId() const { return id; }

  const IndexSet &getIndices() const { return validIndices; }

  const IndexSet &getAllIndices() const { return allIndices; }

  void validateIndices(const IndexSet &indices) {
    assert(allIndices.contains(indices) && "Invalid indices for equation");
    validIndices += indices;
  }

  EquationVertex(const EquationBridge &bridge, const IndexSet &validIndices)
      : id(bridge.getId()), allIndices(bridge.getIndices()),
        validIndices(validIndices) {
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
  Edge(const VariableVertex::Id variableId, const IndexSet &equationRanges,
       const IndexSet &variableRanges, const AccessFunction &accessFunction)
      : incidenceMatrix(std::make_shared<MCIM>(equationRanges, variableRanges)),
        mappings(std::make_shared<std::vector<MCIM>>()) {
    incidenceMatrix->apply(accessFunction);
  }

  const std::vector<MCIM> &getMappings() const {
    if (mappings->empty()) {
      *mappings = incidenceMatrix->splitGroups();
    }
    return *mappings;
  }

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
struct VariableSubset {
  VariableVertex::Id id;
  IndexSet indices;
};

/// An equation and a subset of its indices.
struct EquationSubset {
  EquationVertex::Id id;
  IndexSet indices;
};

/// Visitation state for a single `augmentPath`-execution.
class Coloring final {
public:
  void color(const EquationVertex::Id &equation, const IndexSet &indices) {
    if (indices.empty()) {
      return;
    }
    if (equationColoring.contains(equation)) {
      equationColoring[equation] += indices;
    } else {
      equationColoring.insert_or_assign(equation, indices);
    }
  }

  IndexSet getColoredIndices(const EquationVertex::Id &equation) const {
    if (auto it = equationColoring.find(equation);
        it != equationColoring.end()) {
      return it->second;
    }
    return IndexSet();
  }

  void color(const VariableVertex &variable, const IndexSet &indices) {
    if (indices.empty()) {
      return;
    }
    assert(variable.getIndices().contains(indices) &&
           "Colored indices must be valid for the variable");
    if (variableColoring.contains(variable.getId())) {
      variableColoring[variable.getId()] += indices;
    } else {
      variableColoring.insert_or_assign(variable.getId(), indices);
    }
  }

  void uncolor(const VariableVertex &variable, const IndexSet &indices) {
    assert(variable.getIndices().contains(indices) &&
           "Colored indices must be valid for the variable");
    assert(variableColoring.contains(variable.getId()) &&
           "Variable must be colored before uncoloring");
    variableColoring[variable.getId()] -= indices;
  }

  IndexSet getColoredIndices(const VariableVertex &variable) const {
    if (auto it = variableColoring.find(variable.getId());
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

private:
  llvm::DenseMap<VariableVertex::Id, IndexSet> variableColoring;
  llvm::DenseMap<EquationVertex::Id, IndexSet> equationColoring;
};

} // namespace internal::indexReduction

class IndexReductionGraph final : public internal::Dumpable {
public:
  using VariableVertex = internal::indexReduction::VariableVertex;
  using EquationVertex = internal::indexReduction::EquationVertex;

  struct PantelidesResult {
    llvm::DenseMap<EquationVertex::Id, llvm::SmallVector<IndexSet>>
        equationDerivatives;
    llvm::DenseMap<VariableVertex::Id, llvm::SmallVector<IndexSet>>
        variableDerivatives;
  };

private:
  using Vertex = std::variant<VariableVertex, EquationVertex>;
  using Edge = internal::indexReduction::Edge;
  using Graph = internal::UndirectedGraph<Vertex, Edge>;
  using VertexDescriptor = Graph::VertexDescriptor;
  using EdgeDescriptor = Graph::EdgeDescriptor;

  using VariableSubset = internal::indexReduction::VariableSubset;
  using EquationSubset = internal::indexReduction::EquationSubset;
  using Coloring = internal::indexReduction::Coloring;

  struct AssignmentComponent {
    AssignmentComponent(EquationVertex::Id equationId, const IndexSet &indicesE,
                        const IndexSet &indicesV,
                        const internal::MCIM &evMapping)
        : equationId(equationId), indicesE(indicesE), indicesV(indicesV),
          incidenceMatrix(evMapping) {
      assert(indicesE.flatSize() == indicesV.flatSize() &&
             "Indices of equation and variable must have the same size");
      assert(internal::isValidLocalMatchingSolution(evMapping) &&
             "Assignment mapping must be one-to-one");
    }

    /// The equation that the variable is assigned to.
    EquationVertex::Id equationId;
    /// The indices of the equation that are assigned.
    IndexSet indicesE;
    /// The indices of the variable that are assigned.
    IndexSet indicesV;
    /// The mapping between the variable and equation.
    internal::MCIM incidenceMatrix;
  };
  /// Map from variables to their assignments.
  using VariableAssignments =
      llvm::DenseMap<VariableVertex::Id,
                     llvm::SmallVector<AssignmentComponent>>;

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

  VertexDescriptor getDescriptorFromId(const VariableVertex::Id id) const {
    auto it = variablesMap.find(id);
    assert(it != variablesMap.end() && "Variable not found");
    return it->second;
  }

  VertexDescriptor getDescriptorFromId(const EquationVertex::Id id) const {
    auto it = equationsMap.find(id);
    assert(it != equationsMap.end() && "Equation not found");
    return it->second;
  }

  /// Get the derivative of a variable if it has one, along with the derived
  /// indices.
  std::optional<VariableSubset>
  getVariableDerivative(const VariableVertex::Id id) const {
    if (auto it = variableAssociations.find(id);
        it != variableAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  void setVariableDerivative(const VariableVertex::Id id,
                             const VariableSubset &derivative) {
    variableAssociations.insert_or_assign(id, derivative);
  }

  /// Get the derivative of an equation if it has one.
  std::optional<EquationSubset>
  getEquationDerivative(const EquationVertex::Id id) const {
    if (auto it = equationAssociations.find(id);
        it != equationAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  void setEquationDerivative(const EquationVertex::Id id,
                             const EquationSubset &derivative) {
    equationAssociations.insert_or_assign(id, derivative);
  }

  void addVariable(const VariableBridge &variableBridge) {
    VariableVertex variable(variableBridge);
    VariableVertex::Id id = variable.getId();
    assert(!variablesMap.contains(id) && "Variable id already in the graph");
    VertexDescriptor variableDescriptor = graph.addVertex(std::move(variable));
    variablesMap[id] = variableDescriptor;
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

  /// Get the assignment for a variable, creating it if it does not exist.
  static llvm::SmallVector<AssignmentComponent> &
  getAssignment(VariableVertex::Id id, VariableAssignments &assignments) {
    auto existingAssignment = assignments.find(id);
    if (existingAssignment == assignments.end()) {
      existingAssignment = assignments.insert({id, {}}).first;
    }
    return existingAssignment->getSecond();
  }

  /// Get all assigned variable indices from the assignment.
  static IndexSet allAssignedVariableIndices(
      const llvm::SmallVector<AssignmentComponent> &assignments) {
    IndexSet result;
    for (const AssignmentComponent &component : assignments) {
      result += component.indicesV;
    }
    return result;
  }

  /// Get all assigned equation indices from the assignment.
  static IndexSet allAssignedEquationIndices(
      const llvm::SmallVector<AssignmentComponent> &assignments) {
    IndexSet result;
    for (const AssignmentComponent &component : assignments) {
      result += component.indicesE;
    }
    return result;
  }

  IndexSet tryAssignEdge(const EdgeDescriptor &evDescriptor, IndexSet indicesE,
                         VariableAssignments &assignments) {
    const auto &e = getVertex<EquationVertex>(evDescriptor.from);
    const auto &v = getVertex<VariableVertex>(evDescriptor.to);

    // Get the assignment for v.
    llvm::SmallVector<AssignmentComponent> &assignmentV =
        getAssignment(v.getId(), assignments);
    IndexSet assignedV = allAssignedVariableIndices(assignmentV);

    for (const auto &m : graph[evDescriptor].getMappings()) {
      const IndexSet &indicesV =
          Edge::variableIndices(indicesE, m).intersect(v.getVisibleIndices());
      assert(indicesV.flatSize() <= indicesE.flatSize() &&
             "Each equation point should map to at most one variable point.");
      const IndexSet &assignableV = indicesV - assignedV;
      // No indices can be assigned to e.
      if (assignableV.empty()) {
        continue;
      }
      const IndexSet &assignmentE =
          m.filterColumns(assignableV).flattenColumns();
      // assignableV is not empty -> assign those indices to i.
      assignmentV.emplace_back(e.getId(), assignmentE, assignableV, m);
      assignedV += assignableV;
      indicesE -= assignmentE;
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
  IndexSet tryAssignAdjacent(const EquationVertex::Id eId, IndexSet indicesE,
                             VariableAssignments &assignments) {
    for (EdgeDescriptor evDescriptor :
         getEdgesRange(getDescriptorFromId(eId))) {
      indicesE = tryAssignEdge(evDescriptor, std::move(indicesE), assignments);
      if (indicesE.empty()) {
        break;
      }
    }

    return indicesE;
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
  IndexSet tryAugmentAdjacent(const EquationVertex::Id eId, IndexSet indicesE,
                              VariableAssignments &assignments,
                              Coloring &coloring) {
    // Look at each variable, v, accessed by e, that is also uncolored.
    for (EdgeDescriptor evDescriptor :
         getEdgesRange(getDescriptorFromId(eId))) {
      VariableVertex &v = getVertex<VariableVertex>(evDescriptor.to);

      // The current assignment for the indices of v.
      llvm::SmallVector<AssignmentComponent> &existingAssignment =
          getAssignment(v.getId(), assignments);

      // For each one-to-one mapping from e to v, search for augmenting paths.
      for (const auto &m : graph[evDescriptor].getMappings()) {
        IndexSet indicesV = Edge::variableIndices(indicesE, m)
                                .intersect(v.getVisibleIndices()) -
                            coloring.getColoredIndices(v);
        assert(indicesE.flatSize() >= indicesV.flatSize() &&
               "Each equation point should map to at most one variable point.");
        if (indicesV.empty()) {
          continue;
        }

        // Color the accessed indices of v.
        coloring.color(v, indicesV);

        // The new assignment components that will be added.
        llvm::SmallVector<AssignmentComponent> newAssignments;

        for (AssignmentComponent &assignmentComponent : existingAssignment) {
          // The indices of the variable that can be reassigned.
          const IndexSet &candidatesV =
              assignmentComponent.indicesV.intersect(indicesV);
          // No reassignments can be made
          if (candidatesV.empty()) {
            continue;
          }

          // Traverse the edge from v to e2 (the equation that the indices are
          // currently assigned to), to get the currently assigned indices of
          // e2.
          const IndexSet &candidatesE2 = Edge::equationIndices(
              candidatesV, assignmentComponent.incidenceMatrix);
          assert(candidatesE2.flatSize() == candidatesV.flatSize() &&
                 "The mapping in an assignment should be 1-to-1");

          // Recursively continue the search for an augmenting path from the
          // candidate indices.
          const IndexSet &augmentedE2 =
              candidatesE2 - augmentPath(assignmentComponent.equationId,
                                         candidatesE2, assignments, coloring);
          // Traverse the edge from e2 to v to get the indices of v to assign.
          const IndexSet &augmentedV = Edge::variableIndices(
              augmentedE2, assignmentComponent.incidenceMatrix);

          // Create a new assignment component for the successful indices.
          const IndexSet &augmentedE = Edge::equationIndices(augmentedV, m);
          indicesE -= augmentedE;
          newAssignments.push_back(
              AssignmentComponent(eId, augmentedE, augmentedV, m));
          // Remove the indices from the old assignment component.
          assignmentComponent.indicesV -= augmentedV;

          // Uncolor the indices of v that were augmented.
          coloring.uncolor(v, augmentedV);
          // Remove the indices of v that were augmented.
          indicesV -= augmentedV;
          if (indicesV.empty()) {
            break;
          }
        }
        // Remove empty assignment components.
        llvm::erase_if(existingAssignment,
                       [](const AssignmentComponent &assignmentComponent) {
                         return assignmentComponent.indicesV.empty();
                       });
        // Add the new assignments.
        existingAssignment.append(newAssignments);

        if (indicesE.empty()) {
          break;
        }
      }
      if (indicesE.empty()) {
        break;
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
  IndexSet augmentPath(const EquationVertex::Id e, IndexSet indicesE,
                       VariableAssignments &assignments, Coloring &coloring) {
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
  void differentiateNodes(const Coloring &coloring,
                          VariableAssignments &variableAssignments) {
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
        const VariableBridge &dvBridge = differentiateVariable(v);
        addVariable(dvBridge);
        setVariableDerivative(v,
                              VariableSubset{dvBridge.getId(), coloredIndices});
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
        VertexDescriptor deDescriptor = graph.addVertex(
            EquationVertex(differentiateEquation(e), coloredIndices));
        EquationVertex &de = getVertex<EquationVertex>(deDescriptor);
        equationsMap[de.getId()] = deDescriptor;

        // Add edges to the variables accessed by the original equation,
        // and their derivatives.
        llvm::DenseSet<VariableVertex::Id> addedEdges;
        for (EdgeDescriptor edgeDescriptor :
             getEdgesRange(getDescriptorFromId(e))) {
          const Edge &edge = graph[edgeDescriptor];
          const auto &v = getVertex<VariableVertex>(edgeDescriptor.to);

          if (!addedEdges.contains(v.getId())) {
            graph.addEdge(deDescriptor, edgeDescriptor.to, edge);
            addedEdges.insert(v.getId());
          }

          if (auto dv = getVariableDerivative(v.getId());
              dv && !addedEdges.contains(dv->id)) {
            graph.addEdge(deDescriptor, getDescriptorFromId(dv->id), edge);
            addedEdges.insert(dv->id);
          }
        }

        setEquationDerivative(e, {de.getId(), coloredIndices});
      }
    }

    // Assign derivatives of colored variables to the
    // derivatives of their assigned equations.
    for (const auto &[jId, coloredIndices] : coloring.coloredVariables()) {
      IndexSet unassignedIndicesJ = coloredIndices;
      llvm::SmallVector<AssignmentComponent> djAssignment;
      for (const AssignmentComponent &assignmentComponent :
           variableAssignments[jId]) {
        // The indices of dj are a subset of the indices of j.
        // Therefore we have to check that the indices of an assignment
        // are valid for dj before we use it.
        const IndexSet &assignableJ =
            unassignedIndicesJ.intersect(assignmentComponent.indicesV);
        if (!assignableJ.empty()) {
          unassignedIndicesJ -= assignableJ;
          IndexSet assignableK = Edge::equationIndices(
              assignableJ, assignmentComponent.incidenceMatrix);
          assert(assignableK.flatSize() == assignableJ.flatSize() &&
                 "The mapping in an assignment should be 1-to-1");
          djAssignment.push_back(AssignmentComponent(
              getEquationDerivative(assignmentComponent.equationId)->id,
              assignableK, assignableJ, assignmentComponent.incidenceMatrix));
        }
      }

      assert(unassignedIndicesJ.empty() && "Did not assign all indices.");
      auto dj = getVariableDerivative(jId);
      assert(dj && "Variable derivative not found");
      if (variableAssignments.contains(dj->id)) {
        variableAssignments[dj->id].append(djAssignment);
      } else {
        variableAssignments.insert({dj->id, std::move(djAssignment)});
      }
    }
  }

  /// Collect the derivative-chains for the original equations and variables.
  PantelidesResult buildPantelidesResult() {
    PantelidesResult result;
    // Collect the derivation-chain for each original equation
    for (EquationVertex::Id equationId : initialEquations) {
      llvm::SmallVector<IndexSet> derivativeChain;
      std::optional<EquationSubset> derivative =
          getEquationDerivative(equationId);
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
    // is just a subset of it.
    llvm::DenseSet<VariableVertex::Id> seenDerivatives;
    for (VariableVertex::Id variableId : initialVariables) {
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
  IndexReductionGraph(const std::function<VariableBridge &(VariableBridge::Id)>
                          &differentiateVariable,
                      const std::function<EquationBridge &(EquationBridge::Id)>
                          &differentiateEquation)
      : differentiateVariable(differentiateVariable),
        differentiateEquation(differentiateEquation) {}

  /// Data bundle for adding equations to the graph.
  /// Consists of an equation paired with the list of variable/access function
  /// pairs
  using EquationWithAccesses =
      std::pair<EquationBridge *,
                llvm::SmallVector<std::pair<VariableVertex::Id,
                                            std::unique_ptr<AccessFunction>>>>;
  using VariableDerivative =
      std::tuple<VariableVertex::Id, VariableVertex::Id, IndexSet>;

  /// Initialize the graph with the given variables, variable derivatives, and
  /// equations.
  void
  initialize(const llvm::SmallVector<VariableBridge *> &variables,
             const llvm::SmallVector<VariableDerivative> &variableDerivatives,
             const llvm::SmallVector<EquationWithAccesses> &equations) {
    for (const VariableBridge *variable : variables) {
      addVariable(*variable);
      initialVariables.push_back(variable->getId());
    }

    for (const auto &[variableId, derivativeId, derivedIndices] :
         variableDerivatives) {
      setVariableDerivative(variableId,
                            VariableSubset{derivativeId, derivedIndices});
    }

    for (const auto &[equationBridge, accesses] : equations) {
      EquationVertex eq(*equationBridge, equationBridge->getIndices());
      EquationVertex::Id id = eq.getId();
      assert(!equationsMap.contains(id) && "Equation id already in graph");
      VertexDescriptor equationDescriptor = graph.addVertex(std::move(eq));
      equationsMap[id] = equationDescriptor;
      initialEquations.push_back(id);

      const EquationVertex &equation =
          getVertex<EquationVertex>(equationDescriptor);
      IndexSet equationRanges = equation.getIndices();

      for (const auto &[variableId, accessFunction] : accesses) {
        VertexDescriptor variableDescriptor = getDescriptorFromId(variableId);
        const VariableVertex &variable =
            getVertex<VariableVertex>(variableDescriptor);

        for (const MultidimensionalRange &range :
             llvm::make_range(variable.getIndices().rangesBegin(),
                              variable.getIndices().rangesEnd())) {
          graph.addEdge(equationDescriptor, variableDescriptor,
                        {variable.getId(), equationRanges, IndexSet(range),
                         *accessFunction});
        }
      }
    }
  }

  /// Apply the array-aware pantelides algorithm to the graph.
  ///
  /// Returns the chain of derivations to be performed for each equation.
  PantelidesResult pantelides() {
    // Hide indices of initially derived variables.
    hideDerivedVariables();
    VariableAssignments assignments;
    for (EquationVertex::Id e : initialEquations) {
      IndexSet indicesE =
          getVertex<EquationVertex>(getDescriptorFromId(e)).getIndices();

      while (true) {
        Coloring coloring;
        indicesE = augmentPath(e, std::move(indicesE), assignments, coloring);
        if (indicesE.empty()) {
          break;
        }
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
    for (const auto &descriptor : getDescriptorRange<EquationVertex>()) {
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

  std::function<VariableBridge &(const VariableBridge::Id)>
      differentiateVariable;
  std::function<EquationBridge &(const EquationBridge::Id)>
      differentiateEquation;

  llvm::DenseMap<VariableVertex::Id, VertexDescriptor> variablesMap;
  llvm::DenseMap<EquationVertex::Id, VertexDescriptor> equationsMap;

  /// THe initial variables and equations added to the graph.
  llvm::SmallVector<VariableVertex::Id> initialVariables;
  llvm::SmallVector<EquationVertex::Id> initialEquations;

  /// Associates a variable with its derivative.
  /// var -> (var', indices of var)
  llvm::DenseMap<VariableVertex::Id, VariableSubset> variableAssociations;

  /// Associates an equation with its derivative.
  /// eq -> eq'
  llvm::DenseMap<EquationVertex::Id, EquationSubset> equationAssociations;
};

} // namespace marco::modeling

#endif // MARCO_MODELING_INDEXREDUCTION_H
