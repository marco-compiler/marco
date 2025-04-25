#ifndef MARCO_MODELING_INDEXREDUCTION_H
#define MARCO_MODELING_INDEXREDUCTION_H

#include "marco/Dialect/BaseModelica/Transforms/Modeling/EquationBridge.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/VariableBridge.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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

/// An edge from an equation to a variable in the graph. Thin wrapper around
/// an `MCIM`.
class Edge final : public Dumpable {
public:
  Edge(const VariableVertex::Id variableId, const IndexSet &equationRanges,
       const IndexSet &variableRanges, const AccessFunction &accessFunction)
      : incidenceMatrix(
            std::make_shared<MCIM>(equationRanges, variableRanges)) {
    incidenceMatrix->apply(accessFunction);
  }

  /// Get the indices of the variable that are accessed by the equation.
  IndexSet accessedVariableIndices() const {
    return incidenceMatrix->flattenRows();
  }

  /// Traverse edge from equation to variable.
  IndexSet variableIndices(const IndexSet &equationIndices,
                           const VariableVertex &variable) const {
    return incidenceMatrix->filterRows(equationIndices)
        .flattenRows()
        .intersect(variable.getVisibleIndices());
  }

  /// Traverse edge from variable to equation.
  IndexSet equationIndices(const IndexSet &variableIndices) const {
    return incidenceMatrix->filterColumns(variableIndices).flattenColumns();
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "incidence matrix:\n" << *incidenceMatrix;
  }

private:
  std::shared_ptr<MCIM> incidenceMatrix;
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
class Coloring final : public Dumpable {
public:
  void color(const EquationVertex &equation, const IndexSet &indices) {
    if (indices.empty()) {
      return;
    }
    assert(equation.getIndices().contains(indices) &&
           "Colored indices must be valid for the equation");
    if (equationColoring.contains(equation.getId())) {
      equationColoring[equation.getId()] += indices;
    } else {
      equationColoring.insert_or_assign(equation.getId(), indices);
    }
  }

  IndexSet getColoredIndices(const EquationVertex &equation) const {
    if (auto it = equationColoring.find(equation.getId());
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

  void dump(llvm::raw_ostream &os) const override {
    os << "Coloring:\n colored equations:\n";
    for (const auto &[id, coloredIndices] : coloredEquations()) {
      os << "  " << id << " @ " << coloredIndices << "\n";
    }
    os << " colored variables:\n";
    for (const auto &[id, coloredIndices] : coloredVariables()) {
      os << "  " << id << " @ " << coloredIndices << "\n";
    }
  }

private:
  llvm::DenseMap<VariableVertex::Id, IndexSet> variableColoring;
  llvm::DenseMap<EquationVertex::Id, IndexSet> equationColoring;
};

// TODO: Implement this. But think about it first.
class VariableAssignments final : public Dumpable {
  void dump(llvm::raw_ostream &os) const override {
    assert(false && "VariableAssignments not implemented");
  }

private:
  llvm::DenseMap<VariableVertex::Id,
                 llvm::DenseMap<EquationVertex::Id, IndexSet>>
      assignments;
};

} // namespace internal::indexReduction

class IndexReductionGraph final : public internal::Dumpable {
public:
  using VariableVertex = internal::indexReduction::VariableVertex;
  using EquationVertex = internal::indexReduction::EquationVertex;

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
    /// The equation that the variable is assigned to.
    EquationVertex::Id equationId;
    /// The indices of the variable that are assigned.
    IndexSet indices;
    /// The edge between the variable and the equation.
    EdgeDescriptor edgeDescriptor;
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

  template <typename VertexType>
  auto getVertexRange() const {
    return llvm::map_range(getDescriptorRange<VertexType>(),
                           [&](VertexDescriptor descriptor) {
                             return std::ref(getVertex<VertexType>(descriptor));
                           });
  }

  /// Returns a range of the incident edges of [vertex].
  auto getEdgesRange(const VertexDescriptor &vertex) const {
    return llvm::make_range(graph.outgoingEdgesBegin(vertex),
                            graph.outgoingEdgesEnd(vertex));
  }

  bool hasVariableWithId(const VariableVertex::Id id) const {
    return variablesMap.find(id) != variablesMap.end();
  }

  bool hasEquationWithId(const EquationVertex::Id id) const {
    return equationsMap.find(id) != equationsMap.end();
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

  /// Get all assigned indices from the assignment.
  static IndexSet allAssignedIndices(
      const llvm::SmallVector<AssignmentComponent> &assignments) {
    IndexSet result;
    for (const AssignmentComponent &component : assignments) {
      result += component.indices;
    }
    return result;
  }

  /// Step (2) of the augmenting path algorithm.
  ///
  /// Look for unassigned variables reachable from the given indices of the
  /// equation. Returns the indices where an assignment could not be made.
  IndexSet tryAssignAdjacent(const EquationVertex::Id eId,
                             const IndexSet &equationIndices,
                             VariableAssignments &assignments) {
    IndexSet indicesE = equationIndices;
    for (EdgeDescriptor evDescriptor :
         getEdgesRange(getDescriptorFromId(eId))) {
      const Edge &ev = graph[evDescriptor];
      const auto &v = getVertex<VariableVertex>(evDescriptor.to);
      const IndexSet indicesV = ev.variableIndices(indicesE, v);
      if (indicesV.empty()) {
        continue;
      }

      // Get the assignment for j.
      llvm::SmallVector<AssignmentComponent> &assignmentV =
          getAssignment(v.getId(), assignments);
      // Contains the accessed indices of v that aren't yet assigned.
      const IndexSet &assignableV = indicesV - allAssignedIndices(assignmentV);

      // No indices can be assigned to e.
      if (assignableV.empty()) {
        continue;
      }

      // assignableV is not empty -> assign those indices to i.
      assignmentV.push_back(
          AssignmentComponent{eId, assignableV, evDescriptor});
      // Remove the assigned indices from the search.
      indicesE -= ev.equationIndices(assignableV);
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
  /// Returns the indices where no path was found.
  IndexSet tryAugmentAdjacent(const EquationVertex::Id eId,
                              const IndexSet &equationIndices,
                              VariableAssignments &assignments,
                              Coloring &coloring) {
    // Search for an augmenting path for the unassigned indices.
    IndexSet indicesE = equationIndices;

    // Look at each variable, v, accessed by e.
    for (EdgeDescriptor evDescriptor :
         getEdgesRange(getDescriptorFromId(eId))) {
      Edge &ev = graph[evDescriptor];
      VariableVertex &v = getVertex<VariableVertex>(evDescriptor.to);

      // Consider indices that are uncolored.
      IndexSet indicesV =
          ev.variableIndices(indicesE, v) - coloring.getColoredIndices(v);
      if (indicesV.empty()) {
        continue;
      }

      coloring.color(v, indicesV);

      // The current assignment for the indices of v.
      llvm::SmallVector<AssignmentComponent> &existingAssignment =
          getAssignment(v.getId(), assignments);
      llvm::SmallVector<AssignmentComponent> newAssignments;
      for (AssignmentComponent &assignmentComponent : existingAssignment) {
        // The indices of the variable that can be reassigned.
        const IndexSet &candidateIndices =
            assignmentComponent.indices.intersect(indicesV);
        if (candidateIndices.empty()) {
          continue;
        }

        // Traverse the edge from v to k (the equation that the indices are
        // currently assigned to), to get the currently assigned indices of k.
        const Edge &kv = graph[assignmentComponent.edgeDescriptor];
        const IndexSet &candidatesK = kv.equationIndices(candidateIndices);

        // Recursively continue the search for an augmenting path from the
        // currently assigned indices.
        const IndexSet &augmentedK =
            candidatesK - augmentPath(assignmentComponent.equationId,
                                      candidatesK, assignments, coloring);
        // Traverse the edge from k to j to get the indices of j to assign.
        const IndexSet &augmentedV = kv.variableIndices(augmentedK, v);

        // Create a new assignment component for the successful indices.
        newAssignments.push_back(
            AssignmentComponent{eId, augmentedV, evDescriptor});
        // Remove the indices from the old assignment component.
        assignmentComponent.indices -= augmentedV;

        // Uncolor the indices of v that were augmented.
        coloring.uncolor(v, augmentedV);
        // Remove the indices of v that were augmented.
        indicesV -= augmentedV;
        if (indicesV.empty()) {
          break;
        }
      }

      // Add the new assignments.
      existingAssignment.append(newAssignments);
      // Remove empty assignment components.
      llvm::erase_if(existingAssignment,
                     [](const AssignmentComponent &assignmentComponent) {
                       return assignmentComponent.indices.empty();
                     });

      indicesE -= ev.equationIndices(allAssignedIndices(newAssignments));
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
  IndexSet augmentPath(const EquationVertex::Id eId,
                       const IndexSet &equationIndices,
                       VariableAssignments &assignments, Coloring &coloring) {
    VertexDescriptor eDescriptor = getDescriptorFromId(eId);
    EquationVertex &e = getVertex<EquationVertex>(eDescriptor);

    // Colored indices should not be revisited.
    assert(!coloring.getColoredIndices(e).overlaps(equationIndices) &&
           "Equation already colored");

    // Try to assign adjacent variables to the indices of e.
    IndexSet unassignedEquationIndices =
        tryAssignAdjacent(eId, equationIndices, assignments);
    if (unassignedEquationIndices.empty()) {
      return IndexSet();
    }

    // Try to create augmenting path(s) for the unassigned indices.
    unassignedEquationIndices = tryAugmentAdjacent(
        eId, unassignedEquationIndices, assignments, coloring);

    // Color the indices that were not successfully assigned.
    coloring.color(e, unassignedEquationIndices);

    return unassignedEquationIndices;
  }

  /// Differentiate colored nodes
  ///
  /// - coloring: the variables and equations to differentiate, according to
  /// their coloring state
  /// - variableAssignments: the current variable assignments
  void differentiateNodes(const Coloring &coloring,
                          VariableAssignments &variableAssignments) {
    // Differentiate colored variables
    for (const auto &[jId, coloredIndices] : coloring.coloredVariables()) {
      if (auto dj = getVariableDerivative(jId)) {
        // The variable is an array that was already differentiated along
        // other indices. Therefore update the derived indices.
        assert(!dj->indices.overlaps(coloredIndices) &&
               "Previously derived indices should not be revisited.");
        dj->indices += coloredIndices;
        setVariableDerivative(jId, *dj);
        LLVM_DEBUG(llvm::dbgs()
                   << "Extending derivative of " << jId << " with "
                   << coloredIndices << " to " << dj->indices << "\n");
      } else {
        // The variable is a scalar or an array that was not yet
        // differentiated. Therefore, create a new variable, and establish
        // the derivative relationship.
        const VariableBridge &djBridge = differentiateVariable(jId);
        addVariable(djBridge);
        setVariableDerivative(jId,
                              VariableSubset{djBridge.getId(), coloredIndices});
        LLVM_DEBUG(llvm::dbgs()
                   << "Creating derivative of " << jId << " with name "
                   << djBridge.getName() << " @ " << coloredIndices << "\n");
      }
    }

    // Differentiate colored equations
    for (const auto &[lId, coloredIndices] : coloring.coloredEquations()) {
      // Check if the equation was already differentiated.
      if (auto existingDl = getEquationDerivative(lId)) {
        // An equation derivative already exists.
        // Therefore, validate any newly derived indices and update the
        // association.
        assert(!existingDl->indices.overlaps(coloredIndices) &&
               "Previously derived indices should not be revisited.");
        existingDl->indices += coloredIndices;
        setEquationDerivative(lId, *existingDl);
        getVertex<EquationVertex>(getDescriptorFromId(existingDl->id))
            .validateIndices(coloredIndices);
        LLVM_DEBUG(llvm::dbgs()
                   << "Extending derivative of " << lId << " with "
                   << coloredIndices << " to " << existingDl->indices << "\n");
      } else {
        // Create and add the equation node.
        VertexDescriptor dlDescriptor = graph.addVertex(
            EquationVertex(differentiateEquation(lId), coloredIndices));
        EquationVertex &dl = getVertex<EquationVertex>(dlDescriptor);
        equationsMap[dl.getId()] = dlDescriptor;

        // Add edges to the variables accessed by the original equation,
        // and their derivatives.
        llvm::DenseSet<VariableVertex::Id> addedEdges;
        for (EdgeDescriptor edgeDescriptor :
             getEdgesRange(getDescriptorFromId(lId))) {
          const Edge &edge = graph[edgeDescriptor];
          const auto &j = getVertex<VariableVertex>(edgeDescriptor.to);

          if (!addedEdges.contains(j.getId())) {
            graph.addEdge(dlDescriptor, edgeDescriptor.to, edge);
            addedEdges.insert(j.getId());
          }

          if (auto dj = getVariableDerivative(j.getId());
              dj && !addedEdges.contains(dj->id)) {
            graph.addEdge(dlDescriptor, getDescriptorFromId(dj->id), edge);
            addedEdges.insert(dj->id);
          }
        }

        LLVM_DEBUG({
          llvm::dbgs() << "Creating derivative of " << lId << ": ",
              dl.dump(llvm::dbgs());
          llvm::dbgs() << " and edges to:\n";
          for (EdgeDescriptor edgeDescriptor : getEdgesRange(dlDescriptor)) {
            llvm::dbgs() << "  "
                         << getVertex<VariableVertex>(edgeDescriptor.to).getId()
                         << " @ "
                         << graph[edgeDescriptor].accessedVariableIndices()
                         << "\n";
          }
        });

        setEquationDerivative(lId, {dl.getId(), coloredIndices});
      }
    }

    // Assign derivatives of colored variables to the
    // derivatives of their assigned equations.
    for (const auto &[jId, coloredIndices] : coloring.coloredVariables()) {
      IndexSet unassignedIndices = coloredIndices;
      llvm::SmallVector<AssignmentComponent> djAssignment;
      for (const AssignmentComponent &assignmentComponent :
           variableAssignments[jId]) {
        // The indices of dj are a subset of the indices of j.
        // Therefore we have to check that the indices of an assignment
        // are valid for dj before we use it.
        const IndexSet &assignable =
            unassignedIndices.intersect(assignmentComponent.indices);
        if (!assignable.empty()) {
          unassignedIndices -= assignable;
          djAssignment.push_back(AssignmentComponent{
              getEquationDerivative(assignmentComponent.equationId)->id,
              assignable, assignmentComponent.edgeDescriptor});
        }
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Assigning derivative of " << jId << " (colored @ "
                     << coloredIndices << ")" << ", "
                     << getVariableDerivative(jId)->id << " to "
                     << djAssignment.size() << " equations:\n";
        for (const AssignmentComponent &assignmentComponent : djAssignment) {
          llvm::dbgs() << "  " << assignmentComponent.equationId << " @ "
                       << assignmentComponent.indices << "\n";
        }
      });

      assert(unassignedIndices.empty() && "Did not assign all indices.");
      // J has a derivative by construction.
      auto dj = *getVariableDerivative(jId);
      if (variableAssignments.contains(dj.id)) {
        variableAssignments[dj.id].append(djAssignment);
      } else {
        variableAssignments.insert({dj.id, std::move(djAssignment)});
      }
    }
  }

public:
  IndexReductionGraph(const std::function<VariableBridge &(VariableBridge::Id)>
                          &differentiateVariable,
                      const std::function<EquationBridge &(EquationBridge::Id)>
                          &differentiateEquation)
      : differentiateVariable(differentiateVariable),
        differentiateEquation(differentiateEquation) {}

  /// Add a variable to the graph.
  void addVariable(const VariableBridge &variableBridge) {
    VariableVertex variable(variableBridge);
    VariableVertex::Id id = variable.getId();
    assert(!hasVariableWithId(id) && "Already existing variable");
    VertexDescriptor variableDescriptor = graph.addVertex(std::move(variable));
    variablesMap[id] = variableDescriptor;
  }

  /// Add an equation to the graph, all variables accessed by the equation must
  /// already be present in the graph.
  void addEquation(
      const EquationBridge &equationBridge,
      const llvm::ArrayRef<
          std::pair<VariableBridge::Id, std::unique_ptr<AccessFunction>>>
          accesses) {
    EquationVertex eq(equationBridge, equationBridge.getIndices());
    EquationVertex::Id id = eq.getId();
    assert(!hasEquationWithId(id) && "Already existing equation");
    VertexDescriptor equationDescriptor = graph.addVertex(std::move(eq));
    equationsMap[id] = equationDescriptor;

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

  /// Establish the relationship between a variable and its derivative.
  void setVariableDerivative(const VariableVertex::Id variableId,
                             const VariableVertex::Id derivativeId,
                             const IndexSet &derivedIndices) {
    setVariableDerivative(variableId,
                          VariableSubset{derivativeId, derivedIndices});
  }

  /// Apply the array-aware pantelides algorithm to the graph.
  ///
  /// Returns the chain of derivations to be performed for each equation.
  llvm::SmallVector<std::pair<EquationBridge::Id, llvm::SmallVector<IndexSet>>>
  pantelides() {
    hideDerivedVariables();
    LLVM_DEBUG(dumpInitialState());

    VariableAssignments assignments;
    const size_t numEquations = llvm::count_if(
        getDescriptorRange<EquationVertex>(), [](auto) { return true; });

    for (EquationVertex::Id eInitial = 0; eInitial < numEquations; eInitial++) {
      LLVM_DEBUG(llvm::dbgs() << "----------\n"
                              << "eInitial = " << eInitial << "\n");

      EquationVertex::Id eId = eInitial;
      IndexSet eIndices =
          getVertex<EquationVertex>(getDescriptorFromId(eInitial)).getIndices();

      while (true) {
        const auto &e = getVertex<EquationVertex>(getDescriptorFromId(eId));
        Coloring coloring;
        eIndices = augmentPath(e.getId(), eIndices, assignments, coloring);

        LLVM_DEBUG(dumpPathResult(e, eIndices, assignments, coloring,
                                  eIndices.empty()));

        if (eIndices.empty()) {
          break;
        }
        differentiateNodes(coloring, assignments);
        hideDerivedVariables();
        eId = getEquationDerivative(eId)->id;

        LLVM_DEBUG({
          dumpAssignmentState(assignments);
          dumpVisibilityState();
          dump(llvm::dbgs());
        });
      }
    }

    llvm::SmallVector<
        std::pair<EquationVertex::Id, llvm::SmallVector<IndexSet>>>
        differentiations;
    // Collect the derivation-chain for each original equation
    for (EquationVertex::Id equationId = 0; equationId < numEquations;
         equationId++) {
      llvm::SmallVector<IndexSet> derivativeChain;
      std::optional<EquationSubset> derivative =
          getEquationDerivative(equationId);
      while (derivative) {
        derivativeChain.push_back(derivative->indices);
        derivative = getEquationDerivative(derivative->id);
      }
      if (!derivativeChain.empty()) {
        differentiations.emplace_back(equationId, std::move(derivativeChain));
      }
    }
    return differentiations;
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "Index Reduction Graph:\n";

    os << " Variables:\n";
    for (const VariableVertex &v : getVertexRange<VariableVertex>()) {
      os << "  ", v.dump(os);
      if (auto dv = getVariableDerivative(v.getId())) {
        os << " -der-> " << dv->id << " @ " << dv->indices;
      }
      os << "\n";
    }

    os << " Equations:\n";
    for (const VertexDescriptor &descriptor :
         getDescriptorRange<EquationVertex>()) {
      auto &e = getVertex<EquationVertex>(descriptor);
      os << "  ", e.dump(os);
      if (auto de = getEquationDerivative(e.getId())) {
        os << " -der-> " << de->id << " @ " << de->indices;
      }
      os << "\n";
      for (EdgeDescriptor edgeDescriptor : getEdgesRange(descriptor)) {
        os << "   " << getVertex<VariableVertex>(edgeDescriptor.to).getId()
           << " - ";
        graph[edgeDescriptor].dump(os);
      }
    }
    os << "---\n";
  };

private:
  void dumpVisibilityState() const {
    llvm::raw_ostream &os = llvm::dbgs();
    os << "Variable visibility:\n";
    for (const VariableVertex &j : getVertexRange<VariableVertex>()) {
      auto visibleIndices = j.getVisibleIndices();
      auto hiddenIndices = j.getIndices() - visibleIndices;
      if (hiddenIndices.empty())
        continue;

      os << " ";
      j.dump(os);
      if (visibleIndices.empty())
        os << " hidden\n";
      else if (!hiddenIndices.empty())
        os << " has hidden indices " << hiddenIndices << "\n";
    }
  }

  void
  dumpAssignmentState(const VariableAssignments &variableAssignments) const {
    llvm::raw_ostream &os = llvm::dbgs();
    os << "Variable assignments:\n";
    for (const auto &[id, assignments] : variableAssignments) {
      if (assignments.empty()) {
        continue;
      }
      auto variable = getVertex<VariableVertex>(getDescriptorFromId(id));
      os << " ", variable.dump(os), os << ":\n";
      for (const AssignmentComponent &assignmentComponent : assignments) {
        const auto &edge = graph[assignmentComponent.edgeDescriptor];
        os << "  " << assignmentComponent.indices << " -> "
           << assignmentComponent.equationId << "("
           << edge.equationIndices(assignmentComponent.indices) << ")" << "\n";
      }
    }
  }

  void dumpInitialState() const {
    size_t numEquations = 0;
    size_t numEquationIndices = 0;
    for (const EquationVertex &equation : getVertexRange<EquationVertex>()) {
      numEquations++;
      numEquationIndices += equation.getIndices().flatSize();
    }
    size_t numVariables = 0;
    size_t numIndices = 0;
    for (const VariableVertex &variable : getVertexRange<VariableVertex>()) {
      numVariables++;
      IndexSet indices = variable.getIndices();
      // Derivative variables are only "active" at the derived indices.
      for (const auto &[_, derivative] : variableAssociations) {
        if (derivative.id == variable.getId()) {
          indices = derivative.indices;
        }
      }
      numIndices += indices.flatSize();
    }
    llvm::dbgs() << "Pantelides initial state:\n"
                 << " #equations: " << numEquations << " ->"
                 << " #indices: " << numEquationIndices << "\n"
                 << " #variables: " << numVariables << " ->"
                 << " #indices: " << numIndices << "\n";
  }

  void dumpPathResult(const EquationVertex &e, const IndexSet &indices,
                      const VariableAssignments &variableAssignments,
                      const Coloring &coloring, const bool res) const {
    llvm::dbgs() << "e = ";
    e.dump(llvm::dbgs());
    llvm::dbgs() << " @ " << indices << "\n";
    llvm::dbgs() << "Augmenting path " << (res ? " SUCCEEDED" : " FAILED")
                 << "\n";
    if (!res) {
      coloring.dump(llvm::dbgs());
    }
    dumpAssignmentState(variableAssignments);
  }

private:
  Graph graph;

  std::function<VariableBridge &(const VariableBridge::Id)>
      differentiateVariable;
  std::function<EquationBridge &(const EquationBridge::Id)>
      differentiateEquation;

  llvm::DenseMap<VariableVertex::Id, VertexDescriptor> variablesMap;
  llvm::DenseMap<EquationVertex::Id, VertexDescriptor> equationsMap;

  /// Associates a variable with its derivative.
  /// var -> (var', indices of var)
  llvm::DenseMap<VariableVertex::Id, VariableSubset> variableAssociations;

  /// Associates an equation with its derivative.
  /// eq -> eq'
  llvm::DenseMap<EquationVertex::Id, EquationSubset> equationAssociations;
};

} // namespace marco::modeling

#endif // MARCO_MODELING_INDEXREDUCTION_H
