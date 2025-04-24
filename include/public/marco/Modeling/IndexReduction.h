#ifndef MARCO_MODELING_INDEXREDUCTION_H
#define MARCO_MODELING_INDEXREDUCTION_H

#include "marco/Dialect/BaseModelica/IR/EquationPath.h"
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

namespace marco::modeling {

using mlir::bmodelica::bridge::EquationBridge;
using mlir::bmodelica::bridge::VariableBridge;

namespace internal::indexReduction {

class VariableVertex final : public Dumpable {
public:
  using Id = VariableBridge::Id;

  Id getId() const { return id; }
  const IndexSet &getIndices() const { return indices; }

  const IndexSet &getColoredIndices() const { return coloredIndices; }
  void colorIndices(const IndexSet &indices) { coloredIndices += indices; }
  void uncolorIndices(const IndexSet &indices) { coloredIndices -= indices; }

  void hideIndices(const IndexSet &indices) { visibleIndices -= indices; }
  const IndexSet &getVisibleIndices() const { return visibleIndices; }

  explicit VariableVertex(const VariableBridge &bridge)
      : id(bridge.getId()), indices(bridge.getIndices()), coloredIndices(),
        visibleIndices(indices) {}

  void dump(llvm::raw_ostream &os) const override {
    os << "Variable(id: " << getId() << ", indices: " << getIndices() << ")";
  }

private:
  Id id;
  IndexSet indices;

  IndexSet coloredIndices;
  IndexSet visibleIndices;
};

/// An equation vertex in the graph.
class EquationVertex final {
public:
  using Id = EquationBridge::Id;

  Id getId() const { return id; }

  const IndexSet &getIndices() const { return validIndices; }

  void validateIndices(const IndexSet &indices) {
    assert(allIndices.contains(indices) && "Invalid indices for equation");
    validIndices += indices;
  }

  const IndexSet &getColoredIndices() const { return coloredIndices; }
  void colorIndices(const IndexSet &indices) {
    assert(getIndices().contains(indices) &&
           "Colored indices must be valid for the equation");
    coloredIndices += indices;
  }
  void uncolorIndices(const IndexSet &indices) { coloredIndices -= indices; }

  EquationVertex(const EquationBridge &bridge, const IndexSet &validIndices)
      : id(bridge.getId()), allIndices(bridge.getIndices()),
        validIndices(validIndices) {
    assert(allIndices.contains(validIndices) &&
           "Valid indices must be a subset of equation indices");
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

  /// Indices that have been visited during the current BFS invocation.
  /// TODO: Possibly factor this out, use a map in the augmentPath function
  /// instead.
  IndexSet coloredIndices;
};

/// An edge from an equation to a variable in the graph.
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

  /// Traverse the edge from equation to variable.
  IndexSet variableIndices(const IndexSet &equationIndices) const {
    return incidenceMatrix->filterColumns(equationIndices).flattenRows();
  }

  /// Traverse the edge from variable to equation.
  IndexSet equationIndices(const IndexSet &variableIndices) const {
    return incidenceMatrix->filterRows(variableIndices).flattenColumns();
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "incidence matrix:\n" << *incidenceMatrix;
  }

private:
  std::shared_ptr<MCIM> incidenceMatrix;
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

  struct VariableSubset {
    VariableVertex::Id id;
    IndexSet indices;
  };
  struct EquationSubset {
    EquationVertex::Id id;
    IndexSet indices;
  };
  using VariableAssignments =
      llvm::DenseMap<VariableVertex::Id, llvm::SmallVector<EquationSubset>>;

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
  auto getEdgesRange(const VertexDescriptor vertex) const {
    return llvm::make_range(graph.outgoingEdgesBegin(vertex),
                            graph.outgoingEdgesEnd(vertex));
  }

  bool hasVariableWithId(const VariableVertex::Id id) const {
    return variablesMap.find(id) != variablesMap.end();
  }

  bool hasEquationWithId(const EquationVertex::Id id) const {
    return equationsMap.find(id) != equationsMap.end();
  }

  VertexDescriptor
  getVariableDescriptorFromId(const VariableVertex::Id id) const {
    auto it = variablesMap.find(id);
    assert(it != variablesMap.end() && "Variable not found");
    return it->second;
  }

  VertexDescriptor
  getEquationDescriptorFromId(const EquationVertex::Id id) const {
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

  /// Get the derivative of an equation if it has one.
  std::optional<EquationSubset>
  getEquationDerivative(const EquationVertex::Id id) const {
    if (auto it = equationAssociations.find(id);
        it != equationAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  /// Remove coloring from all vertices in the graph.
  void uncolorAllVertices() {
    for (VertexDescriptor descriptor : getDescriptorRange<VariableVertex>()) {
      VariableVertex &variable = getVertex<VariableVertex>(descriptor);
      variable.uncolorIndices(variable.getColoredIndices());
    }
    for (VertexDescriptor descriptor : getDescriptorRange<EquationVertex>()) {
      EquationVertex &equation = getVertex<EquationVertex>(descriptor);
      equation.uncolorIndices(equation.getColoredIndices());
    }
  }

  /// Hide the (indices of) variables that have derivatives (of those indices).
  void hideDerivedVariables() {
    for (VertexDescriptor descriptor : getDescriptorRange<VariableVertex>()) {
      VariableVertex &variable = getVertex<VariableVertex>(descriptor);
      if (auto derivative = getVariableDerivative(variable.getId())) {
        variable.hideIndices(derivative->indices);
      }
    }
  }

  /// Get the assignment for a variable, creating it if it does not exist.
  static llvm::SmallVector<EquationSubset> &
  getAssignment(VariableVertex::Id id, VariableAssignments &assignments) {
    auto existingAssignment = assignments.find(id);
    if (existingAssignment == assignments.end()) {
      existingAssignment =
          assignments.insert({id, llvm::SmallVector<EquationSubset>()}).first;
    }
    return existingAssignment->getSecond();
  }

  /// Get all assigned indices from the assignment.
  static IndexSet
  allAssignedIndices(const llvm::SmallVector<EquationSubset> &assignments) {
    IndexSet result;
    for (const auto &[_, assignedIndices] : assignments) {
      result += assignedIndices;
    }
    return result;
  }

  /// Attempt to create an augmenting path from the given equation.
  ///
  /// - from: the id of the equation vertex to start from
  /// - assignments: the current assignments
  ///
  /// Returns true if an augmenting path was found for all indices.
  ///
  /// When false is returned, the colored vertices make up a
  /// structurally singular subset of the system.
  bool augmentPath(EquationVertex::Id from, VariableAssignments &assignments) {
    IndexSet equationIndices =
        getVertex<EquationVertex>(getEquationDescriptorFromId(from))
            .getIndices();
    // Start the traversal from all the indices of the equation.
    return augmentPath(from, equationIndices, assignments).empty();
  }

  /// Augment path
  ///
  /// - from: the id of the equation vertex to start from
  /// - equationIndices: the indices of the equation vertex to use
  /// - assignments: the current assignments
  ///
  /// Returns the indices for which an augmenting path was not found.
  ///
  /// When false is returned, the colored vertices make up a
  /// structurally singular subset of the system.
  IndexSet augmentPath(EquationVertex::Id from, const IndexSet &equationIndices,
                       VariableAssignments &assignments) {
    VertexDescriptor iDescriptor = getEquationDescriptorFromId(from);
    EquationVertex &i = getVertex<EquationVertex>(iDescriptor);

    LLVM_DEBUG(llvm::dbgs() << "Augmenting path from equation " << i.getId()
                            << " with indices " << i.getIndices() << " for "
                            << equationIndices << "\n";);

    // Each equation should only be visited once for a given set of indices.
    assert(!i.getColoredIndices().overlaps(equationIndices) &&
           "Equation already colored");

    // (1) Color the equation.
    i.colorIndices(equationIndices);

    // TODO: Remove this
    return IndexSet();

    /*

    // The indices that have yet found an assignment.
    IndexSet remainingEquationIndices = equationIndices;

    // (2) If an unassigned variable exists, assign it to the current equation.
    for (EdgeDescriptor edgeDescriptor : getEdgesRange(iDescriptor)) {
      const Edge &edge = graph[edgeDescriptor];
      const auto &j = getVertex<VariableVertex>(edgeDescriptor.to);
      // The indices of variable j that are accessed at the given equation
      // indices.
      const IndexSet accessedVariableIndices =
          edge.accessedVariableIndices(remainingEquationIndices);
      // The visible subset of the accessed indices of j.
      const IndexSet visibleIndices =
          j.getVisibleIndices().intersect(accessedVariableIndices);

      // If j is completely hidden at the indices, skip it.
      if (visibleIndices.empty()) {
        continue;
      }
      LLVM_DEBUG({
        if (visibleIndices != accessedVariableIndices) {
          llvm::dbgs() << "Accessing variable " << j.getId()
                       << " at hidden indices "
                       << accessedVariableIndices - visibleIndices << "\n";
        }
      });

      // Get the assignment for j.
      llvm::SmallVector<Assignment> &existingAssignments =
          getAssignment(j.getId(), assignments);

      if (const IndexSet &alreadyAssignedIndices =
              visibleIndices.intersect(allAssignedIndices(existingAssignments));
          alreadyAssignedIndices != visibleIndices) {
        // Assign the unassigned indices to equation i.
        const auto &[_, assigned] = existingAssignments.emplace_back(
            i.getId(), visibleIndices - alreadyAssignedIndices);
        // Remove the assigned indices from the search.
        remainingEquationIndices -= edge.equationIndices(assigned);
        if (remainingEquationIndices.empty()) {
          // If no indices remain we are finished.
          return IndexSet();
        }
      }
    }

    // (3) Look for an augmenting path for the unassigned indices.
    for (EdgeDescriptor edgeDescriptor : getEdgesRange(iDescriptor)) {
      Edge &edge = graph[edgeDescriptor];
      VariableVertex &j = getVertex<VariableVertex>(edgeDescriptor.to);
      // The indices of j that are accessed at the given equation
      // indices.
      const IndexSet accessedVariableIndices =
          edge.accessedVariableIndices(remainingEquationIndices);

      // Continue with only the visible subset of the accessed indices of j.
      const IndexSet visibleIndices =
          j.getVisibleIndices().intersect(accessedVariableIndices);
      if (visibleIndices.empty()) {
        // If j is completely hidden at the indices, skip it.
        continue;
      }
      LLVM_DEBUG({
        if (visibleIndices != accessedVariableIndices) {
          llvm::dbgs() << "Accessing variable " << j.getId()
                       << " at hidden indices "
                       << accessedVariableIndices - visibleIndices << "\n";
        }
      });

      // Continue with only the visible indices of j that have not yet been
      // visited.
      const IndexSet uncoloredIndices =
          j.getColoredIndices().intersect(visibleIndices);
      if (uncoloredIndices.empty()) {
        // If all visible indices of j have been visited, skip it.
        continue;
      }
      LLVM_DEBUG({
        if (uncoloredIndices != visibleIndices) {
          llvm::dbgs() << "Variable " << j.getId()
                       << " has uncolored indices "
                          "overlapping with the visible ones\n";
        }
      });

      j.colorIndices(uncoloredIndices);

      // As the variable is colored, it is guaranteed to have an assignment.
      llvm::SmallVectorImpl<Assignment> &existingAssignments =
          getAssignment(j.getId(), assignments);

      bool foundAugmentingPath = false;
      // TODO: See if this is still correct with the loop. It should propably
      // only happen for assignments that intersect with the incidentIndices.
      for (const Assignment &assignment : existingAssignments) {
        if (!assignment.second.overlaps(accessedVariableIndices)) {
          continue;
        }
        EquationVertex::Id k = assignment.first;
        if (augmentPath(k, assignments)) {
          existingAssignments.emplace_back(id, accessedVariableIndices);
          foundAugmentingPath = true;
        }
      }

      if (foundAugmentingPath) {
        return true;
      }
    }

    return remainingEquationIndices;
    */
  }

  void dumpVisibilityState(llvm::raw_ostream &os) const {
    os << "Visibility state:\n";
    for (const VariableVertex &j : getVertexRange<VariableVertex>()) {
      os << " ";
      j.dump(os);
      auto visibleIndices = j.getVisibleIndices();
      auto hiddenIndices = j.getIndices() - visibleIndices;
      if (visibleIndices.empty())
        os << " is completely hidden\n";
      else if (hiddenIndices.empty())
        os << " is completely visible\n";
      else
        os << " has hidden indices " << hiddenIndices << "\n";
    }
  }

  void dumpColoringState(llvm::raw_ostream &os) const {
    os << "Coloring state:\n";

    os << " colored equation id(s):";
    for (const EquationVertex &l : getVertexRange<EquationVertex>()) {
      // TODO: Show colored indices
      if (!l.getColoredIndices().empty()) {
        os << " " << l.getId();
      }
    }
    os << "\n";

    for (const VariableVertex &j : getVertexRange<VariableVertex>()) {
      os << " ";
      j.dump(os);
      auto coloredIndices = j.getColoredIndices();
      auto uncoloredIndices = j.getIndices() - coloredIndices;
      if (uncoloredIndices.empty())
        os << " is completely colored\n";
      else if (coloredIndices.empty())
        os << " is not colored\n";
      else
        os << " has colored indices " << coloredIndices << "\n";
    }
  }

  void
  dumpAssignmentState(llvm::raw_ostream &os,
                      const VariableAssignments &variableAssignments) const {
    os << "Variable assignments:\n";
    for (const auto &[id, assignments] : variableAssignments) {
      if (assignments.empty()) {
        continue;
      }
      auto variable =
          getVertex<VariableVertex>(getVariableDescriptorFromId(id));

      os << " ";
      variable.dump(os), os << " is assigned to equations:\n";
      for (const auto &[equationId, indices] : assignments) {
        os << "  " << equationId << " at " << indices << "\n";
      }
    }
  }

public:
  IndexReductionGraph(const std::function<VariableBridge &(VariableBridge::Id,
                                                           const IndexSet &)>
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

  /// Add an equation to the graph, all variables accessed by the equation are
  /// expected to already be present in the graph.
  void addEquation(
      EquationBridge &equationBridge,
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
      VertexDescriptor variableDescriptor =
          getVariableDescriptorFromId(variableId);
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
  void setVariableDerivative(VariableVertex::Id variableId,
                             VariableVertex::Id derivativeId,
                             const IndexSet &derivedIndices) {
    variableAssociations.try_emplace(
        variableId, VariableSubset{derivativeId, derivedIndices});
  }

  /// Apply the pantelides algorithm to the graph.
  /// Returns how many times each equation should be differentiated.
  llvm::SmallVector<std::pair<EquationVertex::Id, size_t>> pantelides() {
    VariableAssignments variableAssignments;

    const size_t numEquations = llvm::count_if(
        getDescriptorRange<EquationVertex>(), [](auto) { return true; });
    LLVM_DEBUG({
      size_t numVariables = 0;
      size_t numIndices = 0;
      for (const VariableVertex &variable : getVertexRange<VariableVertex>()) {
        numVariables++;
        // TODO: This isn't quite right for derivatives (where we maybe only
        // care about a subset of the indices)
        numIndices += variable.getIndices().flatSize();
      }
      llvm::dbgs() << "Pantelides initial state:\n"
                   << " #equations: " << numEquations << "\n"
                   << " #variables: " << numVariables << " ->"
                   << " #indices: " << numIndices << "\n";
    });

    for (size_t kId = 0; kId < numEquations; kId++) {
      LLVM_DEBUG(llvm::dbgs() << "----------\n" << "k = " << kId << "\n");
      // 3a
      auto i = getVertex<EquationVertex>(getEquationDescriptorFromId(kId));

      // 3b
      while (true) {
        LLVM_DEBUG({
          llvm::dbgs() << "---\n"
                       << "i = " << i.getId() << "\n";
          // Dump the graph
          dump(llvm::dbgs());
        });

        // 3b-1
        hideDerivedVariables();

        // 3b-2
        uncolorAllVertices();

        // 3b-(3 & 4)
        bool res = augmentPath(i.getId(), variableAssignments);

        LLVM_DEBUG({
          dumpVisibilityState(llvm::dbgs());
          llvm::dbgs() << "Augmenting path from " << i.getId()
                       << (res ? " SUCCEEDED" : " FAILED") << "\n";
          if (!res) {
            dumpColoringState(llvm::dbgs());
          }
          dumpAssignmentState(llvm::dbgs(), variableAssignments);
        });

        // 3b-5
        if (res) {
          break;
        }
        // 3b-5 (i) - Differentiate colored variables
        for (const VariableVertex &j : getVertexRange<VariableVertex>()) {
          const IndexSet &coloredIndices = j.getColoredIndices();
          if (coloredIndices.empty()) {
            continue;
          }
          LLVM_DEBUG({
            llvm::dbgs() << "Differentiating ";
            j.dump(llvm::dbgs());
            llvm::dbgs() << " at indices " << coloredIndices << "\n";
          });

          const VariableBridge &djBridge =
              differentiateVariable(j.getId(), coloredIndices);
          assert(djBridge.getIndices().contains(coloredIndices) &&
                 "Variables was not differentiated for the requested indices.");
          if (hasVariableWithId(djBridge.getId())) {
            // The variable is an array that was already differentiated along
            // other indices. Therefore update the derived indices.
            IndexSet newIndices =
                getVariableDerivative(j.getId())->indices + coloredIndices;
            variableAssociations.insert_or_assign(
                j.getId(), VariableSubset{djBridge.getId(), newIndices});
          } else {
            // The variable is a scalar or an array that was not yet
            // differentiated. Therefore, add it variable, and establish the
            // derivative relationship.
            addVariable(djBridge);
            variableAssociations.insert(
                {j.getId(), {djBridge.getId(), coloredIndices}});
          }
        }

        // 3b-5 (ii) - Differentiate colored equations
        for (const auto &lDescriptor : getDescriptorRange<EquationVertex>()) {
          // If the equation is colored, generate and add its derivative
          const EquationVertex &l = getVertex<EquationVertex>(lDescriptor);
          const IndexSet &coloredIndices = l.getColoredIndices();
          if (coloredIndices.empty()) {
            continue;
          }

          // Check if the equation was already differentiated.
          auto dl = getEquationDerivative(l.getId());
          if (!dl) {
            EquationVertex dl(differentiateEquation(l.getId()), coloredIndices);
            assert(!hasEquationWithId(dl.getId()) &&
                   "Already existing equation");
            assert(dl.getIndices() == l.getIndices() &&
                   "Differentiated equation has wrong iteration ranges");

            VertexDescriptor dlDescriptor = graph.addVertex(std::move(dl));
            equationsMap[dl.getId()] = dlDescriptor;

            // Iterate the edges of the initial equation
            for (EdgeDescriptor edgeDescriptor :
                 getEdgesRange(getEquationDescriptorFromId(l.getId()))) {
              const Edge &edge = graph[edgeDescriptor];
              const auto &j = getVertex<VariableVertex>(edgeDescriptor.to);
              LLVM_DEBUG(llvm::dbgs() << "Adding edge(s) from " << dl.getId()
                                      << " to " << j.getId());

              graph.addEdge(dlDescriptor, edgeDescriptor.to, edge);

              if (auto dj = getVariableDerivative(j.getId())) {
                LLVM_DEBUG(llvm::dbgs() << ", " << dj->id);
                assert(dj->indices.contains(edge.accessedVariableIndices()) &&
                       "Variable derivative does not contain accessed "
                       "indices");

                graph.addEdge(dlDescriptor, getVariableDescriptorFromId(dj->id),
                              edge);
              }
              LLVM_DEBUG(llvm::dbgs()
                         << " at " << edge.accessedVariableIndices() << "\n");
            }

            equationAssociations[l.getId()] = {dl.getId(), coloredIndices};
          } else {
            // An equation derivative already exists.
            // Therefore validate any newly derived indices and update the
            // association.
            auto [dlId, derivedIndices] = *dl;
            equationAssociations[l.getId()] = {dlId,
                                               derivedIndices + coloredIndices};
            getVertex<EquationVertex>(getEquationDescriptorFromId(dlId))
                .validateIndices(coloredIndices);
          }
        }

        // 3b-5 (iii) - Assign derivatives of colored variables to the
        // derivatives of their assigned equations.
        for (const VariableVertex &j : getVertexRange<VariableVertex>()) {
          const IndexSet &coloredIndices = j.getColoredIndices();
          if (coloredIndices.empty()) {
            continue;
          }

          // As j was colored we know we just gave it a derivative
          auto [djId, derivedIndices] = *getVariableDerivative(j.getId());
          assert(derivedIndices.contains(coloredIndices) &&
                 "The colored indices should have been derived");

          IndexSet unassignedIndices = coloredIndices;
          llvm::SmallVector<EquationSubset> djAssignment;
          for (const EquationSubset &assignment :
               variableAssignments[j.getId()]) {

            // The indices of dj are a subset of the indices of j.
            // Therefore we have to check that the indices of an assignment
            // are valid for dj before we use it.
            // TODO: Fix this
            if (unassignedIndices.contains(assignment.indices)) {
              unassignedIndices -= assignment.indices;
              djAssignment.emplace_back(
                  EquationSubset{getEquationDerivative(assignment.id)->id,
                                 assignment.indices});
            }
          }

          assert(unassignedIndices.empty() && "Did not assign all indices.");
          variableAssignments[djId] = std::move(djAssignment);
        }

        // 3b-5 (iv) - Continue from the derivative of the current equation
        auto [diId, derivedIndices] = *getEquationDerivative(i.getId());
        i = getVertex<EquationVertex>(getEquationDescriptorFromId(diId));
      }
    }

    llvm::SmallVector<std::pair<EquationVertex::Id, size_t>> neededDerivations;
    // Measure the length of the derivation-chain for each original equation
    for (size_t equationId = 0; equationId < numEquations; equationId++) {
      size_t numDerivations = 0;
      auto derivative = equationAssociations.find(equationId);
      while (derivative != equationAssociations.end()) {
        numDerivations++;
        derivative = equationAssociations.find(derivative->getFirst());
      }
      neededDerivations.emplace_back(equationId, numDerivations);
    }

    return neededDerivations;
  }

  void dump(llvm::raw_ostream &os) const override {
    os << "---\n" << "Index Reduction Graph:\n";

    os << " Variables:\n";
    for (const VariableVertex &v : getVertexRange<VariableVertex>()) {
      os << "  ", v.dump(os);
      if (auto dv = getVariableDerivative(v.getId())) {
        os << " -> " << "Derivative(id: " << dv->id
           << ") for indices: " << dv->indices;
      }
      os << "\n";
    }

    os << " Equations:\n";
    for (const VertexDescriptor &descriptor :
         getDescriptorRange<EquationVertex>()) {
      auto &e = getVertex<EquationVertex>(descriptor);
      os << "  Equation(id: " << e.getId() << ", indices: " << e.getIndices()
         << ")";
      if (auto de = getEquationDerivative(e.getId())) {
        os << " -> ";
        os << "Derivative(id: " << de->id << ") for indices: " << de->indices;
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
  Graph graph;

  std::function<VariableBridge &(const VariableBridge::Id, const IndexSet &)>
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
