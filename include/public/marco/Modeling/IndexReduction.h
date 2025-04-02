#ifndef MARCO_MODELING_INDEXREDUCTION_H
#define MARCO_MODELING_INDEXREDUCTION_H

#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
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
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <functional>
#include <variant>

namespace marco::modeling {

using mlir::bmodelica::bridge::EquationBridge;
using mlir::bmodelica::bridge::VariableBridge;

namespace internal::indexReduction {

class VariableVertex final : public Dumpable {
public:
  using Id = VariableBridge::Id;

  Id getId() const { return bridge->id; }

  void colorIndices(const IndexSet &indices) { coloredIndices += indices; }
  void clearColoring() { coloredIndices.clear(); }
  const IndexSet &getColoredIndices() const { return coloredIndices; }
  bool isColoredAt(const IndexSet &indices) const {
    return coloredIndices.contains(indices);
  }

  void hideIndices(const IndexSet &indices) { visibleIndices -= indices; }
  const IndexSet &getVisibleIndices() const { return visibleIndices; }
  bool isHiddenAt(const IndexSet &indices) const {
    return !visibleIndices.contains(indices);
  }

  const IndexSet &getIndices() const {
    assert(!bridge->indices.empty());
    return bridge->indices;
  }

  explicit VariableVertex(VariableBridge *bridge)
      : bridge(bridge), visibleIndices(bridge->indices) {}

  VariableBridge *getBridge() const { return bridge; }

  void dump(llvm::raw_ostream &os) const override {
    os << "Variable(id: " << bridge->id << ", indices: " << bridge->indices
       << ")";
  }

private:
  VariableBridge *bridge;

  IndexSet coloredIndices;
  IndexSet visibleIndices;
};

class Access {
  using EquationPath = mlir::bmodelica::EquationPath;

public:
  Access(const VariableVertex::Id variableId,
         std::unique_ptr<AccessFunction> accessFunction)
      : variableId(variableId), accessFunction(std::move(accessFunction)) {}

  Access(const Access &other)
      : variableId(other.variableId),
        accessFunction(other.accessFunction->clone()) {}

  Access &operator=(Access &&other) noexcept {
    Access tmp(other);
    std::swap(*this, tmp);
    return *this;
  }

  ~Access() = default;

  VariableVertex::Id getVariableId() const { return variableId; }

  const AccessFunction &getAccessFunction() const {
    assert(accessFunction != nullptr);
    return *accessFunction;
  }

private:
  VariableVertex::Id variableId;
  std::unique_ptr<AccessFunction> accessFunction;
};

class EquationVertex final : public Dumpable {
public:
  using Id = int64_t;

  Id getId() const { return bridge->getId(); }

  bool isColored() const { return colored; }
  void setColored(const bool value) { colored = value; }

  IndexSet getIterationRanges() const {
    IndexSet iterationSpace = bridge->getOp().getProperties().indices;
    if (iterationSpace.empty()) {
      // Scalar equation.
      iterationSpace += MultidimensionalRange(Range(0, 1));
    }
    return iterationSpace;
  }

  llvm::SmallVector<Access> getVariableAccesses() const {
    if (bridge->hasAccessAnalysis()) {
      if (auto cachedAccesses = bridge->getAccessAnalysis().getAccesses(
              bridge->getSymbolTableCollection())) {
        return convertAccesses(*cachedAccesses);
      }
    }

    llvm::SmallVector<mlir::bmodelica::VariableAccess> accesses;
    if (mlir::succeeded(bridge->getOp().getAccesses(
            accesses, bridge->getSymbolTableCollection()))) {
      return convertAccesses(accesses);
    }

    llvm_unreachable("Can't compute the accesses");
  }

  explicit EquationVertex(EquationBridge *bridge) : bridge(bridge) {}

  EquationBridge *getBridge() const { return bridge; }

  void dump(llvm::raw_ostream &os) const override {
    os << "Equation(id: " << bridge->getId() << ")";
  }

private:
  llvm::SmallVector<Access> convertAccesses(
      const llvm::ArrayRef<mlir::bmodelica::VariableAccess> accesses) const {
    llvm::SmallVector<Access> result;
    for (const auto &access : accesses) {
      auto accessFunction =
          convertAccessFunction(bridge->getOp().getContext(), access);

      if (auto variableIt =
              bridge->getVariablesMap().find(access.getVariable());
          variableIt != bridge->getVariablesMap().end()) {
        result.emplace_back(variableIt->getSecond()->id,
                            std::move(accessFunction));
      }
    }
    return result;
  }

  static std::unique_ptr<AccessFunction>
  convertAccessFunction(mlir::MLIRContext *context,
                        const mlir::bmodelica::VariableAccess &access) {
    const AccessFunction &accessFunction = access.getAccessFunction();

    if (accessFunction.getNumOfResults() == 0) {
      // Access to scalar variable.
      return AccessFunction::build(
          mlir::AffineMap::get(accessFunction.getNumOfDims(), 0,
                               mlir::getAffineConstantExpr(0, context)));
    }

    return accessFunction.clone();
  }

private:
  EquationBridge *bridge;
  bool colored = false;
};

class Edge final : public Dumpable {
public:
  Edge(const VariableVertex::Id variableId, IndexSet equationRanges,
       IndexSet variableRanges, const Access &access)
      : incidenceMatrix(std::move(equationRanges), std::move(variableRanges)) {
    incidenceMatrix.apply(access.getAccessFunction());
  }

  // Copy constructor
  Edge(const Edge &other) : incidenceMatrix(other.incidenceMatrix) {}

  /// Get the indices of the variable that are accessed by the equation.
  IndexSet accessedVariableIndices() const {
    return incidenceMatrix.flattenRows();
  }

  void dump(llvm::raw_ostream &os) const override {
    if (incidenceMatrix.getEquationSpace().flatSize() == 1 &&
        incidenceMatrix.getVariableSpace().flatSize() == 1) {
      os << "scalar\n";
    } else {
      os << "incidence matrix:\n" << incidenceMatrix;
    }
  }

private:
  MCIM incidenceMatrix;
};

} // namespace internal::indexReduction

class IndexReductionGraph final : public internal::Dumpable {
private:
  using VariableVertex = internal::indexReduction::VariableVertex;
  using EquationVertex = internal::indexReduction::EquationVertex;
  using Vertex = std::variant<VariableVertex, EquationVertex>;
  using Edge = internal::indexReduction::Edge;

  using Graph = internal::UndirectedGraph<Vertex, Edge>;
  using Access = internal::indexReduction::Access;
  using VertexDescriptor = Graph::VertexDescriptor;
  using EdgeDescriptor = Graph::EdgeDescriptor;
  using VertexIterator = Graph::VertexIterator;
  using EdgeIterator = Graph::EdgeIterator;

  using VariableIterator = Graph::FilteredVertexIterator;
  using EquationIterator = Graph::FilteredVertexIterator;

  using Assignment = std::pair<EquationVertex::Id, IndexSet>;
  using Assignments =
      llvm::DenseMap<VariableVertex::Id, llvm::SmallVector<Assignment>>;

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
  auto getEdgesRange(VertexDescriptor vertex) const {
    return llvm::make_filter_range(
        llvm::make_range(graph.outgoingEdgesBegin(vertex),
                         graph.outgoingEdgesEnd(vertex)),
        [&](const EdgeDescriptor &descriptor) {
          return true;
          // TODO: Maybe filter out invisible edges here.
          // graph[descriptor].isVisible();
        });
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

  /// Get the derivative of a variable if it exists, along with the derived
  /// indices.
  std::optional<std::pair<VariableVertex::Id, IndexSet>>
  getDerivative(const VariableVertex::Id id) const {
    if (auto it = variableAssociations.find(id);
        it != variableAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  /// Get the derivative of an equation if it has one.
  std::optional<EquationVertex::Id>
  getDerivative(const EquationVertex::Id id) const {
    if (auto it = equationAssociations.find(id);
        it != equationAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  /// Augment path procedure from the Pantelides algorithm.
  bool augmentPath(EquationVertex::Id id, Assignments &assignments) {
    VertexDescriptor iDescriptor = getEquationDescriptorFromId(id);
    EquationVertex &i = getVertex<EquationVertex>(iDescriptor);
    assert(!i.isColored() && "Equation already colored");

    // (1) Color the equation
    i.setColored(true);

    // (2) If an unassigned variable exists, assign it to the current equation.
    for (EdgeDescriptor edgeDescriptor : getEdgesRange(iDescriptor)) {
      const IndexSet &incidentIndices =
          graph[edgeDescriptor].accessedVariableIndices();
      const auto &j = getVertex<VariableVertex>(edgeDescriptor.to);

      // Skip hidden variables
      if (j.isHiddenAt(incidentIndices)) {
        // NOTE: Need to figure out what happens at a partial overlap here?
        assert(!j.getVisibleIndices().overlaps(incidentIndices) &&
               "Partial overlap...");
        continue;
      }

      auto existingAssignment = assignments.find(j.getId());
      if (existingAssignment == assignments.end()) {
        existingAssignment =
            assignments.insert({j.getId(), llvm::SmallVector<Assignment>()})
                .first;
      }
      llvm::SmallVector<Assignment> &existingAssignments =
          existingAssignment->getSecond();

      // Assign the variable if it is not assigned at any incident indices
      if (!llvm::any_of(existingAssignments, [&](const Assignment &assignment) {
            return incidentIndices.overlaps(assignment.second);
          })) {
        existingAssignments.emplace_back(
            std::make_pair(i.getId(), incidentIndices));
        return true;
      }
    }

    for (EdgeDescriptor edgeDescriptor : getEdgesRange(iDescriptor)) {
      Edge &edge = graph[edgeDescriptor];
      VariableVertex &j = getVertex<VariableVertex>(edgeDescriptor.to);
      IndexSet incidentIndices = edge.accessedVariableIndices();

      // Skip hidden variables
      if (j.isHiddenAt(incidentIndices)) {
        // NOTE: Need to figure out what happens at a partial overlap here?
        assert(!j.getVisibleIndices().overlaps(incidentIndices) &&
               "Partial overlap...");
        continue;
      }

      if (j.isColoredAt(incidentIndices)) {
        continue;
      }

      j.colorIndices(incidentIndices);

      // TODO: See if this is still correct with the loop.
      // As the variable is colored, it must be assigned.
      llvm::SmallVectorImpl<Assignment> &existingAssignments =
          assignments[j.getId()];
      bool foundAugmentingPath = false;
      for (const Assignment &assignment : existingAssignments) {
        EquationVertex::Id k = assignment.first;
        if (augmentPath(k, assignments)) {
          existingAssignments.emplace_back(
              std::make_pair(id, edge.accessedVariableIndices()));
          foundAugmentingPath = true;
        }
      }

      if (foundAugmentingPath) {
        return true;
      }
    }

    return false;
  }

  /// Remove coloring from all vertices in the graph.
  void uncolorAllVertices() {
    for (VertexDescriptor descriptor : getDescriptorRange<VariableVertex>()) {
      getVertex<VariableVertex>(descriptor).clearColoring();
    }
    for (VertexDescriptor descriptor : getDescriptorRange<EquationVertex>()) {
      getVertex<EquationVertex>(descriptor).setColored(false);
    }
  }

  /// Hide the (indices of) variables that have derivatives (of those indices).
  void hideDerivedVariables() {
    for (VertexDescriptor descriptor : getDescriptorRange<VariableVertex>()) {
      VariableVertex &variable = getVertex<VariableVertex>(descriptor);
      if (auto derivative = getDerivative(variable.getId())) {
        const IndexSet &derivedIndices = derivative->second;
        variable.hideIndices(derivedIndices);
        // Here we might want to hide parts of the edge as well, for now we
        // must check if the variable indices are hidden at arrival.
      }
    }
  }

public:
  IndexReductionGraph(
      const std::function<VariableBridge *(
          const VariableBridge *, const IndexSet &)> &differentiateVariable,
      const std::function<EquationBridge *(const EquationBridge *)>
          &differentiateEquation)
      : differentiateVariable(differentiateVariable),
        differentiateEquation(differentiateEquation) {}

  /// Add a variable to the graph.
  void addVariable(VariableBridge *variableBridge) {
    VariableVertex variable(variableBridge);
    VariableVertex::Id id = variable.getId();
    assert(!hasVariableWithId(id) && "Already existing variable");
    VertexDescriptor variableDescriptor = graph.addVertex(std::move(variable));
    variablesMap[id] = variableDescriptor;
  }

  /// Add an equation to the graph, all variables accessed by the equation are
  /// expected to already be present in the graph.
  void addEquation(EquationBridge *equationBridge) {
    EquationVertex eq(equationBridge);
    EquationVertex::Id id = eq.getId();
    assert(!hasEquationWithId(id) && "Already existing equation");
    VertexDescriptor equationDescriptor = graph.addVertex(std::move(eq));
    equationsMap[id] = equationDescriptor;

    const EquationVertex &equation =
        getVertex<EquationVertex>(equationDescriptor);
    IndexSet equationRanges = equation.getIterationRanges();

    for (const Access &access : equation.getVariableAccesses()) {
      VertexDescriptor variableDescriptor =
          getVariableDescriptorFromId(access.getVariableId());
      const VariableVertex &variable =
          getVertex<VariableVertex>(variableDescriptor);

      for (const MultidimensionalRange &range :
           llvm::make_range(variable.getIndices().rangesBegin(),
                            variable.getIndices().rangesEnd())) {
        graph.addEdge(
            equationDescriptor, variableDescriptor,
            {variable.getId(), equationRanges, IndexSet(range), access});
      }
    }
  }

  /// Establish the relationship between variables and derivative variables.
  void setDerivatives(const mlir::bmodelica::DerivativesMap &derivativesMap) {
    for (const VariableVertex &variable : getVertexRange<VariableVertex>()) {
      const VariableVertex::Id variableId = variable.getId();
      if (auto derivativeId = derivativesMap.getDerivative(variableId)) {

        IndexSet indices;
        if (auto subIndices = derivativesMap.getDerivedIndices(variableId);
            subIndices && !subIndices->get().empty()) {
          indices = *subIndices;
        } else {
          // As no derived indices were specified, the variable must be a
          // scalar. getIndices is used to get indices == [0, 1).
          indices = variable.getIndices();
        }

        variableAssociations.try_emplace(
            variableId, std::make_pair(*derivativeId, indices));
      }
    }
  }

  /// Apply the pantelides algorithm to the graph.
  Assignments pantelides() {
    Assignments variableAssignments;

    size_t numEquations = llvm::count_if(getDescriptorRange<EquationVertex>(),
                                         [](auto) { return true; });
    LLVM_DEBUG({
      size_t numVariables = 0;
      size_t numIndices = 0;
      for (const VariableVertex &variable : getVertexRange<VariableVertex>()) {
        numVariables++;
        numIndices += variable.getIndices().flatSize();
      }
      llvm::dbgs() << "Pantelides initial state:\n"
                   << " #equations: " << numEquations << "\n"
                   << " #variables: " << numVariables << " ->"
                   << " #indices: " << numIndices << "\n";
    });

    // TODO: Check if the end of the iterator is moved when a new equation is
    //       added. This should only iterate over the equations that were
    //       initially part of the model.
    for (size_t kId = 0; kId < numEquations; kId++) {
      LLVM_DEBUG(llvm::dbgs() << "----------\n" << "k = " << kId << "\n");
      // 3a
      EquationVertex i = getVertex<EquationVertex>(
          getEquationDescriptorFromId(static_cast<int64_t>(kId)));
      // 3b
      while (true) {
        LLVM_DEBUG({
          llvm::dbgs() << "---\n"
                       << "i = " << i.getId() << "\n";
          dump(llvm::dbgs());
        });

        // 3b-1
        hideDerivedVariables();
        // 3b-2
        uncolorAllVertices();
        // 3b-(3 & 4)

        LLVM_DEBUG({
          dumpVisibilityState(llvm::dbgs());
          llvm::dbgs() << "Augmenting path from " << i.getId();
        });
        bool res = augmentPath(i.getId(), variableAssignments);
        LLVM_DEBUG({
          llvm::dbgs() << (res ? " SUCCEEDED" : " FAILED") << "\n";
          if (!res)
            dumpColoringState(llvm::dbgs());
          dumpAssignmentState(llvm::dbgs(), variableAssignments);
        });

        // 3b-5
        if (!res) {
          // 3b-5 (i) - Differentiate visited variables
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

            // TODO: This needs to be reworked to handle indices correctly.
            //       The original variable might already be differentiated.
            //       So the functor might return a modified version of the
            //       original. Atm we crash if this happens.
            VariableBridge *dj =
                differentiateVariable(j.getBridge(), coloredIndices);
            addVariable(dj);
            variableAssociations.insert({j.getId(), {dj->id, IndexSet()}});
          }

          // 3b-5 (ii) - Differentiate visited equations
          for (auto lDescriptor : getDescriptorRange<EquationVertex>()) {
            const EquationVertex &l = getVertex<EquationVertex>(lDescriptor);
            if (!l.isColored()) {
              continue;
            }
            EquationVertex dl(differentiateEquation(l.getBridge()));
            assert(!hasEquationWithId(dl.getId()) &&
                   "Already existing equation");
            assert(dl.getIterationRanges() == l.getIterationRanges() &&
                   "Differentiated equation has wrong iteration ranges");
            VertexDescriptor dlDescriptor = graph.addVertex(std::move(dl));
            equationsMap[dl.getId()] = dlDescriptor;

            // TODO: Add edges from dl to the variables that have edges with
            // l. And their derivatives.
            for (EdgeDescriptor edgeDescriptor : getEdgesRange(lDescriptor)) {
              const Edge &edge = graph[edgeDescriptor];
              const auto &j = getVertex<VariableVertex>(edgeDescriptor.to);
              LLVM_DEBUG(llvm::dbgs() << "Adding edge(s) from " << dl.getId()
                                      << " to " << j.getId());

              graph.addEdge(dlDescriptor, edgeDescriptor.to, edge);

              if (auto dj = getDerivative(j.getId())) {
                LLVM_DEBUG(llvm::dbgs() << ", " << dj->first);
                assert(dj->second.contains(edge.accessedVariableIndices()) &&
                       "Variable derivative does not contain accessed "
                       "indices");

                graph.addEdge(dlDescriptor,
                              getVariableDescriptorFromId(dj->first), edge);
              }
              LLVM_DEBUG(llvm::dbgs() << "\n");
            }

            equationAssociations[l.getId()] = dl.getId();
          }

          // 3b-5 (iii) - Assign derivatives of colored variables to the
          // derivatives of their assigned equations.

          // 3b-5 (iv) - Continue from the derivative of the current equation
          EquationVertex::Id nextId = equationAssociations[i.getId()];
          i = getVertex<EquationVertex>(getEquationDescriptorFromId(nextId));
        } else {
          break;
        }
        llvm::dbgs() << "Press enter to continue\n";
        std::cin.get();
      }
    }

    return variableAssignments;
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
      if (l.isColored()) {
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

  void dumpAssignmentState(llvm::raw_ostream &os,
                           const Assignments &variableAssignments) const {
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

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "---\n" << "Index Reduction Graph:\n";

    os << " Variables:\n";
    for (const VariableVertex &variable : getVertexRange<VariableVertex>()) {
      os << "  ", variable.dump(os);
      if (auto derivative = getDerivative(variable.getId())) {
        os << " -> ";
        os << "Derivative(id: " << derivative->first << ")";
        os << " for indices: ";
        derivative->second.dump(os);
      }
      os << "\n";
    }

    os << " Equations:\n";
    for (const VertexDescriptor &descriptor :
         getDescriptorRange<EquationVertex>()) {
      auto &equation = getVertex<EquationVertex>(descriptor);
      os << "  ", equation.dump(os);
      if (auto derivative = getDerivative(equation.getId())) {
        os << " -> ";
        os << "Derivative(id: " << *derivative << ")";
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

  std::function<VariableBridge *(const VariableBridge *, const IndexSet &)>
      differentiateVariable;
  std::function<EquationBridge *(const EquationBridge *)> differentiateEquation;

  llvm::DenseMap<VariableVertex::Id, VertexDescriptor> variablesMap;
  llvm::DenseMap<EquationVertex::Id, VertexDescriptor> equationsMap;

  /// Associates a variable with its derivative.
  /// var -> (var', indices of var)
  llvm::DenseMap<VariableVertex::Id, std::pair<VariableVertex::Id, IndexSet>>
      variableAssociations;

  /// Associates an equation with its derivative.
  /// eq -> eq'
  llvm::DenseMap<EquationVertex::Id, EquationVertex::Id> equationAssociations;
};

} // namespace marco::modeling

#endif // MARCO_MODELING_INDEXREDUCTION_H
