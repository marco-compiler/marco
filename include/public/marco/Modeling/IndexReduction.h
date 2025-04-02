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

  void hideIndices(const IndexSet &indices) { visibleIndices -= indices; }
  const IndexSet &getVisibleIndices() const { return visibleIndices; }

  const IndexSet &getIndices() const {
    assert(!bridge->indices.empty());
    return bridge->indices;
  }

  explicit VariableVertex(VariableBridge *bridge)
      : bridge(bridge), visibleIndices(bridge->indices) {}

  VariableBridge *getBridge() const { return bridge; }

  using Dumpable::dump;

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
         std::unique_ptr<AccessFunction> accessFunction,
         EquationPath equationPath)
      : variableId(variableId), accessFunction(std::move(accessFunction)),
        equationPath(std::move(equationPath)) {}

  Access(const Access &other)
      : variableId(other.variableId),
        accessFunction(other.accessFunction->clone()),
        equationPath(other.equationPath) {}

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

  const EquationPath &getEquationPath() const { return equationPath; }

private:
  VariableVertex::Id variableId;
  std::unique_ptr<AccessFunction> accessFunction;
  EquationPath equationPath;
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

  using Dumpable::dump;

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
                            std::move(accessFunction), access.getPath());
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
      : variableId(variableId),
        accessFunction(access.getAccessFunction().clone()),
        equationPath(access.getEquationPath()),
        incidenceMatrix(std::move(equationRanges), std::move(variableRanges)) {
    incidenceMatrix.apply(*accessFunction);
  }

  IndexSet getAccessRanges() const { return incidenceMatrix.flattenRows(); }

  bool isVisible() const { return visible; }
  void setVisible(bool visible) { this->visible = visible; }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "    " << variableId;
    if (incidenceMatrix.getEquationSpace().flatSize() == 1 &&
        incidenceMatrix.getVariableSpace().flatSize() == 1) {
      os << " - scalar\n";
    } else {
      os << " - incidence matrix:\n" << incidenceMatrix;
    }
  }

private:
  VariableVertex::Id variableId;
  bool visible = true;

  std::unique_ptr<AccessFunction> accessFunction;
  mlir::bmodelica::EquationPath equationPath;
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
  VertexType &getVertexFromDescriptor(const VertexDescriptor descriptor) {
    Vertex &vertex = graph[descriptor];
    assert(std::holds_alternative<VertexType>(vertex) &&
           "Invalid vertex type for descriptor");
    return std::get<VertexType>(vertex);
  }

  template <typename VertexType>
  const VertexType &
  getVertexFromDescriptor(const VertexDescriptor descriptor) const {
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
    return llvm::map_range(
        getDescriptorRange<VertexType>(), [&](VertexDescriptor descriptor) {
          return std::ref(getVertexFromDescriptor<VertexType>(descriptor));
        });
  }

  /// Returns a range of the incident edges of [vertex].
  auto edgesRange(VertexDescriptor vertex) const {
    return llvm::make_filter_range(
        llvm::make_range(graph.outgoingEdgesBegin(vertex),
                         graph.outgoingEdgesEnd(vertex)),
        [&](const EdgeDescriptor &descriptor) {
          return graph[descriptor].isVisible();
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
    auto it = variableAssociations.find(id);
    if (it != variableAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  /// Augment path procedure from the Pantelides algorithm.
  bool augmentPath(EquationVertex::Id id, Assignments &assignments) {
    VertexDescriptor iDescriptor = getEquationDescriptorFromId(id);
    EquationVertex &i = getVertexFromDescriptor<EquationVertex>(iDescriptor);
    assert(!i.isColored() && "Equation already colored");
    // Color initial equation
    i.setColored(true);

    // If an unassigned variable exists, assign it to the current equation.
    for (EdgeDescriptor edgeDescriptor : edgesRange(iDescriptor)) {
      Edge &edge = graph[edgeDescriptor];
      VariableVertex &j =
          getVertexFromDescriptor<VariableVertex>(edgeDescriptor.to);

      llvm::SmallVector<Assignment> existingAssignments;
      if (auto existingAssignment = assignments.find(j.getId());
          existingAssignment != assignments.end()) {
        // Get existingAssignment
        existingAssignments = existingAssignment->getSecond();
      } else {
        // Create new assignment
        existingAssignments =
            assignments.insert({j.getId(), {}}).first->getSecond();
      }

      // Assign the variable if it is not assigned at any incident indices
      if (IndexSet incidentIndices = edge.getAccessRanges();
          !llvm::any_of(existingAssignments, [&](const Assignment &assignment) {
            return incidentIndices.overlaps(assignment.second);
          })) {
        existingAssignments.emplace_back(
            std::make_pair(i.getId(), incidentIndices));
        return true;
      }
    }

    for (EdgeDescriptor edgeDescriptor : edgesRange(iDescriptor)) {
      Edge &edge = graph[edgeDescriptor];
      VariableVertex &j =
          getVertexFromDescriptor<VariableVertex>(edgeDescriptor.to);
      IndexSet incidentIndices = edge.getAccessRanges();

      if (j.getColoredIndices().overlaps(incidentIndices)) {
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
              std::make_pair(id, edge.getAccessRanges()));
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
      getVertexFromDescriptor<VariableVertex>(descriptor).clearColoring();
    }
    for (VertexDescriptor descriptor : getDescriptorRange<EquationVertex>()) {
      getVertexFromDescriptor<EquationVertex>(descriptor).setColored(false);
    }
  }

  /// Hide the (indices of) variables that have derivatives (of those indices).
  void hideDerivedVariables() {
    for (VertexDescriptor descriptor : getDescriptorRange<VariableVertex>()) {
      VariableVertex &variable =
          getVertexFromDescriptor<VariableVertex>(descriptor);
      if (auto derivative = getDerivative(variable.getId())) {
        const IndexSet &derivedIndices = derivative->second;
        variable.hideIndices(derivedIndices);
        // TODO: Hide the relevant indices, this is not correct for subsets of
        // array variables atm
        for (EdgeDescriptor edgeDescriptor : edgesRange(descriptor)) {
          graph[edgeDescriptor].setVisible(false);
        }
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
    numVariables++;
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
        getVertexFromDescriptor<EquationVertex>(equationDescriptor);
    IndexSet equationRanges = equation.getIterationRanges();

    for (const Access &access : equation.getVariableAccesses()) {
      VertexDescriptor variableDescriptor =
          getVariableDescriptorFromId(access.getVariableId());
      const VariableVertex &variable =
          getVertexFromDescriptor<VariableVertex>(variableDescriptor);

      IndexSet indices = variable.getIndices();

      for (const MultidimensionalRange &range :
           llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
        Edge edge(variable.getId(), equationRanges, IndexSet(range), access);

        graph.addEdge(equationDescriptor, variableDescriptor, std::move(edge));
      }
    }
    numEquations++;
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

    LLVM_DEBUG({
      size_t numEquations = llvm::count_if(getDescriptorRange<EquationVertex>(),
                                           [](auto) { return true; });
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
    for (const EquationVertex &k : getVertexRange<EquationVertex>()) {
      // 3a
      EquationVertex i = k;
      // 3b
      while (true) {
        // 3b-1
        hideDerivedVariables();
        // 3b-2
        uncolorAllVertices();
        // 3b-(3 & 4)
        bool res = augmentPath(i.getId(), variableAssignments);
        LLVM_DEBUG({
          llvm::dbgs() << "-----\n"
                       << "Augmenting path from equation " << i.getId() << " "
                       << (res ? "succeeded" : "failed") << "\n";
        });
        // 3b-5
        if (!res) {
          // 3b-5 (i) - Differentiate visited variables
          for (const VariableVertex &j : getVertexRange<VariableVertex>()) {
            const IndexSet &coloredIndices = j.getColoredIndices();
            if (coloredIndices.empty()) {
              continue;
            }

            LLVM_DEBUG(llvm::dbgs()
                           << "Variable " << j.getId() << " is colored "
                           << coloredIndices << "\n";);

            // TODO: This needs to be reworked to handle indices correctly
            VariableBridge *dj =
                differentiateVariable(j.getBridge(), coloredIndices);
            addVariable(dj);
            variableAssociations.insert({j.getId(), {dj->id, IndexSet()}});
          }

          LLVM_DEBUG({
            llvm::dbgs() << "Colored equations:";
            for (const EquationVertex &l : getVertexRange<EquationVertex>()) {
              if (l.isColored()) {
                llvm::dbgs() << " " << l.getId();
              }
            }
            llvm::dbgs() << "\n";
          });
          // 3b-5 (ii) - Differentiate visited equations
          for (VertexDescriptor lDescriptor :
               getDescriptorRange<EquationVertex>()) {
            EquationVertex &l =
                getVertexFromDescriptor<EquationVertex>(lDescriptor);
            if (l.isColored()) {
              EquationVertex dl(differentiateEquation(l.getBridge()));
              assert(!hasEquationWithId(dl.getId()) &&
                     "Already existing equation");
              VertexDescriptor dlDescriptor = graph.addVertex(std::move(dl));
              equationsMap[dl.getId()] = dlDescriptor;
              numEquations++;

              for (EdgeDescriptor edgeDescriptor : edgesRange(lDescriptor)) {
                // TODO: add edges
              }

              equationAssociations.insert({l.getId(), dl.getId()});
            }
          }

          // 3b-5 (iii) - Assign derivatives of colored variables to the derivatives of their assigned equations.

          // 3b-5 (iv) - Continue from the derivative of the current equation
          EquationVertex::Id nextId = equationAssociations[i.getId()];
          i = getVertexFromDescriptor<EquationVertex>(
              getEquationDescriptorFromId(nextId));
        } else {
          break;
        }
      }
    }

    return variableAssignments;
  }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "-----\n" << "Index Reduction Graph:\n";

    os << " Variables:\n";
    for (const VariableVertex &variable : getVertexRange<VariableVertex>()) {
      os << "  ";
      variable.dump(os);
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
      const EquationVertex &equation =
          getVertexFromDescriptor<EquationVertex>(descriptor);
      os << "  ";
      equation.dump(os);
      os << "\n";
      for (EdgeDescriptor edgeDescriptor : edgesRange(descriptor)) {
        graph[edgeDescriptor].dump(os);
      }
    }
    os << "-----\n";
  };

private:
  Graph graph;

  size_t numEquations = 0;
  size_t numVariables = 0;

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
