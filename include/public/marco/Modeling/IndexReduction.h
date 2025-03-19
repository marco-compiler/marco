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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include <variant>

namespace marco::modeling {

namespace internal::indexReduction {

class VariableVertex final : public Dumpable {
public:
  using Bridge = mlir::bmodelica::bridge::VariableBridge;
  using Id = Bridge::Id;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Variable(";
    os << "id: " << bridge->id;
    os << ")";
  }

  Id getId() const { return bridge->id; }

  const IndexSet &getIndices() const { return bridge->indices; }

  explicit VariableVertex(Bridge *bridge) : bridge(bridge) {}

private:
  Bridge *bridge;
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

  Access& operator=(Access&& other) noexcept {
    Access tmp(std::move(other));
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
  using Bridge = mlir::bmodelica::bridge::EquationBridge;
  using Id = int64_t;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Equation(";
    os << "id: " << bridge->getId();
    os << ")";
  }

  Id getId() const { return bridge->getId(); }

  explicit EquationVertex(Bridge *bridge) : bridge(bridge) {}

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
        return convertAccesses(bridge, *cachedAccesses);
      }
    }

    llvm::SmallVector<mlir::bmodelica::VariableAccess> accesses;
    if (mlir::succeeded(bridge->getOp().getAccesses(
            accesses, bridge->getSymbolTableCollection()))) {
      return convertAccesses(bridge, accesses);
    }

    llvm_unreachable("Can't compute the accesses");
  }

  static llvm::SmallVector<Access> convertAccesses(
      const Bridge *bridge,
      const llvm::ArrayRef<mlir::bmodelica::VariableAccess> accesses) {
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
  Bridge *bridge;
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

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "    " << variableId;
    if (incidenceMatrix.getEquationRanges().flatSize() == 1 &&
        incidenceMatrix.getVariableRanges().flatSize() == 1) {
      os << " - scalar\n";
    } else {
      os << " - incidence matrix:\n" << incidenceMatrix;
    }
  }

private:
  VariableVertex::Id variableId;

  std::unique_ptr<AccessFunction> accessFunction;
  mlir::bmodelica::EquationPath equationPath;
  MCIM incidenceMatrix;
};

} // namespace internal::indexReduction

class IndexReductionGraph final : public internal::Dumpable {
public:
  using Variable = internal::indexReduction::VariableVertex;
  using Equation = internal::indexReduction::EquationVertex;
  using Vertex = std::variant<Variable, Equation>;
  using Edge = internal::indexReduction::Edge;

private:
  using Graph = internal::UndirectedGraph<Vertex, Edge>;
  using VertexDescriptor = Graph::VertexDescriptor;
  using EdgeDescriptor = Graph::EdgeDescriptor;
  using VertexIterator = Graph::VertexIterator;
  using EdgeIterator = Graph::EdgeIterator;

  using VariableIterator = Graph::FilteredVertexIterator;
  using EquationIterator = Graph::FilteredVertexIterator;

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
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<VertexType>(vertex);
    };
    return llvm::make_range(graph.verticesBegin(filter),
                            graph.verticesEnd(filter));
  }

  template <typename VertexType>
  auto getVertexRange() const {
    return llvm::map_range(
        getDescriptorRange<VertexType>(), [&](auto descriptor) {
          return std::ref(getVertexFromDescriptor<VertexType>(descriptor));
        });
  }

  auto edgesRange(VertexDescriptor vertex) const {
    return llvm::make_range(graph.outgoingEdgesBegin(vertex),
                            graph.outgoingEdgesEnd(vertex));
  }

public:
  void addVariable(Variable::Bridge *variableBridge) {
    Variable variable(variableBridge);
    auto id = variable.getId();
    assert(!hasVariableWithId(id) && "Already existing variable");
    VertexDescriptor variableDescriptor = graph.addVertex(std::move(variable));
    variablesMap[id] = variableDescriptor;
  }

  void addEquation(Equation::Bridge *equationBridge) {
    Equation eq(equationBridge);
    Equation::Id id = eq.getId();
    assert(!hasEquationWithId(id) && "Already existing equation");
    VertexDescriptor equationDescriptor = graph.addVertex(std::move(eq));
    equationsMap[id] = equationDescriptor;

    const Equation &equation =
        getVertexFromDescriptor<Equation>(equationDescriptor);
    IndexSet equationRanges = equation.getIterationRanges();

    for (const auto &access : equation.getVariableAccesses()) {
      VertexDescriptor variableDescriptor =
          getVariableDescriptorFromId(access.getVariableId());
      const Variable &variable =
          getVertexFromDescriptor<Variable>(variableDescriptor);

      IndexSet indices = variable.getIndices().getCanonicalRepresentation();

      for (const MultidimensionalRange &range :
           llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
        Edge edge(variable.getId(), equationRanges, IndexSet(range), access);

        graph.addEdge(equationDescriptor, variableDescriptor, std::move(edge));
      }
    }
  }

  void setDerivatives(const mlir::bmodelica::DerivativesMap &derivativesMap) {
    for (const VertexDescriptor &variableDescriptor :
         getDescriptorRange<Variable>()) {
      const Variable &variable =
          getVertexFromDescriptor<Variable>(variableDescriptor);

      if (auto derivativeId = derivativesMap.getDerivative(variable.getId())) {

        IndexSet indices;
        if (auto subIndices = derivativesMap.getDerivedIndices(variable.getId())) {
          indices = *subIndices;
        } else {
          indices = variable.getIndices();
        }

        variableAssociations.insert({
            variable.getId(),
            std::make_pair(*derivativeId, indices),
        });
      }
    }
  }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "-----\n" << "Index Reduction Graph:\n";

    os << " Variables:\n";
    for (const Variable &variable : getVertexRange<Variable>()) {
      os << "  ";
      variable.dump(os);
      if (auto derivative = getDerivative(variable.getId())) {
        os << " -> ";
        os << "Derivative(id: " << derivative->first << ")";
        os << " with indices: ";
        derivative->second.dump(os);
      }
      os << "\n";
    }

    os << " Equations:\n";
    for (const VertexDescriptor &descriptor : getDescriptorRange<Equation>()) {
      const Equation &equation = getVertexFromDescriptor<Equation>(descriptor);
      os << "  ";
      equation.dump(os);
      os << "\n";
      for (const EdgeDescriptor &edgeDescriptor : edgesRange(descriptor)) {
        graph[edgeDescriptor].dump(os);
      }
    }
    os << "-----\n";
  };

private:
  bool hasVariableWithId(const Variable::Id id) const {
    return variablesMap.find(id) != variablesMap.end();
  }

  bool hasEquationWithId(const Equation::Id id) const {
    return equationsMap.find(id) != equationsMap.end();
  }

  VertexDescriptor getVariableDescriptorFromId(const Variable::Id id) const {
    auto it = variablesMap.find(id);
    assert(it != variablesMap.end() && "Variable not found");
    return it->second;
  }

  VertexDescriptor getEquationDescriptorFromId(const Equation::Id id) const {
    auto it = equationsMap.find(id);
    assert(it != equationsMap.end() && "Equation not found");
    return it->second;
  }

  std::optional<std::pair<Variable::Id, IndexSet>>
  getDerivative(const Variable::Id id) const {
    auto it = variableAssociations.find(id);
    if (it != variableAssociations.end()) {
      return it->getSecond();
    }
    return std::nullopt;
  }

  Graph graph;

  llvm::DenseMap<Variable::Id, VertexDescriptor> variablesMap;
  llvm::DenseMap<Equation::Id, VertexDescriptor> equationsMap;

  // Associates a variable with its derivative.
  // var -> var' (for the given indices)
  llvm::DenseMap<Variable::Id, std::pair<Variable::Id, IndexSet>>
      variableAssociations;
  // Associates an equation with its derivative.
  // eq -> eq'
  llvm::DenseMap<Equation::Id, Equation::Id> equationAssociations;
};

} // namespace marco::modeling

#endif // MARCO_MODELING_INDEXREDUCTION_H
