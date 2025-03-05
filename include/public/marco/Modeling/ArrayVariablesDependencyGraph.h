#ifndef MARCO_MODELING_ARRAYVARIABLESSDEPENDENCYGRAPH_H
#define MARCO_MODELING_ARRAYVARIABLESSDEPENDENCYGRAPH_H

#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/SCC.h"
#include "marco/Modeling/SingleEntryDigraph.h"
#include "marco/Modeling/SingleEntryWeaklyConnectedDigraph.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"
#include <type_traits>

namespace marco::modeling {
/// Graph storing the dependencies between array equations.
/// An edge from equation A to equation B is created if the computations inside
/// A depends on B, meaning that B needs to be computed first. The order of
/// computations is therefore given by a post-order visit of the graph.
template <typename VariableProperty, typename EquationProperty,
          typename Graph =
              internal::dependency::SingleEntryWeaklyConnectedDigraph<
                  internal::dependency::ArrayVariable<VariableProperty>>>
class ArrayVariablesDependencyGraph {
public:
  using Base = Graph;

  using Variable = typename Graph::VertexProperty;
  using VariableTraits = typename Variable::Traits;

  using Equation = internal::dependency::ArrayEquation<EquationProperty>;
  using EquationTraits = typename Equation::Traits;

  using VariableDescriptor = typename Graph::VertexDescriptor;
  using AccessProperty = typename Equation::Access::Property;

  using Access =
      ::marco::modeling::dependency::Access<VariableProperty, AccessProperty>;

  using SCC = internal::dependency::SCC<Graph>;

private:
  mlir::MLIRContext *context;
  std::shared_ptr<Graph> graph;
  llvm::DenseMap<typename Variable::Id, VariableDescriptor> variableDescriptors;

public:
  template <
      typename G = Graph,
      typename = typename std::enable_if<std::is_same_v<
          G, internal::dependency::SingleEntryWeaklyConnectedDigraph<
                 typename G::VertexProperty, typename G::EdgeProperty>>>::type>
  explicit ArrayVariablesDependencyGraph(mlir::MLIRContext *context)
      : context(context) {
    graph = std::make_shared<Graph>();
  }

  ArrayVariablesDependencyGraph(mlir::MLIRContext *context,
                                std::shared_ptr<Graph> graph)
      : context(context), graph(graph) {}

  [[nodiscard]] mlir::MLIRContext *getContext() const {
    assert(context != nullptr);
    return context;
  }

  /// @name Forwarded methods
  /// {

  Variable &getVariable(VariableDescriptor descriptor) {
    return (*graph)[descriptor];
  }

  const Variable &getVariable(VariableDescriptor descriptor) const {
    return (*graph)[descriptor];
  }

  VariableProperty &operator[](VariableDescriptor descriptor) {
    return getVariable(descriptor).getProperty();
  }

  const VariableProperty &operator[](VariableDescriptor descriptor) const {
    return getVariable(descriptor).getProperty();
  }

  auto variablesBegin() const { return graph->verticesBegin(); }

  auto variablesEnd() const { return graph->verticesEnd(); }

  /// }

  VariableDescriptor addVariable(const VariableProperty &variableProperty) {
    auto descriptor = graph->addVertex(Variable(variableProperty));
    auto id = VariableTraits::getId(&variableProperty);
    variableDescriptors[id] = descriptor;
    return descriptor;
  }

  void addVariables(llvm::ArrayRef<VariableProperty> variables) {
    for (const VariableProperty &variableProperty : variables) {
      (void)addVariable(variableProperty);
    }
  }

  void addEquations(llvm::ArrayRef<EquationProperty> equations) {
    for (const EquationProperty &equationProperty : equations) {
      std::vector<Access> writeAccesses =
          EquationTraits ::getWrites(&equationProperty);

      if (!writeAccesses.empty()) {
        auto writtenVariableId = writeAccesses[0].getVariable();
        auto writtenVariableDescriptor = variableDescriptors[writtenVariableId];

        auto readAccesses = EquationTraits::getReads(&equationProperty);

        for (const Access &readAccess : readAccesses) {
          auto readVariableId = readAccess.getVariable();
          auto readVariableDescriptor = variableDescriptors[readVariableId];
          graph->addEdge(writtenVariableDescriptor, readVariableDescriptor);
        }
      }
    }
  }

  void addEdge(typename Variable::Id from, typename Variable::Id to) {
    auto fromDescriptor = variableDescriptors[from];
    auto toDescriptor = variableDescriptors[to];
    graph->addEdge(fromDescriptor, toDescriptor);
  }

  /// Get all the SCCs.
  std::vector<SCC> getSCCs() const {
    std::vector<SCC> result;

    for (auto scc = llvm::scc_begin(&static_cast<const Graph &>(*graph)),
              end = llvm::scc_end(&static_cast<const Graph &>(*graph));
         scc != end; ++scc) {
      std::vector<VariableDescriptor> variables;

      for (const auto &variable : *scc) {
        variables.push_back(variable);
      }

      // Ignore the entry node.
      if (variables.size() > 1 || variables[0] != graph->getEntryNode()) {
        result.emplace_back(*graph, variables.begin(), variables.end());
      }
    }

    return result;
  }
};
} // namespace marco::modeling

#endif // MARCO_MODELING_ARRAYVARIABLESSDEPENDENCYGRAPH_H
