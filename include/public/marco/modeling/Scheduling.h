#ifndef MARCO_MODELING_SCHEDULING_H
#define MARCO_MODELING_SCHEDULING_H

#include <llvm/ADT/PostOrderIterator.h>

#include "Dependency.h"

namespace marco::modeling
{
  /*
  namespace scheduling
  {
    template<typename EquationType>
    struct EquationTraits
    {
      using Id = typename EquationType::UnknownEquationTypeError;
    };
  }
   */

  namespace internal::scheduling
  {

  }

  template<typename VariableProperty, typename EquationProperty>
  class Scheduler
  {
    private:
      using VectorDependencyGraph = VVarDependencyGraph<VariableProperty, EquationProperty>;
      //using SCCDependencyGraph = internal::dependency::SCCDependencyGraph<typename VectorDependencyGraph::SCC>;

    public:
      bool schedule(llvm::ArrayRef<EquationProperty> equations) const
      {
        VectorDependencyGraph vectorDependencyGraph(equations);
        auto SCCs = vectorDependencyGraph.getSCCs();

        //SCCDependencyGraph sccDependencyGraph(SCCs);

        /*
        // Sort the SCCs in topological order

        for (const auto& graph : graphs) {
          std::set<EquationDescriptor> set;

          for (EquationDescriptor node : llvm::post_order_ext(graph, set)) {
            std::cout << graph[node].getId() << " ";
          }

          std::cout << "\n";
        }
         */

        return true;
      }

    private:
  };
}

#endif // MARCO_MODELING_SCHEDULING_H
