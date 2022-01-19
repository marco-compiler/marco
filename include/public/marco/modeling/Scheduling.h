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
      using VectorDependencyGraph = internal::VVarDependencyGraph<VariableProperty, EquationProperty>;
      using SCCDependencyGraph = internal::SCCDependencyGraph<typename VectorDependencyGraph::SCC>;

    public:
      bool schedule(llvm::ArrayRef<EquationProperty> equations) const
      {
        VectorDependencyGraph vectorDependencyGraph(equations);
        auto SCCs = vectorDependencyGraph.getSCCs();
        SCCDependencyGraph sccDependencyGraph(SCCs);
        auto scheduledSCCs = sccDependencyGraph.postOrder();

        for (const auto& sccDescriptor : scheduledSCCs) {
          const auto& scc = sccDependencyGraph[sccDescriptor];
          const auto& originalGraph = scc.getGraph();

          for (const auto& equationDescriptor : scc) {
            std::cout << originalGraph[equationDescriptor].getId() << " ";
          }

          std::cout << "\n";
        }

        return true;
      }

    private:
  };
}

#endif // MARCO_MODELING_SCHEDULING_H
