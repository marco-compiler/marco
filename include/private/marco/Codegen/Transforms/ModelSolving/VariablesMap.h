#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLESMAP_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLESMAP_H

#include "marco/Modeling/IndexSet.h"
#include "llvm/ADT/StringMap.h"
#include <map>

namespace marco::codegen
{
  enum class VariableType
  {
    /// A variable for which its derivative does not explicitely appear.
    ALGEBRAIC,

    /// A variable for which it exists a derivative variable.
    DIFFERENTIAL
  };

  class SplitVariable
  {
    public:
      SplitVariable(modeling::MultidimensionalRange indices, unsigned int argNumber);

      unsigned int getArgNumber() const;

      VariableType getType() const;

      const modeling::MultidimensionalRange& getIndices() const;

    private:
      modeling::MultidimensionalRange indices;
      unsigned int argNumber;
  };

  class VariablesMap
  {
    public:
      void add(llvm::StringRef name, SplitVariable variable);

      const std::vector<SplitVariable>& operator[](llvm::StringRef name) const;

    private:
      llvm::StringMap<std::vector<SplitVariable>> variables;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_VARIABLESMAP_H
