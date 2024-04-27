#ifndef MARCO_VARIABLEFILTER_VARIABLEFILTER_H
#define MARCO_VARIABLEFILTER_VARIABLEFILTER_H

#include "marco/VariableFilter/Filter.h"
#include "marco/VariableFilter/Tracker.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <set>

namespace marco
{
  /// Keeps track of variables, arrays, derivatives (and regex for matching)
  /// that has to be printed during the simulation.
	class VariableFilter
  {
    public:
      using Tracker = vf::Tracker;
      using Filter = vf::Filter;

      static std::optional<VariableFilter> fromString(llvm::StringRef str);

      void dump() const;

      void dump(llvm::raw_ostream& os) const;

      bool isEnabled() const;

      void setEnabled(bool enabled);

      void addVariable(Tracker var);

      void addDerivative(Tracker var);

      void addRegexString(llvm::StringRef regex);

      std::vector<Filter> getVariableInfo(llvm::StringRef name, unsigned int expectedRank = 0) const;

      std::vector<Filter> getVariableDerInfo(llvm::StringRef name, unsigned int expectedRank = 0) const;

    private:
      /// Check whether a variable identifier matches any of the regular
      /// expressions stored within the variable filter.
      bool matchesRegex(llvm::StringRef identifier) const;

    private:
      llvm::StringMap<std::vector<Tracker>> variables;
      llvm::StringMap<std::vector<Tracker>> derivatives;
      llvm::SmallVector<std::string> regex;
      bool enabled = false;
  };
}

#endif // MARCO_VARIABLEFILTER_VARIABLEFILTER_H
