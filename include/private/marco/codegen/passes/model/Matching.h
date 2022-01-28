#ifndef MARCO_CODEGEN_PASSES_MODEL_MATCHING_H
#define MARCO_CODEGEN_PASSES_MODEL_MATCHING_H

#include "marco/codegen/passes/model/EquationImpl.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class MatchedEquation : public Equation
  {
    public:
      MatchedEquation(std::unique_ptr<Equation> equation, EquationPath matchedPath);

      MatchedEquation(const MatchedEquation& other);

      ~MatchedEquation();

      MatchedEquation& operator=(const MatchedEquation& other);
      MatchedEquation& operator=(MatchedEquation&& other);

      friend void swap(MatchedEquation& first, MatchedEquation& second);

      std::unique_ptr<Equation> clone() const override;

      /// @name Forwarded methods
      /// {

      modelica::EquationOp getOperation() const override;

      const Variables& getVariables() const override;

      void setVariables(Variables variables) override;

      std::vector<Access> getAccesses() const override;

      ::marco::modeling::DimensionAccess resolveDimensionAccess(
          std::pair<mlir::Value, long> access) const override;

      /// }
      /// @name Modified methods
      /// {

      size_t getNumOfIterationVars() const override;
      long getRangeBegin(size_t inductionVarIndex) const override;
      long getRangeEnd(size_t inductionVarIndex) const override;

      /// }

      /// Set the matched indexes
      void setMatchedIndexes(size_t inductionVarIndex, long begin, long end);

      std::vector<Access> getReads() const;

      Access getWrite() const;

    private:
      std::unique_ptr<Equation> equation;

      std::vector<std::pair<long, long>> matchedIndexes;
      EquationPath matchedPath;
  };
}

namespace marco::modeling::matching
{
  template<>
  struct EquationTraits<::marco::codegen::Equation*>
  {
    using Equation = ::marco::codegen::Equation*;
    using Id = mlir::Operation*;

    static Id getId(const Equation* equation)
    {
      return (*equation)->getOperation().getOperation();
    }

    static size_t getNumOfIterationVars(const Equation* equation)
    {
      return (*equation)->getNumOfIterationVars();
    }

    static long getRangeBegin(const Equation* equation, size_t inductionVarIndex)
    {
      return (*equation)->getRangeBegin(inductionVarIndex);
    }

    static long getRangeEnd(const Equation* equation, size_t inductionVarIndex)
    {
      return (*equation)->getRangeEnd(inductionVarIndex);
    }

    using VariableType = codegen::Variable*;

    using AccessProperty = codegen::EquationPath;

    static std::vector<Access<VariableType, AccessProperty>> getAccesses(const Equation* equation)
    {
      std::vector<Access<VariableType, AccessProperty>> accesses;

      for (const auto& access : (*equation)->getAccesses()) {
        accesses.emplace_back(access.getVariable(), access.getAccessFunction(), access.getPath());
      }

      return accesses;
    }
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_MATCHING_H
