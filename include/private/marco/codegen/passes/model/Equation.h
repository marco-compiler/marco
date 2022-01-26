#ifndef MARCO_CODEGEN_EQUATION_H
#define MARCO_CODEGEN_EQUATION_H

#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/model/Access.h"
#include "marco/codegen/passes/model/Path.h"
#include "marco/codegen/passes/model/Variable.h"
#include "marco/modeling/Matching.h"
#include <memory>

namespace marco::codegen
{
  class Equation
  {
    public:
    class Impl;

    Equation(modelica::EquationOp equation, Variables variables);

    Equation(const Equation& other);

    ~Equation();

    Equation& operator=(const Equation& other);
    Equation& operator=(Equation&& other);

    friend void swap(Equation& first, Equation& second);

    modelica::EquationOp getOperation() const;

    void setVariables(Variables variables);

    Equation cloneIR() const;

    void eraseIR();

    size_t getNumOfIterationVars() const;
    long getRangeBegin(size_t inductionVarIndex) const;
    long getRangeEnd(size_t inductionVarIndex) const;

    std::vector<Access> getAccesses() const;

    /*
    void getWrites(llvm::SmallVectorImpl<Access>& accesses) const;
    void getReads(llvm::SmallVectorImpl<Access>& accesses) const;
     */

    mlir::LogicalResult explicitate(const EquationPath& path);

    /// @name Matching
    /// {

    bool isMatched() const;

    const EquationPath& getMatchedPath() const;

    void setMatchedPath(EquationPath path);

    /// }

    private:
    Equation(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl;
  };

  /// Container for the equations of the model.
  class Equations
  {
    public:
      using Container = std::vector<std::unique_ptr<Equation>>;
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      Equations();

      void add(std::unique_ptr<Equation> equation);

      size_t size() const;

      std::unique_ptr<Equation>& operator[](size_t index);

      const std::unique_ptr<Equation>& operator[](size_t index) const;

      iterator begin();

      const_iterator begin() const;

      iterator end();

      const_iterator end() const;

      /// For each equation, set the variables it should consider while determining the accesses.
      void setVariables(Variables variables);

    private:
      class Impl;
      std::shared_ptr<Impl> impl;
  };
}

// Specializations for the modeling library
namespace marco::modeling
{
  namespace matching
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
}

#endif // MARCO_CODEGEN_EQUATION_H
