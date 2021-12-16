#ifndef MARCO_CODEGEN_EQUATION_H
#define MARCO_CODEGEN_EQUATION_H

#include <marco/codegen/dialects/modelica/ModelicaDialect.h>
#include <marco/matching/Matching.h>

#include "Path.h"
#include "Variable.h"

namespace marco::codegen
{
  class Equation
  {
    public:
    class Impl;
    using Id = mlir::Operation*;
    using AccessProperty = EquationPath;
    using Access = matching::Access<Variable, AccessProperty>;

    Equation(modelica::EquationOp equation, llvm::ArrayRef<Variable> variables);

    Equation(const Equation& other);

    ~Equation();

    Equation& operator=(const Equation& other);
    Equation& operator=(Equation&& other);

    friend void swap(Equation& first, Equation& second);

    Equation cloneIR() const;

    void eraseIR();

    Id getId() const;
    size_t getNumOfIterationVars() const;
    long getRangeStart(size_t inductionVarIndex) const;
    long getRangeEnd(size_t inductionVarIndex) const;
    void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const;

    void getWrites(llvm::SmallVectorImpl<Access>& accesses) const;
    void getReads(llvm::SmallVectorImpl<Access>& accesses) const;

    mlir::LogicalResult explicitate(const EquationPath& path);

    private:
    Equation(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl;
  };
}

#endif // MARCO_CODEGEN_EQUATION_H
