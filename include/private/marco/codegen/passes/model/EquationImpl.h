#ifndef MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H
#define MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H

#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/Path.h"

namespace marco::codegen
{
  class Equation::Impl
  {
    public:
      Impl(modelica::EquationOp equation, Variables variables);

      virtual std::unique_ptr<Equation::Impl> clone() const = 0;

      /// Set the variables to be considered by the equation while determining the accesses.
      void setVariables(Variables variables);

      virtual std::unique_ptr<Impl> cloneIR() const = 0;
      virtual void eraseIR() = 0;

      virtual size_t getNumOfIterationVars() const = 0;
      virtual long getRangeBegin(size_t inductionVarIndex) const = 0;
      virtual long getRangeEnd(size_t inductionVarIndex) const = 0;

      virtual std::vector<Access> getAccesses() const = 0;

      modelica::EquationOp getOperation() const;

      llvm::Optional<Variable*> findVariable(mlir::Value value) const;

      bool isVariable(mlir::Value value) const;

      bool isReferenceAccess(mlir::Value value) const;

      void searchAccesses(
          std::vector<Access>& accesses,
          mlir::Value value,
          EquationPath path) const;

      void searchAccesses(
          std::vector<Access>& accesses,
          mlir::Value value,
          std::vector<::marco::modeling::DimensionAccess>& dimensionAccesses,
          EquationPath path) const;

      void searchAccesses(
          std::vector<Access>& accesses,
          mlir::Operation* op,
          std::vector<::marco::modeling::DimensionAccess>& dimensionAccesses,
          EquationPath path) const;

      void resolveAccess(
          std::vector<Access>& accesses,
          mlir::Value value,
          std::vector<::marco::modeling::DimensionAccess>& dimensionsAccesses,
          EquationPath path) const;

      std::pair<mlir::Value, long> evaluateDimensionAccess(mlir::Value value) const;

      virtual ::marco::modeling::DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const = 0;

      //virtual void getWrites(llvm::SmallVectorImpl<Access>& accesses) const = 0;
      //virtual void getReads(llvm::SmallVectorImpl<Access>& accesses) const = 0;

      mlir::LogicalResult explicitate(const EquationPath& path);

      /// @name Matching
      /// {

      bool isMatched() const;

      const EquationPath& getMatchedPath() const;

      void setMatchedPath(EquationPath path);

      /// }

    protected:
      const Variables& getVariables() const;

    private:
      mlir::LogicalResult explicitate(mlir::OpBuilder& builder, size_t argumentIndex, EquationPath::EquationSide side);

      mlir::Operation* equationOp;
      Variables variables;

      llvm::Optional<EquationPath> matchedPath;
  };
}

#endif // MARCO_CODEGEN_PASSES_MODEL_EQUATIONIMPL_H
