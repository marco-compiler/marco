#ifndef MARCO_CODEGEN_TRANSFORMS_MODELING_MATCHEDEQUATIONBRIDGE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELING_MATCHEDEQUATIONBRIDGE_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Modeling/VariableBridge.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::modelica::bridge
{
  class MatchedEquationBridge
  {
    public:
      MatchedEquationInstanceOp op;
      mlir::SymbolTableCollection* symbolTable;
      VariableAccessAnalysis* accessAnalysis;
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>* variablesMap;

    public:
      template<typename... Args>
      static std::unique_ptr<MatchedEquationBridge> build(Args&&... args)
      {
        return std::make_unique<MatchedEquationBridge>(
            std::forward<Args>(args)...);
      }

      MatchedEquationBridge(
          MatchedEquationInstanceOp op,
          mlir::SymbolTableCollection& symbolTable,
          VariableAccessAnalysis& accessAnalysis,
          llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap);

      // Forbid copies to avoid dangling pointers by design.
      MatchedEquationBridge(const MatchedEquationBridge& other) = delete;
      MatchedEquationBridge(MatchedEquationBridge&& other) = delete;

      MatchedEquationBridge& operator=(
          const MatchedEquationBridge& other) = delete;

      MatchedEquationBridge& operator==(
          const MatchedEquationBridge& other) = delete;
  };
}

namespace marco::modeling::dependency
{
  template<>
  struct EquationTraits<::mlir::modelica::bridge::MatchedEquationBridge*>
  {
    using Equation = ::mlir::modelica::bridge::MatchedEquationBridge*;
    using Id = mlir::Operation*;

    static Id getId(const Equation* equation)
    {
      return (*equation)->op.getOperation();
    }

    static size_t getNumOfIterationVars(const Equation* equation)
    {
      uint64_t numOfInductions = static_cast<uint64_t>(
          (*equation)->op.getInductionVariables().size());

      if (numOfInductions == 0) {
        // Scalar equation.
        return 1;
      }

      return static_cast<size_t>(numOfInductions);
    }

    static IndexSet getIterationRanges(const Equation* equation)
    {
      IndexSet iterationSpace = (*equation)->op.getIterationSpace();

      if (iterationSpace.empty()) {
        // Scalar equation.
        iterationSpace += MultidimensionalRange(Range(0, 1));
      }

      return iterationSpace;
    }

    using VariableType = ::mlir::modelica::bridge::VariableBridge*;
    using VariableAccess = mlir::modelica::VariableAccess;
    using AccessProperty = VariableAccess;

    static std::vector<Access<VariableType, AccessProperty>>
    getAccesses(const Equation* equation)
    {
      std::vector<Access<VariableType, AccessProperty>> result;
      llvm::SmallVector<VariableAccess> accesses;

      if (mlir::failed((*equation)->op.getAccesses(
              accesses, *(*equation)->symbolTable))) {
        return result;
      }

      for (VariableAccess& access : accesses) {
        auto accessFunction = getAccessFunction(
            (*equation)->op.getContext(), access);

        auto variableIt =
            (*(*equation)->variablesMap).find(access.getVariable());

        if (variableIt != (*(*equation)->variablesMap).end()) {
          result.emplace_back(
              variableIt->getSecond(),
              std::move(accessFunction),
              access);
        }
      }

      return result;
    }

    static Access<VariableType, AccessProperty> getWrite(
        const Equation* equation)
    {
      auto matchPath = (*equation)->op.getPath();

      auto write = (*equation)->op.getAccessAtPath(
          *(*equation)->symbolTable, matchPath.getValue());

      assert(write.has_value() && "Can't get the write access");

      auto accessFunction = getAccessFunction(
          (*equation)->op.getContext(), *write);

      return Access(
          (*(*equation)->variablesMap)[write->getVariable()],
          std::move(accessFunction),
          *write);
    }

    static std::vector<Access<VariableType, AccessProperty>> getReads(
        const Equation* equation)
    {
      IndexSet equationIndices = getIterationRanges(equation);

      llvm::SmallVector<VariableAccess> accesses;

      if (mlir::failed((*equation)->op.getAccesses(
              accesses, *(*equation)->symbolTable))) {
        llvm_unreachable("Can't compute the accesses");
        return {};
      }

      llvm::SmallVector<VariableAccess> readAccesses;

      if (mlir::failed((*equation)->op.getReadAccesses(
              readAccesses,
              *(*equation)->symbolTable,
              equationIndices,
              accesses))) {
        llvm_unreachable("Can't compute read accesses");
        return {};
      }

      std::vector<Access<VariableType, AccessProperty>> reads;

      for (const VariableAccess& readAccess : readAccesses) {
        auto variableIt =
            (*(*equation)->variablesMap).find(readAccess.getVariable());

        reads.emplace_back(
            variableIt->getSecond(),
            getAccessFunction((*equation)->op.getContext(), readAccess),
            readAccess);
      }

      return reads;
    }

    static std::unique_ptr<AccessFunction> getAccessFunction(
        mlir::MLIRContext* context,
        const VariableAccess& access)
    {
      const AccessFunction& accessFunction = access.getAccessFunction();

      if (accessFunction.getNumOfResults() == 0) {
        // Access to scalar variable.
        return AccessFunction::build(mlir::AffineMap::get(
            accessFunction.getNumOfDims(), 0,
            mlir::getAffineConstantExpr(0, context)));
      }

      return accessFunction.clone();
    }
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELING_MATCHEDEQUATIONBRIDGE_H
