#ifndef MARCO_CODEGEN_TRANSFORMS_MODELING_EQUATIONBRIDGE_H
#define MARCO_CODEGEN_TRANSFORMS_MODELING_EQUATIONBRIDGE_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"
#include "marco/Codegen/Transforms/Modeling/VariableBridge.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::modelica::bridge
{
  class EquationBridge
  {
    public:
      EquationInstanceOp op;
      mlir::SymbolTableCollection* symbolTable;
      VariableAccessAnalysis* accessAnalysis;
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>* variablesMap;

    public:
      template<typename... Args>
      static std::unique_ptr<EquationBridge> build(Args&&... args)
      {
        return std::make_unique<EquationBridge>(std::forward<Args>(args)...);
      }

      EquationBridge(
          EquationInstanceOp op,
          mlir::SymbolTableCollection& symbolTable,
          VariableAccessAnalysis& accessAnalysis,
          llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap);

      // Forbid copies to avoid dangling pointers by design.
      EquationBridge(const EquationBridge& other) = delete;
      EquationBridge(EquationBridge&& other) = delete;
      EquationBridge& operator=(const EquationBridge& other) = delete;
      EquationBridge& operator==(const EquationBridge& other) = delete;
  };
}

namespace marco::modeling::matching
{
  template<>
  struct EquationTraits<::mlir::modelica::bridge::EquationBridge*>
  {
    using Equation = ::mlir::modelica::bridge::EquationBridge*;
    using Id = mlir::Operation*;

    static Id getId(const Equation* equation)
    {
      return (*equation)->op.getOperation();
    }

    static size_t getNumOfIterationVars(const Equation* equation)
    {
      auto numOfInductions = static_cast<uint64_t>(
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
    using AccessProperty = ::mlir::modelica::EquationPath;

    static std::vector<Access<VariableType, AccessProperty>>
    getAccesses(const Equation* equation)
    {
      std::vector<Access<VariableType, AccessProperty>> accesses;

      auto cachedAccesses = (*equation)->accessAnalysis->getAccesses(
          (*equation)->op, *(*equation)->symbolTable);

      if (cachedAccesses) {
        for (auto& access : *cachedAccesses) {
          auto accessFunction = getAccessFunction(
              (*equation)->op.getContext(), access);

          auto variableIt =
              (*(*equation)->variablesMap).find(access.getVariable());

          if (variableIt != (*(*equation)->variablesMap).end()) {
            accesses.emplace_back(
                variableIt->getSecond(),
                std::move(accessFunction),
                access.getPath());
          }
        }
      }

      return accesses;
    }

    static std::unique_ptr<AccessFunction> getAccessFunction(
        mlir::MLIRContext* context,
        const mlir::modelica::VariableAccess& access)
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

#endif // MARCO_CODEGEN_TRANSFORMS_MODELING_EQUATIONBRIDGE_H
