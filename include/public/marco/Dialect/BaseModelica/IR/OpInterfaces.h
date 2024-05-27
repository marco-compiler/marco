#ifndef MARCO_DIALECT_BASEMODELICA_IR_OPINTERFACES_H
#define MARCO_DIALECT_BASEMODELICA_IR_OPINTERFACES_H

#include "marco/Dialect/BaseModelica/IR/Attributes.h"
#include "marco/Dialect/BaseModelica/IR/VariableAccess.h"
#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/DimensionAccessAdd.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"
#include "marco/Modeling/DimensionAccessDiv.h"
#include "marco/Modeling/DimensionAccessMul.h"
#include "marco/Modeling/DimensionAccessIndices.h"
#include "marco/Modeling/DimensionAccessRange.h"
#include "marco/Modeling/DimensionAccessSub.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include <optional>

namespace mlir::bmodelica
{
  using DimensionAccess = ::marco::modeling::DimensionAccess;
  using DimensionAccessConstant = ::marco::modeling::DimensionAccessConstant;
  using DimensionAccessDimension = ::marco::modeling::DimensionAccessDimension;
  using DimensionAccessAdd = ::marco::modeling::DimensionAccessAdd;
  using DimensionAccessSub = ::marco::modeling::DimensionAccessSub;
  using DimensionAccessMul = ::marco::modeling::DimensionAccessMul;
  using DimensionAccessDiv = ::marco::modeling::DimensionAccessDiv;
  using DimensionAccessRange = ::marco::modeling::DimensionAccessRange;
  using DimensionAccessIndices = ::marco::modeling::DimensionAccessIndices;

  class AdditionalInductions
  {
    public:
      using Dependencies = llvm::DenseSet<uint64_t>;

      using IterationSpaceDependencies =
          llvm::DenseMap<uint64_t, Dependencies>;

      uint64_t addIterationSpace(IndexSet iterationSpace);

      [[nodiscard]] uint64_t getNumOfIterationSpaces() const;

      void addInductionVariable(
          mlir::Value inductionValue,
          uint64_t iterationSpace,
          uint64_t dimension);

      [[nodiscard]] bool hasInductionVariable(mlir::Value induction) const;

      [[nodiscard]] const IndexSet& getInductionSpace(
          mlir::Value induction) const;

      [[nodiscard]] uint64_t getInductionDimension(mlir::Value induction) const;

      [[nodiscard]] const Dependencies&
      getInductionDependencies(mlir::Value induction) const;

      void addDimensionDependency(
          uint64_t iterationSpace, uint64_t dimension, uint64_t dependency);

    private:
      // The list of iteration spaces.
      llvm::SmallVector<IndexSet> iterationSpaces;

      // Map an SSA to the dimension of an iteration space.
      // The first element of the pair represents the iteration space ID, while
      // the second one represent the linked dimension.
      llvm::DenseMap<mlir::Value, std::pair<uint64_t, uint64_t>> inductions;

      // For each iteration space, keep track of the dependencies among its
      // dimensions.
      llvm::DenseMap<uint64_t, IterationSpaceDependencies> dependencies;
  };

  namespace ad::forward
  {
    class State
    {
      public:
        State();

        State(const State& other) = delete;

        State(State&& other);

        ~State();

        State& operator=(const State& other) = delete;

        State& operator=(State&& other);

        mlir::SymbolTableCollection& getSymbolTableCollection();

        void mapDerivative(mlir::Value original, mlir::Value mapped);

        void mapDerivatives(mlir::ValueRange original, mlir::ValueRange mapped);

        [[nodiscard]] bool hasDerivative(mlir::Value original) const;

        [[nodiscard]] std::optional<mlir::Value> getDerivative(
            mlir::Value original) const;

        void mapGenericOpDerivative(
            mlir::Operation* original, mlir::Operation* mapped);

        [[nodiscard]] bool hasGenericOpDerivative(mlir::Operation* original) const;

        [[nodiscard]] std::optional<mlir::Operation*> getGenericOpDerivative(
            mlir::Operation* original) const;

      private:
        mlir::SymbolTableCollection symbolTableCollection;
        mlir::IRMapping valueMapping;
        llvm::DenseMap<mlir::Operation*, mlir::Operation*> generalOpMapping;
    };
  }
}

#include "marco/Dialect/BaseModelica/IR/BaseModelicaOpInterfaces.h.inc"

#endif // MARCO_DIALECT_BASEMODELICA_IR_OPINTERFACES_H
