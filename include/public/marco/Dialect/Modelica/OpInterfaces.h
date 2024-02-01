#ifndef MARCO_DIALECTS_MODELICA_OPINTERFACES_H
#define MARCO_DIALECTS_MODELICA_OPINTERFACES_H

#include "marco/Dialect/Modelica/VariableAccess.h"
#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/DimensionAccessAdd.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"
#include "marco/Modeling/DimensionAccessDiv.h"
#include "marco/Modeling/DimensionAccessMul.h"
#include "marco/Modeling/DimensionAccessIndices.h"
#include "marco/Modeling/DimensionAccessRange.h"
#include "marco/Modeling/DimensionAccessSub.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_FWD_DEFINES
#include "marco/Dialect/Modelica/Modelica.h.inc"

namespace mlir::modelica
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
}

#include "marco/Dialect/Modelica/ModelicaOpInterfaces.h.inc"

#endif // MARCO_DIALECTS_MODELICA_OPINTERFACES_H
