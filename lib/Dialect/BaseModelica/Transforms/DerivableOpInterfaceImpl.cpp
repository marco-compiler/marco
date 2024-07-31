#include "marco/Dialect/BaseModelica/Transforms/DerivableOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation/ForwardAD.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::ad::forward;

static mlir::FailureOr<mlir::Type> getDerivedType(mlir::Type type)
{
  auto derivableType = mlir::dyn_cast<DerivableTypeInterface>(type);

  if (!derivableType) {
    return mlir::failure();
  }

  return derivableType.derive();
}

// The function is a wrapper to the State class method, but also emits an error
// in case of failure.
static std::optional<mlir::Value> getDerivative(
    State& state, mlir::Value original)
{
  auto result = state.getDerivative(original);

  if (!result) {
    mlir::emitError(original.getLoc()) << "Can't get derivative";
  }

  return result;
}

// The function is a wrapper to the State class method, but also emits an error
// in case of failure.
static std::optional<mlir::Operation*> getGenericOpDerivative(
    State& state, mlir::Operation* original)
{
  auto result = state.getGenericOpDerivative(original);

  if (!result) {
    mlir::emitError(original->getLoc()) << "Can't get derivative";
  }

  return result;
}

/*
// The function is a wrapper to the State class method, but also emits an error
// in case of failure.
static std::optional<mlir::StringAttr> getDerivative(
    State& state, mlir::StringAttr original, mlir::Location loc)
{
  auto result = state.getDerivative(original);

  if (!result) {
    mlir::emitError(loc) << "Can't get derivative of '" << original.getValue()
                         << "'";
  }

  return result;
}

// The function is a wrapper to the State class method, but also emits an error
// in case of failure.
static std::optional<mlir::SymbolRefAttr> getDerivative(
    State& state, mlir::SymbolRefAttr original, mlir::Location loc)
{
  auto result = state.getDerivative(original);

  if (!result) {
    mlir::emitError(loc) << "Can't get derivative of " << original;
  }

  return result;
}
 */

static mlir::LogicalResult deriveRegion(
    mlir::Region& region, mlir::OpBuilder& builder, State& state,
    llvm::function_ref<
        mlir::LogicalResult(DerivableOpInterface)> deriveFn)
{
  for (auto& op : llvm::make_early_inc_range(region.getOps())) {
    auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(op);

    if (!derivableOp) {
      continue;
    }

    if (mlir::failed(deriveFn(derivableOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

static mlir::LogicalResult createValueTimeDerivative(
    mlir::OpBuilder& builder, State& state, mlir::Value value)
{
  auto derivableOp = value.getDefiningOp<DerivableOpInterface>();
  
  if (!derivableOp) {
    return mlir::failure();
  }
  
  return derivableOp.createTimeDerivative(builder, state, true);
}

namespace
{
  struct TimeOpInterface
      : public DerivableOpInterface::ExternalModel<TimeOpInterface, TimeOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<TimeOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedOp = builder.create<ConstantOp>(
          op->getLoc(), RealAttr::get(op->getContext(), 0));

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TimeOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedOp = builder.create<ConstantOp>(
          op->getLoc(), RealAttr::get(op->getContext(), 1));

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TensorFromElementsOpInterface
      : public DerivableOpInterface::ExternalModel<
          TensorFromElementsOpInterface, TensorFromElementsOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TensorFromElementsOp>(op);
      
      if (deriveDependencies) {
        for (mlir::Value value : castedOp.getValues()) {
          if (mlir::failed(createValueTimeDerivative(
                  builder, state, value))) {
            return mlir::failure();
          }
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<TensorFromElementsOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      llvm::SmallVector<mlir::Value> derivedValues;

      for (mlir::Value value : castedOp.getValues()) {
        auto derivedValue = getDerivative(state, value);

        if (!derivedValue) {
          return mlir::failure();
        }

        derivedValues.push_back(*derivedValue);
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<TensorFromElementsOp>(
          castedOp.getLoc(), *resultType, derivedValues);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TensorBroadcastOpInterface
      : public DerivableOpInterface::ExternalModel<
          TensorBroadcastOpInterface, TensorBroadcastOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TensorBroadcastOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getValue()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<TensorBroadcastOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedValue = getDerivative(state, castedOp.getValue());

      if (!derivedValue) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<TensorBroadcastOp>(
          castedOp.getLoc(), *resultType, *derivedValue);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TensorViewOpInterface
      : public DerivableOpInterface::ExternalModel<
          TensorViewOpInterface, TensorViewOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TensorViewOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getSource()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<TensorViewOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedSource = getDerivative(state, castedOp.getSource());

      if (!derivedSource) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<TensorViewOp>(
          castedOp.getLoc(), *derivedSource, castedOp.getSubscriptions());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TensorExtractOpInterface
      : public DerivableOpInterface::ExternalModel<
            TensorExtractOpInterface, TensorExtractOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TensorExtractOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getTensor()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<TensorExtractOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedTensor = getDerivative(state, castedOp.getTensor());

      if (!derivedTensor) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<TensorExtractOp>(
          castedOp.getLoc(), *derivedTensor, castedOp.getIndices());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TensorInsertOpInterface
      : public DerivableOpInterface::ExternalModel<
            TensorInsertOpInterface, TensorInsertOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TensorInsertOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getDestination()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getValue()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<TensorInsertOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedDestination =
          getDerivative(state, castedOp.getDestination());

      if (!derivedDestination) {
        return mlir::failure();
      }

      auto derivedValue = getDerivative(state, castedOp.getValue());

      auto derivedOp = builder.create<TensorInsertOp>(
          castedOp.getLoc(), *derivedValue, *derivedDestination,
          castedOp.getIndices());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TensorInsertSliceOpInterface
      : public DerivableOpInterface::ExternalModel<
            TensorInsertSliceOpInterface, TensorInsertSliceOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TensorInsertSliceOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getDestination()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getValue()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<TensorInsertSliceOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedDestination =
          getDerivative(state, castedOp.getDestination());

      if (!derivedDestination) {
        return mlir::failure();
      }

      auto derivedValue = getDerivative(state, castedOp.getValue());

      auto derivedOp = builder.create<TensorInsertSliceOp>(
          castedOp.getLoc(), *derivedValue, *derivedDestination,
          castedOp.getSubscriptions());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct AllocaOpInterface
      : public DerivableOpInterface::ExternalModel<
          AllocaOpInterface, AllocaOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<AllocaOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<AllocaOp>(
          castedOp.getLoc(), *resultType, castedOp.getDynamicSizes());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct AllocOpInterface
      : public DerivableOpInterface::ExternalModel<
          AllocOpInterface, AllocOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<AllocOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<AllocOp>(
          castedOp.getLoc(), *resultType, castedOp.getDynamicSizes());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct ArrayFromElementsOpInterface
      : public DerivableOpInterface::ExternalModel<
            ArrayFromElementsOpInterface, ArrayFromElementsOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<ArrayFromElementsOp>(op);
      
      if (deriveDependencies) {
        for (mlir::Value value : castedOp.getValues()) {
          if (mlir::failed(createValueTimeDerivative(
                  builder, state, value))) {
            return mlir::failure();
          }
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<ArrayFromElementsOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      llvm::SmallVector<mlir::Value> derivedValues;

      for (mlir::Value value : castedOp.getValues()) {
        auto derivedValue = getDerivative(state, value);

        if (!derivedValue) {
          return mlir::failure();
        }

        derivedValues.push_back(*derivedValue);
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<ArrayFromElementsOp>(
          castedOp.getLoc(), *resultType, derivedValues);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct ArrayBroadcastOpInterface
      : public DerivableOpInterface::ExternalModel<
            ArrayBroadcastOpInterface, ArrayBroadcastOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<ArrayBroadcastOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getValue()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<ArrayBroadcastOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedValue = getDerivative(state, castedOp.getValue());

      if (!derivedValue) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<ArrayBroadcastOp>(
          castedOp.getLoc(), *resultType, *derivedValue);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct SubscriptionOpInterface
      : public DerivableOpInterface::ExternalModel<
            SubscriptionOpInterface, SubscriptionOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<SubscriptionOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getSource()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<SubscriptionOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedSource = getDerivative(state, castedOp.getSource());

      auto derivedOp = builder.create<SubscriptionOp>(
          castedOp.getLoc(), *derivedSource, castedOp.getIndices());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct LoadOpInterface
      : public DerivableOpInterface::ExternalModel<LoadOpInterface, LoadOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<LoadOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getArray()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<LoadOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedArray = getDerivative(state, castedOp.getArray());

      auto derivedOp = builder.create<LoadOp>(
          castedOp.getLoc(), *derivedArray, castedOp.getIndices());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct StoreOpInterface
      : public DerivableOpInterface::ExternalModel<StoreOpInterface, StoreOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<StoreOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getArray()))) {
          return mlir::failure();
        }
        
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getValue()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<StoreOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedArray = getDerivative(state, castedOp.getArray());

      if (!derivedArray) {
        return mlir::failure();
      }

      auto derivedValue = getDerivative(state, castedOp.getValue());

      if (!derivedValue) {
        return mlir::failure();
      }

      builder.create<StoreOp>(
          castedOp.getLoc(), *derivedValue, *derivedArray,
          castedOp.getIndices());

      return mlir::success();
    }
  };

  struct VariableGetOpInterface
      : public DerivableOpInterface::ExternalModel<
          VariableGetOpInterface, VariableGetOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<VariableGetOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Operation* parentClass =
          castedOp->getParentWithTrait<ClassInterface::Trait>();

      auto variableOp =
          state.getSymbolTableCollection().lookupSymbolIn<VariableOp>(
              parentClass, castedOp.getVariableAttr());

      assert(variableOp && "Variable lookup failed");
      auto derivativeVariableOp = getGenericOpDerivative(state, variableOp);

      if (!derivativeVariableOp) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<VariableGetOp>(
          castedOp.getLoc(),
          mlir::cast<VariableOp>(*derivativeVariableOp));

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct VariableSetOpInterface
      : public DerivableOpInterface::ExternalModel<
          VariableSetOpInterface, VariableSetOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<VariableSetOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getValue()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<VariableSetOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Operation* parentClass =
          castedOp->getParentWithTrait<ClassInterface::Trait>();

      auto variableOp =
          state.getSymbolTableCollection().lookupSymbolIn<VariableOp>(
              parentClass, castedOp.getVariableAttr());

      assert(variableOp && "Variable lookup failed");
      auto derivativeVariableOp = getGenericOpDerivative(state, variableOp);

      if (!derivativeVariableOp) {
        return mlir::failure();
      }

      auto derivedValue = getDerivative(state, castedOp.getValue());

      if (!derivedValue) {
        return mlir::failure();
      }

      builder.create<VariableSetOp>(
          castedOp.getLoc(),
          mlir::cast<VariableOp>(*derivativeVariableOp),
          castedOp.getIndices(), *derivedValue);

      return mlir::success();
    }
  };

  struct GlobalVariableGetOpInterface
      : public DerivableOpInterface::ExternalModel<
            GlobalVariableGetOpInterface, GlobalVariableGetOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<GlobalVariableGetOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto arrayType = castedOp.getResult().getType().cast<ArrayType>();
      auto elementType = arrayType.getElementType();

      auto materializableType = mlir::dyn_cast<ConstantMaterializableTypeInterface>(elementType);

      if (!materializableType) {
        return mlir::failure();
      }

      mlir::Value zero = materializableType.materializeIntConstant(
          builder, castedOp.getLoc(), 0);

      auto derivedOp = builder.create<ArrayBroadcastOp>(
          castedOp.getLoc(),
          arrayType, zero);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct ConstantOpInterface
      : public DerivableOpInterface::ExternalModel<
          ConstantOpInterface, ConstantOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[const] = 0
      auto castedOp = mlir::cast<ConstantOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Type resultType = castedOp.getResult().getType();

      if (resultType.isa<RangeType>()) {
        state.mapDerivative(castedOp, castedOp);
        return mlir::success();
      }

      mlir::Type baseType = resultType;

      if (auto resultArrayType = resultType.dyn_cast<ArrayType>()) {
        baseType = resultArrayType.getElementType();
      }

      if (auto resultTensorType = resultType.dyn_cast<mlir::TensorType>()) {
        baseType = resultTensorType.getElementType();
      }

      auto derivableBaseType =
          mlir::dyn_cast<DerivableTypeInterface>(baseType);

      if (!derivableBaseType) {
        return mlir::failure();
      }

      auto derivedBaseType = derivableBaseType.derive();

      if (!derivableBaseType) {
        return mlir::failure();
      }

      auto constantMaterializableDerivedBaseType =
          mlir::dyn_cast<ConstantMaterializableTypeInterface>(
              *derivedBaseType);

      if (!constantMaterializableDerivedBaseType) {
        return mlir::failure();
      }

      mlir::Value result =
          constantMaterializableDerivedBaseType.materializeIntConstant(
              builder, castedOp.getResult().getLoc(), 0);

      if (auto resultArrayType = resultType.dyn_cast<ArrayType>()) {
        result = builder.create<ArrayBroadcastOp>(
            result.getLoc(), resultArrayType.clone(*derivedBaseType), result);
      }

      if (auto resultTensorType = resultType.dyn_cast<mlir::TensorType>()) {
        result = builder.create<TensorBroadcastOp>(
            result.getLoc(), resultTensorType.clone(*derivedBaseType), result);
      }

      state.mapDerivative(castedOp, result);
      return mlir::success();
    }
  };

  struct NegateOpInterface
      : public DerivableOpInterface::ExternalModel<
          NegateOpInterface, NegateOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<NegateOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<NegateOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<NegateOp>(
          castedOp.getLoc(), *resultType, *derivedOperand);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct AddOpInterface
      : public DerivableOpInterface::ExternalModel<AddOpInterface, AddOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<AddOp>(op);
      
      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }
        
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }
      
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<AddOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<AddOp>(
          loc, *resultType, *derivedLhs, *derivedRhs);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct AddEWOpInterface
      : public DerivableOpInterface::ExternalModel<AddEWOpInterface, AddEWOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<AddEWOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<AddEWOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<AddEWOp>(
          loc, *resultType, *derivedLhs, *derivedRhs);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct SubOpInterface
      : public DerivableOpInterface::ExternalModel<SubOpInterface, SubOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<SubOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<SubOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<SubOp>(
          loc, *resultType, *derivedLhs, *derivedRhs);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct SubEWOpInterface
      : public DerivableOpInterface::ExternalModel<SubEWOpInterface, SubEWOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<SubEWOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<SubEWOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<SubEWOp>(
          loc, *resultType, *derivedLhs, *derivedRhs);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct MulOpInterface
      : public DerivableOpInterface::ExternalModel<MulOpInterface, MulOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<MulOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<MulOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value firstMul = builder.create<MulOp>(
          loc, *resultType, *derivedLhs, castedOp.getRhs());

      mlir::Value secondMul = builder.create<MulOp>(
          loc, *resultType, castedOp.getLhs(), *derivedRhs);

      auto derivedOp = builder.create<AddOp>(
          loc, *resultType, firstMul, secondMul);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct MulEWOpInterface
      : public DerivableOpInterface::ExternalModel<MulEWOpInterface, MulEWOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<MulEWOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<MulEWOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value firstMul = builder.create<MulEWOp>(
          loc, *resultType, *derivedLhs, castedOp.getRhs());

      mlir::Value secondMul = builder.create<MulEWOp>(
          loc, *resultType, castedOp.getLhs(), *derivedRhs);

      auto derivedOp = builder.create<AddEWOp>(
          loc, *resultType, firstMul, secondMul);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct DivOpInterface
      : public DerivableOpInterface::ExternalModel<DivOpInterface, DivOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<DivOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<DivOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto denominatorType = getDerivedType(castedOp.getRhs().getType());

      if (mlir::failed(denominatorType)) {
        return mlir::failure();
      }

      mlir::Value firstMul = builder.create<MulOp>(
          loc, *resultType, *derivedLhs, castedOp.getRhs());

      mlir::Value secondMul = builder.create<MulOp>(
          loc, *resultType, castedOp.getLhs(), *derivedRhs);

      mlir::Value numerator = builder.create<SubOp>(
          loc, *resultType, firstMul, secondMul);

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      mlir::Value denominator = builder.create<PowOp>(
          loc, *denominatorType, castedOp.getRhs(), two);

      auto derivedOp = builder.create<DivOp>(
          loc, *resultType, numerator, denominator);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct DivEWOpInterface
      : public DerivableOpInterface::ExternalModel<DivEWOpInterface, DivEWOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<DivEWOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getLhs()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getRhs()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<DivEWOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedLhs = getDerivative(state, castedOp.getLhs());

      if (!derivedLhs) {
        return mlir::failure();
      }

      auto derivedRhs = getDerivative(state, castedOp.getRhs());

      if (!derivedRhs) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto denominatorType = getDerivedType(castedOp.getRhs().getType());

      if (mlir::failed(denominatorType)) {
        return mlir::failure();
      }

      mlir::Value firstMul = builder.create<MulEWOp>(
          loc, *resultType, *derivedLhs, castedOp.getRhs());

      mlir::Value secondMul = builder.create<MulEWOp>(
          loc, *resultType, castedOp.getLhs(), *derivedRhs);

      mlir::Value numerator = builder.create<SubEWOp>(
          loc, *resultType, firstMul, secondMul);

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      mlir::Value denominator = builder.create<PowEWOp>(
          loc, *denominatorType, two);

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, numerator, denominator);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct PowOpInterface
      : public DerivableOpInterface::ExternalModel<PowOpInterface, PowOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<PowOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getBase()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getExponent()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[x ^ y] = (x ^ (y - 1)) * (y * x' + x * ln(x) * y')
      auto castedOp = mlir::cast<PowOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedBase = getDerivative(state, castedOp.getBase());

      if (!derivedBase) {
        return mlir::failure();
      }

      auto derivedExponent = getDerivative(state, castedOp.getExponent());

      if (!derivedExponent) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      if (auto constantExponent =
              castedOp.getExponent().getDefiningOp<ConstantOp>()) {
        mlir::Value one = builder.create<ConstantOp>(
            loc, RealAttr::get(builder.getContext(), 1));

        mlir::Value exponent = builder.createOrFold<SubOp>(
            loc,
            castedOp.getExponent().getType(),
            castedOp.getExponent(),
            one);

        mlir::Value pow = builder.create<PowOp>(
            loc, *resultType, castedOp.getBase(), exponent);

        auto derivedOp = builder.create<MulOp>(
            loc, *resultType, pow, *derivedBase);

        state.mapDerivative(castedOp, derivedOp);
        return mlir::success();
      }

      mlir::Value one = builder.create<ConstantOp>(
          loc, RealAttr::get(builder.getContext(), 1));

      mlir::Value exponent = builder.create<SubOp>(
          loc,
          castedOp.getExponent().getType(),
          castedOp.getExponent(),
          one);

      mlir::Value pow = builder.create<PowOp>(
          loc, *resultType, castedOp.getBase(),exponent);

      mlir::Value firstMul = builder.create<MulOp>(
          loc, *resultType, castedOp.getExponent(), *derivedBase);

      mlir::Value ln = builder.create<LogOp>(
          loc, *resultType, castedOp.getBase());

      mlir::Value secondMul = builder.create<MulOp>(
          loc, *resultType, castedOp.getBase(), ln);

      mlir::Value thirdMul = builder.create<MulOp>(
          loc, *resultType, secondMul, *derivedExponent);

      mlir::Value sum = builder.create<AddOp>(
          loc, *resultType, firstMul, thirdMul);

      auto derivedOp = builder.create<MulOp>(
          loc, *resultType, pow, sum);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct PowEWOpInterface
      : public DerivableOpInterface::ExternalModel<PowEWOpInterface, PowEWOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<PowEWOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getBase()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getExponent()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)
      auto castedOp = mlir::cast<PowEWOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedBase = getDerivative(state, castedOp.getBase());

      if (!derivedBase) {
        return mlir::failure();
      }

      auto derivedExponent = getDerivative(state, castedOp.getExponent());

      if (!derivedExponent) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value pow = builder.create<PowEWOp>(
          loc, *resultType, castedOp.getBase(), castedOp.getExponent());

      mlir::Value ln = builder.create<LogOp>(
          loc, *resultType, castedOp.getBase());

      mlir::Value firstOperand = builder.create<MulEWOp>(
          loc, *resultType, *derivedExponent, ln);

      mlir::Value numerator = builder.create<MulEWOp>(
          loc, *resultType, castedOp.getExponent(), *derivedBase);

      mlir::Value secondOperand = builder.create<DivEWOp>(
          loc, *resultType, numerator, castedOp.getBase());

      mlir::Value sum = builder.create<AddEWOp>(
          loc, *resultType, firstOperand, secondOperand);

      auto derivedOp = builder.create<MulEWOp>(
          loc, *resultType, pow, sum);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct ReductionOpInterface
      : public DerivableOpInterface::ExternalModel<
            ReductionOpInterface, ReductionOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createPartialDerivative(builder, state);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createTimeDerivative(
            builder, state, deriveDependencies);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        llvm::function_ref<
            mlir::LogicalResult(DerivableOpInterface)> deriveFn) const
    {
      auto castedOp = mlir::cast<ReductionOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto action = castedOp.getAction();

      if (action != "add") {
        castedOp.emitOpError() << "Can't derive '" << action << "' reduction";
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<ReductionOp>(
          castedOp.getLoc(), *resultType, action,
          castedOp.getIterables());

      builder.setInsertionPointToStart(
          derivedOp.createExpressionBlock(builder));

      mlir::IRMapping mapping;

      // Map the induction variables.
      for (auto [oldInductionVar, newInductionVar] :
           llvm::zip(castedOp.getInductions(), derivedOp.getInductions())) {
        mapping.map(oldInductionVar, newInductionVar);
      }

      // Create the derivative for the induction variables.
      for (mlir::Value inductionVar : derivedOp.getInductions()) {
        mlir::Value derivedInductionVar = builder.create<ConstantOp>(
            inductionVar.getLoc(), builder.getIndexAttr(0));

        state.mapDerivative(inductionVar, derivedInductionVar);
      }

      // Clone and derive the body.
      llvm::SmallVector<mlir::Region*> regionsToBeDerived;

      for (auto& nestedOp : castedOp.getOps()) {
        if (auto yieldOp = mlir::dyn_cast<YieldOp>(nestedOp)) {
          llvm::SmallVector<mlir::Value> yieldedValues;

          for (mlir::Value yieldedValue : yieldOp.getValues()) {
            mlir::Value mappedYieldedValue = mapping.lookup(yieldedValue);
            auto derivedYieldedValue = getDerivative(state, mappedYieldedValue);

            if (!derivedYieldedValue) {
              return mlir::failure();
            }

            yieldedValues.push_back(*derivedYieldedValue);
          }

          builder.create<YieldOp>(yieldOp.getLoc(), yieldedValues);
        } else {
          auto clonedOp = builder.clone(nestedOp, mapping);
          llvm::SmallVector<mlir::Value> derivedValues;

          auto derivableClonedNestedOp =
              mlir::dyn_cast<DerivableOpInterface>(clonedOp);

          if (!derivableClonedNestedOp) {
            continue;
          }

          if (mlir::failed(deriveFn(derivableClonedNestedOp))) {
            return mlir::failure();
          }
        }
      }

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct AcosOpInterface
      : public DerivableOpInterface::ExternalModel<AcosOpInterface, AcosOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<AcosOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[acos(x)] = -x' / sqrt(1 - x^2)
      auto castedOp = mlir::cast<AcosOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value one = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 1));

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      mlir::Value argSquared = builder.create<PowEWOp>(
          loc, *resultType, castedOp.getOperand(), two);

      mlir::Value sub = builder.create<SubEWOp>(
          loc, *resultType, one, argSquared);

      mlir::Value denominator = builder.create<SqrtOp>(
          loc, *resultType, sub);

      mlir::Value div = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, denominator);

      auto derivedOp = builder.create<NegateOp>(loc, *resultType, div);
      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct AsinOpInterface
      : public DerivableOpInterface::ExternalModel<AsinOpInterface, AsinOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<AsinOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[arcsin(x)] = x' / sqrt(1 - x^2)
      auto castedOp = mlir::cast<AsinOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value one = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 1));

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      mlir::Value argSquared = builder.create<PowEWOp>(
          loc, *resultType, castedOp.getOperand(), two);

      mlir::Value sub = builder.create<SubEWOp>(
          loc, *resultType, one, argSquared);

      mlir::Value denominator = builder.create<SqrtOp>(
          loc, *resultType, sub);

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, denominator);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct AtanOpInterface
      : public DerivableOpInterface::ExternalModel<AtanOpInterface, AtanOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<AtanOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[atan(x)] = x' / (1 + x^2)
      auto castedOp = mlir::cast<AtanOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value one = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 1));

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      mlir::Value argSquared = builder.create<PowEWOp>(
          loc, *resultType, castedOp.getOperand(), two);

      mlir::Value denominator = builder.create<AddEWOp>(
          loc, *resultType, one, argSquared);

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, denominator);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct Atan2OpInterface
      : public DerivableOpInterface::ExternalModel<Atan2OpInterface, Atan2Op>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<Atan2Op>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getY()))) {
          return mlir::failure();
        }

        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getX()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[atan2(y, x)] = (y' * x - y * x') / (y^2 + x^2)
      auto castedOp = mlir::cast<Atan2Op>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedY = getDerivative(state, castedOp.getY());

      if (!derivedY) {
        return mlir::failure();
      }

      auto derivedX = getDerivative(state, castedOp.getX());

      if (!derivedX) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value firstMul = builder.create<MulEWOp>(
          loc, *resultType, *derivedY, castedOp.getX());

      mlir::Value secondMul = builder.create<MulEWOp>(
          loc, *resultType, castedOp.getY(), *derivedX);

      mlir::Value numerator = builder.create<SubEWOp>(
          loc, *resultType, firstMul, secondMul);

      mlir::Value firstSquared = builder.create<MulEWOp>(
          loc, *resultType, castedOp.getY(), castedOp.getY());

      mlir::Value secondSquared = builder.create<MulEWOp>(
          loc, *resultType, castedOp.getX(), castedOp.getX());

      mlir::Value denominator = builder.create<AddEWOp>(
          loc, *resultType, firstSquared, secondSquared);

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, numerator, denominator);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct CosOpInterface
      : public DerivableOpInterface::ExternalModel<CosOpInterface, CosOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<CosOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[cos(x)] = -x' * sin(x)
      auto castedOp = mlir::cast<CosOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value sin = builder.create<SinOp>(
          loc, *resultType, castedOp.getOperand());

      mlir::Value negatedSin = builder.create<NegateOp>(
          loc, *resultType, sin);

      auto derivedOp = builder.create<MulEWOp>(
          loc, *resultType, negatedSin, *derivedOperand);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct CoshOpInterface
      : public DerivableOpInterface::ExternalModel<CoshOpInterface, CoshOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<CoshOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[cosh(x)] = x' * sinh(x)
      auto castedOp = mlir::cast<CoshOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value sinh = builder.create<SinhOp>(
          loc, *resultType, castedOp.getOperand());

      auto derivedOp = builder.create<MulEWOp>(
          loc, *resultType, sinh, *derivedOperand);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct ExpOpInterface
      : public DerivableOpInterface::ExternalModel<ExpOpInterface, ExpOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<ExpOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getExponent()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[e^x] = x' * e^x
      auto castedOp = mlir::cast<ExpOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedExponent = getDerivative(state, castedOp.getExponent());

      if (!derivedExponent) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value pow = builder.create<ExpOp>(
          loc, *resultType, castedOp.getExponent());

      auto derivedOp = builder.create<MulEWOp>(
          loc, *resultType, pow, *derivedExponent);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct FillOpInterface
      : public DerivableOpInterface::ExternalModel<FillOpInterface, FillOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<FillOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<FillOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto derivedValue = getDerivative(state, castedOp.getValue());

      if (!derivedValue) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<TensorBroadcastOp>(
          castedOp.getLoc(), *resultType, *derivedValue);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct LogOpInterface
      : public DerivableOpInterface::ExternalModel<LogOpInterface, LogOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<LogOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[ln(x)] = x' / x
      auto castedOp = mlir::cast<LogOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, castedOp.getOperand());

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct Log10OpInterface
      : public DerivableOpInterface::ExternalModel<Log10OpInterface, Log10Op>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<Log10Op>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[log10(x)] = x' / (x * ln(10))
      auto castedOp = mlir::cast<Log10Op>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value ten = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 10));

      mlir::Value log = builder.create<LogOp>(
          loc, RealType::get(castedOp.getContext()), ten);

      mlir::Value mul = builder.create<MulEWOp>(
          loc, *resultType, castedOp.getOperand(), log);

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, mul);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct SinOpInterface
      : public DerivableOpInterface::ExternalModel<SinOpInterface, SinOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<SinOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[sin(x)] = x' * cos(x)
      auto castedOp = mlir::cast<SinOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value cos = builder.create<CosOp>(
          loc, *resultType, castedOp.getOperand());

      auto derivedOp = builder.create<MulEWOp>(
          loc, *resultType, cos, *derivedOperand);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct SinhOpInterface
      : public DerivableOpInterface::ExternalModel<SinhOpInterface, SinhOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<SinhOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[sinh(x)] = x' * cosh(x)
      auto castedOp = mlir::cast<SinhOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value cosh = builder.create<CoshOp>(
          loc, *resultType, castedOp.getOperand());

      auto derivedOp = builder.create<MulEWOp>(
          loc, *resultType, cosh, *derivedOperand);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct SqrtOpInterface
      : public DerivableOpInterface::ExternalModel<SqrtOpInterface, SqrtOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<SqrtOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[sqrt(x)] = x' / sqrt(x) / 2
      auto castedOp = mlir::cast<SqrtOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value sqrt = builder.create<SqrtOp>(
          loc, *resultType, castedOp.getOperand());

      mlir::Value numerator = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, sqrt);

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, numerator, two);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TanOpInterface
      : public DerivableOpInterface::ExternalModel<TanOpInterface, TanOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TanOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[tan(x)] = x' / (cos(x))^2
      auto castedOp = mlir::cast<TanOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value cos = builder.create<CosOp>(
          loc, *resultType, castedOp.getOperand());

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      mlir::Value denominator = builder.create<PowEWOp>(
          loc, *resultType, cos, two);

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, denominator);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct TanhOpInterface
      : public DerivableOpInterface::ExternalModel<TanhOpInterface, TanhOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<TanhOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      // D[tanh(x)] = x' / (cosh(x))^2
      auto castedOp = mlir::cast<TanhOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      mlir::Location loc = castedOp.getLoc();

      auto derivedOperand = getDerivative(state, castedOp.getOperand());

      if (!derivedOperand) {
        return mlir::failure();
      }

      auto resultType = getDerivedType(castedOp.getResult().getType());

      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      mlir::Value cosh = builder.create<CoshOp>(
          loc, *resultType, castedOp.getOperand());

      mlir::Value two = builder.create<ConstantOp>(
          loc, RealAttr::get(castedOp.getContext(), 2));

      mlir::Value pow = builder.create<PowEWOp>(
          loc, *resultType, cosh, two);

      auto derivedOp = builder.create<DivEWOp>(
          loc, *resultType, *derivedOperand, pow);

      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct IfOpInterface
      : public DerivableOpInterface::ExternalModel<IfOpInterface, IfOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createPartialDerivative(builder, state);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createTimeDerivative(
            builder, state, deriveDependencies);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        llvm::function_ref<
            mlir::LogicalResult(DerivableOpInterface)> deriveFn) const
    {
      auto castedOp = mlir::cast<IfOp>(op);

      if (mlir::failed(deriveRegion(
              castedOp.getThenRegion(), builder, state, deriveFn))) {
        return mlir::failure();
      }

      if (mlir::failed(deriveRegion(
              castedOp.getElseRegion(), builder, state, deriveFn))) {
        return mlir::failure();
      }

      return mlir::success();
    }
  };

  struct ForOpInterface
      : public DerivableOpInterface::ExternalModel<ForOpInterface, ForOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createPartialDerivative(builder, state);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createTimeDerivative(
            builder, state, deriveDependencies);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        llvm::function_ref<
            mlir::LogicalResult(DerivableOpInterface)> deriveFn) const
    {
      auto castedOp = mlir::cast<ForOp>(op);
      return deriveRegion(castedOp.getBodyRegion(), builder, state, deriveFn);
    }
  };

  struct WhileOpInterface
      : public DerivableOpInterface::ExternalModel<WhileOpInterface, WhileOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createPartialDerivative(builder, state);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createTimeDerivative(
            builder, state, deriveDependencies);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        llvm::function_ref<
            mlir::LogicalResult(DerivableOpInterface)> deriveFn) const
    {
      auto castedOp = mlir::cast<WhileOp>(op);
      return deriveRegion(castedOp.getBodyRegion(), builder, state, deriveFn);
    }
  };

  struct CastOpInterface
      : public DerivableOpInterface::ExternalModel<CastOpInterface, CastOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<CastOp>(op);

      if (deriveDependencies) {
        if (mlir::failed(createValueTimeDerivative(
                builder, state, castedOp.getOperand()))) {
          return mlir::failure();
        }
      }

      return createDerivative(op, builder, state);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<CastOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);
      
      // TODO not sure about the correctness of this cast.
      auto resultType = getDerivedType(castedOp.getResult().getType());
      
      if (mlir::failed(resultType)) {
        return mlir::failure();
      }

      auto derivedValue = getDerivative(state, castedOp.getValue());

      if (!derivedValue) {
        return mlir::failure();
      }

      auto derivedOp = builder.create<CastOp>(
          castedOp.getLoc(), *resultType, *derivedValue);
      
      state.mapDerivative(castedOp, derivedOp);
      return mlir::success();
    }
  };

  struct CallOpInterface
      : public DerivableOpInterface::ExternalModel<CallOpInterface, CallOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto castedOp = mlir::cast<CallOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      auto moduleOp = castedOp->getParentOfType<mlir::ModuleOp>();

      auto calleeOp = resolveSymbol(
          moduleOp, state.getSymbolTableCollection(), castedOp.getCallee());

      if (!calleeOp) {
        castedOp.emitOpError() << "Can't get callee '" << castedOp.getCallee()
                               << "' to be derived";

        return mlir::failure();
      }

      auto functionOp = mlir::dyn_cast<FunctionOp>(calleeOp);

      if (!functionOp) {
        castedOp.emitOpError() << "Can't derive function call, "
                               << castedOp.getCallee()
                               << " is not a compatible function type";

        return mlir::failure();
      }

      // Derive the function.
      auto derivedFunctionOp =
          createFunctionPartialDerivative(builder, state, functionOp);

      if (!derivedFunctionOp) {
        castedOp.emitOpError() << "Can't create callee derivative";
        return mlir::failure();
      }

      // Create the call to the derived function.
      llvm::SmallVector<mlir::Value, 3> args;

      for (auto arg : castedOp.getArgs()) {
        args.push_back(arg);
      }

      for (auto arg : castedOp.getArgs()) {
        auto derivedArg = getDerivative(state, arg);

        if (!derivedArg) {
          return mlir::failure();
        }

        args.push_back(*derivedArg);
      }

      auto derivedOp = builder.create<CallOp>(
          castedOp.getLoc(), *derivedFunctionOp, args);

      state.mapDerivatives(castedOp.getResults(), derivedOp.getResults());
      return mlir::success();
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto castedOp = mlir::cast<CallOp>(op);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(castedOp);

      llvm_unreachable("CallOp time derivative is not implemented yet");
      return mlir::failure();
    }

    std::optional<FunctionOp> createFunctionPartialDerivative(
        mlir::OpBuilder& builder,
        State& state,
        FunctionOp functionOp) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(functionOp);

      // Create the derived function.
      std::string derivedFunctionName =
          getPartialDerFunctionName(functionOp.getSymName());

      return ::ad::forward::createFunctionPartialDerivative(
          builder, state, functionOp, derivedFunctionName);
    }

    std::optional<FunctionOp> createFunctionTimeDerivative(
        mlir::OpBuilder& builder,
        State& state,
        FunctionOp functionOp) const
    {
      uint64_t order = functionOp.getTimeDerivativeOrder();

      std::string derivedFunctionName =
          getTimeDerFunctionName(functionOp.getSymName());

      return ad::forward::createFunctionTimeDerivative(
          builder, state, functionOp, order, derivedFunctionName, order + 1);
    }
  };

  struct AlgorithmOpInterface
      : public DerivableOpInterface::ExternalModel<
            AlgorithmOpInterface, AlgorithmOp>
  {
    mlir::LogicalResult createPartialDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createPartialDerivative(builder, state);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createTimeDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        bool deriveDependencies) const
    {
      auto deriveFn = [&](DerivableOpInterface nestedOp) {
        return nestedOp.createTimeDerivative(
            builder, state, deriveDependencies);
      };

      return createDerivative(op, builder, state, deriveFn);
    }

    mlir::LogicalResult createDerivative(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        State& state,
        llvm::function_ref<
            mlir::LogicalResult(DerivableOpInterface)> deriveFn) const
    {
      auto castedOp = mlir::cast<AlgorithmOp>(op);
      return deriveRegion(castedOp.getBodyRegion(), builder, state, deriveFn);
    }
  };
}

namespace mlir::bmodelica
{
  void registerDerivableOpInterfaceExternalModels(
      mlir::DialectRegistry& registry)
  {
    registry.addExtension(+[](mlir::MLIRContext* context,
                              BaseModelicaDialect* dialect) {
      TimeOp::attachInterface<::TimeOpInterface>(*context);

      // Tensor operations.
      TensorFromElementsOp::attachInterface<
          ::TensorFromElementsOpInterface>(*context);
      
      TensorBroadcastOp::attachInterface<::TensorBroadcastOpInterface>(*context);
      TensorViewOp::attachInterface<::TensorViewOpInterface>(*context);
      TensorExtractOp::attachInterface<::TensorExtractOpInterface>(*context);
      TensorInsertOp::attachInterface<::TensorInsertOpInterface>(*context);
      TensorInsertSliceOp::attachInterface<::TensorInsertSliceOpInterface>(*context);

      // Array operations.
      AllocaOp::attachInterface<::AllocaOpInterface>(*context);
      AllocOp::attachInterface<::AllocOpInterface>(*context);

      ArrayFromElementsOp::attachInterface<
          ::ArrayFromElementsOpInterface>(*context);

      ArrayBroadcastOp::attachInterface<::ArrayBroadcastOpInterface>(*context);
      SubscriptionOp::attachInterface<::SubscriptionOpInterface>(*context);
      LoadOp::attachInterface<::LoadOpInterface>(*context);
      StoreOp::attachInterface<::StoreOpInterface>(*context);

      // Variable operations.
      VariableGetOp::attachInterface<::VariableGetOpInterface>(*context);
      VariableSetOp::attachInterface<::VariableSetOpInterface>(*context);

      // Global variable operations.
      GlobalVariableGetOp::attachInterface<::GlobalVariableGetOpInterface>(*context);

      // Math operations.
      ConstantOp::attachInterface<::ConstantOpInterface>(*context);
      NegateOp::attachInterface<::NegateOpInterface>(*context);
      AddOp::attachInterface<::AddOpInterface>(*context);
      AddEWOp::attachInterface<::AddEWOpInterface>(*context);
      SubOp::attachInterface<::SubOpInterface>(*context);
      SubEWOp::attachInterface<::SubEWOpInterface>(*context);
      MulOp::attachInterface<::MulOpInterface>(*context);
      MulEWOp::attachInterface<::MulEWOpInterface>(*context);
      DivOp::attachInterface<::DivOpInterface>(*context);
      DivEWOp::attachInterface<::DivEWOpInterface>(*context);
      PowOp::attachInterface<::PowOpInterface>(*context);
      PowEWOp::attachInterface<::PowEWOpInterface>(*context);

      ReductionOp::attachInterface<::ReductionOpInterface>(*context);

      // Built-in operations.
      AcosOp::attachInterface<::AcosOpInterface>(*context);
      AsinOp::attachInterface<::AsinOpInterface>(*context);
      AtanOp::attachInterface<::AtanOpInterface>(*context);
      Atan2Op::attachInterface<::Atan2OpInterface>(*context);
      CosOp::attachInterface<::CosOpInterface>(*context);
      CoshOp::attachInterface<::CoshOpInterface>(*context);
      ExpOp::attachInterface<::ExpOpInterface>(*context);
      FillOp::attachInterface<::FillOpInterface>(*context);
      LogOp::attachInterface<::LogOpInterface>(*context);
      Log10Op::attachInterface<::Log10OpInterface>(*context);
      SinOp::attachInterface<::SinOpInterface>(*context);
      SinhOp::attachInterface<::SinhOpInterface>(*context);
      SqrtOp::attachInterface<::SqrtOpInterface>(*context);
      TanOp::attachInterface<::TanOpInterface>(*context);
      TanhOp::attachInterface<::TanhOpInterface>(*context);

      // Control flow operations.
      IfOp::attachInterface<::IfOpInterface>(*context);
      ForOp::attachInterface<::ForOpInterface>(*context);
      WhileOp::attachInterface<::WhileOpInterface>(*context);

      // Function operations.
      CallOp::attachInterface<::CallOpInterface>(*context);
      AlgorithmOp::attachInterface<::AlgorithmOpInterface>(*context);
    });
  }
}
