#include "marco/Dialect/BaseModelica/Transforms/AllocationOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bufferization;

namespace
{
  struct AllocOpInterface
      : public AllocationOpInterface::ExternalModel<AllocOpInterface, AllocOp>
  {
    static std::optional<mlir::Operation*> buildDealloc(
        mlir::OpBuilder& builder, mlir::Value alloc)
    {
      return builder.create<FreeOp>(alloc.getLoc(), alloc).getOperation();
    }

    static std::optional<mlir::Value> buildClone(
        mlir::OpBuilder& builder, mlir::Value alloc)
    {
      return std::nullopt;
    }

    static mlir::HoistingKind getHoistingKind()
    {
      return mlir::HoistingKind::Loop | mlir::HoistingKind::Block;
    }

    static std::optional<mlir::Operation*> buildPromotedAlloc(
        mlir::OpBuilder& builder, mlir::Value alloc)
    {
      mlir::Operation* definingOp = alloc.getDefiningOp();

      return builder.create<AllocaOp>(
          definingOp->getLoc(),
          mlir::cast<ArrayType>(definingOp->getResultTypes()[0]),
          definingOp->getOperands(), definingOp->getAttrs());
    }
  };

  struct ArrayFromElementsOpInterface
      : public AllocationOpInterface::ExternalModel<
            ArrayFromElementsOpInterface, ArrayFromElementsOp>
  {
    static std::optional<mlir::Operation*> buildDealloc(
        mlir::OpBuilder& builder, mlir::Value alloc)
    {
      return builder.create<FreeOp>(alloc.getLoc(), alloc).getOperation();
    }

    static mlir::HoistingKind getHoistingKind()
    {
      return mlir::HoistingKind::Loop | mlir::HoistingKind::Block;
    }
  };

  struct ArrayBroadcastOpInterface
      : public AllocationOpInterface::ExternalModel<
            ArrayBroadcastOpInterface, ArrayBroadcastOp>
  {
    static std::optional<mlir::Operation*> buildDealloc(
        mlir::OpBuilder& builder, mlir::Value alloc)
    {
      return builder.create<FreeOp>(alloc.getLoc(), alloc).getOperation();
    }

    static mlir::HoistingKind getHoistingKind()
    {
      return mlir::HoistingKind::Loop | mlir::HoistingKind::Block;
    }
  };

  struct RawVariableOpInterface
      : public AllocationOpInterface::ExternalModel<
            RawVariableOpInterface, RawVariableOp>
  {
    static std::optional<mlir::Operation*> buildDealloc(
        mlir::OpBuilder& builder,
        mlir::Value alloc)
    {
      auto deallocOp = builder.create<RawVariableDeallocOp>(
          alloc.getLoc(), alloc);

      return deallocOp.getOperation();
    }

    static std::optional<mlir::Operation*> buildPromotedAlloc(
        mlir::OpBuilder& builder,
        mlir::Value alloc)
    {
      auto rawVariableOp = alloc.getDefiningOp<RawVariableOp>();

      if (!rawVariableOp) {
        return std::nullopt;
      }

      if (rawVariableOp.isDynamicArrayVariable()) {
        // Dynamically sized variables must always be allocated on the heap.
        return std::nullopt;
      }

      auto stackRawVariableOp = builder.create<RawVariableOp>(
          rawVariableOp.getLoc(),
          rawVariableOp.getResult().getType(),
          rawVariableOp.getName(),
          rawVariableOp.getDimensionsConstraints(),
          rawVariableOp.getDynamicSizes(),
          rawVariableOp.getOutput(),
          false);

      return stackRawVariableOp.getOperation();
    }
  };

  struct CallOpInterface
      : public AllocationOpInterface::ExternalModel<CallOpInterface, CallOp>
  {
    static std::optional<mlir::Operation*> buildDealloc(
        mlir::OpBuilder& builder, mlir::Value alloc)
    {
      return builder.create<FreeOp>(alloc.getLoc(), alloc).getOperation();
    }

    static mlir::HoistingKind getHoistingKind()
    {
      return mlir::HoistingKind::Loop | mlir::HoistingKind::Block;
    }
  };
}

namespace mlir::bmodelica
{
  void registerAllocationOpInterfaceExternalModels(
      mlir::DialectRegistry& registry)
  {
    registry.addExtension(+[](mlir::MLIRContext* context,
                              BaseModelicaDialect* dialect) {
      AllocOp::attachInterface<::AllocOpInterface>(*context);

      ArrayFromElementsOp::attachInterface<
          ::ArrayFromElementsOpInterface>(*context);

      ArrayBroadcastOp::attachInterface<::ArrayBroadcastOpInterface>(*context);
      CallOp::attachInterface<::CallOpInterface>(*context);
      RawVariableOp::attachInterface<::RawVariableOpInterface>(*context);
    });
  }
}
