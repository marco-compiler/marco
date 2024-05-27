#include "marco/Dialect/BaseModelica/Transforms/InvertibleOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::mlir::bmodelica;

static mlir::Value unwrap(mlir::OpBuilder& builder, mlir::Value operand)
{
  if (auto arrayType = operand.getType().dyn_cast<ArrayType>();
      arrayType && arrayType.getRank() == 0) {
    return builder.create<LoadOp>(operand.getLoc(), operand);
  }

  if (auto tensorType = operand.getType().dyn_cast<mlir::TensorType>();
      tensorType && tensorType.getRank() == 0) {
    return builder.create<TensorExtractOp>(operand.getLoc(), operand);
  }

  return operand;
}

namespace
{
  struct NegateOpInterface
      : public InvertibleOpInterface::ExternalModel<
            NegateOpInterface, NegateOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<NegateOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (argumentIndex > 0) {
        castedOp.emitOpError()
            << "Index out of bounds: " << argumentIndex << ".";

        return nullptr;
      }

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];
      mlir::Value nestedOperand = unwrap(builder, toNest);

      auto right = builder.create<NegateOp>(
          castedOp.getLoc(), castedOp.getOperand().getType(), nestedOperand);

      return right.getResult();
    }
  };

  struct AddOpInterface
      : public InvertibleOpInterface::ExternalModel<AddOpInterface, AddOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<AddOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<SubOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<SubOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), nestedOperand,
            castedOp.getLhs());

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct AddEWOpInterface
      : public InvertibleOpInterface::ExternalModel<AddEWOpInterface, AddEWOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<AddEWOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<SubEWOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<SubEWOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), nestedOperand,
            castedOp.getLhs());

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct SubOpInterface
      : public InvertibleOpInterface::ExternalModel<SubOpInterface, SubOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<SubOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<AddOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<SubOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), castedOp.getLhs(),
            nestedOperand);

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct SubEWOpInterface
      : public InvertibleOpInterface::ExternalModel<SubEWOpInterface, SubEWOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<SubEWOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<AddEWOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<SubEWOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), castedOp.getLhs(),
            nestedOperand);

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct MulOpInterface
      : public InvertibleOpInterface::ExternalModel<MulOpInterface, MulOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<MulOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<DivOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<DivOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), nestedOperand,
            castedOp.getLhs());

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct MulEWOpInterface
      : public InvertibleOpInterface::ExternalModel<MulEWOpInterface, MulEWOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<MulEWOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<DivEWOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<DivEWOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), nestedOperand,
            castedOp.getLhs());

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct DivOpInterface
      : public InvertibleOpInterface::ExternalModel<DivOpInterface, DivOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<DivOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<MulOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);
        
        auto right = builder.create<DivOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), castedOp.getLhs(),
            nestedOperand);

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct DivEWOpInterface
      : public InvertibleOpInterface::ExternalModel<DivEWOpInterface, DivEWOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<DivEWOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1)";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      if (argumentIndex == 0) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<MulEWOp>(
            castedOp.getLoc(), castedOp.getLhs().getType(), nestedOperand,
            castedOp.getRhs());

        return right.getResult();
      }

      if (argumentIndex == 1) {
        mlir::Value nestedOperand = unwrap(builder, toNest);

        auto right = builder.create<DivEWOp>(
            castedOp.getLoc(), castedOp.getRhs().getType(), castedOp.getLhs(),
            nestedOperand);

        return right.getResult();
      }

      castedOp.emitOpError()
          << "Can't invert the operand #" << argumentIndex
          << ". The operation has 2 operands.";

      return nullptr;
    }
  };

  struct CallOpInterface
      : public InvertibleOpInterface::ExternalModel<CallOpInterface, CallOp>
  {
    mlir::Value inverse(
        mlir::Operation* op,
        mlir::OpBuilder& builder,
        unsigned int argumentIndex,
        mlir::ValueRange currentResult) const
    {
      auto castedOp = mlir::cast<CallOp>(op);
      mlir::OpBuilder::InsertionGuard guard(builder);

      if (castedOp.getNumResults() != 1) {
        castedOp.emitOpError()
            << "The callee must have one and only one result.";
        
        return nullptr;
      }

      if (argumentIndex >= castedOp.getArgs().size()) {
        castedOp.emitOpError()
            << "Index out of bounds: " << argumentIndex << ".";
        
        return nullptr;
      }

      if (size_t size = currentResult.size(); size != 1) {
        castedOp.emitOpError()
            << "Invalid amount of values to be nested: " << size
            << " (expected 1).";

        return nullptr;
      }

      mlir::Value toNest = currentResult[0];

      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      auto callee = moduleOp.lookupSymbol<FunctionOp>(castedOp.getCallee());

      if (!callee->hasAttr("inverse")) {
        castedOp.emitOpError()
            << "Function " << callee->getName().getStringRef()
            << " is not invertible.";

        return nullptr;
      }

      auto inverseAnnotation =
          callee->getAttrOfType<InverseFunctionsAttr>("inverse");

      if (!inverseAnnotation.isInvertible(argumentIndex)) {
        castedOp.emitOpError()
            << "Function " << callee->getName().getStringRef()
            << " is not invertible for argument " << argumentIndex << ".";

        return nullptr;
      }

      size_t argsSize = castedOp.getArgs().size();
      llvm::SmallVector<mlir::Value, 3> args;

      for (auto arg : inverseAnnotation.getArgumentsIndexes(argumentIndex)) {
        if (arg < argsSize) {
          args.push_back(castedOp.getArgs()[arg]);
        } else {
          assert(arg == argsSize);
          args.push_back(toNest);
        }
      }

      auto invertedCall = builder.create<CallOp>(
          castedOp.getLoc(),
          mlir::SymbolRefAttr::get(builder.getStringAttr(
              inverseAnnotation.getFunction(argumentIndex))),
          castedOp.getArgs()[argumentIndex].getType(),
          args);

      return invertedCall.getResult(0);
    }
  };
}

namespace mlir::bmodelica
{
  void registerInvertibleOpInterfaceExternalModels(
      mlir::DialectRegistry& registry)
  {
    registry.addExtension(+[](mlir::MLIRContext* context,
                              BaseModelicaDialect* dialect) {
      NegateOp::attachInterface<::NegateOpInterface>(*context);
      AddOp::attachInterface<::AddOpInterface>(*context);
      AddEWOp::attachInterface<::AddEWOpInterface>(*context);
      SubOp::attachInterface<::SubOpInterface>(*context);
      SubEWOp::attachInterface<::SubEWOpInterface>(*context);
      MulOp::attachInterface<::MulOpInterface>(*context);
      MulEWOp::attachInterface<::MulEWOpInterface>(*context);
      DivOp::attachInterface<::DivOpInterface>(*context);
      DivEWOp::attachInterface<::DivEWOpInterface>(*context);
      CallOp::attachInterface<::CallOpInterface>(*context);
    });
  }
}
