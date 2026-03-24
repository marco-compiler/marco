#include "marco/Dialect/BaseModelica/Transforms/ClassInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/Dialect.h"

using namespace ::mlir::bmodelica;

namespace {
struct PackageOpInterface
    : public ClassInterface::ExternalModel<PackageOpInterface, PackageOp> {
  llvm::StringRef getClassName(mlir::Operation *op) const {
    return mlir::cast<PackageOp>(op).getSymName();
  }

  mlir::Region &getBody(mlir::Operation *op) const {
    return mlir::cast<PackageOp>(op).getBodyRegion();
  }
};

struct ModelOpInterface
    : public ClassInterface::ExternalModel<ModelOpInterface, ModelOp> {
  llvm::StringRef getClassName(mlir::Operation *op) const {
    return mlir::cast<ModelOp>(op).getSymName();
  }

  mlir::Region &getBody(mlir::Operation *op) const {
    return mlir::cast<ModelOp>(op).getBodyRegion();
  }
};

struct RecordOpInterface
    : public ClassInterface::ExternalModel<RecordOpInterface, RecordOp> {
  llvm::StringRef getClassName(mlir::Operation *op) const {
    return mlir::cast<RecordOp>(op).getSymName();
  }

  mlir::Region &getBody(mlir::Operation *op) const {
    return mlir::cast<RecordOp>(op).getBodyRegion();
  }
};

struct OperatorRecordOpInterface
    : public ClassInterface::ExternalModel<OperatorRecordOpInterface,
                                           OperatorRecordOp> {
  llvm::StringRef getClassName(mlir::Operation *op) const {
    return mlir::cast<OperatorRecordOp>(op).getSymName();
  }

  mlir::Region &getBody(mlir::Operation *op) const {
    return mlir::cast<OperatorRecordOp>(op).getBodyRegion();
  }
};

struct FunctionOpInterface
    : public ClassInterface::ExternalModel<FunctionOpInterface, FunctionOp> {
  llvm::StringRef getClassName(mlir::Operation *op) const {
    return mlir::cast<FunctionOp>(op).getSymName();
  }

  mlir::Region &getBody(mlir::Operation *op) const {
    return mlir::cast<FunctionOp>(op).getBodyRegion();
  }
};
} // namespace

namespace mlir::bmodelica {
void registerClassInterfaceExternalModels(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            BaseModelicaDialect *dialect) {
    // clang-format off
    PackageOp::attachInterface<::PackageOpInterface>(*context);
    ModelOp::attachInterface<::ModelOpInterface>(*context);
    RecordOp::attachInterface<::RecordOpInterface>(*context);
    OperatorRecordOp::attachInterface<::OperatorRecordOpInterface>(*context);
    FunctionOp::attachInterface<::FunctionOpInterface>(*context);
    // clang-format on
  });
}
} // namespace mlir::bmodelica
