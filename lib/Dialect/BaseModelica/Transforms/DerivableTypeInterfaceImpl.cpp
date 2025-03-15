#include "marco/Dialect/BaseModelica/Transforms/DerivableTypeInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::mlir::bmodelica;

namespace {
struct BooleanTypeInterface
    : public DerivableTypeInterface::ExternalModel<BooleanTypeInterface,
                                                   BooleanType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const {
    return RealType::get(type.getContext());
  }
};

struct IntegerTypeInterface
    : public DerivableTypeInterface::ExternalModel<IntegerTypeInterface,
                                                   IntegerType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const {
    return RealType::get(type.getContext());
  }
};

struct RealTypeInterface
    : public DerivableTypeInterface::ExternalModel<RealTypeInterface,
                                                   RealType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const {
    return RealType::get(type.getContext());
  }
};

struct ArrayTypeInterface
    : public DerivableTypeInterface::ExternalModel<ArrayTypeInterface,
                                                   ArrayType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const {
    auto arrayType = mlir::cast<ArrayType>(type);
    mlir::Type elementType = arrayType.getElementType();

    auto derivableElementType =
        mlir::dyn_cast<DerivableTypeInterface>(elementType);

    if (!derivableElementType) {
      return mlir::failure();
    }

    auto derivedElementType = derivableElementType.derive();

    if (mlir::failed(derivedElementType)) {
      return mlir::failure();
    }

    return arrayType.clone(*derivedElementType);
  }
};

struct DerivableTypeIndexModel
    : public DerivableTypeInterface::ExternalModel<DerivableTypeIndexModel,
                                                   mlir::IndexType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const {
    // TODO there should be a way to query the index bit-width.
    return mlir::Float64Type::get(type.getContext());
  }
};

struct DerivableTypeIntegerModel
    : public DerivableTypeInterface::ExternalModel<DerivableTypeIntegerModel,
                                                   mlir::IntegerType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const {
    switch (type.getIntOrFloatBitWidth()) {
    case 16:
      return mlir::Float16Type::get(type.getContext());

    case 32:
      return mlir::Float32Type::get(type.getContext());

    case 64:
      return mlir::Float64Type::get(type.getContext());

    case 80:
      return mlir::Float80Type::get(type.getContext());

    case 128:
      return mlir::Float128Type::get(type.getContext());

    default:
      return mlir::Float64Type::get(type.getContext());
    }
  }
};

template <typename FloatType>
struct DerivableTypeFloatModel
    : public DerivableTypeInterface::ExternalModel<
          DerivableTypeFloatModel<FloatType>, FloatType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const { return type; }
};

struct DerivableTypeTensorModel
    : public DerivableTypeInterface::ExternalModel<DerivableTypeTensorModel,
                                                   mlir::RankedTensorType> {
  mlir::FailureOr<mlir::Type> derive(mlir::Type type) const {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
    mlir::Type elementType = tensorType.getElementType();

    auto derivableElementType =
        mlir::dyn_cast<DerivableTypeInterface>(elementType);

    if (!derivableElementType) {
      return mlir::failure();
    }

    auto derivedElementType = derivableElementType.derive();

    if (mlir::failed(derivedElementType)) {
      return mlir::failure();
    }

    return tensorType.clone(*derivedElementType);
  }
};
} // namespace

namespace mlir::bmodelica {
void registerDerivableTypeInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            BaseModelicaDialect *dialect) {
    BooleanType::attachInterface<::BooleanTypeInterface>(*context);
    IntegerType::attachInterface<::IntegerTypeInterface>(*context);
    RealType::attachInterface<::RealTypeInterface>(*context);
    ArrayType::attachInterface<::ArrayTypeInterface>(*context);

    // Add the derivable interface to the built-in types.
    mlir::IndexType::attachInterface<DerivableTypeIndexModel>(*context);

    mlir::IntegerType::attachInterface<DerivableTypeIntegerModel>(*context);

    mlir::Float16Type::attachInterface<
        DerivableTypeFloatModel<mlir::Float16Type>>(*context);

    mlir::Float32Type::attachInterface<
        DerivableTypeFloatModel<mlir::Float32Type>>(*context);

    mlir::Float64Type::attachInterface<
        DerivableTypeFloatModel<mlir::Float64Type>>(*context);

    mlir::Float80Type::attachInterface<
        DerivableTypeFloatModel<mlir::Float80Type>>(*context);

    mlir::Float128Type::attachInterface<
        DerivableTypeFloatModel<mlir::Float128Type>>(*context);

    mlir::RankedTensorType::attachInterface<DerivableTypeTensorModel>(*context);
  });
}
} // namespace mlir::bmodelica
