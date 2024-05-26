#include "marco/Dialect/BaseModelica/Transforms/ConstantMaterializableTypeInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelicaDialect.h"

using namespace ::mlir::bmodelica;

namespace
{
  struct BooleanTypeInterface
      : public ConstantMaterializableTypeInterface::ExternalModel<
            BooleanTypeInterface, BooleanType>
  {
    mlir::Value materializeBoolConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        bool value) const
    {
      return builder.create<ConstantOp>(
          loc, BooleanAttr::get(builder.getContext(), value));
    }

    mlir::Value materializeIntConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        int64_t value) const
    {
      return materializeBoolConstant(
          type, builder, loc, static_cast<bool>(value != 0));
    }

    mlir::Value materializeFloatConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        double value) const
    {
      return materializeBoolConstant(
          type, builder, loc, static_cast<bool>(value != 0));
    }
  };

  struct IntegerTypeInterface
      : public ConstantMaterializableTypeInterface::ExternalModel<
            IntegerTypeInterface, IntegerType>
  {
    mlir::Value materializeBoolConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        bool value) const
    {
      return materializeIntConstant(
          type, builder, loc, static_cast<int64_t>(value));
    }

    mlir::Value materializeIntConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        int64_t value) const
    {
      return builder.create<ConstantOp>(
          loc, IntegerAttr::get(builder.getContext(), value));
    }

    mlir::Value materializeFloatConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        double value) const
    {
      return materializeIntConstant(
          type, builder, loc, static_cast<int64_t>(value));
    }
  };

  struct RealTypeInterface
      : public ConstantMaterializableTypeInterface::ExternalModel<
            RealTypeInterface, RealType>
  {
    mlir::Value materializeBoolConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        bool value) const
    {
      return materializeFloatConstant(
          type, builder, loc, static_cast<double>(value));
    }

    mlir::Value materializeIntConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        int64_t value) const
    {
      return materializeFloatConstant(
          type, builder, loc, static_cast<double>(value));
    }

    mlir::Value materializeFloatConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        double value) const
    {
      return builder.create<ConstantOp>(
          loc, RealAttr::get(builder.getContext(), value));
    }
  };

  struct ConstantMaterializableBuiltinIndexModel
      : public ConstantMaterializableTypeInterface::ExternalModel<
            ConstantMaterializableBuiltinIndexModel, mlir::IndexType>
  {
    static mlir::Value materializeBoolConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        bool value)
    {
      return builder.create<ConstantOp>(
          loc, builder.getIndexAttr(static_cast<int64_t>(value)));
    }

    static mlir::Value materializeIntConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        int64_t value)
    {
      return builder.create<ConstantOp>(loc, builder.getIndexAttr(value));
    }

    static mlir::Value materializeFloatConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        double value)
    {
      return builder.create<ConstantOp>(
          loc, builder.getIndexAttr(static_cast<int64_t>(value)));
    }
  };

  struct ConstantMaterializableBuiltinIntegerModel
      : public ConstantMaterializableTypeInterface::ExternalModel<
            ConstantMaterializableBuiltinIntegerModel, mlir::IntegerType>
  {
    static mlir::Value materializeBoolConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        bool value)
    {
      return builder.create<ConstantOp>(
          loc, builder.getIntegerAttr(type, static_cast<int64_t>(value)));
    }

    static mlir::Value materializeIntConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        int64_t value)
    {
      return builder.create<ConstantOp>(loc, builder.getIntegerAttr(type, value));
    }

    static mlir::Value materializeFloatConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        double value)
    {
      return builder.create<ConstantOp>(
          loc, builder.getIntegerAttr(type, static_cast<int64_t>(value)));
    }
  };

  template<typename FloatType>
  struct ConstantMaterializableBuiltinFloatModel
      : public ConstantMaterializableTypeInterface::ExternalModel<
            ConstantMaterializableBuiltinFloatModel<FloatType>, FloatType>
  {
    static mlir::Value materializeBoolConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        bool value)
    {
      return builder.create<ConstantOp>(
          loc, builder.getFloatAttr(type, static_cast<double>(value)));
    }

    static mlir::Value materializeIntConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        int64_t value)
    {
      return builder.create<ConstantOp>(
          loc, builder.getFloatAttr(type, static_cast<double>(value)));
    }

    static mlir::Value materializeFloatConstant(
        mlir::Type type,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        double value)
    {
      return builder.create<ConstantOp>(
          loc, builder.getFloatAttr(type, value));
    }
  };
}

namespace mlir::bmodelica
{
  void registerConstantMaterializableTypeInterfaceExternalModels(
      mlir::DialectRegistry& registry)
  {
    registry.addExtension(+[](mlir::MLIRContext* context,
                              BaseModelicaDialect* dialect) {
      BooleanType::attachInterface<BooleanTypeInterface>(*context);
      IntegerType::attachInterface<IntegerTypeInterface>(*context);
      RealType::attachInterface<RealTypeInterface>(*context);

      // Add the constant-materialization type interface to built-in types.
      mlir::IndexType::attachInterface<
          ::ConstantMaterializableBuiltinIndexModel>(*context);

      mlir::IntegerType::attachInterface<
          ::ConstantMaterializableBuiltinIntegerModel>(*context);

      mlir::Float16Type::attachInterface<
          ::ConstantMaterializableBuiltinFloatModel<Float16Type>>(*context);

      mlir::Float32Type::attachInterface<
          ::ConstantMaterializableBuiltinFloatModel<Float32Type>>(*context);

      mlir::Float64Type::attachInterface<
          ::ConstantMaterializableBuiltinFloatModel<Float64Type>>(*context);

      mlir::Float80Type::attachInterface<
          ::ConstantMaterializableBuiltinFloatModel<Float80Type>>(*context);

      mlir::Float128Type::attachInterface<
          ::ConstantMaterializableBuiltinFloatModel<Float128Type>>(*context);
    });
  }
}
