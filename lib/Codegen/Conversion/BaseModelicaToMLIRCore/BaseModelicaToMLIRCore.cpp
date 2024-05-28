#include "marco/Codegen/Conversion/BaseModelicaToMLIRCore/BaseModelicaToMLIRCore.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/BaseModelicaToArith/BaseModelicaToArith.h"
#include "marco/Codegen/Conversion/BaseModelicaToFunc/BaseModelicaToFunc.h"
#include "marco/Codegen/Conversion/BaseModelicaToLinalg/BaseModelicaToLinalg.h"
#include "marco/Codegen/Conversion/BaseModelicaToMemRef/BaseModelicaToMemRef.h"
#include "marco/Codegen/Conversion/BaseModelicaToRuntimeCall/BaseModelicaToRuntimeCall.h"
#include "marco/Codegen/Conversion/BaseModelicaToTensor/BaseModelicaToTensor.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_BASEMODELICATOMLIRCORECONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class BaseModelicaToMLIRCoreConversionPass
      : public mlir::impl::BaseModelicaToMLIRCoreConversionPassBase<
            BaseModelicaToMLIRCoreConversionPass>
  {
    public:
      using BaseModelicaToMLIRCoreConversionPassBase<
          BaseModelicaToMLIRCoreConversionPass>
          ::BaseModelicaToMLIRCoreConversionPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult convertOperations();
  };
}

void BaseModelicaToMLIRCoreConversionPass::runOnOperation()
{
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult BaseModelicaToMLIRCoreConversionPass::convertOperations()
{
  mlir::ModuleOp module = getOperation();
  mlir::ConversionTarget target(getContext());
  TypeConverter typeConverter;

  target.addLegalDialect<
      mlir::BuiltinDialect,
      mlir::arith::ArithDialect,
      mlir::bufferization::BufferizationDialect,
      mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::runtime::RuntimeDialect,
      mlir::tensor::TensorDialect>();

  target.addLegalOp<RangeOp, RangeBeginOp, RangeEndOp, RangeStepOp>();
  target.addIllegalOp<CastOp>();

  target.addIllegalOp<
      TensorFromElementsOp,
      TensorBroadcastOp,
      TensorViewOp,
      TensorExtractOp,
      TensorInsertOp,
      TensorInsertSliceOp>();

  target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
    return op.getResult().getType().isa<RangeType>();
  });

  target.addIllegalOp<EqOp, NotEqOp, GtOp, GteOp, LtOp, LteOp>();
  target.addIllegalOp<NotOp, AndOp, OrOp>();

  target.addIllegalOp<NegateOp>();
  target.addIllegalOp<AddOp, AddEWOp>();
  target.addIllegalOp<SubOp, SubEWOp>();
  target.addIllegalOp<MulOp, MulEWOp>();
  target.addIllegalOp<DivOp, DivEWOp>();
  target.addIllegalOp<PowOp>();
  target.addIllegalOp<TransposeOp>();

  target.addIllegalOp<SelectOp>();
  target.addIllegalOp<RangeSizeOp>();

  target.addIllegalOp<GlobalVariableOp, GlobalVariableGetOp>();

  target.addIllegalOp<
      AllocaOp,
      AllocOp,
      ArrayFromElementsOp,
      ArrayBroadcastOp,
      FreeOp,
      DimOp,
      SubscriptionOp,
      LoadOp,
      StoreOp,
      ArrayCastOp,
      ArrayFillOp,
      ArrayCopyOp>();

  target.addIllegalOp<
      EquationFunctionOp,
      EquationCallOp,
      RawFunctionOp,
      RawReturnOp,
      CallOp>();

  target.addDynamicallyLegalOp<RawVariableOp>([&](RawVariableOp op) {
    mlir::Type variableType = op.getVariable().getType();
    return variableType == typeConverter.convertType(variableType);
  });

  target.addDynamicallyLegalOp<RawVariableGetOp>([&](RawVariableGetOp op) {
    mlir::Type resultType = op.getResult().getType();
    return resultType == typeConverter.convertType(resultType);
  });

  target.addDynamicallyLegalOp<RawVariableSetOp>([&](RawVariableSetOp op) {
    mlir::Type valueType = op.getValue().getType();
    return valueType == typeConverter.convertType(valueType);
  });

  target.addIllegalOp<
      AbsOp,
      AcosOp,
      AsinOp,
      AtanOp,
      Atan2Op,
      CeilOp,
      CosOp,
      CoshOp,
      DiagonalOp,
      DivTruncOp,
      ExpOp,
      FillOp,
      FloorOp,
      IdentityOp,
      IntegerOp,
      LinspaceOp,
      LogOp,
      Log10Op,
      OnesOp,
      MaxOp,
      MinOp,
      ModOp,
      NDimsOp,
      ProductOp,
      RemOp,
      SignOp,
      SinOp,
      SinhOp,
      SizeOp,
      SqrtOp,
      SumOp,
      SymmetricOp,
      TanOp,
      TanhOp,
      TransposeOp,
      ZerosOp>();

  target.addIllegalOp<PrintOp>();
  target.addIllegalOp<ArrayToTensorOp, TensorToArrayOp>();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  populateBaseModelicaToRuntimeCallConversionPatterns(
      patterns, &getContext(), typeConverter, symbolTableCollection);

  populateBaseModelicaToTensorConversionPatterns(
      patterns, &getContext(), typeConverter);

  populateBaseModelicaToArithConversionPatterns(
      patterns, &getContext(), typeConverter);

  populateBaseModelicaToFuncConversionPatterns(
      patterns, &getContext(), typeConverter);

  populateBaseModelicaRawVariablesTypeLegalizationPatterns(
      patterns, &getContext(), typeConverter);

  populateBaseModelicaToLinalgConversionPatterns(
      patterns, &getContext(), typeConverter);

  populateBaseModelicaToMemRefConversionPatterns(
      patterns, &getContext(), typeConverter, symbolTableCollection);

  return applyPartialConversion(module, target, std::move(patterns));
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createBaseModelicaToMLIRCoreConversionPass()
  {
    return std::make_unique<BaseModelicaToMLIRCoreConversionPass>();
  }
}
