#include "marco/Codegen/Conversion/ModelicaToFunc/ModelicaToFunc.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

/// Get or declare an LLVM function inside the module.
static RuntimeFunctionOp getOrDeclareRuntimeFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    llvm::StringRef name,
    mlir::TypeRange results,
    mlir::TypeRange args)
{
  if (auto funcOp = module.lookupSymbol<RuntimeFunctionOp>(name)) {
    return funcOp;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto functionType = builder.getFunctionType(args, results);
  return builder.create<RuntimeFunctionOp>(module.getLoc(), name, functionType);
}

/// Get or declare an LLVM function inside the module.
static RuntimeFunctionOp getOrDeclareRuntimeFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    llvm::StringRef name,
    mlir::TypeRange results,
    mlir::ValueRange args)
{
  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (const auto& arg : args) {
    argsTypes.push_back(arg.getType());
  }

  return getOrDeclareRuntimeFunction(builder, module, name, results, argsTypes);
}

namespace
{
  /// Generic rewrite pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpRewritePattern : public mlir::OpRewritePattern<Op>
  {
    using mlir::OpRewritePattern<Op>::OpRewritePattern;
  };

  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpConversionPattern : public mlir::OpConversionPattern<Op>
  {
    public:
      using mlir::OpConversionPattern<Op>::OpConversionPattern;

    protected:
      mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) const
      {
        mlir::Type type = this->getTypeConverter()->convertType(value.getType());
        return this->getTypeConverter()->materializeTargetConversion(builder, loc, type, value);
      }

      void materializeTargetConversion(mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Value>& values) const
      {
        for (auto& value : values) {
          value = materializeTargetConversion(builder, value);
        }
      }

      mlir::Value convertToUnrankedMemRef(
        mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memRef) const
      {
        auto memRefType = memRef.getType().cast<mlir::MemRefType>();

        auto unrankedMemRefType = mlir::UnrankedMemRefType::get(
            memRefType.getElementType(), memRefType.getMemorySpace());

        mlir::Value unrankedMemRef = builder.create<mlir::memref::CastOp>(
            loc, unrankedMemRefType, memRef);

        return unrankedMemRef;
      }
  };

  template<typename Op>
  class RuntimeOpConversionPattern : public ModelicaOpConversionPattern<Op>
  {
    public:
      using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;

    protected:
      const RuntimeFunctionsMangling* getMangler() const
      {
        return &mangler;
      }

      std::string getMangledType(mlir::Type type) const
      {
        if (auto indexType = type.dyn_cast<mlir::IndexType>()) {
          return getMangledType(this->getTypeConverter()->convertType(type));
        }

        if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
          return getMangler()->getIntegerType(integerType.getWidth());
        }

        if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
          return getMangler()->getFloatingPointType(floatType.getWidth());
        }

        if (auto memRefType = type.dyn_cast<mlir::UnrankedMemRefType>()) {
          return getMangler()->getArrayType(getMangledType(memRefType.getElementType()));
        }

        llvm_unreachable("Unknown type for mangling");
        return "unknown";
      }

      std::string getMangledFunctionName(
          llvm::StringRef name,
          mlir::TypeRange resultTypes,
          mlir::TypeRange argTypes) const
      {
        llvm::SmallVector<std::string, 3> mangledArgTypes;

        for (mlir::Type type : argTypes) {
          mangledArgTypes.push_back(getMangledType(type));
        }

        assert(resultTypes.size() <= 1);

        if (resultTypes.empty()) {
          return getMangler()->getMangledFunction(
              name, getMangler()->getVoidType(), mangledArgTypes);
        }

        return getMangler()->getMangledFunction(
            name, getMangledType(resultTypes[0]), mangledArgTypes);
      }

      std::string getMangledFunctionName(
          llvm::StringRef name,
          mlir::TypeRange resultTypes,
          mlir::ValueRange args) const
      {
        return getMangledFunctionName(name, resultTypes, args.getTypes());
      }

    private:
      RuntimeFunctionsMangling mangler;
  };
}

//===----------------------------------------------------------------------===//
// Func operations
//===----------------------------------------------------------------------===//

namespace
{
  struct RawFunctionOpLowering : public ModelicaOpConversionPattern<RawFunctionOp>
  {
    using ModelicaOpConversionPattern<RawFunctionOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(RawFunctionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type, 3> argsTypes;
      llvm::SmallVector<mlir::Type, 3> resultsTypes;

      for (const auto& argType : op.getFunctionType().getInputs()) {
        argsTypes.push_back(getTypeConverter()->convertType(argType));
      }

      for (const auto& resultType : op.getFunctionType().getResults()) {
        resultsTypes.push_back(getTypeConverter()->convertType(resultType));
      }

      auto functionType = rewriter.getFunctionType(argsTypes, resultsTypes);
      auto funcOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(op, op.getSymName(), functionType);

      rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

      if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter))) {
        return mlir::failure();
      }

      return mlir::success();
    }
  };

  struct RawReturnOpLowering : public mlir::OpConversionPattern<RawReturnOp>
  {
    using mlir::OpConversionPattern<RawReturnOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(RawReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
      return mlir::success();
    }
  };

  struct CallOpLowering : public mlir::OpConversionPattern<CallOp>
  {
    using mlir::OpConversionPattern<CallOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type, 3> resultsTypes;

      if (auto res = getTypeConverter()->convertTypes(op->getResultTypes(), resultsTypes); mlir::failed(res)) {
        return res;
      }

      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, op.getCallee(), resultsTypes, adaptor.getOperands());
      return mlir::success();
    }
  };
}

//===----------------------------------------------------------------------===//
// Math operations
//===----------------------------------------------------------------------===//

namespace
{
  struct PowOpLowering: public RuntimeOpConversionPattern<PowOp>
  {
    using RuntimeOpConversionPattern<PowOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        PowOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!isNumeric(op.getBase())) {
        return rewriter.notifyMatchFailure(op, "Base is not a scalar");
      }

      llvm::SmallVector<mlir::Value, 2> newOperands;

      // Base.
      newOperands.push_back(adaptor.getBase());

      // Exponent.
      newOperands.push_back(adaptor.getExponent());

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("pow", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };
}

//===----------------------------------------------------------------------===//
// Built-in functions
//===----------------------------------------------------------------------===//

namespace
{
  struct AbsOpCastPattern : public ModelicaOpRewritePattern<AbsOp>
  {
    using ModelicaOpRewritePattern<AbsOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AbsOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType != resultType);
    }

    void rewrite(AbsOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AbsOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AbsOpLowering : public RuntimeOpConversionPattern<AbsOp>
  {
    using RuntimeOpConversionPattern<AbsOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(AbsOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(
        AbsOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("abs", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct AcosOpCastPattern : public ModelicaOpRewritePattern<AcosOp>
  {
    using ModelicaOpRewritePattern<AcosOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AcosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(AcosOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AcosOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AcosOpLowering : public RuntimeOpConversionPattern<AcosOp>
  {
    using RuntimeOpConversionPattern<AcosOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(AcosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        AcosOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("acos", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct AsinOpCastPattern : public ModelicaOpRewritePattern<AsinOp>
  {
    using ModelicaOpRewritePattern<AsinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AsinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(AsinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AsinOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AsinOpLowering : public RuntimeOpConversionPattern<AsinOp>
  {
    using RuntimeOpConversionPattern<AsinOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(AsinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        AsinOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("asin", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct AtanOpCastPattern : public ModelicaOpRewritePattern<AtanOp>
  {
    using ModelicaOpRewritePattern<AtanOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AtanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(AtanOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AtanOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AtanOpLowering : public RuntimeOpConversionPattern<AtanOp>
  {
    using RuntimeOpConversionPattern<AtanOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(AtanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        AtanOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("atan", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct Atan2OpCastPattern : public ModelicaOpRewritePattern<Atan2Op>
  {
    using ModelicaOpRewritePattern<Atan2Op>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(Atan2Op op) const override
    {
      mlir::Type yType = op.getY().getType();
      mlir::Type xType = op.getX().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !yType.isa<RealType>() || !xType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(Atan2Op op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value y = rewriter.create<CastOp>(loc, realType, op.getY());
      mlir::Value x = rewriter.create<CastOp>(loc, realType, op.getX());

      mlir::Value result = rewriter.create<Atan2Op>(loc, realType, y, x);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct Atan2OpLowering : public RuntimeOpConversionPattern<Atan2Op>
  {
    using RuntimeOpConversionPattern<Atan2Op>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(Atan2Op op) const override
    {
      mlir::Type yType = op.getY().getType();
      mlir::Type xType = op.getX().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          yType.isa<RealType>() &&
          xType.isa<RealType>() &&
          resultType.isa<RealType>());
    }

    void rewrite(
        Atan2Op op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;

      // Y.
      assert(op.getY().getType().isa<RealType>());
      newOperands.push_back(adaptor.getY());

      // X.
      assert(op.getX().getType().isa<RealType>());
      newOperands.push_back(adaptor.getX());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("atan2", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct CeilOpCastPattern : public ModelicaOpRewritePattern<CeilOp>
  {
    using ModelicaOpRewritePattern<CeilOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(CeilOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(CeilOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value result = rewriter.create<CeilOp>(loc, op.getOperand().getType(), op.getOperand());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct CeilOpLowering : public RuntimeOpConversionPattern<CeilOp>
  {
    using RuntimeOpConversionPattern<CeilOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(CeilOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(
        CeilOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("ceil", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct CosOpCastPattern : public ModelicaOpRewritePattern<CosOp>
  {
    using ModelicaOpRewritePattern<CosOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(CosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    mlir::LogicalResult matchAndRewrite(CosOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<CosOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct CosOpLowering : public RuntimeOpConversionPattern<CosOp>
  {
    using RuntimeOpConversionPattern<CosOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(CosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        CosOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("cos", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct CoshOpCastPattern : public ModelicaOpRewritePattern<CoshOp>
  {
    using ModelicaOpRewritePattern<CoshOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(CoshOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(CoshOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<CoshOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct CoshOpLowering : public RuntimeOpConversionPattern<CoshOp>
  {
    using RuntimeOpConversionPattern<CoshOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(CoshOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        CoshOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("cosh", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct DiagonalOpLowering : public RuntimeOpConversionPattern<DiagonalOp>
  {
    using RuntimeOpConversionPattern<DiagonalOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        DiagonalOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> newOperands;

      // Result.
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          if (dynamicDimensions.empty()) {
            assert(op.getValues().getType().cast<ArrayType>().getRank() == 1);
            mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

            mlir::Value values = getTypeConverter()->materializeSourceConversion(
                rewriter, op.getValues().getLoc(), op.getValues().getType(), adaptor.getValues());

            dynamicDimensions.push_back(rewriter.create<DimOp>(loc, values, zeroValue));
          } else {
            dynamicDimensions.push_back(dynamicDimensions[0]);
          }
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      result = materializeTargetConversion(rewriter, loc, result);
      result = convertToUnrankedMemRef(rewriter, loc, result);

      newOperands.push_back(result);

      // Values.
      mlir::Value values = adaptor.getValues();
      values = convertToUnrankedMemRef(rewriter, loc, values);

      newOperands.push_back(values);

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("diagonal", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.create<CallOp>(loc, callee, newOperands);
      return mlir::success();
    }
  };

  struct DivTruncOpCastPattern : public ModelicaOpRewritePattern<DivTruncOp>
  {
    using ModelicaOpRewritePattern<DivTruncOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(DivTruncOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType != yType || xType != resultType);
    }

    void rewrite(DivTruncOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getX(), op.getY() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<DivTruncOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct DivTruncOpLowering : public RuntimeOpConversionPattern<DivTruncOp>
  {
    using RuntimeOpConversionPattern<DivTruncOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(DivTruncOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType == yType && xType == resultType);
    }

    void rewrite(
        DivTruncOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;

      assert(op.getX().getType() == op.getY().getType());

      // Dividend.
      newOperands.push_back(adaptor.getX());

      // Divisor.
      newOperands.push_back(adaptor.getY());

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("div", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct ExpOpCastPattern : public ModelicaOpRewritePattern<ExpOp>
  {
    using ModelicaOpRewritePattern<ExpOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(ExpOp op) const override
    {
      mlir::Type operandType = op.getExponent().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(ExpOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value exponent = rewriter.create<CastOp>(loc, realType, op.getExponent());
      mlir::Value result = rewriter.create<ExpOp>(loc, realType, exponent);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct ExpOpLowering : public RuntimeOpConversionPattern<ExpOp>
  {
    using RuntimeOpConversionPattern<ExpOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(ExpOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        ExpOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Exponent.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getExponent());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("exp", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct FloorOpCastPattern : public ModelicaOpRewritePattern<FloorOp>
  {
    using ModelicaOpRewritePattern<FloorOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(FloorOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(FloorOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value result = rewriter.create<FloorOp>(loc, op.getOperand().getType(), op.getOperand());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct FloorOpLowering : public RuntimeOpConversionPattern<FloorOp>
  {
    using RuntimeOpConversionPattern<FloorOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(FloorOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(
        FloorOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("floor", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct IdentityOpCastPattern : public ModelicaOpRewritePattern<IdentityOp>
  {
    using ModelicaOpRewritePattern<IdentityOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(IdentityOp op) const override
    {
      mlir::Type sizeType = op.getSize().getType();
      return mlir::LogicalResult::success(!sizeType.isa<mlir::IndexType>());
    }

    void rewrite(IdentityOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value operand = rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.getSize());
      rewriter.replaceOpWithNewOp<IdentityOp>(op, op.getResult().getType(), operand);
    }
  };

  struct IdentityOpLowering : public RuntimeOpConversionPattern<IdentityOp>
  {
    using RuntimeOpConversionPattern<IdentityOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(IdentityOp op) const override
    {
      mlir::Type sizeType = op.getSize().getType();
      return mlir::LogicalResult::success(sizeType.isa<mlir::IndexType>());
    }

    void rewrite(
        IdentityOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Result.
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          if (dynamicDimensions.empty()) {
            dynamicDimensions.push_back(adaptor.getSize());
          } else {
            dynamicDimensions.push_back(dynamicDimensions[0]);
          }
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultType, dynamicDimensions);

      result = materializeTargetConversion(rewriter, loc, result);
      result = convertToUnrankedMemRef(rewriter, loc, result);

      newOperands.push_back(result);

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("identity", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.create<CallOp>(loc, callee, newOperands);
    }
  };

  struct IntegerOpCastPattern : public ModelicaOpRewritePattern<IntegerOp>
  {
    using ModelicaOpRewritePattern<IntegerOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(IntegerOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(IntegerOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value result = rewriter.create<IntegerOp>(loc, op.getOperand().getType(), op.getOperand());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct IntegerOpLowering : public RuntimeOpConversionPattern<IntegerOp>
  {
    using RuntimeOpConversionPattern<IntegerOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(IntegerOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(
        IntegerOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("integer", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct LinspaceOpCastPattern : public ModelicaOpRewritePattern<LinspaceOp>
  {
    using ModelicaOpRewritePattern<LinspaceOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(LinspaceOp op) const override
    {
      mlir::Type beginType = op.getBegin().getType();
      mlir::Type endType = op.getEnd().getType();
      mlir::Type amountType = op.getAmount().getType();

      return mlir::LogicalResult::success(
          !beginType.isa<RealType>() || !endType.isa<RealType>() || !amountType.isa<mlir::IndexType>());
    }

    void rewrite(LinspaceOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      auto realType = RealType::get(op.getContext());

      mlir::Value begin = op.getBegin();

      if (!begin.getType().isa<RealType>()) {
        begin = rewriter.create<CastOp>(loc, realType, begin);
      }

      mlir::Value end = op.getEnd();

      if (!end.getType().isa<RealType>()) {
        end = rewriter.create<CastOp>(loc, realType, end);
      }

      mlir::Value amount = op.getAmount();

      if (!amount.getType().isa<mlir::IndexType>()) {
        amount = rewriter.create<CastOp>(loc, rewriter.getIndexType(), amount);
      }

      rewriter.replaceOpWithNewOp<LinspaceOp>(op, op.getResult().getType(), begin, end, amount);
    }
  };

  struct LinspaceOpLowering : public RuntimeOpConversionPattern<LinspaceOp>
  {
    using RuntimeOpConversionPattern<LinspaceOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(LinspaceOp op) const override
    {
      mlir::Type beginType = op.getBegin().getType();
      mlir::Type endType = op.getEnd().getType();
      mlir::Type amountType = op.getAmount().getType();

      return mlir::LogicalResult::success(
          beginType.isa<RealType>() &&
          endType.isa<RealType>() &&
          amountType.isa<mlir::IndexType>());
    }

    void rewrite(
        LinspaceOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 3> newOperands;

      // Result.
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 1);
      llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          dynamicDimensions.push_back(adaptor.getAmount());
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      result = materializeTargetConversion(rewriter, loc, result);
      result = convertToUnrankedMemRef(rewriter, loc, result);

      newOperands.push_back(result);

      // Begin value.
      newOperands.push_back(adaptor.getBegin());

      // End value.
      newOperands.push_back(adaptor.getEnd());

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("linspace", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.create<CallOp>(loc, callee, newOperands);
    }
  };

  struct LogOpCastPattern : public ModelicaOpRewritePattern<LogOp>
  {
    using ModelicaOpRewritePattern<LogOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(LogOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(LogOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<LogOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct LogOpLowering : public RuntimeOpConversionPattern<LogOp>
  {
    using RuntimeOpConversionPattern<LogOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(LogOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        LogOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("log", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct Log10OpCastPattern : public ModelicaOpRewritePattern<Log10Op>
  {
    using ModelicaOpRewritePattern<Log10Op>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(Log10Op op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(Log10Op op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<Log10Op>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct Log10OpLowering : public RuntimeOpConversionPattern<Log10Op>
  {
    using RuntimeOpConversionPattern<Log10Op>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(Log10Op op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        Log10Op op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("log10", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct OnesOpCastPattern : public ModelicaOpRewritePattern<OnesOp>
  {
    using ModelicaOpRewritePattern<OnesOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(OnesOp op) const override
    {
      return mlir::LogicalResult::success(
          !llvm::all_of(op.getSizes(), [](mlir::Value dimension) {
            return dimension.getType().isa<mlir::IndexType>();
          }));
    }

    void rewrite(OnesOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 3> dimensions;

      for (const auto& dimension : op.getSizes()) {
        if (dimension.getType().isa<mlir::IndexType>()) {
          dimensions.push_back(dimension);
        } else {
          dimensions.push_back(rewriter.create<CastOp>(loc, rewriter.getIndexType(), dimension));
        }
      }

      rewriter.replaceOpWithNewOp<OnesOp>(op, op.getResult().getType(), dimensions);
    }
  };

  struct OnesOpLowering : public RuntimeOpConversionPattern<OnesOp>
  {
    using RuntimeOpConversionPattern<OnesOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(OnesOp op) const override
    {
      return mlir::LogicalResult::success(
          llvm::all_of(op.getSizes(), [](mlir::Value dimension) {
            return dimension.getType().isa<mlir::IndexType>();
          }));
    }

    void rewrite(
        OnesOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Result.
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          auto index = dynamicDimensions.size();
          dynamicDimensions.push_back(adaptor.getSizes()[index]);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      result = materializeTargetConversion(rewriter, loc, result);
      result = convertToUnrankedMemRef(rewriter, loc, result);

      newOperands.push_back(result);

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("ones", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.create<CallOp>(loc, callee, newOperands);
    }
  };

  struct MaxOpArrayCastPattern : public ModelicaOpRewritePattern<MaxOp>
  {
    using ModelicaOpRewritePattern<MaxOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(MaxOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto elementType = op.getFirst().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<MaxOp>(loc, elementType, op.getFirst());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MaxOpArrayLowering : public RuntimeOpConversionPattern<MaxOp>
  {
    using RuntimeOpConversionPattern<MaxOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          arrayType.getElementType() == resultType);
    }

    void rewrite(
        MaxOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      assert(op.getFirst().getType().isa<ArrayType>());

      // Array.
      mlir::Value array = adaptor.getFirst();
      array = convertToUnrankedMemRef(rewriter, loc, array);

      newOperands.push_back(array);

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("maxArray", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct MaxOpScalarsCastPattern : public ModelicaOpRewritePattern<MaxOp>
  {
    using ModelicaOpRewritePattern<MaxOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType != secondValueType || firstValueType != resultType);
    }

    void rewrite(MaxOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getFirst(), op.getSecond() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<MaxOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MaxOpScalarsLowering : public RuntimeOpConversionPattern<MaxOp>
  {
    using RuntimeOpConversionPattern<MaxOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType == secondValueType && firstValueType == resultType);
    }

    void rewrite(
        MaxOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;

      assert(op.getFirst().getType() == op.getSecond().getType());

      // First value.
      newOperands.push_back(adaptor.getFirst());

      // Second value.
      newOperands.push_back(adaptor.getSecond());

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("maxScalars", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct MinOpArrayCastPattern : public ModelicaOpRewritePattern<MinOp>
  {
    using ModelicaOpRewritePattern<MinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(MinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto elementType = op.getFirst().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<MinOp>(loc, elementType, op.getFirst());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MinOpArrayLowering : public RuntimeOpConversionPattern<MinOp>
  {
    using RuntimeOpConversionPattern<MinOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() == resultType);
    }

    void rewrite(
        MinOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      assert(op.getFirst().getType().isa<ArrayType>());

      // Array.
      mlir::Value array = adaptor.getFirst();
      array = convertToUnrankedMemRef(rewriter, loc, array);

      newOperands.push_back(array);

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("minArray", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct MinOpScalarsCastPattern : public ModelicaOpRewritePattern<MinOp>
  {
    using ModelicaOpRewritePattern<MinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType != secondValueType || firstValueType != resultType);
    }

    void rewrite(MinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getFirst(), op.getSecond() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<MinOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MinOpScalarsLowering : public RuntimeOpConversionPattern<MinOp>
  {
    using RuntimeOpConversionPattern<MinOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType == secondValueType && firstValueType == resultType);
    }

    void rewrite(
        MinOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;

      assert(op.getFirst().getType() == op.getSecond().getType());

      // First value.
      newOperands.push_back(adaptor.getFirst());

      // Second value.
      newOperands.push_back(adaptor.getSecond());

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("minScalars", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct ModOpCastPattern : public ModelicaOpRewritePattern<ModOp>
  {
    using ModelicaOpRewritePattern<ModOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(ModOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType != yType || xType != resultType);
    }

    void rewrite(ModOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getX(), op.getY() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<ModOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct ModOpLowering : public RuntimeOpConversionPattern<ModOp>
  {
    using RuntimeOpConversionPattern<ModOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(ModOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType == yType && xType == resultType);
    }

    void rewrite(
        ModOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;

      assert(op.getX().getType() == op.getY().getType());

      // Dividend.
      newOperands.push_back(adaptor.getX());

      // Divisor.
      newOperands.push_back(adaptor.getY());

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("mod", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct ProductOpCastPattern : public ModelicaOpRewritePattern<ProductOp>
  {
    using ModelicaOpRewritePattern<ProductOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(ProductOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(ProductOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto resultType = op.getArray().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<ProductOp>(loc, resultType, op.getArray());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct ProductOpLowering : public RuntimeOpConversionPattern<ProductOp>
  {
    using RuntimeOpConversionPattern<ProductOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(ProductOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          arrayType.getElementType() == resultType);
    }

    void rewrite(
        ProductOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Array.
      mlir::Value array = adaptor.getArray();
      array = convertToUnrankedMemRef(rewriter, loc, array);

      newOperands.push_back(array);

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("product", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct RemOpCastPattern : public ModelicaOpRewritePattern<RemOp>
  {
    using ModelicaOpRewritePattern<RemOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(RemOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType != yType || xType != resultType);
    }

    void rewrite(RemOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getX(), op.getY() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<RemOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct RemOpLowering : public RuntimeOpConversionPattern<RemOp>
  {
    using RuntimeOpConversionPattern<RemOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(RemOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType == yType && xType == resultType);
    }

    void rewrite(
        RemOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;

      assert(op.getX().getType() == op.getY().getType());

      // 'x' value.
      newOperands.push_back(adaptor.getX());

      // 'y' value.
      newOperands.push_back(adaptor.getY());

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("rem", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct SignOpCastPattern : public ModelicaOpRewritePattern<SignOp>
  {
    using ModelicaOpRewritePattern<SignOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SignOp op) const override
    {
      mlir::Type resultType = op.getResult().getType();
      return mlir::LogicalResult::success(!resultType.isa<IntegerType>());
    }

    void rewrite(SignOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto integerType = IntegerType::get(op.getContext());
      mlir::Value result = rewriter.create<SignOp>(loc, integerType, op.getOperand());

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct SignOpLowering : public RuntimeOpConversionPattern<SignOp>
  {
    using RuntimeOpConversionPattern<SignOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(SignOp op) const override
    {
      mlir::Type resultType = op.getResult().getType();
      return mlir::LogicalResult::success(resultType.isa<IntegerType>());
    }

    void rewrite(
        SignOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<IntegerType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sign", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct SinOpCastPattern : public ModelicaOpRewritePattern<SinOp>
  {
    using ModelicaOpRewritePattern<SinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(SinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<SinOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SinOpLowering : public RuntimeOpConversionPattern<SinOp>
  {
    using RuntimeOpConversionPattern<SinOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(SinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        SinOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sin", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct SinhOpCastPattern : public ModelicaOpRewritePattern<SinhOp>
  {
    using ModelicaOpRewritePattern<SinhOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SinhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(SinhOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<SinhOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SinhOpLowering : public RuntimeOpConversionPattern<SinhOp>
  {
    using RuntimeOpConversionPattern<SinhOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(SinhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        SinhOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sinh", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct SqrtOpCastPattern : public ModelicaOpRewritePattern<SqrtOp>
  {
    using ModelicaOpRewritePattern<SqrtOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SqrtOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(SqrtOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<SqrtOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SqrtOpLowering : public RuntimeOpConversionPattern<SqrtOp>
  {
    using RuntimeOpConversionPattern<SqrtOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(SqrtOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        SqrtOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sqrt", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct SumOpCastPattern : public ModelicaOpRewritePattern<SumOp>
  {
    using ModelicaOpRewritePattern<SumOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SumOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(SumOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto resultType = op.getArray().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<SumOp>(loc, resultType, op.getArray());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SumOpLowering : public RuntimeOpConversionPattern<SumOp>
  {
    using RuntimeOpConversionPattern<SumOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(SumOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          arrayType.getElementType() == resultType);
    }

    void rewrite(
        SumOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Array.
      mlir::Value array = adaptor.getArray();
      array = convertToUnrankedMemRef(rewriter, loc, array);

      newOperands.push_back(array);

      // Create the call to the runtime library.
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("sum", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  class SymmetricOpLowering : public RuntimeOpConversionPattern<SymmetricOp>
  {
    public:
      SymmetricOpLowering(
        mlir::TypeConverter& typeConverter,
        mlir::MLIRContext* context,
        bool assertions)
          : RuntimeOpConversionPattern(typeConverter, context),
            assertions(assertions)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          SymmetricOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        mlir::Value sourceMatrixValue = nullptr;

        auto sourceMatrixFn = [&]() -> mlir::Value {
          if (sourceMatrixValue == nullptr) {
            sourceMatrixValue =
                getTypeConverter()->materializeSourceConversion(
                    rewriter, op.getMatrix().getLoc(),
                    op.getMatrix().getType(),
                    adaptor.getMatrix());
          }

          return sourceMatrixValue;
        };

        if (assertions) {
          // Check if the matrix is a square one.
          if (!op.getMatrix().getType().cast<ArrayType>().hasStaticShape()) {
            mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(0));

            mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(1));

            mlir::Value lhsDimensionSize = materializeTargetConversion(
                rewriter, loc,
                rewriter.create<DimOp>(loc, sourceMatrixFn(), one));

            mlir::Value rhsDimensionSize = materializeTargetConversion(
                rewriter, loc,
                rewriter.create<DimOp>(loc, sourceMatrixFn(), zero));

            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq,
                lhsDimensionSize, rhsDimensionSize);

            rewriter.create<mlir::cf::AssertOp>(
                loc, condition,
                rewriter.getStringAttr("Base matrix is not squared"));
          }
        }

        llvm::SmallVector<mlir::Value, 2> newOperands;

        // Result.
        auto resultType = op.getResult().getType().cast<ArrayType>();
        assert(resultType.getRank() == 2);
        llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

        if (!resultType.hasStaticShape()) {
          for (const auto& dimension :
               llvm::enumerate(resultType.getShape())) {
            if (dimension.value() == ArrayType::kDynamicSize) {
              mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
                  loc, rewriter.getIndexAttr(dimension.index()));

              dynamicDimensions.push_back(
                  rewriter.create<DimOp>(loc, sourceMatrixFn(), one));
            }
          }
        }

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultType, dynamicDimensions);

        result = materializeTargetConversion(rewriter, loc, result);
        result = convertToUnrankedMemRef(rewriter, loc, result);

        newOperands.push_back(result);

        // Matrix.
        mlir::Value matrix = adaptor.getMatrix();
        matrix = convertToUnrankedMemRef(rewriter, loc, matrix);

        newOperands.push_back(matrix);

        // Create the call to the runtime library.
        auto callee = getOrDeclareRuntimeFunction(
            rewriter,
            op->getParentOfType<mlir::ModuleOp>(),
            getMangledFunctionName("symmetric", llvm::None, newOperands),
            llvm::None, newOperands);

        rewriter.create<CallOp>(loc, callee, newOperands);
        return mlir::success();
      }

    private:
      bool assertions;
  };

  struct TanOpCastPattern : public ModelicaOpRewritePattern<TanOp>
  {
    using ModelicaOpRewritePattern<TanOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(TanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(TanOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<TanOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct TanOpLowering : public RuntimeOpConversionPattern<TanOp>
  {
    using RuntimeOpConversionPattern<TanOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(TanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        TanOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("tan", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct TanhOpCastPattern : public ModelicaOpRewritePattern<TanhOp>
  {
    using ModelicaOpRewritePattern<TanhOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(TanhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(TanhOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<TanhOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct TanhOpLowering : public RuntimeOpConversionPattern<TanhOp>
  {
    using RuntimeOpConversionPattern<TanhOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(TanhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(
        TanhOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());

      // Create the call to the runtime library.
      assert(op.getResult().getType().isa<RealType>());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("tanh", resultType, newOperands),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
    }
  };

  struct TransposeOpLowering : public RuntimeOpConversionPattern<TransposeOp>
  {
    using RuntimeOpConversionPattern<TransposeOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        TransposeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> newOperands;

      // Result.
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      if (!resultType.hasStaticShape()) {
        mlir::Value sourceMatrix =
            getTypeConverter()->materializeSourceConversion(
                rewriter, op.getMatrix().getLoc(),
                op.getMatrix().getType(), adaptor.getMatrix());

        if (resultType.getShape()[0] == ArrayType::kDynamicSize) {
          mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getIndexAttr(1));

          dynamicDimensions.push_back(
              rewriter.create<DimOp>(loc, sourceMatrix, one));
        }

        if (resultType.getShape()[1] == ArrayType::kDynamicSize) {
          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getIndexAttr(0));

          dynamicDimensions.push_back(rewriter.create<DimOp>(
              loc, sourceMatrix, zero));
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultType, dynamicDimensions);

      result = materializeTargetConversion(rewriter, loc, result);
      result = convertToUnrankedMemRef(rewriter, loc, result);

      newOperands.push_back(result);

      // Matrix.
      assert(op.getMatrix().getType().isa<ArrayType>());

      mlir::Value matrix = adaptor.getMatrix();
      matrix = convertToUnrankedMemRef(rewriter, loc, matrix);

      newOperands.push_back(matrix);

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("transpose", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.create<CallOp>(loc, callee, newOperands);
      return mlir::success();
    }
  };

  struct ZerosOpCastPattern : public ModelicaOpRewritePattern<ZerosOp>
  {
    using ModelicaOpRewritePattern<ZerosOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(ZerosOp op) const override
    {
      return mlir::LogicalResult::success(
          !llvm::all_of(op.getSizes(), [](mlir::Value dimension) {
            return dimension.getType().isa<mlir::IndexType>();
          }));
    }

    void rewrite(ZerosOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 3> dimensions;

      for (const auto& dimension : op.getSizes()) {
        if (dimension.getType().isa<mlir::IndexType>()) {
          dimensions.push_back(dimension);
        } else {
          dimensions.push_back(rewriter.create<CastOp>(loc, rewriter.getIndexType(), dimension));
        }
      }

      rewriter.replaceOpWithNewOp<ZerosOp>(op, op.getResult().getType(), dimensions);
    }
  };

  struct ZerosOpLowering : public RuntimeOpConversionPattern<ZerosOp>
  {
    using RuntimeOpConversionPattern<ZerosOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult match(ZerosOp op) const override
    {
      return mlir::LogicalResult::success(
          llvm::all_of(op.getSizes(), [](mlir::Value dimension) {
            return dimension.getType().isa<mlir::IndexType>();
          }));
    }

    void rewrite(
        ZerosOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Result.
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          auto index = dynamicDimensions.size();
          dynamicDimensions.push_back(adaptor.getSizes()[index]);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultType, dynamicDimensions);

      result = materializeTargetConversion(rewriter, loc, result);
      result = convertToUnrankedMemRef(rewriter, loc, result);

      newOperands.push_back(result);

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("zeros", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.create<CallOp>(loc, callee, newOperands);
    }
  };
}

//===----------------------------------------------------------------------===//
// Utility operations
//===----------------------------------------------------------------------===//

namespace
{
  struct ArrayFillOpLowering : public RuntimeOpConversionPattern<ArrayFillOp>
  {
    using RuntimeOpConversionPattern<ArrayFillOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        ArrayFillOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Array.
      mlir::Value array = adaptor.getArray();
      array = convertToUnrankedMemRef(rewriter, loc, array);

      newOperands.push_back(array);

      // Value.
      newOperands.push_back(adaptor.getValue());

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("fill", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.replaceOpWithNewOp<CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };

  struct PrintOpLowering : public RuntimeOpConversionPattern<PrintOp>
  {
    using RuntimeOpConversionPattern<PrintOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        PrintOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;

      // Operand.
      if (op.getValue().getType().isa<ArrayType>()) {
        mlir::Value array = adaptor.getValue();
        array = convertToUnrankedMemRef(rewriter, loc, array);
        newOperands.push_back(array);
      } else {
        newOperands.push_back(adaptor.getValue());
      }

      // Create the call to the runtime library.
      auto callee = getOrDeclareRuntimeFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangledFunctionName("print", llvm::None, newOperands),
          llvm::None, newOperands);

      rewriter.create<CallOp>(loc, callee, newOperands);
      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
}

static void populateModelicaToFuncPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context,
    mlir::TypeConverter& typeConverter,
    bool assertions)
{
  // Func operations
  patterns.insert<
      RawFunctionOpLowering,
      RawReturnOpLowering,
      CallOpLowering>(typeConverter, context);

  // Math operations
  patterns.insert<
      PowOpLowering>(typeConverter, context, assertions);

  // Built-in functions
  patterns.insert<
      AbsOpCastPattern,
      AcosOpCastPattern,
      AsinOpCastPattern,
      AtanOpCastPattern,
      Atan2OpCastPattern,
      CeilOpCastPattern,
      CosOpCastPattern,
      CoshOpCastPattern,
      DivTruncOpCastPattern,
      ExpOpCastPattern,
      FloorOpCastPattern,
      IdentityOpCastPattern,
      IntegerOpCastPattern,
      LinspaceOpCastPattern,
      LogOpCastPattern,
      Log10OpCastPattern,
      MaxOpArrayCastPattern,
      MaxOpScalarsCastPattern,
      MinOpArrayCastPattern,
      MinOpScalarsCastPattern,
      ModOpCastPattern,
      OnesOpCastPattern,
      ProductOpCastPattern,
      RemOpCastPattern,
      SignOpCastPattern,
      SinOpCastPattern,
      SinhOpCastPattern,
      SqrtOpCastPattern,
      SumOpCastPattern,
      TanOpCastPattern,
      TanhOpCastPattern,
      ZerosOpCastPattern>(context);

  patterns.insert<
      AbsOpLowering,
      AcosOpLowering,
      AsinOpLowering,
      AtanOpLowering,
      Atan2OpLowering,
      CeilOpLowering,
      CosOpLowering,
      CoshOpLowering,
      DiagonalOpLowering,
      DivTruncOpLowering,
      ExpOpLowering,
      FloorOpLowering,
      IdentityOpLowering,
      IntegerOpLowering,
      LinspaceOpLowering,
      LogOpLowering,
      Log10OpLowering,
      OnesOpLowering,
      MaxOpArrayLowering,
      MaxOpScalarsLowering,
      MinOpArrayLowering,
      MinOpScalarsLowering,
      ModOpLowering,
      ProductOpLowering,
      RemOpLowering,
      SignOpLowering,
      SignOpLowering,
      SinOpLowering,
      SinhOpLowering,
      SqrtOpLowering,
      SumOpLowering>(typeConverter, context);

  patterns.insert<
      SymmetricOpLowering>(typeConverter, context, assertions);

  patterns.insert<
      TanOpLowering,
      TanhOpLowering,
      TransposeOpLowering,
      ZerosOpLowering>(typeConverter, context);

  // Utility operations
  patterns.insert<
      ArrayFillOpLowering,
      PrintOpLowering>(typeConverter, context);
}

namespace
{
  class ModelicaToFuncConversionPass : public mlir::impl::ModelicaToFuncConversionPassBase<ModelicaToFuncConversionPass>
  {
    public:
      using ModelicaToFuncConversionPassBase::ModelicaToFuncConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica operations");
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect<mlir::BuiltinDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::cf::ControlFlowDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        target.addLegalDialect<mlir::memref::MemRefDialect>();

        target.addLegalDialect<ModelicaDialect>();
        target.addIllegalOp<RawFunctionOp, RawReturnOp>();

        target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
          auto module = op->getParentOfType<mlir::ModuleOp>();
          auto callee = module.lookupSymbol(op.getCallee());

          return callee && mlir::isa<RuntimeFunctionOp>(callee);
        });

        target.addDynamicallyLegalOp<PowOp>([](PowOp op) {
          return !isNumeric(op.getBase());
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
            ProductOp,
            RemOp,
            SignOp,
            SinOp,
            SinhOp,
            SqrtOp,
            SumOp,
            SymmetricOp,
            TanOp,
            TanhOp,
            TransposeOp,
            ZerosOp>();

        target.addIllegalOp<
            ArrayFillOp,
            PrintOp>();

        mlir::modelica::TypeConverter typeConverter(bitWidth);

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToFuncPatterns(patterns, &getContext(), typeConverter, assertions);

        return applyPartialConversion(module, target, std::move(patterns));
      }
    };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToFuncConversionPass()
  {
    return std::make_unique<ModelicaToFuncConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToFuncConversionPass(const ModelicaToFuncConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToFuncConversionPass>(options);
  }
}
