#include "marco/Codegen/Conversion/IDA/IDAToLLVM.h"
#include "marco/Codegen/Conversion/IDA/TypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::ida;

/// Generic conversion pattern that provides some utility functions.
template<typename FromOp>
class IDAOpConversion : public mlir::ConvertOpToLLVMPattern<FromOp>
{
  public:
    using Adaptor = typename FromOp::Adaptor;
    using mlir::ConvertOpToLLVMPattern<FromOp>::ConvertOpToLLVMPattern;

    mlir::ida::TypeConverter& typeConverter() const
    {
      return *static_cast<mlir::ida::TypeConverter*>(this->getTypeConverter());
    }

    mlir::Type convertType(mlir::Type type) const
    {
      return typeConverter().convertType(type);
    }

    mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
    {
      mlir::Type type = this->getTypeConverter()->convertType(value.getType());
      return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
    }

    void materializeTargetConversion(mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Value>& values) const
    {
      for (auto& value : values) {
        value = materializeTargetConversion(builder, value);
      }
    }
};

static mlir::FuncOp getOrDeclareFunction(
    mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::TypeRange args)
{
  if (auto funcOp = module.lookupSymbol<mlir::FuncOp>(name)) {
    return funcOp;
  }

  mlir::PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcOp = builder.create<mlir::FuncOp>(module.getLoc(), name, builder.getFunctionType(args, results));
  funcOp.setPrivate();

  return funcOp;
}

static mlir::FuncOp getOrDeclareFunction(
    mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::TypeRange results, mlir::ValueRange args)
{
  return getOrDeclareFunction(builder, module, name, results, args.getTypes());
}

struct CreateOpLowering : public IDAOpConversion<CreateOp>
{
  using IDAOpConversion<CreateOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(CreateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    mlir::Type voidPtrType = mlir::LLVM::LLVMPointerType::get(
        getTypeConverter()->convertType(rewriter.getIntegerType(8)));

    auto mangledResultType = mangling.getVoidPointerType();

    llvm::SmallVector<mlir::Value, 2> newOperands;

    // Scalar equations amount
    newOperands.push_back(rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(op.scalarEquations())));

    // Data bitwidth
    // TODO fetch from options
    newOperands.push_back(rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(64)));

    llvm::SmallVector<std::string, 2> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getIntegerType(newOperands[0].getType().getIntOrFloatBitWidth()));
    mangledArgsTypes.push_back(mangling.getIntegerType(newOperands[1].getType().getIntOrFloatBitWidth()));

    auto functionName = mangling.getMangledFunction("idaCreate", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, voidPtrType, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), voidPtrType, newOperands);
    return mlir::success();
  }
};

struct SetStartTimeOpLowering : public IDAOpConversion<SetStartTimeOp>
{
  using IDAOpConversion<SetStartTimeOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(SetStartTimeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto mangledResultType = mangling.getVoidType();

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 2> newOperands;
    newOperands.push_back(operands[0]);

    // Start time
    newOperands.push_back(rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getF64FloatAttr(op.time().convertToDouble())));

    llvm::SmallVector<std::string, 2> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());
    mangledArgsTypes.push_back(mangling.getFloatingPointType(newOperands[1].getType().getIntOrFloatBitWidth()));

    auto functionName = mangling.getMangledFunction("idaSetStartTime", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, llvm::None, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, newOperands);
    return mlir::success();
  }
};

struct SetEndTimeOpLowering : public IDAOpConversion<SetEndTimeOp>
{
  using IDAOpConversion<SetEndTimeOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(SetEndTimeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto mangledResultType = mangling.getVoidType();

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 2> newOperands;
    newOperands.push_back(operands[0]);

    // End time
    newOperands.push_back(rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getF64FloatAttr(op.time().convertToDouble())));

    llvm::SmallVector<std::string, 2> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());
    mangledArgsTypes.push_back(mangling.getFloatingPointType(newOperands[1].getType().getIntOrFloatBitWidth()));

    auto functionName = mangling.getMangledFunction("idaSetEndTime", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, llvm::None, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, newOperands);
    return mlir::success();
  }
};

struct SetRelativeToleranceOpLowering : public IDAOpConversion<SetRelativeToleranceOp>
{
  using IDAOpConversion<SetRelativeToleranceOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(SetRelativeToleranceOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto mangledResultType = mangling.getVoidType();

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 2> newOperands;
    newOperands.push_back(operands[0]);

    // Tolerance
    newOperands.push_back(rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getF64FloatAttr(op.tolerance().convertToDouble())));

    llvm::SmallVector<std::string, 2> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());
    mangledArgsTypes.push_back(mangling.getFloatingPointType(operands[1].getType().getIntOrFloatBitWidth()));

    auto functionName = mangling.getMangledFunction("idaSetRelativeTolerance", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, llvm::None, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, newOperands);
    return mlir::success();
  }
};

struct SetAbsoluteToleranceOpLowering : public IDAOpConversion<SetAbsoluteToleranceOp>
{
  using IDAOpConversion<SetAbsoluteToleranceOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(SetAbsoluteToleranceOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto mangledResultType = mangling.getVoidType();

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 2> newOperands;
    newOperands.push_back(operands[0]);

    // Tolerance
    newOperands.push_back(rewriter.create<mlir::ConstantOp>(
        loc, rewriter.getF64FloatAttr(op.tolerance().convertToDouble())));

    llvm::SmallVector<std::string, 2> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());
    mangledArgsTypes.push_back(mangling.getFloatingPointType(operands[1].getType().getIntOrFloatBitWidth()));

    auto functionName = mangling.getMangledFunction("idaSetAbsoluteTolerance", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, llvm::None, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, newOperands);
    return mlir::success();
  }
};

struct GetCurrentTimeOpLowering : public IDAOpConversion<GetCurrentTimeOp>
{
  using IDAOpConversion<GetCurrentTimeOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(GetCurrentTimeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    RuntimeFunctionsMangling mangling;

    auto resultType =  getTypeConverter()->convertType(op.getResult().getType());
    auto mangledResultType = mangling.getFloatingPointType(resultType.getIntOrFloatBitWidth());

    assert(operands.size() == 1);
    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    auto functionName = mangling.getMangledFunction("idaGetCurrentTime", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, resultType, operands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), resultType, operands);
    return mlir::success();
  }
};

struct AddEquationOpLowering : public IDAOpConversion<AddEquationOp>
{
  using IDAOpConversion<AddEquationOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(AddEquationOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto mangledResultType = mangling.getFloatingPointType(resultType.getIntOrFloatBitWidth());

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 2> newOperands;
    newOperands.push_back(operands[0]);

    // Create the array with the equation ranges
    mlir::Type indexType = getTypeConverter()->convertType(rewriter.getIndexType());

    mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(op.equationRanges().size() * 2));
    mlir::Value totalSize = rewriter.create<mlir::ConstantOp>(loc, indexType, rewriter.getIndexAttr(2));
    totalSize = rewriter.create<mlir::MulIOp>(loc, indexType, totalSize, numOfElements);

    mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(indexType);
    mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
    mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::ArrayRef<mlir::Value>{ nullPtr, totalSize });
    mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, indexType, gepPtr);

    auto heapAllocFn = lookupOrCreateHeapAllocFn(op->getParentOfType<mlir::ModuleOp>(), indexType);
    mlir::Value equationRangesPtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
    equationRangesPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, equationRangesPtr);

    // Populate the equation ranges
    size_t flatPosition = 0;

    for (const auto& range : op.equationRanges()) {
      auto rangeAttr = range.cast<mlir::ArrayAttr>();
      assert(rangeAttr.size() == 2);

      for (const auto& value : rangeAttr) {
        auto valueAttr = value.cast<mlir::IntegerAttr>();
        rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(valueAttr.getInt()));
      }
    }

    assert(operands.size() == 1);
    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    auto functionName = mangling.getMangledFunction("idaAddEquation", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, resultType, operands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, callee.getName(), resultType, newOperands);

    return mlir::success();
  }
};

static void populateIDAConversionPatterns(
    mlir::OwningRewritePatternList& patterns,
    mlir::ida::TypeConverter& typeConverter)
{
  patterns.insert<
      CreateOpLowering,
      SetStartTimeOpLowering,
      SetEndTimeOpLowering,
      SetRelativeToleranceOpLowering,
      SetAbsoluteToleranceOpLowering,
      GetCurrentTimeOpLowering>(typeConverter);
}

namespace marco::codegen
{
  class IDAConversionPass : public mlir::PassWrapper<IDAConversionPass, mlir::OperationPass<mlir::ModuleOp>>
  {
    public:
      void getDependentDialects(mlir::DialectRegistry& registry) const override
      {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
      }

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the IDA operations\n");
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addIllegalDialect<mlir::ida::IDADialect>();
        target.addIllegalDialect<mlir::StandardOpsDialect>();

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::OwningRewritePatternList patterns(&getContext());
        populateIDAConversionPatterns(patterns, typeConverter);
        mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

        if (auto status = applyPartialConversion(module, target, std::move(patterns)); mlir::failed(status)) {
          return status;
        }

        return mlir::success();
      }
  };
}

namespace
{
  //===----------------------------------------------------------------------===//
  // Patterns for structural type conversions for the IDA dialect operations.
  //===----------------------------------------------------------------------===//

  struct ConvertGetVariableOpTypes : public mlir::OpConversionPattern<GetVariableOp>
  {
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(GetVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<GetVariableOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(operands);

      for (auto result : newOp->getResults()) {
        result.setType(typeConverter->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct ConvertGetCurrentTimeOpTypes : public mlir::OpConversionPattern<GetCurrentTimeOp>
  {
    using OpConversionPattern::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(GetCurrentTimeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<GetCurrentTimeOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(operands);

      for (auto result : newOp->getResults()) {
        result.setType(typeConverter->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createIDAConversionPass()
  {
    return std::make_unique<IDAConversionPass>();
  }

  void populateIDAStructuralTypeConversionsAndLegality(
      mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns, mlir::ConversionTarget& target)
  {
    typeConverter.addConversion([&](InstanceType type) {
     return type;
    });

    typeConverter.addConversion([&](EquationType type) {
      return type;
    });

    typeConverter.addConversion([&](VariableType type) {
      return type;
    });

    patterns.add<
        ConvertGetVariableOpTypes,
        ConvertGetCurrentTimeOpTypes>(typeConverter, patterns.getContext());

    target.addDynamicallyLegalOp<GetVariableOp>([&](mlir::Operation *op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<GetCurrentTimeOp>([&](mlir::Operation *op) {
      return typeConverter.isLegal(op);
    });
  }
}
