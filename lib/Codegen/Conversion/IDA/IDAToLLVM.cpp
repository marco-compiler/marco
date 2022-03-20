#include "marco/Codegen/Conversion/IDA/IDAToLLVM.h"
#include "marco/Codegen/Conversion/IDA/TypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/ArrayDescriptor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "llvm/ADT/Optional.h"

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

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
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

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 3> newOperands;
    newOperands.push_back(operands[0]);

    // Create the array with the equation ranges
    mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
    mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(op.equationRanges().size() * 2));
    mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
    mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
    mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::ArrayRef<mlir::Value>{ nullPtr, numOfElements });
    mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

    auto heapAllocFn = lookupOrCreateHeapAllocFn(op->getParentOfType<mlir::ModuleOp>(), getIndexType());
    mlir::Value equationRangesPtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
    equationRangesPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, equationRangesPtr);
    newOperands.push_back(equationRangesPtr);

    // Populate the equation ranges
    for (const auto& range : llvm::enumerate(op.equationRanges())) {
      auto rangeAttr = range.value().cast<mlir::ArrayAttr>();
      assert(rangeAttr.size() == 2);

      for (const auto& index : llvm::enumerate(rangeAttr)) {
        auto indexAttr = index.value().cast<mlir::IntegerAttr>();
        mlir::Value offset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(range.index() * 2 + index.index()));
        mlir::Value indexValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(indexAttr.getInt()));
        mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, equationRangesPtr.getType(), equationRangesPtr, offset);
        rewriter.create<mlir::LLVM::StoreOp>(loc, indexValue, ptr);
      }
    }

    // Rank
    mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.equationRanges().size()));
    newOperands.push_back(rank);

    // Mangled types
    auto mangledResultType = mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

    llvm::SmallVector<std::string, 3> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());
    mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(64)));
    mangledArgsTypes.push_back(mangling.getIntegerType(64));

    // Create the call to the runtime library
    auto functionName = mangling.getMangledFunction(
        "idaAddEquation", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, resultType, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, callee.getName(), resultType, newOperands);

    return mlir::success();
  }
};

struct AddVariableOpLowering : public IDAOpConversion<AddVariableOp>
{
  using IDAOpConversion<AddVariableOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(AddVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 4> newOperands;
    newOperands.push_back(operands[0]);

    // Create the array with the equation ranges
    mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
    mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(op.arrayDimensions().size()));
    mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
    mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
    mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::ArrayRef<mlir::Value>{ nullPtr, numOfElements });
    mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

    auto heapAllocFn = lookupOrCreateHeapAllocFn(op->getParentOfType<mlir::ModuleOp>(), getIndexType());
    mlir::Value arrayDimensionsPtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
    arrayDimensionsPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, arrayDimensionsPtr);
    newOperands.push_back(arrayDimensionsPtr);

    // Populate the equation ranges
    for (const auto& sizeAttr : llvm::enumerate(op.arrayDimensions())) {
      mlir::Value offset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(sizeAttr.index()));
      mlir::Value size = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(sizeAttr.value().cast<mlir::IntegerAttr>().getInt()));
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, arrayDimensionsPtr.getType(), arrayDimensionsPtr, offset);
      rewriter.create<mlir::LLVM::StoreOp>(loc, size, ptr);
    }

    // Rank
    mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.arrayDimensions().size()));
    newOperands.push_back(rank);

    // State variable property
    mlir::Value isState = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(op.state()));
    newOperands.push_back(isState);

    // Mangled types
    auto mangledResultType = mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

    llvm::SmallVector<std::string, 3> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());
    mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(64)));
    mangledArgsTypes.push_back(mangling.getIntegerType(64));
    mangledArgsTypes.push_back(mangling.getIntegerType(1));

    // Create the call to the runtime library
    auto functionName = mangling.getMangledFunction(
        "idaAddVariable", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, resultType, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, callee.getName(), resultType, newOperands);

    return mlir::success();
  }
};

struct AddVariableAccessOpLowering : public IDAOpConversion<AddVariableAccessOp>
{
  using IDAOpConversion<AddVariableAccessOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(AddVariableAccessOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    assert(operands.size() == 3);
    llvm::SmallVector<mlir::Value, 5> newOperands;
    newOperands.push_back(operands[0]);
    newOperands.push_back(operands[1]);
    newOperands.push_back(operands[2]);

    // Create the array with the variable accesses
    auto dimensionAccesses = op.access().getResults();
    llvm::SmallVector<mlir::Value, 6> accessValues;

    for (const auto dimensionAccess : dimensionAccesses) {
      if (dimensionAccess.isa<mlir::AffineConstantExpr>()) {
        auto constantAccess = dimensionAccess.cast<mlir::AffineConstantExpr>();
        accessValues.push_back(rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(-1)));
        accessValues.push_back(rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(constantAccess.getValue())));
      } else {
        auto dynamicAccess = dimensionAccess.cast<mlir::AffineBinaryOpExpr>();
        auto dimension = dynamicAccess.getLHS().cast<mlir::AffineDimExpr>();
        auto offset = dynamicAccess.getRHS().cast<mlir::AffineConstantExpr>();
        accessValues.push_back(rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(dimension.getPosition())));
        accessValues.push_back(rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(offset.getValue())));
      }
    }

    mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
    mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(accessValues.size()));
    mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
    mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
    mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::ArrayRef<mlir::Value>{ nullPtr, numOfElements });
    mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

    auto heapAllocFn = lookupOrCreateHeapAllocFn(op->getParentOfType<mlir::ModuleOp>(), getIndexType());
    mlir::Value accessesPtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
    accessesPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, accessesPtr);
    newOperands.push_back(accessesPtr);

    // Populate the equation ranges
    for (const auto& accessValue : llvm::enumerate(accessValues)) {
      mlir::Value offset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(accessValue.index()));
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, accessesPtr.getType(), accessesPtr, offset);
      rewriter.create<mlir::LLVM::StoreOp>(loc, accessValue.value(), ptr);
    }

    // Rank
    mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.access().getResults().size()));
    newOperands.push_back(rank);

    // Mangled types
    auto mangledResultType = mangling.getVoidType();

    llvm::SmallVector<std::string, 3> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());
    mangledArgsTypes.push_back(mangling.getIntegerType(64));
    mangledArgsTypes.push_back(mangling.getIntegerType(64));
    mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(64)));
    mangledArgsTypes.push_back(mangling.getIntegerType(64));

    // Create the call to the runtime library
    auto functionName = mangling.getMangledFunction(
        "idaAddVariableAccess", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, llvm::None, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, callee.getName(), llvm::None, newOperands);

    return mlir::success();
  }
};

struct ReturnOpLowering : public IDAOpConversion<ReturnOp>
{
  using IDAOpConversion<ReturnOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(ReturnOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, operands);
    return mlir::success();
  }
};

struct InitOpLowering : public IDAOpConversion<InitOp>
{
  using IDAOpConversion<InitOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(InitOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    RuntimeFunctionsMangling mangling;

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto mangledResultType = mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

    assert(operands.size() == 1);
    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    auto functionName = mangling.getMangledFunction("idaInit", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, resultType, operands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), resultType, operands);
    return mlir::success();
  }
};

struct StepOpLowering : public IDAOpConversion<StepOp>
{
  using IDAOpConversion<StepOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(StepOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto resultType = rewriter.getI1Type();
    auto mangledResultType = mangling.getIntegerType(1);

    assert(operands.size() == 1);
    llvm::SmallVector<mlir::Value, 2> newOperands;
    newOperands.push_back(operands[0]);

    llvm::SmallVector<std::string, 2> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    if (auto timeStep = op.timeStep(); timeStep.hasValue()) {
      newOperands.push_back(rewriter.create<mlir::ConstantOp>(loc, rewriter.getF64FloatAttr(timeStep->convertToDouble())));
      mangledArgsTypes.push_back(mangling.getFloatingPointType(64));
    }

    auto functionName = mangling.getMangledFunction("idaStep", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, resultType, newOperands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), resultType, newOperands);
    return mlir::success();
  }
};

struct FreeOpLowering : public IDAOpConversion<FreeOp>
{
  using IDAOpConversion<FreeOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(FreeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto resultType = rewriter.getI1Type();
    auto mangledResultType = mangling.getIntegerType(1);

    assert(operands.size() == 1);
    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    auto functionName = mangling.getMangledFunction("idaFree", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, resultType, operands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), resultType, operands);
    return mlir::success();
  }
};

struct PrintStatisticsOpLowering : public IDAOpConversion<PrintStatisticsOp>
{
  using IDAOpConversion<PrintStatisticsOp>::IDAOpConversion;

  mlir::LogicalResult matchAndRewrite(PrintStatisticsOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op.getLoc();
    RuntimeFunctionsMangling mangling;

    auto mangledResultType = mangling.getVoidType();

    assert(operands.size() == 1);
    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    auto functionName = mangling.getMangledFunction("idaPrintStatistics", mangledResultType, mangledArgsTypes);

    auto callee = getOrDeclareFunction(
        rewriter,
        op->getParentOfType<mlir::ModuleOp>(),
        functionName, llvm::None, operands);

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, callee.getName(), llvm::None, operands);
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
      GetCurrentTimeOpLowering,
      AddVariableAccessOpLowering,
      ReturnOpLowering,
      InitOpLowering,
      StepOpLowering,
      FreeOpLowering,
      PrintStatisticsOpLowering>(typeConverter);
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
  template<typename Op>
  struct GetVariableLikeOpLowering : public IDAOpConversion<Op>
  {
    using IDAOpConversion<Op>::IDAOpConversion;

    virtual std::string getRuntimeFunctionName() const = 0;

    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto arrayType = op.getResult().getType().template cast<mlir::modelica::ArrayType>();
      auto arrayDescriptorType = this->getTypeConverter()->convertType(arrayType).template cast<mlir::LLVM::LLVMStructType>();
      auto dataPtrType = arrayDescriptorType.getBody()[0];

      auto loc = op.getLoc();
      RuntimeFunctionsMangling mangling;

      auto resultType = this->getTypeConverter()->convertType(op.getResult().getType());

      assert(operands.size() == 2);
      llvm::SmallVector<mlir::Value, 2> newOperands;
      newOperands.push_back(operands[0]);
      newOperands.push_back(operands[1]);

      // Mangled types
      auto mangledResultType = mangling.getVoidPointerType();

      llvm::SmallVector<std::string, 3> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getVoidPointerType());
      mangledArgsTypes.push_back(mangling.getIntegerType(64));

      // Create the call to the runtime library
      auto functionName = mangling.getMangledFunction(
          getRuntimeFunctionName(), mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->template getParentOfType<mlir::ModuleOp>(),
          functionName, resultType, newOperands);

      mlir::Value ptr = rewriter.create<mlir::CallOp>(loc, callee.getName(), resultType, newOperands).getResult(0);

      // Create the array descriptor
      auto arrayDescriptor = ArrayDescriptor::undef(rewriter, &this->typeConverter(), loc, arrayDescriptorType);
      arrayDescriptor.setPtr(rewriter, loc, ptr);

      mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, arrayDescriptor.getRankType(), rewriter.getIndexAttr(arrayType.getRank()));
      arrayDescriptor.setRank(rewriter, loc, rank);

      assert(arrayType.hasConstantShape());

      for (auto size : llvm::enumerate(arrayType.getShape())) {
        assert(size.value() != -1);
        mlir::Value sizeValue = rewriter.create<mlir::ConstantOp>(loc, arrayDescriptor.getSizeType(), rewriter.getIndexAttr(size.value()));
        arrayDescriptor.setSize(rewriter, loc, size.index(), sizeValue);
      }

      rewriter.replaceOp(op, *arrayDescriptor);
      return mlir::success();
    }
  };

  struct GetVariableOpLowering : public GetVariableLikeOpLowering<GetVariableOp>
  {
    using GetVariableLikeOpLowering<GetVariableOp>::GetVariableLikeOpLowering;

    std::string getRuntimeFunctionName() const
    {
      return "idaGetVariable";
    }
  };

  struct GetDerivativeOpLowering : public GetVariableLikeOpLowering<GetDerivativeOp>
  {
    using GetVariableLikeOpLowering<GetDerivativeOp>::GetVariableLikeOpLowering;

    std::string getRuntimeFunctionName() const
    {
      return "idaGetVariable";
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
      mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns, mlir::ConversionTarget& target)
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
        GetVariableOpLowering,
        GetDerivativeOpLowering>(typeConverter);

    patterns.add<ConvertGetCurrentTimeOpTypes>(typeConverter, patterns.getContext());

    target.addDynamicallyLegalOp<GetVariableOp>([&](mlir::Operation *op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<GetCurrentTimeOp>([&](mlir::Operation *op) {
      return typeConverter.isLegal(op);
    });
  }
}
