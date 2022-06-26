#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/IDAToLLVM/TypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "marco/Codegen/Conversion/PassDetail.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::ida;

static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    mlir::Location loc,
    llvm::StringRef name,
    mlir::Type result,
    llvm::ArrayRef<mlir::Type> args)
{
  if (auto funcOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
    return funcOp;
  }

  mlir::PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, mlir::LLVM::LLVMFunctionType::get(result, args));
}

static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    mlir::Location loc,
    llvm::StringRef name,
    mlir::Type result,
    mlir::ValueRange args)
{
  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (auto type : args.getTypes()) {
    argsTypes.push_back(type);
  }

  return getOrDeclareFunction(builder, module, loc, name, result, argsTypes);
}

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class IDAOpConversion : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      IDAOpConversion(mlir::LLVMTypeConverter& typeConverter, unsigned int bitWidth)
          : mlir::ConvertOpToLLVMPattern<Op>(typeConverter),
            bitWidth(bitWidth)
      {
      }

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

    protected:
      unsigned int bitWidth;
  };

  struct CreateOpLowering : public IDAOpConversion<CreateOp>
  {
    using IDAOpConversion<CreateOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(CreateOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // Scalar equations amount
      mlir::Value scalarEquationsAmount = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.getScalarEquations()));
      newOperands.push_back(scalarEquationsAmount);
      mangledArgsTypes.push_back(mangling.getIntegerType(scalarEquationsAmount.getType().getIntOrFloatBitWidth()));

      // Data bit-width
      mlir::Value dataBitWidth = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(bitWidth));
      newOperands.push_back(dataBitWidth);
      mangledArgsTypes.push_back(mangling.getIntegerType(dataBitWidth.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library
      auto resultType = getVoidPtrType();
      auto mangledResultType = mangling.getVoidPointerType();
      auto functionName = mangling.getMangledFunction("idaCreate", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct SetStartTimeOpLowering : public IDAOpConversion<SetStartTimeOp>
  {
    using IDAOpConversion<SetStartTimeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(SetStartTimeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Start time
      mlir::Value startTime = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64FloatAttr(op.getTime().convertToDouble()));
      newOperands.push_back(startTime);
      mangledArgsTypes.push_back(mangling.getFloatingPointType(startTime.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaSetStartTime", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct SetEndTimeOpLowering : public IDAOpConversion<SetEndTimeOp>
  {
    using IDAOpConversion<SetEndTimeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(SetEndTimeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // End time
      mlir::Value endTime = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64FloatAttr(op.getTime().convertToDouble()));
      newOperands.push_back(endTime);
      mangledArgsTypes.push_back(mangling.getFloatingPointType(endTime.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaSetEndTime", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct SetTimeStepOpLowering : public IDAOpConversion<SetTimeStepOp>
  {
    using IDAOpConversion<SetTimeStepOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(SetTimeStepOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Time step
      mlir::Value timeStep = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64FloatAttr(op.getTimeStep().convertToDouble()));
      newOperands.push_back(timeStep);
      mangledArgsTypes.push_back(mangling.getFloatingPointType(timeStep.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaSetTimeStep", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct GetCurrentTimeOpLowering : public IDAOpConversion<GetCurrentTimeOp>
  {
    using IDAOpConversion<GetCurrentTimeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(GetCurrentTimeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = mangling.getFloatingPointType(resultType.getIntOrFloatBitWidth());
      auto functionName = mangling.getMangledFunction("idaGetCurrentTime", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, adaptor.getOperands());

      return mlir::success();
    }
  };

  struct AddEquationOpLowering : public IDAOpConversion<AddEquationOp>
  {
    using IDAOpConversion<AddEquationOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddEquationOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 3> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the array with the equation ranges
      mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
      mlir::Value numOfElements = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), op.getEquationRanges().size() * 2));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, nullPtr, numOfElements);
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      auto heapAllocFn = lookupOrCreateHeapAllocFn(module, getIndexType());
      mlir::Value equationRangesOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value equationRangesPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, equationRangesOpaquePtr);

      newOperands.push_back(equationRangesPtr);
      mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(dimensionSizeType.getIntOrFloatBitWidth())));

      // Populate the equation ranges
      for (const auto& range : llvm::enumerate(op.getEquationRanges())) {
        auto rangeAttr = range.value().cast<mlir::ArrayAttr>();
        assert(rangeAttr.size() == 2);

        for (const auto& index : llvm::enumerate(rangeAttr)) {
          auto indexAttr = index.value().cast<mlir::IntegerAttr>();
          mlir::Value offset = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), range.index() * 2 + index.index()));
          mlir::Value indexValue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(indexAttr.getInt()));
          mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, equationRangesPtr.getType(), equationRangesPtr, offset);
          rewriter.create<mlir::LLVM::StoreOp>(loc, indexValue, ptr);
        }
      }

      // Rank
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.getEquationRanges().size()));
      newOperands.push_back(rank);
      mangledArgsTypes.push_back(mangling.getIntegerType(rank.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = mangling.getIntegerType(resultType.getIntOrFloatBitWidth());
      auto functionName = mangling.getMangledFunction("idaAddEquation", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the ranges array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(module);
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, equationRangesOpaquePtr);

      return mlir::success();
    }
  };

  struct AddAlgebraicVariableOpLowering : public IDAOpConversion<AddAlgebraicVariableOp>
  {
    using IDAOpConversion<AddAlgebraicVariableOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddAlgebraicVariableOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      auto heapAllocFn = lookupOrCreateHeapAllocFn(module, getIndexType());

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 5> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable
      mlir::Type variableType = adaptor.getVariable().getType();
      mlir::Type variablePtrType = mlir::LLVM::LLVMPointerType::get(variableType);
      mlir::Value variableNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, variablePtrType);
      mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 1));
      mlir::Value variableGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, variablePtrType, variableNullPtr, one);
      mlir::Value variableSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), variableGepPtr);

      mlir::Value variableOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, variableSizeBytes, getVoidPtrType())[0];
      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, variablePtrType, variableOpaquePtr);
      rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getVariable(), variablePtr);

      newOperands.push_back(variableOpaquePtr);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the array with the variable dimensions
      mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
      mlir::Value numOfElements = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), op.getDimensions().size()));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, nullPtr, numOfElements);
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      mlir::Value arrayDimensionsOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value arrayDimensionsPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, arrayDimensionsOpaquePtr);

      newOperands.push_back(arrayDimensionsPtr);
      mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(dimensionSizeType.getIntOrFloatBitWidth())));

      // Populate the dimensions list
      for (const auto& sizeAttr : llvm::enumerate(op.getDimensions())) {
        mlir::Value offset = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), sizeAttr.index()));
        mlir::Value size = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(sizeAttr.value().cast<mlir::IntegerAttr>().getInt()));
        mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, arrayDimensionsPtr.getType(), arrayDimensionsPtr, offset);
        rewriter.create<mlir::LLVM::StoreOp>(loc, size, ptr);
      }

      // Rank
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.getDimensions().size()));
      newOperands.push_back(rank);
      mangledArgsTypes.push_back(mangling.getIntegerType(rank.getType().getIntOrFloatBitWidth()));

      // Variable getter function address
      auto getter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getGetter());
      mlir::Value getterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, getter);
      getterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), getterAddress);

      newOperands.push_back(getterAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable setter function address
      auto setter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getSetter());
      mlir::Value setterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, setter);
      setterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), setterAddress);

      newOperands.push_back(setterAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = mangling.getIntegerType(resultType.getIntOrFloatBitWidth());
      auto functionName = mangling.getMangledFunction("idaAddAlgebraicVariable", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the dimensions array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(module);
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, arrayDimensionsOpaquePtr);

      return mlir::success();
    }
  };

  struct AddStateVariableOpLowering : public IDAOpConversion<AddStateVariableOp>
  {
    using IDAOpConversion<AddStateVariableOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddStateVariableOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      auto heapAllocFn = lookupOrCreateHeapAllocFn(module, getIndexType());

      RuntimeFunctionsMangling mangling;

      // Set the arguments of the runtime function call
      llvm::SmallVector<mlir::Value, 5> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable
      mlir::Type variableType = adaptor.getVariable().getType();
      mlir::Type variablePtrType = mlir::LLVM::LLVMPointerType::get(variableType);
      mlir::Value variableNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, variablePtrType);
      mlir::Value variableOneOffset = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 1));
      mlir::Value variableGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, variablePtrType, variableNullPtr, variableOneOffset);
      mlir::Value variableSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), variableGepPtr);

      mlir::Value variableOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, variableSizeBytes, getVoidPtrType())[0];
      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, variablePtrType, variableOpaquePtr);
      rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getVariable(), variablePtr);

      newOperands.push_back(variableOpaquePtr);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the array with the variable dimensions
      mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
      mlir::Value numOfElements = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), op.getDimensions().size()));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, nullPtr, numOfElements);
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      mlir::Value arrayDimensionsOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value arrayDimensionsPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, arrayDimensionsOpaquePtr);

      newOperands.push_back(arrayDimensionsPtr);
      mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(dimensionSizeType.getIntOrFloatBitWidth())));

      // Populate the dimensions list
      for (const auto& sizeAttr : llvm::enumerate(op.getDimensions())) {
        mlir::Value offset = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), sizeAttr.index()));
        mlir::Value size = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(sizeAttr.value().cast<mlir::IntegerAttr>().getInt()));
        mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, arrayDimensionsPtr.getType(), arrayDimensionsPtr, offset);
        rewriter.create<mlir::LLVM::StoreOp>(loc, size, ptr);
      }

      // Rank
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.getDimensions().size()));
      newOperands.push_back(rank);
      mangledArgsTypes.push_back(mangling.getIntegerType(rank.getType().getIntOrFloatBitWidth()));

      // Variable getter function address
      auto getter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getGetter());
      mlir::Value getterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, getter);
      getterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), getterAddress);

      newOperands.push_back(getterAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable setter function address
      auto setter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getSetter());
      mlir::Value setterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, setter);
      setterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), setterAddress);

      newOperands.push_back(setterAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = mangling.getIntegerType(resultType.getIntOrFloatBitWidth());
      auto functionName = mangling.getMangledFunction("idaAddStateVariable", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the dimensions array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(module);
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, arrayDimensionsOpaquePtr);

      return mlir::success();
    }
  };

  struct SetDerivativeOpLowering : public IDAOpConversion<SetDerivativeOp>
  {
    using IDAOpConversion<SetDerivativeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(SetDerivativeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      auto heapAllocFn = lookupOrCreateHeapAllocFn(module, getIndexType());

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 3> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // State variable
      newOperands.push_back(adaptor.getStateVariable());
      mangledArgsTypes.push_back(mangling.getIntegerType(adaptor.getStateVariable().getType().getIntOrFloatBitWidth()));

      // Derivative
      mlir::Type derivativeType = adaptor.getDerivative().getType();
      mlir::Type derivativePtrType = mlir::LLVM::LLVMPointerType::get(derivativeType);
      mlir::Value derivativeNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, derivativePtrType);
      mlir::Value derivativeOneOffset = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 1));
      mlir::Value derivativeGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, derivativePtrType, derivativeNullPtr, derivativeOneOffset);
      mlir::Value derivativeSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), derivativeGepPtr);

      mlir::Value derivativeOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, derivativeSizeBytes, getVoidPtrType())[0];
      mlir::Value derivativePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, derivativePtrType, derivativeOpaquePtr);
      rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getDerivative(), derivativePtr);

      newOperands.push_back(derivativeOpaquePtr);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable getter function address
      auto getter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getGetter());
      mlir::Value getterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, getter);
      getterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), getterAddress);

      newOperands.push_back(getterAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable setter function address
      auto setter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getSetter());
      mlir::Value setterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, setter);
      setterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), setterAddress);

      newOperands.push_back(setterAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaSetDerivative", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct VariableGetterOpLowering : public IDAOpConversion<VariableGetterOp>
  {
    using IDAOpConversion<VariableGetterOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(VariableGetterOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::SmallVector<mlir::Type, 3> argsTypes;

      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));

      auto functionType = mlir::LLVM::LLVMFunctionType::get(getTypeConverter()->convertType(op.getFunctionType().getResult(0)), argsTypes);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::LLVMFuncOp>(op, op.getSymName(), functionType);
      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // The lowered function will receive a void pointer to the array descriptor of the variable
      mlir::Value variableOpaquePtr = newOp.getArgument(0);
      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, mlir::LLVM::LLVMPointerType::get(op.getVariable().getType()), variableOpaquePtr);
      mlir::Value variable = rewriter.create<mlir::LLVM::LoadOp>(loc, variablePtr);
      mapping.map(op.getVariable(), variable);

      // The equation indices are also passed through an array
      mlir::Value variableIndicesPtr = newOp.getArgument(1);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(variableIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));
        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(variableIndicesPtr.getLoc(), variableIndicesPtr.getType(), variableIndicesPtr, index);
        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(variableIndexPtr.getLoc(), variableIndexPtr);
        mappedVariableIndex = getTypeConverter()->materializeSourceConversion(rewriter, mappedVariableIndex.getLoc(), rewriter.getIndexType(), mappedVariableIndex);
        mapping.map(variableIndex.value(), mappedVariableIndex);
      }

      // Clone the blocks structure
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getBody(),
              newOp.getBody().end(),
              block.value().getArgumentTypes(),
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned] : llvm::zip(block.value().getArguments(), clonedBlock->getArguments())) {
            mapping.map(original, cloned);
          }
        }
      }

      // Clone the original operations
      for (auto& block : llvm::enumerate(op.getBodyRegion())) {
        if (block.index() == 0) {
          rewriter.setInsertionPointToEnd(entryBlock);
        } else {
          rewriter.setInsertionPointToStart(mapping.lookup(&block.value()));
        }

        for (auto& bodyOp : block.value().getOperations()) {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };

  struct VariableSetterOpLowering : public IDAOpConversion<VariableSetterOp>
  {
    using IDAOpConversion<VariableSetterOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(VariableSetterOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::SmallVector<mlir::Type, 3> argsTypes;

      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(op.getValue().getType());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));

      auto functionType = mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), argsTypes);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::LLVMFuncOp>(op, op.getSymName(), functionType);
      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // The lowered function will receive a void pointer to the array descriptor of the variable
      mlir::Value variableOpaquePtr = newOp.getArgument(0);
      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, mlir::LLVM::LLVMPointerType::get(op.getVariable().getType()), variableOpaquePtr);
      mlir::Value variable = rewriter.create<mlir::LLVM::LoadOp>(loc, variablePtr);
      mapping.map(op.getVariable(), variable);

      // Map the value
      mapping.map(op.getValue(), newOp.getArgument(1));

      // The equation indices are also passed through an array
      mlir::Value variableIndicesPtr = newOp.getArgument(2);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(variableIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));
        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(variableIndicesPtr.getLoc(), variableIndicesPtr.getType(), variableIndicesPtr, index);
        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(variableIndexPtr.getLoc(), variableIndexPtr);
        mappedVariableIndex = getTypeConverter()->materializeSourceConversion(rewriter, mappedVariableIndex.getLoc(), rewriter.getIndexType(), mappedVariableIndex);
        mapping.map(variableIndex.value(), mappedVariableIndex);
      }

      // Clone the blocks structure
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getBody(),
              newOp.getBody().end(),
              block.value().getArgumentTypes(),
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned] : llvm::zip(block.value().getArguments(), clonedBlock->getArguments())) {
            mapping.map(original, cloned);
          }
        }
      }

      // Clone the original operations
      for (auto& block : llvm::enumerate(op.getBodyRegion())) {
        if (block.index() == 0) {
          rewriter.setInsertionPointToEnd(entryBlock);
        } else {
          rewriter.setInsertionPointToStart(mapping.lookup(&block.value()));
        }

        for (auto& bodyOp : block.value().getOperations()) {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };

  struct AddVariableAccessOpLowering : public IDAOpConversion<AddVariableAccessOp>
  {
    using IDAOpConversion<AddVariableAccessOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddVariableAccessOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 5> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation
      newOperands.push_back(adaptor.getEquation());
      mangledArgsTypes.push_back(mangling.getIntegerType(adaptor.getEquation().getType().getIntOrFloatBitWidth()));

      // Variable
      newOperands.push_back(adaptor.getVariable());
      mangledArgsTypes.push_back(mangling.getIntegerType(adaptor.getVariable().getType().getIntOrFloatBitWidth()));

      // Create the array with the variable accesses
      auto dimensionAccesses = op.getAccess().getResults();
      llvm::SmallVector<mlir::Value, 6> accessValues;

      for (const auto dimensionAccess : dimensionAccesses) {
        if (dimensionAccess.isa<mlir::AffineConstantExpr>()) {
          auto constantAccess = dimensionAccess.cast<mlir::AffineConstantExpr>();
          accessValues.push_back(rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(-1)));
          accessValues.push_back(rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(constantAccess.getValue())));
        } else if (dimensionAccess.isa<mlir::AffineDimExpr>()) {
          auto dimension = dimensionAccess.cast<mlir::AffineDimExpr>();
          accessValues.push_back(rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(dimension.getPosition())));
          accessValues.push_back(rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0)));
        } else {
          auto dynamicAccess = dimensionAccess.cast<mlir::AffineBinaryOpExpr>();
          auto dimension = dynamicAccess.getLHS().cast<mlir::AffineDimExpr>();
          auto offset = dynamicAccess.getRHS().cast<mlir::AffineConstantExpr>();
          accessValues.push_back(rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(dimension.getPosition())));
          accessValues.push_back(rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(offset.getValue())));
        }
      }

      mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
      mlir::Value numOfElements = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), accessValues.size()));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, nullPtr, numOfElements);
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      auto heapAllocFn = lookupOrCreateHeapAllocFn(op->getParentOfType<mlir::ModuleOp>(), getIndexType());
      mlir::Value accessesOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value accessesPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, accessesOpaquePtr);

      newOperands.push_back(accessesPtr);
      mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(dimensionSizeType.getIntOrFloatBitWidth())));

      // Populate the equation ranges
      for (const auto& accessValue : llvm::enumerate(accessValues)) {
        mlir::Value offset = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), accessValue.index()));
        mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, accessesPtr.getType(), accessesPtr, offset);
        rewriter.create<mlir::LLVM::StoreOp>(loc, accessValue.value(), ptr);
      }

      // Rank
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.getAccess().getResults().size()));
      newOperands.push_back(rank);
      mangledArgsTypes.push_back(mangling.getIntegerType(rank.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaAddVariableAccess", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the accesses array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(op->getParentOfType<mlir::ModuleOp>());
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, accessesOpaquePtr);

      return mlir::success();
    }
  };

  struct ResidualFunctionOpLowering : public IDAOpConversion<ResidualFunctionOp>
  {
    using IDAOpConversion<ResidualFunctionOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(ResidualFunctionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::SmallVector<mlir::Type, 3> argsTypes;

      argsTypes.push_back(op.getTime().getType());
      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));

      auto functionType = mlir::LLVM::LLVMFunctionType::get(op.getFunctionType().getResult(0), argsTypes);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::LLVMFuncOp>(op, op.getSymName(), functionType);
      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Map the time variable
      mapping.map(op.getTime(), newOp.getArgument(0));

      // The lowered function will receive a pointer to the array of variables.
      assert(!op.getVariables().empty());

      mlir::Value variablesPtr = newOp.getArgument(1);
      variablesPtr = rewriter.create<mlir::LLVM::BitcastOp>(variablesPtr.getLoc(), mlir::LLVM::LLVMPointerType::get(getVoidPtrType()), variablesPtr);

      for (auto variable : llvm::enumerate(op.getVariables())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(variablesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variable.index()));
        mlir::Value variablePtr = rewriter.create<mlir::LLVM::GEPOp>(variablesPtr.getLoc(), variablesPtr.getType(), variablesPtr, index);
        mlir::Value mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(variablePtr.getLoc(), variablePtr);
        mappedVariable = rewriter.create<mlir::LLVM::BitcastOp>(mappedVariable.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getVariables()[variable.index()].getType()), mappedVariable);
        mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(mappedVariable.getLoc(), mappedVariable);

        mapping.map(variable.value(), mappedVariable);
      }

      // The equation indices are also passed through an array
      mlir::Value equationIndicesPtr = newOp.getArgument(2);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(equationIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));
        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(equationIndicesPtr.getLoc(), equationIndicesPtr.getType(), equationIndicesPtr, index);
        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(equationIndexPtr.getLoc(), equationIndexPtr);
        mappedEquationIndex = getTypeConverter()->materializeSourceConversion(rewriter, mappedEquationIndex.getLoc(), rewriter.getIndexType(), mappedEquationIndex);
        mapping.map(equationIndex.value(), mappedEquationIndex);
      }

      // Clone the blocks structure
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getBody(),
              newOp.getBody().end(),
              block.value().getArgumentTypes(),
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned] : llvm::zip(block.value().getArguments(), clonedBlock->getArguments())) {
            mapping.map(original, cloned);
          }
        }
      }

      // Clone the original operations
      for (auto& block : llvm::enumerate(op.getBodyRegion())) {
        if (block.index() == 0) {
          rewriter.setInsertionPointToEnd(entryBlock);
        } else {
          rewriter.setInsertionPointToStart(mapping.lookup(&block.value()));
        }

        for (auto& bodyOp : block.value().getOperations()) {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };

  struct JacobianFunctionOpLowering : public IDAOpConversion<JacobianFunctionOp>
  {
    using IDAOpConversion<JacobianFunctionOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(JacobianFunctionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::SmallVector<mlir::Type, 5> argsTypes;

      argsTypes.push_back(op.getTime().getType());
      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));
      argsTypes.push_back(op.getAlpha().getType());

      auto functionType = mlir::LLVM::LLVMFunctionType::get(op.getFunctionType().getResult(0), argsTypes);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::LLVMFuncOp>(op, op.getSymName(), functionType);
      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Map the "time" variable
      mapping.map(op.getTime(), newOp.getArgument(0));

      // The lowered function will receive a pointer to the array of variables.
      assert(!op.getVariables().empty());

      mlir::Value variablesPtr = newOp.getArgument(1);
      variablesPtr = rewriter.create<mlir::LLVM::BitcastOp>(variablesPtr.getLoc(), mlir::LLVM::LLVMPointerType::get(getVoidPtrType()), variablesPtr);

      for (auto variable : llvm::enumerate(op.getVariables())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(variablesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variable.index()));
        mlir::Value variablePtr = rewriter.create<mlir::LLVM::GEPOp>(variablesPtr.getLoc(), variablesPtr.getType(), variablesPtr, index);
        mlir::Value mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(variablePtr.getLoc(), variablePtr);
        mappedVariable = rewriter.create<mlir::LLVM::BitcastOp>(mappedVariable.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getVariables()[variable.index()].getType()), mappedVariable);
        mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(mappedVariable.getLoc(), mappedVariable);

        mapping.map(variable.value(), mappedVariable);
      }

      // The equation indices are also passed through an array
      mlir::Value equationIndicesPtr = newOp.getArgument(2);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(equationIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));
        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(equationIndicesPtr.getLoc(), equationIndicesPtr.getType(), equationIndicesPtr, index);
        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(equationIndexPtr.getLoc(), equationIndexPtr);
        mappedEquationIndex = getTypeConverter()->materializeSourceConversion(rewriter, mappedEquationIndex.getLoc(), rewriter.getIndexType(), mappedEquationIndex);
        mapping.map(equationIndex.value(), mappedEquationIndex);
      }

      // The variable indices are also passed through an array
      mlir::Value variableIndicesPtr = newOp.getArgument(3);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(equationIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));
        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(variableIndicesPtr.getLoc(), variableIndicesPtr.getType(), variableIndicesPtr, index);
        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(variableIndexPtr.getLoc(), variableIndexPtr);
        mappedVariableIndex = getTypeConverter()->materializeSourceConversion(rewriter, mappedVariableIndex.getLoc(), rewriter.getIndexType(), mappedVariableIndex);
        mapping.map(variableIndex.value(), mappedVariableIndex);
      }

      // Add the "alpha" variable
      mapping.map(op.getAlpha(), newOp.getArgument(4));

      // Clone the blocks structure
      for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
        if (block.index() != 0) {
          std::vector<mlir::Location> argLocations;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getBody(),
              newOp.getBody().end(),
              block.value().getArgumentTypes(),
              argLocations);

          mapping.map(&block.value(), clonedBlock);

          for (const auto& [original, cloned] : llvm::zip(block.value().getArguments(), clonedBlock->getArguments())) {
            mapping.map(original, cloned);
          }
        }
      }

      // Clone the original operations
      for (auto& block : llvm::enumerate(op.getBodyRegion())) {
        if (block.index() == 0) {
          rewriter.setInsertionPointToEnd(entryBlock);
        } else {
          rewriter.setInsertionPointToStart(mapping.lookup(&block.value()));
        }

        for (auto& bodyOp : block.value().getOperations()) {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };

  struct ReturnOpLowering : public IDAOpConversion<ReturnOp>
  {
    using IDAOpConversion<ReturnOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, adaptor.getOperands());
      return mlir::success();
    }
  };

  struct AddResidualOpLowering : public IDAOpConversion<AddResidualOp>
  {
    using IDAOpConversion<AddResidualOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddResidualOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 3> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation
      newOperands.push_back(adaptor.getEquation());
      mangledArgsTypes.push_back(mangling.getIntegerType(adaptor.getEquation().getType().getIntOrFloatBitWidth()));

      // Residual function address
      auto function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getFunction());
      mlir::Value functionAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, function);
      functionAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), functionAddress);

      newOperands.push_back(functionAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaAddResidual", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct AddJacobianOpLowering : public IDAOpConversion<AddJacobianOp>
  {
    using IDAOpConversion<AddJacobianOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddJacobianOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 4> newOperands;
      llvm::SmallVector<std::string, 4> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation
      newOperands.push_back(adaptor.getEquation());
      mangledArgsTypes.push_back(mangling.getIntegerType(adaptor.getEquation().getType().getIntOrFloatBitWidth()));

      // Variable
      newOperands.push_back(adaptor.getVariable());
      mangledArgsTypes.push_back(mangling.getIntegerType(adaptor.getVariable().getType().getIntOrFloatBitWidth()));

      // Jacobian function address
      auto function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getFunction());
      mlir::Value functionAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, function);
      functionAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), functionAddress);

      newOperands.push_back(functionAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaAddJacobian", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct InitOpLowering : public IDAOpConversion<InitOp>
  {
    using IDAOpConversion<InitOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(InitOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 3> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaInit", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, adaptor.getOperands());

      return mlir::success();
    }
  };

  struct StepOpLowering : public IDAOpConversion<StepOp>
  {
    using IDAOpConversion<StepOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(StepOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaStep", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct FreeOpLowering : public IDAOpConversion<FreeOp>
  {
    using IDAOpConversion<FreeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(FreeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaFree", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, newOperands);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, adaptor.getOperands());

      return mlir::success();
    }
  };

  struct PrintStatisticsOpLowering : public IDAOpConversion<PrintStatisticsOp>
  {
    using IDAOpConversion<PrintStatisticsOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(PrintStatisticsOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();
      auto functionName = mangling.getMangledFunction("idaPrintStatistics", mangledResultType, mangledArgsTypes);
      auto callee = getOrDeclareFunction(rewriter, module, loc, functionName, resultType, adaptor.getOperands());
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, adaptor.getOperands());

      return mlir::success();
    }
  };
}

static void populateIDAFunctionLikeOpsConversionPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::ida::TypeConverter& typeConverter,
    unsigned int bitWidth)
{
  patterns.insert<
      VariableGetterOpLowering,
      VariableSetterOpLowering,
      ResidualFunctionOpLowering,
      JacobianFunctionOpLowering,
      ReturnOpLowering>(typeConverter, bitWidth);
}

static void populateIDAConversionPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::ida::TypeConverter& typeConverter,
    unsigned int bitWidth)
{
  patterns.insert<
      CreateOpLowering,
      SetStartTimeOpLowering,
      SetEndTimeOpLowering,
      SetTimeStepOpLowering,
      GetCurrentTimeOpLowering,
      AddEquationOpLowering,
      AddAlgebraicVariableOpLowering,
      AddStateVariableOpLowering,
      SetDerivativeOpLowering,
      AddVariableAccessOpLowering,
      AddResidualOpLowering,
      AddJacobianOpLowering,
      InitOpLowering,
      StepOpLowering,
      FreeOpLowering,
      PrintStatisticsOpLowering>(typeConverter, bitWidth);
}

namespace marco::codegen
{
  class IDAToLLVMConversionPass : public IDAToLLVMBase<IDAToLLVMConversionPass> // mlir::PassWrapper<IDAToLLVMConversionPass, mlir::OperationPass<mlir::ModuleOp>> //
  {
    public:
      void runOnOperation() override
      {
        // Convert the function-like operations.
        // This must be done first and as an independent step, as other operations within
        // the IDA dialect will need the addresses of such functions when being converted.

        if (mlir::failed(convertFunctionsLikeOps())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the IDA function-like operations");
          return signalPassFailure();
        }

        // Convert the rest of the IDA dialect
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the IDA operations");
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertFunctionsLikeOps()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addLegalDialect<mlir::LLVM::LLVMDialect>();

        target.addIllegalOp<
            VariableGetterOp,
            VariableSetterOp,
            ResidualFunctionOp,
            JacobianFunctionOp,
            ReturnOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::RewritePatternSet patterns(&getContext());
        populateIDAFunctionLikeOpsConversionPatterns(patterns, typeConverter, 64);

        if (auto status = applyPartialConversion(module, target, std::move(patterns)); mlir::failed(status)) {
          return status;
        }

        return mlir::success();
      }

      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addIllegalDialect<mlir::ida::IDADialect>();
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::RewritePatternSet patterns(&getContext());
        populateIDAConversionPatterns(patterns, typeConverter, 64);

        if (auto status = applyPartialConversion(module, target, std::move(patterns)); mlir::failed(status)) {
          return status;
        }

        return mlir::success();
      }
  };
}

namespace
{
  struct AddAlgebraicVariableOpTypes : public mlir::OpConversionPattern<AddAlgebraicVariableOp>
  {
    using mlir::OpConversionPattern<AddAlgebraicVariableOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AddAlgebraicVariableOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<AddAlgebraicVariableOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct AddStateVariableOpTypes : public mlir::OpConversionPattern<AddStateVariableOp>
  {
    using mlir::OpConversionPattern<AddStateVariableOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AddStateVariableOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<AddStateVariableOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct SetDerivativeOpTypes : public mlir::OpConversionPattern<SetDerivativeOp>
  {
    using mlir::OpConversionPattern<SetDerivativeOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SetDerivativeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<SetDerivativeOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct ConvertGetCurrentTimeOpTypes : public mlir::OpConversionPattern<GetCurrentTimeOp>
  {
    using mlir::OpConversionPattern<GetCurrentTimeOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(GetCurrentTimeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<GetCurrentTimeOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(typeConverter->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct VariableGetterOpTypes : public mlir::OpConversionPattern<VariableGetterOp>
  {
    using mlir::OpConversionPattern<VariableGetterOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(VariableGetterOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type resultType = getTypeConverter()->convertType(op.getFunctionType().getResult(0));
      mlir::Type variableType = getTypeConverter()->convertType(op.getVariable().getType());

      auto newOp = rewriter.replaceOpWithNewOp<VariableGetterOp>(
          op, op.getSymName(), resultType, variableType, op.getVariableIndices().size());

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned] : llvm::zip(op.getBody().getArguments(), newOp.getBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(), mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getVariable());
      assert(op.getBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getBody().getOps()) {
        if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
          std::vector<mlir::Value> returnValues;

          for (auto returnValue : returnOp.operands()) {
            returnValues.push_back(getTypeConverter()->materializeTargetConversion(
                rewriter, returnOp.getLoc(),
                getTypeConverter()->convertType(returnValue.getType()),
                mapping.lookup(returnValue)));
          }

          rewriter.create<ReturnOp>(returnOp.getLoc(), returnValues);
        } else {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };

  struct VariableSetterOpTypes : public mlir::OpConversionPattern<VariableSetterOp>
  {
    using mlir::OpConversionPattern<VariableSetterOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(VariableSetterOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type variableType = getTypeConverter()->convertType(op.getVariable().getType());
      mlir::Type valueType = getTypeConverter()->convertType(op.getValue().getType());

      auto newOp = rewriter.replaceOpWithNewOp<VariableSetterOp>(
          op, op.getSymName(), variableType, valueType, op.getVariableIndices().size());

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned] : llvm::zip(op.getBody().getArguments(), newOp.getBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(), mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getVariable());
      castArgFn(op.getValue());

      assert(op.getBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getBody().getOps()) {
        rewriter.clone(bodyOp, mapping);
      }

      return mlir::success();
    }
  };

  struct ResidualFunctionOpTypes : public mlir::OpConversionPattern<ResidualFunctionOp>
  {
    using mlir::OpConversionPattern<ResidualFunctionOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ResidualFunctionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType = getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type differenceType = getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<ResidualFunctionOp>(
          op, op.getSymName(), timeType, variablesTypes, op.getEquationRank().getSExtValue(), differenceType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned] : llvm::zip(op.getBody().getArguments(), newOp.getBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(), mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getTime());

      for (auto variable : op.getVariables()) {
        castArgFn(variable);
      }

      assert(op.getBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getBody().getOps()) {
        if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
          std::vector<mlir::Value> returnValues;

          for (auto returnValue : returnOp.operands()) {
            returnValues.push_back(getTypeConverter()->materializeTargetConversion(
                rewriter, returnOp.getLoc(), differenceType, mapping.lookup(returnValue)));
          }

          rewriter.create<ReturnOp>(returnOp.getLoc(), returnValues);
        } else {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };

  struct JacobianFunctionOpTypes : public mlir::OpConversionPattern<JacobianFunctionOp>
  {
    using mlir::OpConversionPattern<JacobianFunctionOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(JacobianFunctionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType = getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type alphaType = getTypeConverter()->convertType(op.getAlpha().getType());
      mlir::Type resultType = getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<JacobianFunctionOp>(
          op, op.getSymName(), timeType, variablesTypes,
          op.getEquationRank().getSExtValue(),
          op.getVariableRank().getSExtValue(),
          alphaType, resultType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned] : llvm::zip(op.getBody().getArguments(), newOp.getBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(), mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getTime());

      for (auto variable : op.getVariables()) {
        castArgFn(variable);
      }

      castArgFn(op.getAlpha());

      assert(op.getBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getBody().getOps()) {
        if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
          std::vector<mlir::Value> returnValues;

          for (auto returnValue : returnOp.operands()) {
            returnValues.push_back(getTypeConverter()->materializeTargetConversion(
                rewriter, returnOp.getLoc(), resultType, mapping.lookup(returnValue)));
          }

          rewriter.create<ReturnOp>(returnOp.getLoc(), returnValues);
        } else {
          rewriter.clone(bodyOp, mapping);
        }
      }

      return mlir::success();
    }
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createIDAToLLVMPass()
  {
    return std::make_unique<IDAToLLVMConversionPass>();
  }

  void populateIDAStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      mlir::ConversionTarget& target)
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
        AddAlgebraicVariableOpTypes,
        AddStateVariableOpTypes,
        SetDerivativeOpTypes>(typeConverter, patterns.getContext());

    patterns.add<ConvertGetCurrentTimeOpTypes>(typeConverter, patterns.getContext());

    patterns.add<
        VariableGetterOpTypes,
        VariableSetterOpTypes,
        ResidualFunctionOpTypes,
        JacobianFunctionOpTypes>(typeConverter, patterns.getContext());

    target.addDynamicallyLegalOp<AddAlgebraicVariableOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<AddStateVariableOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<SetDerivativeOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<GetCurrentTimeOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<VariableGetterOp>([&](VariableGetterOp op) {
      if (!typeConverter.isLegal(op.getFunctionType().getResult(0))) {
        return false;
      }

      if (!typeConverter.isLegal(op.getVariable().getType())) {
        return false;
      }

      return true;
    });

    target.addDynamicallyLegalOp<VariableSetterOp>([&](VariableSetterOp op) {
      if (!typeConverter.isLegal(op.getVariable().getType())) {
        return false;
      }

      if (!typeConverter.isLegal(op.getValue().getType())) {
        return false;
      }

      return true;
    });

    target.addDynamicallyLegalOp<ResidualFunctionOp>([&](ResidualFunctionOp op) {
      if (!typeConverter.isLegal(op.getTime().getType())) {
        return false;
      }

      for (auto variable : op.getVariables()) {
        if (!typeConverter.isLegal(variable.getType())) {
          return false;
        }
      }

      if (!typeConverter.isLegal(op.getFunctionType().getResult(0))) {
        return false;
      }

      return true;
    });

    target.addDynamicallyLegalOp<JacobianFunctionOp>([&](JacobianFunctionOp op) {
      if (!typeConverter.isLegal(op.getTime().getType())) {
        return false;
      }

      for (auto variable : op.getVariables()) {
        if (!typeConverter.isLegal(variable.getType())) {
          return false;
        }
      }

      if (!typeConverter.isLegal(op.getAlpha().getType())) {
        return false;
      }

      if (!typeConverter.isLegal(op.getFunctionType().getResult(0))) {
        return false;
      }

      return true;
    });
  }
}
