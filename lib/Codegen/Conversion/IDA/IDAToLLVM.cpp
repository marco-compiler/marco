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

#include "llvm/Support/Debug.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::ida;

static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
    mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::Type result, llvm::ArrayRef<mlir::Type> args)
{
  if (auto funcOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
    return funcOp;
  }

  mlir::PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return builder.create<mlir::LLVM::LLVMFuncOp>(
      module.getLoc(), name, mlir::LLVM::LLVMFunctionType::get(result, args));
}

static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
    mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::Type result, mlir::ValueRange args)
{
  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (auto type : args.getTypes()) {
    argsTypes.push_back(type);
  }

  return getOrDeclareFunction(builder, module, name, result, argsTypes);
}

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class IDAOpConversion : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      using Adaptor = typename Op::Adaptor;

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

    mlir::LogicalResult matchAndRewrite(CreateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      RuntimeFunctionsMangling mangling;

      mlir::Type voidPtrType = getVoidPtrType();
      auto mangledResultType = mangling.getVoidPointerType();

      llvm::SmallVector<mlir::Value, 2> newOperands;

      // Scalar equations amount
      newOperands.push_back(rewriter.create<mlir::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(op.scalarEquations())));

      // Data bitwidth
      newOperands.push_back(rewriter.create<mlir::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(bitWidth)));

      llvm::SmallVector<std::string, 2> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getIntegerType(newOperands[0].getType().getIntOrFloatBitWidth()));
      mangledArgsTypes.push_back(mangling.getIntegerType(newOperands[1].getType().getIntOrFloatBitWidth()));

      auto functionName = mangling.getMangledFunction("idaCreate", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, voidPtrType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
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
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
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
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };

  struct SetTimeStepOpLowering : public IDAOpConversion<SetTimeStepOp>
  {
    using IDAOpConversion<SetTimeStepOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(SetTimeStepOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      RuntimeFunctionsMangling mangling;

      auto mangledResultType = mangling.getVoidType();

      assert(operands.size() == 1);
      llvm::SmallVector<mlir::Value, 2> newOperands;
      newOperands.push_back(operands[0]);

      // Time step
      newOperands.push_back(rewriter.create<mlir::ConstantOp>(
          loc, rewriter.getF64FloatAttr(op.timeStep().convertToDouble())));

      llvm::SmallVector<std::string, 2> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getVoidPointerType());
      mangledArgsTypes.push_back(mangling.getFloatingPointType(newOperands[1].getType().getIntOrFloatBitWidth()));

      auto functionName = mangling.getMangledFunction("idaSetTimeStep", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
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
      mangledArgsTypes.push_back(mangling.getFloatingPointType(newOperands[1].getType().getIntOrFloatBitWidth()));

      auto functionName = mangling.getMangledFunction("idaSetRelativeTolerance", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
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
      mangledArgsTypes.push_back(mangling.getFloatingPointType(newOperands[1].getType().getIntOrFloatBitWidth()));

      auto functionName = mangling.getMangledFunction("idaSetAbsoluteTolerance", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
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

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, operands);
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
      mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), op.equationRanges().size() * 2));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::makeArrayRef({ nullPtr, numOfElements }));
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      auto heapAllocFn = lookupOrCreateHeapAllocFn(op->getParentOfType<mlir::ModuleOp>(), getIndexType());
      mlir::Value equationRangesOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value equationRangesPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, equationRangesOpaquePtr);
      newOperands.push_back(equationRangesPtr);

      // Populate the equation ranges
      for (const auto& range : llvm::enumerate(op.equationRanges())) {
        auto rangeAttr = range.value().cast<mlir::ArrayAttr>();
        assert(rangeAttr.size() == 2);

        for (const auto& index : llvm::enumerate(rangeAttr)) {
          auto indexAttr = index.value().cast<mlir::IntegerAttr>();
          mlir::Value offset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), range.index() * 2 + index.index()));
          mlir::Value indexValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(indexAttr.getInt()));
          mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, equationRangesPtr.getType(), equationRangesPtr, offset);
          rewriter.create<mlir::LLVM::StoreOp>(loc, indexValue, ptr);
        }
      }

      // Rank
      mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.equationRanges().size()));
      newOperands.push_back(rank);

      // Mangled types
      auto mangledResultType = mangling.getIntegerType(64);

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

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the ranges array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(op->getParentOfType<mlir::ModuleOp>());
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, equationRangesOpaquePtr, getVoidType())[0];

      return mlir::success();
    }
  };

  struct AddAlgebraicVariableOpLowering : public IDAOpConversion<AddAlgebraicVariableOp>
  {
    using IDAOpConversion<AddAlgebraicVariableOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddAlgebraicVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto heapAllocFn = lookupOrCreateHeapAllocFn(module, getIndexType());

      RuntimeFunctionsMangling mangling;

     auto setter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.setter());

     if (!setter) {
       return rewriter.notifyMatchFailure(op, "Variable setter function " + op.setter().getLeafReference() + " not found");
     }

     mlir::Value setterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, setter);
     setterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), setterAddress);

      auto resultType = getTypeConverter()->convertType(op.getResult().getType());

      assert(operands.size() == 2);
      llvm::SmallVector<mlir::Value, 5> newOperands;

      // IDA instance
      newOperands.push_back(operands[0]);

      // Variable
      mlir::Type variableType = operands[1].getType();
      mlir::Type variablePtrType = mlir::LLVM::LLVMPointerType::get(variableType);
      mlir::Value variableNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, variablePtrType);
      mlir::Value one = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 1));
      mlir::Value variableGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, variablePtrType, llvm::makeArrayRef({ variableNullPtr, one }));
      mlir::Value variableSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), variableGepPtr);

      mlir::Value variableOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, variableSizeBytes, getVoidPtrType())[0];
      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, variablePtrType, variableOpaquePtr);
      rewriter.create<mlir::LLVM::StoreOp>(loc, operands[1], variablePtr);
      newOperands.push_back(variableOpaquePtr);

      // Create the array with the variable dimensions
      mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
      mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), op.arrayDimensions().size()));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::makeArrayRef({ nullPtr, numOfElements }));
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      mlir::Value arrayDimensionsOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value arrayDimensionsPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, arrayDimensionsOpaquePtr);
      newOperands.push_back(arrayDimensionsPtr);

      // Populate the dimensions list
      for (const auto& sizeAttr : llvm::enumerate(op.arrayDimensions())) {
        mlir::Value offset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), sizeAttr.index()));
        mlir::Value size = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(sizeAttr.value().cast<mlir::IntegerAttr>().getInt()));
        mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, arrayDimensionsPtr.getType(), arrayDimensionsPtr, offset);
        rewriter.create<mlir::LLVM::StoreOp>(loc, size, ptr);
      }

      // Rank
      mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.arrayDimensions().size()));
      newOperands.push_back(rank);

      // Variable setter function address
      newOperands.push_back(setterAddress);

      // Mangled types
      auto mangledResultType = mangling.getIntegerType(64);

      llvm::SmallVector<std::string, 3> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getVoidPointerType());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());
      mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(64)));
      mangledArgsTypes.push_back(mangling.getIntegerType(64));
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto functionName = mangling.getMangledFunction(
          "idaAddAlgebraicVariable", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the dimensions array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(op->getParentOfType<mlir::ModuleOp>());
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, arrayDimensionsOpaquePtr, getVoidType())[0];

      return mlir::success();
    }
  };

  struct AddStateVariableOpLowering : public IDAOpConversion<AddStateVariableOp>
  {
    using IDAOpConversion<AddStateVariableOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddStateVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto heapAllocFn = lookupOrCreateHeapAllocFn(module, getIndexType());

      RuntimeFunctionsMangling mangling;

      auto setter = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.setter());

      if (!setter) {
        return rewriter.notifyMatchFailure(op, "Variable setter function " + op.setter().getLeafReference() + " not found");
      }

      mlir::Value setterAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, setter);
      setterAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), setterAddress);

      auto resultType = getTypeConverter()->convertType(op.getResult().getType());

      assert(operands.size() == 3);
      llvm::SmallVector<mlir::Value, 5> newOperands;

      // IDA instance
      newOperands.push_back(operands[0]);

      // Variable
      mlir::Type variableType = operands[1].getType();
      mlir::Type variablePtrType = mlir::LLVM::LLVMPointerType::get(variableType);
      mlir::Value variableNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, variablePtrType);
      mlir::Value variableOneOffset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 1));
      mlir::Value variableGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, variablePtrType, llvm::makeArrayRef({ variableNullPtr, variableOneOffset }));
      mlir::Value variableSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), variableGepPtr);

      mlir::Value variableOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, variableSizeBytes, getVoidPtrType())[0];
      mlir::Value variablePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, variablePtrType, variableOpaquePtr);
      rewriter.create<mlir::LLVM::StoreOp>(loc, operands[1], variablePtr);
      newOperands.push_back(variableOpaquePtr);

      // Derivative
      mlir::Type derivativeType = operands[2].getType();
      mlir::Type derivativePtrType = mlir::LLVM::LLVMPointerType::get(derivativeType);
      mlir::Value derivativeNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, derivativePtrType);
      mlir::Value derivativeOneOffset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 1));
      mlir::Value derivativeGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, derivativePtrType, llvm::makeArrayRef({ derivativeNullPtr, derivativeOneOffset }));
      mlir::Value derivativeSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), derivativeGepPtr);

      mlir::Value derivativeOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, derivativeSizeBytes, getVoidPtrType())[0];
      mlir::Value derivativePtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, derivativePtrType, derivativeOpaquePtr);
      rewriter.create<mlir::LLVM::StoreOp>(loc, operands[2], derivativePtr);
      newOperands.push_back(derivativeOpaquePtr);

      // Create the array with the variable dimensions
      mlir::Type dimensionSizeType = getTypeConverter()->convertType(rewriter.getI64Type());
      mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), op.arrayDimensions().size()));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::makeArrayRef({ nullPtr, numOfElements }));
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      mlir::Value arrayDimensionsOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value arrayDimensionsPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, arrayDimensionsOpaquePtr);
      newOperands.push_back(arrayDimensionsPtr);

      // Populate the dimensions list
      for (const auto& sizeAttr : llvm::enumerate(op.arrayDimensions())) {
        mlir::Value offset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), sizeAttr.index()));
        mlir::Value size = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(sizeAttr.value().cast<mlir::IntegerAttr>().getInt()));
        mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, arrayDimensionsPtr.getType(), arrayDimensionsPtr, offset);
        rewriter.create<mlir::LLVM::StoreOp>(loc, size, ptr);
      }

      // Rank
      mlir::Value rank = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI64IntegerAttr(op.arrayDimensions().size()));
      newOperands.push_back(rank);

      // Variable setter function address
      newOperands.push_back(setterAddress);

      // Mangled types
      auto mangledResultType = mangling.getIntegerType(64);

      llvm::SmallVector<std::string, 3> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getVoidPointerType());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());
      mangledArgsTypes.push_back(mangling.getPointerType(mangling.getIntegerType(64)));
      mangledArgsTypes.push_back(mangling.getIntegerType(64));
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library
      auto functionName = mangling.getMangledFunction(
          "idaAddStateVariable", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the dimensions array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(op->getParentOfType<mlir::ModuleOp>());
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, arrayDimensionsOpaquePtr, getVoidType())[0];

      return mlir::success();
    }
  };

  struct VariableSetterOpLowering : public IDAOpConversion<VariableSetterOp>
  {
    using IDAOpConversion<VariableSetterOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(VariableSetterOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::SmallVector<mlir::Type, 3> argsTypes;

      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(op.getValue().getType());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));

      auto functionType = rewriter.getFunctionType(argsTypes, llvm::None);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::FuncOp>(op, op.name(), functionType);
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
        mlir::Value index = rewriter.create<mlir::ConstantOp>(variableIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));
        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(variableIndicesPtr.getLoc(), variableIndicesPtr.getType(), variableIndicesPtr, index);
        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(variableIndexPtr.getLoc(), variableIndexPtr);
        mapping.map(variableIndex.value(), mappedVariableIndex);
      }

      // Clone the original operations
      assert(op.bodyRegion().getBlocks().size() == 1);

      for (auto& bodyOp : op.bodyRegion().getOps()) {
        rewriter.clone(bodyOp, mapping);
      }

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
      mlir::Value numOfElements = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), accessValues.size()));
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(dimensionSizeType);
      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::makeArrayRef({ nullPtr, numOfElements }));
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

      auto heapAllocFn = lookupOrCreateHeapAllocFn(op->getParentOfType<mlir::ModuleOp>(), getIndexType());
      mlir::Value accessesOpaquePtr = mlir::LLVM::createLLVMCall(rewriter, loc, heapAllocFn, sizeBytes, getVoidPtrType())[0];
      mlir::Value accessesPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, accessesOpaquePtr);
      newOperands.push_back(accessesPtr);

      // Populate the equation ranges
      for (const auto& accessValue : llvm::enumerate(accessValues)) {
        mlir::Value offset = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), accessValue.index()));
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
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the accesses array
      auto heapFreeFn = lookupOrCreateHeapFreeFn(op->getParentOfType<mlir::ModuleOp>());
      mlir::LLVM::createLLVMCall(rewriter, loc, heapFreeFn, accessesOpaquePtr, getVoidType())[0];

      return mlir::success();
    }
  };

  template<typename Op>
  struct GetVariableLikeOpLowering : public IDAOpConversion<Op>
  {
    using IDAOpConversion<Op>::IDAOpConversion;

    virtual std::string getRuntimeFunctionName() const = 0;

    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto arrayDescriptorType = op.getResult().getType().template cast<mlir::LLVM::LLVMStructType>();
      auto dataPtrType = arrayDescriptorType.getBody()[0];

      auto loc = op.getLoc();
      RuntimeFunctionsMangling mangling;

      auto resultType = this->getVoidPtrType();

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

      mlir::Value ptr = rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands).getResult(0);
      ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, dataPtrType, ptr);

      // Create the array descriptor
      auto arrayDescriptor = ArrayDescriptor::undef(rewriter, &this->typeConverter(), loc, arrayDescriptorType);
      arrayDescriptor.setPtr(rewriter, loc, ptr);

      rewriter.replaceOp(op, *arrayDescriptor);
      return mlir::success();
    }
  };

  struct GetVariableOpLowering : public GetVariableLikeOpLowering<GetVariableOp>
  {
    using GetVariableLikeOpLowering<GetVariableOp>::GetVariableLikeOpLowering;

    std::string getRuntimeFunctionName() const override
    {
      return "idaGetVariable";
    }
  };

  struct GetDerivativeOpLowering : public GetVariableLikeOpLowering<GetDerivativeOp>
  {
    using GetVariableLikeOpLowering<GetDerivativeOp>::GetVariableLikeOpLowering;

    std::string getRuntimeFunctionName() const override
    {
      return "idaGetDerivative";
    }
  };

  struct ResidualFunctionOpLowering : public IDAOpConversion<ResidualFunctionOp>
  {
    using IDAOpConversion<ResidualFunctionOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(ResidualFunctionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::SmallVector<mlir::Type, 3> argsTypes;
      mlir::SmallVector<mlir::Type, 1> resultsTypes;

      argsTypes.push_back(op.getTime().getType());
      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));

      resultsTypes.push_back(op.getType().getResult(0));

      auto functionType = rewriter.getFunctionType(argsTypes, resultsTypes);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::FuncOp>(op, op.name(), functionType);
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
        mlir::Value index = rewriter.create<mlir::ConstantOp>(variablesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variable.index()));
        mlir::Value variablePtr = rewriter.create<mlir::LLVM::GEPOp>(variablesPtr.getLoc(), variablesPtr.getType(), variablesPtr, index);
        mlir::Value mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(variablePtr.getLoc(), variablePtr);
        mappedVariable = rewriter.create<mlir::LLVM::BitcastOp>(mappedVariable.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getVariables()[variable.index()].getType()), mappedVariable);
        mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(mappedVariable.getLoc(), mappedVariable);

        mapping.map(variable.value(), mappedVariable);
      }

      // The equation indices are also passed through an array
      mlir::Value equationIndicesPtr = newOp.getArgument(2);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::ConstantOp>(equationIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));
        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(equationIndicesPtr.getLoc(), equationIndicesPtr.getType(), equationIndicesPtr, index);
        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(equationIndexPtr.getLoc(), equationIndexPtr);
        mapping.map(equationIndex.value(), mappedEquationIndex);
      }

      // Clone the original operations
      assert(op.bodyRegion().getBlocks().size() == 1);

      for (auto& bodyOp : op.bodyRegion().getOps()) {
        rewriter.clone(bodyOp, mapping);
      }

      return mlir::success();
    }
  };

  struct JacobianFunctionOpLowering : public IDAOpConversion<JacobianFunctionOp>
  {
    using IDAOpConversion<JacobianFunctionOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(JacobianFunctionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::SmallVector<mlir::Type, 5> argsTypes;
      mlir::SmallVector<mlir::Type, 1> resultsTypes;

      argsTypes.push_back(op.getTime().getType());
      argsTypes.push_back(getVoidPtrType());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(getTypeConverter()->getIndexType()));
      argsTypes.push_back(op.getAlpha().getType());

      resultsTypes.push_back(op.getType().getResult(0));

      auto functionType = rewriter.getFunctionType(argsTypes, resultsTypes);

      auto newOp = rewriter.replaceOpWithNewOp<mlir::FuncOp>(op, op.name(), functionType);
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
        mlir::Value index = rewriter.create<mlir::ConstantOp>(variablesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variable.index()));
        mlir::Value variablePtr = rewriter.create<mlir::LLVM::GEPOp>(variablesPtr.getLoc(), variablesPtr.getType(), variablesPtr, index);
        mlir::Value mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(variablePtr.getLoc(), variablePtr);
        mappedVariable = rewriter.create<mlir::LLVM::BitcastOp>(mappedVariable.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getVariables()[variable.index()].getType()), mappedVariable);
        mappedVariable = rewriter.create<mlir::LLVM::LoadOp>(mappedVariable.getLoc(), mappedVariable);

        mapping.map(variable.value(), mappedVariable);
      }

      // The equation indices are also passed through an array
      mlir::Value equationIndicesPtr = newOp.getArgument(2);

      for (auto equationIndex : llvm::enumerate(op.getEquationIndices())) {
        mlir::Value index = rewriter.create<mlir::ConstantOp>(equationIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), equationIndex.index()));
        mlir::Value equationIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(equationIndicesPtr.getLoc(), equationIndicesPtr.getType(), equationIndicesPtr, index);
        mlir::Value mappedEquationIndex = rewriter.create<mlir::LLVM::LoadOp>(equationIndexPtr.getLoc(), equationIndexPtr);
        mapping.map(equationIndex.value(), mappedEquationIndex);
      }

      // The variable indices are also passed through an array
      mlir::Value variableIndicesPtr = newOp.getArgument(3);

      for (auto variableIndex : llvm::enumerate(op.getVariableIndices())) {
        mlir::Value index = rewriter.create<mlir::ConstantOp>(equationIndicesPtr.getLoc(), rewriter.getIntegerAttr(getIndexType(), variableIndex.index()));
        mlir::Value variableIndexPtr = rewriter.create<mlir::LLVM::GEPOp>(variableIndicesPtr.getLoc(), variableIndicesPtr.getType(), variableIndicesPtr, index);
        mlir::Value mappedVariableIndex = rewriter.create<mlir::LLVM::LoadOp>(variableIndexPtr.getLoc(), variableIndexPtr);
        mapping.map(variableIndex.value(), mappedVariableIndex);
      }

      // Add the "alpha" variable
      mapping.map(op.getAlpha(), newOp.getArgument(4));

      // Clone the original operations
      assert(op.bodyRegion().getBlocks().size() == 1);

      for (auto& bodyOp : op.bodyRegion().getOps()) {
        rewriter.clone(bodyOp, mapping);
      }

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

  struct AddResidualOpLowering : public IDAOpConversion<AddResidualOp>
  {
    using IDAOpConversion<AddResidualOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddResidualOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      RuntimeFunctionsMangling mangling;

      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.function());

      if (!function) {
        return rewriter.notifyMatchFailure(op, "Residual function " + op.function().str() + " not found");
      }

      mlir::Value functionAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, function);
      functionAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), functionAddress);

      // Call the runtime library
      auto mangledResultType = mangling.getVoidType();

      assert(operands.size() == 2);
      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(operands[0]);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation indices
      newOperands.push_back(operands[1]);
      mangledArgsTypes.push_back(mangling.getIntegerType(newOperands[1].getType().getIntOrFloatBitWidth()));

      // Residual function address
      newOperands.push_back(functionAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      auto functionName = mangling.getMangledFunction("idaAddResidual", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };

  struct AddJacobianOpLowering : public IDAOpConversion<AddJacobianOp>
  {
    using IDAOpConversion<AddJacobianOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(AddJacobianOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      RuntimeFunctionsMangling mangling;

      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.function());

      if (!function) {
        return rewriter.notifyMatchFailure(op, "Jacobian function " + op.function().str() + " not found");
      }

      mlir::Value functionAddress = rewriter.create<mlir::LLVM::AddressOfOp>(loc, function);
      functionAddress = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), functionAddress);

      // Call the runtime library
      auto mangledResultType = mangling.getVoidType();

      assert(operands.size() == 3);
      llvm::SmallVector<mlir::Value, 3> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(operands[0]);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation indices
      newOperands.push_back(operands[1]);
      mangledArgsTypes.push_back(mangling.getIntegerType(newOperands[1].getType().getIntOrFloatBitWidth()));

      // Variable indices
      newOperands.push_back(operands[2]);
      mangledArgsTypes.push_back(mangling.getIntegerType(newOperands[2].getType().getIntOrFloatBitWidth()));

      // Jacobian function address
      newOperands.push_back(functionAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      auto functionName = mangling.getMangledFunction("idaAddJacobian", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };

  struct InitOpLowering : public IDAOpConversion<InitOp>
  {
    using IDAOpConversion<InitOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(InitOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      RuntimeFunctionsMangling mangling;

      auto mangledResultType = mangling.getVoidType();

      assert(operands.size() == 1);
      llvm::SmallVector<std::string, 1> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      auto functionName = mangling.getMangledFunction("idaInit", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), operands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, operands);
      return mlir::success();
    }
  };

  struct StepOpLowering : public IDAOpConversion<StepOp>
  {
    using IDAOpConversion<StepOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(StepOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      RuntimeFunctionsMangling mangling;

      auto mangledResultType = mangling.getVoidType();

      assert(operands.size() == 1);
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      newOperands.push_back(operands[0]);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      auto functionName = mangling.getMangledFunction("idaStep", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };

  struct FreeOpLowering : public IDAOpConversion<FreeOp>
  {
    using IDAOpConversion<FreeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(FreeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      RuntimeFunctionsMangling mangling;

      auto mangledResultType = mangling.getVoidType();

      assert(operands.size() == 1);
      llvm::SmallVector<std::string, 1> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      auto functionName = mangling.getMangledFunction("idaFree", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), operands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, operands);
      return mlir::success();
    }
  };

  struct PrintStatisticsOpLowering : public IDAOpConversion<PrintStatisticsOp>
  {
    using IDAOpConversion<PrintStatisticsOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(PrintStatisticsOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      RuntimeFunctionsMangling mangling;

      auto mangledResultType = mangling.getVoidType();

      assert(operands.size() == 1);
      llvm::SmallVector<std::string, 1> mangledArgsTypes;
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      auto functionName = mangling.getMangledFunction("idaPrintStatistics", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          functionName, getVoidType(), operands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, operands);
      return mlir::success();
    }
  };
}

static void populateIDAResidualAndJacobianConversionPatterns(
    mlir::OwningRewritePatternList& patterns,
    mlir::ida::TypeConverter& typeConverter,
    unsigned int bitWidth)
{
  patterns.insert<
      VariableSetterOpLowering,
      ResidualFunctionOpLowering,
      JacobianFunctionOpLowering>(typeConverter, bitWidth);
}

static void populateIDAConversionPatterns(
    mlir::OwningRewritePatternList& patterns,
    mlir::ida::TypeConverter& typeConverter,
    unsigned int bitWidth)
{
  patterns.insert<
      CreateOpLowering,
      SetStartTimeOpLowering,
      SetEndTimeOpLowering,
      SetTimeStepOpLowering,
      SetRelativeToleranceOpLowering,
      SetAbsoluteToleranceOpLowering,
      GetCurrentTimeOpLowering,
      AddEquationOpLowering,
      AddAlgebraicVariableOpLowering,
      AddStateVariableOpLowering,
      AddVariableAccessOpLowering,
      GetVariableOpLowering,
      GetDerivativeOpLowering,
      ReturnOpLowering,
      AddResidualOpLowering,
      AddJacobianOpLowering,
      InitOpLowering,
      StepOpLowering,
      FreeOpLowering,
      PrintStatisticsOpLowering>(typeConverter, bitWidth);
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
        // Convert the Residual and Jacobian functions.
        // This must be done first and as an independent step, as other operations within
        // the IDA dialect will need the addresses of such functions when being converted.

        if (mlir::failed(convertFunctionsLikeOps())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the Residual and Jacobian functions");
          return signalPassFailure();
        }

        // Convert the rest of the IDA dialect.

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
        target.addIllegalOp<mlir::FuncOp>();

        target.addIllegalOp<
            VariableSetterOp,
            ResidualFunctionOp,
            JacobianFunctionOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::OwningRewritePatternList patterns(&getContext());
        populateIDAResidualAndJacobianConversionPatterns(patterns, typeConverter, 64);
        mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

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
        target.addIllegalDialect<mlir::StandardOpsDialect>();
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::OwningRewritePatternList patterns(&getContext());
        populateIDAConversionPatterns(patterns, typeConverter, 64);
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
  struct AddAlgebraicVariableOpTypes : public mlir::OpConversionPattern<AddAlgebraicVariableOp>
  {
    using mlir::OpConversionPattern<AddAlgebraicVariableOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AddAlgebraicVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<AddAlgebraicVariableOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(operands);

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

    mlir::LogicalResult matchAndRewrite(AddStateVariableOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<AddStateVariableOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(operands);

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  template<typename Op>
  struct GetVariableLikeOpTypes : public mlir::ConvertOpToLLVMPattern<Op>
  {
    using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      assert(op.getResult().getType().template isa<mlir::modelica::ArrayType>());
      auto arrayType = op.getResult().getType().template cast<mlir::modelica::ArrayType>();
      auto arrayDescriptorType = this->getTypeConverter()->convertType(arrayType).template cast<mlir::LLVM::LLVMStructType>();

      auto loc = op.getLoc();

      auto newOp = mlir::cast<Op>(rewriter.cloneWithoutRegions(*op.getOperation()));
      newOp->setOperands(operands);

      for (auto result : newOp->getResults()) {
        result.setType(arrayDescriptorType);
      }

      assert(newOp->getNumResults() == 1);
      ArrayDescriptor arrayDescriptor(
          this->getTypeConverter(),
          newOp->getResults()[0]);

      mlir::Value rank = rewriter.create<mlir::ConstantOp>(
          loc, arrayDescriptor.getRankType(),
          rewriter.getIntegerAttr(arrayDescriptor.getRankType(), arrayType.getRank()));

      arrayDescriptor.setRank(rewriter, loc, rank);

      assert(arrayType.hasConstantShape());

      for (auto size : llvm::enumerate(arrayType.getShape())) {
        mlir::Value sizeValue = rewriter.create<mlir::ConstantOp>(
            loc, arrayDescriptor.getSizeType(),
            rewriter.getIntegerAttr(arrayDescriptor.getSizeType(), size.value()));

        arrayDescriptor.setSize(rewriter, loc, size.index(), sizeValue);
      }

      rewriter.replaceOp(op, *arrayDescriptor);
      return mlir::success();
    }
  };

  struct GetVariableOpTypes : public GetVariableLikeOpTypes<GetVariableOp>
  {
    using GetVariableLikeOpTypes<GetVariableOp>::GetVariableLikeOpTypes;
  };

  struct GetDerivativeOpTypes : public GetVariableLikeOpTypes<GetDerivativeOp>
  {
    using GetVariableLikeOpTypes<GetDerivativeOp>::GetVariableLikeOpTypes;
  };

  struct ConvertGetCurrentTimeOpTypes : public mlir::OpConversionPattern<GetCurrentTimeOp>
  {
    using mlir::OpConversionPattern<GetCurrentTimeOp>::OpConversionPattern;

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

  struct VariableSetterOpTypes : public mlir::OpConversionPattern<VariableSetterOp>
  {
    using mlir::OpConversionPattern<VariableSetterOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(VariableSetterOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type variableType = getTypeConverter()->convertType(op.getVariable().getType());
      mlir::Type valueType = getTypeConverter()->convertType(op.getValue().getType());

      auto newOp = rewriter.replaceOpWithNewOp<VariableSetterOp>(
          op, op.name(), variableType, valueType, op.getVariableIndices().size());

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

    mlir::LogicalResult matchAndRewrite(ResidualFunctionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType = getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type differenceType = getTypeConverter()->convertType(op.getType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<ResidualFunctionOp>(
          op, op.name(), timeType, variablesTypes, op.equationRank().getSExtValue(), differenceType);

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

    mlir::LogicalResult matchAndRewrite(JacobianFunctionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType = getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type alphaType = getTypeConverter()->convertType(op.getAlpha().getType());
      mlir::Type resultType = getTypeConverter()->convertType(op.getType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<JacobianFunctionOp>(
          op, op.name(), timeType, variablesTypes,
          op.equationRank().getSExtValue(),
          op.variableRank().getSExtValue(),
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
  std::unique_ptr<mlir::Pass> createIDAConversionPass()
  {
    return std::make_unique<IDAConversionPass>();
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
        AddStateVariableOpTypes>(typeConverter, patterns.getContext());

    patterns.add<
        GetVariableOpTypes,
        GetDerivativeOpTypes>(typeConverter);

    patterns.add<ConvertGetCurrentTimeOpTypes>(typeConverter, patterns.getContext());

    patterns.add<
        VariableSetterOpTypes,
        ResidualFunctionOpTypes,
        JacobianFunctionOpTypes>(typeConverter, patterns.getContext());

    target.addDynamicallyLegalOp<AddAlgebraicVariableOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<AddStateVariableOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<GetVariableOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<GetDerivativeOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
    });

    target.addDynamicallyLegalOp<GetCurrentTimeOp>([&](mlir::Operation* op) {
      return typeConverter.isLegal(op);
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

      if (!typeConverter.isLegal(op.getType().getResult(0))) {
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

      if (!typeConverter.isLegal(op.getType().getResult(0))) {
        return false;
      }

      return true;
    });
  }
}
