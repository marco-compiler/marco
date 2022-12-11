#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/IDACommon/LLVMTypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir
{
#define GEN_PASS_DEF_IDATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::ida;

static mlir::LLVM::LLVMFuncOp getOrDeclareLLVMFunction(
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

  return builder.create<mlir::LLVM::LLVMFuncOp>(
      loc, name, mlir::LLVM::LLVMFunctionType::get(result, args));
}

static mlir::LLVM::LLVMFuncOp getOrDeclareLLVMFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    mlir::Location loc,
    llvm::StringRef name,
    mlir::Type result,
    mlir::ValueRange args)
{
  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (mlir::Type type : args.getTypes()) {
    argsTypes.push_back(type);
  }

  return getOrDeclareLLVMFunction(
      builder, module, loc, name, result, argsTypes);
}

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class IDAOpConversion : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      IDAOpConversion(
        mlir::LLVMTypeConverter& typeConverter,
        unsigned int bitWidth)
        : mlir::ConvertOpToLLVMPattern<Op>(typeConverter),
          bitWidth(bitWidth)
      {
      }

      mlir::ida::LLVMTypeConverter& typeConverter() const
      {
        return *static_cast<mlir::ida::LLVMTypeConverter*>(
            this->getTypeConverter());
      }

      mlir::Type convertType(mlir::Type type) const
      {
        return typeConverter().convertType(type);
      }

      mlir::Value materializeTargetConversion(
          mlir::OpBuilder& builder, mlir::Value value) const
      {
        mlir::Type type =
            this->getTypeConverter()->convertType(value.getType());

        return this->getTypeConverter()->materializeTargetConversion(
            builder, value.getLoc(), type, value);
      }

      void materializeTargetConversion(
          mlir::OpBuilder& builder,
          llvm::SmallVectorImpl<mlir::Value>& values) const
      {
        for (auto& value : values) {
          value = materializeTargetConversion(builder, value);
        }
      }

      mlir::Value allocRangesArray(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp module,
          mlir::ArrayAttr ranges) const
      {
        mlir::Type dimensionSizeType =
            this->getTypeConverter()->convertType(builder.getI64Type());

        mlir::Value numOfElements = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIntegerAttr(
                     this->getIndexType(), ranges.size() * 2));

        mlir::Type elementPtrType =
            mlir::LLVM::LLVMPointerType::get(dimensionSizeType);

        mlir::Value nullPtr =
            builder.create<mlir::LLVM::NullOp>(loc, elementPtrType);

        mlir::Value gepPtr = builder.create<mlir::LLVM::GEPOp>(
            loc, elementPtrType, nullPtr, numOfElements);

        mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(
            loc, this->getIndexType(), gepPtr);

        auto heapAllocFn = lookupOrCreateHeapAllocFn(module, this->getIndexType());

        auto allocCall = builder.create<mlir::LLVM::CallOp>(
            loc, heapAllocFn, sizeBytes);

        mlir::Value rangesPtr = builder.create<mlir::LLVM::BitcastOp>(
            loc, elementPtrType, allocCall.getResult());

        // Populate the array.
        for (const auto& range : llvm::enumerate(ranges)) {
          auto rangeAttr = range.value().cast<mlir::ArrayAttr>();
          assert(rangeAttr.size() == 2);

          for (const auto& index : llvm::enumerate(rangeAttr)) {
            auto indexAttr = index.value().cast<mlir::IntegerAttr>();

            mlir::Value offset = builder.create<mlir::arith::ConstantOp>(
                loc, builder.getIntegerAttr(
                         this->getIndexType(),
                         range.index() * 2 + index.index()));

            mlir::Value indexValue = builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI64IntegerAttr(indexAttr.getInt()));

            mlir::Value ptr = builder.create<mlir::LLVM::GEPOp>(
                loc, rangesPtr.getType(), rangesPtr, offset);

            builder.create<mlir::LLVM::StoreOp>(loc, indexValue, ptr);
          }
        }

        return rangesPtr;
      }

      mlir::Value allocAccessArray(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp module,
          mlir::AffineMap access) const
      {
        auto dimensionAccesses = access.getResults();
        llvm::SmallVector<mlir::Value, 6> accessValues;

        for (const auto dimensionAccess : dimensionAccesses) {
          if (dimensionAccess.isa<mlir::AffineConstantExpr>()) {
            auto constantAccess =
                dimensionAccess.cast<mlir::AffineConstantExpr>();

            accessValues.push_back(builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI64IntegerAttr(-1)));

            accessValues.push_back(builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI64IntegerAttr(constantAccess.getValue())));

          } else if (dimensionAccess.isa<mlir::AffineDimExpr>()) {
            auto dimension = dimensionAccess.cast<mlir::AffineDimExpr>();

            accessValues.push_back(builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI64IntegerAttr(dimension.getPosition())));

            accessValues.push_back(builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI64IntegerAttr(0)));

          } else {
            auto dynamicAccess =
                dimensionAccess.cast<mlir::AffineBinaryOpExpr>();

            auto dimension =
                dynamicAccess.getLHS().cast<mlir::AffineDimExpr>();

            auto offset =
                dynamicAccess.getRHS().cast<mlir::AffineConstantExpr>();

            accessValues.push_back(builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI64IntegerAttr(dimension.getPosition())));

            accessValues.push_back(builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI64IntegerAttr(offset.getValue())));
          }
        }

        mlir::Type dimensionSizeType =
            this->getTypeConverter()->convertType(builder.getI64Type());

        mlir::Value numOfElements = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIntegerAttr(this->getIndexType(),
                                        accessValues.size()));

        mlir::Type elementPtrType =
            mlir::LLVM::LLVMPointerType::get(dimensionSizeType);

        mlir::Value nullPtr =
            builder.create<mlir::LLVM::NullOp>(loc, elementPtrType);

        mlir::Value gepPtr = builder.create<mlir::LLVM::GEPOp>(
            loc, elementPtrType, nullPtr, numOfElements);

        mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(
            loc, this->getIndexType(), gepPtr);

        auto heapAllocFn = lookupOrCreateHeapAllocFn(
            module, this->getIndexType());

        auto allocCall = builder.create<mlir::LLVM::CallOp>(
            loc, heapAllocFn, sizeBytes);

        mlir::Value accessesPtr = builder.create<mlir::LLVM::BitcastOp>(
            loc, elementPtrType, allocCall.getResult());

        for (const auto& accessValue : llvm::enumerate(accessValues)) {
          mlir::Value offset = builder.create<mlir::arith::ConstantOp>(
              loc, builder.getIntegerAttr(
                       this->getIndexType(), accessValue.index()));

          mlir::Value ptr = builder.create<mlir::LLVM::GEPOp>(
              loc, accessesPtr.getType(), accessesPtr, offset);

          builder.create<mlir::LLVM::StoreOp>(loc, accessValue.value(), ptr);
        }

        return accessesPtr;
      }

      void deallocate(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value ptr) const
      {
        mlir::Location loc = ptr.getLoc();
        auto heapFreeFn = lookupOrCreateHeapFreeFn(module);

        mlir::Value opaquePtr = builder.create<mlir::LLVM::BitcastOp>(
            loc, this->getVoidPtrType(), ptr);

        builder.create<mlir::LLVM::CallOp>(loc, heapFreeFn, opaquePtr);
      }

      mlir::Value getFunctionAddress(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::SymbolRefAttr functionName) const
      {
        auto funcOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(functionName);
        mlir::Location loc = funcOp.getLoc();

        mlir::Value address =
            builder.create<mlir::LLVM::AddressOfOp>(loc, funcOp);

        mlir::Value opaquePtr = builder.create<mlir::LLVM::BitcastOp>(
            loc, this->getVoidPtrType(), address);

        return opaquePtr;
      }

    protected:
      unsigned int bitWidth;
  };

  struct CreateOpLowering : public IDAOpConversion<CreateOp>
  {
    using IDAOpConversion<CreateOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        CreateOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Scalar equations amount.
      mlir::Value scalarEquationsAmount =
          rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getI64IntegerAttr(op.getScalarEquations()));

      newOperands.push_back(scalarEquationsAmount);

      mangledArgsTypes.push_back(mangling.getIntegerType(
          scalarEquationsAmount.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library.
      auto resultType = getVoidPtrType();
      auto mangledResultType = mangling.getVoidPointerType();

      auto functionName = mangling.getMangledFunction(
          "idaCreate", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct SetStartTimeOpLowering : public IDAOpConversion<SetStartTimeOp>
  {
    using IDAOpConversion<SetStartTimeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        SetStartTimeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Start time.
      mlir::Value startTime = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getF64FloatAttr(op.getTime().convertToDouble()));

      newOperands.push_back(startTime);

      mangledArgsTypes.push_back(mangling.getFloatingPointType(
          startTime.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaSetStartTime", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct SetEndTimeOpLowering : public IDAOpConversion<SetEndTimeOp>
  {
    using IDAOpConversion<SetEndTimeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        SetEndTimeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // End time.
      mlir::Value endTime = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getF64FloatAttr(op.getTime().convertToDouble()));

      newOperands.push_back(endTime);

      mangledArgsTypes.push_back(mangling.getFloatingPointType(
          endTime.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaSetEndTime", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct GetCurrentTimeOpLowering : public IDAOpConversion<GetCurrentTimeOp>
  {
    using IDAOpConversion<GetCurrentTimeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        GetCurrentTimeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = rewriter.getF64Type();

      auto mangledResultType = mangling.getFloatingPointType(
          resultType.getIntOrFloatBitWidth());

      auto functionName = mangling.getMangledFunction(
          "idaGetCurrentTime", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      auto callOp = rewriter.create<mlir::LLVM::CallOp>(
          loc, callee, newOperands);

      mlir::Value result = callOp.getResult();

      auto requiredResultType =
          getTypeConverter()->convertType(op.getResult().getType());

      if (requiredResultType.getIntOrFloatBitWidth() <
          resultType.getIntOrFloatBitWidth()) {
        result = rewriter.create<mlir::LLVM::FPTruncOp>(
            loc, requiredResultType, result);
      } else if (requiredResultType.getIntOrFloatBitWidth() >
                 resultType.getIntOrFloatBitWidth()) {
        result = rewriter.create<mlir::LLVM::FPExtOp>(
            loc, requiredResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct AddEquationOpLowering : public IDAOpConversion<AddEquationOp>
  {
    using IDAOpConversion<AddEquationOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        AddEquationOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 5> newOperands;
      llvm::SmallVector<std::string, 5> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the array with the equation ranges.
      mlir::Value equationRangesPtr = allocRangesArray(
          rewriter, loc, module, op.getEquationRanges());

      newOperands.push_back(equationRangesPtr);

      mangledArgsTypes.push_back(
          mangling.getPointerType(
              mangling.getIntegerType(
                  equationRangesPtr.getType()
                      .cast<mlir::LLVM::LLVMPointerType>()
                      .getElementType()
                      .getIntOrFloatBitWidth())));

      // Rank.
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(op.getEquationRanges().size()));

      newOperands.push_back(rank);

      mangledArgsTypes.push_back(mangling.getIntegerType(
          rank.getType().getIntOrFloatBitWidth()));

      // Written variable.
      newOperands.push_back(adaptor.getWrittenVariable());

      mangledArgsTypes.push_back(mangling.getIntegerType(
          adaptor.getWrittenVariable().getType().getIntOrFloatBitWidth()));

      // Write access.
      mlir::Value writeAccessPtr = allocAccessArray(
          rewriter, loc, module, op.getWriteAccess());

      newOperands.push_back(writeAccessPtr);

      mangledArgsTypes.push_back(
          mangling.getPointerType(
              mangling.getIntegerType(
                  writeAccessPtr.getType()
                      .cast<mlir::LLVM::LLVMPointerType>()
                      .getElementType()
                      .getIntOrFloatBitWidth())));

      // Create the call to the runtime library.
      auto resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto mangledResultType =
          mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

      auto functionName = mangling.getMangledFunction(
          "idaAddEquation", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the equation ranges array.
      deallocate(rewriter, module, equationRangesPtr);
      deallocate(rewriter, module, writeAccessPtr);

      return mlir::success();
    }
  };

  template<typename Op>
  struct AddVariableOpLowering : public IDAOpConversion<Op>
  {
    using IDAOpConversion<Op>::IDAOpConversion;

    mlir::Value allocVariableDescriptor(
        mlir::OpBuilder& builder,
        mlir::ModuleOp module,
        mlir::Value variable) const
    {
      mlir::Location loc = variable.getLoc();

      mlir::Type ptrType =
          mlir::LLVM::LLVMPointerType::get(variable.getType());

      mlir::Value nullPtr =
          builder.create<mlir::LLVM::NullOp>(loc, ptrType);

      mlir::Value one = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIntegerAttr(this->getIndexType(), 1));

      mlir::Value gepPtr =
          builder.create<mlir::LLVM::GEPOp>(loc, ptrType, nullPtr, one);

      mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(
          loc, this->getIndexType(), gepPtr);

      auto heapAllocFn =
          lookupOrCreateHeapAllocFn(module, this->getIndexType());

      auto allocCall =
          builder.create<mlir::LLVM::CallOp>(loc, heapAllocFn, sizeBytes);

      mlir::Value opaquePtr = allocCall.getResult();

      mlir::Value ptr =
          builder.create<mlir::LLVM::BitcastOp>(loc, ptrType, opaquePtr);

      builder.create<mlir::LLVM::StoreOp>(loc, variable, ptr);

      return opaquePtr;
    }

    mlir::Value allocVariableDimensionsArray(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::ModuleOp module,
        mlir::ArrayAttr dimensions) const
    {
      mlir::Type dimensionSizeType =
          this->getTypeConverter()->convertType(builder.getI64Type());

      mlir::Value numOfElements = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIntegerAttr(
                   this->getIndexType(), dimensions.size()));

      mlir::Type elementPtrType =
          mlir::LLVM::LLVMPointerType::get(dimensionSizeType);

      mlir::Value nullPtr =
          builder.create<mlir::LLVM::NullOp>(loc, elementPtrType);

      mlir::Value gepPtr = builder.create<mlir::LLVM::GEPOp>(
          loc, elementPtrType, nullPtr, numOfElements);

      mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(
          loc, this->getIndexType(), gepPtr);

      auto heapAllocFn =
          lookupOrCreateHeapAllocFn(module, this->getIndexType());

      auto allocCallOp = builder.create<mlir::LLVM::CallOp>(
          loc, heapAllocFn, sizeBytes);

      mlir::Value arrayDimensionsPtr = builder.create<mlir::LLVM::BitcastOp>(
          loc, elementPtrType, allocCallOp.getResult());

      // Populate the dimensions list.
      for (const auto& sizeAttr : llvm::enumerate(dimensions)) {
        mlir::Value offset = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIntegerAttr(
                     this->getIndexType(), sizeAttr.index()));

        mlir::Value size = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getI64IntegerAttr(
                     sizeAttr.value().cast<mlir::IntegerAttr>().getInt()));

        mlir::Value ptr = builder.create<mlir::LLVM::GEPOp>(
            loc, arrayDimensionsPtr.getType(), arrayDimensionsPtr, offset);

        builder.create<mlir::LLVM::StoreOp>(loc, size, ptr);
      }

      return arrayDimensionsPtr;
    }
  };

  struct AddAlgebraicVariableOpLowering
      : public AddVariableOpLowering<AddAlgebraicVariableOp>
  {
    using AddVariableOpLowering<AddAlgebraicVariableOp>::AddVariableOpLowering;

    mlir::LogicalResult matchAndRewrite(
        AddAlgebraicVariableOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 6> newOperands;
      llvm::SmallVector<std::string, 6> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable.
      newOperands.push_back(allocVariableDescriptor(
          rewriter, module, adaptor.getVariable()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the array with the variable dimensions.
      mlir::Value dimensionsPtr = allocVariableDimensionsArray(
          rewriter, loc, module, op.getDimensions());

      newOperands.push_back(dimensionsPtr);

      mangledArgsTypes.push_back(mangling.getPointerType(
          mangling.getIntegerType(
              dimensionsPtr.getType()
                  .cast<mlir::LLVM::LLVMPointerType>()
                  .getElementType().getIntOrFloatBitWidth())));

      // Rank.
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(op.getDimensions().size()));

      newOperands.push_back(rank);

      mangledArgsTypes.push_back(mangling.getIntegerType(
          rank.getType().getIntOrFloatBitWidth()));

      // Variable getter function address.
      newOperands.push_back(getFunctionAddress(
          rewriter, module, op.getGetter()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable setter function address.
      newOperands.push_back(getFunctionAddress(
          rewriter, module, op.getSetter()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto mangledResultType =
          mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

      auto functionName = mangling.getMangledFunction(
          "idaAddAlgebraicVariable", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the dimensions array.
      deallocate(rewriter, module, dimensionsPtr);

      return mlir::success();
    }
  };

  struct AddStateVariableOpLowering
      : public AddVariableOpLowering<AddStateVariableOp>
  {
    using AddVariableOpLowering<AddStateVariableOp>::AddVariableOpLowering;

    mlir::LogicalResult matchAndRewrite(
        AddStateVariableOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 6> newOperands;
      llvm::SmallVector<std::string, 6> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable.
      newOperands.push_back(allocVariableDescriptor(
          rewriter, module, adaptor.getVariable()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the array with the variable dimensions.
      mlir::Value dimensionsPtr = allocVariableDimensionsArray(
          rewriter, loc, module, op.getDimensions());

      newOperands.push_back(dimensionsPtr);

      mangledArgsTypes.push_back(mangling.getPointerType(
          mangling.getIntegerType(
              dimensionsPtr.getType()
                  .cast<mlir::LLVM::LLVMPointerType>()
                  .getElementType().getIntOrFloatBitWidth())));

      // Rank.
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(op.getDimensions().size()));

      newOperands.push_back(rank);

      mangledArgsTypes.push_back(mangling.getIntegerType(
          rank.getType().getIntOrFloatBitWidth()));

      // Variable getter function address.
      newOperands.push_back(getFunctionAddress(
          rewriter, module, op.getGetter()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable setter function address.
      newOperands.push_back(getFunctionAddress(
          rewriter, module, op.getSetter()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType =
          getTypeConverter()->convertType(op.getResult().getType());

      auto mangledResultType =
          mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

      auto functionName = mangling.getMangledFunction(
          "idaAddStateVariable", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the dimensions array.
      deallocate(rewriter, module, dimensionsPtr);

      return mlir::success();
    }
  };

  struct SetDerivativeOpLowering
      : public AddVariableOpLowering<SetDerivativeOp>
  {
    using AddVariableOpLowering<SetDerivativeOp>::AddVariableOpLowering;

    mlir::LogicalResult matchAndRewrite(
        SetDerivativeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 5> newOperands;
      llvm::SmallVector<std::string, 5> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // State variable.
      newOperands.push_back(adaptor.getStateVariable());

      mangledArgsTypes.push_back(mangling.getIntegerType(
          adaptor.getStateVariable().getType().getIntOrFloatBitWidth()));

      // Derivative.
      newOperands.push_back(allocVariableDescriptor(
          rewriter, module, adaptor.getDerivative()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable getter function address.
      newOperands.push_back(getFunctionAddress(
          rewriter, module, op.getGetter()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable setter function address.
      newOperands.push_back(getFunctionAddress(
          rewriter, module, op.getSetter()));

      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaSetDerivative", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct AddParametricVariableOpLowering
      : public AddVariableOpLowering<AddParametricVariableOp>
  {
    using AddVariableOpLowering<AddParametricVariableOp>::AddVariableOpLowering;

    mlir::LogicalResult matchAndRewrite(
        AddParametricVariableOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Variable.
      newOperands.push_back(allocVariableDescriptor(
          rewriter, module, adaptor.getVariable()));
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaAddParametricVariable", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct AddVariableAccessOpLowering
      : public IDAOpConversion<AddVariableAccessOp>
  {
    using IDAOpConversion<AddVariableAccessOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        AddVariableAccessOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 5> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation.
      newOperands.push_back(adaptor.getEquation());

      mangledArgsTypes.push_back(mangling.getIntegerType(
          adaptor.getEquation().getType().getIntOrFloatBitWidth()));

      // Variable.
      newOperands.push_back(adaptor.getVariable());

      mangledArgsTypes.push_back(mangling.getIntegerType(
          adaptor.getVariable().getType().getIntOrFloatBitWidth()));

      // Create the array with the variable accesses.
      mlir::Value accessesPtr = allocAccessArray(
          rewriter, loc, module, op.getAccess());

      newOperands.push_back(accessesPtr);

      mangledArgsTypes.push_back(
          mangling.getPointerType(
              mangling.getIntegerType(
                  accessesPtr.getType()
                      .cast<mlir::LLVM::LLVMPointerType>()
                      .getElementType()
                      .getIntOrFloatBitWidth())));

      // Rank.
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(op.getAccess().getResults().size()));

      newOperands.push_back(rank);

      mangledArgsTypes.push_back(mangling.getIntegerType(
          rank.getType().getIntOrFloatBitWidth()));

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaAddVariableAccess", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      // Deallocate the accesses array
      deallocate(rewriter, module, accessesPtr);

      return mlir::success();
    }
  };

  struct SetResidualOpLowering : public IDAOpConversion<SetResidualOp>
  {
    using IDAOpConversion<SetResidualOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        SetResidualOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 3> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // IDA instance
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation
      newOperands.push_back(adaptor.getEquation());

      mangledArgsTypes.push_back(mangling.getIntegerType(
          adaptor.getEquation().getType().getIntOrFloatBitWidth()));

      // Residual function address.
      auto function =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getFunction());

      mlir::Value functionAddress =
          rewriter.create<mlir::LLVM::AddressOfOp>(loc, function);

      functionAddress = rewriter.create<mlir::LLVM::BitcastOp>(
          loc, getVoidPtrType(), functionAddress);

      newOperands.push_back(functionAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaSetResidual", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct AddJacobianOpLowering : public IDAOpConversion<AddJacobianOp>
  {
    using IDAOpConversion<AddJacobianOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        AddJacobianOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 4> newOperands;
      llvm::SmallVector<std::string, 4> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Equation.
      newOperands.push_back(adaptor.getEquation());

      mangledArgsTypes.push_back(mangling.getIntegerType(
          adaptor.getEquation().getType().getIntOrFloatBitWidth()));

      // Variable
      newOperands.push_back(adaptor.getVariable());
      mangledArgsTypes.push_back(mangling.getIntegerType(
          adaptor.getVariable().getType().getIntOrFloatBitWidth()));

      // Jacobian function address.
      auto function =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.getFunction());

      mlir::Value functionAddress =
          rewriter.create<mlir::LLVM::AddressOfOp>(loc, function);

      functionAddress = rewriter.create<mlir::LLVM::BitcastOp>(
          loc, getVoidPtrType(), functionAddress);

      newOperands.push_back(functionAddress);
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaAddJacobian", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct InitOpLowering : public IDAOpConversion<InitOp>
  {
    using IDAOpConversion<InitOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        InitOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaInit", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, callee, adaptor.getOperands());

      return mlir::success();
    }
  };

  struct CalcICOpLowering : public IDAOpConversion<CalcICOp>
  {
    using IDAOpConversion<CalcICOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        CalcICOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaCalcIC", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct StepOpLowering : public IDAOpConversion<StepOp>
  {
    using IDAOpConversion<StepOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        StepOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaStep", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct FreeOpLowering : public IDAOpConversion<FreeOp>
  {
    using IDAOpConversion<FreeOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        FreeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      auto module = op->getParentOfType<mlir::ModuleOp>();

      RuntimeFunctionsMangling mangling;

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // IDA instance.
      newOperands.push_back(adaptor.getInstance());
      mangledArgsTypes.push_back(mangling.getVoidPointerType());

      // Create the call to the runtime library.
      auto resultType = getVoidType();
      auto mangledResultType = mangling.getVoidType();

      auto functionName = mangling.getMangledFunction(
          "idaFree", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };

  struct PrintStatisticsOpLowering : public IDAOpConversion<PrintStatisticsOp>
  {
    using IDAOpConversion<PrintStatisticsOp>::IDAOpConversion;

    mlir::LogicalResult matchAndRewrite(
        PrintStatisticsOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
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

      auto functionName = mangling.getMangledFunction(
          "idaPrintStatistics", mangledResultType, mangledArgsTypes);

      auto callee = getOrDeclareLLVMFunction(
          rewriter, module, loc, functionName, resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);

      return mlir::success();
    }
  };
}

static void populateIDAConversionPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::ida::LLVMTypeConverter& typeConverter,
    unsigned int bitWidth)
{
  patterns.insert<
      CreateOpLowering,
      SetStartTimeOpLowering,
      SetEndTimeOpLowering,
      GetCurrentTimeOpLowering,
      AddEquationOpLowering,
      AddAlgebraicVariableOpLowering,
      AddStateVariableOpLowering,
      SetDerivativeOpLowering,
      AddParametricVariableOpLowering,
      AddVariableAccessOpLowering,
      SetResidualOpLowering,
      AddJacobianOpLowering,
      InitOpLowering,
      CalcICOpLowering,
      StepOpLowering,
      FreeOpLowering,
      PrintStatisticsOpLowering>(typeConverter, bitWidth);
}

namespace marco::codegen
{
  class IDAToLLVMConversionPass
      : public mlir::impl::IDAToLLVMConversionPassBase<IDAToLLVMConversionPass>
  {
    public:
      using IDAToLLVMConversionPassBase::IDAToLLVMConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(
              getOperation().getLoc(),
              "Error in converting the IDA operations");

          return signalPassFailure();
        }
      }

    private:
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
        llvmLoweringOptions.dataLayout.reset(dataLayout);

        LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        mlir::RewritePatternSet patterns(&getContext());
        populateIDAConversionPatterns(patterns, typeConverter, 64);

        return applyPartialConversion(module, target, std::move(patterns));
      }
  };
}

namespace
{
  struct AddAlgebraicVariableOpTypes
      : public mlir::OpConversionPattern<AddAlgebraicVariableOp>
  {
    using mlir::OpConversionPattern<AddAlgebraicVariableOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AddAlgebraicVariableOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<AddAlgebraicVariableOp>(
          rewriter.cloneWithoutRegions(*op.getOperation()));

      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct AddStateVariableOpTypes
      : public mlir::OpConversionPattern<AddStateVariableOp>
  {
    using mlir::OpConversionPattern<AddStateVariableOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AddStateVariableOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<AddStateVariableOp>(
          rewriter.cloneWithoutRegions(*op.getOperation()));

      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct SetDerivativeOpTypes
      : public mlir::OpConversionPattern<SetDerivativeOp>
  {
    using mlir::OpConversionPattern<SetDerivativeOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        SetDerivativeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<SetDerivativeOp>(
          rewriter.cloneWithoutRegions(*op.getOperation()));

      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct AddParametricVariableOpTypes
      : public mlir::OpConversionPattern<AddParametricVariableOp>
  {
    using mlir::OpConversionPattern<AddParametricVariableOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AddParametricVariableOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<AddParametricVariableOp>(
          rewriter.cloneWithoutRegions(*op.getOperation()));

      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(getTypeConverter()->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct ConvertGetCurrentTimeOpTypes
      : public mlir::OpConversionPattern<GetCurrentTimeOp>
  {
    using mlir::OpConversionPattern<GetCurrentTimeOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        GetCurrentTimeOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<GetCurrentTimeOp>(
          rewriter.cloneWithoutRegions(*op.getOperation()));

      newOp->setOperands(adaptor.getOperands());

      for (auto result : newOp->getResults()) {
        result.setType(typeConverter->convertType(result.getType()));
      }

      rewriter.replaceOp(op, newOp->getResults());
      return mlir::success();
    }
  };

  struct VariableGetterOpTypes
      : public mlir::OpConversionPattern<VariableGetterOp>
  {
    using mlir::OpConversionPattern<VariableGetterOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableGetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type resultType =
          getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      mlir::Type variableType =
          getTypeConverter()->convertType(op.getVariable().getType());

      auto newOp = rewriter.replaceOpWithNewOp<VariableGetterOp>(
          op, op.getSymName(), resultType, variableType,
          op.getVariableIndices().size());

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getVariable());
      assert(op.getFunctionBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getFunctionBody().getOps()) {
        if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
          std::vector<mlir::Value> returnValues;

          for (auto returnValue : returnOp.operands()) {
            returnValues.push_back(
                getTypeConverter()->materializeTargetConversion(
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

  struct VariableSetterOpTypes
      : public mlir::OpConversionPattern<VariableSetterOp>
  {
    using mlir::OpConversionPattern<VariableSetterOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        VariableSetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type variableType =
          getTypeConverter()->convertType(op.getVariable().getType());

      mlir::Type valueType =
          getTypeConverter()->convertType(op.getValue().getType());

      auto newOp = rewriter.replaceOpWithNewOp<VariableSetterOp>(
          op, op.getSymName(), variableType, valueType,
          op.getVariableIndices().size());

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getVariable());
      castArgFn(op.getValue());

      assert(op.getFunctionBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getFunctionBody().getOps()) {
        rewriter.clone(bodyOp, mapping);
      }

      return mlir::success();
    }
  };

  struct ResidualFunctionOpTypes
      : public mlir::OpConversionPattern<ResidualFunctionOp>
  {
    using mlir::OpConversionPattern<ResidualFunctionOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        ResidualFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType =
          getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(
            getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type differenceType =
          getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<ResidualFunctionOp>(
          op, op.getSymName(), timeType, variablesTypes,
          op.getEquationRank().getSExtValue(), differenceType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getTime());

      for (auto variable : op.getVariables()) {
        castArgFn(variable);
      }

      assert(op.getFunctionBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getFunctionBody().getOps()) {
        if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
          std::vector<mlir::Value> returnValues;

          for (auto returnValue : returnOp.operands()) {
            returnValues.push_back(
                getTypeConverter()->materializeTargetConversion(
                    rewriter, returnOp.getLoc(), differenceType,
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

  struct JacobianFunctionOpTypes
      : public mlir::OpConversionPattern<JacobianFunctionOp>
  {
    using mlir::OpConversionPattern<JacobianFunctionOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        JacobianFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type timeType =
          getTypeConverter()->convertType(op.getTime().getType());

      llvm::SmallVector<mlir::Type> variablesTypes;

      for (auto variable : op.getVariables()) {
        variablesTypes.push_back(getTypeConverter()->convertType(variable.getType()));
      }

      mlir::Type alphaType =
          getTypeConverter()->convertType(op.getAlpha().getType());

      mlir::Type resultType =
          getTypeConverter()->convertType(op.getFunctionType().getResult(0));

      auto newOp = rewriter.replaceOpWithNewOp<JacobianFunctionOp>(
          op, op.getSymName(), timeType, variablesTypes,
          op.getEquationRank().getSExtValue(),
          op.getVariableRank().getSExtValue(),
          alphaType, resultType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      for (const auto& [original, cloned]
           : llvm::zip(op.getFunctionBody().getArguments(),
                       newOp.getFunctionBody().getArguments())) {
        mapping.map(original, cloned);
      }

      auto castArgFn = [&](mlir::Value originalArg) {
        auto castedClonedArg = getTypeConverter()->materializeSourceConversion(
            rewriter, originalArg.getLoc(), originalArg.getType(),
            mapping.lookup(originalArg));

        mapping.map(originalArg, castedClonedArg);
      };

      castArgFn(op.getTime());

      for (auto variable : op.getVariables()) {
        castArgFn(variable);
      }

      castArgFn(op.getAlpha());

      assert(op.getFunctionBody().getBlocks().size() == 1);

      for (auto& bodyOp : op.getFunctionBody().getOps()) {
        if (auto returnOp = mlir::dyn_cast<ReturnOp>(bodyOp)) {
          std::vector<mlir::Value> returnValues;

          for (auto returnValue : returnOp.operands()) {
            returnValues.push_back(
                getTypeConverter()->materializeTargetConversion(
                    rewriter, returnOp.getLoc(), resultType,
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
}

namespace mlir
{
  void populateIDAToLLVMStructuralTypeConversionsAndLegality(
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
        SetDerivativeOpTypes,
        AddParametricVariableOpTypes>(typeConverter, patterns.getContext());

    patterns.add<ConvertGetCurrentTimeOpTypes>(typeConverter, patterns.getContext());

    patterns.add<
        VariableGetterOpTypes,
        VariableSetterOpTypes,
        ResidualFunctionOpTypes,
        JacobianFunctionOpTypes>(typeConverter, patterns.getContext());

    target.addDynamicallyLegalOp<AddAlgebraicVariableOp>(
        [&](mlir::Operation* op) {
          return typeConverter.isLegal(op);
        });

    target.addDynamicallyLegalOp<AddStateVariableOp>(
        [&](mlir::Operation* op) {
          return typeConverter.isLegal(op);
        });

    target.addDynamicallyLegalOp<SetDerivativeOp>(
        [&](mlir::Operation* op) {
          return typeConverter.isLegal(op);
        });

    target.addDynamicallyLegalOp<AddParametricVariableOp>(
        [&](mlir::Operation* op) {
          return typeConverter.isLegal(op);
        });

    target.addDynamicallyLegalOp<GetCurrentTimeOp>(
        [&](mlir::Operation* op) {
          return typeConverter.isLegal(op);
        });
  }

  std::unique_ptr<mlir::Pass> createIDAToLLVMConversionPass()
  {
    return std::make_unique<IDAToLLVMConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createIDAToLLVMConversionPass(
      const IDAToLLVMConversionPassOptions& options)
  {
    return std::make_unique<IDAToLLVMConversionPass>(options);
  }
}
