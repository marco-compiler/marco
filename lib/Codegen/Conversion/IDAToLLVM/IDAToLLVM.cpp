#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/IDACommon/LLVMTypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_IDATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::ida;

namespace {
/// Generic conversion pattern that provides some utility functions.
template <typename Op>
class IDAOpConversion : public mlir::ConvertOpToLLVMPattern<Op> {
public:
  IDAOpConversion(mlir::LLVMTypeConverter &typeConverter,
                  mlir::SymbolTableCollection &symbolTableCollection)
      : mlir::ConvertOpToLLVMPattern<Op>(typeConverter),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::Value getInstance(mlir::OpBuilder &builder, mlir::Location loc,
                          llvm::StringRef name) const {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    mlir::Value address =
        builder.create<mlir::LLVM::AddressOfOp>(loc, ptrType, name);

    mlir::Type elementType = this->getTypeConverter()->convertType(
        InstanceType::get(builder.getContext()));

    return builder.create<mlir::LLVM::LoadOp>(loc, elementType, address);
  }

  mlir::LLVM::GlobalOp declareDimensionsArray(mlir::OpBuilder &builder,
                                              mlir::ModuleOp moduleOp,
                                              mlir::Location loc,
                                              mlir::ArrayAttr dimensions,
                                              llvm::StringRef prefix) const {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto arrayType =
        mlir::LLVM::LLVMArrayType::get(builder.getI64Type(), dimensions.size());

    std::string symbolName = prefix.str() + "_dimensions";
    llvm::SmallVector<int64_t> values;

    for (const auto &dimension : dimensions.getAsRange<mlir::IntegerAttr>()) {
      values.push_back(dimension.getInt());
    }

    auto tensorType = mlir::RankedTensorType::get(arrayType.getNumElements(),
                                                  builder.getI64Type());

    mlir::SymbolTable &symbolTable =
        symbolTableCollection->getSymbolTable(moduleOp);

    auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
        loc, arrayType, true, mlir::LLVM::Linkage::Private, symbolName,
        mlir::DenseIntElementsAttr::get(tensorType, values));

    symbolTable.insert(globalOp);
    return globalOp;
  }

  mlir::Value declareAndGetDimensionsArray(mlir::OpBuilder &builder,
                                           mlir::ModuleOp moduleOp,
                                           mlir::Location loc,
                                           mlir::ArrayAttr dimensions,
                                           llvm::StringRef prefix) const {
    auto globalOp =
        declareDimensionsArray(builder, moduleOp, loc, dimensions, prefix);

    mlir::Value address =
        builder.create<mlir::LLVM::AddressOfOp>(loc, globalOp);

    return address;
  }

  mlir::LLVM::GlobalOp declareRangesArray(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::ModuleOp moduleOp,
                                          const MultidimensionalRange &indices,
                                          llvm::StringRef prefix) const {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto arrayType = mlir::LLVM::LLVMArrayType::get(builder.getI64Type(),
                                                    indices.rank() * 2);

    llvm::SmallVector<int64_t, 10> values;

    for (size_t dim = 0, rank = indices.rank(); dim < rank; ++dim) {
      values.push_back(indices[dim].getBegin());
      values.push_back(indices[dim].getEnd());
    }

    std::string symbolName = prefix.str() + "_range";

    auto tensorType = mlir::RankedTensorType::get(arrayType.getNumElements(),
                                                  builder.getI64Type());

    auto globalOp = builder.create<mlir::LLVM::GlobalOp>(
        loc, arrayType, true, mlir::LLVM::Linkage::Private, symbolName,
        mlir::DenseIntElementsAttr::get(tensorType, values));

    symbolTableCollection->getSymbolTable(moduleOp).insert(globalOp);
    return globalOp;
  }

  mlir::Value declareAndGetRangesArray(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::ModuleOp moduleOp,
                                       const MultidimensionalRange &indices,
                                       llvm::StringRef instanceName) const {
    auto globalOp =
        declareRangesArray(builder, loc, moduleOp, indices, instanceName);

    mlir::Value address =
        builder.create<mlir::LLVM::AddressOfOp>(loc, globalOp);

    return address;
  }

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::LLVM::LLVMFunctionType functionType) const {
    auto funcOp = symbolTableCollection->lookupSymbolIn<mlir::LLVM::LLVMFuncOp>(
        moduleOp, builder.getStringAttr(name));

    if (funcOp) {
      return funcOp;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto newFuncOp =
        builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, functionType);

    symbolTableCollection->getSymbolTable(moduleOp).insert(newFuncOp);
    return newFuncOp;
  }

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::Type> argTypes) const {
    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getOrDeclareFunction(builder, moduleOp, loc, name, functionType);
  }

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::Value> args) const {
    llvm::SmallVector<mlir::Type> argTypes;

    for (mlir::Value arg : args) {
      argTypes.push_back(arg.getType());
    }

    return getOrDeclareFunction(builder, moduleOp, loc, name, resultType,
                                argTypes);
  }

  mlir::Value getFunctionAddress(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::LLVM::LLVMFunctionType functionType,
                                 llvm::StringRef name) const {
    auto functionPtrType =
        mlir::LLVM::LLVMPointerType::get(builder.getContext());

    mlir::Value address =
        builder.create<mlir::LLVM::AddressOfOp>(loc, functionPtrType, name);

    return address;
  }

  mlir::Value getGetterFunctionAddress(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       llvm::StringRef name) const {
    mlir::Type resultType = builder.getF64Type();

    llvm::SmallVector<mlir::Type, 1> argTypes;

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getFunctionAddress(builder, loc, functionType, name);
  }

  mlir::Value getSetterFunctionAddress(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       llvm::StringRef name) const {
    mlir::Type resultType = mlir::LLVM::LLVMVoidType::get(builder.getContext());

    llvm::SmallVector<mlir::Type, 2> argTypes;
    argTypes.push_back(builder.getF64Type());

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getFunctionAddress(builder, loc, functionType, name);
  }

  mlir::Value getResidualFunctionAddress(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         llvm::StringRef name) const {
    mlir::Type resultType = builder.getF64Type();

    llvm::SmallVector<mlir::Type, 2> argTypes;
    argTypes.push_back(builder.getF64Type());

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getFunctionAddress(builder, loc, functionType, name);
  }

  mlir::Value getJacobianFunctionAddress(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         llvm::StringRef name) const {
    mlir::Type resultType = builder.getF64Type();

    llvm::SmallVector<mlir::Type, 4> argTypes;
    argTypes.push_back(builder.getF64Type());

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    argTypes.push_back(builder.getF64Type());

    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getFunctionAddress(builder, loc, functionType, name);
  }

  mlir::Value getAccessFunctionAddress(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       llvm::StringRef name) const {
    mlir::Type resultType = mlir::LLVM::LLVMVoidType::get(builder.getContext());

    llvm::SmallVector<mlir::Type, 4> argTypes;

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getContext()));

    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getFunctionAddress(builder, loc, functionType, name);
  }

  mlir::Value createGlobalString(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::ModuleOp moduleOp, mlir::StringRef name,
                                 mlir::StringRef value) const {
    mlir::LLVM::GlobalOp global;

    {
      // Create the global at the entry of the module.
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(moduleOp.getBody());

      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(
              llvm::StringRef(value.data(), value.size() + 1)));

      symbolTableCollection->getSymbolTable(moduleOp).insert(global);
    }

    // Get the pointer to the first character of the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);

    mlir::Type type = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()), type,
        globalPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
  }

protected:
  mlir::SymbolTableCollection *symbolTableCollection;
};

struct InstanceOpLowering : public IDAOpConversion<InstanceOp> {
  using IDAOpConversion<InstanceOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    mlir::SymbolTable &symbolTable =
        symbolTableCollection->getSymbolTable(moduleOp);

    // Create the global variable.
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, getVoidPtrType(), false, mlir::LLVM::Linkage::Private,
        op.getSymName(), nullptr);

    symbolTable.remove(op);
    return mlir::success();
  }
};

struct InitOpLowering : public IDAOpConversion<InitOp> {
  using IDAOpConversion<InitOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(InitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 1> args;
    llvm::SmallVector<std::string, 1> mangledArgsTypes;

    auto resultType = getVoidPtrType();
    auto mangledResultType = mangling.getVoidPointerType();

    auto functionName = mangling.getMangledFunction(
        "idaCreate", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    auto callOp = rewriter.create<mlir::LLVM::CallOp>(loc, funcOp, args);

    auto instancePtrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

    mlir::Value address = rewriter.create<mlir::LLVM::AddressOfOp>(
        loc, instancePtrType, op.getInstance());

    rewriter.create<mlir::LLVM::StoreOp>(loc, callOp.getResult(), address);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct SetStartTimeOpLowering : public IDAOpConversion<SetStartTimeOp> {
  using IDAOpConversion<SetStartTimeOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SetStartTimeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 2> args;
    llvm::SmallVector<std::string, 2> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Start time.
    mlir::Value startTime = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getF64FloatAttr(op.getTime().convertToDouble()));

    args.push_back(startTime);

    mangledArgsTypes.push_back(mangling.getFloatingPointType(
        startTime.getType().getIntOrFloatBitWidth()));

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaSetStartTime", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct SetEndTimeOpLowering : public IDAOpConversion<SetEndTimeOp> {
  using IDAOpConversion<SetEndTimeOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SetEndTimeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 2> args;
    llvm::SmallVector<std::string, 2> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // End time.
    mlir::Value endTime = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getF64FloatAttr(op.getTime().convertToDouble()));

    args.push_back(endTime);

    mangledArgsTypes.push_back(mangling.getFloatingPointType(
        endTime.getType().getIntOrFloatBitWidth()));

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaSetEndTime", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct GetCurrentTimeOpLowering : public IDAOpConversion<GetCurrentTimeOp> {
  using IDAOpConversion<GetCurrentTimeOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(GetCurrentTimeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 1> args;
    llvm::SmallVector<std::string, 1> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the call to the runtime library.
    auto resultType = rewriter.getF64Type();

    auto mangledResultType =
        mangling.getFloatingPointType(resultType.getIntOrFloatBitWidth());

    auto functionName = mangling.getMangledFunction(
        "idaGetCurrentTime", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct AddEquationOpLowering : public IDAOpConversion<AddEquationOp> {
  using IDAOpConversion<AddEquationOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(AddEquationOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 4> args;
    llvm::SmallVector<std::string, 4> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the array with the equation ranges.
    mlir::Value equationRangesPtr = declareAndGetRangesArray(
        rewriter, loc, moduleOp, adaptor.getEquationRanges().getValue(),
        adaptor.getInstance());

    args.push_back(equationRangesPtr);

    mangledArgsTypes.push_back(
        mangling.getPointerType(mangling.getIntegerType(64)));

    // Rank.
    mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(
        loc,
        rewriter.getI64IntegerAttr(op.getEquationRanges().getValue().rank()));

    args.push_back(rank);

    mangledArgsTypes.push_back(
        mangling.getIntegerType(rank.getType().getIntOrFloatBitWidth()));

    // String representation.
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    if (auto stringRepr = op.getStringRepresentation()) {
      args.push_back(
          createGlobalString(rewriter, loc, moduleOp, "eqStr", *stringRepr));
    } else {
      args.push_back(
          rewriter.create<mlir::LLVM::ZeroOp>(loc, getVoidPtrType()));
    }

    // Create the call to the runtime library.
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto mangledResultType =
        mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

    auto functionName = mangling.getMangledFunction(
        "idaAddEquation", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct AddAlgebraicVariableOpLowering
    : public IDAOpConversion<AddAlgebraicVariableOp> {
  using IDAOpConversion<AddAlgebraicVariableOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(AddAlgebraicVariableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 7> args;
    llvm::SmallVector<std::string, 7> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Rank.
    mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(op.getDimensions().size()));

    args.push_back(rank);

    mangledArgsTypes.push_back(
        mangling.getIntegerType(rank.getType().getIntOrFloatBitWidth()));

    // Create the array with the variable dimensions.
    mlir::Value dimensionsPtr = declareAndGetDimensionsArray(
        rewriter, moduleOp, loc, adaptor.getDimensions(),
        adaptor.getInstance());

    args.push_back(dimensionsPtr);

    mangledArgsTypes.push_back(
        mangling.getPointerType(mangling.getIntegerType(64)));

    // Variable getter function address.
    auto getterName = op.getGetter();
    assert(getterName.getNestedReferences().empty());
    args.push_back(getGetterFunctionAddress(
        rewriter, loc, getterName.getRootReference().getValue()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Variable setter function address.
    auto setterName = op.getSetter();
    assert(setterName.getNestedReferences().empty());
    args.push_back(getSetterFunctionAddress(
        rewriter, loc, setterName.getRootReference().getValue()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Name of the variable.
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    if (auto name = op.getName()) {
      args.push_back(
          createGlobalString(rewriter, loc, moduleOp, "varName", *name));
    } else {
      args.push_back(
          rewriter.create<mlir::LLVM::ZeroOp>(loc, getVoidPtrType()));
    }

    // Create the call to the runtime library.
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto mangledResultType =
        mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

    auto functionName = mangling.getMangledFunction(
        "idaAddAlgebraicVariable", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct AddStateVariableOpLowering : public IDAOpConversion<AddStateVariableOp> {
  using IDAOpConversion<AddStateVariableOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(AddStateVariableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 7> args;
    llvm::SmallVector<std::string, 7> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Rank.
    mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(op.getDimensions().size()));

    args.push_back(rank);

    mangledArgsTypes.push_back(
        mangling.getIntegerType(rank.getType().getIntOrFloatBitWidth()));

    // Create the array with the variable dimensions.
    mlir::Value dimensionsPtr = declareAndGetDimensionsArray(
        rewriter, moduleOp, loc, adaptor.getDimensions(),
        adaptor.getInstance());

    args.push_back(dimensionsPtr);

    mangledArgsTypes.push_back(
        mangling.getPointerType(mangling.getIntegerType(64)));

    // State variable getter function address.
    auto stateGetterName = op.getStateGetter();
    assert(stateGetterName.getNestedReferences().empty());
    args.push_back(getGetterFunctionAddress(
        rewriter, loc, stateGetterName.getRootReference().getValue()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // State variable setter function address.
    auto stateSetterName = op.getStateSetter();
    assert(stateSetterName.getNestedReferences().empty());
    args.push_back(getSetterFunctionAddress(
        rewriter, loc, stateSetterName.getRootReference().getValue()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Derivative variable getter function address.
    auto derivativeGetterName = op.getDerivativeGetter();
    assert(derivativeGetterName.getNestedReferences().empty());
    args.push_back(getGetterFunctionAddress(
        rewriter, loc, derivativeGetterName.getRootReference().getValue()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Derivative variable setter function address.
    auto derivativeSetterName = op.getDerivativeSetter();
    assert(derivativeSetterName.getNestedReferences().empty());
    args.push_back(getSetterFunctionAddress(
        rewriter, loc, derivativeSetterName.getRootReference().getValue()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Name of the variable.
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    if (auto name = op.getName()) {
      args.push_back(
          createGlobalString(rewriter, loc, moduleOp, "varName", *name));
    } else {
      args.push_back(
          rewriter.create<mlir::LLVM::ZeroOp>(loc, getVoidPtrType()));
    }

    // Create the call to the runtime library.
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto mangledResultType =
        mangling.getIntegerType(resultType.getIntOrFloatBitWidth());

    auto functionName = mangling.getMangledFunction(
        "idaAddStateVariable", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct AddVariableAccessOpLowering
    : public IDAOpConversion<AddVariableAccessOp> {
  using IDAOpConversion<AddVariableAccessOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(AddVariableAccessOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 5> args;
    llvm::SmallVector<std::string, 3> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Equation.
    args.push_back(adaptor.getEquation());

    mangledArgsTypes.push_back(mangling.getIntegerType(
        adaptor.getEquation().getType().getIntOrFloatBitWidth()));

    // Variable.
    args.push_back(adaptor.getVariable());

    mangledArgsTypes.push_back(mangling.getIntegerType(
        adaptor.getVariable().getType().getIntOrFloatBitWidth()));

    // Access function.
    args.push_back(
        getAccessFunctionAddress(rewriter, loc, op.getAccessFunction()));

    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaAddVariableAccess", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct SetResidualOpLowering : public IDAOpConversion<SetResidualOp> {
  using IDAOpConversion<SetResidualOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(SetResidualOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 3> args;
    llvm::SmallVector<std::string, 3> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Equation.
    args.push_back(adaptor.getEquation());

    mangledArgsTypes.push_back(mangling.getIntegerType(
        adaptor.getEquation().getType().getIntOrFloatBitWidth()));

    // Residual function address.
    mlir::Value functionAddress =
        getResidualFunctionAddress(rewriter, loc, op.getFunction());

    args.push_back(functionAddress);
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaSetResidual", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct AddJacobianOpLowering : public IDAOpConversion<AddJacobianOp> {
  using IDAOpConversion<AddJacobianOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(AddJacobianOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 6> args;
    llvm::SmallVector<std::string, 6> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Equation.
    args.push_back(adaptor.getEquation());

    mangledArgsTypes.push_back(mangling.getIntegerType(
        adaptor.getEquation().getType().getIntOrFloatBitWidth()));

    // Variable.
    args.push_back(adaptor.getVariable());
    mangledArgsTypes.push_back(mangling.getIntegerType(
        adaptor.getVariable().getType().getIntOrFloatBitWidth()));

    // Jacobian function address.
    mlir::Value functionAddress =
        getJacobianFunctionAddress(rewriter, loc, op.getFunction());

    args.push_back(functionAddress);
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Number of seeds.
    mlir::Value numOfSeeds = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(op.getSeedSizes().size()));

    args.push_back(numOfSeeds);
    mangledArgsTypes.push_back(mangling.getIntegerType(64));

    // Seed sizes.
    mlir::Value seedSizes = declareAndGetDimensionsArray(
        rewriter, moduleOp, loc, op.getSeedSizes(), op.getFunction());

    args.push_back(seedSizes);
    mangledArgsTypes.push_back(
        mangling.getPointerType(mangling.getIntegerType(64)));

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaAddJacobian", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct CalcICOpLowering : public IDAOpConversion<CalcICOp> {
  using IDAOpConversion<CalcICOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(CalcICOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 1> args;
    llvm::SmallVector<std::string, 1> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaCalcIC", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct StepOpLowering : public IDAOpConversion<StepOp> {
  using IDAOpConversion<StepOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(StepOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 1> args;
    llvm::SmallVector<std::string, 1> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaStep", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct FreeOpLowering : public IDAOpConversion<FreeOp> {
  using IDAOpConversion<FreeOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(FreeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 1> args;
    llvm::SmallVector<std::string, 1> mangledArgsTypes;

    // IDA instance.
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the call to the runtime library.
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaFree", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

struct PrintStatisticsOpLowering : public IDAOpConversion<PrintStatisticsOp> {
  using IDAOpConversion<PrintStatisticsOp>::IDAOpConversion;

  mlir::LogicalResult
  matchAndRewrite(PrintStatisticsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 1> args;
    llvm::SmallVector<std::string, 1> mangledArgsTypes;

    // IDA instance
    args.push_back(getInstance(rewriter, loc, adaptor.getInstance()));
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    // Create the call to the runtime library
    auto resultType = getVoidType();
    auto mangledResultType = mangling.getVoidType();

    auto functionName = mangling.getMangledFunction(
        "idaPrintStatistics", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};
} // namespace

namespace marco::codegen {
class IDAToLLVMConversionPass
    : public mlir::impl::IDAToLLVMConversionPassBase<IDAToLLVMConversionPass> {
public:
  using IDAToLLVMConversionPassBase<
      IDAToLLVMConversionPass>::IDAToLLVMConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOperations();
};
} // namespace marco::codegen

void IDAToLLVMConversionPass::runOnOperation() {
  if (mlir::failed(convertOperations())) {
    mlir::emitError(getOperation().getLoc(),
                    "Error in converting the IDA operations");

    return signalPassFailure();
  }
}

mlir::LogicalResult IDAToLLVMConversionPass::convertOperations() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<mlir::ida::IDADialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();

  mlir::DataLayout dataLayout(moduleOp);
  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext(), dataLayout);

  LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);
  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  patterns
      .insert<InstanceOpLowering, SetStartTimeOpLowering, SetEndTimeOpLowering,
              GetCurrentTimeOpLowering, AddEquationOpLowering,
              AddAlgebraicVariableOpLowering, AddStateVariableOpLowering,
              AddVariableAccessOpLowering, SetResidualOpLowering,
              AddJacobianOpLowering, InitOpLowering, CalcICOpLowering,
              StepOpLowering, FreeOpLowering, PrintStatisticsOpLowering>(
          typeConverter, symbolTableCollection);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir {
std::unique_ptr<mlir::Pass> createIDAToLLVMConversionPass() {
  return std::make_unique<IDAToLLVMConversionPass>();
}
} // namespace mlir
