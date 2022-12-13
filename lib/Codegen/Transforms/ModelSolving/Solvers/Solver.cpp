#include "marco/Codegen/Transforms/ModelSolving/Solvers/ModelSolver.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

static IndexSet getFilteredIndices(
    mlir::Type variableType,
    llvm::ArrayRef<VariableFilter::Filter> filters)
{
  IndexSet result;

  auto arrayType = variableType.cast<ArrayType>();
  assert(arrayType.hasStaticShape());

  for (const auto& filter : filters) {
    if (!filter.isVisible()) {
      continue;
    }

    auto filterRanges = filter.getRanges();
    assert(filterRanges.size() == static_cast<size_t>(arrayType.getRank()));

    std::vector<Range> ranges;

    for (const auto& range : llvm::enumerate(filterRanges)) {
      // In Modelica, arrays are 1-based. If present, we need to lower by 1
      // the value given by the variable filter.

      auto lowerBound = range.value().hasLowerBound() ? range.value().getLowerBound() - 1 : 0;
      auto upperBound = range.value().hasUpperBound() ? range.value().getUpperBound() : arrayType.getShape()[range.index()];
      ranges.emplace_back(lowerBound, upperBound);
    }

    if (ranges.empty()) {
      // Scalar value.
      ranges.emplace_back(0, 1);
    }

    result += MultidimensionalRange(std::move(ranges));
  }

  return result;
}

namespace marco::codegen
{
  ModelSolver::ModelSolver(
      mlir::LLVMTypeConverter& typeConverter,
      VariableFilter& variablesFilter)
      : typeConverter(&typeConverter),
        variablesFilter(&variablesFilter)
  {
  }

  ModelSolver::~ModelSolver() = default;

  mlir::Type ModelSolver::getVoidType() const
  {
    return mlir::LLVM::LLVMVoidType::get(&typeConverter->getContext());
  }

  mlir::Type ModelSolver::getVoidPtrType() const
  {
    mlir::MLIRContext* context = &typeConverter->getContext();
    mlir::Type i8Type = mlir::IntegerType::get(context, 8);
    return mlir::LLVM::LLVMPointerType::get(i8Type);
  }

  mlir::LLVM::LLVMFuncOp ModelSolver::getOrDeclareExternalFunction(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      llvm::StringRef name,
      mlir::LLVM::LLVMFunctionType type) const
  {
    if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
      return foo;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, type);
  }

  mlir::Value ModelSolver::alloc(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Location loc,
      mlir::Type type) const
  {
    // Add the function to the module.
    mlir::LLVM::LLVMFuncOp allocFn = lookupOrCreateHeapAllocFn(
        module, typeConverter->getIndexType());

    // Determine the size (in bytes) of the memory to be allocated.
    mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(type);
    mlir::Value nullPtr = builder.create<mlir::LLVM::NullOp>(loc, ptrType);

    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
        loc, typeConverter->getIndexType(), builder.getIndexAttr(1));

    mlir::Value gepPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, ptrType, nullPtr, one);

    mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(
        loc, typeConverter->getIndexType(), gepPtr);

    auto callOp = builder.create<mlir::LLVM::CallOp>(loc, allocFn, sizeBytes);
    mlir::Value resultOpaquePtr = callOp.getResult();

    // Cast the allocated memory pointer to a pointer of the original type.
    return builder.create<mlir::LLVM::BitcastOp>(
        loc, ptrType, resultOpaquePtr);
  }

  void ModelSolver::dealloc(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Location loc,
      mlir::Value ptr) const
  {
    assert(ptr.getType().isa<mlir::LLVM::LLVMPointerType>());

    // Add the function to the module.
    mlir::LLVM::LLVMFuncOp deallocFn = lookupOrCreateHeapFreeFn(module);

    // Call the function.
    builder.create<mlir::LLVM::CallOp>(loc, deallocFn, ptr);
  }

  mlir::LLVM::LLVMStructType ModelSolver::getRuntimeDataStructType(
      mlir::MLIRContext* context,
      mlir::modelica::ModelOp modelOp) const
  {
    auto variablesTypes = modelOp.getBodyRegion().getArgumentTypes();
    llvm::SmallVector<mlir::Type> types;

    // Solver-reserved data.
    types.push_back(getVoidPtrType());

    // Time.
    mlir::Type timeType = RealType::get(context);
    mlir::Type convertedTimeType = typeConverter->convertType(timeType);
    types.push_back(convertedTimeType);

    // Variables.
    for (const auto& varType : variablesTypes) {
      mlir::Type convertedVarType = typeConverter->convertType(varType);
      types.push_back(convertedVarType);
    }

    return mlir::LLVM::LLVMStructType::getLiteral(context, types);
  }

  mlir::Value ModelSolver::loadDataFromOpaquePtr(
      mlir::OpBuilder& builder,
      mlir::Value ptr,
      ModelOp modelOp) const
  {
    mlir::Location loc = ptr.getLoc();

    mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(
        getRuntimeDataStructType(builder.getContext(), modelOp));

    mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(
        loc, structPtrType, ptr);

    return builder.create<mlir::LLVM::LoadOp>(loc, structPtr);
  }

  void ModelSolver::setRuntimeData(
      mlir::OpBuilder& builder,
      mlir::Value opaquePtr,
      mlir::modelica::ModelOp modelOp,
      mlir::Value runtimeData) const
  {
    mlir::Location loc = runtimeData.getLoc();

    auto runtimeDataStructPtrType = mlir::LLVM::LLVMPointerType::get(
        getRuntimeDataStructType(builder.getContext(), modelOp));

    assert(runtimeData.getType() == runtimeDataStructPtrType.getElementType());

    mlir::Value runtimeDataStructPtr = builder.create<mlir::LLVM::BitcastOp>(
        loc, runtimeDataStructPtrType, opaquePtr);

    builder.create<mlir::LLVM::StoreOp>(
        loc, runtimeData, runtimeDataStructPtr);
  }

  mlir::Value ModelSolver::extractValue(
      mlir::OpBuilder& builder,
      mlir::Value structValue,
      mlir::Type type,
      unsigned int position) const
  {
    mlir::Location loc = structValue.getLoc();

    assert(structValue.getType().isa<mlir::LLVM::LLVMStructType>() && "Not an LLVM struct");
    auto structType = structValue.getType().cast<mlir::LLVM::LLVMStructType>();
    auto structTypes = structType.getBody();
    assert (position < structTypes.size() && "LLVM struct: index is out of bounds");

    mlir::Value var = builder.create<mlir::LLVM::ExtractValueOp>(loc, structTypes[position], structValue, position);
    return typeConverter->materializeSourceConversion(builder, loc, type, var);
  }

  mlir::Value ModelSolver::extractSolverDataPtr(
      mlir::OpBuilder& builder,
      mlir::Value structValue,
      mlir::Type solverDataType) const
  {
    mlir::Value opaquePtr = extractValue(
              builder, structValue, getVoidPtrType(), solversDataPosition);

    return builder.create<mlir::LLVM::BitcastOp>(
        opaquePtr.getLoc(),
        mlir::LLVM::LLVMPointerType::get(solverDataType),
        opaquePtr);
  }

  mlir::Value ModelSolver::extractVariable(
      mlir::OpBuilder& builder,
      mlir::Value structValue,
      mlir::Type type,
      unsigned int varIndex) const
  {
    return extractValue(builder, structValue, type, varIndex + variablesOffset);
  }

  mlir::Value ModelSolver::getOrCreateGlobalString(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ModuleOp module,
      mlir::StringRef name,
      mlir::StringRef value) const
  {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;

    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());

      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(llvm::StringRef(
              value.data(), value.size() + 1)));
    }

    // Get the pointer to the first character of the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);

    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc,
        mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc,
        getVoidPtrType(),
        globalPtr, llvm::makeArrayRef({cst0, cst0}));
  }

  mlir::LogicalResult ModelSolver::createGetModelNameFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getModelNameFunctionName,
        builder.getFunctionType(llvm::None, getVoidPtrType()));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create a global string containing the name of the model.
    mlir::Value name = getOrCreateGlobalString(
        builder, loc, module, "modelName", modelOp.getSymName());

    builder.create<mlir::func::ReturnOp>(loc, name);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGetNumOfVariablesFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getNumOfVariablesFunctionName,
        builder.getFunctionType(llvm::None, builder.getI64Type()));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create the result.
    unsigned int numOfVariables = modelOp.getBodyRegion().getNumArguments();

    mlir::Value result = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(numOfVariables));

    builder.create<mlir::func::ReturnOp>(loc, result);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGetVariableNameFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    // char* type.
    mlir::Type charPtrType = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(builder.getContext(), 8));

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getVariableNameFunctionName,
        builder.getFunctionType(builder.getI64Type(), charPtrType));

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(
        &function.getBody(), function.getBody().end(), charPtrType, loc);

    // Create the blocks and the switch.
    llvm::SmallVector<llvm::StringRef> names = modelOp.variableNames();

    size_t numCases = names.size();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);

    mlir::Value defaultOperand = getOrCreateGlobalString(
        builder, loc, module, "varUnknown", "unknown");

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks.
    for (const auto& name : llvm::enumerate(names)) {
      size_t i = name.index();
      builder.setInsertionPointToStart(caseBlocks[i]);

      std::string symbolName = "var" + std::to_string(name.index());

      mlir::Value result = getOrCreateGlobalString(
          builder, loc, module, symbolName, name.value());

      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGetVariableRankFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getVariableRankFunctionName,
        builder.getFunctionType(builder.getI64Type(), builder.getI64Type()));

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(
        &function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch.
    mlir::TypeRange types = modelOp.getBodyRegion().getArgumentTypes();

    size_t numCases = types.size();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);

    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    builder.setInsertionPointToStart(entryBlock);

    // Populate the case blocks.
    for (const auto& type : llvm::enumerate(types)) {
      size_t i = type.index();
      builder.setInsertionPointToStart(caseBlocks[i]);

      int64_t rank = type.value().cast<ArrayType>().getRank();

      mlir::Value result = builder.create<mlir::LLVM::ConstantOp>(
          loc, builder.getI64IntegerAttr(rank));

      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGetVariableNumOfPrintableRangesFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const DerivativesMap& derivativesMap) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getVariableNumOfPrintableRangesFunctionName,
        builder.getFunctionType(builder.getI64Type(), builder.getI64Type()));

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch.
    llvm::SmallVector<llvm::StringRef> names = modelOp.variableNames();

    size_t numCases = names.size();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks.
    auto filteredIndicesFn = [&](size_t varIndex, llvm::StringRef name) -> IndexSet {
      auto arrayType = modelOp.getBodyRegion().getArgument(varIndex).getType().cast<ArrayType>();
      int64_t rank = arrayType.getRank();

      if (derivativesMap.isDerivative(varIndex)) {
        auto derivedVariable = derivativesMap.getDerivedVariable(varIndex);
        llvm::StringRef derivedVariableName = names[derivedVariable];
        auto filters = variablesFilter->getVariableDerInfo(derivedVariableName, rank);

        IndexSet filteredIndices = getFilteredIndices(arrayType, filters);
        IndexSet derivedIndices = derivativesMap.getDerivedIndices(derivedVariable);
        return filteredIndices.intersect(derivedIndices);
      }

      auto filters = variablesFilter->getVariableInfo(name, rank);
      return getFilteredIndices(arrayType, filters);
    };

    for (const auto& name : llvm::enumerate(names)) {
      size_t i = name.index();
      builder.setInsertionPointToStart(caseBlocks[i]);

      IndexSet indices = filteredIndicesFn(i, name.value());
      auto numOfRanges = std::distance(indices.rangesBegin(), indices.rangesEnd());
      mlir::Value result = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(numOfRanges));
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGetVariablePrintableRangeBeginFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const DerivativesMap& derivativesMap) const
  {
    auto callback = [](const Range& range) -> int64_t {
      return range.getBegin();
    };

    return createGetVariablePrintableRangeBoundariesFunction(
        builder, modelOp, derivativesMap,
        getVariablePrintableRangeBeginFunctionName,
        callback);
  }

  mlir::LogicalResult ModelSolver::createGetVariablePrintableRangeEndFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const DerivativesMap& derivativesMap) const
  {
    auto callback = [](const Range& range) -> int64_t {
      return range.getEnd();
    };

    return createGetVariablePrintableRangeBoundariesFunction(
        builder, modelOp, derivativesMap,
        getVariablePrintableRangeEndFunctionName,
        callback);
  }

  mlir::LogicalResult ModelSolver::createGetVariablePrintableRangeBoundariesFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const DerivativesMap& derivativesMap,
      llvm::StringRef functionName,
      std::function<int64_t(const Range&)> boundaryGetterCallback) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 3> argTypes;
    argTypes.push_back(builder.getI64Type());
    argTypes.push_back(builder.getI64Type());
    argTypes.push_back(builder.getI64Type());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, functionName,
        builder.getFunctionType(argTypes, builder.getI64Type()));

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(
        &function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch
    llvm::SmallVector<llvm::StringRef> names = modelOp.variableNames();

    size_t numCases = names.size();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);

    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks.
    auto filteredIndicesFn = [&](size_t varIndex, llvm::StringRef name) -> IndexSet {
      auto arrayType = modelOp.getBodyRegion().getArgument(varIndex).getType().cast<ArrayType>();
      int64_t rank = arrayType.getRank();

      if (derivativesMap.isDerivative(varIndex)) {
        auto derivedVariable = derivativesMap.getDerivedVariable(varIndex);
        llvm::StringRef derivedVariableName = names[derivedVariable];
        auto filters = variablesFilter->getVariableDerInfo(derivedVariableName, rank);

        IndexSet filteredIndices = getFilteredIndices(arrayType, filters);
        IndexSet derivedIndices = derivativesMap.getDerivedIndices(derivedVariable);
        return filteredIndices.intersect(derivedIndices);
      }

      auto filters = variablesFilter->getVariableInfo(name, rank);
      return getFilteredIndices(arrayType, filters);
    };

    std::map<unsigned int, std::map<MultidimensionalRange, mlir::func::FuncOp>> rangeBoundaryFuncOps;
    std::string baseRangeFunctionName = functionName.str() + "_range";
    size_t rangesCounter = 0;

    for (const auto& name : llvm::enumerate(names)) {
      size_t i = name.index();
      builder.setInsertionPointToStart(caseBlocks[i]);

      std::string calleeName = functionName.str() + "_var" + std::to_string(i);
      IndexSet indices = filteredIndicesFn(i, name.value());

      if (mlir::failed(createGetPrintableIndexSetBoundariesFunction(builder, loc, module, calleeName, indices, boundaryGetterCallback, rangeBoundaryFuncOps, baseRangeFunctionName, rangesCounter))) {
        return mlir::failure();
      }

      std::vector<mlir::Value> args;
      args.push_back(function.getArgument(1));
      args.push_back(function.getArgument(2));
      mlir::Value result = builder.create<mlir::func::CallOp>(loc, calleeName, builder.getI64Type(), args).getResult(0);
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGetPrintableIndexSetBoundariesFunction(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ModuleOp module,
      llvm::StringRef functionName,
      const IndexSet& indexSet,
      std::function<int64_t(const modeling::Range&)> boundaryGetterCallback,
      std::map<unsigned int, std::map<MultidimensionalRange, mlir::func::FuncOp>>& rangeBoundaryFuncOps,
      llvm::StringRef baseRangeFunctionName,
      size_t& rangeFunctionsCounter) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the function inside the parent module.
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type resultType = builder.getI64Type();

    llvm::SmallVector<mlir::Type, 2> argTypes;
    argTypes.push_back(builder.getI64Type());
    argTypes.push_back(builder.getI64Type());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, functionName, builder.getFunctionType(argTypes, resultType));

    // Collect the multidimensional ranges and sort them.
    llvm::SmallVector<MultidimensionalRange> ranges;

    for (const auto& range : llvm::make_range(indexSet.rangesBegin(), indexSet.rangesEnd())) {
      ranges.push_back(range);
    }

    llvm::sort(ranges);

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(
        &function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch
    size_t numCases = ranges.size();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks.
    for (const auto& range : llvm::enumerate(ranges)) {
      size_t i = range.index();
      builder.setInsertionPointToStart(caseBlocks[i]);

      mlir::func::FuncOp callee = rangeBoundaryFuncOps[range.value().rank()][range.value()];

      if (!callee) {
        std::string calleeName = baseRangeFunctionName.str() + "_" + std::to_string(rangeFunctionsCounter++);
        callee = createGetPrintableMultidimensionalRangeBoundariesFunction(builder, loc, module, calleeName, range.value(), boundaryGetterCallback);
        rangeBoundaryFuncOps[range.value().rank()][range.value()] = callee;
      }

      mlir::Value result = builder.create<mlir::func::CallOp>(loc, callee, function.getArgument(1)).getResult(0);
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::func::FuncOp ModelSolver::createGetPrintableMultidimensionalRangeBoundariesFunction(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ModuleOp module,
      llvm::StringRef functionName,
      const MultidimensionalRange& ranges,
      std::function<int64_t(const Range&)> boundaryGetterCallback) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the function inside the parent module.
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type resultType = builder.getI64Type();

    llvm::SmallVector<mlir::Type, 1> argTypes;
    argTypes.push_back(builder.getI64Type());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, functionName,
        builder.getFunctionType(argTypes, resultType));

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(
            &function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch.
    size_t numCases = ranges.rank();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks.
    for (unsigned int i = 0, e = ranges.rank(); i < e; ++i) {
      builder.setInsertionPointToStart(caseBlocks[i]);
      int64_t boundary = boundaryGetterCallback(ranges[i]);
      mlir::Value result = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(boundary));
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return function;
  }

  mlir::LogicalResult ModelSolver::createGetVariableValueFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type int64PtrType = mlir::LLVM::LLVMPointerType::get(builder.getI64Type());

    llvm::SmallVector<mlir::Type, 3> argTypes;
    argTypes.push_back(getVoidPtrType());
    argTypes.push_back(builder.getI64Type());
    argTypes.push_back(int64PtrType);

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getVariableValueFunctionName,
        builder.getFunctionType(argTypes, builder.getF64Type()));

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(
        &function.getBody(), function.getBody().end(), builder.getF64Type(), loc);

    // Create the blocks.
    size_t numCases = modelOp.getBodyRegion().getNumArguments();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);

    // Load the runtime data structure.
    mlir::Value structValue = loadDataFromOpaquePtr(
        builder, function.getArgument(0), modelOp);

    // Create the switch.
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getF64FloatAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(1), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks.
    llvm::SmallVector<mlir::Value, 2> args(2);
    args[1] = function.getArgument(2);

    llvm::DenseMap<ArrayType, mlir::func::FuncOp> callees;
    llvm::SmallString<20> calleeName;

    for (size_t i = 0; i < numCases; ++i) {
      builder.setInsertionPointToStart(caseBlocks[i]);

      auto arrayType = modelOp.getBodyRegion().getArgument(i).getType().cast<ArrayType>();
      mlir::func::FuncOp callee = callees[arrayType];

      if (!callee) {
        calleeName = getVariableValueFunctionName;

        if (!arrayType.isScalar()) {
          calleeName += '_';

          for (const auto& dimension : llvm::enumerate(arrayType.getShape())) {
            if (dimension.index() != 0) {
              calleeName += 'x';
            }

            calleeName += std::to_string(dimension.value());
          }
        }

        calleeName += '_';
        mlir::Type elementType = arrayType.getElementType();

        if (elementType.isa<BooleanType>()) {
          calleeName += "boolean";
        } else if (elementType.isa<IntegerType>()) {
          calleeName += "integer";
        } else if (elementType.isa<RealType>()) {
          calleeName += "real";
        } else if (elementType.isa<mlir::IndexType>()) {
          calleeName += "index";
        } else {
          return mlir::failure();
        }

        callee = createScalarVariableGetter(builder, loc, module, calleeName, arrayType);
        callees[arrayType] = callee;
      }

      args[0] = extractVariable(builder, structValue, arrayType, i);
      args[0] = typeConverter->materializeTargetConversion(builder, loc, typeConverter->convertType(args[0].getType()), args[0]);

      auto callOp = builder.create<mlir::func::CallOp>(loc, callee, args);
      mlir::Value result = callOp.getResult(0);
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::func::FuncOp ModelSolver::createScalarVariableGetter(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ModuleOp module,
      llvm::StringRef functionName,
      mlir::modelica::ArrayType arrayType) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the function inside the parent module.
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type int64PtrType = mlir::LLVM::LLVMPointerType::get(builder.getI64Type());
    mlir::Type convertedArrayType = typeConverter->convertType(arrayType);

    auto functionType = builder.getFunctionType({ convertedArrayType, int64PtrType }, builder.getF64Type());
    auto function = builder.create<mlir::func::FuncOp>(loc, functionName, functionType);

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the indices.
    llvm::SmallVector<mlir::Value, 3> indices;

    for (int64_t i = 0, e = arrayType.getRank(); i < e; ++i) {
      mlir::Value offset = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(i));
      mlir::Value address = builder.create<mlir::LLVM::GEPOp>(loc, int64PtrType, int64PtrType, function.getArgument(1), offset);
      mlir::Value index = builder.create<mlir::LLVM::LoadOp>(loc, address);
      index = builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), index);
      indices.push_back(index);
    }

    mlir::Value array = typeConverter->materializeSourceConversion(builder, loc, arrayType, function.getArgument(0));
    mlir::Value result = builder.create<LoadOp>(loc, array, indices);

    if (!result.getType().isa<RealType>()) {
      result = builder.create<CastOp>(loc, RealType::get(builder.getContext()), result);
    }

    result = typeConverter->materializeTargetConversion(builder, loc, typeConverter->convertType(result.getType()), result);

    if (result.getType().getIntOrFloatBitWidth() < 64) {
      result = builder.create<mlir::LLVM::FPExtOp>(loc, builder.getF64Type(), result);
    } else if (result.getType().getIntOrFloatBitWidth() > 64) {
      result = builder.create<mlir::LLVM::FPTruncOp>(loc, builder.getF64Type(), result);
    }

    builder.create<mlir::LLVM::ReturnOp>(loc, result);
    return function;
  }

  mlir::LogicalResult ModelSolver::createGetDerivativeFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const DerivativesMap& derivativesMap) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getDerivativeFunctionName,
        builder.getFunctionType(builder.getI64Type(), builder.getI64Type()));

    // Create the entry block.
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned.
    mlir::Block* returnBlock = builder.createBlock(
        &function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch
    size_t numCases = modelOp.getBodyRegion().getNumArguments();
    llvm::SmallVector<int64_t> caseValues(numCases);
    llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
    llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

    for (size_t i = 0; i < numCases; ++i) {
      caseValues[i] = i;
      caseBlocks[i] = builder.createBlock(returnBlock);
      caseOperandsRefs[i] = llvm::None;
    }

    builder.setInsertionPointToStart(entryBlock);

    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64IntegerAttr(-1));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    builder.setInsertionPointToStart(entryBlock);

    // Populate the case blocks.
    for (size_t i = 0; i < numCases; ++i) {
      builder.setInsertionPointToStart(caseBlocks[i]);
      int64_t derivative = -1;

      if (derivativesMap.hasDerivative(i)) {
        derivative = derivativesMap.getDerivative(i);
      }

      mlir::Value result = builder.create<mlir::LLVM::ConstantOp>(
          loc, builder.getI64IntegerAttr(derivative));

      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block.
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createGetTimeFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, getTimeFunctionName,
        builder.getFunctionType(getVoidPtrType(), builder.getF64Type()));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Load the runtime data structure.
    mlir::Value structValue = loadDataFromOpaquePtr(
        builder, function.getArgument(0), modelOp);

    // Extract the time variable.
    mlir::Value time = extractValue(
        builder, structValue,
        RealType::get(builder.getContext()),
        timeVariablePosition);

    time = typeConverter->materializeTargetConversion(
        builder, loc, typeConverter->convertType(time.getType()), time);

    unsigned int timeBitWidth = time.getType().getIntOrFloatBitWidth();

    if (timeBitWidth < 64) {
      time = builder.create<mlir::LLVM::FPExtOp>(
          loc, builder.getF64Type(), time);
    } else if (timeBitWidth > 64) {
      time = builder.create<mlir::LLVM::FPTruncOp>(
          loc, builder.getF64Type(), time);
    }

    builder.create<mlir::func::ReturnOp>(loc, time);
    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createSetTimeFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 2> argTypes;
    argTypes.push_back(getVoidPtrType());
    argTypes.push_back(builder.getF64Type());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, setTimeFunctionName,
        builder.getFunctionType(argTypes, llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Load the runtime data structure.
    mlir::Value runtimeData = loadDataFromOpaquePtr(
        builder, function.getArgument(0), modelOp);

    // Set the time inside the data structure.
    mlir::Value time = function.getArgument(1);

    auto runtimeDataStructType =
        getRuntimeDataStructType(builder.getContext(), modelOp);

    mlir::Type timeType =
        runtimeDataStructType.getBody()[timeVariablePosition];

    unsigned int timeBitWidth = timeType.getIntOrFloatBitWidth();

    if (timeBitWidth > 64) {
      time = builder.create<mlir::LLVM::FPExtOp>(loc, timeType, time);
    } else if (timeBitWidth < 64) {
      time = builder.create<mlir::LLVM::FPTruncOp>(loc, timeType, time);
    }

    runtimeData = builder.create<mlir::LLVM::InsertValueOp>(
        loc, runtimeData, time, timeVariablePosition);

    setRuntimeData(builder, function.getArgument(0), modelOp, runtimeData);

    // Terminate the function.
    builder.create<mlir::func::ReturnOp>(loc);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createInitFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();

    // Create the function inside the parent module.
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, initFunctionName,
        builder.getFunctionType(llvm::None, getVoidPtrType()));

    mlir::Block* bodyBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(bodyBlock);

    std::vector<MemberCreateOp> originalMembers;

    // The descriptors of the arrays that compose that runtime data structure.
    std::vector<mlir::Value> structVariables;

    mlir::BlockAndValueMapping membersOpsMapping;

    for (auto& op : modelOp.getVarsRegion().getOps()) {
      if (auto memberCreateOp = mlir::dyn_cast<MemberCreateOp>(op)) {
        auto arrayType = memberCreateOp.getMemberType().toArrayType();
        assert(arrayType.hasStaticShape());

        std::vector<mlir::Value> dynamicDimensions;

        for (const auto& dynamicDimension : memberCreateOp.getDynamicSizes()) {
          dynamicDimensions.push_back(membersOpsMapping.lookup(dynamicDimension));
        }

        mlir::Value array = builder.create<AllocOp>(
            memberCreateOp.getLoc(), arrayType, dynamicDimensions);

        originalMembers.push_back(memberCreateOp);
        structVariables.push_back(array);

        membersOpsMapping.map(memberCreateOp.getResult(), array);
      }
    }

    // Map the body arguments to the new arrays.
    mlir::ValueRange bodyArgs = modelOp.getBodyRegion().getArguments();
    mlir::BlockAndValueMapping startOpsMapping;

    for (const auto& [arg, array] : llvm::zip(bodyArgs, structVariables)) {
      startOpsMapping.map(arg, array);
    }

    // Keep track of the variables for which a start value has been provided.
    llvm::SmallVector<bool> initializedVars(bodyArgs.size(), false);

    modelOp.getBodyRegion().walk([&](StartOp startOp) {
      unsigned int argNumber = startOp.getVariable().cast<mlir::BlockArgument>().getArgNumber();

      // Note that parameters must be set independently of the 'fixed'
      // attribute being true or false.

      if (startOp.getFixed() && !originalMembers[argNumber].isConstant()) {
        return;
      }

      builder.setInsertionPointAfterValue(structVariables[argNumber]);

      for (auto& op : startOp.getBodyRegion().getOps()) {
        if (auto yieldOp = mlir::dyn_cast<YieldOp>(op)) {
          mlir::Value valueToBeStored = startOpsMapping.lookup(yieldOp.getValues()[0]);
          mlir::Value destination = startOpsMapping.lookup(startOp.getVariable());

          if (startOp.getEach()) {
            builder.create<ArrayFillOp>(startOp.getLoc(), destination, valueToBeStored);
          } else {
            builder.create<StoreOp>(startOp.getLoc(), valueToBeStored, destination, llvm::None);
          }
        } else {
          builder.clone(op, startOpsMapping);
        }
      }

      // Set the variable as initialized.
      initializedVars[argNumber] = true;
    });

    // The variables without a 'start' attribute must be initialized to zero.
    for (const auto& initialized : llvm::enumerate(initializedVars)) {
      if (initialized.value()) {
        continue;
      }

      mlir::Value destination = structVariables[initialized.index()];
      auto arrayType = destination.getType().cast<ArrayType>();

      builder.setInsertionPointAfterValue(destination);

      mlir::Value zero = builder.create<ConstantOp>(
          destination.getLoc(), getZeroAttr(arrayType.getElementType()));

      if (arrayType.isScalar()) {
        builder.create<StoreOp>(destination.getLoc(), zero, destination, llvm::None);
      } else {
        builder.create<ArrayFillOp>(destination.getLoc(), destination, zero);
      }
    }

    // Create the runtime data structure.
    builder.setInsertionPointToEnd(&function.getBody().back());

    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp);

    mlir::Value runtimeDataStructValue = builder.create<mlir::LLVM::UndefOp>(
        loc, runtimeDataStructType);

    // Add the model variables.
    for (const auto& var : llvm::enumerate(structVariables)) {
      mlir::Type varType = var.value().getType();
      mlir::Type convertedType = typeConverter->convertType(varType);

      mlir::Value convertedVar = typeConverter->materializeTargetConversion(
          builder, loc, convertedType, var.value());

      runtimeDataStructValue = builder.create<mlir::LLVM::InsertValueOp>(
          loc, runtimeDataStructValue, convertedVar, var.index() + variablesOffset);
    }

    // Allocate the main runtime data structure.
    mlir::Value runtimeDataStructPtr = alloc(builder, module, loc, runtimeDataStructType);
    builder.create<mlir::LLVM::StoreOp>(loc, runtimeDataStructValue, runtimeDataStructPtr);

    mlir::Value runtimeDataOpaquePtr = builder.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), runtimeDataStructPtr);

    builder.create<mlir::func::ReturnOp>(loc, runtimeDataOpaquePtr);

    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createDeinitFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();

    // Create the function inside the parent module.
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, deinitFunctionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the data from the struct.
    mlir::Value runtimeDataStruct = loadDataFromOpaquePtr(
        builder, function.getArgument(0), modelOp);

    // Deallocate the arrays.
    for (const auto& varType : llvm::enumerate(modelOp.getBodyRegion().getArgumentTypes())) {
      if (auto arrayType = varType.value().dyn_cast<ArrayType>()) {

        mlir::Value var = extractValue(
            builder, runtimeDataStruct, varType.value(),
            varType.index() + variablesOffset);

        builder.create<FreeOp>(loc, var);
      }
    }

    // Deallocate the data structure.
    dealloc(builder, module, loc, function.getArgument(0));

    builder.create<mlir::LLVM::ReturnOp>(loc, llvm::None);
    return mlir::success();
  }

  mlir::LogicalResult ModelSolver::createMainFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module.
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type resultType = builder.getI32Type();

    llvm::SmallVector<mlir::Type, 3> argTypes;
    argTypes.push_back(builder.getI32Type());

    mlir::Type charType = builder.getIntegerType(8);
    mlir::Type charPtrType = mlir::LLVM::LLVMPointerType::get(charType);
    mlir::Type charPtrPtrType = mlir::LLVM::LLVMPointerType::get(charPtrType);
    argTypes.push_back(charPtrPtrType);

    auto mainFunction = builder.create<mlir::func::FuncOp>(
        loc, mainFunctionName,
        builder.getFunctionType(argTypes, resultType));

    auto* entryBlock = mainFunction.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Call the function to start the simulation.
    // Its definition lives within the runtime library.

    auto runFunction = getOrDeclareExternalFunction(
        builder, module, runFunctionName,
        mlir::LLVM::LLVMFunctionType::get(resultType, argTypes));

    auto runSimulationCall = builder.create<mlir::LLVM::CallOp>(
        loc, runFunction, mainFunction.getArguments());

    mlir::Value returnValue = runSimulationCall.getResult();

    // Create the return statement.
    builder.create<mlir::LLVM::ReturnOp>(loc, returnValue);

    return mlir::success();
  }
}
