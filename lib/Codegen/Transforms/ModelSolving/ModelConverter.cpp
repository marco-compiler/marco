#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDASolver.h"
#include "marco/Codegen/Runtime.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  /// Class to be used to uniquely identify an equation template function.
  /// Two templates are considered to be equal if they refer to the same EquationOp and have
  /// the same scheduling direction, which impacts on the function body itself due to the way
  /// the iteration indices are updated.
  class EquationTemplateInfo
  {
    public:
      EquationTemplateInfo(EquationInterface equation, modeling::scheduling::Direction schedulingDirection)
          : equation(equation.getOperation()), schedulingDirection(schedulingDirection)
      {
      }

      bool operator<(const EquationTemplateInfo& other) const {
        return equation < other.equation && schedulingDirection < other.schedulingDirection;
      }

    private:
      mlir::Operation* equation;
      modeling::scheduling::Direction schedulingDirection;
  };

  /// Get the MLIR function with the given name, or declare it inside the module if not present.
  mlir::LLVM::LLVMFuncOp getOrInsertFunction(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      llvm::StringRef name,
      mlir::LLVM::LLVMFunctionType type)
  {
    if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
      return foo;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, type);
  }

  /// Remove the unused arguments of a function and also update the function calls
  /// to reflect the function signature change.
  template<typename CallIt>
  void removeUnusedArguments(
      mlir::func::FuncOp function, CallIt callsBegin, CallIt callsEnd)
  {
    llvm::BitVector removedArgs(function.getNumArguments(), false);

    // Determine the unused arguments
    for (const auto& arg : function.getArguments()) {
      if (arg.getUsers().empty()) {
        removedArgs[arg.getArgNumber()] = true;
      }
    }

    if (!removedArgs.empty()) {
      // Erase the unused function arguments
      function.eraseArguments(removedArgs);

      // Update the function calls
      for (auto callIt = callsBegin; callIt != callsEnd; ++callIt) {
        mlir::func::CallOp call = callIt->second;

        for (size_t i = 0, e = removedArgs.size(); i < e; ++i) {
          if (removedArgs[e - i - 1]) {
            call->eraseOperand(e - i - 1);
          }
        }
      }
    }
  }

  IndexSet getFilteredIndices(mlir::Type variableType, llvm::ArrayRef<VariableFilter::Filter> filters)
  {
    IndexSet result;

    auto arrayType = variableType.cast<ArrayType>();
    assert(arrayType.hasStaticShape());

    for (const auto& filter : filters) {
      if (!filter.isVisible()) {
        continue;
      }

      auto filterRanges = filter.getRanges();
      assert(filterRanges.size() == arrayType.getRank());

      std::vector<Range> ranges;

      for (const auto& range : llvm::enumerate(filterRanges)) {
        // In Modelica, arrays are 1-based. If present, we need to lower by 1
        // the value given by the variable filter.

        auto lowerBound = range.value().hasLowerBound() ? range.value().getLowerBound() - 1 : 0;
        auto upperBound = range.value().hasUpperBound() ? range.value().getUpperBound() : arrayType.getShape()[range.index()];
        ranges.emplace_back(lowerBound, upperBound);
      }

      if (ranges.empty()) {
        // Scalar value
        ranges.emplace_back(0, 1);
      }

      result += MultidimensionalRange(std::move(ranges));
    }

    return result;
  }
}

namespace marco::codegen
{
  ModelConverter::ModelConverter(ModelSolvingOptions options, mlir::LLVMTypeConverter& typeConverter)
      : options(std::move(options)),
        typeConverter(&typeConverter)
  {
  }

  mlir::LogicalResult ModelConverter::convertInitialModel(mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model) const
  {
    // Determine the external solvers to be used
    ExternalSolvers solvers;

    DerivativesMap emptyDerivativesMap;

    auto ida = std::make_unique<IDASolver>(typeConverter, emptyDerivativesMap, options.ida, 0, 0, options.timeStep);
    ida->setEnabled(options.solver == Solver::ida);

    ConversionInfo conversionInfo;

    // Determine which equations can be potentially processed by MARCO.
    // Those are the ones that can me bade explicit with respect to the matched variable and
    // the non-cyclic ones.
    for (auto& scheduledBlock : model.getScheduledBlocks()) {
      if (!scheduledBlock->hasCycle()) {
        for (auto& scheduledEquation : *scheduledBlock) {
          auto explicitClone = scheduledEquation->cloneIRAndExplicitate(builder);

          if (explicitClone == nullptr) {
            conversionInfo.implicitEquations.emplace(scheduledEquation.get());
          } else {
            auto& movedClone = *conversionInfo.explicitEquations.emplace(std::move(explicitClone)).first;
            conversionInfo.explicitEquationsMap[scheduledEquation.get()] = movedClone.get();
          }
        }
      } else {
        for (const auto& equation : *scheduledBlock) {
          conversionInfo.cyclicEquations.emplace(equation.get());
        }
      }
    }

    if (ida->isEnabled()) {
      // Add the implicit equations to the set of equations managed by IDA, together with their
      // written variables.
      for (const auto& implicitEquation : conversionInfo.implicitEquations) {
        auto var = implicitEquation->getWrite().getVariable();
        ida->addVariable(var->getValue());
        ida->addEquation(implicitEquation);
      }

      // Add the cyclic equations to the set of equations managed by IDA, together with their
      // written variables.
      for (const auto& cyclicEquation : conversionInfo.cyclicEquations) {
        auto var = cyclicEquation->getWrite().getVariable();
        ida->addVariable(var->getValue());
        ida->addEquation(cyclicEquation);
      }

      // If any of the remaining equations manageable by MARCO does write on a variable managed
      // by IDA, then the equation must be passed to IDA even if not strictly necessary.
      // Avoiding this would require either memory duplication or a more severe restructuring
      // of the solving infrastructure, which would have to be able to split variables and equations
      // according to which runtime solver manages such variables.
      for (const auto& scheduledBlock : model.getScheduledBlocks()) {
        for (auto& scheduledEquation : *scheduledBlock) {
          auto var = scheduledEquation->getWrite().getVariable();

          if (ida->hasVariable(var->getValue())) {
            ida->addEquation(scheduledEquation.get());
          }
        }
      }
    }

    solvers.addSolver(std::move(ida));

    if (auto res = createInitICSolversFunction(builder, model, solvers); failed(res)) {
      model.getOperation().emitError("Could not create the '" + initICSolversFunctionName + "' function");
      return res;
    }

    if (auto res = createCalcICFunction(builder, model, conversionInfo, solvers); failed(res)) {
      model.getOperation().emitError("Could not create the '" + calcICFunctionName + "' function");
      return res;
    }

    if (auto res = createDeinitICSolversFunction(builder, model, solvers); failed(res)) {
      model.getOperation().emitError("Could not create the '" + deinitICSolversFunctionName + "' function");
      return res;
    }

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::convertMainModel(mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model) const
  {
    const auto& derivativesMap = model.getDerivativesMap();

    // Determine the external solvers to be used
    ExternalSolvers solvers;

    auto ida = std::make_unique<IDASolver>(typeConverter, model.getDerivativesMap(), options.ida, options.startTime, options.endTime, options.timeStep);
    ida->setEnabled(options.solver == Solver::ida);

    ConversionInfo conversionInfo;

    // Determine which equations can be potentially processed by MARCO.
    // Those are the ones that can me bade explicit with respect to the matched variable and
    // the non-cyclic ones.
    for (auto& scheduledBlock : model.getScheduledBlocks()) {
      if (!scheduledBlock->hasCycle()) {
        for (auto& scheduledEquation : *scheduledBlock) {
          auto explicitClone = scheduledEquation->cloneIRAndExplicitate(builder);

          if (explicitClone == nullptr) {
            conversionInfo.implicitEquations.emplace(scheduledEquation.get());
          } else {
            auto& movedClone = *conversionInfo.explicitEquations.emplace(std::move(explicitClone)).first;
            conversionInfo.explicitEquationsMap[scheduledEquation.get()] = movedClone.get();
          }
        }
      } else {
        for (const auto& equation : *scheduledBlock) {
          conversionInfo.cyclicEquations.emplace(equation.get());
        }
      }
    }

    if (ida->isEnabled()) {
      // Add the implicit equations to the set of equations managed by IDA, together with their
      // written variables.
      for (const auto& implicitEquation : conversionInfo.implicitEquations) {
        auto var = implicitEquation->getWrite().getVariable();
        ida->addVariable(var->getValue());
        ida->addEquation(implicitEquation);
      }

      // Add the cyclic equations to the set of equations managed by IDA, together with their
      // written variables.
      for (const auto& cyclicEquation : conversionInfo.cyclicEquations) {
        auto var = cyclicEquation->getWrite().getVariable();
        ida->addVariable(var->getValue());
        ida->addEquation(cyclicEquation);
      }

      // Add the differential equations (i.e. the ones matched with a derivative) to the set
      // of equations managed by IDA, together with their written variables.
      for (const auto& scheduledBlock : model.getScheduledBlocks()) {
        for (auto& scheduledEquation : *scheduledBlock) {
          auto var = scheduledEquation->getWrite().getVariable();
          auto argNumber = var->getValue().cast<mlir::BlockArgument>().getArgNumber();

          if (derivativesMap.isDerivative(argNumber)) {
            // State variable
            ida->addVariable(model.getVariables().getValues()[derivativesMap.getDerivedVariable(argNumber)]);

            // Derivative
            ida->addVariable(var->getValue());

            ida->addEquation(scheduledEquation.get());
          }
        }
      }

      // If any of the remaining equations manageable by MARCO does write on a variable managed
      // by IDA, then the equation must be passed to IDA even if not strictly necessary.
      // Avoiding this would require either memory duplication or a more severe restructuring
      // of the solving infrastructure, which would have to be able to split variables and equations
      // according to which runtime solver manages such variables.
      for (const auto& scheduledBlock : model.getScheduledBlocks()) {
        for (auto& scheduledEquation : *scheduledBlock) {
          auto var = scheduledEquation->getWrite().getVariable();

          if (ida->hasVariable(var->getValue())) {
            ida->addEquation(scheduledEquation.get());
          }
        }
      }
    }

    solvers.addSolver(std::move(ida));

    if (auto res = createInitMainSolversFunction(builder, model, solvers); mlir::failed(res)) {
      model.getOperation().emitError("Could not create the '" + initMainSolversFunctionName + "' function");
      return res;
    }

    if (auto res = createDeinitMainSolversFunction(builder, model, solvers); mlir::failed(res)) {
      model.getOperation().emitError("Could not create the '" + deinitMainSolversFunctionName + "' function");
      return res;
    }

    if (auto res = createUpdateNonStateVariablesFunction(builder, model, conversionInfo, solvers); mlir::failed(res)) {
      model.getOperation().emitError("Could not create the '" + updateNonStateVariablesFunctionName + "' function");
      return res;
    }

    if (auto res = createUpdateStateVariablesFunction(builder, model, solvers); mlir::failed(res)) {
      model.getOperation().emitError("Could not create the '" + updateStateVariablesFunctionName + "' function");
      return res;
    }

    if (auto res = createIncrementTimeFunction(builder, model, solvers); mlir::failed(res)) {
      model.getOperation().emitError("Could not create the '" + incrementTimeFunctionName + "' function");
      return res;
    }

    if (auto res = createPrintHeaderFunction(builder, model); mlir::failed(res)) {
      model.getOperation().emitError("Could not create the '" + printHeaderFunctionName + "' function");
      return res;
    }

    if (auto res = createPrintFunction(builder, model); mlir::failed(res)) {
      model.getOperation().emitError("Could not create the '" + printFunctionName + "' function");
      return res;
    }

    return mlir::success();
  }

  mlir::Type ModelConverter::getVoidPtrType() const
  {
    return mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(&typeConverter->getContext(), 8));
  }

  mlir::LLVM::LLVMFuncOp ModelConverter::lookupOrCreateHeapAllocFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
  {
    std::string name = "_MheapAlloc_pvoid_i64";

    if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
      return foo;
    }

    mlir::PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidPtrType(), builder.getI64Type());
    return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
  }

  mlir::LLVM::LLVMFuncOp ModelConverter::lookupOrCreateHeapFreeFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
  {
    std::string name = "_MheapFree_void_pvoid";

    if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
      return foo;
    }

    mlir::PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    mlir::Type voidType = mlir::LLVM::LLVMVoidType::get(module.getContext());
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(voidType, getVoidPtrType());
    return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
  }

  mlir::Value ModelConverter::alloc(mlir::OpBuilder& builder, mlir::ModuleOp module, mlir::Location loc, mlir::Type type) const
  {
    // Add the heap-allocating function to the module
    auto heapAllocFunc = lookupOrCreateHeapAllocFn(builder, module);

    // Determine the size (in bytes) of the memory to be allocated
    mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(type);
    mlir::Value nullPtr = builder.create<mlir::LLVM::NullOp>(loc, ptrType);

    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(loc, typeConverter->getIndexType(), builder.getIndexAttr(1));

    mlir::Value gepPtr = builder.create<mlir::LLVM::GEPOp>(loc, ptrType, nullPtr, one);
    mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(loc, typeConverter->getIndexType(), gepPtr);
    mlir::Value resultOpaquePtr = createLLVMCall(builder, loc, heapAllocFunc, sizeBytes, getVoidPtrType())[0];

    // Cast the allocated memory pointer to a pointer of the original type
    return builder.create<mlir::LLVM::BitcastOp>(loc, ptrType, resultOpaquePtr);
  }

  mlir::LogicalResult ModelConverter::createGetModelNameFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type resultType = getVoidPtrType();
    auto function = builder.create<mlir::func::FuncOp>(loc, getModelNameFunctionName, builder.getFunctionType(llvm::None, resultType));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value name = getOrCreateGlobalString(loc, builder, "modelName", modelOp.getSymName(), module);
    builder.create<mlir::func::ReturnOp>(loc, name);

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createMainFunction(mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 3> argsTypes;
    mlir::Type resultType = builder.getI32Type();

    argsTypes.push_back(builder.getI32Type());
    argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(builder.getIntegerType(8))));

    auto mainFunction = builder.create<mlir::func::FuncOp>(loc, mainFunctionName, builder.getFunctionType(argsTypes, resultType));

    auto* entryBlock = mainFunction.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Call the function to start the simulation.
    // Its definition lives within the runtime library.

    auto runFunction = getOrInsertFunction(
        builder, module, runFunctionName, mlir::LLVM::LLVMFunctionType::get(resultType, argsTypes));

    mlir::Value returnValue = builder.create<mlir::LLVM::CallOp>(loc, runFunction, mainFunction.getArguments()).getResult(0);

    // Create the return statement
    builder.create<mlir::func::ReturnOp>(loc, returnValue);

    return mlir::success();
  }

  mlir::LLVM::LLVMStructType ModelConverter::getRuntimeDataStructType(mlir::MLIRContext* context, mlir::TypeRange variables) const
  {
    std::vector<mlir::Type> types;

    // External solvers
    types.push_back(getVoidPtrType());

    // Time
    auto convertedTimeType = typeConverter->convertType(RealType::get(context));
    types.push_back(convertedTimeType);

    // Variables
    for (const auto& varType : variables) {
      auto convertedVarType = typeConverter->convertType(varType);
      types.push_back(convertedVarType);
    }

    return mlir::LLVM::LLVMStructType::getLiteral(context, types);
  }

  mlir::LLVM::LLVMStructType ModelConverter::getExternalSolversStructType(
      mlir::MLIRContext* context,
      const ExternalSolvers& externalSolvers) const
  {
    std::vector<mlir::Type> externalSolversTypes;

    for (auto& solver : externalSolvers) {
      externalSolversTypes.push_back(mlir::LLVM::LLVMPointerType::get(solver->getRuntimeDataType(context)));
    }

    return  mlir::LLVM::LLVMStructType::getLiteral(context, externalSolversTypes);
  }

  mlir::Value ModelConverter::loadDataFromOpaquePtr(
      mlir::OpBuilder& builder,
      mlir::Value ptr,
      mlir::LLVM::LLVMStructType runtimeDataType) const
  {
    auto loc = ptr.getLoc();

    mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(runtimeDataType);
    mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(loc, structPtrType, ptr);
    mlir::Value structValue = builder.create<mlir::LLVM::LoadOp>(loc, structPtr);

    return structValue;
  }

  mlir::Value ModelConverter::extractValue(
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

    mlir::Value var = builder.create<mlir::LLVM::ExtractValueOp>(loc, structTypes[position], structValue, builder.getIndexArrayAttr(position));
    return typeConverter->materializeSourceConversion(builder, loc, type, var);
  }

  mlir::Value ModelConverter::createExternalSolvers(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Location loc,
      const ExternalSolvers& externalSolvers) const
  {
    auto externalSolversStructType = getExternalSolversStructType(builder.getContext(), externalSolvers);
    mlir::Value externalSolversDataPtr = alloc(builder, module, loc, externalSolversStructType);

    // Allocate the data structures of each external solver and store each pointer into the
    // previous solvers structure.
    mlir::Value externalSolversData = builder.create<mlir::LLVM::UndefOp>(loc, externalSolversStructType);
    std::vector<mlir::Value> externalSolverDataPtrs;

    for (const auto& solver : llvm::enumerate(externalSolvers)) {
      if (!solver.value()->isEnabled()) {
        continue;
      }

      mlir::Value externalSolverDataPtr = alloc(builder, module, loc, solver.value()->getRuntimeDataType(builder.getContext()));
      externalSolverDataPtrs.push_back(externalSolverDataPtr);
      externalSolversData = builder.create<mlir::LLVM::InsertValueOp>(loc, externalSolversData, externalSolverDataPtr, builder.getIndexArrayAttr(solver.index()));
    }

    builder.create<mlir::LLVM::StoreOp>(loc, externalSolversData, externalSolversDataPtr);

    return builder.create<mlir::LLVM::BitcastOp>(externalSolversDataPtr.getLoc(), getVoidPtrType(), externalSolversDataPtr);
  }

  mlir::Value ModelConverter::convertMember(mlir::OpBuilder& builder, MemberCreateOp op) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    using LoadReplacer = std::function<void(MemberLoadOp)>;
    using StoreReplacer = std::function<void(MemberStoreOp)>;

    mlir::Location loc = op->getLoc();

    auto arrayType = op.getMemberType().toArrayType();

    // Create the memory buffer for the variable
    builder.setInsertionPoint(op);

    mlir::Value reference = builder.create<AllocOp>(loc, arrayType, op.getDynamicSizes());

    // Replace loads and stores with appropriate instructions operating on the new memory buffer.
    // The way such replacements are executed depend on the nature of the variable.

    auto replacers = [&]() {
      if (arrayType.isScalar()) {
        assert(op.getDynamicSizes().empty());

        auto loadReplacer = [&builder, reference](MemberLoadOp loadOp) -> void {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(loadOp);
          mlir::Value value = builder.create<LoadOp>(loadOp.getLoc(), reference);
          loadOp.replaceAllUsesWith(value);
          loadOp.erase();
        };

        auto storeReplacer = [&builder, reference](MemberStoreOp storeOp) -> void {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(storeOp);
          auto assignment = builder.create<AssignmentOp>(storeOp.getLoc(), reference, storeOp.getValue());
          storeOp->replaceAllUsesWith(assignment);
          storeOp.erase();
        };

        return std::make_pair<LoadReplacer, StoreReplacer>(loadReplacer, storeReplacer);
      }

      auto loadReplacer = [&builder, reference](MemberLoadOp loadOp) -> void {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(loadOp);
        loadOp.replaceAllUsesWith(reference);
        loadOp.erase();
      };

      auto storeReplacer = [&builder, reference](MemberStoreOp storeOp) -> void {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(storeOp);
        auto assignment = builder.create<AssignmentOp>(storeOp.getLoc(), reference, storeOp.getValue());
        storeOp->replaceAllUsesWith(assignment);
        storeOp.erase();
      };

      return std::make_pair<LoadReplacer, StoreReplacer>(loadReplacer, storeReplacer);
    };

    LoadReplacer loadReplacer;
    StoreReplacer storeReplacer;
    std::tie(loadReplacer, storeReplacer) = replacers();

    for (auto* user : llvm::make_early_inc_range(op->getUsers())) {
      if (auto loadOp = mlir::dyn_cast<MemberLoadOp>(user)) {
        loadReplacer(loadOp);
      } else if (auto storeOp = mlir::dyn_cast<MemberStoreOp>(user)) {
        storeReplacer(storeOp);
      }
    }

    op.replaceAllUsesWith(reference);
    op.erase();
    return reference;
  }

  mlir::LogicalResult ModelConverter::createInitSolversFunction(
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      const Model<ScheduledEquationsBlock>& model,
      const ExternalSolvers& externalSolvers) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto modelOp = model.getOperation();
    auto loc = modelOp.getLoc();
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, functionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto returnOp = builder.create<mlir::func::ReturnOp>(loc);
    builder.setInsertionPoint(returnOp);

    // Extract the runtime data structure
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value runtimeDataStruct = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    // Create the external solvers
    runtimeDataStruct = builder.create<mlir::LLVM::InsertValueOp>(
        loc, runtimeDataStruct,
        createExternalSolvers(builder, module, loc, externalSolvers),
        builder.getIndexArrayAttr(externalSolversPosition));

    // Initialize the external solvers
    mlir::Value externalSolversOpaquePtr = extractValue(builder, runtimeDataStruct, getVoidPtrType(), externalSolversPosition);

    mlir::Value externalSolversStruct = loadDataFromOpaquePtr(
        builder, externalSolversOpaquePtr,
        getExternalSolversStructType(builder.getContext(), externalSolvers));

    llvm::SmallVector<mlir::Value, 3> variables;

    for (const auto& varType : llvm::enumerate(modelOp.getBodyRegion().getArgumentTypes())) {
      variables.push_back(extractValue(builder, runtimeDataStruct, varType.value(), varType.index() + variablesOffset));
    }

    for (auto& solver : llvm::enumerate(externalSolvers)) {
      if (!solver.value()->isEnabled()) {
        continue;
      }

      mlir::Value solverDataPtr = extractValue(
          builder, externalSolversStruct,
          mlir::LLVM::LLVMPointerType::get(solver.value()->getRuntimeDataType(builder.getContext())),
          solver.index());

      if (auto res = solver.value()->processInitFunction(builder, solverDataPtr, function, variables, model); mlir::failed(res)) {
        return res;
      }
    }

    // Store the new runtime data struct
    mlir::Value runtimeDataStructPtr = builder.create<mlir::LLVM::BitcastOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(runtimeDataStructType),
        function.getArgument(0));

    builder.create<mlir::LLVM::StoreOp>(loc, runtimeDataStruct, runtimeDataStructPtr);

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createDeinitSolversFunction(
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      const Model<ScheduledEquationsBlock>& model,
      const ExternalSolvers& externalSolvers) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto modelOp = model.getOperation();
    auto loc = modelOp.getLoc();
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, functionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto returnOp = builder.create<mlir::func::ReturnOp>(loc);
    builder.setInsertionPoint(returnOp);

    // Extract the data from the struct
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value runtimeDataStruct = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    // Add "free" function to the module
    auto freeFunc = lookupOrCreateHeapFreeFn(builder, module);

    mlir::Value externalSolversPtr = extractValue(builder, runtimeDataStruct, getVoidPtrType(), externalSolversPosition);

    mlir::Value externalSolversStruct = loadDataFromOpaquePtr(
        builder, externalSolversPtr,
        getExternalSolversStructType(builder.getContext(), externalSolvers));

    // Deallocate each solver data structure
    for (auto& solver : llvm::enumerate(externalSolvers)) {
      if (!solver.value()->isEnabled()) {
        continue;
      }

      mlir::Type externalSolverRuntimeDataType = solver.value()->getRuntimeDataType(builder.getContext());

      mlir::Value solverDataPtr = extractValue(
          builder, externalSolversStruct,
          typeConverter->convertType(mlir::LLVM::LLVMPointerType::get(externalSolverRuntimeDataType)),
          solver.index());

      if (auto res = solver.value()->processDeinitFunction(builder, solverDataPtr, function); mlir::failed(res)) {
        return res;
      }

      solverDataPtr = builder.create<mlir::LLVM::BitcastOp>(solverDataPtr.getLoc(), getVoidPtrType(), solverDataPtr);
      mlir::LLVM::createLLVMCall(builder, solverDataPtr.getLoc(), freeFunc, solverDataPtr);
    }

    // Deallocate the structure containing all the solvers
    externalSolversPtr = builder.create<mlir::LLVM::BitcastOp>(externalSolversPtr.getLoc(), getVoidPtrType(), externalSolversPtr);
    mlir::LLVM::createLLVMCall(builder, externalSolversPtr.getLoc(), freeFunc, externalSolversPtr);

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createInitICSolversFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      const ExternalSolvers& externalSolvers) const
  {
    return createInitSolversFunction(builder, initICSolversFunctionName, model, externalSolvers);
  }

  mlir::LogicalResult ModelConverter::createCalcICFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      const ConversionInfo& conversionInfo,
      const ExternalSolvers& externalSolvers) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    ModelOp modelOp = model.getOperation();
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(modelOp->getParentOfType<mlir::ModuleOp>().getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, calcICFunctionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the data from the struct
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    llvm::SmallVector<mlir::Value, 3> vars;

    mlir::Value time = extractValue(builder, structValue, RealType::get(builder.getContext()), timeVariablePosition);
    vars.push_back(time);

    for (const auto& varType : llvm::enumerate(modelOp.getBodyRegion().getArgumentTypes())) {
      mlir::Value var = extractValue(builder, structValue, varType.value(), varType.index() + variablesOffset);
      vars.push_back(var);
    }

    // Convert the equations into algorithmic code
    size_t equationTemplateCounter = 0;
    size_t equationCounter = 0;

    std::map<EquationTemplateInfo, mlir::func::FuncOp> equationTemplatesMap;
    std::set<mlir::func::FuncOp> equationTemplateFunctions;
    std::multimap<mlir::func::FuncOp, mlir::func::CallOp> equationTemplateCalls;
    std::set<mlir::func::FuncOp> equationFunctions;
    std::multimap<mlir::func::FuncOp, mlir::func::CallOp> equationCalls;

    // Get or create the template equation function for a scheduled equation
    auto getEquationTemplateFn = [&](const ScheduledEquation* equation) -> mlir::func::FuncOp {
      EquationTemplateInfo requestedTemplate(equation->getOperation(), equation->getSchedulingDirection());
      auto it = equationTemplatesMap.find(requestedTemplate);

      if (it != equationTemplatesMap.end()) {
        return it->second;
      }

      std::string templateFunctionName = "initial_eq_template_" + std::to_string(equationTemplateCounter);
      ++equationTemplateCounter;

      auto explicitEquation = llvm::find_if(conversionInfo.explicitEquationsMap, [&](const auto& equationPtr) {
        return equationPtr.first == equation;
      });

      if (explicitEquation == conversionInfo.explicitEquationsMap.end()) {
        // The equation can't be made explicit and is not passed to any external solver
        return nullptr;
      }

      // Create the equation template function
      auto templateFunction = explicitEquation->second->createTemplateFunction(
          builder, templateFunctionName,
          modelOp.getBodyRegion().getArguments(),
          equation->getSchedulingDirection());

      auto timeArgumentIndex = equation->getNumOfIterationVars() * 3;

      templateFunction.insertArgument(
          timeArgumentIndex,
          RealType::get(builder.getContext()),
          builder.getDictionaryAttr(llvm::None),
          equation->getOperation().getLoc());

      templateFunction.walk([&](TimeOp timeOp) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(timeOp);
        timeOp.replaceAllUsesWith(templateFunction.getArgument(timeArgumentIndex));
        timeOp.erase();
      });

      return templateFunction;
    };

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *scheduledBlock) {
        if (externalSolvers.containsEquation(equation.get())) {
          // Let the external solver process the equation
          continue;

        } else {
          // The equation is handled by MARCO
          auto templateFunction = getEquationTemplateFn(equation.get());

          if (templateFunction == nullptr) {
            equation->getOperation().emitError("The equation can't be made explicit");
            equation->getOperation().dump();
            return mlir::failure();
          }

          equationTemplateFunctions.insert(templateFunction);

          // Create the function that calls the template.
          // This function dictates the indices the template will work with.
          std::string equationFunctionName = "initial_eq_" + std::to_string(equationCounter);
          ++equationCounter;

          auto equationFunction = createEquationFunction(
              builder, *equation, equationFunctionName, templateFunction,
              equationTemplateCalls,
              modelOp.getBodyRegion().getArgumentTypes());

          equationFunctions.insert(equationFunction);

          // Create the call to the instantiated template function
          auto equationCall = builder.create<mlir::func::CallOp>(loc, equationFunction, vars);
          equationCalls.emplace(equationFunction, equationCall);
        }
      }
    }

    builder.create<mlir::func::ReturnOp>(loc);

    // Remove the unused function arguments
    for (const auto& equationTemplateFunction : equationTemplateFunctions) {
      auto calls = equationTemplateCalls.equal_range(equationTemplateFunction);
      removeUnusedArguments(equationTemplateFunction, calls.first, calls.second);
    }

    for (const auto& equationFunction : equationFunctions) {
      auto calls = equationCalls.equal_range(equationFunction);
      removeUnusedArguments(equationFunction, calls.first, calls.second);
    }

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createDeinitICSolversFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      const ExternalSolvers& externalSolvers) const
  {
    return createDeinitSolversFunction(builder, deinitICSolversFunctionName, model, externalSolvers);
  }

  mlir::LogicalResult ModelConverter::createInitFunction(mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = builder.getFunctionType(llvm::None, getVoidPtrType());
    auto function = builder.create<mlir::func::FuncOp>(loc, initFunctionName, functionType);
    mlir::Block* bodyBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(bodyBlock);

    std::vector<MemberCreateOp> originalMembers;

    // The descriptors of the arrays that compose that runtime data structure
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

        mlir::Value array = builder.create<AllocOp>(memberCreateOp.getLoc(), arrayType, dynamicDimensions);

        originalMembers.push_back(memberCreateOp);
        structVariables.push_back(array);

        membersOpsMapping.map(memberCreateOp.getResult(), array);

      } else if (auto memberLoadOp = mlir::dyn_cast<MemberLoadOp>(op)) {
        mlir::Value array = membersOpsMapping.lookup(memberLoadOp.getMember());
        membersOpsMapping.map(memberLoadOp.getResult(), array);

      } else if (!mlir::isa<YieldOp>(op)) {
        builder.clone(op, membersOpsMapping);
      }
    }

    // Map the body arguments to the new arrays
    mlir::BlockAndValueMapping startOpsMapping;

    for (const auto& [arg, array] : llvm::zip(modelOp.getBodyRegion().getArguments(), structVariables)) {
      startOpsMapping.map(arg, array);
    }

    // Keep track of the variables for which a start value has been provided
    std::vector<bool> initializedVars(structVariables.size(), false);

    modelOp.getBodyRegion().walk([&](StartOp startOp) {
      unsigned int argNumber = startOp.getVariable().cast<mlir::BlockArgument>().getArgNumber();

      // Note that parameters must be set independently of the 'fixed' attribute
      // being true or false.

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

      // Set the variable as initialized
      initializedVars[argNumber] = true;
    });

    // The variables without a start attribute must be initialized to zero
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

    // Create the runtime data structure
    builder.setInsertionPointToEnd(bodyBlock);

    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value runtimeDataStructValue = builder.create<mlir::LLVM::UndefOp>(loc, runtimeDataStructType);

    // Set the start time
    mlir::Value startTime = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), options.startTime));
    startTime = typeConverter->materializeTargetConversion(builder, loc, runtimeDataStructType.getBody()[timeVariablePosition], startTime);
    runtimeDataStructValue = builder.create<mlir::LLVM::InsertValueOp>(loc, runtimeDataStructValue, startTime, builder.getIndexArrayAttr(1));

    // Add the model variables
    for (const auto& var : llvm::enumerate(structVariables)) {
      mlir::Type convertedType = typeConverter->convertType(var.value().getType());
      mlir::Value convertedVar = typeConverter->materializeTargetConversion(builder, loc, convertedType, var.value());
      runtimeDataStructValue = builder.create<mlir::LLVM::InsertValueOp>(loc, runtimeDataStructValue, convertedVar, builder.getIndexArrayAttr(var.index() + 2));
    }

    // Allocate the main runtime data structure
    mlir::Value runtimeDataStructPtr = alloc(builder, module, loc, runtimeDataStructType);
    builder.create<mlir::LLVM::StoreOp>(loc, runtimeDataStructValue, runtimeDataStructPtr);

    mlir::Value runtimeDataOpaquePtr = builder.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), runtimeDataStructPtr);

    builder.create<mlir::func::ReturnOp>(loc, runtimeDataOpaquePtr);

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createInitMainSolversFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      const ExternalSolvers& externalSolvers) const
  {
    return createInitSolversFunction(builder, initMainSolversFunctionName, model, externalSolvers);
  }

  mlir::LogicalResult ModelConverter::createDeinitMainSolversFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      const ExternalSolvers& externalSolvers) const
  {
    return createDeinitSolversFunction(builder, deinitMainSolversFunctionName, model, externalSolvers);
  }

  mlir::LogicalResult ModelConverter::createDeinitFunction(mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto loc = modelOp.getLoc();
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, deinitFunctionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the data from the struct
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value runtimeDataStruct = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    // Deallocate the arrays
    for (const auto& varType : llvm::enumerate(modelOp.getBodyRegion().getArgumentTypes())) {
      if (auto arrayType = varType.value().dyn_cast<ArrayType>()) {
        mlir::Value var = extractValue(builder, runtimeDataStruct, varType.value(), varType.index() + variablesOffset);
        builder.create<FreeOp>(loc, var);
      }
    }

    // Add "free" function to the module
    auto freeFunc = lookupOrCreateHeapFreeFn(builder, module);

    // Deallocate the data structure
    builder.create<mlir::LLVM::CallOp>(loc, freeFunc, function.getArgument(0));

    builder.create<mlir::func::ReturnOp>(loc);
    return mlir::success();
  }

  mlir::func::FuncOp ModelConverter::createEquationFunction(
      mlir::OpBuilder& builder,
      const ScheduledEquation& equation,
      llvm::StringRef equationFunctionName,
      mlir::func::FuncOp templateFunction,
      std::multimap<mlir::func::FuncOp, mlir::func::CallOp>& equationTemplateCalls,
      mlir::TypeRange varsTypes) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = equation.getOperation().getLoc();

    auto module = equation.getOperation()->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    // Function arguments ('time' + variables)
    llvm::SmallVector<mlir::Type, 3> argsTypes;
    //argsTypes.push_back(typeConverter->convertType(RealType::get(builder.getContext())));
    argsTypes.push_back(typeConverter->convertType(RealType::get(builder.getContext())));

    for (const auto& type : varsTypes) {
      //argsTypes.push_back(typeConverter->convertType(type));
      argsTypes.push_back(type);
    }

    // Function type. The equation doesn't need to return any value.
    auto functionType = builder.getFunctionType(argsTypes, llvm::None);

    auto function = builder.create<mlir::func::FuncOp>(loc, equationFunctionName, functionType);
    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto valuesFn = [&](marco::modeling::scheduling::Direction iterationDirection, Range range) -> std::tuple<mlir::Value, mlir::Value, mlir::Value> {
      assert(iterationDirection == marco::modeling::scheduling::Direction::Forward ||
          iterationDirection == marco::modeling::scheduling::Direction::Backward);

      if (iterationDirection == marco::modeling::scheduling::Direction::Forward) {
        mlir::Value begin = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getBegin()));
        mlir::Value end = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getEnd()));
        mlir::Value step = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));

        return std::make_tuple(begin, end, step);
      }

      mlir::Value begin = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getEnd() - 1));
      mlir::Value end = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(range.getBegin() - 1));
      mlir::Value step = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));

      return std::make_tuple(begin, end, step);
    };

    std::vector<mlir::Value> args;
    auto iterationRangesSet = equation.getIterationRanges();
    assert(iterationRangesSet.isSingleMultidimensionalRange());//todo: handle ragged case
    auto iterationRanges = iterationRangesSet.minContainingRange();

    for (size_t i = 0, e = equation.getNumOfIterationVars(); i < e; ++i) {
      auto values = valuesFn(equation.getSchedulingDirection(), iterationRanges[i]);

      args.push_back(std::get<0>(values));
      args.push_back(std::get<1>(values));
      args.push_back(std::get<2>(values));
    }

    mlir::ValueRange vars = function.getArguments();
    args.insert(args.end(), vars.begin(), vars.end());

    // Call the equation template function
    auto templateFunctionCall = builder.create<mlir::func::CallOp>(loc, templateFunction, args);
    equationTemplateCalls.emplace(templateFunction, templateFunctionCall);

    builder.create<mlir::func::ReturnOp>(loc);
    return function;
  }

  mlir::LogicalResult ModelConverter::createUpdateNonStateVariablesFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      const ConversionInfo& conversionInfo,
      ExternalSolvers& externalSolvers) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    ModelOp modelOp = model.getOperation();
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(modelOp->getParentOfType<mlir::ModuleOp>().getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, updateNonStateVariablesFunctionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the data from the struct
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    llvm::SmallVector<mlir::Value, 3> vars;

    mlir::Value time = extractValue(builder, structValue, RealType::get(builder.getContext()), timeVariablePosition);
    //time = typeConverter->materializeTargetConversion(builder, time.getLoc(), typeConverter->convertType(time.getType()), time);
    vars.push_back(time);

    for (const auto& varType : llvm::enumerate(modelOp.getBodyRegion().getArgumentTypes())) {
      mlir::Value var = extractValue(builder, structValue, varType.value(), varType.index() + variablesOffset);
      //var = typeConverter->materializeTargetConversion(builder, var.getLoc(), typeConverter->convertType(var.getType()), var);
      vars.push_back(var);
    }

    // Convert the equations into algorithmic code
    size_t equationTemplateCounter = 0;
    size_t equationCounter = 0;

    std::map<EquationTemplateInfo, mlir::func::FuncOp> equationTemplatesMap;
    std::set<mlir::func::FuncOp> equationTemplateFunctions;
    std::multimap<mlir::func::FuncOp, mlir::func::CallOp> equationTemplateCalls;
    std::set<mlir::func::FuncOp> equationFunctions;
    std::multimap<mlir::func::FuncOp, mlir::func::CallOp> equationCalls;

    // Get or create the template equation function for a scheduled equation
    auto getEquationTemplateFn = [&](const ScheduledEquation* equation) -> mlir::func::FuncOp {
      EquationTemplateInfo requestedTemplate(equation->getOperation(), equation->getSchedulingDirection());
      auto it = equationTemplatesMap.find(requestedTemplate);

      if (it != equationTemplatesMap.end()) {
        return it->second;
      }

      std::string templateFunctionName = "eq_template_" + std::to_string(equationTemplateCounter);
      ++equationTemplateCounter;

      auto explicitEquation = llvm::find_if(conversionInfo.explicitEquationsMap, [&](const auto& equationPtr) {
        return equationPtr.first == equation;
      });

      if (explicitEquation == conversionInfo.explicitEquationsMap.end()) {
        // The equation can't be made explicit and is not passed to any external solver
        return nullptr;
      }

      // Create the equation template function
      auto templateFunction = explicitEquation->second->createTemplateFunction(
          builder, templateFunctionName,
          modelOp.getBodyRegion().getArguments(),
          equation->getSchedulingDirection());

      auto timeArgumentIndex = equation->getNumOfIterationVars() * 3;

      templateFunction.insertArgument(
          timeArgumentIndex,
          RealType::get(builder.getContext()),
          builder.getDictionaryAttr(llvm::None),
          explicitEquation->second->getOperation().getLoc());

      templateFunction.walk([&](TimeOp timeOp) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(timeOp);
        timeOp.replaceAllUsesWith(templateFunction.getArgument(timeArgumentIndex));
        timeOp.erase();
      });

      return templateFunction;
    };

    for (const auto& scheduledBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *scheduledBlock) {
        if (externalSolvers.containsEquation(equation.get())) {
          // Let the external solver process the equation
          continue;

        } else {
          // The equation is handled by MARCO
          auto templateFunction = getEquationTemplateFn(equation.get());

          if (templateFunction == nullptr) {
            equation->getOperation().emitError("The equation can't be made explicit");
            equation->getOperation().dump();
            return mlir::failure();
          }

          equationTemplateFunctions.insert(templateFunction);

          // Create the function that calls the template.
          // This function dictates the indices the template will work with.
          std::string equationFunctionName = "eq_" + std::to_string(equationCounter);
          ++equationCounter;

          auto equationFunction = createEquationFunction(
              builder, *equation, equationFunctionName, templateFunction,
              equationTemplateCalls,
              modelOp.getBodyRegion().getArgumentTypes());

          equationFunctions.insert(equationFunction);

          // Create the call to the instantiated template function
          auto equationCall = builder.create<mlir::func::CallOp>(loc, equationFunction, vars);
          equationCalls.emplace(equationFunction, equationCall);
        }
      }
    }

    builder.create<mlir::func::ReturnOp>(loc);

    // Remove the unused function arguments
    for (const auto& equationTemplateFunction : equationTemplateFunctions) {
      auto calls = equationTemplateCalls.equal_range(equationTemplateFunction);
      removeUnusedArguments(equationTemplateFunction, calls.first, calls.second);
    }

    for (const auto& equationFunction : equationFunctions) {
      auto calls = equationCalls.equal_range(equationFunction);
      removeUnusedArguments(equationFunction, calls.first, calls.second);
    }

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createUpdateStateVariablesFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      ExternalSolvers& externalSolvers) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto modelOp = model.getOperation();
    auto loc = modelOp.getLoc();

    auto varTypes = modelOp.getBodyRegion().getArgumentTypes();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(modelOp->getParentOfType<mlir::ModuleOp>().getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, updateStateVariablesFunctionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the state variables from the opaque pointer
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    auto returnOp = builder.create<mlir::func::ReturnOp>(loc);
    builder.setInsertionPoint(returnOp);

    // External solvers
    mlir::Value externalSolversPtr = extractValue(builder, structValue, getVoidPtrType(), externalSolversPosition);

    externalSolversPtr = builder.create<mlir::LLVM::BitcastOp>(
        externalSolversPtr.getLoc(),
        mlir::LLVM::LLVMPointerType::get(getExternalSolversStructType(builder.getContext(), externalSolvers)),
        externalSolversPtr);

    assert(externalSolversPtr.getType().isa<mlir::LLVM::LLVMPointerType>());
    mlir::Value externalSolversStruct = builder.create<mlir::LLVM::LoadOp>(externalSolversPtr.getLoc(), externalSolversPtr);

    std::vector<mlir::Value> variables;

    for (const auto& variable : modelOp.getBodyRegion().getArguments()) {
      size_t index = variable.getArgNumber();
      mlir::Value var = extractValue(builder, structValue, varTypes[index], index + variablesOffset);
      variables.push_back(var);
    }

    for (auto& solver : llvm::enumerate(externalSolvers)) {
      if (!solver.value()->isEnabled()) {
        continue;
      }

      mlir::Type externalSolverRuntimeDataType = solver.value()->getRuntimeDataType(builder.getContext());

      mlir::Value solverDataPtr = extractValue(
          builder, externalSolversStruct,
          typeConverter->convertType(mlir::LLVM::LLVMPointerType::get(externalSolverRuntimeDataType)),
          solver.index());

      if (auto res = solver.value()->processUpdateStatesFunction(builder, solverDataPtr, function, variables); mlir::failed(res)) {
        return res;
      }
    }

    if (options.solver == Solver::forwardEuler) {
      // Update the state variables by applying the forward Euler method
      builder.setInsertionPoint(returnOp);
      mlir::Value timeStep = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), options.timeStep));

      auto apply = [&](mlir::OpBuilder& nestedBuilder, mlir::Value scalarState, mlir::Value scalarDerivative) -> mlir::Value {
        mlir::Value result = builder.create<MulOp>(scalarDerivative.getLoc(), scalarDerivative.getType(), scalarDerivative, timeStep);
        result = builder.create<AddOp>(scalarDerivative.getLoc(), scalarState.getType(), scalarState, result);
        return result;
      };

      const auto& derivativesMap = model.getDerivativesMap();

      for (const auto& var : model.getVariables()) {
        auto varArgNumber = var->getValue().cast<mlir::BlockArgument>().getArgNumber();
        mlir::Value variable = variables[varArgNumber];

        if (derivativesMap.hasDerivative(varArgNumber)) {
          auto derArgNumber = derivativesMap.getDerivative(varArgNumber);
          mlir::Value derivative = variables[derArgNumber];

          assert(variable.getType().cast<ArrayType>().getShape() == derivative.getType().cast<ArrayType>().getShape());

          if (auto arrayType = variable.getType().cast<ArrayType>(); arrayType.isScalar()) {
            mlir::Value scalarState = builder.create<LoadOp>(derivative.getLoc(), variable, llvm::None);
            mlir::Value scalarDerivative = builder.create<LoadOp>(derivative.getLoc(), derivative, llvm::None);
            mlir::Value updatedValue = apply(builder, scalarState, scalarDerivative);
            builder.create<StoreOp>(derivative.getLoc(), updatedValue, variable, llvm::None);

          } else {
            // Create the loops to iterate on each scalar variable
            std::vector<mlir::Value> lowerBounds;
            std::vector<mlir::Value> upperBounds;
            std::vector<mlir::Value> steps;

            for (unsigned int i = 0; i < arrayType.getRank(); ++i) {
              lowerBounds.push_back(builder.create<ConstantOp>(derivative.getLoc(), builder.getIndexAttr(0)));
              mlir::Value dimension = builder.create<ConstantOp>(derivative.getLoc(), builder.getIndexAttr(i));
              upperBounds.push_back(builder.create<DimOp>(variable.getLoc(), variable, dimension));
              steps.push_back(builder.create<ConstantOp>(variable.getLoc(), builder.getIndexAttr(1)));
            }

            mlir::scf::buildLoopNest(
                builder, loc, lowerBounds, upperBounds, steps,
                [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange indices) {
                  mlir::Value scalarState = nestedBuilder.create<LoadOp>(loc, variable, indices);
                  mlir::Value scalarDerivative = nestedBuilder.create<LoadOp>(loc, derivative, indices);
                  mlir::Value updatedValue = apply(nestedBuilder, scalarState, scalarDerivative);
                  nestedBuilder.create<StoreOp>(loc, updatedValue, variable, indices);
                });
          }
        }
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createIncrementTimeFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      ExternalSolvers& externalSolvers) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    ModelOp modelOp = model.getOperation();
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(modelOp->getParentOfType<mlir::ModuleOp>().getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, incrementTimeFunctionName,
        builder.getFunctionType(getVoidPtrType(), builder.getI1Type()));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the data from the struct
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    // Extract the external solvers data
    mlir::Value runtimeDataStruct = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    mlir::Value externalSolversPtr = extractValue(builder, runtimeDataStruct, getVoidPtrType(), externalSolversPosition);

    externalSolversPtr = builder.create<mlir::LLVM::BitcastOp>(
        externalSolversPtr.getLoc(),
        mlir::LLVM::LLVMPointerType::get(getExternalSolversStructType(builder.getContext(), externalSolvers)),
        externalSolversPtr);

    assert(externalSolversPtr.getType().isa<mlir::LLVM::LLVMPointerType>());
    mlir::Value externalSolversStruct = builder.create<mlir::LLVM::LoadOp>(externalSolversPtr.getLoc(), externalSolversPtr);

    std::vector<mlir::Value> externalSolverDataPtrs;
    externalSolverDataPtrs.resize(externalSolvers.size());

    // Deallocate each solver data structure
    for (auto& solver : llvm::enumerate(externalSolvers)) {
      if (!solver.value()->isEnabled()) {
        continue;
      }

      mlir::Type externalSolverRuntimeDataType = solver.value()->getRuntimeDataType(builder.getContext());

      mlir::Value solverDataPtr = extractValue(
          builder, externalSolversStruct,
          typeConverter->convertType(mlir::LLVM::LLVMPointerType::get(externalSolverRuntimeDataType)),
          solver.index());

      externalSolverDataPtrs[solver.index()] = solverDataPtr;
    }

    mlir::Value increasedTime = externalSolvers.getCurrentTime(builder, externalSolverDataPtrs);

    if (!increasedTime) {
      mlir::Value timeStep = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), options.timeStep));
      mlir::Value currentTime = extractValue(builder, runtimeDataStruct, RealType::get(builder.getContext()), timeVariablePosition);
      increasedTime = builder.create<AddOp>(loc, currentTime.getType(), currentTime, timeStep);
    }

    // Store the increased time into the runtime data structure
    increasedTime = typeConverter->materializeTargetConversion(builder, loc, runtimeDataStructType.getBody()[timeVariablePosition], increasedTime);
    runtimeDataStruct = builder.create<mlir::LLVM::InsertValueOp>(loc, runtimeDataStruct, increasedTime, builder.getIndexArrayAttr(timeVariablePosition));

    mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(runtimeDataStruct.getType());
    mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(loc, structPtrType, function.getArgument(0));
    builder.create<mlir::LLVM::StoreOp>(loc, runtimeDataStruct, structPtr);

    // Check if the current time is less than the end time
    mlir::Value endTime = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), options.endTime));
    mlir::Value epsilon = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 10e-06));
    endTime = builder.create<SubOp>(loc, endTime.getType(), endTime, epsilon);

    endTime = typeConverter->materializeTargetConversion(builder, loc, typeConverter->convertType(endTime.getType()), endTime);

    mlir::Value condition = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, increasedTime, endTime);
    builder.create<mlir::func::ReturnOp>(loc, condition);

    return mlir::success();
  }

  void ModelConverter::printSeparator(mlir::OpBuilder& builder, mlir::ModuleOp module) const
  {
    // Get the mangled function name
    RuntimeFunctionsMangling mangling;
    auto functionName = mangling.getMangledFunction("print_csv_separator", mangling.getVoidType(), llvm::None);

    // Get or declare the external function
    auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(voidType, llvm::None);
    auto function = getOrInsertFunction(builder, module, functionName, llvmFnType);

    // Call it
    builder.create<mlir::LLVM::CallOp>(function.getLoc(), function, llvm::None);
  }

  void ModelConverter::printNewline(mlir::OpBuilder& builder, mlir::ModuleOp module) const
  {
    // Get the mangled function name
    RuntimeFunctionsMangling mangling;
    auto functionName = mangling.getMangledFunction("print_csv_newline", mangling.getVoidType(), llvm::None);

    // Get or declare the external function
    auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(voidType, llvm::None);
    auto function = getOrInsertFunction(builder, module, functionName, llvmFnType);

    // Call it
    builder.create<mlir::LLVM::CallOp>(function.getLoc(), function, llvm::None);
  }

  mlir::Value ModelConverter::getOrCreateGlobalString(
      mlir::Location loc,
      mlir::OpBuilder& builder,
      mlir::StringRef name,
      mlir::StringRef value,
      mlir::ModuleOp module) const
  {
    // Create the global at the entry of the module
    mlir::LLVM::GlobalOp global;

    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(loc, type, true, mlir::LLVM::Linkage::Internal, name, builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string
    mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);

    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc,
        mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc,
        getVoidPtrType(),
        globalPtr, llvm::makeArrayRef({cst0, cst0}));
  }

  mlir::LLVM::LLVMFuncOp ModelConverter::getOrInsertPrintNameFunction(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module) const
  {
    // Get the mangled function name
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<std::string, 3> mangledArgTypes;
    mangledArgTypes.push_back(mangling.getVoidPointerType());
    mangledArgTypes.push_back(mangling.getIntegerType(64));
    mangledArgTypes.push_back(mangling.getPointerType(mangling.getIntegerType(64)));

    auto functionName = mangling.getMangledFunction("print_csv_name", mangling.getVoidType(), mangledArgTypes);

    // Get or declare the external function
    llvm::SmallVector<mlir::Type, 3> argTypes;
    argTypes.push_back(getVoidPtrType());
    argTypes.push_back(builder.getI64Type());
    argTypes.push_back(mlir::LLVM::LLVMPointerType::get(builder.getI64Type()));

    auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(voidType, argTypes);
    return getOrInsertFunction(builder, module, functionName, llvmFnType);
  }

  void ModelConverter::printVariableName(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Value name,
      mlir::Value value,
      const IndexSet& filteredIndices,
      bool shouldPrependSeparator) const
  {
    if (auto arrayType = value.getType().dyn_cast<ArrayType>()) {
      if (arrayType.getRank() == 0) {
        printScalarVariableName(builder, module, name, shouldPrependSeparator);
      } else {
        printArrayVariableName(builder, module, name, value, filteredIndices, shouldPrependSeparator);
      }
    } else {
      printScalarVariableName(builder, module, name, shouldPrependSeparator);
    }
  }

  void ModelConverter::printScalarVariableName(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Value name,
      bool shouldPrependSeparator) const
  {
    if (shouldPrependSeparator) {
      printSeparator(builder, module);
    }

    auto loc = name.getLoc();
    mlir::Value rank = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    mlir::Value indices = builder.create<mlir::LLVM::NullOp>(loc, mlir::LLVM::LLVMPointerType::get(builder.getI64Type()));

    auto function = getOrInsertPrintNameFunction(builder, module);
    builder.create<mlir::LLVM::CallOp>(loc, function, mlir::ValueRange({ name, rank, indices }));
  }

  void ModelConverter::printArrayVariableName(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Value name,
      mlir::Value value,
      const IndexSet& filteredIndices,
      bool shouldPrependSeparator) const
  {
    auto loc = name.getLoc();
    assert(value.getType().isa<ArrayType>());
    auto arrayType = value.getType().cast<ArrayType>();

    // Get a reference to the function to print the name
    auto function = getOrInsertPrintNameFunction(builder, module);

    // The arguments to be passed to the function
    llvm::SmallVector<mlir::Value, 3> args;
    args.push_back(name);

    // Create the rank constant and the array of the indices
    mlir::Value rank = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(arrayType.getRank()));
    args.push_back(rank);

    auto heapAllocFn = lookupOrCreateHeapAllocFn(builder, module);

    mlir::Type indexPtrType = mlir::LLVM::LLVMPointerType::get(builder.getI64Type());
    mlir::Value indexNullPtr = builder.create<mlir::LLVM::NullOp>(loc, indexPtrType);
    mlir::Value indicesGepPtr = builder.create<mlir::LLVM::GEPOp>(loc, indexPtrType, indexNullPtr, rank);
    mlir::Value indicesSizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(loc, builder.getI64Type(), indicesGepPtr);
    mlir::Value indicesOpaquePtr = builder.create<mlir::LLVM::CallOp>(loc, heapAllocFn, indicesSizeBytes).getResult(0);
    mlir::Value indicesPtr = builder.create<mlir::LLVM::BitcastOp>(loc, indexPtrType, indicesOpaquePtr);
    args.push_back(indicesPtr);

    for (const auto& filteredRange : llvm::make_range(filteredIndices.rangesBegin(), filteredIndices.rangesEnd())) {
      // Create the lower and upper bounds
      assert(filteredRange.rank() == arrayType.getRank());

      llvm::SmallVector<mlir::Value, 3> lowerBounds;
      llvm::SmallVector<mlir::Value, 3> upperBounds;

      mlir::Value one = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
      llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

      for (size_t i = 0; i < filteredRange.rank(); ++i) {
        lowerBounds.push_back(builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(filteredRange[i].getBegin())));
        upperBounds.push_back(builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(filteredRange[i].getEnd())));
      }

      // Create nested loops in order to iterate on each dimension of the array
      mlir::scf::buildLoopNest(
          builder, loc, lowerBounds, upperBounds, steps,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange indices) {
            // Print the separator and the variable name
            printSeparator(builder, module);

            for (auto index : llvm::enumerate(indices)) {
              mlir::Type convertedType = typeConverter->convertType(index.value().getType());
              mlir::Value indexValue = typeConverter->materializeTargetConversion(builder, loc, convertedType, index.value());

              if (convertedType.getIntOrFloatBitWidth() < 64) {
                indexValue = builder.create<mlir::arith::ExtSIOp>(loc, builder.getI64Type(), indexValue);
              } else if (convertedType.getIntOrFloatBitWidth() > 64) {
                indexValue = builder.create<mlir::arith::TruncIOp>(loc, builder.getI64Type(), indexValue);
              }

              mlir::Value offset = builder.create<mlir::arith::ConstantOp>(
                  loc, typeConverter->getIndexType(), builder.getIntegerAttr(typeConverter->getIndexType(), index.index()));

              mlir::Value indexPtr = builder.create<mlir::LLVM::GEPOp>(
                  loc, indexPtrType, indicesPtr, offset);

              // Arrays are 1-based in Modelica, so we add 1 in order to print indexes that are
              // coherent with the model source.
              mlir::Value increment = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(indexValue.getType(), 1));
              indexValue = builder.create<mlir::arith::AddIOp>(loc, indexValue.getType(), indexValue, increment);

              builder.create<mlir::LLVM::StoreOp>(loc, indexValue, indexPtr);
            }

            builder.create<mlir::LLVM::CallOp>(loc, function, args);
          });
    }

    // Deallocate the indices array
    auto heapFreeFn = lookupOrCreateHeapFreeFn(builder, module);
    builder.create<mlir::LLVM::CallOp>(loc, heapFreeFn, indicesOpaquePtr);
  }

  mlir::LogicalResult ModelConverter::createPrintHeaderFunction(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model) const
  {
    auto modelOp = model.getOperation();
    auto module = modelOp.getOperation()->getParentOfType<mlir::ModuleOp>();

    auto callback = [&](llvm::StringRef name, mlir::Value value, const IndexSet& filteredIndices, mlir::ModuleOp module, size_t processedValues) -> mlir::LogicalResult {
      auto loc = modelOp.getLoc();

      std::string symbolName = "var" + std::to_string(processedValues);
      llvm::SmallString<10> terminatedName(name);
      terminatedName.append("\0");
      mlir::Value symbol = getOrCreateGlobalString(loc, builder, symbolName, llvm::StringRef(terminatedName.c_str(), terminatedName.size() + 1), module);

      bool shouldPrintSeparator = processedValues != 0;
      printVariableName(builder, module, symbol, value, filteredIndices, shouldPrintSeparator);
      return mlir::success();
    };

    return createPrintFunctionBody(builder, module, model, printHeaderFunctionName, callback);
  }

  void ModelConverter::printVariable(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Value var,
      const IndexSet& filteredIndices,
      bool shouldPrependSeparator) const
  {
    if (auto arrayType = var.getType().dyn_cast<ArrayType>()) {
      if (arrayType.getRank() == 0) {
        mlir::Value value = builder.create<LoadOp>(var.getLoc(), var);
        printScalarVariable(builder, module, value, shouldPrependSeparator);
      } else {
        printArrayVariable(builder, module, var, filteredIndices, shouldPrependSeparator);
      }
    } else {
      printScalarVariable(builder, module, var, shouldPrependSeparator);
    }
  }

  void ModelConverter::printScalarVariable(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Value var,
      bool shouldPrependSeparator) const
  {
    if (shouldPrependSeparator) {
      printSeparator(builder, module);
    }

    printElement(builder, module, var);
  }

  void ModelConverter::printArrayVariable(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      mlir::Value var,
      const IndexSet& filteredIndices,
      bool shouldPrependSeparator) const
  {
    mlir::Location loc = var.getLoc();
    assert(var.getType().isa<ArrayType>());

    for (const auto& filteredRange : llvm::make_range(filteredIndices.rangesBegin(), filteredIndices.rangesEnd())) {
      auto arrayType = var.getType().cast<ArrayType>();
      assert(filteredRange.rank() == arrayType.getRank());

      llvm::SmallVector<mlir::Value, 3> lowerBounds;
      llvm::SmallVector<mlir::Value, 3> upperBounds;

      mlir::Value one = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
      llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

      for (size_t i = 0; i < filteredRange.rank(); ++i) {
        lowerBounds.push_back(builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(filteredRange[i].getBegin())));
        upperBounds.push_back(builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(filteredRange[i].getEnd())));
      }

      // Create nested loops in order to iterate on each dimension of the array
      mlir::scf::buildLoopNest(
          builder, loc, lowerBounds, upperBounds, steps,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position) {
            mlir::Value value = nestedBuilder.create<LoadOp>(loc, var, position);

            printSeparator(nestedBuilder, module);
            printElement(nestedBuilder, module, value);
          });
    }
  }

  void ModelConverter::printElement(mlir::OpBuilder& builder, mlir::ModuleOp module, mlir::Value value) const
  {
    auto loc = value.getLoc();
    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Type, 1> argTypes;
    llvm::SmallVector<std::string, 1> mangledArgTypes;

    mlir::Type convertedType = typeConverter->convertType(value.getType());
    argTypes.push_back(convertedType);
    value = typeConverter->materializeTargetConversion(builder, loc, convertedType, value);

    if (convertedType.isa<mlir::IntegerType>()) {
      mangledArgTypes.push_back(mangling.getIntegerType(convertedType.getIntOrFloatBitWidth()));
    } else if (convertedType.isa<mlir::FloatType>()) {
      mangledArgTypes.push_back(mangling.getFloatingPointType(convertedType.getIntOrFloatBitWidth()));
    } else {
      llvm_unreachable("The value can't be printed because of its unknown type");
    }

    auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(voidType, argTypes);
    auto functionName = mangling.getMangledFunction("print_csv", mangling.getVoidType(), mangledArgTypes);
    auto function = getOrInsertFunction(builder, module, functionName, llvmFnType);

    builder.create<mlir::LLVM::CallOp>(function.getLoc(), function, value);
  }

  mlir::LogicalResult ModelConverter::createPrintFunction(
      mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model) const
  {
    auto modelOp = model.getOperation();
    auto module = modelOp.getOperation()->getParentOfType<mlir::ModuleOp>();

    auto callback = [&](llvm::StringRef name, mlir::Value value, const IndexSet& filteredIndices, mlir::ModuleOp module, size_t processedValues) -> mlir::LogicalResult {
      bool shouldPrintSeparator = processedValues != 0;
      printVariable(builder, module, value, filteredIndices, shouldPrintSeparator);
      return mlir::success();
    };

    return createPrintFunctionBody(builder, module, model, printFunctionName, callback);
  }

  mlir::LogicalResult ModelConverter::createPrintFunctionBody(
      mlir::OpBuilder& builder,
      mlir::ModuleOp module,
      const Model<ScheduledEquationsBlock>& model,
      llvm::StringRef functionName,
      std::function<mlir::LogicalResult(llvm::StringRef, mlir::Value, const IndexSet&, mlir::ModuleOp, size_t)> elementCallback) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto modelOp = model.getOperation();
    auto loc = modelOp.getLoc();

    auto variableTypes = model.getVariables().getTypes();

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    auto function = builder.create<mlir::func::FuncOp>(
        loc, functionName,
        builder.getFunctionType(getVoidPtrType(), llvm::None));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Load the runtime data structure
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    // Get the names of the variables
    auto variableNames = model.getOperation().variableNames();

    // Map each variable to its argument number.
    // It must be noted that the arguments list also contains the derivatives,
    // so its size can be greater than the number of names.

    llvm::StringMap<size_t> variablePositionByName;

    for (const auto& variable : llvm::enumerate(variableNames)) {
      variablePositionByName[variable.value()] = variable.index();
    }

    // The positions have been saved, so we can now sort the names
    std::vector<llvm::StringRef> sortedVariableNames(variableNames.begin(), variableNames.end());

    llvm::sort(sortedVariableNames, [](llvm::StringRef x, llvm::StringRef y) -> bool {
      return x.compare_insensitive(y) < 0;
    });

    size_t processedValues = 0;

    mlir::Value time = extractValue(builder, structValue, RealType::get(builder.getContext()), timeVariablePosition);

    if (auto res = elementCallback("time", time, IndexSet(MultidimensionalRange(Range(0, 1))), module, processedValues++); mlir::failed(res)) {
      return res;
    }

    // Print the other variables
    const auto& derivativesMap = model.getDerivativesMap();

    for (const auto& name : sortedVariableNames) {
      size_t position = variablePositionByName[name];

      if (derivativesMap.isDerivative(position)) {
        continue;
      }

      unsigned int rank = 0;

      if (auto arrayType = variableTypes[position].dyn_cast<ArrayType>()) {
        rank = arrayType.getRank();
      }

      auto filters = options.variableFilter->getVariableInfo(name, rank);
      IndexSet filteredIndices = getFilteredIndices(variableTypes[position], filters);

      if (filteredIndices.empty()) {
        // Nothing to print, so we can also skip the extraction of the variable
        // from the runtime data structure.
        continue;
      }

      mlir::Value value = extractValue(builder, structValue, variableTypes[position], position + variablesOffset);

      if (auto res = elementCallback(name, value, filteredIndices, module, processedValues++); mlir::failed(res)) {
        return res;
      }
    }

    // Print the derivatives
    for (const auto& name : variableNames) {
      size_t varPosition = variablePositionByName[name];

      if (!derivativesMap.hasDerivative(varPosition)) {
        continue;
      }

      auto derPosition = derivativesMap.getDerivative(varPosition);

      unsigned int rank = 0;

      if (auto arrayType = variableTypes[derPosition].dyn_cast<ArrayType>()) {
        rank = arrayType.getRank();
      }

      auto filters = options.variableFilter->getVariableDerInfo(name, rank);
      IndexSet filteredIndices = getFilteredIndices(variableTypes[derPosition], filters);
      filteredIndices -= derivativesMap.getDerivedIndices(varPosition);

      if (filteredIndices.empty()) {
        continue;
      }

      llvm::SmallString<15> derName;
      derName.append("der(");
      derName.append(name);
      derName.append(")");

      mlir::Value value = extractValue(builder, structValue, variableTypes[derPosition], derPosition + variablesOffset);

      if (auto res = elementCallback(derName, value, filteredIndices, module, processedValues++); mlir::failed(res)) {
        return res;
      }
    }

    // Print a newline character after all the variables have been processed
    printNewline(builder, module);

    builder.create<mlir::func::ReturnOp>(loc);
    return mlir::success();
  }
}
