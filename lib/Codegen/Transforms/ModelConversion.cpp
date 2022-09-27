#include "marco/Codegen/Transforms/ModelConversion.h"
#include "marco/Codegen/Conversion/IDAToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/DerivativesMap.h"
#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/IDASolver.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Codegen/Runtime.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_MODELCONVERSIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  struct FuncOpTypesPattern : public mlir::OpConversionPattern<mlir::func::FuncOp>
  {
    using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type, 3> resultTypes;
      llvm::SmallVector<mlir::Type, 3> argTypes;

      for (const auto& type : op.getFunctionType().getResults()) {
        resultTypes.push_back(getTypeConverter()->convertType(type));
      }

      for (const auto& type : op.getFunctionType().getInputs()) {
        argTypes.push_back(getTypeConverter()->convertType(type));
      }

      auto functionType = rewriter.getFunctionType(argTypes, resultTypes);
      auto newOp = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(op, op.getSymName(), functionType);

      mlir::Block* entryBlock = newOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::BlockAndValueMapping mapping;

      // Clone the blocks structure
      for (auto& block : llvm::enumerate(op.getBody())) {
        if (block.index() == 0) {
          mapping.map(&block.value(), entryBlock);
        } else {
          std::vector<mlir::Location> argLocations;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
          }

          auto signatureConversion = typeConverter->convertBlockSignature(&block.value());

          if (!signatureConversion) {
            return mlir::failure();
          }

          mlir::Block* clonedBlock = rewriter.createBlock(
              &newOp.getBody(),
              newOp.getBody().end(),
              signatureConversion->getConvertedTypes(),
              argLocations);

          mapping.map(&block.value(), clonedBlock);
        }
      }

      for (auto& block : op.getBody().getBlocks()) {
        mlir::Block* clonedBlock = mapping.lookup(&block);
        rewriter.setInsertionPointToStart(clonedBlock);

        // Cast the block arguments
        for (const auto& [original, cloned] : llvm::zip(block.getArguments(), clonedBlock->getArguments())) {
          mlir::Value arg = typeConverter->materializeSourceConversion(
              rewriter, cloned.getLoc(), original.getType(), cloned);

          mapping.map(original, arg);
        }

        // Clone the operations
        for (auto& bodyOp : block.getOperations()) {
          if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(bodyOp)) {
            std::vector<mlir::Value> returnValues;

            for (auto returnValue : returnOp.operands()) {
              returnValues.push_back(getTypeConverter()->materializeTargetConversion(
                  rewriter, returnOp.getLoc(),
                  getTypeConverter()->convertType(returnValue.getType()),
                  mapping.lookup(returnValue)));
            }

            rewriter.create<mlir::func::ReturnOp>(returnOp.getLoc(), returnValues);
          } else {
            rewriter.clone(bodyOp, mapping);
          }
        }
      }

      return mlir::success();
    }
  };

  struct CallOpTypesPattern : public mlir::OpConversionPattern<mlir::func::CallOp>
  {
    using mlir::OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 3> values;

      for (const auto& operand : op.operands()) {
        values.push_back(getTypeConverter()->materializeTargetConversion(
            rewriter, operand.getLoc(), getTypeConverter()->convertType(operand.getType()), operand));
      }

      llvm::SmallVector<mlir::Type, 3> resultTypes;

      for (const auto& type : op.getResults().getTypes()) {
        resultTypes.push_back(getTypeConverter()->convertType(type));
      }

      auto newOp = rewriter.create<mlir::func::CallOp>(op.getLoc(), op.getCallee(), resultTypes, values);

      llvm::SmallVector<mlir::Value, 3> results;

      for (const auto& [oldResult, newResult] : llvm::zip(op.getResults(), newOp.getResults())) {
        if (oldResult.getType() != newResult.getType()) {
          results.push_back(getTypeConverter()->materializeSourceConversion(
              rewriter, newResult.getLoc(), oldResult.getType(), newResult));
        } else {
          results.push_back(newResult);
        }
      }

      rewriter.replaceOp(op, results);
      return mlir::success();
    }
  };
}

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

  /// Get the LLVM function with the given name, or declare it inside the module if not present.
  mlir::LLVM::LLVMFuncOp getOrCreateLLVMFunctionDecl(
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
  ModelConverter::ModelConverter(
      mlir::LLVMTypeConverter& typeConverter,
      VariableFilter& variablesFilter,
      Solver solver,
      double startTime,
      double endTime,
      double timeStep,
      IDAOptions idaOptions)
      : typeConverter(&typeConverter),
        variablesFilter(&variablesFilter),
        solver(solver),
        startTime(startTime),
        endTime(endTime),
        timeStep(timeStep),
        idaOptions(idaOptions)
  {
  }

  mlir::LogicalResult ModelConverter::convertInitialModel(mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model) const
  {
    // Determine the external solvers to be used
    ExternalSolvers solvers;

    DerivativesMap emptyDerivativesMap;

    auto ida = std::make_unique<IDASolver>(typeConverter, emptyDerivativesMap, idaOptions, 0, 0, timeStep);
    ida->setEnabled(solver.getKind() == Solver::Kind::ida);

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

    auto ida = std::make_unique<IDASolver>(typeConverter, model.getDerivativesMap(), idaOptions, startTime, endTime, timeStep);
    ida->setEnabled(solver.getKind() == Solver::Kind::ida);

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
    mlir::Value resultOpaquePtr = builder.create<mlir::LLVM::CallOp>(loc, heapAllocFunc, sizeBytes).getResult();

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

    auto functionType = builder.getFunctionType(llvm::None, getVoidPtrType());
    auto function = builder.create<mlir::func::FuncOp>(loc, getModelNameFunctionName, functionType);

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value name = getOrCreateGlobalString(builder, loc, module, "modelName", modelOp.getSymName());
    builder.create<mlir::func::ReturnOp>(loc, name);

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createGetNumOfVariablesFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = builder.getFunctionType(llvm::None, builder.getI64Type());
    auto function = builder.create<mlir::func::FuncOp>(loc, getNumOfVariablesFunctionName, functionType);

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value result = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(modelOp.getBodyRegion().getNumArguments()));
    builder.create<mlir::func::ReturnOp>(loc, result);

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createGetVariableNameFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type charPtrType = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8));

    auto functionType = mlir::LLVM::LLVMFunctionType::get(charPtrType, builder.getI64Type());
    auto function = builder.create<mlir::LLVM::LLVMFuncOp>(loc, getVariableNameFunctionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), charPtrType, loc);

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
    mlir::Value defaultOperand = getOrCreateGlobalString(builder, loc, module, "varUnknown", "unknown");

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks
    llvm::SmallString<10> terminatedName;

    for (const auto& name : llvm::enumerate(names)) {
      size_t i = name.index();
      builder.setInsertionPointToStart(caseBlocks[i]);

      std::string symbolName = "var" + std::to_string(name.index());
      terminatedName = name.value();
      terminatedName.append("\0");
      mlir::Value result = getOrCreateGlobalString(builder, loc, module, symbolName, llvm::StringRef(terminatedName.c_str(), terminatedName.size() + 1));

      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createGetVariableRankFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = mlir::LLVM::LLVMFunctionType::get(builder.getI64Type(), builder.getI64Type());
    auto function = builder.create<mlir::LLVM::LLVMFuncOp>(loc, getVariableRankFunctionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch
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
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    builder.setInsertionPointToStart(entryBlock);

    // Populate the case blocks
    for (const auto& type : llvm::enumerate(types)) {
      size_t i = type.index();
      builder.setInsertionPointToStart(caseBlocks[i]);

      int64_t rank = type.value().cast<ArrayType>().getRank();
      mlir::Value result = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(rank));
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createGetVariableNumOfPrintableRangesFunction(
      mlir::OpBuilder& builder, ModelOp modelOp, const DerivativesMap& derivativesMap) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = mlir::LLVM::LLVMFunctionType::get(builder.getI64Type(), builder.getI64Type());
    auto function = builder.create<mlir::LLVM::LLVMFuncOp>(loc, getVariableNumOfPrintableRangesFunctionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

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
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks
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

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createGetVariablePrintableRangeBeginFunction(
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

  mlir::LogicalResult ModelConverter::createGetVariablePrintableRangeEndFunction(
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

  mlir::LogicalResult ModelConverter::createGetVariablePrintableRangeBoundariesFunction(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const DerivativesMap& derivativesMap,
      llvm::StringRef functionName,
      std::function<int64_t(const Range&)> boundaryGetterCallback) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = mlir::LLVM::LLVMFunctionType::get(builder.getI64Type(), { builder.getI64Type(), builder.getI64Type(), builder.getI64Type() });
    auto function = builder.create<mlir::LLVM::LLVMFuncOp>(loc, functionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

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
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks
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

      if (auto res = createGetPrintableIndexSetBoundariesFunction(builder, loc, module, calleeName, indices, boundaryGetterCallback, rangeBoundaryFuncOps, baseRangeFunctionName, rangesCounter); mlir::failed(res)) {
        return res;
      }

      std::vector<mlir::Value> args;
      args.push_back(function.getArgument(1));
      args.push_back(function.getArgument(2));
      mlir::Value result = builder.create<mlir::func::CallOp>(loc, calleeName, builder.getI64Type(), args).getResult(0);
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createGetPrintableIndexSetBoundariesFunction(
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

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = builder.getFunctionType({ builder.getI64Type(), builder.getI64Type() }, builder.getI64Type());
    auto function = builder.create<mlir::func::FuncOp>(loc, functionName, functionType);

    // Collect the multidimensional ranges and sort them
    llvm::SmallVector<MultidimensionalRange> ranges;

    for (const auto& range : llvm::make_range(indexSet.rangesBegin(), indexSet.rangesEnd())) {
      ranges.push_back(range);
    }

    llvm::sort(ranges);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

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

    // Populate the case blocks
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

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::func::FuncOp ModelConverter::createGetPrintableMultidimensionalRangeBoundariesFunction(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ModuleOp module,
      llvm::StringRef functionName,
      const MultidimensionalRange& ranges,
      std::function<int64_t(const Range&)> boundaryGetterCallback) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = builder.getFunctionType(builder.getI64Type(), builder.getI64Type());
    auto function = builder.create<mlir::func::FuncOp>(loc, functionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

    // Create the blocks and the switch
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

    // Populate the case blocks
    for (unsigned int i = 0, e = ranges.rank(); i < e; ++i) {
      builder.setInsertionPointToStart(caseBlocks[i]);
      int64_t boundary = boundaryGetterCallback(ranges[i]);
      mlir::Value result = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(boundary));
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

    return function;
  }

  mlir::LogicalResult ModelConverter::createGetVariableValueFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type int64PtrType = mlir::LLVM::LLVMPointerType::get(builder.getI64Type());

    auto functionType = mlir::LLVM::LLVMFunctionType::get(builder.getF64Type(), { getVoidPtrType(), builder.getI64Type(), int64PtrType });
    auto function = builder.create<mlir::LLVM::LLVMFuncOp>(loc, getVariableValueFunctionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getF64Type(), loc);

    // Create the blocks
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

    // Load the runtime data structure
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    // Create the switch
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getF64FloatAttr(0));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(1), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    // Populate the case blocks
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

      args[0] = extractValue(builder, structValue, arrayType, variablesOffset + i);
      args[0] = typeConverter->materializeTargetConversion(builder, loc, typeConverter->convertType(args[0].getType()), args[0]);

      auto callOp = builder.create<mlir::func::CallOp>(loc, callee, args);
      mlir::Value result = callOp.getResult(0);
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::func::FuncOp ModelConverter::createScalarVariableGetter(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ModuleOp module,
      llvm::StringRef functionName,
      mlir::modelica::ArrayType arrayType) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Create the function inside the parent module
    builder.setInsertionPointToEnd(module.getBody());

    mlir::Type int64PtrType = mlir::LLVM::LLVMPointerType::get(builder.getI64Type());
    mlir::Type convertedArrayType = typeConverter->convertType(arrayType);

    auto functionType = builder.getFunctionType({ convertedArrayType, int64PtrType }, builder.getF64Type());
    auto function = builder.create<mlir::func::FuncOp>(loc, functionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Extract the indices
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

    builder.create<mlir::func::ReturnOp>(loc, result);
    return function;
  }

  mlir::LogicalResult ModelConverter::createGetDerivativeFunction(
      mlir::OpBuilder& builder, ModelOp modelOp, const DerivativesMap& derivativesMap) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = mlir::LLVM::LLVMFunctionType::get(builder.getI64Type(), builder.getI64Type());
    auto function = builder.create<mlir::LLVM::LLVMFuncOp>(loc, getDerivativeFunctionName, functionType);

    // Create the entry block
    auto* entryBlock = function.addEntryBlock();

    // Create the last block receiving the value to be returned
    mlir::Block* returnBlock = builder.createBlock(&function.getBody(), function.getBody().end(), builder.getI64Type(), loc);

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
    mlir::Value defaultOperand = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(-1));

    builder.create<mlir::cf::SwitchOp>(
        loc,
        entryBlock->getArgument(0), returnBlock, defaultOperand,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);

    builder.setInsertionPointToStart(entryBlock);

    // Populate the case blocks
    for (size_t i = 0; i < numCases; ++i) {
      builder.setInsertionPointToStart(caseBlocks[i]);
      int64_t derivative = -1;

      if (derivativesMap.hasDerivative(i)) {
        derivative = derivativesMap.getDerivative(i);
      }

      mlir::Value result = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64IntegerAttr(derivative));
      builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
    }

    // Populate the return block
    builder.setInsertionPointToStart(returnBlock);
    builder.create<mlir::LLVM::ReturnOp>(loc, returnBlock->getArgument(0));

    return mlir::success();
  }

  mlir::LogicalResult ModelConverter::createGetCurrentTimeFunction(
      mlir::OpBuilder& builder, ModelOp modelOp) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Location loc = modelOp.getLoc();

    // Create the function inside the parent module
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    auto functionType = builder.getFunctionType(getVoidPtrType(), builder.getF64Type());
    auto function = builder.create<mlir::func::FuncOp>(loc, getCurrentTimeFunctionName, functionType);

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Load the runtime data structure
    auto runtimeDataStructType = getRuntimeDataStructType(
        builder.getContext(), modelOp.getBodyRegion().getArgumentTypes());

    mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), runtimeDataStructType);

    // Extract the time variable
    mlir::Value time = extractValue(builder, structValue, RealType::get(builder.getContext()), timeVariablePosition);

    time = typeConverter->materializeTargetConversion(builder, loc, typeConverter->convertType(time.getType()), time);

    if (auto timeBitWidth = time.getType().getIntOrFloatBitWidth(); timeBitWidth < 64) {
      time = builder.create<mlir::LLVM::FPExtOp>(loc, builder.getF64Type(), time);
    } else if (timeBitWidth > 64) {
      time = builder.create<mlir::LLVM::FPTruncOp>(loc, builder.getF64Type(), time);
    }

    builder.create<mlir::func::ReturnOp>(loc, time);
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

    auto functionType = builder.getFunctionType(argsTypes, resultType);
    auto mainFunction = builder.create<mlir::func::FuncOp>(loc, mainFunctionName, functionType);

    auto* entryBlock = mainFunction.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Call the function to start the simulation.
    // Its definition lives within the runtime library.

    auto runFunction = getOrCreateLLVMFunctionDecl(
        builder, module, runFunctionName, mlir::LLVM::LLVMFunctionType::get(resultType, argsTypes));

    mlir::Value returnValue = builder.create<mlir::LLVM::CallOp>(loc, runFunction, mainFunction.getArguments()).getResult();

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

    mlir::Value var = builder.create<mlir::LLVM::ExtractValueOp>(loc, structTypes[position], structValue, position);
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
      externalSolversData = builder.create<mlir::LLVM::InsertValueOp>(loc, externalSolversData, externalSolverDataPtr, solver.index());
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

    auto functionType = builder.getFunctionType(getVoidPtrType(), llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(loc, functionName, functionType);

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
        externalSolversPosition);

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

    auto functionType = builder.getFunctionType(getVoidPtrType(), llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(loc, functionName, functionType);

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
      builder.create<mlir::LLVM::CallOp>(solverDataPtr.getLoc(), freeFunc, solverDataPtr);
    }

    // Deallocate the structure containing all the solvers
    externalSolversPtr = builder.create<mlir::LLVM::BitcastOp>(externalSolversPtr.getLoc(), getVoidPtrType(), externalSolversPtr);
    builder.create<mlir::LLVM::CallOp>(externalSolversPtr.getLoc(), freeFunc, externalSolversPtr);

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

    auto functionType = builder.getFunctionType(getVoidPtrType(), llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(loc, calcICFunctionName, functionType);

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
    mlir::Value startTimeValue = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), startTime));
    startTimeValue = typeConverter->materializeTargetConversion(builder, loc, runtimeDataStructType.getBody()[timeVariablePosition], startTimeValue);
    runtimeDataStructValue = builder.create<mlir::LLVM::InsertValueOp>(loc, runtimeDataStructValue, startTimeValue, 1);

    // Add the model variables
    for (const auto& var : llvm::enumerate(structVariables)) {
      mlir::Type convertedType = typeConverter->convertType(var.value().getType());
      mlir::Value convertedVar = typeConverter->materializeTargetConversion(builder, loc, convertedType, var.value());
      runtimeDataStructValue = builder.create<mlir::LLVM::InsertValueOp>(loc, runtimeDataStructValue, convertedVar, var.index() + 2);
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

    auto functionType = builder.getFunctionType(getVoidPtrType(), llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(loc, deinitFunctionName, functionType);

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

    auto functionType = builder.getFunctionType(getVoidPtrType(), llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(loc, updateNonStateVariablesFunctionName, functionType);

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

    auto functionType = builder.getFunctionType(getVoidPtrType(), llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(loc, updateStateVariablesFunctionName, functionType);

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

    if (solver.getKind() == Solver::Kind::forwardEuler) {
      // Update the state variables by applying the forward Euler method
      builder.setInsertionPoint(returnOp);
      mlir::Value timeStepValue = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), timeStep));

      auto apply = [&](mlir::OpBuilder& nestedBuilder, mlir::Value scalarState, mlir::Value scalarDerivative) -> mlir::Value {
        mlir::Value result = builder.create<MulOp>(scalarDerivative.getLoc(), scalarDerivative.getType(), scalarDerivative, timeStepValue);
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

    auto functionType = builder.getFunctionType(getVoidPtrType(), builder.getI1Type());
    auto function = builder.create<mlir::func::FuncOp>(loc, incrementTimeFunctionName, functionType);

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
      mlir::Value timeStepValue = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), timeStep));
      mlir::Value currentTime = extractValue(builder, runtimeDataStruct, RealType::get(builder.getContext()), timeVariablePosition);
      increasedTime = builder.create<AddOp>(loc, currentTime.getType(), currentTime, timeStepValue);
    }

    // Store the increased time into the runtime data structure
    increasedTime = typeConverter->materializeTargetConversion(builder, loc, runtimeDataStructType.getBody()[timeVariablePosition], increasedTime);
    runtimeDataStruct = builder.create<mlir::LLVM::InsertValueOp>(loc, runtimeDataStruct, increasedTime, timeVariablePosition);

    mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(runtimeDataStruct.getType());
    mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(loc, structPtrType, function.getArgument(0));
    builder.create<mlir::LLVM::StoreOp>(loc, runtimeDataStruct, structPtr);

    // Check if the current time is less than the end time
    mlir::Value endTimeValue = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), endTime));
    mlir::Value epsilon = builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 10e-06));
    endTimeValue = builder.create<SubOp>(loc, endTimeValue.getType(), endTimeValue, epsilon);

    endTimeValue = typeConverter->materializeTargetConversion(builder, loc, typeConverter->convertType(endTimeValue.getType()), endTimeValue);

    mlir::Value condition = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, increasedTime, endTimeValue);
    builder.create<mlir::func::ReturnOp>(loc, condition);

    return mlir::success();
  }

  mlir::Value ModelConverter::getOrCreateGlobalString(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ModuleOp module,
      mlir::StringRef name,
      mlir::StringRef value) const
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
}

namespace
{
  class ModelConversionPass : public mlir::modelica::impl::ModelConversionPassBase<ModelConversionPass>
  {
    public:
      using ModelConversionPassBase::ModelConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(createSimulationHooks())) {
          mlir::emitError(getOperation().getLoc(), "Can't create the simulation hooks");
          return signalPassFailure();
        }

        if (mlir::failed(convertFuncOps())) {
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult createSimulationHooks();

      mlir::LogicalResult convertFuncOps();
  };
}

mlir::LogicalResult ModelConversionPass::createSimulationHooks()
{
  mlir::ModuleOp module = getOperation();
  std::vector<ModelOp> modelOps;

  module.walk([&](ModelOp modelOp) {
    modelOps.push_back(modelOp);
  });

  assert(llvm::count_if(modelOps, [&](ModelOp modelOp) {
           return modelOp.getSymName() == model;
         }) <= 1 && "More than one model matches the requested model name, but only one can be converted into a simulation");

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  // Modelica types converter
  mlir::modelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

  // Add the conversions for the IDA types
  mlir::ida::LLVMTypeConverter idaTypeConverter(&getContext(), llvmLoweringOptions);

  typeConverter.addConversion([&](mlir::ida::InstanceType type) {
    return idaTypeConverter.convertType(type);
  });

  typeConverter.addConversion([&](mlir::ida::VariableType type) {
    return idaTypeConverter.convertType(type);
  });

  typeConverter.addConversion([&](mlir::ida::EquationType type) {
    return idaTypeConverter.convertType(type);
  });

  for (auto& modelOp : modelOps) {
    if (modelOp.getSymName() != model) {
      modelOp.erase();
      continue;
    }

    auto expectedVariablesFilter = VariableFilter::fromString(variablesFilter);
    std::unique_ptr<VariableFilter> variablesFilterInstance;

    if (!expectedVariablesFilter) {
      modelOp.emitWarning("Invalid variable filter string. No filtering will take place");
      variablesFilterInstance = std::make_unique<VariableFilter>();
    } else {
      variablesFilterInstance = std::make_unique<VariableFilter>(std::move(*expectedVariablesFilter));
    }

    IDAOptions idaOptions;
    idaOptions.equidistantTimeGrid = idaEquidistantTimeGrid;

    ModelConverter modelConverter(typeConverter, *variablesFilterInstance, solver, startTime, endTime, timeStep, idaOptions);

    mlir::OpBuilder builder(modelOp);

    // Parse the derivatives map.
    DerivativesMap derivativesMap;

    if (auto res = readDerivativesMap(modelOp, derivativesMap); mlir::failed(res)) {
      return res;
    }

    if (processICModel) {
      // Obtain the scheduled model.
      Model<ScheduledEquationsBlock> model(modelOp);
      model.setVariables(discoverVariables(modelOp));
      model.setDerivativesMap(derivativesMap);

      auto equationsFilter = [](EquationInterface op) {
        return mlir::isa<InitialEquationOp>(op);
      };

      if (auto res = readSchedulingAttributes(model, equationsFilter); mlir::failed(res)) {
        return res;
      }

      // Create the simulation functions.
      if (auto res = modelConverter.convertInitialModel(builder, model); mlir::failed(res)) {
        return res;
      }
    }

    if (processMainModel) {
      // Obtain the scheduled model.
      Model<ScheduledEquationsBlock> model(modelOp);
      model.setVariables(discoverVariables(modelOp));
      model.setDerivativesMap(derivativesMap);

      auto equationsFilter = [](EquationInterface op) {
        return mlir::isa<EquationOp>(op);
      };

      if (auto res = readSchedulingAttributes(model, equationsFilter); mlir::failed(res)) {
        return res;
      }

      // Create the simulation functions.
      if (auto res = modelConverter.convertMainModel(builder, model); mlir::failed(res)) {
        return res;
      }
    }

    if (auto res = modelConverter.createGetModelNameFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getModelNameFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetNumOfVariablesFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getNumOfVariablesFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetVariableNameFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getVariableNameFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetVariableRankFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getVariableRankFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetVariableNumOfPrintableRangesFunction(builder, modelOp, derivativesMap); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getVariableNumOfPrintableRangesFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetVariablePrintableRangeBeginFunction(builder, modelOp, derivativesMap); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getVariablePrintableRangeBeginFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetVariablePrintableRangeEndFunction(builder, modelOp, derivativesMap); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getVariablePrintableRangeEndFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetVariableValueFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getVariableValueFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetDerivativeFunction(builder, modelOp, derivativesMap); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getDerivativeFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createGetCurrentTimeFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::getCurrentTimeFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createInitFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::initFunctionName + "' function");
      return res;
    }

    if (auto res = modelConverter.createDeinitFunction(builder, modelOp); mlir::failed(res)) {
      modelOp.emitError("Could not create the '" + ModelConverter::deinitFunctionName + "' function");
      return res;
    }

    if (emitSimulationMainFunction) {
      if (auto res = modelConverter.createMainFunction(builder, modelOp); mlir::failed(res)) {
        modelOp.emitError("Could not create the '" + ModelConverter::mainFunctionName + "' function");
        return res;
      }
    }

    // Erase the model operation, which has been converted to algorithmic code
    modelOp.erase();
  }

  return mlir::success();
}

mlir::LogicalResult ModelConversionPass::convertFuncOps()
{
  mlir::ConversionTarget target(getContext());

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  mlir::modelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<mlir::func::CallOp>([&](mlir::func::CallOp op) {
    for (const auto& type : op.operands().getTypes()) {
      if (!typeConverter.isLegal(type)) {
        return false;
      }
    }

    for (const auto& type : op.getResults().getTypes()) {
      if (!typeConverter.isLegal(type)) {
        return false;
      }
    }

    return true;
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<FuncOpTypesPattern>(typeConverter, &getContext());
  patterns.insert<CallOpTypesPattern>(typeConverter, &getContext());

  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createModelConversionPass()
  {
    return std::make_unique<ModelConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelConversionPass(const ModelConversionPassOptions& options)
  {
    return std::make_unique<ModelConversionPass>(options);
  }
}