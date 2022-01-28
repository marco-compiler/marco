#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/codegen/dialects/modelica/ModelicaBuilder.h"
#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/SolveModel.h"
#include "marco/codegen/passes/TypeConverter.h"
#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/Matching.h"
#include "marco/codegen/passes/model/Model.h"
#include "marco/modeling/Matching.h"
#include "marco/utils/VariableFilter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>
#include <memory>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

// TODO factor out from here and AD pass (and maybe also from somewhere else)
template <class T>
unsigned int numDigits(T number)
{
  unsigned int digits = 0;

  while (number != 0)
  {
    number /= 10;
    ++digits;
  }

  return digits;
}

// TODO factor out from here and AD pass
static std::string getFullDerVariableName(llvm::StringRef baseName, unsigned int order)
{
  assert(order > 0);

  if (order == 1)
    return "der_" + baseName.str();

  return "der_" + std::to_string(order) + "_" + baseName.str();
}

// TODO factor out from here and AD pass
static std::string getNextFullDerVariableName(llvm::StringRef currentName, unsigned int requestedOrder)
{
  if (requestedOrder == 1)
    return getFullDerVariableName(currentName, requestedOrder);

  assert(currentName.rfind("der_") == 0);

  if (requestedOrder == 2)
    return getFullDerVariableName(currentName.substr(4), requestedOrder);

  return getFullDerVariableName(currentName.substr(5 + numDigits(requestedOrder - 1)), requestedOrder);
}


struct ModelOpPattern : public mlir::ConvertOpToLLVMPattern<ModelOp>
{
	private:
	// The derivatives map keeps track of whether a variable is the derivative
	// of another one. Each variable is identified by its position within the
	// list of the "body" region arguments.

	using DerivativesPositionsMap = std::map<size_t, size_t>;

	// Name for the functions of the simulation
	static constexpr llvm::StringLiteral mainFunctionName = "main";
	static constexpr llvm::StringLiteral initFunctionName = "init";
	static constexpr llvm::StringLiteral stepFunctionName = "step";
	static constexpr llvm::StringLiteral printHeaderFunctionName = "printHeader";
	static constexpr llvm::StringLiteral printFunctionName = "print";
	static constexpr llvm::StringLiteral deinitFunctionName = "deinit";
  static constexpr llvm::StringLiteral runtimeInitFunctionName = "runtimeInit";
  static constexpr llvm::StringLiteral runtimeDeinitFunctionName = "runtimeDeinit";

	public:
  ModelOpPattern(mlir::MLIRContext* ctx,
                 TypeConverter& typeConverter,
                 SolveModelOptions options,
                 mlir::BlockAndValueMapping& derivativesMap)
			: mlir::ConvertOpToLLVMPattern<ModelOp>(typeConverter),
				options(std::move(options)),
				derivativesMap(&derivativesMap)
	{
	}

	mlir::LogicalResult matchAndRewrite(ModelOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		// Convert the original derivatives map between values into a map
		// between positions.
		DerivativesPositionsMap derivativesPositions;

		for (size_t i = 0, e = op.body().getNumArguments(); i < e; ++i)
		{
			mlir::Value var = op.body().getArgument(i);

			if (derivativesMap->contains(var))
			{
				mlir::Value derivative = derivativesMap->lookup(var);
				bool derivativeFound = false;
				unsigned int position = 0;

				for (size_t j = 0; j < e && !derivativeFound; ++j)
				{
					mlir::Value arg = op.body().getArgument(j);

					if (arg == derivative)
					{
						derivativeFound = true;
						position = j;
					}
				}

				assert(derivativeFound && "Derivative not found among arguments");
				derivativesPositions[i] = position;
			}
		}

    mlir::TypeRange varTypes = op.body().getArgumentTypes();

		if (auto status = createInitFunction(rewriter, op, varTypes); failed(status))
			return status;

		if (auto status = createDeinitFunction(rewriter, op, varTypes); failed(status))
			return status;

		if (auto status = createStepFunction(rewriter, op, varTypes); failed(status))
			return status;

    /*
		if (auto status = createPrintHeaderFunction(rewriter, op, varTypes, derivativesPositions); failed(status))
			return status;

		if (auto status = createPrintFunction(rewriter, op, varTypes, derivativesPositions); failed(status))
			return status;

		if (options.emitMain)
			if (auto status = createMainFunction(rewriter, op); failed(status))
				return status;
     */

		rewriter.eraseOp(op);
		return mlir::success();
	}

	private:
	/**
	 * Load the data structure from the opaque pointer that is passed around the
	 * simulation functions.
	 *
	 * @param builder		operation builder
	 * @param ptr 			opaque pointer
	 * @param varTypes 	types of the variables
	 * @return data structure containing the variables
	 */
	mlir::Value loadDataFromOpaquePtr(mlir::OpBuilder& builder, mlir::Value ptr, mlir::TypeRange varTypes) const
	{
		mlir::Location loc = ptr.getLoc();
		llvm::SmallVector<mlir::Type, 3> structTypes;

		for (const auto& type : varTypes)
			structTypes.push_back(this->getTypeConverter()->convertType(type));

		mlir::Type structType = mlir::LLVM::LLVMStructType::getLiteral(ptr.getContext(), structTypes);
		mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(structType);
		mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(loc, structPtrType, ptr);
		mlir::Value structValue = builder.create<mlir::LLVM::LoadOp>(loc, structPtr);

		return structValue;
	}

	/**
	 * Extract a value from the data structure shared between the various
	 * simulation main functions.
	 *
	 * @param builder 			operation builder
	 * @param structValue 	data structure
	 * @param type 					value type
	 * @param position 			value position
	 * @return extracted value
	 */
	mlir::Value extractValue(mlir::OpBuilder& builder, mlir::Value structValue, mlir::Type type, unsigned int position) const
	{
		mlir::Location loc = structValue.getLoc();

		assert(structValue.getType().isa<mlir::LLVM::LLVMStructType>() && "Not an LLVM struct");
		auto structType = structValue.getType().cast<mlir::LLVM::LLVMStructType>();
		auto structTypes = structType.getBody();
		assert (position < structTypes.size() && "LLVM struct: index is out of bounds");

		mlir::Value var = builder.create<mlir::LLVM::ExtractValueOp>(loc, structTypes[position], structValue, builder.getIndexArrayAttr(position));
		return this->getTypeConverter()->materializeSourceConversion(builder, loc, type, var);
	}

  mlir::Value convertMember(mlir::ConversionPatternRewriter& rewriter, MemberCreateOp op) const
  {
    using LoadReplacer = std::function<void(MemberLoadOp)>;
    using StoreReplacer = std::function<void(MemberStoreOp)>;

    mlir::Location loc = op->getLoc();

    auto memberType = op.resultType().cast<MemberType>();
    auto arrayType = memberType.toArrayType();
    assert(arrayType.getAllocationScope() == BufferAllocationScope::heap);

    mlir::Value reference = rewriter.create<AllocOp>(
            loc, arrayType.getElementType(), arrayType.getShape(), op.dynamicDimensions(), false);

    auto replacers = [&]() {
        if (arrayType.isScalar())
        {
          assert(op.dynamicDimensions().empty());

          return std::make_pair<LoadReplacer, StoreReplacer>(
                  [&rewriter, reference](MemberLoadOp loadOp) -> void {
                      mlir::OpBuilder::InsertionGuard guard(rewriter);
                      rewriter.setInsertionPoint(loadOp);
                      rewriter.replaceOpWithNewOp<LoadOp>(loadOp, reference);
                  },
                  [&rewriter, reference](MemberStoreOp storeOp) -> void {
                      mlir::OpBuilder::InsertionGuard guard(rewriter);
                      rewriter.setInsertionPoint(storeOp);
                      rewriter.replaceOpWithNewOp<AssignmentOp>(storeOp, storeOp.value(), reference);
                  });
        }

        return std::make_pair<LoadReplacer, StoreReplacer>(
                [&rewriter, reference](MemberLoadOp loadOp) -> void {
                    mlir::OpBuilder::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPoint(loadOp);
                    rewriter.replaceOp(loadOp, reference);
                },
                [&rewriter, reference](MemberStoreOp storeOp) -> void {
                    mlir::OpBuilder::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPoint(storeOp);
                    rewriter.replaceOpWithNewOp<AssignmentOp>(storeOp, storeOp.value(), reference);
                });
    };

    LoadReplacer loadReplacer;
    StoreReplacer storeReplacer;
    std::tie(loadReplacer, storeReplacer) = replacers();

    for (auto* user : op->getUsers())
    {
      if (auto loadOp = mlir::dyn_cast<MemberLoadOp>(user))
        loadReplacer(loadOp);
      else if (auto storeOp = mlir::dyn_cast<MemberStoreOp>(user))
        storeReplacer(storeOp);
    }

    rewriter.replaceOp(op, reference);
    return reference;
  }

	/**
	 * Create the initialization function that allocates the variables and
	 * stores them into an appropriate data structure to be passed to the other
	 * simulation functions.
	 *
	 * @param rewriter 	operation rewriter
	 * @param op 				simulation op
	 * @param varTypes 	types of the variables
	 * @return conversion result status
	 */
	mlir::LogicalResult createInitFunction(mlir::ConversionPatternRewriter& rewriter, ModelOp op, mlir::TypeRange varTypes) const
	{
		mlir::Location loc = op->getLoc();
		mlir::OpBuilder::InsertionGuard guard(rewriter);

		// Create the function inside the parent module
		rewriter.setInsertionPointToEnd(op->getParentOfType<mlir::ModuleOp>().getBody());

		auto functionType = rewriter.getFunctionType(llvm::None, getVoidPtrType());
		auto function = rewriter.create<mlir::FuncOp>(loc, initFunctionName, functionType);

		auto* entryBlock = function.addEntryBlock();
		rewriter.setInsertionPointToStart(entryBlock);

		// Move the initialization instructions into the new function
		rewriter.mergeBlocks(&op.init().front(), &function.body().front(), llvm::None);

		llvm::SmallVector<mlir::Value, 3> values;
		auto terminator = mlir::cast<YieldOp>(entryBlock->getTerminator());

		auto removeAllocationScopeFn = [&](mlir::Value value) -> mlir::Value {
			return rewriter.create<ArrayCastOp>(
					loc, value,
					value.getType().cast<ArrayType>().toUnknownAllocationScope());
		};

		// Add variables to the struct to be passed around (i.e. to the step and
		// print functions).

		for (const auto& var : terminator.values())
    {
      auto memberCreateOp = var.getDefiningOp<MemberCreateOp>();
      mlir::Value array = convertMember(rewriter, memberCreateOp);
      values.push_back(removeAllocationScopeFn(array));
    }

		// Set the start time
		mlir::Value startTime = rewriter.create<ConstantOp>(loc, op.startTime());
		rewriter.create<StoreOp>(loc, startTime, values[0]);

		// Pack the values
		llvm::SmallVector<mlir::Type, 3> structTypes;

		for (const auto& type : varTypes)
			structTypes.push_back(this->getTypeConverter()->convertType(type));

		auto structType = mlir::LLVM::LLVMStructType::getLiteral(op->getContext(), structTypes);
		mlir::Value structValue = rewriter.create<mlir::LLVM::UndefOp>(loc, structType);

		for (const auto& var : llvm::enumerate(values))
		{
			mlir::Type convertedType = this->getTypeConverter()->convertType(var.value().getType());
			mlir::Value convertedVar = this->getTypeConverter()->materializeTargetConversion(rewriter, loc, convertedType, var.value());
			structValue = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structValue, convertedVar, rewriter.getIndexArrayAttr(var.index()));
		}

		// The data structure must be stored on the heap in order to escape
		// from the function.

		// Add the "malloc" function to the module
		auto mallocFunc = mlir::LLVM::lookupOrCreateMallocFn(op->getParentOfType<mlir::ModuleOp>(), getIndexType());

		// Determine the size (in bytes) of the memory to be allocated
		mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(structType);
		mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, structPtrType);

		mlir::Value one = rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(1));
		one = this->getTypeConverter()->materializeTargetConversion(rewriter, loc, getIndexType(), one);

		mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, structPtrType, llvm::ArrayRef<mlir::Value>{nullPtr, one});
		mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);
		mlir::Value resultOpaquePtr = createLLVMCall(rewriter, loc, mallocFunc, sizeBytes, getVoidPtrType())[0];

		// Store the struct into the heap memory
		mlir::Value resultCastedPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, structPtrType, resultOpaquePtr);
		rewriter.create<mlir::LLVM::StoreOp>(loc, structValue, resultCastedPtr);

		rewriter.replaceOpWithNewOp<mlir::ReturnOp>(terminator, resultOpaquePtr);
		return mlir::success();
	}

	/**
	 * Create a function to be called when the simulation has finished and the
	 * variables together with its data structure are not required anymore and
	 * thus can be deallocated.
	 *
	 * @param builder		operation builder
	 * @param op 				simulation op
	 * @param varTypes 	types of the variables
	 * @return conversion result status
	 */
	mlir::LogicalResult createDeinitFunction(mlir::OpBuilder& builder, ModelOp op, mlir::TypeRange varTypes) const
	{
		mlir::Location loc = op.getLoc();
		mlir::OpBuilder::InsertionGuard guard(builder);

		// Create the function inside the parent module
		builder.setInsertionPointToEnd(op->getParentOfType<mlir::ModuleOp>().getBody());

		auto function = builder.create<mlir::FuncOp>(
				loc, deinitFunctionName,
				builder.getFunctionType(getVoidPtrType(), llvm::None));

		auto* entryBlock = function.addEntryBlock();
		builder.setInsertionPointToStart(entryBlock);

		// Extract the data from the struct
		mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), varTypes);

		// Deallocate the arrays
		for (const auto& type : llvm::enumerate(varTypes))
		{
			if (auto arrayType = type.value().dyn_cast<ArrayType>())
			{
				mlir::Value var = extractValue(builder, structValue, varTypes[type.index()], type.index());
				var = builder.create<ArrayCastOp>(loc, var, arrayType.toAllocationScope(BufferAllocationScope::heap));
				builder.create<FreeOp>(loc, var);
			}
		}

		// Add "free" function to the module
		auto freeFunc = mlir::LLVM::lookupOrCreateFreeFn(op->getParentOfType<mlir::ModuleOp>());

		// Deallocate the data structure
		builder.create<mlir::LLVM::CallOp>(loc, llvm::None, builder.getSymbolRefAttr(freeFunc), function.getArgument(0));

		builder.create<mlir::ReturnOp>(loc);
		return mlir::success();
	}

	/**
	 * Create the function to be called at each time step.
	 *
	 * @param rewriter		operation rewriter
	 * @param op					simulation op
	 * @param varTypes		types of the variables
	 * @return conversion result status
	 */
	mlir::LogicalResult createStepFunction(mlir::ConversionPatternRewriter& rewriter, ModelOp op, mlir::TypeRange varTypes) const
	{
		mlir::Location loc = op.getLoc();
		mlir::OpBuilder::InsertionGuard guard(rewriter);

		// Create the function inside the parent module
		rewriter.setInsertionPointToEnd(op->getParentOfType<mlir::ModuleOp>().getBody());

		auto function = rewriter.create<mlir::FuncOp>(
				loc, stepFunctionName,
				rewriter.getFunctionType(getVoidPtrType(), rewriter.getI1Type()));

		auto* entryBlock = function.addEntryBlock();
		rewriter.setInsertionPointToStart(entryBlock);

		// Extract the data from the struct
		mlir::Value structValue = loadDataFromOpaquePtr(rewriter, function.getArgument(0), varTypes);

		llvm::SmallVector<mlir::Value, 3> vars;

		for (const auto& varType : llvm::enumerate(varTypes))
			vars.push_back(extractValue(rewriter, structValue, varTypes[varType.index()], varType.index()));

		// Increment the time
		mlir::Value timeStep = rewriter.create<ConstantOp>(loc, op.timeStep());
		mlir::Value currentTime = rewriter.create<LoadOp>(loc, vars[0]);
		mlir::Value increasedTime = rewriter.create<AddOp>(loc, currentTime.getType(), currentTime, timeStep);
		rewriter.create<StoreOp>(loc, increasedTime, vars[0]);

		// Check if the current time is less than the end time
		mlir::Value endTime = rewriter.create<ConstantOp>(loc, op.endTime());
		//endTime = rewriter.create<AddOp>(loc, endTime.getType(), endTime, timeStep);

		mlir::Value condition = rewriter.create<LteOp>(loc, BooleanType::get(op->getContext()), increasedTime, endTime);
		condition = getTypeConverter()->materializeTargetConversion(rewriter, condition.getLoc(), rewriter.getI1Type(), condition);

		auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, rewriter.getI1Type(), condition, true);
		rewriter.create<mlir::ReturnOp>(loc, ifOp.getResult(0));

		// If we didn't reach the end time update the variables and return
		// true to continue the simulation.
		rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());

		auto trueValue = rewriter.create<mlir::ConstantOp>(
				loc, rewriter.getBoolAttr(true));

		rewriter.create<mlir::scf::YieldOp>(loc, trueValue.getResult());

    // Copy the body of the simulation
    mlir::BlockAndValueMapping mapping;

    for (const auto& [oldVar, newVar] : llvm::zip(op.body().getArguments(), vars))
      mapping.map(oldVar, newVar);

    // We need to keep track of the amount of equations processed, so that
    // we can give a unique name to each function generated.
    size_t equationCounter = 0;

    rewriter.setInsertionPoint(trueValue);

    for (auto& bodyOp : op.body().front().without_terminator())
    {
      if (!mlir::isa<EquationOp>(bodyOp) && !mlir::isa<ForEquationOp>(bodyOp))
        rewriter.clone(bodyOp, mapping);
    }

    op.body().walk([&](EquationOp equationOp) {
      auto insertionPoint = rewriter.saveInsertionPoint();

      llvm::SmallVector<ForEquationOp, 3> loops;
      ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();

      while (parent != nullptr)
      {
        loops.push_back(parent);
        parent = parent->getParentOfType<ForEquationOp>();
      }

      llvm::SmallVector<mlir::Value, 3> originalInductions;
      llvm::SmallVector<mlir::Value, 3> inductions;

      // Create the loops
      for (auto it = loops.rbegin(); it != loops.rend(); ++it)
      {
        mlir::Value lowerBound = rewriter.create<ConstantOp>(it->getLoc(), rewriter.getIndexAttr(it->start()));
        mlir::Value upperBound = rewriter.create<ConstantOp>(it->getLoc(), rewriter.getIndexAttr(it->end()));
        mlir::Value step = rewriter.create<ConstantOp>(it->getLoc(), rewriter.getIndexAttr(1));

        auto forOp = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
        originalInductions.push_back(it->induction());
        inductions.push_back(forOp.getInductionVar());
        rewriter.setInsertionPointToStart(forOp.getBody());
      }

      llvm::SmallVector<mlir::Value, 3> usedVars;
      mlir::FuncOp equationFunction = convertEquation(rewriter, equationOp, equationCounter, vars, originalInductions, usedVars);
      usedVars.append(inductions);
      auto call = rewriter.create<mlir::CallOp>(equationOp->getLoc(), equationFunction, usedVars);
      ++equationCounter;

      rewriter.restoreInsertionPoint(insertionPoint);
    });

		// Otherwise, return false to stop the simulation
		mlir::OpBuilder::InsertionGuard g(rewriter);
		rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
		mlir::Value falseValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getBoolAttr(false));
		rewriter.create<mlir::scf::YieldOp>(loc, falseValue);

		return mlir::success();
	}

  mlir::FuncOp convertEquation(
          mlir::ConversionPatternRewriter& rewriter,
          EquationOp eq, size_t counter,
          mlir::ValueRange vars,
          mlir::ValueRange originalInductions,
          llvm::SmallVectorImpl<mlir::Value>& usedVars) const
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Location loc = eq->getLoc();

    auto module = eq->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(module.getBody());

    auto varsTypes = vars.getTypes();
    llvm::SmallVector<mlir::Type, 3> argsTypes(varsTypes.begin(), varsTypes.end());
    argsTypes.append(originalInductions.size(), rewriter.getIndexType());

    auto functionType = rewriter.getFunctionType(argsTypes, llvm::None);
    auto name = "eq" + std::to_string(counter);

    auto function = rewriter.create<mlir::FuncOp>(loc, name, functionType);

    auto* entryBlock = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    // Before moving the equation body, we need to map the old variables of
    // the ModelOp to the new ones living in the dedicated function.
    // The same must be done with the induction variables.
    mlir::BlockAndValueMapping mapping;

    auto originalVars = eq->getParentOfType<ModelOp>().body().getArguments();

    for (size_t i = 0; i < originalVars.size(); ++i)
      mapping.map(originalVars[i], function.getArgument(i));

    for (size_t i = 0; i < originalInductions.size(); ++i)
      mapping.map(originalInductions[i], function.getArgument(originalVars.size() + i));

    for (auto& op : eq.body()->getOperations())
      rewriter.clone(op, mapping);

    // Create the return instruction
    rewriter.setInsertionPointToEnd(&function.body().back());
    rewriter.create<mlir::ReturnOp>(loc);

    // Remove the arguments that are not used
    llvm::SmallVector<unsigned int, 3> argsToBeRemoved;

    for (size_t i = 0; i < vars.size(); ++i)
    {
      mlir::Value arg = function.getArgument(i);

      if (arg.getUsers().empty())
        argsToBeRemoved.push_back(i);
      else
        usedVars.push_back(vars[i]);
    }

    function.eraseArguments(argsToBeRemoved);

    // Move loop invariants. This is needed because we scalarize the array
    // assignments, and thus arrays may be allocated way more times than
    // needed.

    function.walk([&](mlir::LoopLikeOpInterface loopLike) {
      moveLoopInvariantCode(loopLike);
    });

    return function;
  }

  mlir::LogicalResult moveLoopInvariantCode(mlir::LoopLikeOpInterface loopLike) const
  {
    auto& loopBody = loopLike.getLoopBody();

    // We use two collections here as we need to preserve the order for insertion
    // and this is easiest.
    llvm::SmallPtrSet<mlir::Operation*, 8> willBeMovedSet;
    llvm::SmallVector<mlir::Operation*, 8> opsToMove;

    // Helper to check whether an operation is loop invariant wrt. SSA properties.
    auto isDefinedOutsideOfBody = [&](mlir::Value value) {
        auto definingOp = value.getDefiningOp();

        return (definingOp && !!willBeMovedSet.count(definingOp)) ||
               loopLike.isDefinedOutsideOfLoop(value);
    };

    // Do not use walk here, as we do not want to go into nested regions and hoist
    // operations from there. These regions might have semantics unknown to this
    // rewriting. If the nested regions are loops, they will have been processed.

    for (auto& block : loopBody)
    {
      for (auto& op : block.without_terminator())
      {
        if (canBeHoisted(&op, isDefinedOutsideOfBody))
        {
          opsToMove.push_back(&op);
          willBeMovedSet.insert(&op);
        }
      }
    }

    return loopLike.moveOutOfLoop(opsToMove);
  }

  static bool canBeHoisted(mlir::Operation* op, std::function<bool(mlir::Value)> definedOutside)
  {
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;

    for (auto& region : op->getRegions())
    {
      for (auto& block : region)
      {
        for (auto& innerOp : block.without_terminator())
          if (!canBeHoisted(&innerOp, definedOutside))
            return false;
      }
    }

    return true;
  }

	void printSeparator(mlir::OpBuilder& builder, mlir::Value separator) const
	{
		auto module = separator.getParentRegion()->getParentOfType<mlir::ModuleOp>();
		auto printfRef = getOrInsertPrintf(builder, module);
		builder.create<mlir::LLVM::CallOp>(separator.getLoc(), printfRef, separator);
	}

	mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder& builder, mlir::StringRef name, mlir::StringRef value, mlir::ModuleOp module) const
	{
		// Create the global at the entry of the module
		mlir::LLVM::GlobalOp global;

		if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name)))
		{
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
				mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)),
				globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
	}

	mlir::Value getSeparatorString(mlir::Location loc, mlir::OpBuilder& builder, mlir::ModuleOp module) const
	{
		return getOrCreateGlobalString(loc, builder, "semicolon", mlir::StringRef(";\0", 2), module);
	}

	mlir::Value getNewlineString(mlir::Location loc, mlir::OpBuilder& builder, mlir::ModuleOp module) const
	{
		return getOrCreateGlobalString(loc, builder, "newline", mlir::StringRef("\n\0", 2), module);
	}

	mlir::LLVM::LLVMFuncOp getOrInsertFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::LLVM::LLVMFunctionType type) const
  {
    auto *context = module.getContext();

    if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
      return foo;

    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, type);
  }

	mlir::LLVM::LLVMFuncOp getOrInsertPrintf(mlir::OpBuilder& builder, mlir::ModuleOp module) const
	{
    auto *context = module.getContext();

		// Create a function declaration for printf, the signature is:
		//   * `i32 (i8*, ...)`
		auto llvmI32Ty = mlir::IntegerType::get(context, 32);
		auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
		auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

		// Insert the printf function into the body of the parent module
		return getOrInsertFunction(builder, module, "printf", llvmFnType);
	}

	void printVariableName(
			mlir::OpBuilder& builder,
			mlir::Value name,
			mlir::Type type,
			VariableFilter::Filter filter,
			std::function<mlir::Value()> structValue,
			unsigned int position,
			mlir::ModuleOp module,
			mlir::Value separator,
			bool shouldPreprendSeparator = true) const
	{
		if (auto arrayType = type.dyn_cast<ArrayType>())
		{
			if (arrayType.getRank() == 0)
				printScalarVariableName(builder, name, module, separator, shouldPreprendSeparator);
			else
				printArrayVariableName(builder, name, type, filter, structValue, position, module, separator, shouldPreprendSeparator);
		}
		else
		{
			printScalarVariableName(builder, name, module, separator, shouldPreprendSeparator);
		}
	}

	void printScalarVariableName(
			mlir::OpBuilder& builder,
			mlir::Value name,
			mlir::ModuleOp module,
			mlir::Value separator,
			bool shouldPrependSeparator) const
	{
		if (shouldPrependSeparator)
			printSeparator(builder, separator);

		mlir::Location loc = name.getLoc();
		mlir::Value formatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_str", mlir::StringRef("%s\0", 3), module);
		auto printfRef = getOrInsertPrintf(builder, module);
		builder.create<mlir::LLVM::CallOp>(loc, printfRef, mlir::ValueRange({ formatSpecifier, name }));
	}

	void printArrayVariableName(
			mlir::OpBuilder& builder,
			mlir::Value name,
			mlir::Type type,
			VariableFilter::Filter filter,
			std::function<mlir::Value()> structValue,
			unsigned int position,
			mlir::ModuleOp module,
			mlir::Value separator,
			bool shouldPrependSeparator) const
	{
		mlir::Location loc = name.getLoc();
		assert(type.isa<ArrayType>());

		// Get a reference to the printf function
		auto printfRef = getOrInsertPrintf(builder, module);

		// Create the brackets and comma strings
		mlir::Value lSquare = getOrCreateGlobalString(loc, builder, "lsquare", llvm::StringRef("[\0", 2), module);
		mlir::Value rSquare = getOrCreateGlobalString(loc, builder, "rsquare", llvm::StringRef("]\0", 2), module);
		mlir::Value comma = getOrCreateGlobalString(loc, builder, "comma", llvm::StringRef(",\0", 2), module);

		// Create the format strings
		mlir::Value stringFormatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_str", mlir::StringRef("%s\0", 3), module);
		mlir::Value integerFormatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_int", mlir::StringRef("%ld\0", 4), module);

		// Allow for the variable to lazily extracted if one of its dimension size
		// must be determined.
		bool valueLoaded = false;
		mlir::Value extractedValue = nullptr;
		auto insertionPoint = builder.saveInsertionPoint();

		auto var = [&]() -> mlir::Value {
			if (!valueLoaded)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.restoreInsertionPoint(insertionPoint);
				extractedValue = extractValue(builder, structValue(), type, position);
				valueLoaded = true;
			}

			return extractedValue;
		};

		// Create the lower and upper bounds
		auto ranges = filter.getRanges();
		auto arrayType = type.cast<ArrayType>();
		assert(arrayType.getRank() == ranges.size());

		llvm::SmallVector<mlir::Value, 3> lowerBounds;
		llvm::SmallVector<mlir::Value, 3> upperBounds;

		mlir::Value one = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));
		llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

		for (const auto& range : llvm::enumerate(ranges))
		{
			// In Modelica, arrays are 1-based. If present, we need to lower by 1
			// the value given by the variable filter.

			unsigned int lowerBound = range.value().hasLowerBound() ? range.value().getLowerBound() - 1 : 0;
			lowerBounds.push_back(builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(lowerBound)));

			// The upper bound is not lowered because the SCF's for operation assumes
			// them as excluded.

			if (range.value().hasUpperBound())
			{
				mlir::Value upperBound = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.value().getUpperBound()));
				upperBounds.push_back(upperBound);
			}
			else
			{
				mlir::Value dim = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.index()));
				mlir::Value upperBound = builder.create<DimOp>(loc, var(), dim);
				upperBounds.push_back(upperBound);
			}
		}

		bool shouldPrintSeparator = false;

		// Create nested loops in order to iterate on each dimension of the array
		mlir::scf::buildLoopNest(
				builder, loc, lowerBounds, upperBounds, steps,
				[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange indexes) {
					// Print the separator, the variable name and the left square bracket
					printSeparator(builder, separator);
					builder.create<mlir::LLVM::CallOp>(loc, printfRef, mlir::ValueRange({ stringFormatSpecifier, name }));
					builder.create<mlir::LLVM::CallOp>(loc, printfRef, lSquare);

					for (mlir::Value index : indexes)
					{
						if (shouldPrintSeparator)
							builder.create<mlir::LLVM::CallOp>(loc, printfRef, comma);

						shouldPrintSeparator = true;

						mlir::Type convertedType = this->getTypeConverter()->convertType(index.getType());
						index = this->getTypeConverter()->materializeTargetConversion(builder, loc, convertedType, index);

						// Arrays are 1-based in Modelica, so we add 1 in order to print
						// indexes that are coherent with the model source.
						mlir::Value increment = builder.create<mlir::ConstantOp>(loc, builder.getIntegerAttr(index.getType(), 1));
						index = builder.create<mlir::AddIOp>(loc, index.getType(), index, increment);

						builder.create<mlir::LLVM::CallOp>(loc, printfRef, mlir::ValueRange({ integerFormatSpecifier, index }));
					}

					// Print the right square bracket
					builder.create<mlir::LLVM::CallOp>(loc, printfRef, rSquare);
				});

	}

	mlir::LogicalResult createPrintHeaderFunction(
			mlir::OpBuilder& builder,
      ModelOp op,
			mlir::TypeRange varTypes,
			DerivativesPositionsMap& derivativesPositions) const
	{
		auto callback = [&](std::function<mlir::Value()> structValue, llvm::StringRef name, unsigned int position, VariableFilter::Filter filter, mlir::Value separator) -> mlir::LogicalResult {
			mlir::Location loc = op.getLoc();
			auto module = op->getParentOfType<mlir::ModuleOp>();

			std::string symbolName = "var" + std::to_string(position);
			llvm::SmallString<10> terminatedName(name);
			terminatedName.append("\0");
			mlir::Value symbol = getOrCreateGlobalString(loc, builder, symbolName, llvm::StringRef(terminatedName.c_str(), terminatedName.size() + 1), module);

			bool shouldPrintSeparator = position != 0;
			printVariableName(builder, symbol, varTypes[position], filter, structValue, position, module, separator, shouldPrintSeparator);
			return mlir::success();
		};

		return createPrintFunctionBody(builder, op, varTypes, derivativesPositions, printHeaderFunctionName, callback);
	}

	void printVariable(mlir::OpBuilder& builder, mlir::Value var, VariableFilter::Filter filter, mlir::Value separator, bool shouldPreprendSeparator = true) const
	{
		if (auto arrayType = var.getType().dyn_cast<ArrayType>())
		{
			if (arrayType.getRank() == 0)
			{
				mlir::Value value = builder.create<LoadOp>(var.getLoc(), var);
				printScalarVariable(builder, value, separator, shouldPreprendSeparator);
			}
			else
			{
				printArrayVariable(builder, var, filter, separator, shouldPreprendSeparator);
			}
		}
		else
		{
			printScalarVariable(builder, var, separator, shouldPreprendSeparator);
		}
	}

	void printScalarVariable(mlir::OpBuilder& builder, mlir::Value var, mlir::Value separator, bool shouldPreprendSeparator = true) const
	{
		if (shouldPreprendSeparator)
			printSeparator(builder, separator);

		printElement(builder, var);
	}

	void printArrayVariable(mlir::OpBuilder& builder, mlir::Value var, VariableFilter::Filter filter, mlir::Value separator, bool shouldPreprendSeparator = true) const
	{
		mlir::Location loc = var.getLoc();
		assert(var.getType().isa<ArrayType>());

		auto ranges = filter.getRanges();
		auto arrayType = var.getType().cast<ArrayType>();
		assert(arrayType.getRank() == ranges.size());

		llvm::SmallVector<mlir::Value, 3> lowerBounds;
		llvm::SmallVector<mlir::Value, 3> upperBounds;

		mlir::Value one = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));
		llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

		for (const auto& range : llvm::enumerate(ranges))
		{
			// In Modelica, arrays are 1-based. If present, we need to lower by 1
			// the value given by the variable filter.

			unsigned int lowerBound = range.value().hasLowerBound() ? range.value().getLowerBound() - 1 : 0;
			lowerBounds.push_back(builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(lowerBound)));

			// The upper bound is not lowered because the SCF's for operation assumes
			// them as excluded.

			if (range.value().hasUpperBound())
			{
				mlir::Value upperBound = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.value().getUpperBound()));
				upperBounds.push_back(upperBound);
			}
			else
			{
				mlir::Value dim = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.index()));
				mlir::Value upperBound = builder.create<DimOp>(loc, var, dim);
				upperBounds.push_back(upperBound);
			}
		}

		// Create nested loops in order to iterate on each dimension of the array
		mlir::scf::buildLoopNest(
				builder, loc, lowerBounds, upperBounds, steps,
				[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position) {
					mlir::Value value = nestedBuilder.create<LoadOp>(loc, var, position);

					printSeparator(nestedBuilder, separator);
					printElement(nestedBuilder, value);
				});
	}

	void printElement(mlir::OpBuilder& builder, mlir::Value value) const
	{
		mlir::Location loc = value.getLoc();
		auto module = value.getParentRegion()->getParentOfType<mlir::ModuleOp>();
		auto printfRef = getOrInsertPrintf(builder, module);

		mlir::Type convertedType = this->getTypeConverter()->convertType(value.getType());
		value = this->getTypeConverter()->materializeTargetConversion(builder, loc, convertedType, value);
		mlir::Type type = value.getType();

		mlir::Value formatSpecifier;

		if (type.isa<mlir::IntegerType>())
			formatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_int", mlir::StringRef("%ld\0", 4), module);
		else if (type.isa<mlir::FloatType>())
			formatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_float", mlir::StringRef("%.12f\0", 6), module);
		else
			assert(false && "Unknown type");

		builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, mlir::ValueRange({ formatSpecifier, value }));
	}

	mlir::LogicalResult createPrintFunction(
			mlir::OpBuilder& builder,
      ModelOp op,
			mlir::TypeRange varTypes,
			DerivativesPositionsMap& derivativesPositions) const
	{
		auto callback = [&](std::function<mlir::Value()> structValue, llvm::StringRef name, unsigned int position, VariableFilter::Filter filter, mlir::Value separator) -> mlir::LogicalResult {
			mlir::Value var = extractValue(builder, structValue(), varTypes[position], position);
			bool shouldPrintSeparator = position != 0;
			printVariable(builder, var, filter, separator, shouldPrintSeparator);
			return mlir::success();
		};

		return createPrintFunctionBody(builder, op, varTypes, derivativesPositions, printFunctionName, callback);
	}

	mlir::LogicalResult createPrintFunctionBody(
			mlir::OpBuilder& builder,
      ModelOp op,
			mlir::TypeRange varTypes,
			DerivativesPositionsMap& derivativesPositions,
			llvm::StringRef functionName,
			std::function<mlir::LogicalResult(std::function<mlir::Value()>, llvm::StringRef, unsigned int, VariableFilter::Filter, mlir::Value)> elementCallback) const
	{
		mlir::Location loc = op.getLoc();
		mlir::OpBuilder::InsertionGuard guard(builder);
		auto module = op->getParentOfType<mlir::ModuleOp>();

		// Create the function inside the parent module
		builder.setInsertionPointToEnd(module.getBody());

		auto function = builder.create<mlir::FuncOp>(
				loc, functionName,
				builder.getFunctionType(getVoidPtrType(), llvm::None));

		auto* entryBlock = function.addEntryBlock();
		builder.setInsertionPointToStart(entryBlock);

		// Create the separator and newline global strings
		mlir::Value separator = getSeparatorString(loc, builder, module);
		mlir::Value newline = getNewlineString(loc, builder, module);

		// Create the callback to load the data structure whenever needed
		bool structValueLoaded = false;
		mlir::Value structValue = nullptr;
		auto structValueInsertionPoint = builder.saveInsertionPoint();

		auto structValueCallback = [&]() -> mlir::Value {
			if (!structValueLoaded)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.restoreInsertionPoint(structValueInsertionPoint);
				structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), varTypes);
			}

			return structValue;
		};

		// Get the names of the variables
		llvm::SmallVector<llvm::StringRef, 8> variableNames =
				llvm::to_vector<8>(op.variableNames().getAsValueRange<mlir::StringAttr>());

		// Map each variable to its position inside the data structure.
		// It must be noted that the data structure also contains derivative (if
		// existent), so its size can be greater than the number of names.

		assert(op.variableNames().size() <= varTypes.size());
		llvm::StringMap<size_t> variablePositionByName;

		for (const auto& var : llvm::enumerate(variableNames))
			variablePositionByName[var.value()] = var.index() + 1; // + 1 to skip the "time" variable

		// The positions have been saved, so we can now sort the names
		llvm::sort(variableNames, [](llvm::StringRef x, llvm::StringRef y) -> bool {
			return x.compare_insensitive(y) < 0;
		});

		if (auto status = elementCallback(structValueCallback, "time", 0, VariableFilter::Filter::visibleScalar(), separator); failed(status))
			return status;

		// Print the other variables
		for (const auto& name : variableNames)
		{
			assert(variablePositionByName.count(name) != 0);
			size_t position = variablePositionByName[name];

			unsigned int rank = 0;

			if (auto arrayType = varTypes[position].dyn_cast<ArrayType>())
				rank = arrayType.getRank();

			auto filter = options.variableFilter->getVariableInfo(name, rank);

			if (!filter.isVisible())
				continue;

			if (auto status = elementCallback(structValueCallback, name, position, filter, separator); failed(status))
				return status;
		}

		// Print the derivatives
		for (const auto& name : variableNames)
		{
			size_t varPosition = variablePositionByName[name];

			if (derivativesPositions.count(varPosition) == 0)
			{
				// The variable has no derivative
				continue;
			}

			size_t derivedVarPosition = derivativesPositions[varPosition];

			unsigned int rank = 0;

			if (auto arrayType = varTypes[derivedVarPosition].dyn_cast<ArrayType>())
				rank = arrayType.getRank();

			auto filter = options.variableFilter->getVariableDerInfo(name, rank);

			if (!filter.isVisible())
				continue;

			llvm::SmallString<15> derName;
			derName.append("der(");
			derName.append(name);
			derName.append(")");

			if (auto status = elementCallback(structValueCallback, derName, derivedVarPosition, filter, separator); failed(status))
				return status;
		}

		// Print a newline character after all the variables have been processed
		builder.create<mlir::LLVM::CallOp>(loc, getOrInsertPrintf(builder, module), newline);

		builder.create<mlir::ReturnOp>(loc);
		return mlir::success();
	}

	/**
	 * Create the main function to be called to run the simulation.
	 * More precisely, the function first calls the "init" function, and then
	 * keeps running the updates until the step function return the stop
	 * condition (that is, a false value). After each step, it also prints the
	 * values and increments the time.
	 *
	 * @param builder 	operation builder
	 * @param op 				simulation operation
	 * @return conversion result status
	 */
	mlir::LogicalResult createMainFunction(mlir::OpBuilder& builder, ModelOp op) const
	{
		mlir::Location loc = op.getLoc();
		mlir::OpBuilder::InsertionGuard guard(builder);

		// Create the function inside the parent module
    auto module = op->getParentOfType<mlir::ModuleOp>();
		builder.setInsertionPointToEnd(module.getBody());

		llvm::SmallVector<mlir::Type, 3> argsTypes;
		llvm::SmallVector<mlir::Type, 3> resultsTypes;

		argsTypes.push_back(builder.getI32Type());
		argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(builder.getIntegerType(8))));
		resultsTypes.push_back(builder.getI32Type());

		auto function = builder.create<mlir::FuncOp>(
				loc, mainFunctionName, builder.getFunctionType(argsTypes, resultsTypes));

		auto* entryBlock = function.addEntryBlock();
		builder.setInsertionPointToStart(entryBlock);

		// Initialize the runtime environment
		auto runtimeInitFunction = getOrInsertFunction(builder, module, runtimeInitFunctionName, mlir::LLVM::LLVMFunctionType::get(getVoidType(), llvm::None));
		builder.create<mlir::LLVM::CallOp>(loc, runtimeInitFunction, llvm::None);

		// Initialize the variables
		mlir::Value data = builder.create<mlir::CallOp>(loc, initFunctionName, getVoidPtrType(), llvm::None).getResult(0);
		builder.create<mlir::CallOp>(loc, printHeaderFunctionName, llvm::None, data);
		builder.create<mlir::CallOp>(loc, printFunctionName, llvm::None, data);

		// Create the simulation loop
		auto loop = builder.create<mlir::scf::WhileOp>(loc, llvm::None, llvm::None);

		{
			mlir::OpBuilder::InsertionGuard g(builder);

			mlir::Block* conditionBlock = builder.createBlock(&loop.before());
			builder.setInsertionPointToStart(conditionBlock);
			mlir::Value shouldContinue = builder.create<mlir::CallOp>(loc, stepFunctionName, builder.getI1Type(), data).getResult(0);
			builder.create<mlir::scf::ConditionOp>(loc, shouldContinue, llvm::None);

			// The body contains just the print call, because the update is
			// already done by the "step "function in the condition region.

			mlir::Block* bodyBlock = builder.createBlock(&loop.after());
			builder.setInsertionPointToStart(bodyBlock);
			builder.create<mlir::CallOp>(loc, printFunctionName, llvm::None, data);
			builder.create<mlir::scf::YieldOp>(loc);
		}

		// Deallocate the variables
		builder.create<mlir::CallOp>(loc, deinitFunctionName, llvm::None, data);

	  // Deinitialize the runtime environment
    auto runtimeDeinitFunction = getOrInsertFunction(builder, module, runtimeDeinitFunctionName, mlir::LLVM::LLVMFunctionType::get(getVoidType(), llvm::None));
    builder.create<mlir::LLVM::CallOp>(loc, runtimeDeinitFunction, llvm::None);

		mlir::Value returnValue = builder.create<mlir::ConstantOp>(loc, builder.getI32IntegerAttr(0));
		builder.create<mlir::ReturnOp>(loc, returnValue);

		return mlir::success();
	}

	SolveModelOptions options;
	mlir::BlockAndValueMapping* derivativesMap;
};

struct EquationOpPattern : public mlir::OpRewritePattern<EquationOp>
{
	using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		// Create the assignment
		auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
		rewriter.setInsertionPoint(sides);

		for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
		{
			if (auto loadOp = mlir::dyn_cast<LoadOp>(lhs.getDefiningOp()))
			{
				assert(loadOp.indexes().empty());
				rewriter.create<AssignmentOp>(sides.getLoc(), rhs, loadOp.memory());
			}
			else
			{
				rewriter.create<AssignmentOp>(sides->getLoc(), rhs, lhs);
			}
		}

		rewriter.eraseOp(sides);

		// Inline the equation body
		rewriter.setInsertionPoint(op);
		rewriter.mergeBlockBefore(op.body(), op);

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct ForEquationOpPattern : public mlir::OpRewritePattern<ForEquationOp>
{
	using mlir::OpRewritePattern<ForEquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(ForEquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto loc = op->getLoc();

		llvm::SmallVector<mlir::Value, 3> lowerBounds;
		llvm::SmallVector<mlir::Value, 3> upperBounds;
		llvm::SmallVector<mlir::Value, 3> steps;

    /*
		for (auto induction : op.inductionsDefinitions())
		{
			auto inductionOp = mlir::cast<InductionOp>(induction.getDefiningOp());

			mlir::Value start = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(inductionOp.start()));
			mlir::Value end = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(inductionOp.end() + 1));
			mlir::Value step = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(1));

			lowerBounds.push_back(start);
			upperBounds.push_back(end);
			steps.push_back(step);
		}
     */

		llvm::SmallVector<mlir::Value, 3> inductionVariables;

		for (const auto& [lowerBound, upperBound, step] : llvm::zip(lowerBounds, upperBounds, steps))
		{
			auto forOp = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
			inductionVariables.push_back(forOp.getInductionVar());
			rewriter.setInsertionPointToStart(forOp.getBody());
		}

		rewriter.mergeBlockBefore(op.body(), rewriter.getInsertionBlock()->getTerminator(), inductionVariables);
		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct EquationSidesOpPattern : public mlir::OpRewritePattern<EquationSidesOp>
{
  using mlir::OpRewritePattern<EquationSidesOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(EquationSidesOp op, mlir::PatternRewriter& rewriter) const override
  {
    auto loc = op->getLoc();

    for (auto [lhs, rhs] : llvm::zip(op.lhs(), op.rhs()))
    {
      if (auto loadOp = mlir::dyn_cast<LoadOp>(lhs.getDefiningOp()))
      {
         assert(loadOp.indexes().empty());
         rewriter.create<AssignmentOp>(loc, rhs, loadOp.memory());
      }
      else
      {
        rewriter.create<AssignmentOp>(loc, rhs, lhs);
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// Remove the derivative operations by replacing them with appropriate
/// buffers, and set the derived variables as state variables.
template<typename EquationType>
static mlir::LogicalResult removeDerivatives(
    mlir::OpBuilder& builder, Model<EquationType>& model, mlir::BlockAndValueMapping& derivatives)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  auto appendIndexesFn = [](llvm::SmallVectorImpl<mlir::Value>& destination, mlir::ValueRange indexes) {
    for (size_t i = 0, e = indexes.size(); i < e; ++i) {
      mlir::Value index = indexes[e - 1 - i];
      destination.push_back(index);
    }
  };

  auto derivativeOrder = [&](mlir::Value value) -> unsigned int {
    auto inverseMap = derivatives.getInverse();
    unsigned int result = 0;

    while (inverseMap.contains(value)) {
      ++result;
      value = inverseMap.lookup(value);
    }

    return result;
  };

  model.getOperation().walk([&](DerOp op) {
    mlir::Location loc = op->getLoc();
    mlir::Value operand = op.operand();

    // If the value to be derived belongs to an array, then also the derived
    // value is stored within an array. Thus, we need to store its position.

    llvm::SmallVector<mlir::Value, 3> subscriptions;

    while (!operand.isa<mlir::BlockArgument>())
    {
      mlir::Operation* definingOp = operand.getDefiningOp();
      assert(mlir::isa<LoadOp>(definingOp) || mlir::isa<SubscriptionOp>(definingOp));

      if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp))
      {
        appendIndexesFn(subscriptions, loadOp.indexes());
        operand = loadOp.memory();
      }

      auto subscriptionOp = mlir::cast<SubscriptionOp>(definingOp);
      appendIndexesFn(subscriptions, subscriptionOp.indexes());
      operand = subscriptionOp.source();
    }

    if (!derivatives.contains(operand))
    {
      auto model = op->getParentOfType<ModelOp>();
      auto terminator = mlir::cast<YieldOp>(model.init().back().getTerminator());
      builder.setInsertionPoint(terminator);

      size_t index = operand.cast<mlir::BlockArgument>().getArgNumber();
      auto memberCreateOp = terminator.values()[index].getDefiningOp<MemberCreateOp>();
      auto nextDerName = getNextFullDerVariableName(memberCreateOp.name(), derivativeOrder(operand) + 1);

      assert(operand.getType().isa<ArrayType>());
      auto arrayType = operand.getType().cast<ArrayType>().toElementType(RealType::get(builder.getContext()));

      // Create the member and initialize it
      auto memberType = MemberType::get(arrayType.toAllocationScope(BufferAllocationScope::heap));
      mlir::Value memberDer = builder.create<MemberCreateOp>(loc, nextDerName, memberType, memberCreateOp.dynamicDimensions(), memberCreateOp.isConstant());
      mlir::Value zero = builder.create<ConstantOp>(loc, RealAttribute::get(builder.getContext(), 0));
      mlir::Value array = builder.create<MemberLoadOp>(loc, memberType.unwrap(), memberDer);
      builder.create<FillOp>(loc, zero, array);

      // Update the terminator values
      llvm::SmallVector<mlir::Value, 3> args(terminator.values().begin(), terminator.values().end());
      args.push_back(memberDer);
      builder.create<YieldOp>(loc, args);
      terminator.erase();

      // Add the new argument to the body of the model
      auto bodyArgument = model.body().addArgument(arrayType);
      derivatives.map(operand, bodyArgument);
    }

    builder.setInsertionPoint(op);
    mlir::Value derVar = derivatives.lookup(operand);

    llvm::SmallVector<mlir::Value, 3> reverted(subscriptions.rbegin(), subscriptions.rend());

    if (!subscriptions.empty()) {
      derVar = builder.create<SubscriptionOp>(loc, derVar, reverted);
    }

    if (auto arrayType = derVar.getType().cast<ArrayType>(); arrayType.getRank() == 0) {
      derVar = builder.create<LoadOp>(loc, derVar);
    }

    op.replaceAllUsesWith(derVar);
    op.erase();
  });

  return mlir::success();
}

/// Get all the variables that are declared inside the Model operation, independently
/// from their nature (state variables, constants, etc.).
static Variables discoverVariables(ModelOp model)
{
  Variables result;

  mlir::ValueRange vars = model.body().getArguments();

  for (size_t i = 1; i < vars.size(); ++i) {
    result.add(std::make_unique<Variable>(vars[i]));
  }

  return result;
}

/// Get the equations that are declared inside the Model operation.
static Equations<Equation> discoverEquations(ModelOp model, Variables variables)
{
  Equations<Equation> result;

  model.walk([&](EquationOp equationOp) {
    result.add(Equation::build(equationOp, variables));
  });

  return result;
}

/// Match each scalar variable to a scalar equation.
template<typename EquationType>
static mlir::LogicalResult matching(
    Model<MatchedEquation>& result,
    mlir::OpBuilder& builder,
    Model<EquationType>& model,
    const mlir::BlockAndValueMapping& derivatives)
{
  Variables allVariables = model.getVariables();

  // Filter the variables. State and constant ones must not in fact
  // take part into the matching process as their values are already
  // determined (state variables depend on their derivatives, while
  // constants have a fixed value).

  Variables variables;

  for (const auto& variable : allVariables) {
    mlir::Value var = variable->getValue();

    if (!derivatives.contains(var)) {
      auto nonStateVariable = std::make_unique<Variable>(var);

      if (!nonStateVariable->isConstant()) {
        variables.add(std::move(nonStateVariable));
      }
    }
  }

  model.setVariables(variables);
  result.setVariables(variables);

  model.getEquations().setVariables(model.getVariables());

  MatchingGraph<Variable*, Equation*> matchingGraph;

  for (const auto& variable : model.getVariables()) {
    matchingGraph.addVariable(variable.get());
  }

  for (const auto& equation : model.getEquations()) {
    matchingGraph.addEquation(equation.get());
  }

  if (!matchingGraph.simplify())
  {
    model.getOperation().emitError("Inconsistency found during the matching simplification process");
    return mlir::failure();
  }

  if (!matchingGraph.match())
  {
    model.getOperation().emitError("Matching failed");
    return mlir::failure();
  }

  Equations<MatchedEquation> matchedEquations;

  for (auto& solution : matchingGraph.getMatch()) {
    auto clone = solution.getEquation()->clone();
    auto matchedEquation = std::make_unique<MatchedEquation>(std::move(clone), solution.getAccess());

    for (size_t i = 0, e = solution.getEquation()->getNumOfIterationVars(); i < e; ++i) {
      matchedEquation->setMatchedIndexes(i, solution.getRangeBegin(i), solution.getRangeEnd(i));
    }

    matchedEquations.add(std::move(matchedEquation));
  }

  result.setEquations(matchedEquations);
  return mlir::success();
}

/// Model solving pass.
/// Its objective is to convert a descriptive (and thus not sequential) model
/// into an algorithmic one and to create the functions controlling the simulation.
class SolveModelPass: public mlir::PassWrapper<SolveModelPass, mlir::OperationPass<ModelOp>>
{
	public:
	explicit SolveModelPass(SolveModelOptions options, unsigned int bitWidth)
			: options(std::move(options)),
				bitWidth(std::move(bitWidth))
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::scf::SCFDialect>();
		registry.insert<mlir::LLVM::LLVMDialect>();
	}

	void runOnOperation() override
	{
    Model model(getOperation());
    mlir::OpBuilder builder(model.getOperation());

    // Remove the derivative operations and allocate the appropriate memory buffers
    mlir::BlockAndValueMapping derivatives;

    if (failed(removeDerivatives(builder, model, derivatives))) {
      model.getOperation().emitError("Derivative could not be converted to variables");
      return signalPassFailure();
    }

    // Now that the additional variables have been created, we can start a discovery process
    model.setVariables(discoverVariables(model.getOperation()));
    model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

    // Matching process
    Model<MatchedEquation> matchedModel(model.getOperation());

    if (mlir::failed(::matching(matchedModel, builder, model, derivatives))) {
      return signalPassFailure();
    }

    matchedModel.getOperation().erase();

    /*
    // Select and use the solver
    if (failed(selectSolver(builder, derivatives)))
      return signalPassFailure();

    // The model has been solved and we can now proceed to create the update
    // functions and, if requested, the main simulation loop.
    if (auto status = createSimulationFunctions(derivatives); failed(status))
    	return signalPassFailure();
     */

    /*
    // Create the model
    Model model = Model::build(simulation);

    // Remove the derivative operations and allocate the appropriate buffers
    if (failed(removeDerivatives(builder, model, derivatives)))
      return signalPassFailure();

    // Match
    if (failed(match(model, options.matchingMaxIterations)))
      return signalPassFailure();
     */

    /*
    // Solve circular dependencies
    if (failed(solveSCCs(builder, model, options.sccMaxIterations)))
      return signalPassFailure();

    // Schedule
    if (failed(schedule(model)))
      return signalPassFailure();

    // Explicitate the equations so that the updated variable is the only
    // one on the left-hand side of the equation.
    if (failed(explicitateEquations(model)))
      return signalPassFailure();

    // Select and use the solver
    if (failed(selectSolver(builder, model)))
      return signalPassFailure();
     */

		// The model has been solved and we can now proceed to create the update
		// functions and, if requested, the main simulation loop.
		//if (auto status = createSimulationFunctions(derivatives); failed(status))
		//	return signalPassFailure();
	}




	mlir::LogicalResult selectSolver(mlir::OpBuilder& builder, const mlir::BlockAndValueMapping& derivatives)
	{
		if (options.solver == ForwardEuler)
			return updateStates(builder, derivatives);

		if (options.solver == CleverDAE)
			return addBltBlocks(builder);

		return mlir::failure();
	}

	/**
	 * Calculate the values that the state variables will have in the next
	 * iteration.
	 */
	mlir::LogicalResult updateStates(mlir::OpBuilder& builder, const mlir::BlockAndValueMapping& derivatives)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
    ModelOp model = getOperation();
		mlir::Location loc = model->getLoc();

		// Theoretically, given a state variable x and the current step time n,
		// the value of x(n + 1) should be determined at the end of the step of
		// time n, and the assignment of the new value should be done at time
		// n + 1. Anyway, doing so would require to create an additional buffer
		// to store x(n + 1), so that it can be assigned at the beginning of step
		// n + 1. This allocation can be avoided by computing x(n + 1) right at
		// the beginning of step n + 1, when the derivatives still have the values
		// of step n.

		builder.setInsertionPointToStart(&model.body().front());
		mlir::Value timeStep = builder.create<ConstantOp>(loc, model.timeStep());

    for (mlir::Value variable : model.body().getArguments())
    {
      if (derivatives.contains(variable))
      {
        mlir::Value derivative = derivatives.lookup(variable);
        mlir::Value nextValue = builder.create<MulOp>(loc, derivative.getType(), derivative, timeStep);
        nextValue = builder.create<AddOp>(loc, variable.getType(), nextValue, variable);
        builder.create<AssignmentOp>(loc, nextValue, variable);
      }
    }

		return mlir::success();
	}

	/**
	 * This method transforms all differential equations and implicit equations
	 * into BLT blocks within the model. Then the assigned model, ready for
	 * lowering, is returned.
	 */
	mlir::LogicalResult addBltBlocks(mlir::OpBuilder& builder)
	{
		assert(false && "To be implemented");
	}

	mlir::LogicalResult createSimulationFunctions(mlir::BlockAndValueMapping& derivativesMap)
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		target.addIllegalOp<
		        ModelOp,
            MemberCreateOp,
            MemberLoadOp,
            MemberStoreOp,
            EquationOp,
            ForEquationOp,
            EquationSidesOp>();

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<ModelOpPattern>(&getContext(), typeConverter, options, derivativesMap);
		patterns.insert<EquationOpPattern, EquationSidesOpPattern>(&getContext());

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
        return status;

		return mlir::success();
	}

	private:
	SolveModelOptions options;
	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createSolveModelPass(SolveModelOptions options, unsigned int bitWidth)
{
	return std::make_unique<SolveModelPass>(options, bitWidth);
}
