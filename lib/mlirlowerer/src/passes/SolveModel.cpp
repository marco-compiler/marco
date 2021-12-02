#include <llvm/ADT/STLExtras.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <marco/mlirlowerer/dialects/ida/IdaBuilder.h>
#include <marco/mlirlowerer/dialects/ida/IdaDialect.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaBuilder.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <marco/mlirlowerer/passes/SolveModel.h>
#include <marco/mlirlowerer/passes/TypeConverter.h>
#include <marco/mlirlowerer/passes/matching/LinSolver.h>
#include <marco/mlirlowerer/passes/matching/Matching.h>
#include <marco/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <marco/mlirlowerer/passes/matching/Schedule.h>
#include <marco/mlirlowerer/passes/model/BltBlock.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/mlirlowerer/passes/TypeConverter.h>
#include <marco/utils/VariableFilter.h>

using namespace marco;
using namespace codegen;
using namespace model;
using namespace modelica;
using namespace ida;

struct SimulationOpLoopifyPattern : public mlir::OpRewritePattern<SimulationOp>
{
	using mlir::OpRewritePattern<SimulationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SimulationOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());
		rewriter.eraseOp(terminator);

		llvm::SmallVector<mlir::Value, 3> newVars;
		newVars.push_back(terminator.values()[0]);

		unsigned int index = 1;

		std::map<ForEquationOp, mlir::Value> inductions;

		for (auto it = std::next(terminator.values().begin()); it != terminator.values().end(); ++it)
		{
			mlir::Value value = *it;

			if (auto arrayType = value.getType().dyn_cast<ArrayType>(); arrayType && arrayType.getRank() == 0)
			{
				{
					mlir::OpBuilder::InsertionGuard guard(rewriter);
					rewriter.setInsertionPoint(value.getDefiningOp());
					mlir::Value newVar = rewriter.create<AllocOp>(value.getLoc(), arrayType.getElementType(), 1, llvm::None, false);
					mlir::Value zeroValue = rewriter.create<ConstantOp>(value.getLoc(), rewriter.getIndexAttr(0));
					mlir::Value subscription = rewriter.create<SubscriptionOp>(value.getLoc(), newVar, zeroValue);

					value.replaceAllUsesWith(subscription);
					rewriter.eraseOp(value.getDefiningOp());
					value = newVar;
				}

				{
					// Fix body argument
					mlir::OpBuilder::InsertionGuard guard(rewriter);

					auto originalArgument = op.body().getArgument(index);
					auto newArgument = op.body().insertArgument(index + 1, value.getType().cast<ArrayType>().toUnknownAllocationScope());

					for (auto useIt = originalArgument.use_begin(), end = originalArgument.use_end(); useIt != end;)
					{
						auto& use = *useIt;
						++useIt;
						rewriter.setInsertionPoint(use.getOwner());

						auto* parentEquation = use.getOwner()->getParentWithTrait<EquationInterface::Trait>();

						if (auto equationOp = mlir::dyn_cast<EquationOp>(parentEquation))
						{
							auto forEquationOp = convertToForEquation(rewriter, equationOp.getLoc(), equationOp);
							inductions[forEquationOp] = forEquationOp.body()->getArgument(0);
						}
					}

					for (auto useIt = originalArgument.use_begin(), end = originalArgument.use_end(); useIt != end;)
					{
						auto& use = *useIt;
						++useIt;
						rewriter.setInsertionPoint(use.getOwner());

						auto* parentEquation = use.getOwner()->getParentWithTrait<EquationInterface::Trait>();
						auto forEquation = mlir::cast<ForEquationOp>(parentEquation);

						if (inductions.find(forEquation) == inductions.end())
						{
							mlir::Value i = rewriter.create<ConstantOp>(use.get().getLoc(), rewriter.getIndexAttr(0));
							mlir::Value subscription = rewriter.create<SubscriptionOp>(use.get().getLoc(), newArgument, i);
							use.set(subscription);
						}
						else
						{
							mlir::Value i = inductions[forEquation];
							i = rewriter.create<SubOp>(i.getLoc(), i.getType(), i, rewriter.create<ConstantOp>(i.getLoc(), rewriter.getIndexAttr(1)));
							mlir::Value subscription = rewriter.create<SubscriptionOp>(use.get().getLoc(), newArgument, i);
							use.set(subscription);
						}
					}

					op.body().eraseArgument(index);
				}

			}

			newVars.push_back(value);
			++index;
		}

		{
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToEnd(&op.init().back());
			rewriter.create<YieldOp>(terminator.getLoc(), newVars);
		}

		// Replace operation with a new one
		auto result = rewriter.create<SimulationOp>(op->getLoc(), op.variableNames(), op.startTime(), op.endTime(), op.timeStep(), op.relTol(), op.absTol(), op.body().getArgumentTypes());
		rewriter.mergeBlocks(&op.init().front(), &result.init().front(), result.init().getArguments());
		rewriter.mergeBlocks(&op.body().front(), &result.body().front(), result.body().getArguments());

		rewriter.eraseOp(op);
		return mlir::success();
	}

	private:
	static ForEquationOp convertToForEquation(mlir::PatternRewriter& rewriter, mlir::Location loc, EquationOp equation)
	{
		mlir::OpBuilder::InsertionGuard guard(rewriter);
		rewriter.setInsertionPoint(equation);

		auto forEquation = rewriter.create<ForEquationOp>(loc, 1);

		// Inductions
		rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
		mlir::Value induction = rewriter.create<InductionOp>(loc, 1, 1);
		rewriter.create<YieldOp>(loc, induction);

		// Body
		rewriter.mergeBlocks(equation.body(), forEquation.body());

		rewriter.eraseOp(equation);
		return forEquation;
	}
};

struct EquationOpLoopifyPattern : public mlir::OpRewritePattern<EquationOp>
{
	using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		auto forEquation = rewriter.create<ForEquationOp>(loc, 1);

		// Inductions
		rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
		mlir::Value induction = rewriter.create<InductionOp>(loc, 1, 1);
		rewriter.create<YieldOp>(loc, induction);

		// Body
		rewriter.mergeBlocks(op.body(), forEquation.body());

		forEquation.body()->walk([&](SubscriptionOp subscription) {
			if (subscription.indexes().size() == 1)
			{
				auto index = subscription.indexes()[0].getDefiningOp<ConstantOp>();
				auto indexValue = index.value().cast<modelica::IntegerAttribute>().getValue();
				rewriter.setInsertionPoint(subscription);

				mlir::Value newIndex = rewriter.create<AddOp>(
						subscription->getLoc(), rewriter.getIndexType(), forEquation.induction(0),
						rewriter.create<ConstantOp>(subscription->getLoc(), rewriter.getIndexAttr(indexValue - 1)));

				rewriter.replaceOp(index, newIndex);
			}
		});

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct EquationOpScalarizePattern : public mlir::OpRewritePattern<EquationOp>
{
	using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
		assert(sides.lhs().size() == 1 && sides.rhs().size() == 1);

		auto lhs = sides.lhs()[0];
		auto rhs = sides.rhs()[0];

		auto lhsArrayType = lhs.getType().cast<ArrayType>();
		auto rhsArrayType = rhs.getType().cast<ArrayType>();
		assert(lhsArrayType.getRank() == rhsArrayType.getRank());

		auto forEquation = rewriter.create<ForEquationOp>(location, lhsArrayType.getRank());

		{
			// Inductions
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
			llvm::SmallVector<mlir::Value, 3> inductions;

			for (const auto& [left, right] : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape()))
			{
				assert(left != -1 || right != -1);

				if (left != -1 && right != -1)
					assert(left == right);

				long size = std::max(left, right);
				mlir::Value induction = rewriter.create<InductionOp>(location, 0, size - 1);
				inductions.push_back(induction);
			}

			rewriter.create<YieldOp>(location, inductions);
		}

		{
			// Body
			mlir::OpBuilder::InsertionGuard guard(rewriter);

			rewriter.mergeBlocks(op.body(), forEquation.body());
			rewriter.setInsertionPoint(sides);

			llvm::SmallVector<mlir::Value, 1> newLhs;
			llvm::SmallVector<mlir::Value, 1> newRhs;

			for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
			{
				auto leftSubscription = rewriter.create<SubscriptionOp>(location, lhs, forEquation.inductions());
				newLhs.push_back(rewriter.create<LoadOp>(location, leftSubscription));

				auto rightSubscription = rewriter.create<SubscriptionOp>(location, rhs, forEquation.inductions());
				newRhs.push_back(rewriter.create<LoadOp>(location, rightSubscription));
			}

			rewriter.setInsertionPointAfter(sides);
			rewriter.replaceOpWithNewOp<EquationSidesOp>(sides, newLhs, newRhs);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct ForEquationOpScalarizePattern : public mlir::OpRewritePattern<ForEquationOp>
{
	using mlir::OpRewritePattern<ForEquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(ForEquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
		assert(sides.lhs().size() == 1 && sides.rhs().size() == 1);

		auto lhs = sides.lhs()[0];
		auto rhs = sides.rhs()[0];

		auto lhsArrayType = lhs.getType().cast<ArrayType>();
		auto rhsArrayType = rhs.getType().cast<ArrayType>();
		assert(lhsArrayType.getRank() == rhsArrayType.getRank());

		auto forEquation = rewriter.create<ForEquationOp>(location, lhsArrayType.getRank() + op.inductionsDefinitions().size());

		{
			// Inductions
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
			llvm::SmallVector<mlir::Value, 3> inductions;

			for (const auto& [left, right] : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape()))
			{
				assert(left != -1 || right != -1);

				if (left != -1 && right != -1)
					assert(left == right);

				long size = std::max(left, right);
				mlir::Value induction = rewriter.create<InductionOp>(location, 0, size - 1);
				inductions.push_back(induction);
			}

			for (auto induction : op.inductionsDefinitions())
			{
				mlir::Operation* clone = rewriter.clone(*induction.getDefiningOp());
				inductions.push_back(clone->getResult(0));
			}

			rewriter.create<YieldOp>(location, inductions);
		}

		{
			// Body
			mlir::OpBuilder::InsertionGuard guard(rewriter);

			rewriter.mergeBlocks(op.body(), forEquation.body());
			rewriter.setInsertionPoint(sides);

			llvm::SmallVector<mlir::Value, 1> newLhs;
			llvm::SmallVector<mlir::Value, 1> newRhs;

			mlir::ValueRange allInductions = forEquation.inductions();
			mlir::ValueRange newInductions = mlir::ValueRange(allInductions.begin(), allInductions.begin() + lhsArrayType.getRank());

			for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
			{
				auto leftSubscription = rewriter.create<SubscriptionOp>(location, lhs, newInductions);
				newLhs.push_back(rewriter.create<LoadOp>(location, leftSubscription));

				auto rightSubscription = rewriter.create<SubscriptionOp>(location, rhs, newInductions);
				newRhs.push_back(rewriter.create<LoadOp>(location, rightSubscription));
			}

			rewriter.setInsertionPointAfter(sides);
			rewriter.replaceOpWithNewOp<EquationSidesOp>(sides, newLhs, newRhs);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct DerOpPattern : public mlir::OpRewritePattern<DerOp>
{
	DerOpPattern(mlir::MLIRContext* context, Model& model, mlir::BlockAndValueMapping& derivatives)
			: mlir::OpRewritePattern<DerOp>(context), model(&model), derivatives(&derivatives)
	{
	}

	mlir::LogicalResult matchAndRewrite(DerOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		mlir::Value operand = op.operand();

		// If the value to be derived belongs to an array, then also the derived
		// value is stored within an array. Thus we need to store its position.

		llvm::SmallVector<mlir::Value, 3> subscriptions;

		while (!operand.isa<mlir::BlockArgument>())
		{
			if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(operand.getDefiningOp()))
			{
				mlir::ValueRange indexes = subscriptionOp.indexes();
				subscriptions.append(indexes.begin(), indexes.end());
				operand = subscriptionOp.source();
			}
			else
			{
				return rewriter.notifyMatchFailure(op, "Unexpected operation");
			}
		}

		auto simulation = op->getParentOfType<SimulationOp>();
		mlir::Value var = simulation.getVariableAllocation(operand);
		auto variable = model->getVariable(var);
		mlir::Value derVar;

		if (!derivatives->contains(operand))
		{
			auto terminator = mlir::cast<YieldOp>(var.getParentBlock()->getTerminator());
			rewriter.setInsertionPointAfter(terminator);

			llvm::SmallVector<mlir::Value, 3> args;

			for (mlir::Value arg : terminator.values())
				args.push_back(arg);

			if (auto arrayType = variable.getReference().getType().dyn_cast<ArrayType>())
			{
				derVar = rewriter.create<AllocOp>(loc, arrayType.getElementType(), arrayType.getShape(), llvm::None, false);
				mlir::Value zero = createZeroValue(rewriter, loc, arrayType.getElementType());
				rewriter.create<FillOp>(loc, zero, derVar);
			}
			else
			{
				derVar = rewriter.create<AllocOp>(loc, variable.getReference().getType(), llvm::None, llvm::None, false);
				mlir::Value zero = createZeroValue(rewriter, loc, variable.getReference().getType());
				rewriter.create<AssignmentOp>(loc, zero, derVar);
			}

			model->addVariable(derVar);
			variable.setDer(model->getVariable(derVar));

			args.push_back(derVar);
			rewriter.create<YieldOp>(terminator.getLoc(), args);
			rewriter.eraseOp(terminator);

			auto newArgumentType = derVar.getType().cast<ArrayType>().toUnknownAllocationScope();
			auto bodyArgument = simulation.body().addArgument(newArgumentType);

			derivatives->map(operand, bodyArgument);
		}
		else
		{
			derVar = variable.getDerivative();
		}

		rewriter.setInsertionPoint(op);
		derVar = derivatives->lookup(operand);

		if (!subscriptions.empty())
				derVar = rewriter.create<SubscriptionOp>(loc, derVar, subscriptions);

		if (auto arrayType = derVar.getType().cast<ArrayType>(); arrayType.getRank() == 0)
				derVar = rewriter.create<LoadOp>(loc, derVar);

		rewriter.replaceOp(op, derVar);

		return mlir::success();
	}

	private:
	mlir::Value createZeroValue(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type) const
	{
		if (type.isa<modelica::BooleanType>())
			return builder.create<ConstantOp>(loc, modelica::BooleanAttribute::get(type, false));

		if (type.isa<modelica::IntegerType>())
			return builder.create<ConstantOp>(loc, modelica::IntegerAttribute::get(type, 0));

		assert(type.isa<modelica::RealType>());
		return builder.create<ConstantOp>(loc, modelica::RealAttribute::get(type, 0));
	}

	Model* model;
	mlir::BlockAndValueMapping* derivatives;
};

struct SimulationOpPattern : public mlir::ConvertOpToLLVMPattern<SimulationOp>
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
	SimulationOpPattern(mlir::MLIRContext* ctx,
											TypeConverter& typeConverter,
											SolveModelOptions options,
											mlir::BlockAndValueMapping& derivativesMap)
			: mlir::ConvertOpToLLVMPattern<SimulationOp>(typeConverter),
				options(std::move(options)),
				derivativesMap(&derivativesMap)
	{
	}

	mlir::LogicalResult matchAndRewrite(SimulationOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		// Save the types of the variables which compose the data structure
		llvm::SmallVector<mlir::Type, 3> varTypes;
		auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());

		for (const auto& type : terminator.values().getTypes())
			varTypes.push_back(type);

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

		if (auto status = createInitFunction(rewriter, op, varTypes); failed(status))
			return status;

		if (auto status = createDeinitFunction(rewriter, op, varTypes); failed(status))
			return status;

		if (auto status = createStepFunction(rewriter, op, varTypes); failed(status))
			return status;

		if (auto status = createPrintHeaderFunction(rewriter, op, varTypes, derivativesPositions); failed(status))
			return status;

		if (auto status = createPrintFunction(rewriter, op, varTypes, derivativesPositions); failed(status))
			return status;

		if (options.emitMain)
			if (auto status = createMainFunction(rewriter, op); failed(status))
				return status;

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
	mlir::LogicalResult createInitFunction(mlir::ConversionPatternRewriter& rewriter, SimulationOp op, mlir::TypeRange varTypes) const
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
			values.push_back(removeAllocationScopeFn(var));

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
	mlir::LogicalResult createDeinitFunction(mlir::OpBuilder& builder, SimulationOp op, mlir::TypeRange varTypes) const
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

		// Print statistics and free the allocated IDA user data
		if (options.solver == CleverDAE)
		{
			mlir::Value userData = extractValue(builder, structValue, varTypes[1], 1);
			userData = builder.create<LoadOp>(loc, userData);

			if (options.printStatistics)
				builder.create<PrintStatisticsOp>(loc, userData);

			builder.create<FreeDataOp>(loc, userData);
		}

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
	mlir::LogicalResult createStepFunction(mlir::ConversionPatternRewriter& rewriter, SimulationOp op, mlir::TypeRange varTypes) const
	{
		mlir::Location loc = op.getLoc();
		mlir::OpBuilder::InsertionGuard guard(rewriter);

		// Create the function inside the parent module
		rewriter.setInsertionPointToEnd(op->getParentOfType<mlir::ModuleOp>().getBody());

		auto function = rewriter.create<mlir::FuncOp>(
				loc, "step",
				rewriter.getFunctionType(getVoidPtrType(), rewriter.getI1Type()));

		auto* entryBlock = function.addEntryBlock();
		rewriter.setInsertionPointToStart(entryBlock);

		// Extract the data from the struct
		mlir::Value structValue = loadDataFromOpaquePtr(rewriter, function.getArgument(0), varTypes);

		llvm::SmallVector<mlir::Value, 3> vars;

		for (const auto& varType : llvm::enumerate(varTypes))
			vars.push_back(extractValue(rewriter, structValue, varTypes[varType.index()], varType.index()));

		mlir::Value increasedTime;
		mlir::Value idaStepResult;

		if (options.solver == ForwardEuler)
		{
			// Increment the time
			mlir::Value timeStep = rewriter.create<ConstantOp>(loc, op.timeStep());
			mlir::Value currentTime = rewriter.create<LoadOp>(loc, vars[0]);
			increasedTime = rewriter.create<AddOp>(loc, currentTime.getType(), currentTime, timeStep);
			rewriter.create<StoreOp>(loc, increasedTime, vars[0]);
		}
		else if (options.solver == CleverDAE)
		{
			// Run one step of the ida solver
			mlir::Value userData = rewriter.create<LoadOp>(loc, vars[1]);
			mlir::Value stepOp = rewriter.create<StepOp>(loc, userData);
			idaStepResult = getTypeConverter()->materializeTargetConversion(
					rewriter, loc, rewriter.getI1Type(), stepOp);

			// Update the time based on ida
			increasedTime = rewriter.create<GetTimeOp>(loc, userData);
			rewriter.create<StoreOp>(loc, increasedTime, vars[0]);
		}

		// Check if the current time is less than the end time, minus an epsilon
		// because of floating points
		mlir::Value endTime = rewriter.create<ConstantOp>(loc, op.endTime());
		mlir::Value epsilon = rewriter.create<ConstantOp>(loc, modelica::RealAttribute::get(op.getContext(), 1e-12));
		endTime = rewriter.create<SubOp>(loc, endTime.getType(), endTime, epsilon);

		mlir::Value condition = rewriter.create<LtOp>(loc, modelica::BooleanType::get(op->getContext()), increasedTime, endTime);
		condition = getTypeConverter()->materializeTargetConversion(rewriter, condition.getLoc(), rewriter.getI1Type(), condition);

		// Check if the IDA solver failed
		if (options.solver == CleverDAE)
			condition = rewriter.create<mlir::AndOp>(loc, rewriter.getI1Type(), condition, idaStepResult);

		// If we didn't reach the end time, update the variables and return true to
		// continue the simulation, otherwise return false to stop the simulation.
		mlir::ReturnOp returnOp = rewriter.create<mlir::ReturnOp>(loc, condition);

		// Copy the body of the simulation
		mlir::BlockAndValueMapping mapping;

		for (const auto& [oldVar, newVar] : llvm::zip(op.body().getArguments(), vars))
			mapping.map(oldVar, newVar);

		// We need to keep track of the amount of equations processed, so that
		// we can give a unique name to each function generated.
		llvm::SmallVector<mlir::Value, 3> usedVars;
		size_t equationCounter = 0;

		rewriter.setInsertionPoint(returnOp);

		for (auto& bodyOp : op.body().front().without_terminator())
		{
			if (auto equation = mlir::dyn_cast<ForEquationOp>(bodyOp))
			{
				usedVars.clear();

				mlir::FuncOp equationFunction = convertForEquation(rewriter, equation, equationCounter, vars, usedVars);
				rewriter.create<mlir::CallOp>(equation->getLoc(), equationFunction, usedVars);
				++equationCounter;
			}
			else
			{
				rewriter.clone(bodyOp, mapping);
			}
		}

		return mlir::success();
	}

  mlir::FuncOp convertForEquation(
          mlir::ConversionPatternRewriter& rewriter,
          ForEquationOp eq, size_t counter,
          mlir::ValueRange vars,
          llvm::SmallVectorImpl<mlir::Value>& usedVars) const
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Location loc = eq->getLoc();

    auto module = eq->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(module.getBody());

    auto functionType = rewriter.getFunctionType(vars.getTypes(), llvm::None);
    auto name = "eq" + std::to_string(counter);

    auto function = rewriter.create<mlir::FuncOp>(loc, name, functionType);

    auto* entryBlock = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    // Create the loops
    llvm::SmallVector<mlir::Value, 3> lowerBounds;
    llvm::SmallVector<mlir::Value, 3> upperBounds;
    llvm::SmallVector<mlir::Value, 3> steps;

    for (auto induction : eq.inductionsDefinitions())
    {
      auto inductionOp = mlir::cast<InductionOp>(induction.getDefiningOp());

      mlir::Value start = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(inductionOp.start()));
      mlir::Value end = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(inductionOp.end() + 1));
      mlir::Value step = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(1));

      lowerBounds.push_back(start);
      upperBounds.push_back(end);
      steps.push_back(step);
    }

    llvm::SmallVector<mlir::Value, 3> inductions;

    for (const auto& [lowerBound, upperBound, step] : llvm::zip(lowerBounds, upperBounds, steps))
    {
      auto forOp = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
      inductions.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // Before moving the equation body, we need to map the old variables of
    // the SimulationOp to the new ones living in the dedicated function.
    // The same must be done with the induction variables.
    mlir::BlockAndValueMapping mapping;

    auto originalVars = eq->getParentOfType<SimulationOp>().body().getArguments();
    assert(originalVars.size() == function.getNumArguments());

    for (const auto& [oldVar, newVar] : llvm::zip(originalVars, function.getArguments()))
      mapping.map(oldVar, newVar);

    auto originalInductions = eq.body()->getArguments();
    assert(originalInductions.size() == inductions.size());

    for (const auto& [oldInduction, newInduction] : llvm::zip(originalInductions, inductions))
      mapping.map(oldInduction, newInduction);

    for (auto& op : eq.body()->getOperations())
    {
      rewriter.clone(op, mapping);
    }

    // Create the return instruction
    rewriter.setInsertionPointToEnd(&function.body().back());
    rewriter.create<mlir::ReturnOp>(loc);

    // Remove the arguments that are not used
    llvm::SmallVector<unsigned int, 3> argsToBeRemoved;

    for (const auto& arg : function.getArguments())
    {
      if (arg.getUsers().empty())
        argsToBeRemoved.push_back(arg.getArgNumber());
      else
        usedVars.push_back(vars[arg.getArgNumber()]);
    }

    function.eraseArguments(argsToBeRemoved);

    // Move loop invariants. This is needed because we scalarize the array
    // assignments, and thus arrays may be allocated way more times than
    // needed. Once the new matching & scheduling process will be complete,
    // this will hopefully not be required anymore.

    function.walk([&](mlir::LoopLikeOpInterface loopLike) {
      moveLoopInvariantCode(loopLike);
    });

    return function;
  }

  // To be removed once the new matching & scheduling process is finished
  void moveLoopInvariantCode(mlir::LoopLikeOpInterface looplike) const
  {
    auto& loopBody = looplike.getLoopBody();

    // We use two collections here as we need to preserve the order for insertion
    // and this is easiest.
    llvm::SmallPtrSet<mlir::Operation*, 8> willBeMovedSet;
    llvm::SmallVector<mlir::Operation*, 8> opsToMove;

    // Helper to check whether an operation is loop invariant wrt. SSA properties.
    auto isDefinedOutsideOfBody = [&](mlir::Value value) {
        auto definingOp = value.getDefiningOp();
        return (definingOp && !!willBeMovedSet.count(definingOp)) ||
               looplike.isDefinedOutsideOfLoop(value);
    };

    // Do not use walk here, as we do not want to go into nested regions and hoist
    // operations from there. These regions might have semantics unknown to this
    // rewriting. If the nested regions are loops, they will have been processed.
    for (auto &block : loopBody) {
      for (auto &op : block.without_terminator()) {
        if (canBeHoisted(&op, isDefinedOutsideOfBody)) {
          opsToMove.push_back(&op);
          willBeMovedSet.insert(&op);
        }
      }
    }

    auto result = looplike.moveOutOfLoop(opsToMove);
	assert(mlir::succeeded(result));
  }

  // To be removed once the new matching & scheduling process is finished
  static bool canBeHoisted(mlir::Operation* op, std::function<bool(mlir::Value)> definedOutside)
  {
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;

    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block.without_terminator())
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
		return getOrCreateGlobalString(loc, builder, "comma", mlir::StringRef(",\0", 2), module);
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
			SimulationOp op,
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
			SimulationOp op,
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
			SimulationOp op,
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
		unsigned int varsToSkip = 1; // + 1 to skip the "time" variable.

		if (options.solver == CleverDAE)
			varsToSkip++; // + 1 to skip the ida user data opaque pointer variable.

		for (const auto& var : llvm::enumerate(variableNames))
			variablePositionByName[var.value()] = var.index() + varsToSkip;

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
	mlir::LogicalResult createMainFunction(mlir::OpBuilder& builder, SimulationOp op) const
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

			// The body contains just the print call, because the update is
			// already done by the "step" function in the condition region.

			mlir::Block* conditionBlock = builder.createBlock(&loop.before());
			builder.setInsertionPointToStart(conditionBlock);
			mlir::Value shouldContinue = builder.create<mlir::CallOp>(loc, stepFunctionName, builder.getI1Type(), data).getResult(0);
			builder.create<mlir::CallOp>(loc, printFunctionName, llvm::None, data);
			builder.create<mlir::scf::ConditionOp>(loc, shouldContinue, llvm::None);

			mlir::Block* bodyBlock = builder.createBlock(&loop.after());
			builder.setInsertionPointToStart(bodyBlock);
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

/**
 * Model solver pass.
 * Its objective is to convert a descriptive (and thus not sequential) model
 * into an algorithmic one.
 */
class SolveModelPass: public mlir::PassWrapper<SolveModelPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit SolveModelPass(SolveModelOptions options, unsigned int bitWidth)
			: options(std::move(options)),
				bitWidth(std::move(bitWidth))
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<IdaDialect>();
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::scf::SCFDialect>();
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<mlir::LLVM::LLVMDialect>();
	}

	void runOnOperation() override
	{
		// Scalarize the equations consisting in array assignments, by adding
		// the required inductions.
		if (failed(scalarizeArrayEquations(getOperation())))
			return signalPassFailure();

		// Convert the scalar values into arrays of one element
		if (failed(loopify(getOperation())))
			return signalPassFailure();

		mlir::BlockAndValueMapping derivatives;

		getOperation()->walk([&](SimulationOp simulation) {
			// Create the model
			Model model = Model::build(simulation);

			// Remove the derivative operations and allocate the appropriate buffers
			if (failed(removeDerivatives(model, derivatives, getOperation())))
				return signalPassFailure();

			// Match
			if (failed(match(model, options.matchingMaxIterations)))
				return signalPassFailure();

			// Solve circular dependencies
			if (failed(solveSCCs(model, options.sccMaxIterations)))
				return signalPassFailure();

			// Explicitate the equations so that the updated variable is the only
			// one on the left-hand side of the equation. If the equation is implicit
			// add it to a non-trivial blt block.
			if (failed(explicitateEquations(model)))
				return signalPassFailure();

			// Add differential equations to non-trivial blt blocks. Also add all
			// equations where the matched variable is marked as non-trivial.
			if (options.solver == CleverDAE)
				if (failed(addDifferentialEqToBltBlocks(model)))
					return signalPassFailure();

			// Schedule
			if (failed(schedule(model)))
				return signalPassFailure();

			// Calculate the values that the state variables will have in the next
			// iteration.
			if (options.solver == ForwardEuler)
				if (failed(updateStates(model)))
					return signalPassFailure();

			// In order to hide trivial variable from the IDA solver, all trivial
			// variables present in non-trivial equations must be substituted with
			// the corresponding equation that computes them.
			if (options.solver == CleverDAE)
				if (failed(substituteTrivialVariables(model)))
					return signalPassFailure();

			// Add IDA to the module. This consists in adding all the necessary calls
			// to the runtime library which will handle the allocation, interacion and
			// deallocation of the classes required by IDA.
			if (options.solver == CleverDAE)
				if (failed(addIdaSolver(model)))
					return signalPassFailure();
		});

		// The model has been solved and we can now proceed to create the update
		// functions and, if requested, the main simulation loop.
		if (auto status = createSimulationFunctions(derivatives); failed(status))
			return signalPassFailure();
	}

	/**
	 * Convert the scalar variables to arrays with just one element and the
	 * scalar equations to for equations with a one-sized induction. This
	 * allows the subsequent transformation to ignore whether it was a scalar
	 * or a loop equation, and a later optimization pass can easily remove
	 * the new useless induction.
	 */
	static mlir::LogicalResult loopify(mlir::ModuleOp moduleOp)
	{
		mlir::ConversionTarget target(*moduleOp.getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		target.addDynamicallyLegalOp<SimulationOp>([](SimulationOp op) {
			auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());
			mlir::ValueRange args = terminator.values();

			for (auto it = std::next(args.begin()), end = args.end(); it != end; ++it)
				if (auto arrayType = (*it).getType().dyn_cast<ArrayType>())
					if (arrayType.getRank() == 0)
						return false;

			return true;
		});

		target.addIllegalOp<EquationOp>();

		mlir::OwningRewritePatternList patterns(moduleOp.getContext());
		patterns.insert<SimulationOpLoopifyPattern, EquationOpLoopifyPattern>(moduleOp.getContext());
		patterns.insert<SimulationOpLoopifyPattern>(moduleOp.getContext());

		if (auto status = applyPartialConversion(moduleOp, target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	/**
	 * If an equation consists in an assignment between two arrays, then
	 * convert it into a for equation, in order to scalarize the assignments.
	 */
	static mlir::LogicalResult scalarizeArrayEquations(mlir::ModuleOp moduleOp)
	{
		mlir::ConversionTarget target(*moduleOp.getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
			auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
			auto pairs = llvm::zip(sides.lhs(), sides.rhs());

			return llvm::all_of(pairs, [](const auto& pair) {
				mlir::Type lhs = std::get<0>(pair).getType();
				mlir::Type rhs = std::get<1>(pair).getType();

				return !lhs.isa<ArrayType>() && !rhs.isa<ArrayType>();
			});
		});

		target.addDynamicallyLegalOp<ForEquationOp>([](ForEquationOp op) {
			auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
			auto pairs = llvm::zip(sides.lhs(), sides.rhs());

			return llvm::all_of(pairs, [](const auto& pair) {
				mlir::Type lhs = std::get<0>(pair).getType();
				mlir::Type rhs = std::get<1>(pair).getType();

				return !lhs.isa<ArrayType>() && !rhs.isa<ArrayType>();
			});
		});

		mlir::OwningRewritePatternList patterns(moduleOp.getContext());

		patterns.insert<
		    EquationOpScalarizePattern,
				ForEquationOpScalarizePattern>(moduleOp.getContext());

		if (auto status = applyPartialConversion(moduleOp, target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	/**
	 * Remove the derivative operations by replacing them with appropriate
	 * buffers, and set the derived variables as state variables.
	 *
	 * @param model model
	 * @return conversion result
	 */
	static mlir::LogicalResult removeDerivatives(Model& model, mlir::BlockAndValueMapping& derivatives, mlir::ModuleOp moduleOp)
	{
		mlir::ConversionTarget target(*moduleOp.getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<DerOp>();

		mlir::OwningRewritePatternList patterns(moduleOp.getContext());
		patterns.insert<DerOpPattern>(moduleOp.getContext(), model, derivatives);

		if (auto status = applyPartialConversion(model.getOp(), target, std::move(patterns)); failed(status))
			return status;

		model.reloadIR();
		return mlir::success();
	}

	static mlir::LogicalResult explicitateEquations(Model& model)
	{
		llvm::SmallVector<Equation, 3> equations;
		llvm::SmallVector<BltBlock, 3> bltBlocks = model.getBltBlocks();

		for (Equation& equation : model.getEquations())
		{
			// If the equation cannot be explicitated, mark it as implicit.
			if (auto status = equation.explicitate(); failed(status))
			{
				model.getVariable(equation.getDeterminedVariable().getVar()).setTrivial(false);
				bltBlocks.push_back(BltBlock(llvm::SmallVector<Equation, 3>(1, equation)));
			}
			else
			{
				equations.push_back(equation);
			}
		}

		Model result(model.getOp(), model.getVariables(), equations, bltBlocks);
		model = result;
		return mlir::success();
	}

	static mlir::LogicalResult updateStates(Model& model)
	{
		// If the model contains algebraic loops, Forward Euler cannot solve this
		// model, an other solver must be used.
		if (!model.getBltBlocks().empty())
			return model.getOp()->emitError("Algebraic loops or implicit equations are present, the selected solver cannot be used");

		mlir::OpBuilder builder(model.getOp().getContext());
		mlir::Location loc = model.getOp()->getLoc();

		// Theoretically, given a state variable x and the current step time n,
		// the value of x(n + 1) should be determined at the end of the step of
		// time n, and the assignment of the new value should be done at time
		// n + 1. Anyway, doing so would require to create an additional buffer
		// to store x(n + 1), so that it can be assigned at the beginning of step
		// n + 1. This allocation can be avoided by computing x(n + 1) right at
		// the beginning of step n + 1, when the derivatives still have the values
		// of step n.

		builder.setInsertionPointToStart(&model.getOp().body().front());

		mlir::Value timeStep = builder.create<ConstantOp>(loc, model.getOp().timeStep());

		for (auto& variable : model.getVariables())
		{
			if (!variable->isState())
				continue;

			if (variable->isTime())
				continue;

			mlir::Value var = variable->getReference();
			mlir::Value der = variable->getDerivative();

			auto terminator = mlir::cast<YieldOp>(model.getOp().init().front().getTerminator());

			for (auto value : llvm::enumerate(terminator.values()))
			{
				if (value.value() == var)
					var = model.getOp().body().front().getArgument(value.index());

				if (value.value() == der)
					der = model.getOp().body().front().getArgument(value.index());
			}

			mlir::Value nextValue = builder.create<MulOp>(loc, der.getType(), der, timeStep);
			nextValue = builder.create<AddOp>(loc, var.getType(), nextValue, var);
			builder.create<AssignmentOp>(loc, nextValue, var);
		}

		return mlir::success();
	}

	static mlir::LogicalResult addDifferentialEqToBltBlocks(Model& model)
	{
		llvm::SmallVector<Equation, 3> equations;
		llvm::SmallVector<BltBlock, 3> bltBlocks = model.getBltBlocks();

		// If the equation is a differential equation, or if (part of) the matched variable
		// is not trivial, add the equation to a non-trivial blt block. 
		for (Equation& equation : model.getEquations())
		{
			Variable var = model.getVariable(equation.getDeterminedVariable().getVar());

			if (var.isDerivative() || !var.isTrivial())
			{
				var.setTrivial(false);
				bltBlocks.push_back(BltBlock(llvm::SmallVector<Equation, 3>(1, equation)));
			}
			else
			{
				equations.push_back(equation);
			}
		}

		Model result(model.getOp(), model.getVariables(), equations, bltBlocks);
		model = result;
		return mlir::success();
	}

	static mlir::LogicalResult substituteTrivialVariables(Model& model)
	{
		mlir::OpBuilder builder(model.getOp().getContext());
		llvm::SmallVector<BltBlock, 3> bltBlocks = model.getBltBlocks();

		// Add all trivial variables to a map in order to retrieve later the 
		// equation they are matched with.
		std::map<Variable, llvm::SmallVector<Equation*, 3>> trivialVariablesMap;
		std::map<Variable, ConstantOp> parametersMap;

		for (Equation& equation : model.getEquations())
		{
			Variable var = model.getVariable(equation.getDeterminedVariable().getVar());
			trivialVariablesMap[var].push_back(&equation);
			assert(var.isTrivial());
		}

		// Map also all parameters that are only initialized and not matched with any equation.
		model.getOp().init().walk([&](FillOp fillOp) {
			if (!model.hasVariable(fillOp.memory()))
					return;

			Variable var = model.getVariable(fillOp.memory());
			ConstantOp constantOp = mlir::cast<ConstantOp>(fillOp.value().getDefiningOp());
			parametersMap[var] = constantOp;
		});

		// For each equation in each bltBlocks, if an expression refers to a trivial
		// variable, substitute it with the corresponding equation.
		for (size_t i = 0; i < bltBlocks.size(); i++)
		{
			for (size_t j = 0; j < bltBlocks[i].size(); j++)
			{
				ReferenceMatcher matcher(bltBlocks[i][j]);

				for (size_t k = 0; k < matcher.size(); k++)
				{
					Variable var = model.getVariable(matcher[k].getExpression().getReferredVectorAccess());
					if (!var.isTrivial() || var.isState())
						continue;

					// If the variable is trivial, replace all occurrences in the non-trivial equation.
					if (trivialVariablesMap.find(var) == trivialVariablesMap.end())
					{
						// If the variable is a constant parameter, substitute the
						// subscription operation with the corresponding constant.
						mlir::Operation* subOp = matcher[k].getExpression().getOp();
						mlir::Operation* loadOp = subOp->getNextNode();

						assert(parametersMap.find(var) != parametersMap.end());
						assert(mlir::isa<SubscriptionOp>(subOp) && subOp->hasOneUse());
						assert(mlir::isa<LoadOp>(loadOp) && loadOp->getNumResults() == 1);

						builder.setInsertionPoint(loadOp);
						mlir::Operation* constantOp = builder.clone(*parametersMap[var]);
						loadOp->getResult(0).replaceAllUsesWith(constantOp->getResult(0));
					}
					else if (trivialVariablesMap[var].size() == 1)
					{
						// If the variable is computed by an equation, use that equation.
						replaceUses(builder, *trivialVariablesMap[var][0], bltBlocks[i][j]);
					}
					else
					{
						// In case a trivial variable is computed by more than one
						// equation depending on the index, we must split the non-trivial
						// equation based on the trivial variable.
						for (Equation* trivialEq : trivialVariablesMap[var])
						{
							// Total dimension of the trivial equation.
							VectorAccess sourceAccess = AccessToVar::fromExp(trivialEq->getMatchedExp()).getAccess();
							MultiDimInterval sourceInductions = sourceAccess.map(trivialEq->getInductions());

							// Dimension of trivial variable in non-trivial equation.
							VectorAccess destinationAccess = AccessToVar::fromExp(matcher[k].getExpression()).getAccess();
							MultiDimInterval destinationInductions = destinationAccess.map(bltBlocks[i][j].getInductions());

							if (marco::areDisjoint(sourceInductions, destinationInductions))
								continue;

							// Compute the new inductions.
							MultiDimInterval usedInductions = marco::intersection(sourceInductions, destinationInductions);
							MultiDimInterval newInductions = bltBlocks[i][j].getInductions();
							for (auto& access : llvm::enumerate(destinationAccess))
							{
								if (access.value().isOffset())
								{
									size_t index = access.value().getInductionVar();
									newInductions = newInductions.replacedDimension(
										index,
										std::max(newInductions.at(index).min(),
											usedInductions.at(access.index()).min() - access.value().getOffset()),
										std::min(newInductions.at(index).max(),
											usedInductions.at(access.index()).max() - access.value().getOffset()));
								}
							}

							// Compose the new equation and add it to the BLT block.
							Equation newEquation = bltBlocks[i][j].clone();
							newEquation.setInductions(newInductions);
							replaceUses(builder, *trivialEq, newEquation);
							bltBlocks[i].insert(j + 1, newEquation);
						}

						// Erase the old equation.
						bltBlocks[i][j].erase();
						bltBlocks[i].erase(j);
					}

					// Then re-start iterating over the same equation
					bltBlocks[i][j].update();
					matcher = ReferenceMatcher(bltBlocks[i][j]);
					k = 0;
				}
			}
		}

		// Clean the equation and fold the constants.
		for (BltBlock& bltBlock : bltBlocks)
			for (Equation& equation : bltBlock.getEquations())
				equation.update();

		Model result(model.getOp(), model.getVariables(), model.getEquations(), bltBlocks);
		model = result;
		return mlir::success();
	}

	static std::string getResidualFunctionName(int64_t num)
	{
		return "residual_function_" + std::to_string(num);
	}

	static std::string getJacobianFunctionName(int64_t num)
	{
		return "jacobian_function_" + std::to_string(num);
	}

	mlir::LogicalResult addIdaSolver(Model& model)
	{
		SimulationOp simulationOp = model.getOp();
		mlir::MLIRContext* context = getOperation().getContext();
		IdaBuilder builder(context);
		YieldOp terminator = mlir::cast<YieldOp>(simulationOp.init().back().getTerminator());
		builder.setInsertionPoint(terminator);
		mlir::Location loc = simulationOp.getLoc();

		//===--------------------------------------------------------------===//
		// IDA INIT
		//===--------------------------------------------------------------===//

		// Allocate IDA user data, starting from the number of scalar equations.
		int64_t equationsNumber = 0;
		for (BltBlock& bltBlock : model.getBltBlocks())
			equationsNumber += bltBlock.equationsCount();

		mlir::Value neq = builder.create<ConstantValueOp>(loc, builder.getIntegerAttribute(equationsNumber));
		mlir::Value userData = builder.create<AllocDataOp>(loc, neq);

		// Add start time, end time and time step.
		mlir::Value startTime = builder.create<ConstantValueOp>(loc, builder.getRealAttribute(simulationOp.startTime().getValue()));
		mlir::Value endTime = builder.create<ConstantValueOp>(loc, builder.getRealAttribute(simulationOp.endTime().getValue()));

		double timeStepValue = options.equidistantTimeGrid ? simulationOp.timeStep().getValue() : -1;
		mlir::Value timeStep = builder.create<ConstantValueOp>(loc, builder.getRealAttribute(timeStepValue));

		builder.create<AddTimeOp>(loc, userData, startTime, endTime, timeStep);

		// Add relative and absolute tolerances.
		mlir::Value relTol = builder.create<ConstantValueOp>(loc, builder.getRealAttribute(simulationOp.relTol().getValue()));
		mlir::Value absTol = builder.create<ConstantValueOp>(loc, builder.getRealAttribute(simulationOp.absTol().getValue()));
		builder.create<AddToleranceOp>(loc, userData, relTol, absTol);

		// Initialize IDA user data.
		int64_t variableCount = 0;
		int64_t variableOffset = 0;
		int64_t equationCount = 0;
		std::map<Variable, int64_t> offsetsMap;
		std::map<Variable, mlir::Value> variableIndexMap;

		// Compute all non-trivial variable offsets and dimensions.
		for (BltBlock& bltBlock : model.getBltBlocks())
		{
			for (Equation& equation : bltBlock.getEquations())
			{
				// Get the variable matched with every equation.
				Variable var = model.getVariable(equation.getDeterminedVariable().getVar());

				assert(!var.isTrivial());

				// If the variable has not been insterted yet, initialize it.
				if (variableIndexMap.find(var) != variableIndexMap.end())
					continue;

				// Initialize variableOffset, variableDimensions, variablesValues, derivativesValues, idValues.
				loc = var.getReference().getLoc();
				mlir::Value varOffsetOp = builder.create<ConstantValueOp>(loc, builder.getIntegerAttribute(variableOffset));
				mlir::Value isState = builder.create<ConstantValueOp>(loc, builder.getBooleanAttribute(var.isState() || var.isDerivative()));
				builder.create<AddVariableOp>(loc, userData, varOffsetOp, var.isDerivative() ? var.getState() : var.getReference(), isState);

				// Store the variable index.
				mlir::Value varIndex = builder.create<ConstantValueOp>(loc, builder.getIntegerAttribute(variableCount++));
				offsetsMap[var] = variableOffset;
				variableIndexMap[var] = varIndex;

				if (var.isState())
				{
					offsetsMap[model.getVariable(var.getDerivative())] = variableOffset;
					variableIndexMap[model.getVariable(var.getDerivative())] = varIndex;
				}
				else if (var.isDerivative())
				{
					offsetsMap[model.getVariable(var.getState())] = variableOffset;
					variableIndexMap[model.getVariable(var.getState())] = varIndex;
				}

				// Increase the length of the current row.
				variableOffset += var.toMultiDimInterval().size();
			}
		}

		// Add to IDA all non-trivial variable accesses and the dimensions of each
		// equation. Then create the two functions that compute the Residual and the
		// Jacobian of each equation and pass the pointer to such functions to IDA.
		for (BltBlock& bltBlock : model.getBltBlocks())
		{
			for (Equation& equation : bltBlock.getEquations())
			{
				loc = equation.getOp().getLoc();
				mlir::Value rowIndex = builder.create<ConstantValueOp>(loc, builder.getIntegerAttribute(equationCount));

				// Variable access.
				ReferenceMatcher matcher(equation);
				for (ExpressionPath& path : matcher)
				{
					Variable var = model.getVariable(path.getExpression().getReferredVectorAccess());

					if (var.isTime())
						continue;

					assert(variableIndexMap.find(var) != variableIndexMap.end());

					// Compute the access offset based on the induction variables of the for-equation.
					VectorAccess vectorAccess = AccessToVar::fromExp(path.getExpression()).getAccess();

					llvm::SmallVector<std::pair<int64_t, int64_t>, 3> access;
					for (auto& acc : vectorAccess.getMappingOffset())
					{
						int64_t accOffset = acc.isDirectAccess() ? acc.getOffset() : (acc.getOffset() + 1);
						int64_t accInduction = acc.isOffset() ? acc.getInductionVar() : -1;
						access.push_back({ accOffset, accInduction });
					}

					// Add accesses of the variable to the ida user data.
					ArrayType arrayType = ArrayType::get(
							context, BufferAllocationScope::stack, modelica::IntegerType::get(context), { (long) access.size() });
					AllocaOp offsets = builder.create<AllocaOp>(loc, arrayType.getElementType(), arrayType.getShape(), llvm::None, true);
					AllocaOp inductions = builder.create<AllocaOp>(loc, arrayType.getElementType(), arrayType.getShape(), llvm::None, true);

					for (size_t i = 0; i < access.size(); i++)
					{
						mlir::Value accIndex = builder.create<ConstantOp>(loc, builder.getIndexAttr(i));
						mlir::Value acc = builder.create<ConstantOp>(loc, modelica::IntegerAttribute::get(context, access[i].first));
						mlir::Value subscriptionOp = builder.create<SubscriptionOp>(loc, offsets, accIndex);
						builder.create<AssignmentOp>(loc, acc, subscriptionOp);

						mlir::Value ind = builder.create<ConstantOp>(loc, modelica::IntegerAttribute::get(context, access[i].second));
						subscriptionOp = builder.create<SubscriptionOp>(loc, inductions, accIndex);
						builder.create<AssignmentOp>(loc, ind, subscriptionOp);
					}

					// Add to IDA the indexes of non-zero values of the current equation.
					mlir::Value accessIndex = builder.create<AddVarAccessOp>(loc, userData, variableIndexMap[var], offsets, inductions);
					builder.create<AddColumnIndexOp>(loc, userData, rowIndex, accessIndex);
				}

				// Dimensions.
				MultiDimInterval inductions = equation.getInductions();

				ArrayType arrayType = ArrayType::get(
						context, BufferAllocationScope::stack, modelica::IntegerType::get(context), { (long) inductions.dimensions() });
				AllocaOp start = builder.create<AllocaOp>(loc, arrayType.getElementType(), arrayType.getShape(), llvm::None, true);
				AllocaOp end = builder.create<AllocaOp>(loc, arrayType.getElementType(), arrayType.getShape(), llvm::None, true);

				for (size_t i = 0; i < inductions.dimensions(); i++)
				{
					mlir::Value dimIndex = builder.create<ConstantOp>(loc, builder.getIndexAttr(i));
					mlir::Value min = builder.create<ConstantOp>(loc, modelica::IntegerAttribute::get(context, inductions[i].min() - 1));
					mlir::Value subscriptionOp = builder.create<SubscriptionOp>(loc, start, dimIndex);
					builder.create<AssignmentOp>(loc, min, subscriptionOp);

					mlir::Value max = builder.create<ConstantOp>(loc, modelica::IntegerAttribute::get(context, inductions[i].max() - 1));
					subscriptionOp = builder.create<SubscriptionOp>(loc, end, dimIndex);
					builder.create<AssignmentOp>(loc, max, subscriptionOp);
				}

				builder.create<AddEqDimensionOp>(loc, userData, start, end);

				// Add Residual and Jacobian function to IDA.
				std::string residualName = getResidualFunctionName(equationCount);
				mlir::Value residualAddress = builder.create<FuncAddressOfOp>(loc, residualName, builder.getResidualFunctionType());
				builder.create<AddResidualOp>(loc, userData, residualAddress);

				std::string jacobianName = getJacobianFunctionName(equationCount++);
				mlir::Value jacobianAddress = builder.create<FuncAddressOfOp>(loc, jacobianName, builder.getJacobianFunctionType());
				builder.create<AddJacobianOp>(loc, userData, jacobianAddress);

				// Create the Residual and Jacobian function.
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPoint(simulationOp);
				builder.create<ResidualFunctionOp>(loc, residualName, model, equation, offsetsMap);
				builder.create<JacobianFunctionOp>(loc, jacobianName, model, equation, offsetsMap);
			}
		}

		offsetsMap.clear();
		assert(variableOffset == equationsNumber);

		loc = simulationOp.getLoc();
		mlir::Value threads = builder.create<ConstantValueOp>(loc, builder.getIntegerAttribute(options.threads));
		builder.create<InitOp>(loc, userData, threads);

		// Add userData inside the returned pointer in second position
		mlir::Value userDataAlloc = builder.create<AllocOp>(loc, builder.getOpaquePointerType(), llvm::None, llvm::None, false, false);
		builder.create<StoreOp>(loc, userData, userDataAlloc);

		llvm::SmallVector<mlir::Value, 3> args = terminator.values();
		args.insert(args.begin() + 1, userDataAlloc);
		YieldOp newTerminator = builder.create<YieldOp>(loc, args);
		terminator->erase();

		//===--------------------------------------------------------------===//
		// IDA STEP
		//===--------------------------------------------------------------===//

		// Add the IDA user data to the argument list of the step block and load it.
		mlir::Block& stepBlock = simulationOp.body().front();
		stepBlock.insertArgument(1, userDataAlloc.getType().cast<ArrayType>().toUnknownAllocationScope());
		builder.setInsertionPointToStart(&stepBlock);
		userData = builder.create<LoadOp>(loc, stepBlock.getArgument(1));

		// Map all variables to the argument they are loaded from.
		mlir::BlockAndValueMapping varArgMap;
		for (const auto& [var, arg] : llvm::zip(newTerminator.values(), stepBlock.getArguments()))
			varArgMap.map(var, arg);

		// Update all non-trivial variables by fetching the correct values from IDA.
		for (auto& variable : model.getVariables())
		{
			if ((variable->isTrivial() && !variable->isState()) || 
					variable->isDerivative() || variable->isTime())
				continue;

			assert(variableIndexMap.find(*variable) != variableIndexMap.end());
			assert(varArgMap.contains(variable->getReference()));

			loc = variable->getReference().getLoc();
			mlir::Value varIndex = builder.clone(*variableIndexMap[*variable].getDefiningOp())->getResult(0);
			builder.create<UpdateVariableOp>(loc, userData, varIndex, varArgMap.lookup(variable->getReference()));

			// If the variable is a state, also update its derivative.
			if (variable->isState())
				builder.create<UpdateDerivativeOp>(loc, userData, varIndex, varArgMap.lookup(variable->getDerivative()));
		}

		variableIndexMap.clear();

		// Delete all equations that were replaced by IDA.
		for (BltBlock& bltBlock : model.getBltBlocks())
			for (Equation& equation : bltBlock.getEquations())
				equation.erase();

		return mlir::success();
	}

	mlir::LogicalResult createSimulationFunctions(mlir::BlockAndValueMapping& derivativesMap)
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<SimulationOp, EquationOp, ForEquationOp, EquationSidesOp>();

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<SimulationOpPattern>(&getContext(), typeConverter, options, derivativesMap);
		patterns.insert<EquationOpPattern, EquationSidesOpPattern>(&getContext());

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
        return status;

		return mlir::success();
	}

	static llvm::Optional<Model> getUnmatchedModel(mlir::ModuleOp moduleOp)
	{
		if (failed(scalarizeArrayEquations(moduleOp)))
			return llvm::None;

		if (failed(loopify(moduleOp)))
			return llvm::None;

		llvm::Optional<Model> model;

		moduleOp->walk([&](SimulationOp simulation) {
			mlir::OpBuilder builder(simulation);

			model = Model::build(simulation);
			mlir::BlockAndValueMapping derivatives;

			if (failed(removeDerivatives(*model, derivatives, moduleOp)))
				model = llvm::None;
		});

		return model;
	}

	static llvm::Optional<Model> getSolvedModel(mlir::ModuleOp moduleOp, SolveModelOptions options)
	{
		llvm::Optional<Model> model = SolveModelPass::getUnmatchedModel(moduleOp);
		if (!model)
			return llvm::None;

		mlir::OpBuilder builder(model->getOp());

		if (failed(match(*model, options.matchingMaxIterations)))
			return llvm::None;

		if (failed(solveSCCs(*model, options.sccMaxIterations)))
			return llvm::None;

		if (failed(explicitateEquations(*model)))
			return llvm::None;

		if (options.solver == CleverDAE)
			if (failed(addDifferentialEqToBltBlocks(*model)))
				return llvm::None;

		if (failed(schedule(*model)))
			return llvm::None;

		if (options.solver == ForwardEuler)
			if (failed(updateStates(*model)))
				return llvm::None;

		if (options.solver == CleverDAE)
			if (failed(substituteTrivialVariables(*model)))
				return llvm::None;

		return model;
	}

	private:
	SolveModelOptions options;
	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createSolveModelPass(SolveModelOptions options, unsigned int bitWidth)
{
	return std::make_unique<SolveModelPass>(options, bitWidth);
}

llvm::Optional<Model> marco::codegen::getUnmatchedModel(mlir::ModuleOp moduleOp)
{
	return SolveModelPass::getUnmatchedModel(moduleOp);
}

llvm::Optional<Model> marco::codegen::getSolvedModel(mlir::ModuleOp moduleOp, SolveModelOptions options)
{
	return SolveModelPass::getSolvedModel(moduleOp, options);
}
