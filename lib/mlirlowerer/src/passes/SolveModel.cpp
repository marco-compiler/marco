#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaBuilder.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <marco/mlirlowerer/passes/SolveModel.h>
#include <marco/mlirlowerer/passes/matching/Matching.h>
#include <marco/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <marco/mlirlowerer/passes/matching/Schedule.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/TypeConverter.h>
#include <unordered_map>
#include <modelica/utils/VariableFilter.h>
#include <queue>

using namespace marco;
using namespace codegen;
using namespace model;
using namespace modelica;


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

				{
					/* Fix print argument TODO
					auto originalArgument = op.print().getArgument(index);
					auto newArgument = op.print().insertArgument(index + 1, value.getType().cast<ArrayType>().toUnknownAllocationScope());
					originalArgument.replaceAllUsesWith(newArgument);
					op.print().eraseArgument(index); AL */
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
		auto result = rewriter.create<SimulationOp>(op->getLoc(),op.variableNames(), op.startTime(), op.endTime(), op.timeStep(), op.body().getArgumentTypes());
		rewriter.mergeBlocks(&op.init().front(), &result.init().front(), result.init().getArguments());
		rewriter.mergeBlocks(&op.body().front(), &result.body().front(), result.body().getArguments());
		/* no yield op
        rewriter.setInsertionPointToStart(&result.body().front());
        rewriter.create<YieldOp>(op->getLoc()); */

		//AL rewriter.mergeBlocks(&op.print().front(), &result.print().front(), result.print().getArguments());

		//rewriter.setInsertionPointToStart(&result.body().front()); (copied upwards)
		//rewriter.create<YieldOp>(op->getLoc()); TODO

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
				auto indexValue = index.value().cast<IntegerAttribute>().getValue();
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
				mlir::Value induction = rewriter.create<InductionOp>(location, 1, size);
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
				mlir::Value induction = rewriter.create<InductionOp>(location, 1, size);
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

//TODO x1
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
			variable.setDer(derVar);


			args.push_back(derVar);
			rewriter.create<YieldOp>(terminator.getLoc(), args);
			rewriter.eraseOp(terminator);

			auto newArgumentType = derVar.getType().cast<ArrayType>().toUnknownAllocationScope();
			auto bodyArgument = simulation.body().addArgument(newArgumentType);



			derivatives->map(operand, bodyArgument);

            //AL simulation.print().addArgument(newArgumentType); TODO

		}
		else
		{
			derVar = variable.getDer();
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
		if (type.isa<BooleanType>())
			return builder.create<ConstantOp>(loc, BooleanAttribute::get(type, false));

		if (type.isa<IntegerType>())
			return builder.create<ConstantOp>(loc, IntegerAttribute::get(type, 0));

		assert(type.isa<RealType>());
		return builder.create<ConstantOp>(loc, RealAttribute::get(type, 0));
	}

	Model* model;
	mlir::BlockAndValueMapping* derivatives;

};

//// PRINT VARIABLES ///
mlir::LLVM::LLVMFuncOp getOrInsertPrintf(mlir::OpBuilder& rewriter, mlir::ModuleOp module)
{
    auto *context = module.getContext();

    if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
        return foo;

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    return rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
}

mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder& builder, mlir::StringRef name, mlir::StringRef value, mlir::ModuleOp module)
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

bool performRangeBoundCheck(ArrayType array, VariableFilter filter, string name) {

    std::cout << "Checking print ranges for variable " << name << std::endl;
    //for each array dimension
    for (unsigned int i = 0, e = array.getRank(); i < e; ++i) {
        std::cout << "\tdimension #" << i;
        //get the number of elements, 'length' of the dimension
        auto shapeOfYou = array.getShape()[i];
        std::cout << " is made of " << shapeOfYou <<  " elements.\n";

        Range dimensionRange = filter.lookupByIdentifier(name).getRangeOfDimensionN(i);
        if (dimensionRange.rightValue > shapeOfYou ||  //if the specified range has a bound that it's bigger than the array dimension
            dimensionRange.leftValue  > shapeOfYou) {
            return false;
        }
    }
    return true;
}

/////

struct SimulationOpPattern : public mlir::OpConversionPattern<SimulationOp>
{
    //Constructor
	SimulationOpPattern(mlir::MLIRContext* ctx,
											TypeConverter& typeConverter,
											SolveModelOptions options, mlir::BlockAndValueMapping derVarMapping)
			: mlir::OpConversionPattern<SimulationOp>(typeConverter, ctx, 1),
				options(std::move(options)), derivatives(derVarMapping)
	{
	}

    [[nodiscard]] mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
    {
        mlir::Type type = this->getTypeConverter()->convertType(value.getType());
        return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
    }

    void printElement(mlir::OpBuilder& builder, mlir::Value value, mlir::Value printSeparator, mlir::Value separator, mlir::ModuleOp module) const
    {
        auto printfRef = getOrInsertPrintf(builder, module);

        mlir::Type type = value.getType();

        // Check if the separator should be printed
        mlir::Value shouldPrintSeparator = builder.create<LoadOp>(printSeparator.getLoc(), printSeparator);
        shouldPrintSeparator = materializeTargetConversion(builder, shouldPrintSeparator);
        auto ifOp = builder.create<mlir::scf::IfOp>(value.getLoc(), shouldPrintSeparator);
        builder.setInsertionPointToStart(ifOp.getBody());
        builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, separator);
        builder.setInsertionPointAfter(ifOp);

        mlir::Value formatSpecifier;

        if (type.isa<mlir::IntegerType>())
            formatSpecifier = getOrCreateGlobalString(value.getLoc(), builder, "frmt_spec_int", mlir::StringRef("%ld\0", 4), module);
        else if (type.isa<mlir::FloatType>())
            formatSpecifier = getOrCreateGlobalString(value.getLoc(), builder, "frmt_spec_float", mlir::StringRef("%.12f\0", 6), module);
        else
            assert(false && "Unknown type");

        builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, mlir::ValueRange({ formatSpecifier, value }));

        // Set the separator as to be printed before the next value
        mlir::Value trueValue = builder.create<ConstantOp>(value.getLoc(), BooleanAttribute::get(BooleanType::get(builder.getContext()), true));
        builder.create<StoreOp>(value.getLoc(), trueValue, printSeparator);
    }

	mlir::LogicalResult matchAndRewrite(SimulationOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc(); //In MLIR, every operation has a mandatory source location associated with it

		llvm::SmallVector<mlir::Type, 3> varTypes;

		{
			auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());
			varTypes.push_back(terminator.values()[0].getType().cast<ArrayType>().toUnknownAllocationScope());

			// Add the time step as second argument
			varTypes.push_back(op.timeStep().getType());

			for (auto it = ++terminator.values().begin(); it != terminator.values().end(); ++it)
				varTypes.push_back((*it).getType().cast<ArrayType>().toUnknownAllocationScope());
		}

        mlir::ArrayRef<mlir::Attribute> variableNamesAttributes = op.variableNames().getValue();

	    std::unordered_map<std::string, bool> hasDerivativeMap;

	    int a = 1; //skipping 'time' variable

	    //for each variable of the model, check if it has a derivative too and create a map with key variableName
        for (const mlir::Attribute &item : variableNamesAttributes) {
            std::string variableIdentifier = item.cast<mlir::StringAttr>().getValue().str();
            mlir::Value bodyValue = op.body().getArgument(a++); //then increment a
            bool hasDer = derivatives.contains(bodyValue); //check in the value / derivative map if there is an entry
            //std::cout << variableIdentifier << " has der? " << hasDer << std::endl;
            hasDerivativeMap.insert_or_assign(variableIdentifier, hasDer);
        }

		auto structType = StructType::get(op->getContext(), varTypes);
		auto structPtrType = ArrayType::get(structType.getContext(), BufferAllocationScope::unknown, structType);
		auto opaquePtrType = OpaquePointerType::get(structPtrType.getContext());

		{
			// Init function
			auto functionType = rewriter.getFunctionType(llvm::None, opaquePtrType);
			auto function = rewriter.create<mlir::FuncOp>(loc, "init", functionType);
			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			rewriter.mergeBlocks(&op.init().front(), &function.body().front(), llvm::None);

			llvm::SmallVector<mlir::Value, 3> values;
			auto terminator = mlir::cast<YieldOp>(entryBlock->getTerminator());

			auto removeAllocationScopeFn = [&](mlir::Value value) -> mlir::Value {
				return rewriter.create<ArrayCastOp>(
						loc, value,
						value.getType().cast<ArrayType>().toUnknownAllocationScope());
			};

			// Time variable
			mlir::Value time = terminator.values()[0];
			values.push_back(removeAllocationScopeFn(time));

			// Time step
			mlir::Value timeStep = rewriter.create<ConstantOp>(loc, op.timeStep());
			values.push_back(timeStep);

			// Add the remaining variables to the struct. Time and time step
			// variables have already been managed, but only the time one was in the
			// yield operation, so we need to start from its second argument.

			for (auto it = std::next(terminator.values().begin()); it != terminator.values().end(); ++it)
				values.push_back(removeAllocationScopeFn(*it));

			// Set the start time
			mlir::Value startTime = rewriter.create<ConstantOp>(loc, op.startTime());
			//startTime = rewriter.create<SubOp>(loc, startTime.getType(), startTime, timeStep);
			rewriter.create<StoreOp>(loc, startTime, values[0]);

			mlir::Value structValue = rewriter.create<PackOp>(terminator->getLoc(), values);
			mlir::Value result = rewriter.create<AllocOp>(structValue.getLoc(), structType, llvm::None, llvm::None, false);
			rewriter.create<StoreOp>(result.getLoc(), structValue, result);
			result = rewriter.create<ArrayCastOp>(result.getLoc(), result, opaquePtrType);

			rewriter.replaceOpWithNewOp<mlir::ReturnOp>(terminator, result);
		}

		{
			// Step function
			auto function = rewriter.create<mlir::FuncOp>(
					loc, "step",
					rewriter.getFunctionType(opaquePtrType, rewriter.getI1Type()));

			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			mlir::Value structValue = loadDataFromOpaquePtr(rewriter, loc, function.getArgument(0), structType);

			llvm::SmallVector<mlir::Value, 3> args;
			args.push_back(rewriter.create<ExtractOp>(loc, varTypes[0], structValue, 0));

			for (size_t i = 2, e = structType.getElementTypes().size(); i < e; ++i)
				args.push_back(rewriter.create<ExtractOp>(loc, varTypes[i], structValue, i));

			mlir::Value timeStep = rewriter.create<ExtractOp>(
					loc, op.timeStep().getType(), structValue, 1);

			{
				// Increment the time
				mlir::Value currentTime = rewriter.create<LoadOp>(loc, args[0]);

				mlir::Value increasedTime = rewriter.create<AddOp>(
						loc, currentTime.getType(), currentTime, timeStep);

				rewriter.create<StoreOp>(loc, increasedTime, args[0]);
			}

			// Check if the current time is less than the end time
			mlir::Value currentTime = rewriter.create<LoadOp>(loc, args[0]);
			mlir::Value endTime = rewriter.create<ConstantOp>(loc, op.endTime());
			endTime = rewriter.create<AddOp>(loc, endTime.getType(), endTime, timeStep);

			mlir::Value condition = rewriter.create<LtOp>(
					loc, BooleanType::get(op->getContext()), currentTime, endTime);

			condition = getTypeConverter()->materializeTargetConversion(
					rewriter, condition.getLoc(), rewriter.getI1Type(), condition);

			auto ifOp = rewriter.create<mlir::scf::IfOp>(
					loc, rewriter.getI1Type(), condition, true);

			{
				// If we didn't reach the end time update the variables and return
				// true to continue the simulation.
				mlir::OpBuilder::InsertionGuard g(rewriter);
				rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());

				mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(
						loc, rewriter.getBoolAttr(true));

				auto terminator = rewriter.create<mlir::scf::YieldOp>(loc, trueValue);

				rewriter.eraseOp(op.body().front().getTerminator());
				rewriter.mergeBlockBefore(&op.body().front(), terminator, args);
			}

			{
				// Otherwise, return false to stop the simulation
				mlir::OpBuilder::InsertionGuard g(rewriter);
				rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());

				mlir::Value falseValue = rewriter.create<mlir::ConstantOp>(
						loc, rewriter.getBoolAttr(false));

				rewriter.create<mlir::scf::YieldOp>(loc, falseValue);
			}

			rewriter.create<mlir::ReturnOp>(loc, ifOp.getResult(0));
		}

		{
			// Print function
			auto function = rewriter.create<mlir::FuncOp>( //given the insertion point create an Operation of type FuncOp
					loc, "print",
					rewriter.getFunctionType(opaquePtrType, llvm::None));

			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			//Loaded with values from 'Main'
			mlir::Value structValue = loadDataFromOpaquePtr(rewriter, loc, function.getArgument(0), structType);

            llvm::SmallVector<mlir::Value, 3> valuesToBePrinted;
            llvm::SmallVector<mlir::Value, 3> derivativesValues;

			//RETRIEVE THE NAMES

			modelica::VariableFilter variableFilter = *options.variableFilter; // get variable filter state
			variableFilter.dump();

            mlir::ArrayRef<mlir::Attribute> variableNames = op.variableNames().getValue();
            std::vector<std::string> variableNamesVector;
            for (const mlir::Attribute &item : variableNames) {
                auto stringAttribute = item.cast<mlir::StringAttr>();
                std::string variableIdentifier = stringAttribute.getValue().str();
                variableNamesVector.emplace_back(variableIdentifier);
            }


            //all variables

            mlir::Value time = rewriter.create<ExtractOp>(loc, varTypes[0], structValue, 0);
            valuesToBePrinted.push_back(time); //0.00 0.10 0.20 and so on

            std::map<std::string, mlir::Value> nameValueMap;

            //Values are in the struct, let's fetch them

            size_t NUM_VAR = variableNamesVector.size();
            for (size_t i = 2; i<structType.getElementTypes().size(); ++i) {

                if (i-2 < NUM_VAR) {
                    std::string name = variableNamesVector[i-2];
                    mlir::Value extracted = rewriter.create<ExtractOp>(loc, varTypes[i], structValue, i);
                    nameValueMap.insert(std::pair<std::string,mlir::Value>(name,extracted)); //match a variable name with corresponding mlir::Value
                    valuesToBePrinted.push_back(extracted); // for now: print only variables (not derivatives)

                }
                else /* derivatives */ {
                    mlir::Value derivativeExtractedValue = rewriter.create<ExtractOp>(loc, varTypes[i], structValue, i);
                    derivativesValues.push_back(derivativeExtractedValue);

                }

            }

            //rewriter.create<PrintOp>(loc, valuesToBePrinted);

            auto module = op->getParentOfType<mlir::ModuleOp>();

            auto printfRef = getOrInsertPrintf(rewriter, module);
            mlir::Value semicolonCst = getOrCreateGlobalString(loc, rewriter, "semicolon", mlir::StringRef(";\0", 2), module);
            mlir::Value newLineCst = getOrCreateGlobalString(loc, rewriter, "newline", mlir::StringRef("\n\0", 2), module);

            mlir::Value printSeparator = rewriter.create<AllocaOp>(loc, BooleanType::get(op->getContext()));
            mlir::Value falseValue = rewriter.create<ConstantOp>(loc, BooleanAttribute::get(BooleanType::get(op->getContext()), false));
            rewriter.create<StoreOp>(loc, falseValue, printSeparator);

            int cur = 0; //auxiliary variable to keep the index of the currently printed variable name

            std::cout << "**** 🖨 GENERATING PRINT CODE **** " << std::endl;
            //for each Value to be printed (with bypass activated means 'time' + all variables)
            for (auto var : valuesToBePrinted) {
                    //if current value is an array
                    if (auto arrayType = var.getType().dyn_cast<ArrayType>())
                    {
                        //get the name of the variable the value belongs too
                        std::string varName = variableNamesVector[cur];
                        unsigned int rank = arrayType.getRank();

                        //Arrays of Rank > 0, 1D,2D,3D.. Arrays
                        if (rank > 0) {

                            mlir::Location varLoc = var.getLoc();
                            auto arrayTypeLocal = var.getType().cast<ArrayType>();

                            mlir::Value zero = rewriter.create<mlir::ConstantOp>(varLoc, rewriter.getIndexAttr(0));
                            mlir::Value one = rewriter.create<mlir::ConstantOp>(varLoc, rewriter.getIndexAttr(1));

                            //no filtering or no filter or specified to print by CL args
                            if (variableFilter.isBypass() || variableFilter.checkTrackedIdentifier(varName) ) {



                                //check if custom ranges are specified, boolean 'isArray' e.g.
                                    // x prints all the array with all its dimensions from 0 to len(dim) (is Array = false)
                                    // x[$:3,1:50] prints custom ranges for dimensions 0 and 1 (all of them)
                                bool fullRange = variableFilter.isBypass() || !variableFilter.lookupByIdentifier(varName).getIsArray(); // only the name has been provided
                                std::cout << "printing " << varName << " - fullRange=" << fullRange <<std::endl;
                                //Upper and lower bound vectors (for each iteration of the loop, from LB to UB)
                                llvm::SmallVector<mlir::Value, 3> lowerBounds; //starting from position zero
                                llvm::SmallVector<mlir::Value, 3> upperBounds;

                                //specified ranges must match the number of dimension and be 'feasible'
                                bool rangesAreOk = fullRange || performRangeBoundCheck(arrayTypeLocal, variableFilter, varName); //checks if provided bounds are contained in array dimensions
                                if (!rangesAreOk) assert(false); //please use the variable filter correctly

                                //for each of the array dimensions (rank)
                                for (unsigned int i = 0, e = arrayTypeLocal.getRank(); i < e; ++i) {

                                    Range dimensionRange(-1,-1); //default value

                                    //get specified range for dimension 'i' if is specified and it's not full range
                                    if(!(variableFilter.isBypass() || fullRange)) {
                                        VariableTracker currentTracker = variableFilter.lookupByIdentifier(varName);
                                        dimensionRange = currentTracker.getRangeOfDimensionN(i);
                                    }

                                    /// ============ FIX LOWER BOUNDs ============== //
                                    //print the full dimension, no upper bound specified, starting from [0] element
                                    if(variableFilter.isBypass() || fullRange || dimensionRange.noLowerBound()) {
                                        lowerBounds.push_back(zero); // start from the beginning of the array
                                    }
                                    //else if a bound of the dimension is specified es. "starting from the fourth element"
                                    else {
                                        mlir::Value dim = rewriter.create<mlir::ConstantOp>(varLoc, rewriter.getIndexAttr((int)dimensionRange.leftValue));
                                        lowerBounds.push_back(dim);
                                    }

                                    /// ============ FIX UPPER BOUNDs ============== //
                                    //print the full dimension, no upper bound specified
                                    if(variableFilter.isBypass() || fullRange || dimensionRange.noUpperBound()) {
                                        mlir::Value dim = rewriter.create<mlir::ConstantOp>(varLoc, rewriter.getIndexAttr(i));
                                        upperBounds.push_back(rewriter.create<DimOp>(varLoc, var, dim));
                                    }
                                    else { //else if a bound of the dimension is specified es. "until the third element"
                                        mlir::Value dim = rewriter.create<mlir::ConstantOp>(varLoc, rewriter.getIndexAttr((int)dimensionRange.rightValue));
                                        upperBounds.push_back(dim);
                                    }

                                }

                                //step e.g. what in C++ is "i++"
                                llvm::SmallVector<mlir::Value, 3> steps(arrayTypeLocal.getRank(), one);

                                mlir::scf::buildLoopNest(
                                        rewriter, varLoc, lowerBounds, upperBounds, steps, llvm::None,
                                        [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position, mlir::ValueRange args) -> std::vector<mlir::Value> {
                                            //print element at 'position'
                                            mlir::Value value = rewriter.create<LoadOp>(var.getLoc(), var, position);
                                            value = materializeTargetConversion(rewriter, value);
                                            printElement(rewriter, value, printSeparator, semicolonCst, module);
                                            return std::vector<mlir::Value>();
                                        });

                            } //if variable "is tracked by VF"
                            cur = cur + 1; //keep the match between variable value and variable name consistent
                        }
                        else { //Arrays of Rank = 0
                            std::cout << "printing time" << std::endl;
                            mlir::Value value = rewriter.create<LoadOp>(var.getLoc(), var);
                            value = materializeTargetConversion(rewriter, value);
                            printElement(rewriter, value, printSeparator, semicolonCst, module);
                        }
                    } else //non arrays
                    {
                        std::cout << "printing a non array value" << std::endl;
                        printElement(rewriter, var, printSeparator, semicolonCst, module);
                    }

            }

            std::cout << "**** 🖨 GENERATING DER PRINT CODE **** " << std::endl;

            //add the derivatives to a 'Queue', FIFO Policy is needed for coherency
            std::queue<mlir::Value> derValueQueue;
            for (auto var : derivativesValues) {
                derValueQueue.emplace(var);
            }

            //After the variables have been printed, start to print derivatives
            for (std::string name : variableNamesVector) {

                //for each variable check if it has derivative in the model
                auto got = hasDerivativeMap.find(name);
                if (!(got==hasDerivativeMap.end()) && got->second) { //if it has derivative
                    //where is the derivative value placed? After the variables
                    //pop the value from the queue
                    mlir::Value derivativeValue = derValueQueue.front();
                    derValueQueue.pop(); //remove it from the queue, First come, first served!

                    //now check if the VF is filtering it or if it's not filtering at all
                    if(variableFilter.isBypass() || variableFilter.printDerivative(name)) {
                        //Derivative of variable 'name' must be printed, generate the print code
                        //if current value is an array
                        std::cout << "printing in full derivative of var " << name << std::endl;
                        if (auto arrayType = derivativeValue.getType().dyn_cast<ArrayType>())
                        {
                            //get the name of the variable the value belongs too
                            std::string varName = variableNamesVector[cur];
                            unsigned int rank = arrayType.getRank();

                            //Arrays of Rank > 0, 1D,2D,3D.. Arrays
                            if (rank > 0) {

                                mlir::Location varLoc = derivativeValue.getLoc();
                                auto arrayTypeLocal = derivativeValue.getType().cast<ArrayType>();

                                mlir::Value zero = rewriter.create<mlir::ConstantOp>(varLoc, rewriter.getIndexAttr(0));
                                mlir::Value one = rewriter.create<mlir::ConstantOp>(varLoc, rewriter.getIndexAttr(1));


                                //Upper and lower bound vectors (for each iteration of the loop, from LB to UB)
                                llvm::SmallVector<mlir::Value, 3> lowerBounds; //starting from position zero
                                llvm::SmallVector<mlir::Value, 3> upperBounds;


                                //for each of the array dimensions (rank)
                                for (unsigned int i = 0, e = arrayTypeLocal.getRank(); i < e; ++i) {

                                    Range dimensionRange(-1, -1); //default value

                                    /// ============ FIX LOWER BOUNDs ============== //
                                    lowerBounds.push_back(zero); // start from the beginning of the array
                                    /// ============ FIX UPPER BOUNDs ============== //
                                    mlir::Value dim = rewriter.create<mlir::ConstantOp>(varLoc,
                                                                                        rewriter.getIndexAttr(i));
                                    upperBounds.push_back(rewriter.create<DimOp>(varLoc, derivativeValue, dim));
                                }
                                    //step e.g. what in C++ is "i++"
                                llvm::SmallVector<mlir::Value, 3> steps(arrayTypeLocal.getRank(), one);

                                mlir::scf::buildLoopNest(
                                            rewriter, varLoc, lowerBounds, upperBounds, steps, llvm::None,
                                            [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position, mlir::ValueRange args) -> std::vector<mlir::Value> {
                                                //print element at 'position'
                                                mlir::Value value = rewriter.create<LoadOp>(derivativeValue.getLoc(), derivativeValue, position);
                                                value = materializeTargetConversion(rewriter, value);
                                                printElement(rewriter, value, printSeparator, semicolonCst, module);
                                                return std::vector<mlir::Value>();
                                            });

                                cur = cur + 1; //keep the match between variable value and variable name consistent
                            }
                            else { //Arrays of Rank = 0
                                mlir::Value value = rewriter.create<LoadOp>(derivativeValue.getLoc(), derivativeValue);
                                value = materializeTargetConversion(rewriter, value);
                                printElement(rewriter, value, printSeparator, semicolonCst, module);
                            }
                        }
                    }//prints derivativeValue


                }

            }

            rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), printfRef, newLineCst);

            rewriter.create<mlir::ReturnOp>(loc);


		}

		if (options.emitMain)
		{
			// The main function takes care of running the simulation loop. More
			// precisely, it first calls the "init" function, and then keeps
			// running the updates until the step function return the stop
			// condition (that is, a false value). After each step, it also
			// prints the values and increments the time.

			llvm::SmallVector<mlir::Type, 3> argsTypes;
			llvm::SmallVector<mlir::Type, 3> resultsTypes;

			argsTypes.push_back(rewriter.getI32Type());
			argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(rewriter.getIntegerType(8))));
			resultsTypes.push_back(rewriter.getI32Type());

			auto function = rewriter.create<mlir::FuncOp>(
					loc, "main", rewriter.getFunctionType(argsTypes, resultsTypes));

			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			// Initialize the variables
			mlir::Value data = rewriter.create<mlir::CallOp>(loc, "init", opaquePtrType, llvm::None).getResult(0);
			rewriter.create<mlir::CallOp>(loc, "print", llvm::None, data);

			// Create the simulation loop
			auto loop = rewriter.create<mlir::scf::WhileOp>(loc, llvm::None, llvm::None);

			{
				mlir::OpBuilder::InsertionGuard g(rewriter);

				mlir::Block* conditionBlock = rewriter.createBlock(&loop.before());
				rewriter.setInsertionPointToStart(conditionBlock);
				mlir::Value shouldContinue = rewriter.create<mlir::CallOp>(loc, "step", rewriter.getI1Type(), data).getResult(0);
				rewriter.create<mlir::scf::ConditionOp>(loc, shouldContinue, llvm::None);

				// The body contains just the print call, because the update is
				// already done by the "step "function in the condition region.

				mlir::Block* bodyBlock = rewriter.createBlock(&loop.after());
				rewriter.setInsertionPointToStart(bodyBlock);
				rewriter.create<mlir::CallOp>(loc, "print", llvm::None, data);
				rewriter.create<mlir::scf::YieldOp>(loc);
			}

			mlir::Value returnValue = rewriter.create<mlir::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
			rewriter.create<mlir::ReturnOp>(loc, returnValue);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}

	private:
	static BooleanAttribute getBooleanAttribute(mlir::MLIRContext* context, bool value)
	{
		mlir::Type booleanType = BooleanType::get(context);
		return BooleanAttribute::get(booleanType, value);
	}

	static mlir::Value loadDataFromOpaquePtr(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value ptr, StructType structType)
	{
		assert(ptr.getType().isa<OpaquePointerType>());
		mlir::Type structPtrType = ArrayType::get(structType.getContext(), BufferAllocationScope::unknown, structType);
		mlir::Value castedPtr = builder.create<ArrayCastOp>(loc, ptr, structPtrType);
		return builder.create<LoadOp>(loc, castedPtr);
	}

	mlir::BlockAndValueMapping derivatives;
	SolveModelOptions options;
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

		mlir::scf::buildLoopNest(
				rewriter, op->getLoc(), lowerBounds, upperBounds, steps, llvm::None,
				[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position, mlir::ValueRange args) -> std::vector<mlir::Value> {
					mlir::BlockAndValueMapping mapping;

					for (auto [oldInduction, newInduction] : llvm::zip(op.body()->getArguments(), position))
						mapping.map(oldInduction, newInduction);

					for (auto& sourceOp : op.body()->getOperations())
					{
						if (auto sides = mlir::dyn_cast<EquationSidesOp>(sourceOp))
						{
							// Create the assignments
							for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
							{
								mlir::Value value = mapping.contains(rhs) ? mapping.lookup(rhs) : rhs;

								if (auto loadOp = mlir::dyn_cast<LoadOp>(lhs.getDefiningOp()))
								{
									assert(loadOp.indexes().empty());
									mlir::Value destination = mapping.contains(loadOp.memory()) ? mapping.lookup(loadOp.memory()) : loadOp.memory();
									rewriter.create<AssignmentOp>(sides.getLoc(), value, destination);
								}
								else
								{
									mlir::Value destination = mapping.contains(lhs) ? mapping.lookup(lhs) : lhs;
									rewriter.create<AssignmentOp>(sides->getLoc(), value, destination);
								}
							}
						}
						else
						{
							rewriter.clone(sourceOp, mapping);
						}
					}

					return std::vector<mlir::Value>();
				});

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
			: options(std::move(options)), bitWidth(std::move(bitWidth))
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::scf::SCFDialect>();
	}

	void runOnOperation() override
	{
		// Scalarize the equations consisting in array assignments, by adding
		// the required inductions.
		if (failed(scalarizeArrayEquations()))
			return signalPassFailure();

		// Convert the scalar values into arrays of one element
		if (failed(loopify())) //TODO FIX
			return signalPassFailure();

        mlir::BlockAndValueMapping derivatives;


		getOperation()->walk([&](SimulationOp simulation) {
			mlir::OpBuilder builder(simulation);

			// Create the model
			Model model = Model::build(simulation);


			// Remove the derivative operations and allocate the appropriate buffers
			if (failed(removeDerivatives(builder, model, derivatives))) //TODO FIX
				return signalPassFailure();

			// Match
			if (failed(match(model, options.matchingMaxIterations)))
				return signalPassFailure();

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
		});

		// The model has been solved and we can now proceed to create the update
		// functions and, if requested, the main simulation loop.
		if (auto status = createSimulationFunctions(derivatives); failed(status)) //TODO FIX
			return signalPassFailure();
	}

	/**
	 * Convert the scalar variables to arrays with just one element and the
	 * scalar equations to for equations with a one-sized induction. This
	 * allows the subsequent transformation to ignore whether it was a scalar
	 * or a loop equation, and a later optimization pass can easily remove
	 * the new useless induction.
	 */
	mlir::LogicalResult loopify()
	{
		mlir::ConversionTarget target(getContext());
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

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<SimulationOpLoopifyPattern, EquationOpLoopifyPattern>(&getContext());
		patterns.insert<SimulationOpLoopifyPattern>(&getContext());

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	/**
	 * If an equation consists in an assignment between two arrays, then
	 * convert it into a for equation, in order to scalarize the assignments.
	 */
	mlir::LogicalResult scalarizeArrayEquations()
	{
		mlir::ConversionTarget target(getContext());
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

		mlir::OwningRewritePatternList patterns(&getContext());

		patterns.insert<
		    EquationOpScalarizePattern,
				ForEquationOpScalarizePattern>(&getContext());

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	/**
	 * Remove the derivative operations by replacing them with appropriate
	 * buffers, and set the derived variables as state variables.
	 *
	 * @param builder operation builder
	 * @param model   model
	 * @return conversion result
	 */
	mlir::LogicalResult removeDerivatives(mlir::OpBuilder& builder, Model& model, mlir::BlockAndValueMapping& derivatives)
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<DerOp>();

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<DerOpPattern>(&getContext(), model, derivatives);

		if (auto status = applyPartialConversion(model.getOp(), target, std::move(patterns)); failed(status))
			return status;

		model.reloadIR();
		return mlir::success();
	}

	mlir::LogicalResult explicitateEquations(Model& model)
	{
		for (auto& equation : model.getEquations())
			if (auto status = equation.explicitate(); failed(status))
				return status;

		return mlir::success();
	}

	mlir::LogicalResult selectSolver(mlir::OpBuilder& builder, Model& model)
	{
		if (options.solver == ForwardEuler)
			return updateStates(builder, model);

		if (options.solver == CleverDAE)
			return addBltBlocks(builder, model);

		return mlir::failure();
	}

	/**
	 * Calculate the values that the state variables will have in the next
	 * iteration.
	 */
	mlir::LogicalResult updateStates(mlir::OpBuilder& builder, Model& model)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
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
			mlir::Value der = variable->getDer();

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

	/**
	 * This method transforms all differential equations and implicit equations
	 * into BLT blocks within the model. Then the assigned model, ready for
	 * lowering, is returned.
	 */
	mlir::LogicalResult addBltBlocks(mlir::OpBuilder& builder, Model& model)
	{
		assert(false && "To be implemented");
	}

	mlir::LogicalResult createSimulationFunctions(mlir::BlockAndValueMapping derVarMapping)
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<SimulationOp, EquationOp, ForEquationOp>();

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<SimulationOpPattern>(&getContext(), typeConverter, options, derVarMapping);
		patterns.insert<EquationOpPattern, ForEquationOpPattern>(&getContext());

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
