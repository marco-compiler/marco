#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "marco/ast/AST.h"
#include "marco/ast/passes/ConstantFoldingPass.h"

using namespace marco::ast;

template<>
llvm::Error ConstantFolder::run<Class>(Class& cls)
{
	return cls.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(cls);
	});
}

llvm::Error ConstantFolder::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
{
	for (const auto& cls : classes)
		if (auto error = run<Class>(*cls); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<PartialDerFunction>(Class& cls)
{
	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<StandardFunction>(Class& cls)
{
	auto* function = cls.get<StandardFunction>();
	SymbolTableScope varScope(symbolTable);

	// Populate the symbol table
	symbolTable.insert(function->getName(), Symbol(cls));

	for (auto& member : function->getMembers())
		if (auto error = run(*member); error)
			return error;

	for (auto& algorithm : function->getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Model>(Class& cls)
{
	auto* model = cls.get<Model>();
	SymbolTableScope varScope(symbolTable);

	symbolTable.insert(model->getName(), Symbol(cls));

	for (auto& member : model->getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	for (auto& member : model->getMembers())
		if (auto error = run(*member); error)
			return error;

	for (auto& equation : model->getEquations())
		if (auto error = run(*equation); error)
			return error;

	for (auto& forEquation : model->getForEquations())
		if (auto error = run(*forEquation); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Package>(Class& cls)
{
	auto* package = cls.get<Package>();

	for (auto& innerClass : *package)
		if (auto error = run<Class>(*innerClass); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Record>(Class& cls)
{
	return llvm::Error::success();
}

llvm::Error ConstantFolder::run(Equation& equation)
{
	if (auto error = run<Expression>(*equation.getLhsExpression()); error)
		return error;

	if (auto error = run<Expression>(*equation.getRhsExpression()); error)
		return error;

	return llvm::Error::success();
}

llvm::Error ConstantFolder::run(ForEquation& forEquation)
{
	SymbolTableScope varScope(symbolTable);

	for (auto& ind : forEquation.getInductions())
	{
		symbolTable.insert(ind->getName(), Symbol(*ind));

		if (auto error = run<Expression>(*ind->getBegin()); error)
			return error;

		if (auto error = run<Expression>(*ind->getEnd()); error)
			return error;
	}

	if (auto error = run(*forEquation.getEquation()); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Expression>(Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(expression);
	});
}

template<>
llvm::Error ConstantFolder::run<Array>(Expression& expression)
{
	auto* array = expression.get<Array>();

	for (auto& element : *array)
		if (auto error = run<Expression>(*element); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Call>(Expression& expression)
{
	auto* call = expression.get<Call>();

	if (auto error = run<Expression>(*call->getFunction()); error)
		return error;

	for (auto& arg : *call)
		if (auto error = run<Expression>(*arg); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Constant>(Expression& expression)
{
	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Operation>(Expression& expression)
{
	auto* operation = expression.get<Operation>();

	auto foldOperation = [&](Expression& expression, std::function<llvm::Error(ConstantFolder&, Expression&)> folder) -> llvm::Error {
		if (auto error = folder(*this, expression); error)
			return error;

		return llvm::Error::success();
	};

	switch (operation->getOperationKind())
	{
		case OperationKind::add:
			return foldOperation(expression, &ConstantFolder::foldAddOp);

		case OperationKind::different:
			return foldOperation(expression, &ConstantFolder::foldDifferentOp);

		case OperationKind::divide:
			return foldOperation(expression, &ConstantFolder::foldDivOp);

		case OperationKind::equal:
			return foldOperation(expression, &ConstantFolder::foldEqualOp);

		case OperationKind::greater:
			return foldOperation(expression, &ConstantFolder::foldGreaterOp);

		case OperationKind::greaterEqual:
			return foldOperation(expression, &ConstantFolder::foldGreaterEqualOp);

		case OperationKind::ifelse:
			return foldOperation(expression, &ConstantFolder::foldIfElseOp);

		case OperationKind::less:
			return foldOperation(expression, &ConstantFolder::foldLessOp);

		case OperationKind::lessEqual:
			return foldOperation(expression, &ConstantFolder::foldLessEqualOp);

		case OperationKind::land:
			return foldOperation(expression, &ConstantFolder::foldLogicalAndOp);

		case OperationKind::lor:
			return foldOperation(expression, &ConstantFolder::foldLogicalOrOp);

		case OperationKind::memberLookup:
			return foldOperation(expression, &ConstantFolder::foldMemberLookupOp);

		case OperationKind::multiply:
			return foldOperation(expression, &ConstantFolder::foldMulOp);

		case OperationKind::negate:
			return foldOperation(expression, &ConstantFolder::foldNegateOp);

		case OperationKind::powerOf:
			return foldOperation(expression, &ConstantFolder::foldPowerOfOp);

		case OperationKind::subscription:
			return foldOperation(expression, &ConstantFolder::foldSubscriptionOp);

		case OperationKind::subtract:
			return foldOperation(expression, &ConstantFolder::foldSubOp);
	}

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<ReferenceAccess>(Expression& expression)
{
	auto* reference = expression.get<ReferenceAccess>();

	if (symbolTable.count(reference->getName()) == 0)
	{
		// Built-in variables (such as time) or functions are not in the symbol
		// table.
		return llvm::Error::success();
	}

	const auto& symbol = symbolTable.lookup(reference->getName());

	if (!symbol.isa<Member>())
		return llvm::Error::success();

  // Try to fold references of known variables that have a initializer
	const auto* member = symbol.get<Member>();

	if (!member->hasInitializer())
		return llvm::Error::success();

	if (member->getInitializer()->isa<Constant>() && member->isParameter())
		expression = *member->getInitializer();

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Tuple>(Expression& expression)
{
	auto* tuple = expression.get<Tuple>();

	for (auto& element : *tuple)
		if (auto error = run<Expression>(*element); error)
			return error;

	return llvm::Error::success();
}

llvm::Error ConstantFolder::run(Member& member)
{
	if (member.hasInitializer())
		if (auto error = run<Expression>(*member.getInitializer()); error)
			return error;

	if (member.hasStartOverload())
		return run<Expression>(*member.getStartOverload());

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<Statement>(Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(statement);
	});
}

template<>
llvm::Error ConstantFolder::run<AssignmentStatement>(Statement& statement)
{
	auto* assignmentStatement = statement.get<AssignmentStatement>();

	if (auto error = run<Expression>(*assignmentStatement->getDestinations()); error)
		return error;

	if (auto error = run<Expression>(*assignmentStatement->getExpression()); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<BreakStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<ForStatement>(Statement& statement)
{
	auto* forStatement = statement.get<ForStatement>();

	if (auto error = run(*forStatement->getInduction()); error)
		return error;

	for (auto& stmnt : forStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error ConstantFolder::run(Induction& induction)
{
	if (auto error = run<Expression>(*induction.getBegin()); error)
		return error;

	if (auto error = run<Expression>(*induction.getEnd()); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<IfStatement>(Statement& statement)
{
	auto* ifStatement = statement.get<IfStatement>();

	for (auto& block : *ifStatement)
	{
		if (auto error = run<Expression>(*block.getCondition()); error)
			return error;

		for (auto& stmnt : block)
			if (auto error = run<Statement>(*stmnt); error)
				return error;
	}

	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<ReturnStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<WhenStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error ConstantFolder::run<WhileStatement>(Statement& statement)
{
	auto* whileStatement = statement.get<WhileStatement>();

	if (auto error = run<Expression>(*whileStatement->getCondition()); error)
		return error;

	for (auto& stmnt : whileStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error ConstantFolder::run(Algorithm& algorithm)
{
	for (const auto& statement : algorithm.getBody())
		if (auto error = run<Statement>(*statement); error)
			return error;

	return llvm::Error::success();
}

template<BuiltInType Type>
static llvm::Error foldAddOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::add);

	using ResultType = frontendTypeToType_v<Type>;

	// For each argument, if it is a constant then accumulate it, otherwise
	// add it to the non constant values.
	ResultType constantsSum = 0;
	llvm::SmallVector<std::unique_ptr<Expression>, 3> newArgs;

	// Depth first search, in order to preserve the order
	std::stack<std::unique_ptr<Expression>> stack;

	for (auto it = operation->getArguments().rbegin(), end = operation->getArguments().rend(); it != end; ++it)
		stack.push(std::move(*it));

	while (!stack.empty())
	{
		auto current = std::move(stack.top());
		stack.pop();

		if (auto* constant = current->dyn_get<Constant>())
		{
			// Constants can be folded
			constantsSum += constant->as<Type>();
		}
		else if (auto* op = current->dyn_get<Operation>();
						 op && op->getOperationKind() == OperationKind::add)
		{
			for (auto it = op->getArguments().rbegin(), end = op->getArguments().rend(); it != end; ++it)
				stack.push(std::move(*it));
		}
		else
		{
			newArgs.push_back(std::move(current));
		}
	}

	// If there are no non-constant arguments, then transform the expression
	// into a constant.
	if (newArgs.empty())
	{
		expression = *Expression::constant(expression.getLocation(), makeType<ResultType>(), constantsSum);
		return llvm::Error::success();
	}

	// If the accumulator is not the neutral value, then insert a new constant
	// argument.
	if (constantsSum != 0)
		newArgs.push_back(Expression::constant(expression.getLocation(), makeType<ResultType>(), constantsSum));

	// If there is just one argument, then replace the sum with that value
	if (newArgs.size() == 1)
	{
		expression = *newArgs[0];
		return llvm::Error::success();
	}

	// Otherwise apply the operation to all the arguments
	expression = *Expression::operation(expression.getLocation(), expression.getType(), OperationKind::add, std::move(newArgs));
	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldAddOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::add);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	if (expression.getType() == makeType<BuiltInType::Float>())
		return ::foldAddOp<BuiltInType::Float>(expression);

	if (expression.getType() == makeType<BuiltInType::Integer>())
		return ::foldAddOp<BuiltInType::Integer>(expression);

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldDifferentOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::different);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto args = operation->getArguments();

	if (args[0]->isa<Constant>() && args[1]->isa<Constant>())
	{
		assert(args.size() == 2);

		// Cast the values to floats, in order to avoid information loss
		// during the static comparison.

		auto x = args[0]->get<Constant>()->as<BuiltInType::Float>();
		auto y = args[1]->get<Constant>()->as<BuiltInType::Float>();

		if (x == y)
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), true);
		else
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), false);
	}

	return llvm::Error::success();
}

template<BuiltInType Type>
static llvm::Error foldDivOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::divide);
	using ResultType = frontendTypeToType_v<Type>;

	llvm::SmallVector<std::unique_ptr<Expression>, 3> newArgs;

	for (auto& arg : operation->getArguments())
	{
		if (newArgs.empty())
		{
			newArgs.push_back(std::move(arg));
			continue;
		}

		if (!arg->isa<Constant>())
		{
			// If the current argument is not a constant, then there's nothing
			// we can do.
			newArgs.push_back(std::move(arg));
		}
		else
		{
			// If the previous element was a constant, then we can fold them.
			auto& last = newArgs.back();

			if (last->isa<Constant>())
			{
				*last = *Expression::constant(
						arg->getLocation(),
						makeType<ResultType>(),
						last->get<Constant>()->as<Type>() / arg->get<Constant>()->as<Type>());
			}
			else
			{
				newArgs.push_back(std::move(arg));
			}
		}
	}

	assert(!newArgs.empty());

	// If there is just one argument, then replace the sum with that value
	if (newArgs.size() == 1)
	{
		expression = *newArgs[0];
		return llvm::Error::success();
	}

	// Otherwise apply the operation to all the arguments
	expression = *Expression::operation(expression.getLocation(), expression.getType(), OperationKind::divide, std::move(newArgs));
	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldDivOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::divide);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	if (expression.getType() == makeType<BuiltInType::Float>())
		return ::foldDivOp<BuiltInType::Float>(expression);

	if (expression.getType() == makeType<BuiltInType::Integer>())
		return ::foldDivOp<BuiltInType::Integer>(expression);

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::equal);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto args = operation->getArguments();

	if (args[0]->isa<Constant>() && args[1]->isa<Constant>())
	{
		assert(args.size() == 2);

		// Cast the values to floats, in order to avoid information loss
		// during the static comparison.

		auto x = args[0]->get<Constant>()->as<BuiltInType::Float>();
		auto y = args[1]->get<Constant>()->as<BuiltInType::Float>();

		if (x == y)
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), true);
		else
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), false);
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldGreaterOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::greater);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto args = operation->getArguments();

	if (args[0]->isa<Constant>() && args[1]->isa<Constant>())
	{
		assert(args.size() == 2);

		// Cast the values to floats, in order to avoid information loss
		// during the static comparison.

		auto x = args[0]->get<Constant>()->as<BuiltInType::Float>();
		auto y = args[1]->get<Constant>()->as<BuiltInType::Float>();

		if (x > y)
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), true);
		else
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), false);
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldGreaterEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::greaterEqual);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto args = operation->getArguments();

	if (args[0]->isa<Constant>() && args[1]->isa<Constant>())
	{
		assert(args.size() == 2);

		// Cast the values to floats, in order to avoid information loss
		// during the static comparison.

		auto x = args[0]->get<Constant>()->as<BuiltInType::Float>();
		auto y = args[1]->get<Constant>()->as<BuiltInType::Float>();

		if (x >= y)
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), true);
		else
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), false);
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldIfElseOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::ifelse);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto args = operation->getArguments();

	if (args[0]->isa<Constant>())
	{
		assert(args.size() == 3);
		bool value = args[0]->get<Constant>()->as<BuiltInType::Boolean>();
		Type type = operation->getType();
		expression = value ? *args[1] : *args[2];
		expression.setType(std::move(type));
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldLessOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::less);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto args = operation->getArguments();

	if (args[0]->isa<Constant>() && args[1]->isa<Constant>())
	{
		assert(args.size() == 2);

		// Cast the values to floats, in order to avoid information loss
		// during the static comparison.

		auto x = args[0]->get<Constant>()->as<BuiltInType::Float>();
		auto y = args[1]->get<Constant>()->as<BuiltInType::Float>();

		if (x < y)
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), true);
		else
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), false);
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldLessEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::lessEqual);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto args = operation->getArguments();

	if (args[0]->isa<Constant>() && args[1]->isa<Constant>())
	{
		assert(args.size() == 2);

		// Cast the values to floats, in order to avoid information loss
		// during the static comparison.

		auto x = args[0]->get<Constant>()->as<BuiltInType::Float>();
		auto y = args[1]->get<Constant>()->as<BuiltInType::Float>();

		if (x <= y)
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), true);
		else
			expression = *Expression::constant(expression.getLocation(), makeType<bool>(), false);
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldLogicalAndOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::land);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	// For each argument, if it is a constant then accumulate it, otherwise
	// add it to the non constant values.
	bool constant = true;
	llvm::SmallVector<std::unique_ptr<Expression>, 3> newArgs;

	for (auto& arg : operation->getArguments())
	{
		if (arg->isa<Constant>())
			constant &= arg->get<Constant>()->as<BuiltInType::Boolean>();
		else
			newArgs.push_back(std::move(arg));
	}

	// If there are no non-constant arguments, then transform the expression
	// into a constant.
	if (newArgs.empty())
	{
		expression = *Expression::constant(expression.getLocation(), makeType<bool>(), constant);
		return llvm::Error::success();
	}

	// If the accumulator is not the neutral value, then insert a new constant
	// argument. We keep the other arguments also if the constant is false,
	// because they may have side effects.

	if (!constant)
		newArgs.push_back(Expression::constant(expression.getLocation(), makeType<bool>(), constant));

	// If there is just one argument, then replace the sum with that value
	if (newArgs.size() == 1)
	{
		expression = *newArgs[0];
		return llvm::Error::success();
	}

	// Otherwise apply the operation to all the arguments
	expression = *Expression::operation(expression.getLocation(), expression.getType(), OperationKind::land, std::move(newArgs));
	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldLogicalOrOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::lor);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	// For each argument, if it is a constant then accumulate it, otherwise
	// add it to the non constant values.
	bool constant = false;
	llvm::SmallVector<std::unique_ptr<Expression>, 3> newArgs;

	for (auto& arg : operation->getArguments())
	{
		if (arg->isa<Constant>())
			constant |= arg->get<Constant>()->as<BuiltInType::Boolean>();
		else
			newArgs.push_back(std::move(arg));
	}

	// If there are no non-constant arguments, then transform the expression
	// into a constant.
	if (newArgs.empty())
	{
		expression = *Expression::constant(expression.getLocation(), makeType<bool>(), constant);
		return llvm::Error::success();
	}

	// If the accumulator is not the neutral value, then insert a new constant
	// argument. We keep the other arguments also if the constant is true,
	// because they may have side effects.

	if (constant)
		newArgs.push_back(Expression::constant(expression.getLocation(), makeType<bool>(), constant));

	// If there is just one argument, then replace the sum with that value
	if (newArgs.size() == 1)
	{
		expression = *newArgs[0];
		return llvm::Error::success();
	}

	// Otherwise apply the operation to all the arguments
	expression = *Expression::operation(expression.getLocation(), expression.getType(), OperationKind::lor, std::move(newArgs));
	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldMemberLookupOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::memberLookup);
	return llvm::Error::success();
}

template<BuiltInType Type>
static llvm::Error foldMulOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::multiply);
	using ResultType = frontendTypeToType_v<Type>;

	llvm::SmallVector<std::unique_ptr<Expression>, 3> newArgs;

	for (auto& arg : operation->getArguments())
	{
		if (newArgs.empty())
		{
			newArgs.push_back(std::move(arg));
			continue;
		}

		if (!arg->isa<Constant>())
		{
			// If the current argument is not a constant, then there's nothing
			// we can do.
			newArgs.push_back(std::move(arg));
		}
		else
		{
			// If the previous element was a constant, then we can fold them.
			auto& last = newArgs.back();

			if (last->isa<Constant>())
			{
				*last = *Expression::constant(
						arg->getLocation(),
						makeType<ResultType>(),
						last->get<Constant>()->as<Type>() * arg->get<Constant>()->as<Type>());
			}
			else
			{
				newArgs.push_back(std::move(arg));
			}
		}
	}

	assert(!newArgs.empty());

	// If there is just one argument, then replace the sum with that value
	if (newArgs.size() == 1)
	{
		expression = *newArgs[0];
		return llvm::Error::success();
	}

	// Otherwise apply the operation to all the arguments
	expression = *Expression::operation(expression.getLocation(), expression.getType(), OperationKind::multiply, std::move(newArgs));
	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldMulOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::multiply);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	if (expression.getType() == makeType<BuiltInType::Float>())
		return ::foldMulOp<BuiltInType::Float>(expression);

	if (expression.getType() == makeType<BuiltInType::Integer>())
		return ::foldMulOp<BuiltInType::Integer>(expression);

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldNegateOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::negate);

	if (auto error = run<Expression>(*operation->getArg(0)); error)
		return error;

	if (auto* arg = operation->getArg(0); arg->isa<Constant>())
	{
		auto* constant = arg->get<Constant>();
		assert(constant->isa<BuiltInType::Boolean>());

		bool value = constant->get<BuiltInType::Boolean>();
		expression = *Expression::constant(operation->getLocation(), operation->getType(), !value);
		return llvm::Error::success();
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldPowerOfOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::powerOf);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	return llvm::Error::success();
}

template<BuiltInType Type>
static llvm::Error foldSubOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::subtract);
	using ResultType = frontendTypeToType_v<Type>;

	assert(operation->argumentsCount() <= 2);
	auto args = operation->getArguments();

	if (args.size() == 1)
	{
		if (args[0]->isa<Constant>())
		{
			expression = *Expression::constant(expression.getLocation(), args[0]->getType(), -args[0]->get<Constant>()->get<Type>());
			return llvm::Error::success();
		}
	}

	if (args[0]->isa<Constant>() && args[1]->isa<Constant>())
	{
		expression = *Expression::constant(
				expression.getLocation(),
				makeType<ResultType>(),
				args[0]->get<Constant>()->as<Type>() - args[1]->get<Constant>()->as<Type>());

		return llvm::Error::success();
	}

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldSubOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::subtract);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	if (expression.getType() == makeType<BuiltInType::Float>())
		return ::foldSubOp<BuiltInType::Float>(expression);

	if (expression.getType() == makeType<BuiltInType::Integer>())
		return ::foldSubOp<BuiltInType::Integer>(expression);

	return llvm::Error::success();
}

llvm::Error ConstantFolder::foldSubscriptionOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::subscription);
	auto args = operation->getArguments();

	for (auto it = std::next(args.begin()), end = args.end(); it != end; ++it)
		if (auto error = run<Expression>(**it); error)
			return error;

	auto new_args = operation->getArguments();

	//substitutes the subscription with the constant value, if possible
	if(new_args.size() == 2 && new_args[0]->isa<ReferenceAccess>() && new_args[1]->isa<Constant>()){
		auto* reference = new_args[0]->get<ReferenceAccess>();
		auto* index = new_args[1]->get<Constant>();

		if(index->isa<BuiltInType::Integer>()){
			int int_index = index->as<BuiltInType::Integer>();

			if(int_index < 0){
				// negative index : probably an error
				return llvm::Error::success();
			}

			if (symbolTable.count(reference->getName()) == 0)
			{
				// Built-in variables (such as time) or functions are not in the symbol
				// table.
				return llvm::Error::success();
			}

			const auto& symbol = symbolTable.lookup(reference->getName());

			if (!symbol.isa<Member>())
				return llvm::Error::success();

			// Try to fold references of known variables that have a initializer
			const auto* member = symbol.get<Member>();

			if (!member->hasInitializer())
				return llvm::Error::success();

			auto* initializer = member->getInitializer();

			if (initializer->isa<Array>() && member->isParameter()){
				auto* array = initializer->get<Array>();

				if(int_index < array->size()){
					//if all conditions are met, substitute the subscription with the constant value
					expression = *(*array)[int_index];
				}
				return llvm::Error::success();
			}
		}

	}

	return llvm::Error::success();
}

std::unique_ptr<Pass> marco::ast::createConstantFolderPass()
{
	return std::make_unique<ConstantFolder>();
}
