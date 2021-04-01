#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/ReturnRemovingPass.h>

using namespace modelica;

llvm::Error ReturnRemover::run(ClassContainer& cls)
{
	return cls.visit([&](auto& obj) { return run(obj); });
}

llvm::Error ReturnRemover::run(Class& cls)
{
	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(Function& function)
{
	for (auto& algorithm : function.getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(Package& package)
{
	for (auto& cls : package)
		if (auto error = run(cls); error)
			return error;

	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(Record& record)
{
	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(Algorithm& algorithm)
{
	bool canReturn = false;

	llvm::SmallVector<std::shared_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::shared_ptr<Statement>, 3> avoidableStatements;

	for (auto& statement : algorithm)
	{
		if (canReturn)
			avoidableStatements.push_back(statement);
		else
			statements.push_back(statement);

		canReturn |= run<Statement>(*statement);
	}

	if (canReturn && !avoidableStatements.empty())
	{
		Expression reference = Expression::reference(algorithm.getLocation(), makeType<bool>(), "__mustReturn");
		Expression falseConstant = Expression::constant(algorithm.getLocation(), makeType<bool>(), false);
		Expression condition = Expression::operation(algorithm.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a return is not called
		IfStatement::Block returnNotCalledBlock(condition, {});
		returnNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(algorithm.getLocation(), returnNotCalledBlock)));
	}

	algorithm.getStatements() = statements;
	algorithm.setReturnCheckName("__mustReturn");
	return llvm::Error::success();
}

template<>
bool ReturnRemover::run<Statement>(Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(statement);
	});
}

template<>
bool ReturnRemover::run<AssignmentStatement>(Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::run<IfStatement>(Statement& statement)
{
	auto& ifStatement = statement.get<IfStatement>();
	bool canReturn = false;

	for (auto& block : ifStatement)
	{
		bool blockCanReturn = false;

		llvm::SmallVector<std::shared_ptr<Statement>, 3> statements;
		llvm::SmallVector<std::shared_ptr<Statement>, 3> avoidableStatements;

		for (auto& stmnt : block)
		{
			if (blockCanReturn)
				avoidableStatements.push_back(stmnt);
			else
				statements.push_back(stmnt);

			blockCanReturn |= run<Statement>(*stmnt);
		}

		if (blockCanReturn && !avoidableStatements.empty())
		{
			Expression reference = Expression::reference(ifStatement.getLocation(), makeType<bool>(), "__mustReturn");
			Expression falseConstant = Expression::constant(ifStatement.getLocation(), makeType<bool>(), false);
			Expression condition = Expression::operation(ifStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

			// Create the block of code to be executed if a return is not called
			IfStatement::Block returnNotCalledBlock(condition, {});
			returnNotCalledBlock.getBody() = avoidableStatements;
			statements.push_back(std::make_shared<Statement>(IfStatement(ifStatement.getLocation(), returnNotCalledBlock)));

			block.getBody() = statements;
		}

		canReturn |= blockCanReturn;
	}

	return canReturn;
}

template<>
bool ReturnRemover::run<ForStatement>(Statement& statement)
{
	auto& forStatement = statement.get<ForStatement>();
	bool canReturn = false;

	llvm::SmallVector<std::shared_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::shared_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : forStatement)
	{
		if (canReturn)
			avoidableStatements.push_back(stmnt);
		else
			statements.push_back(stmnt);

		canReturn |= run<Statement>(*stmnt);
	}

	if (canReturn && !avoidableStatements.empty())
	{
		Expression reference = Expression::reference(forStatement.getLocation(), makeType<bool>(), "__mustReturn");
		Expression falseConstant = Expression::constant(forStatement.getLocation(), makeType<bool>(), false);
		Expression condition = Expression::operation(forStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a return is not called
		IfStatement::Block returnNotCalledBlock(condition, {});
		returnNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(forStatement.getLocation(), returnNotCalledBlock)));
	}

	forStatement.getBody() = statements;
	forStatement.setReturnCheckName("__mustReturn");
	return canReturn;
}

template<>
bool ReturnRemover::run<WhileStatement>(Statement& statement)
{
	auto& whileStatement = statement.get<WhileStatement>();
	bool canReturn = false;

	llvm::SmallVector<std::shared_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::shared_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : whileStatement)
	{
		if (canReturn)
			avoidableStatements.push_back(stmnt);
		else
			statements.push_back(stmnt);

		canReturn |= run<Statement>(*stmnt);
	}

	if (canReturn && !avoidableStatements.empty())
	{
		Expression reference = Expression::reference(whileStatement.getLocation(), makeType<bool>(), "__mustReturn");
		Expression falseConstant = Expression::constant(whileStatement.getLocation(), makeType<bool>(), false);
		Expression condition = Expression::operation(whileStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a return is not called
		IfStatement::Block returnNotCalledBlock(condition, {});
		returnNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(whileStatement.getLocation(), returnNotCalledBlock)));
	}

	whileStatement.getBody() = statements;
	whileStatement.setReturnCheckName("__mustReturn");
	return canReturn;
}

template<>
bool ReturnRemover::run<WhenStatement>(Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::run<BreakStatement>(Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::run<ReturnStatement>(Statement& statement)
{
	auto location = statement.get<ReturnStatement>().getLocation();

	statement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "__mustReturn"),
			Expression::constant(location, makeType<bool>(), true));

	return true;
}

std::unique_ptr<Pass> modelica::createReturnRemovingPass()
{
	return std::make_unique<ReturnRemover>();
}
