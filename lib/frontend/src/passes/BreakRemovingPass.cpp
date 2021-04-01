#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/BreakRemovingPass.h>

using namespace modelica::frontend;

llvm::Error BreakRemover::run(ClassContainer& cls)
{
	return cls.visit([&](auto& obj) { return run(obj); });
}

llvm::Error BreakRemover::run(Class& cls)
{
	return llvm::Error::success();
}

llvm::Error BreakRemover::run(Function& function)
{
	for (auto& algorithm : function.getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	return llvm::Error::success();
}

llvm::Error BreakRemover::run(Package& package)
{
	for (auto& cls : package)
		if (auto error = run(cls); error)
			return error;

	return llvm::Error::success();
}

llvm::Error BreakRemover::run(Record& record)
{
	return llvm::Error::success();
}

llvm::Error BreakRemover::run(Algorithm& algorithm)
{
	for (auto& statement : algorithm)
		run<Statement>(*statement);

	return llvm::Error::success();
}

template<>
bool BreakRemover::run<Statement>(Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(statement);
	});
}

template<>
bool BreakRemover::run<AssignmentStatement>(Statement& statement)
{
	return false;
}

template<>
bool BreakRemover::run<IfStatement>(Statement& statement)
{
	auto& ifStatement = statement.get<IfStatement>();
	bool breakable = false;

	for (auto& block : ifStatement)
	{
		bool blockBreakable = false;

		llvm::SmallVector<std::shared_ptr<Statement>, 3> statements;
		llvm::SmallVector<std::shared_ptr<Statement>, 3> avoidableStatements;

		for (auto& stmnt : block)
		{
			if (blockBreakable)
				avoidableStatements.push_back(stmnt);
			else
				statements.push_back(stmnt);

			blockBreakable |= run<Statement>(*stmnt);
		}

		if (blockBreakable && !avoidableStatements.empty())
		{
			Expression reference = Expression::reference(ifStatement.getLocation(), makeType<bool>(), "__mustBreak" + std::to_string(nestLevel));
			Expression falseConstant = Expression::constant(ifStatement.getLocation(), makeType<bool>(), false);
			Expression condition = Expression::operation(ifStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

			// Create the block of code to be executed if a break is not called
			IfStatement::Block breakNotCalledBlock(condition, {});
			breakNotCalledBlock.getBody() = avoidableStatements;
			statements.push_back(std::make_shared<Statement>(IfStatement(ifStatement.getLocation(), breakNotCalledBlock)));

			block.getBody() = statements;
		}

		breakable |= blockBreakable;
	}

	return breakable;
}

template<>
bool BreakRemover::run<ForStatement>(Statement& statement)
{
	auto& forStatement = statement.get<ForStatement>();
	bool breakable = false;
	nestLevel++;

	llvm::SmallVector<std::shared_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::shared_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : forStatement)
	{
		if (breakable)
			avoidableStatements.push_back(stmnt);
		else
			statements.push_back(stmnt);

		breakable |= run<Statement>(*stmnt);
	}

	if (breakable && !avoidableStatements.empty())
	{
		Expression reference = Expression::reference(forStatement.getLocation(), makeType<bool>(), "__mustBreak" + std::to_string(nestLevel));
		Expression falseConstant = Expression::constant(forStatement.getLocation(), makeType<bool>(), false);
		Expression condition = Expression::operation(forStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a break is not called
		IfStatement::Block breakNotCalledBlock(condition, {});
		breakNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(forStatement.getLocation(), breakNotCalledBlock)));
	}

	forStatement.getBody() = statements;
	forStatement.setBreakCheckName("__mustBreak" + std::to_string(nestLevel));
	nestLevel--;

	// A for statement can't break a parent one
	return false;
}

template<>
bool BreakRemover::run<WhileStatement>(Statement& statement)
{
	auto& whileStatement = statement.get<WhileStatement>();
	bool breakable = false;
	nestLevel++;

	llvm::SmallVector<std::shared_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::shared_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : whileStatement)
	{
		if (breakable)
			avoidableStatements.push_back(stmnt);
		else
			statements.push_back(stmnt);

		breakable |= run<Statement>(*stmnt);
	}

	if (breakable && !avoidableStatements.empty())
	{
		Expression reference = Expression::reference(whileStatement.getLocation(), makeType<bool>(), "__mustBreak" + std::to_string(nestLevel));
		Expression falseConstant = Expression::constant(whileStatement.getLocation(), makeType<bool>(), false);
		Expression condition = Expression::operation(whileStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a break is not called
		IfStatement::Block breakNotCalledBlock(condition, {});
		breakNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(whileStatement.getLocation(), breakNotCalledBlock)));
	}

	whileStatement.getBody() = statements;
	whileStatement.setBreakCheckName("__mustBreak" + std::to_string(nestLevel));
	nestLevel--;

	// A while statement can't break a parent one
	return false;
}

template<>
bool BreakRemover::run<WhenStatement>(Statement& statement)
{
	return false;
}

template<>
bool BreakRemover::run<BreakStatement>(Statement& statement)
{
	auto location = statement.get<BreakStatement>().getLocation();

	statement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "__mustBreak" + std::to_string(nestLevel)),
			Expression::constant(location, makeType<bool>(), true));

	return true;
}

template<>
bool BreakRemover::run<ReturnStatement>(Statement& statement)
{
	return false;
}

std::unique_ptr<Pass> modelica::frontend::createBreakRemovingPass()
{
	return std::make_unique<BreakRemover>();
}
