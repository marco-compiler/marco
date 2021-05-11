#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/BreakRemovingPass.h>

using namespace modelica::frontend;

llvm::Error BreakRemover::run(Class& cls)
{
	if (auto* function = cls.dyn_cast<Function>())
		return run(*function);

	if (auto* model = cls.dyn_cast<Model>())
		return run(*model);

	if (auto* package = cls.dyn_cast<Package>())
		return run(*package);

	if (auto* record = cls.dyn_cast<Record>())
		return run(*record);

	return llvm::Error::success();
}

llvm::Error BreakRemover::run(Model& cls)
{
	return llvm::Error::success();
}

llvm::Error BreakRemover::run(Function& function)
{
	if (auto* derFunction = function.dyn_cast<DerFunction>())
		return run(*derFunction);

	if (auto* standardFunction = function.dyn_cast<StandardFunction>())
		return run(*standardFunction);

	return llvm::Error::success();
}

llvm::Error BreakRemover::run(DerFunction& function)
{
	return llvm::Error::success();
}

llvm::Error BreakRemover::run(StandardFunction& function)
{
	for (auto& algorithm : function.getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	return llvm::Error::success();
}

llvm::Error BreakRemover::run(Package& package)
{
	for (auto& cls : package)
		if (auto error = run(*cls); error)
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
		run(*statement);

	return llvm::Error::success();
}

bool BreakRemover::run(Statement& statement)
{
	if (auto* assignmentStatement = statement.dyn_cast<AssignmentStatement>())
		return run(*assignmentStatement);

	if (auto* breakStatement = statement.dyn_cast<BreakStatement>())
		return run(*breakStatement);

	if (auto* forStatement = statement.dyn_cast<ForStatement>())
		return run(*forStatement);

	if (auto* ifStatement = statement.dyn_cast<IfStatement>())
		return run(*ifStatement);

	if (auto* returnStatement = statement.dyn_cast<ReturnStatement>())
		return run(*returnStatement);

	if (auto* whenStatement = statement.dyn_cast<WhenStatement>())
		return run(*whenStatement);

	if (auto* whileStatement = statement.dyn_cast<WhileStatement>())
		return run(*whileStatement);

	return false;
}

bool BreakRemover::run(AssignmentStatement& statement)
{
	return false;
}

bool BreakRemover::run(BreakStatement& statement)
{
	auto location = statement.getLocation();

	statement = AssignmentStatement(
			location,
			std::make_unique<ReferenceAccess>(location, makeType<bool>(), "__mustBreak" + std::to_string(nestLevel)),
			std::make_unique<Constant>(location, makeType<bool>(), true));

	return true;
}

bool BreakRemover::run(IfStatement& statement)
{
	bool breakable = false;

	for (auto& block : statement)
	{
		bool blockBreakable = false;

		llvm::SmallVector<Statement*, 3> statements;
		llvm::SmallVector<Statement*, 3> avoidableStatements;

		for (auto& stmnt : block)
		{
			if (blockBreakable)
				avoidableStatements.push_back(stmnt.get());
			else
				statements.push_back(stmnt.get());

			blockBreakable |= run(*stmnt);
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

bool BreakRemover::run(ForStatement& statement)
{
	bool breakable = false;
	nestLevel++;

	llvm::SmallVector<Statement*, 3> statements;
	llvm::SmallVector<Statement*, 3> avoidableStatements;

	for (auto& stmnt : statement)
	{
		if (breakable)
			avoidableStatements.push_back(stmnt.get());
		else
			statements.push_back(stmnt.get());

		breakable |= run(*stmnt);
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

	statement.getBody() = statements;
	statement.setBreakCheckName("__mustBreak" + std::to_string(nestLevel));
	nestLevel--;

	// A for statement can't break a parent one
	return false;
}

bool BreakRemover::run(ReturnStatement& statement)
{
	return false;
}

bool BreakRemover::run(WhenStatement& statement)
{
	return false;
}

bool BreakRemover::run(WhileStatement& statement)
{
	bool breakable = false;
	nestLevel++;

	llvm::SmallVector<Statement*, 3> statements;
	llvm::SmallVector<Statement*, 3> avoidableStatements;

	for (auto& stmnt : statement)
	{
		if (breakable)
			avoidableStatements.push_back(stmnt.get());
		else
			statements.push_back(stmnt.get());

		breakable |= run(*stmnt);
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

	statement.getBody() = statements;
	statement.setBreakCheckName("__mustBreak" + std::to_string(nestLevel));
	nestLevel--;

	// A while statement can't break a parent one
	return false;
}

std::unique_ptr<Pass> modelica::frontend::createBreakRemovingPass()
{
	return std::make_unique<BreakRemover>();
}
