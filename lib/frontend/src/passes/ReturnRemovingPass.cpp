#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/ReturnRemovingPass.h>

using namespace modelica::frontend;

llvm::Error ReturnRemover::run(Class& cls)
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

llvm::Error ReturnRemover::run(Model& cls)
{
	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(Function& function)
{
	if (auto* derFunction = function.dyn_cast<DerFunction>())
		return run(*derFunction);

	if (auto* standardFunction = function.dyn_cast<StandardFunction>())
		return run(*standardFunction);

	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(DerFunction& function)
{
	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(StandardFunction& function)
{
	for (auto& algorithm : function.getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	return llvm::Error::success();
}

llvm::Error ReturnRemover::run(Package& package)
{
	for (auto& cls : package)
		if (auto error = run(*cls); error)
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

bool ReturnRemover::run(Statement& statement)
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

bool ReturnRemover::run(AssignmentStatement& statement)
{
	return false;
}

bool ReturnRemover::run(BreakStatement& statement)
{
	return false;
}

bool ReturnRemover::run(IfStatement& statement)
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

bool ReturnRemover::run(ForStatement& statement)
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

		canReturn |= run(*stmnt);
	}

	if (canReturn && !avoidableStatements.empty())
	{
		auto reference = std::make_unique<ReferenceAccess>(forStatement.getLocation(), makeType<bool>(), "__mustReturn");
		auto falseConstant = std::make_unique<Constant>(forStatement.getLocation(), makeType<bool>(), false);
		auto condition = std::make_unique<Operation>(forStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a return is not called
		IfStatement::Block returnNotCalledBlock(condition, {});
		returnNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(forStatement.getLocation(), returnNotCalledBlock)));
	}

	forStatement.getBody() = statements;
	forStatement.setReturnCheckName("__mustReturn");
	return canReturn;
}

bool ReturnRemover::run(ReturnStatement& statement)
{
	auto location = statement.get<ReturnStatement>().getLocation();

	statement = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "__mustReturn"),
			Expression::constant(location, makeType<bool>(), true));

	return true;
}

bool ReturnRemover::run(WhenStatement& statement)
{
	return false;
}

bool ReturnRemover::run(WhileStatement& statement)
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

std::unique_ptr<Pass> modelica::frontend::createReturnRemovingPass()
{
	return std::make_unique<ReturnRemover>();
}
