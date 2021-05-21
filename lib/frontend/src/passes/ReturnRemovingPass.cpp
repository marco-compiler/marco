#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/ReturnRemovingPass.h>

using namespace modelica::frontend;

template<>
llvm::Error ReturnRemover::run<Class>(Class& cls)
{
	return cls.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(cls);
	});
}

llvm::Error ReturnRemover::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
{
	for (const auto& cls : classes)
		if (auto error = run<Class>(*cls); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ReturnRemover::run<DerFunction>(Class& cls)
{
	return llvm::Error::success();
}

template<>
llvm::Error ReturnRemover::run<StandardFunction>(Class& cls)
{
	for (auto& algorithm : cls.get<StandardFunction>()->getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ReturnRemover::run<Model>(Class& cls)
{
	return llvm::Error::success();
}

template<>
llvm::Error ReturnRemover::run<Package>(Class& cls)
{
	for (auto& innerClass : cls.get<Package>()->getInnerClasses())
		if (auto error = run<Class>(*innerClass); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error ReturnRemover::run<Record>(Class& cls)
{
	for (auto& innerClass : cls.get<Package>()->getInnerClasses())
		if (auto error = run<Class>(*innerClass); error)
			return error;

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
bool ReturnRemover::run<BreakStatement>(Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::run<ForStatement>(Statement& statement)
{
	auto* forStatement = statement.get<ForStatement>();
	bool canReturn = false;

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::unique_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : *forStatement)
	{
		bool currentCanReturn = run<Statement>(*stmnt);

		if (canReturn)
			avoidableStatements.push_back(std::move(stmnt));
		else
			statements.push_back(std::move(stmnt));

		canReturn |= currentCanReturn;
	}

	if (canReturn && !avoidableStatements.empty())
	{
		auto condition = Expression::operation(
				forStatement->getLocation(),
				makeType<bool>(),
				OperationKind::equal,
				llvm::ArrayRef({
						Expression::reference(forStatement->getLocation(), makeType<bool>(), "__mustReturn"),
						Expression::constant(forStatement->getLocation(), makeType<bool>(), false)
				}));

		// Create the block of code to be executed if a return is not called
		IfStatement::Block returnNotCalledBlock(std::move(condition), {});
		returnNotCalledBlock.setBody(avoidableStatements);
		statements.push_back(Statement::ifStatement(forStatement->getLocation(), returnNotCalledBlock));
	}

	forStatement->setBody(statements);
	forStatement->setReturnCheckName("__mustReturn");
	return canReturn;
}

template<>
bool ReturnRemover::run<IfStatement>(Statement& statement)
{
	auto* ifStatement = statement.get<IfStatement>();
	bool canReturn = false;

	for (auto& block : *ifStatement)
	{
		bool blockCanReturn = false;

		llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;
		llvm::SmallVector<std::unique_ptr<Statement>, 3> avoidableStatements;

		for (auto& stmnt : block)
		{
			bool currentCanReturn = run<Statement>(*stmnt);

			if (blockCanReturn)
				avoidableStatements.push_back(std::move(stmnt));
			else
				statements.push_back(std::move(stmnt));

			blockCanReturn |= currentCanReturn;
		}

		if (blockCanReturn && !avoidableStatements.empty())
		{
			auto condition = Expression::operation(
					ifStatement->getLocation(),
					makeType<bool>(),
					OperationKind::equal,
					llvm::ArrayRef({
							Expression::reference(ifStatement->getLocation(), makeType<bool>(), "__mustReturn"),
							Expression::constant(ifStatement->getLocation(), makeType<bool>(), false)
					}));

			// Create the block of code to be executed if a return is not called
			IfStatement::Block returnNotCalledBlock(std::move(condition), {});
			returnNotCalledBlock.setBody(avoidableStatements);
			statements.push_back(Statement::ifStatement(ifStatement->getLocation(), returnNotCalledBlock));
		}

		block.setBody(statements);
		canReturn |= blockCanReturn;
	}

	return canReturn;
}

template<>
bool ReturnRemover::run<ReturnStatement>(Statement& statement)
{
	auto location = statement.getLocation();

	statement = *Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "__mustReturn"),
			Expression::constant(location, makeType<bool>(), true));

	return true;
}

template<>
bool ReturnRemover::run<WhenStatement>(Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::run<WhileStatement>(Statement& statement)
{
	auto* whileStatement = statement.get<WhileStatement>();
	bool canReturn = false;

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::unique_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : *whileStatement)
	{
		bool currentCanReturn = run<Statement>(*stmnt);

		if (canReturn)
			avoidableStatements.push_back(std::move(stmnt));
		else
			statements.push_back(std::move(stmnt));

		canReturn |= currentCanReturn;
	}

	if (canReturn && !avoidableStatements.empty())
	{
		auto condition = Expression::operation(
				whileStatement->getLocation(),
				makeType<bool>(),
				OperationKind::equal,
				llvm::ArrayRef({
						Expression::reference(whileStatement->getLocation(), makeType<bool>(), "__mustReturn"),
						Expression::constant(whileStatement->getLocation(), makeType<bool>(), false)
				}));

		// Create the block of code to be executed if a return is not called
		IfStatement::Block returnNotCalledBlock(std::move(condition), {});
		returnNotCalledBlock.setBody(avoidableStatements);
		statements.push_back(Statement::ifStatement(whileStatement->getLocation(), returnNotCalledBlock));
	}

	whileStatement->setBody(statements);
	whileStatement->setReturnCheckName("__mustReturn");
	return canReturn;
}

llvm::Error ReturnRemover::run(Algorithm& algorithm)
{
	bool canReturn = false;

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::unique_ptr<Statement>, 3> avoidableStatements;

	for (auto& statement : algorithm)
	{
		bool currentCanReturn = run<Statement>(*statement);

		if (canReturn)
			avoidableStatements.push_back(std::move(statement));
		else
			statements.push_back(std::move(statement));

		canReturn |= currentCanReturn;
	}

	if (canReturn && !avoidableStatements.empty())
	{
		auto condition = Expression::operation(
				algorithm.getLocation(),
				makeType<bool>(),
				OperationKind::equal,
				llvm::ArrayRef({
						Expression::reference(algorithm.getLocation(), makeType<bool>(), "__mustReturn"),
						Expression::constant(algorithm.getLocation(), makeType<bool>(), false)
				}));

		// Create the block of code to be executed if a return is not called
		IfStatement::Block returnNotCalledBlock(std::move(condition), {});
		returnNotCalledBlock.setBody(avoidableStatements);
		statements.push_back(Statement::ifStatement(algorithm.getLocation(), returnNotCalledBlock));
	}

	algorithm.setBody(statements);
	algorithm.setReturnCheckName("__mustReturn");

	return llvm::Error::success();
}

std::unique_ptr<Pass> modelica::frontend::createReturnRemovingPass()
{
	return std::make_unique<ReturnRemover>();
}
