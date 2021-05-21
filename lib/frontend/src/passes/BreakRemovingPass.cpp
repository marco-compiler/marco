#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/BreakRemovingPass.h>

using namespace modelica::frontend;

template<>
llvm::Error BreakRemover::run<Class>(Class& cls)
{
	return cls.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(cls);
	});
}

llvm::Error BreakRemover::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
{
	for (const auto& cls : classes)
	 if (auto error = run<Class>(*cls); error)
		 return error;

	return llvm::Error::success();
}

template<>
llvm::Error BreakRemover::run<DerFunction>(Class& cls)
{
	return llvm::Error::success();
}

template<>
llvm::Error BreakRemover::run<StandardFunction>(Class& cls)
{
	for (auto& algorithm : cls.get<StandardFunction>()->getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error BreakRemover::run<Model>(Class& cls)
{
	return llvm::Error::success();
}

template<>
llvm::Error BreakRemover::run<Package>(Class& cls)
{
	for (auto& innerClass : cls.get<Package>()->getInnerClasses())
		if (auto error = run<Class>(*innerClass); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error BreakRemover::run<Record>(Class& cls)
{
	for (auto& innerClass : cls.get<Package>()->getInnerClasses())
		if (auto error = run<Class>(*innerClass); error)
			return error;

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
bool BreakRemover::run<BreakStatement>(Statement& statement)
{
	auto location = statement.getLocation();

	statement = *Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "__mustBreak" + std::to_string(nestLevel)),
			Expression::constant(location, makeType<bool>(), true));

	return true;
}

template<>
bool BreakRemover::run<ForStatement>(Statement& statement)
{
	auto* forStatement = statement.get<ForStatement>();
	bool breakable = false;
	nestLevel++;

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::unique_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : *forStatement)
	{
		bool currentBreakable = run<Statement>(*stmnt);

		if (breakable)
			avoidableStatements.push_back(std::move(stmnt));
		else
			statements.push_back(std::move(stmnt));

		breakable |= currentBreakable;
	}

	if (breakable && !avoidableStatements.empty())
	{
		auto condition = Expression::operation(
				forStatement->getLocation(),
				makeType<bool>(),
				OperationKind::equal,
				llvm::ArrayRef({
						Expression::reference(forStatement->getLocation(), makeType<bool>(), "__mustBreak" + std::to_string(nestLevel)),
						Expression::constant(forStatement->getLocation(), makeType<bool>(), false)
				}));

		// Create the block of code to be executed if a break is not called
		IfStatement::Block breakNotCalledBlock(std::move(condition), {});
		breakNotCalledBlock.setBody(avoidableStatements);
		statements.push_back(Statement::ifStatement(forStatement->getLocation(), breakNotCalledBlock));
	}

	forStatement->setBody(statements);
	forStatement->setBreakCheckName("__mustBreak" + std::to_string(nestLevel));
	nestLevel--;

	// A for statement can't break a parent one
	return false;
}

template<>
bool BreakRemover::run<IfStatement>(Statement& statement)
{
	auto* ifStatement = statement.get<IfStatement>();
	bool breakable = false;

	for (auto& block : *ifStatement)
	{
		bool blockBreakable = false;

		llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;
		llvm::SmallVector<std::unique_ptr<Statement>, 3> avoidableStatements;

		for (auto& stmnt : block)
		{
			bool currentBreakable = run<Statement>(*stmnt);

			if (blockBreakable)
				avoidableStatements.push_back(std::move(stmnt));
			else
				statements.push_back(std::move(stmnt));

			blockBreakable |= currentBreakable;
		}

		if (blockBreakable && !avoidableStatements.empty())
		{
			auto condition = Expression::operation(
					ifStatement->getLocation(),
					makeType<bool>(),
					    OperationKind::equal,
					llvm::ArrayRef({
							Expression::reference(ifStatement->getLocation(), makeType<bool>(), "__mustBreak" + std::to_string(nestLevel)),
							Expression::constant(ifStatement->getLocation(), makeType<bool>(), false)
					}));

			// Create the block of code to be executed if a break is not called
			IfStatement::Block breakNotCalledBlock(std::move(condition), {});
			breakNotCalledBlock.setBody(avoidableStatements);
			statements.push_back(Statement::ifStatement(ifStatement->getLocation(), breakNotCalledBlock));
		}

		block.setBody(statements);
		breakable |= blockBreakable;
	}

	return breakable;
}

template<>
bool BreakRemover::run<ReturnStatement>(Statement& statement)
{
	return false;
}

template<>
bool BreakRemover::run<WhenStatement>(Statement& statement)
{
	return false;
}

template<>
bool BreakRemover::run<WhileStatement>(Statement& statement)
{
	auto* whileStatement = statement.get<WhileStatement>();
	bool breakable = false;
	nestLevel++;

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;
	llvm::SmallVector<std::unique_ptr<Statement>, 3> avoidableStatements;

	for (auto& stmnt : *whileStatement)
	{
		bool currentBreakable = run<Statement>(*stmnt);

		if (breakable)
			avoidableStatements.push_back(std::move(stmnt));
		else
			statements.push_back(std::move(stmnt));

		breakable |= currentBreakable;
	}

	if (breakable && !avoidableStatements.empty())
	{
		auto condition = Expression::operation(
				whileStatement->getLocation(),
				makeType<bool>(),
				    OperationKind::equal,
				llvm::ArrayRef({
						Expression::reference(whileStatement->getLocation(), makeType<bool>(), "__mustBreak" + std::to_string(nestLevel)),
						Expression::constant(whileStatement->getLocation(), makeType<bool>(), false)
				}));

		// Create the block of code to be executed if a break is not called
		IfStatement::Block breakNotCalledBlock(std::move(condition), {});
		breakNotCalledBlock.setBody(avoidableStatements);
		statements.push_back(Statement::ifStatement(whileStatement->getLocation(), breakNotCalledBlock));
	}

	whileStatement->setBody(statements);
	whileStatement->setBreakCheckName("__mustBreak" + std::to_string(nestLevel));
	nestLevel--;

	// A while statement can't break a parent one
	return false;
}

llvm::Error BreakRemover::run(Algorithm& algorithm)
{
	for (auto& statement : algorithm)
		run<Statement>(*statement);

	return llvm::Error::success();
}

std::unique_ptr<Pass> modelica::frontend::createBreakRemovingPass()
{
	return std::make_unique<BreakRemover>();
}
