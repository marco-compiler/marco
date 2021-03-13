#include <modelica/mlirlowerer/passes/BreakRemover.h>

using namespace modelica;
using namespace std;

void BreakRemover::fix(modelica::ClassContainer& cls)
{
	cls.visit([&](auto& obj) { fix(obj); });
}

void BreakRemover::fix(modelica::Class& cls)
{

}

void BreakRemover::fix(modelica::Function& function)
{
	for (auto& algorithm : function.getAlgorithms())
		fix(*algorithm);
}

void BreakRemover::fix(modelica::Algorithm& algorithm)
{
	for (auto& statement : algorithm)
		fix<modelica::Statement>(statement);
}

template<>
bool BreakRemover::fix<modelica::Statement>(modelica::Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return fix<deconst>(statement);
	});
}

template<>
bool BreakRemover::fix<modelica::AssignmentStatement>(modelica::Statement& statement)
{
	return false;
}

template<>
bool BreakRemover::fix<modelica::IfStatement>(modelica::Statement& statement)
{
	auto& ifStatement = statement.get<modelica::IfStatement>();
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

			blockBreakable |= fix<modelica::Statement>(*stmnt);
		}

		if (blockBreakable && !avoidableStatements.empty())
		{
			Expression reference = Expression::reference(ifStatement.getLocation(), makeType<bool>(), "__mustBreak" + to_string(nestLevel));
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
bool BreakRemover::fix<modelica::ForStatement>(modelica::Statement& statement)
{
	auto& forStatement = statement.get<modelica::ForStatement>();
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

		breakable |= fix<modelica::Statement>(*stmnt);
	}

	if (breakable && !avoidableStatements.empty())
	{
		Expression reference = Expression::reference(forStatement.getLocation(), makeType<bool>(), "__mustBreak" + to_string(nestLevel));
		Expression falseConstant = Expression::constant(forStatement.getLocation(), makeType<bool>(), false);
		Expression condition = Expression::operation(forStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a break is not called
		IfStatement::Block breakNotCalledBlock(condition, {});
		breakNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(forStatement.getLocation(), breakNotCalledBlock)));
	}

	forStatement.getBody() = statements;
	forStatement.setBreakCheckName("__mustBreak" + to_string(nestLevel));
	nestLevel--;

	// A for statement can't break a parent one
	return false;
}

template<>
bool BreakRemover::fix<modelica::WhileStatement>(modelica::Statement& statement)
{
	auto& whileStatement = statement.get<modelica::WhileStatement>();
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

		breakable |= fix<modelica::Statement>(*stmnt);
	}

	if (breakable && !avoidableStatements.empty())
	{
		Expression reference = Expression::reference(whileStatement.getLocation(), makeType<bool>(), "__mustBreak" + to_string(nestLevel));
		Expression falseConstant = Expression::constant(whileStatement.getLocation(), makeType<bool>(), false);
		Expression condition = Expression::operation(whileStatement.getLocation(), makeType<bool>(), OperationKind::equal, reference, falseConstant);

		// Create the block of code to be executed if a break is not called
		IfStatement::Block breakNotCalledBlock(condition, {});
		breakNotCalledBlock.getBody() = avoidableStatements;
		statements.push_back(std::make_shared<Statement>(IfStatement(whileStatement.getLocation(), breakNotCalledBlock)));
	}

	whileStatement.getBody() = statements;
	whileStatement.setBreakCheckName("__mustBreak" + to_string(nestLevel));
	nestLevel--;

	// A while statement can't break a parent one
	return false;
}

template<>
bool BreakRemover::fix<modelica::WhenStatement>(modelica::Statement& statement)
{
	return false;
}

template<>
bool BreakRemover::fix<modelica::BreakStatement>(modelica::Statement& statement)
{
	auto& breakStatement = statement.get<modelica::BreakStatement>();
	breakStatement.setBreakCheckName("__mustBreak" + to_string(nestLevel));
	return true;
}

template<>
bool BreakRemover::fix<modelica::ReturnStatement>(modelica::Statement& statement)
{
	return false;
}
