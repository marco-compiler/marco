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

		for (auto& stmnt : block)
		{
			if (blockBreakable)
			{
				Expression reference = Expression::reference(SourcePosition::unknown(), makeType<bool>(), "__mustBreak" + to_string(nestLevel));
				Expression falseConstant = Expression::constant(SourcePosition::unknown(), makeType<bool>(), false);
				Expression condition = Expression::operation(SourcePosition::unknown(), makeType<bool>(), OperationKind::equal, reference, falseConstant);
				stmnt = IfStatement(SourcePosition::unknown(), IfStatement::Block(condition, stmnt));
			}

			blockBreakable |= fix<modelica::Statement>(stmnt);
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

	for (auto& stmnt : forStatement)
	{
		if (breakable)
		{
			Expression reference = Expression::reference(SourcePosition::unknown(), makeType<bool>(), "__mustBreak" + to_string(nestLevel));
			Expression falseConstant = Expression::constant(SourcePosition::unknown(), makeType<bool>(), false);
			Expression condition = Expression::operation(SourcePosition::unknown(), makeType<bool>(), OperationKind::equal, reference, falseConstant);
			stmnt = IfStatement(SourcePosition::unknown(), IfStatement::Block(condition, stmnt));
		}

		breakable |= fix<modelica::Statement>(stmnt);
	}

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

	for (auto& stmnt : whileStatement)
	{
		if (breakable)
		{
			Expression reference = Expression::reference(SourcePosition::unknown(), makeType<bool>(), "__mustBreak" + to_string(nestLevel));
			Expression falseConstant = Expression::constant(SourcePosition::unknown(), makeType<bool>(), false);
			Expression condition = Expression::operation(SourcePosition::unknown(), makeType<bool>(), OperationKind::equal, reference, falseConstant);
			stmnt = IfStatement(SourcePosition::unknown(), IfStatement::Block(condition, stmnt));
		}

		breakable |= fix<modelica::Statement>(stmnt);
	}

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
