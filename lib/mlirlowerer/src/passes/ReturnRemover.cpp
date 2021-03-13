#include <modelica/mlirlowerer/passes/ReturnRemover.h>

using namespace modelica;
using namespace std;

void ReturnRemover::fix(modelica::ClassContainer& cls)
{
	cls.visit([&](auto& obj) { fix(obj); });
}

void ReturnRemover::fix(modelica::Class& cls)
{

}

void ReturnRemover::fix(modelica::Function& function)
{
	for (auto& algorithm : function.getAlgorithms())
		fix(*algorithm);
}

void ReturnRemover::fix(modelica::Algorithm& algorithm)
{
	bool canReturn = false;

	for (auto& statement : algorithm)
	{
		if (canReturn)
		{
			Expression reference = Expression::reference(SourcePosition::unknown(), makeType<bool>(), "__mustReturn");
			Expression falseConstant = Expression::constant(SourcePosition::unknown(), makeType<bool>(), false);
			Expression condition = Expression::operation(SourcePosition::unknown(), makeType<bool>(), OperationKind::equal, reference, falseConstant);
			statement = IfStatement(SourcePosition::unknown(), IfStatement::Block(condition, statement));
		}

		canReturn |= fix<modelica::Statement>(statement);
	}

	algorithm.setReturnCheckName("__mustReturn");
}

template<>
bool ReturnRemover::fix<modelica::Statement>(modelica::Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return fix<deconst>(statement);
	});
}

template<>
bool ReturnRemover::fix<modelica::AssignmentStatement>(modelica::Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::fix<modelica::IfStatement>(modelica::Statement& statement)
{
	auto& ifStatement = statement.get<modelica::IfStatement>();
	bool canReturn = false;

	for (auto& block : ifStatement)
	{
		bool blockCanReturn = false;

		for (auto& stmnt : block)
		{
			if (blockCanReturn)
			{
				Expression reference = Expression::reference(SourcePosition::unknown(), makeType<bool>(), "__mustReturn");
				Expression falseConstant = Expression::constant(SourcePosition::unknown(), makeType<bool>(), false);
				Expression condition = Expression::operation(SourcePosition::unknown(), makeType<bool>(), OperationKind::equal, reference, falseConstant);
				stmnt = std::make_shared<Statement>(IfStatement(SourcePosition::unknown(), IfStatement::Block(condition, *stmnt)));
			}

			blockCanReturn |= fix<modelica::Statement>(*stmnt);
		}

		canReturn |= blockCanReturn;
	}

	return canReturn;
}

template<>
bool ReturnRemover::fix<modelica::ForStatement>(modelica::Statement& statement)
{
	auto& forStatement = statement.get<modelica::ForStatement>();
	bool canReturn = false;

	for (auto& stmnt : forStatement)
	{
		if (canReturn)
		{
			Expression reference = Expression::reference(SourcePosition::unknown(), makeType<bool>(), "__mustReturn");
			Expression falseConstant = Expression::constant(SourcePosition::unknown(), makeType<bool>(), false);
			Expression condition = Expression::operation(SourcePosition::unknown(), makeType<bool>(), OperationKind::equal, reference, falseConstant);
			stmnt = std::make_shared<Statement>(IfStatement(SourcePosition::unknown(), IfStatement::Block(condition, *stmnt)));
		}

		canReturn |= fix<modelica::Statement>(*stmnt);
	}

	forStatement.setReturnCheckName("__mustReturn");
	return canReturn;
}

template<>
bool ReturnRemover::fix<modelica::WhileStatement>(modelica::Statement& statement)
{
	auto& whileStatement = statement.get<modelica::WhileStatement>();
	bool canReturn = false;

	for (auto& stmnt : whileStatement)
	{
		if (canReturn)
		{
			Expression reference = Expression::reference(SourcePosition::unknown(), makeType<bool>(), "__mustReturn");
			Expression falseConstant = Expression::constant(SourcePosition::unknown(), makeType<bool>(), false);
			Expression condition = Expression::operation(SourcePosition::unknown(), makeType<bool>(), OperationKind::equal, reference, falseConstant);
			stmnt = std::make_shared<Statement>(IfStatement(SourcePosition::unknown(), IfStatement::Block(condition, *stmnt)));
		}

		canReturn |= fix<modelica::Statement>(*stmnt);
	}

	whileStatement.setReturnCheckName("__mustReturn");
	return canReturn;
}

template<>
bool ReturnRemover::fix<modelica::WhenStatement>(modelica::Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::fix<modelica::BreakStatement>(modelica::Statement& statement)
{
	return false;
}

template<>
bool ReturnRemover::fix<modelica::ReturnStatement>(modelica::Statement& statement)
{
	auto& returnStatement = statement.get<modelica::ReturnStatement>();
	returnStatement.setReturnCheckName("__mustReturn");
	return true;
}
