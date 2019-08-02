#include "modelica/Parser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

ExpectedUnique<Statement> Parser::whileStatement()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::WhileKeyword); !e)
		return e.takeError();

	auto exp = expression();
	if (!exp)
		return exp.takeError();

	if (auto e = expect(Token::LoopKeyword); !e)
		return e.takeError();

	auto list = statementList({ Token::EndKeyword });
	if (!list)
		return list.takeError();

	if (auto e = expect(Token::EndKeyword); !e)
		return e.takeError();

	if (auto e = expect(Token::WhileKeyword); !e)
		return e.takeError();

	return makeNode<WhileStatement>(currentPos, move(*exp), move(*list));
}

Expected<vectorUnique<Statement>> Parser::statementList(
		const std::vector<Token>& stopTokens)
{
	vectorUnique<Statement> equations;
	while (find(stopTokens.begin(), stopTokens.end(), current) ==
				 stopTokens.end())
	{
		auto eq = statement();
		if (!eq)
			return eq.takeError();

		if (!expect(Token::Semicolons))
			return make_error<UnexpectedToken>(current);

		equations.push_back(move(*eq));
	}
	return move(equations);
}

Expected<std::pair<UniqueStmt, UniqueExpr>> Parser::ifStmtBrach(
		const std::vector<Token>& stopTokes)
{
	SourcePosition currentPos = getPosition();
	auto expr = expression();
	if (!expr)
		return expr.takeError();

	if (!expect(Token::ThenKeyword))
		return make_error<UnexpectedToken>(current);

	auto ifElseBranchEqus = statementList(stopTokes);
	if (!ifElseBranchEqus)
		return ifElseBranchEqus.takeError();
	auto equations =
			makeNode<CompositeStatement>(currentPos, move(*ifElseBranchEqus));
	if (!equations)
		return equations.takeError();
	return pair(move(*equations), move(*expr));
}

ExpectedUnique<Statement> Parser::ifStatement()
{
	SourcePosition currentPos = getPosition();
	if (!accept<Token::IfKeyword>())
		return make_error<UnexpectedToken>(current);

	vectorUnique<Expr> expressions;
	vectorUnique<Statement> equations;

	auto branch = ifStmtBrach(
			{ Token::EndKeyword, Token::ElseKeyword, Token::ElseIfKeyword });
	if (!branch)
		return branch.takeError();

	expressions.push_back(move(branch->second));
	equations.push_back(move(branch->first));

	while (accept<Token::ElseIfKeyword>())
	{
		auto branch = ifStmtBrach(
				{ Token::EndKeyword, Token::ElseKeyword, Token::ElseIfKeyword });
		if (!branch)
			return branch.takeError();

		expressions.push_back(move(branch->second));
		equations.push_back(move(branch->first));
	}

	if (accept<Token::ElseKeyword>())
	{
		auto elseBranchEqus = statementList({ Token::EndKeyword });
		if (!elseBranchEqus)
			return elseBranchEqus.takeError();

		auto compositeEquation =
				makeNode<CompositeStatement>(currentPos, move(*elseBranchEqus));
		if (!compositeEquation)
			return compositeEquation.takeError();

		equations.push_back(move(*compositeEquation));
	}

	if (!expect(Token::EndKeyword))
		return make_error<UnexpectedToken>(current);
	if (!expect(Token::IfKeyword))
		return make_error<UnexpectedToken>(current);

	return makeNode<IfStatement>(currentPos, move(expressions), move(equations));
}

ExpectedUnique<Statement> Parser::forStatement()
{
	SourcePosition currentPos = getPosition();
	if (!accept<Token::ForKeyword>())
		return make_error<UnexpectedToken>(current);

	auto expression = vector<UniqueExpr>();
	auto names = vector<string>();

	do
	{
		auto pair = forIndex();
		if (!pair)
			return pair.takeError();

		names.emplace_back(move(pair->first));
		expression.emplace_back(move(pair->second));
	} while (accept<Token::Comma>());

	if (!expect(Token::LoopKeyword))
		return make_error<UnexpectedToken>(current);

	auto list = statementList({ Token::EndKeyword });
	if (!list)
		return list.takeError();

	if (!expect(Token::EndKeyword))
		return make_error<UnexpectedToken>(current);

	if (!expect(Token::ForKeyword))
		return make_error<UnexpectedToken>(current);

	return makeNode<ForStatement>(
			currentPos, move(expression), move(*list), move(names));
}

ExpectedUnique<Statement> Parser::whenStatement()
{
	SourcePosition currentPos = getPosition();
	if (!accept<Token::WhenKeyword>())
		return make_error<UnexpectedToken>(current);

	vectorUnique<Expr> expressions;
	vectorUnique<Statement> equations;

	auto branch = ifStmtBrach({ Token::EndKeyword, Token::ElseWhenKeyword });
	if (!branch)
		return branch.takeError();

	expressions.push_back(move(branch->second));
	equations.push_back(move(branch->first));

	if (accept<Token::ElseWhenKeyword>())
	{
		auto elseBranch = ifStmtBrach({ Token::EndKeyword });
		if (!elseBranch)
			return elseBranch.takeError();

		expressions.push_back(move(elseBranch->second));
		equations.push_back(move(elseBranch->first));
	}

	if (!expect(Token::EndKeyword))
		return make_error<UnexpectedToken>(current);
	if (!expect(Token::WhenKeyword))
		return make_error<UnexpectedToken>(current);

	return makeNode<WhenStatement>(
			currentPos, move(expressions), move(equations));
}

ExpectedUnique<Statement> Parser::statement()
{
	SourcePosition currentPos = getPosition();

	if (accept<Token::ReturnKeyword>())
		return makeNode<ReturnStatement>(currentPos);

	if (accept<Token::BreakKeyword>())
		return makeNode<BreakStatement>(currentPos);

	if (current == Token::IfKeyword)
		return ifStatement();

	if (current == Token::ForKeyword)
		return forStatement();

	if (current == Token::WhenKeyword)
		return whenStatement();

	if (current == Token::WhileKeyword)
		return whileStatement();

	ExpectedUnique<Expr> ref = nullptr;
	if (accept<Token::LPar>())
	{
		if (accept<Token::RPar>())
			return llvm::make_error<EmptyList>();
		ref = expressionList();
		if (auto e = expect(Token::RPar); !e)
			return e.takeError();
	}
	else
	{
		ref = componentReference();
	}
	if (!ref)
		return ref.takeError();

	if (accept<Token::Assignment>())
	{
		auto expr = expression();
		if (!expr)
			return expr.takeError();

		return makeNode<AssignStatement>(currentPos, move(*ref), move(*expr));
	}

	auto lPar = expect(Token::LPar);
	if (!lPar)
		return lPar.takeError();

	if (accept<Token::RPar>())
	{
		auto callExpr = makeNode<ComponentFunctionCallExpr>(
				currentPos, vectorUnique<Expr>(), move(*ref));
		if (!callExpr)
			return callExpr.takeError();
		return makeNode<CallStatement>(currentPos, move(*callExpr));
	}

	auto arguments = functionArguments();
	if (!arguments)
		return arguments.takeError();

	if (!expect(Token::RPar))
		return make_error<UnexpectedToken>(current);

	auto callExpr = makeNode<ComponentFunctionCallExpr>(
			currentPos, vectorUnique<Expr>(), move(*ref));

	if (!callExpr)
		return callExpr.takeError();

	return makeNode<CallStatement>(currentPos, move(*callExpr));
}
