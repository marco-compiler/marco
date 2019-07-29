#include "modelica/Parser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

Expected<vectorUnique<Equation>> Parser::equationList(
		const std::vector<Token>& stopTokens)
{
	vectorUnique<Equation> equations;
	while (find(stopTokens.begin(), stopTokens.end(), current) ==
				 stopTokens.end())
	{
		auto eq = equation();
		if (!eq)
			return eq.takeError();

		if (!expect(Token::Semicolons))
			return make_error<UnexpectedToken>(current);

		equations.push_back(move(*eq));
	}
	return move(equations);
}

Expected<std::pair<UniqueEq, UniqueExpr>> Parser::ifBrach(
		const std::vector<Token>& stopTokes)
{
	SourcePosition currentPos = getPosition();
	auto expr = expression();
	if (!expr)
		return expr.takeError();

	if (!expect(Token::ThenKeyword))
		return make_error<UnexpectedToken>(current);

	auto ifElseBranchEqus = equationList(stopTokes);
	if (!ifElseBranchEqus)
		return ifElseBranchEqus.takeError();
	auto equations =
			makeNode<CompositeEquation>(currentPos, move(*ifElseBranchEqus));
	if (!equations)
		return equations.takeError();
	return pair(move(*equations), move(*expr));
}

ExpectedUnique<Equation> Parser::ifEquation()
{
	SourcePosition currentPos = getPosition();
	if (!accept<Token::IfKeyword>())
		return make_error<UnexpectedToken>(current);

	vectorUnique<Expr> expressions;
	vectorUnique<Equation> equations;

	auto branch =
			ifBrach({ Token::EndKeyword, Token::ElseKeyword, Token::ElseIfKeyword });
	if (!branch)
		return branch.takeError();

	expressions.push_back(move(branch->second));
	equations.push_back(move(branch->first));

	while (accept<Token::ElseIfKeyword>())
	{
		auto branch = ifBrach(
				{ Token::EndKeyword, Token::ElseKeyword, Token::ElseIfKeyword });
		if (!branch)
			return branch.takeError();

		expressions.push_back(move(branch->second));
		equations.push_back(move(branch->first));
	}

	if (accept<Token::ElseKeyword>())
	{
		auto elseBranchEqus = equationList({ Token::EndKeyword });
		if (!elseBranchEqus)
			return elseBranchEqus.takeError();

		auto compositeEquation =
				makeNode<CompositeEquation>(currentPos, move(*elseBranchEqus));
		if (!compositeEquation)
			return compositeEquation.takeError();

		equations.push_back(move(*compositeEquation));
	}

	if (!expect(Token::EndKeyword))
		return make_error<UnexpectedToken>(current);
	if (!expect(Token::IfKeyword))
		return make_error<UnexpectedToken>(current);

	return makeNode<IfEquation>(currentPos, move(expressions), move(equations));
}

ExpectedUnique<Equation> Parser::forEquation()
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

	auto list = equationList({ Token::EndKeyword });
	if (!list)
		return list.takeError();

	if (!expect(Token::EndKeyword))
		return make_error<UnexpectedToken>(current);

	if (!expect(Token::ForKeyword))
		return make_error<UnexpectedToken>(current);

	return makeNode<ForEquation>(
			currentPos, move(expression), move(*list), move(names));
}

ExpectedUnique<Equation> Parser::whenEquation()
{
	SourcePosition currentPos = getPosition();
	if (!accept<Token::WhenKeyword>())
		return make_error<UnexpectedToken>(current);

	vectorUnique<Expr> expressions;
	vectorUnique<Equation> equations;

	auto branch = ifBrach({ Token::EndKeyword, Token::ElseWhenKeyword });
	if (!branch)
		return branch.takeError();

	expressions.push_back(move(branch->second));
	equations.push_back(move(branch->first));

	if (accept<Token::ElseWhenKeyword>())
	{
		auto elseBranch = ifBrach({ Token::EndKeyword });
		if (!elseBranch)
			return elseBranch.takeError();

		expressions.push_back(move(elseBranch->second));
		equations.push_back(move(elseBranch->first));
	}

	if (!expect(Token::EndKeyword))
		return make_error<UnexpectedToken>(current);
	if (!expect(Token::WhenKeyword))
		return make_error<UnexpectedToken>(current);

	return makeNode<WhenEquation>(currentPos, move(expressions), move(equations));
}

ExpectedUnique<Equation> Parser::equation()
{
	SourcePosition currentPos = getPosition();
	if (current == Token::IfKeyword)
		return ifEquation();

	if (current == Token::ForKeyword)
		return forEquation();

	if (current == Token::WhenKeyword)
		return whenEquation();

	if (accept<Token::ConnectKeyword>())
	{
		if (!expect(Token::LPar))
			return make_error<UnexpectedToken>(current);

		auto firstParam = componentReference();
		if (!firstParam)
			return firstParam.takeError();

		if (!expect(Token::Comma))
			return make_error<UnexpectedToken>(current);

		auto secondParam = componentReference();
		if (!secondParam)
			return secondParam.takeError();

		if (!expect(Token::RPar))
			return make_error<UnexpectedToken>(current);

		return makeNode<ConnectClause>(
				currentPos, move(*firstParam), move(*secondParam));
	}

	if (current == Token::Ident)
	{
		if (lexer.getLastIdentifier() == "assert" ||
				lexer.getLastIdentifier() == "terminate")
		{
			auto functionCall = primary();
			if (!functionCall)
				return functionCall.takeError();

			if (!llvm::isa<ComponentFunctionCallExpr>(functionCall->get()))
				return make_error<UnexpectedToken>(current);

			return makeNode<CallEquation>(currentPos, move(*functionCall));
		}
	}

	auto exp1 = simpleExpression();
	if (!exp1)
		return exp1.takeError();

	if (!expect(Token::Equal))
		return make_error<UnexpectedToken>(current);

	auto exp2 = expression();
	if (!exp2)
		return exp2.takeError();

	return makeNode<SimpleEquation>(currentPos, move(*exp1), move(*exp2));
}
