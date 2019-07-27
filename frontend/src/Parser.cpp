#include "modelica/Parser.hpp"

#include <cassert>

using namespace modelica;
using namespace llvm;
using namespace std;

llvm::Expected<bool> Parser::expect(Token t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedToken>(current);
}

static void notImplemented() { assert(false); }
llvm::Expected<pair<string, UniqueExpr>> Parser::forIndex()
{
	std::string name = lexer.getLastIdentifier();
	auto id = expect(Token::Ident);
	if (!id)
		return llvm::make_error<UnexpectedToken>(current);

	auto in = expect(Token::InKeyword);
	if (!in)
		return llvm::make_error<UnexpectedToken>(current);

	auto exp = expression();
	if (!exp)
		return exp.takeError();

	return pair(move(name), move(*exp));
}

ExpectedUnique<ArrayConstructorExpr> Parser::arrayArguments()
{
	SourcePosition currentPos = getPosition();
	auto firstExp = expression();
	if (!firstExp)
		return firstExp.takeError();

	if (accept<Token::Comma>())
	{
		auto argsList = expressionList();
		if (!argsList)
			return argsList.takeError();

		(*argsList)->emplace(move(*firstExp));
		return makeNode<DirectArrayConstructorExpr>(currentPos, move(*argsList));
	}

	if (accept<Token::ForKeyword>())
	{
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

		return makeNode<ForInArrayConstructorExpr>(
				currentPos, move(*firstExp), move(expression), move(names));
	}

	auto vector = std::vector<UniqueExpr>();
	vector.emplace_back(move(*firstExp));
	return makeNode<DirectArrayConstructorExpr>(currentPos, move(vector));
}

ExpectedUnique<Expr> functionCallArgs()
{
	notImplemented();
	return make_error<NotImplemented>("function call args was not implemented");
}

ExpectedUnique<Expr> componentReference()
{
	notImplemented();
	return make_error<NotImplemented>("component reference was not implemented");
}

ExpectedUnique<Expr> Parser::expression()
{
	SourcePosition currentPos = getPosition();

	if (accept<Token::IfKeyword>())
	{
		vectorUnique<Expr> vector;
		auto ifCondition = expression();
		if (!ifCondition)
			return ifCondition;

		auto then = expect(Token::ThenKeyword);
		if (!then)
			return then.takeError();

		auto ifExpression = expression();
		if (!ifExpression)
			return ifExpression;

		vector.emplace_back(move(*ifCondition));
		vector.emplace_back(move(*ifExpression));

		while (accept(Token::ElseIfKeyword))
		{
			auto elseIfCondition = expression();
			if (!elseIfCondition)
				return elseIfCondition;

			auto elseIfthen = expect(Token::ThenKeyword);
			if (!elseIfthen)
				return elseIfthen.takeError();

			auto elseIfExpression = expression();
			if (!elseIfExpression)
				return elseIfExpression;

			vector.emplace_back(move(*elseIfCondition));
			vector.emplace_back(move(*elseIfExpression));
		}

		auto expectElse = expect(Token::ElseKeyword);
		if (!expectElse)
			return expectElse.takeError();

		auto elseExpression = expression();
		if (!elseExpression)
			return elseExpression;

		auto toReturn =
				makeNode<IfElseExpr>(currentPos, move(vector), move(*elseExpression));

		return move(toReturn);
	}
	return simpleExpression();
}

ExpectedUnique<Expr> Parser::simpleExpression()
{
	SourcePosition currentPos = getPosition();
	auto first = logicalExpression();
	if (!first || !accept<Token::Colons>())
		return first;

	auto second = logicalExpression();
	if (!second)
		return second;

	if (!accept<Token::Colons>())
		return makeNode<RangeExpr>(currentPos, move(*first), move(*second));

	auto third = logicalExpression();

	if (!third)
		return third;
	return makeNode<RangeExpr>(
			currentPos, move(*first), move(*third), move(*second));
}

ExpectedUnique<Expr> Parser::logicalExpression()
{
	SourcePosition currentPos = getPosition();
	auto leftHandValue = logicalTerm();
	if (!leftHandValue || !accept(Token::OrKeyword))
		return leftHandValue;

	auto rightHandValue = logicalExpression();
	if (!rightHandValue)
		return rightHandValue;

	return makeNode<BinaryExpr>(
			currentPos,
			BinaryExprOp::LogicalOr,
			move(*leftHandValue),
			move(*rightHandValue));
}

ExpectedUnique<Expr> Parser::logicalTerm()
{
	SourcePosition currentPos = getPosition();
	auto leftHandValue = logicalFactor();
	if (!leftHandValue || !accept(Token::AndKeyword))
		return leftHandValue;

	auto rightHandValue = logicalTerm();
	if (!rightHandValue)
		return rightHandValue;

	return makeNode<BinaryExpr>(
			currentPos,
			BinaryExprOp::LogicalAnd,
			move(*leftHandValue),
			move(*rightHandValue));
}

ExpectedUnique<Expr> Parser::logicalFactor()
{
	SourcePosition currentPos = getPosition();
	if (accept<Token::NotKeyword>())
	{
		auto exp = relation();
		if (!exp)
			return exp;

		return makeNode<UnaryExpr>(currentPos, UnaryExprOp::LogicalNot, move(*exp));
	}
	return relation();
}

ExpectedUnique<Expr> Parser::relation()
{
	SourcePosition currentPos = getPosition();

	auto exp = arithmeticExpression();
	if (!exp)
		return exp;

	auto op = relationalOperator();
	if (!op.has_value())
		return exp;

	auto exp2 = arithmeticExpression();
	if (!exp2)
		return exp2;

	return makeNode<BinaryExpr>(currentPos, op.value(), move(*exp), move(*exp2));
}

ExpectedUnique<Expr> Parser::term()
{
	SourcePosition currentPos = getPosition();

	auto leftValue = factor();
	if (!leftValue)
		return leftValue;

	optional<BinaryExprOp> mulOp;
	while ((mulOp = maybeMulOperator()).has_value())
	{
		auto rightValue = factor();
		if (!rightValue)
			return rightValue;

		leftValue = makeNode<BinaryExpr>(
				currentPos, mulOp.value(), move(*leftValue), move(*rightValue));
	}

	return leftValue;
}
ExpectedUnique<Expr> Parser::factor()
{
	SourcePosition currentPos = getPosition();
	auto left = primary();
	bool exp =
			accept<Token::Exponential>() || accept<Token::ElementWiseExponential>();

	if (!exp)
		return left;

	auto right = primary();
	if (!right)
		return right;

	return makeNode<BinaryExpr>(
			currentPos, BinaryExprOp::PowerOf, move(*left), move(*right));
}

ExpectedUnique<Expr> Parser::arithmeticExpression()
{
	SourcePosition currentPos = getPosition();

	auto unaryOp = maybeUnaryAddOperator();

	auto current = term();
	if (!current)
		return current;

	if (unaryOp.has_value())
	{
		current = makeNode<UnaryExpr>(currentPos, unaryOp.value(), move(*current));
		if (!current)
			return current;
	}

	optional<BinaryExprOp> nextOp = nullopt;
	while ((nextOp = maybeAddOperator()).has_value())
	{
		auto rightHandValue = term();
		if (!rightHandValue)
			return rightHandValue;

		current = makeNode<BinaryExpr>(
				currentPos, nextOp.value(), move(*current), move(*rightHandValue));

		if (!current)
			return current;
	}

	return current;
}

optional<BinaryExprOp> Parser::maybeMulOperator()
{
	if (accept<Token::Multiply>())
		return BinaryExprOp::Multiply;
	if (accept<Token::Division>())
		return BinaryExprOp::Division;
	if (accept<Token::ElementWiseDivision>())
		return BinaryExprOp::Division;
	if (accept<Token::ElementWiseMultilpy>())
		return BinaryExprOp::Multiply;
	return nullopt;
}

optional<BinaryExprOp> Parser::maybeAddOperator()
{
	if (accept<Token::Plus>())
		return BinaryExprOp::Sum;
	if (accept<Token::Minus>())
		return BinaryExprOp::Subtraction;
	if (accept<Token::ElementWiseSum>())
		return BinaryExprOp::Sum;
	if (accept<Token::ElementWiseMinus>())
		return BinaryExprOp::Subtraction;
	return nullopt;
}

optional<UnaryExprOp> Parser::maybeUnaryAddOperator()
{
	if (accept<Token::Plus>())
		return UnaryExprOp::Plus;
	if (accept<Token::Minus>())
		return UnaryExprOp::Minus;
	if (accept<Token::ElementWiseSum>())
		return UnaryExprOp::Plus;
	if (accept<Token::ElementWiseMinus>())
		return UnaryExprOp::Minus;
	return nullopt;
}

optional<BinaryExprOp> Parser::relationalOperator()
{
	if (accept<Token::LessEqual>())
		return BinaryExprOp::LessEqual;
	if (accept<Token::GreaterEqual>())
		return BinaryExprOp::GreatureEqual;
	if (accept<Token::LessThan>())
		return BinaryExprOp::Less;
	if (accept<Token::GreaterThan>())
		return BinaryExprOp::Greater;
	if (accept<Token::OperatorEqual>())
		return BinaryExprOp::Equal;
	if (accept<Token::Different>())
		return BinaryExprOp::Different;
	return nullopt;
}

ExpectedUnique<ExprList> Parser::expressionList()
{
	SourcePosition currentPos = getPosition();

	vector<unique_ptr<Expr>> vector;
	auto exp1 = expression();

	if (!exp1)
		return exp1.takeError();

	vector.emplace_back(move(*exp1));
	while (accept<Token::Comma>())
	{
		auto exp = expression();
		if (!exp)
			return exp.takeError();

		vector.emplace_back(move(*exp));
	}
	return makeNode<ExprList>(currentPos, move(vector));
}

ExpectedUnique<Expr> Parser::primary()
{
	SourcePosition currentPos = getPosition();

	if (accept<Token::Integer>())
		return makeNode<IntLiteralExpr>(currentPos, lexer.getLastInt());

	if (accept<Token::FloatingPoint>())
		return makeNode<FloatLiteralExpr>(currentPos, lexer.getLastFloat());

	if (accept<Token::String>())
		return makeNode<StringLiteralExpr>(currentPos, lexer.getLastString());

	if (accept<Token::FalseKeyword>())
		return makeNode<BoolLiteralExpr>(currentPos, false);

	if (accept<Token::TrueKeyword>())
		return makeNode<BoolLiteralExpr>(currentPos, true);

	if (accept<Token::LPar>())
	{
		if (accept<Token::RPar>())
			return makeNode<ExprList>(currentPos);

		auto expList = expressionList();
		if (!expList)
			return expList;

		auto closedPar = expect(Token::RPar);
		if (!closedPar)
			return make_error<UnexpectedToken>(current);

		return expList;
	}

	if (accept<Token::LSquare>())
	{
		if (accept<Token::RSquare>())
			return llvm::make_error<EmptyList>();
		auto firstList = expressionList();
		if (!firstList)
			return firstList;

		vectorUnique<ExprList> vector;
		vector.emplace_back(move(*firstList));
		while (accept<Token::Semicolons>())
		{
			auto list = expressionList();
			if (!list)
				return list;

			vector.emplace_back(move(*list));
		}

		auto closedPar = expect(Token::RSquare);
		if (!closedPar)
			return llvm::make_error<UnexpectedToken>(current);

		return makeNode<ArrayConcatExpr>(currentPos, move(vector));
	}

	if (accept<Token::LCurly>())
	{
		auto args = arrayArguments();

		if (!args)
			return args;

		auto closedPar = expect(Token::RCurly);
		if (!closedPar)
			return llvm::make_error<UnexpectedToken>(current);

		return args;
	}

	if (accept<Token::DerKeyword>())
		return functionCallArgs();

	if (accept<Token::InitialKeyword>())
		return functionCallArgs();

	if (accept<Token::PureKeyword>())
		return functionCallArgs();

	if (current != Token::Dot)
		return make_error<UnexpectedToken>(current);

	if (auto exprOrErr = componentReference())
	{
		if (current == Token::LPar)
			return functionCallArgs();
		return exprOrErr;
	}
	else
		return exprOrErr;
}
