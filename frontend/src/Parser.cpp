#include "modelica/Parser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

llvm::Expected<bool> Parser::expect(Token t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedToken>(current, t);
}

llvm::Expected<pair<string, UniqueExpr>> Parser::forIndex()
{
	std::string name = lexer.getLastIdentifier();
	auto id = expect(Token::Ident);
	if (!id)
		return id.takeError();

	auto in = expect(Token::InKeyword);
	if (!in)
		return in.takeError();

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

Expected<vectorUnique<Expr>> Parser::functionArguments()
{
	SourcePosition currentPos = getPosition();
	vectorUnique<Expr> arguments;

	if (current == Token::FunctionKeyword)
	{
		auto call = partialCall();
		if (!call)
			return call.takeError();

		arguments.emplace_back(move(*call));
		if (!accept<Token::Comma>())
			return move(arguments);

		auto otherArgs = functionArgumentsNonFirst();
		if (!otherArgs)
			return otherArgs.takeError();

		move(otherArgs->begin(), otherArgs->end(), back_inserter(arguments));
		return move(arguments);
	}
	if (current == Token::Ident)
	{
		auto args = namedArguments();
		if (args)
			return args;
	}

	auto firstExp = expression();
	if (!firstExp)
		return firstExp.takeError();

	if (accept<Token::Comma>())
	{
		arguments.emplace_back(move(*firstExp));
		auto otherArgs = functionArgumentsNonFirst();
		if (!otherArgs)
			return otherArgs;

		move(otherArgs->begin(), otherArgs->end(), back_inserter(arguments));
		return move(arguments);
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

		auto node = makeNode<ForInArrayConstructorExpr>(
				currentPos, move(*firstExp), move(expression), move(names));
		if (!node)
			return node.takeError();

		arguments.emplace_back(move(*node));
		return move(arguments);
	}
	arguments.emplace_back(move(*firstExp));
	return move(arguments);
}

Expected<vectorUnique<Expr>> Parser::functionArgumentsNonFirst()
{
	vectorUnique<Expr> toReturn;

	if (current == Token::Ident)
	{
		auto args = namedArguments();
		if (args)
			return args;
	}

	auto arg1 = functionArgument();
	if (!arg1)
		return arg1.takeError();

	toReturn.emplace_back(move(*arg1));

	if (!accept<Token::Comma>())
		return move(toReturn);

	Expected<vectorUnique<Expr>> nextArgs = vectorUnique<Expr>();
	if (current == Token::Ident)
	{
		nextArgs = namedArguments();
		if (!nextArgs)
			nextArgs = functionArgumentsNonFirst();
	}
	else
		nextArgs = functionArgumentsNonFirst();
	if (!nextArgs)
		return nextArgs;

	move(nextArgs->begin(), nextArgs->end(), back_inserter(toReturn));
	return move(toReturn);
}

Expected<vectorUnique<Expr>> Parser::namedArguments()
{
	auto first = namedArgument();
	if (!first)
		return first.takeError();

	vectorUnique<Expr> toReturn;
	toReturn.emplace_back(move(*first));

	while (accept<Token::Comma>())
	{
		auto next = namedArgument();
		if (!next)
			return next.takeError();

		toReturn.emplace_back(move(*next));
	}

	return move(toReturn);
}

ExpectedUnique<NamedArgumentExpr> Parser::namedArgument()
{
	SourcePosition currentPos = getPosition();
	string name = lexer.getLastIdentifier();
	if (auto e = expect(Token::Ident); !e)
		return e.takeError();
	if (auto e = expect(Token::Equal); !e)
	{
		undoScan(Token::Ident);
		return e.takeError();
	}

	auto exp = functionArgument();
	if (!exp)
		return exp.takeError();

	return makeNode<NamedArgumentExpr>(currentPos, move(name), move(*exp));
}

Expected<vector<string>> Parser::name()
{
	vector<string> toReturn;
	toReturn.push_back(lexer.getLastIdentifier());
	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	while (accept<Token::Dot>())
	{
		toReturn.push_back(lexer.getLastIdentifier());
		if (auto e = expect(Token::Ident); !e)
			return e.takeError();
	}

	return move(toReturn);
}

ExpectedUnique<Expr> Parser::partialCall()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::FunctionKeyword); !e)
		return e.takeError();

	auto nm = name();
	if (!nm)
		return nm.takeError();

	if (auto e = expect(Token::LPar); !e)
		return e.takeError();

	if (accept(Token::RPar))
		return makeNode<PartialFunctioCallExpr>(
				currentPos, vectorUnique<Expr>(), move(*nm));

	auto args = namedArguments();
	if (!args)
		return args.takeError();

	if (auto e = expect(Token::RPar); !e)
		return e.takeError();

	return makeNode<PartialFunctioCallExpr>(currentPos, move(*args), move(*nm));
}

ExpectedUnique<Expr> Parser::functionArgument()
{
	return (current == Token::FunctionKeyword) ? partialCall() : expression();
}

ExpectedUnique<Expr> Parser::componentReference()
{
	SourcePosition currentPos = getPosition();
	bool hasDot = accept<Token::Dot>();
	UniqueExpr toReturn;

	auto id = lexer.getLastIdentifier();
	if (auto parseId = expect(Token::Ident); !parseId)
		return parseId.takeError();

	auto node =
			makeNode<ComponentReferenceExpr>(currentPos, move(id), nullptr, hasDot);
	if (!node)
		return node;

	toReturn = move(*node);

	if (accept<Token::LSquare>())
	{
		auto subScriptVector = arraySubscript();
		if (!subScriptVector)
			return subScriptVector.takeError();

		auto newNode = makeNode<ArraySubscriptionExpr>(
				currentPos, move(toReturn), move(*subScriptVector));
		if (!newNode)
			return newNode.takeError();

		toReturn = move(*newNode);

		if (auto e = expect(Token::RSquare); !e)
			return e.takeError();
	}

	while (accept<Token::Dot>())
	{
		auto id = lexer.getLastIdentifier();
		if (auto parsedId = expect(Token::Ident); !parsedId)
			return parsedId.takeError();

		node = makeNode<ComponentReferenceExpr>(
				currentPos, move(id), move(toReturn), false);
		if (!node)
			return node;

		toReturn = move(*node);

		if (accept<Token::LSquare>())
		{
			auto subScriptVector = arraySubscript();
			if (!subScriptVector)
				return subScriptVector.takeError();

			auto newNode = makeNode<ArraySubscriptionExpr>(
					currentPos, move(toReturn), move(*subScriptVector));
			if (!newNode)
				return newNode.takeError();

			toReturn = move(*newNode);

			if (auto e = expect(Token::RSquare); !e)
				return e.takeError();
		}
	}

	return move(toReturn);
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

ExpectedUnique<Expr> Parser::subScript()
{
	SourcePosition currentPos = getPosition();
	if (accept<Token::Colons>())
		return makeNode<AcceptAllExpr>(currentPos);

	return expression();
}

Expected<vectorUnique<Expr>> Parser::arraySubscript()
{
	auto subscript = subScript();
	if (!subscript)
		return subscript.takeError();

	vectorUnique<Expr> vector;
	vector.emplace_back(move(*subscript));
	while (accept<Token::Comma>())
	{
		subscript = subScript();
		if (!subscript)
			return subscript.takeError();

		vector.emplace_back(move(*subscript));
	}

	return move(vector);
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

		if (auto e = expect(Token::RPar); !e)
			return e.takeError();

		return expList;
	}

	if (accept<Token::EndKeyword>())
		return makeNode<EndExpr>(currentPos);

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

		if (auto e = expect(Token::RSquare); !e)
			return e.takeError();

		return makeNode<ArrayConcatExpr>(currentPos, move(vector));
	}

	if (accept<Token::LCurly>())
	{
		auto args = arrayArguments();

		if (!args)
			return args;

		if (auto e = expect(Token::RCurly); !e)
			return e.takeError();

		return args;
	}

	if (auto derCall = functionCall<Token::DerKeyword, DerFunctionCallExpr>();
			derCall.has_value())
		return move(derCall.value());

	if (auto initCall =
					functionCall<Token::InitialKeyword, InitialFunctionCallExpr>();
			initCall.has_value())
		return move(initCall.value());

	if (auto pureCall = functionCall<Token::PureKeyword, PureFunctionCallExpr>();
			pureCall.has_value())
		return move(pureCall.value());

	auto exprOrErr = componentReference();
	if (!exprOrErr)
		return exprOrErr;

	if (accept<Token::LPar>())
	{
		if (accept<Token::RPar>())
			return makeNode<ComponentFunctionCallExpr>(
					currentPos, vectorUnique<Expr>(), move(*exprOrErr));

		auto arguments = functionArguments();
		if (!arguments)
			return arguments.takeError();

		if (auto e = expect(Token::RPar); !e)
			return e.takeError();

		return makeNode<ComponentFunctionCallExpr>(
				currentPos, move(*arguments), move(*exprOrErr));
	}

	return exprOrErr;
}
