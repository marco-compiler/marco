#include "modelica/frontend/Parser.hpp"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/LexerStateMachine.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/Type.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

Expected<bool> Parser::expect(Token t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedToken>(current, t, getPosition());
}

#include "modelica/utils/ParserUtils.hpp"

Expected<vector<Expression>> Parser::arraySubscript()
{
	EXPECT(Token::LSquare);

	vector<Expression> expressions;

	do
	{
		TRY(exp, expression());
		expressions.emplace_back(move(*exp));
	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);

	return expressions;
}

Expected<Expression> Parser::componentReference()
{
	bool globalLookup = accept<Token::Dot>();
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	Expression exp(Type::unkown(), ReferenceAccess(move(name), globalLookup));

	if (current == Token::LSquare)
	{
		TRY(access, arraySubscript());
		exp = Expression::op<OperationKind::subscription>(
				Type::unkown(), move(*access));
	}

	while (accept<Token::Dot>())
	{
		Expression memberName(makeType<std::string>(), lexer.getLastString());
		EXPECT(Token::String);
		exp = Expression::op<OperationKind::memberLookup>(
				Type::unkown(), move(exp), move(memberName));

		if (current != Token::LSquare)
			continue;

		TRY(access, arraySubscript());
		exp = Expression::op<OperationKind::subscription>(
				Type::unkown(), move(*access));
	}

	return exp;
}

optional<OperationKind> Parser::relationalOperator()
{
	if (accept<Token::GreaterEqual>())
		return OperationKind::greaterEqual;
	if (accept<Token::GreaterThan>())
		return OperationKind::greater;
	if (accept<Token::LessEqual>())
		return OperationKind::lessEqual;
	if (accept<Token::LessThan>())
		return OperationKind::less;
	if (accept<Token::Equal>())
		return OperationKind::equal;
	if (accept<Token::Different>())
		return OperationKind::different;
	return nullopt;
}

Expected<Expression> Parser::expression()
{
	TRY(l, logicalExpression());
	return move(*l);
}

Expected<Expression> Parser::logicalExpression()
{
	vector<Expression> factors;
	TRY(l, logicalFactor());
	if (current != Token::OrKeyword)
		return move(*l);

	factors.push_back(move(*l));
	while (accept<Token::OrKeyword>())
	{
		TRY(arg, logicalFactor());
		factors.emplace_back(move(*arg));
	}
	return Expression::op<OperationKind::lor>(Type::unkown(), move(factors));
}

Expected<Expression> Parser::logicalTerm()
{
	vector<Expression> factors;
	TRY(l, logicalFactor());
	if (current != Token::AndKeyword)
		return move(*l);

	factors.push_back(move(*l));
	while (accept<Token::AndKeyword>())
	{
		TRY(arg, logicalFactor());
		factors.emplace_back(move(*arg));
	}
	return Expression::op<OperationKind::land>(Type::unkown(), move(factors));
}

Expected<Expression> Parser::logicalFactor()
{
	bool negated = accept<Token::NotKeyword>();

	TRY(exp, relation());
	return negated
						 ? Expression::op<OperationKind::negate>(Type::unkown(), move(*exp))
						 : move(*exp);
}

Expected<Expression> Parser::relation()
{
	TRY(left, arithmeticExpression());
	auto op = relationalOperator();
	if (!op.has_value())
		return *left;

	TRY(right, arithmeticExpression());
	return Expression(Type::unkown(), op.value(), move(*left), move(*right));
}

Expected<Expression> Parser::arithmeticExpression()
{
	bool negative = false;
	if (accept<Token::Minus>())
		negative = true;
	else
		accept<Token::Plus>();

	TRY(left, term());
	Expression first = negative ? Expression::op<OperationKind::negate>(
																		Type::unkown(), move(*left))
															: move(*left);
	if (current != Token::Minus && current != Token::Plus)
		return first;

	vector<Expression> args;

	while (current == Token::Minus || current != Token::Plus)
	{
		if (accept<Token::Plus>())
		{
			TRY(arg, term());
			args.push_back(move(*arg));
			continue;
		}

		EXPECT(Token::Minus);
		TRY(arg, term());
		auto exp =
				Expression::op<OperationKind::negate>(Type::unkown(), move(*arg));
		args.emplace_back(move(exp));
	}

	return Expression::op<OperationKind::add>(Type::unkown(), move(args));
}

Expected<Expression> Parser::term()
{
	vector<Expression> argumets;
	TRY(toReturn, factor());
	if (current == Token::Multiply || current == Token::Division)
		return *toReturn;

	argumets.emplace_back(move(*toReturn));

	while (current == Token::Multiply || current == Token::Division)
	{
		if (accept<Token::Multiply>())
		{
			TRY(arg, factor());
			argumets.emplace_back(move(*arg));
			continue;
		}
		EXPECT(Token::Division);
		TRY(arg, factor());
		auto exp = Expression::op<OperationKind::divide>(
				Type::unkown(), Expression(makeType<int>(), 1), move(*arg));
		argumets.emplace_back(move(exp));
	}

	return Expression::op<OperationKind::multiply>(
			Type::unkown(), move(argumets));
}

Expected<Expression> Parser::factor()
{
	TRY(l, primary());
	if (!accept<Token::Exponential>())
		return *l;

	TRY(r, primary());
	return Expression::op<OperationKind::powerOf>(
			Type::unkown(), move(*l), move(*r));
}

Expected<SmallVector<Expression, 3>> Parser::functionCallArguments()
{
	EXPECT(Token::LPar);

	SmallVector<Expression, 3> exps;
	while (!accept<Token::RPar>())
	{
		TRY(arg, expression());
		exps.push_back(move(*arg));
	}

	return exps;
}

Expected<Expression> Parser::primary()
{
	if (current == Token::Integer)
	{
		Constant c(lexer.getLastInt());
		accept<Token::Integer>();
		return Expression(makeType<int>(), c);
	}

	if (current == Token::FloatingPoint)
	{
		Constant c(lexer.getLastFloat());
		accept<Token::FloatingPoint>();
		return Expression(makeType<float>(), c);
	}

	if (current == Token::String)
	{
		Constant s(lexer.getLastString());
		accept<Token::String>();
		return Expression(makeType<std::string>(), s);
	}

	if (accept<Token::TrueKeyword>())
		return Expression::trueExp();

	if (accept<Token::FalseKeyword>())
		return Expression::falseExp();

	if (current == Token::Ident)
	{
		TRY(exp, componentReference());
		if (!accept<Token::LPar>())
			return exp;

		TRY(args, functionCallArguments());
		return makeCall(move(*exp), move(*args));
	}

	return make_error<UnexpectedToken>(current, Token::End, getPosition());
}
