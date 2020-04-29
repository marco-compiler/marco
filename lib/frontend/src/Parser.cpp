#include "modelica/frontend/Parser.hpp"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "modelica/frontend/Class.hpp"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/LexerStateMachine.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/ParserErrors.hpp"
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

Expected<Class> Parser::classDefinition()
{
	EXPECT(Token::ModelKeyword);
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	SmallVector<Equation, 3> equs;
	SmallVector<Member, 3> members;

	while (current != Token::EndKeyword)
	{
		if (current == Token::EquationKeyword)
		{
			TRY(eq, equationSection());
			for (auto& e : *eq)
				equs.emplace_back(move(e));
		}
		TRY(mem, elementList());
		for (auto& m : *mem)
			members.emplace_back(move(m));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::Ident);
	return Class(move(name), move(members), move(equs));
}

Expected<SmallVector<size_t, 3>> Parser::arrayDimensions()
{
	SmallVector<size_t, 3> toReturn;
	EXPECT(Token::LPar);
	do
	{
		toReturn.push_back(lexer.getLastInt());
		EXPECT(Token::Integer);
	} while (accept<Token::Comma>());

	EXPECT(Token::RPar);
	return toReturn;
}

static Expected<BuiltinType> nameToBuiltin(const std::string& name)
{
	if (name == "int")
		return BuiltinType::Integer;
	if (name == "string")
		return BuiltinType::String;
	if (name == "real")
		return BuiltinType::Float;
	if (name == "bool")
		return BuiltinType::Boolean;

	return make_error<NotImplemented>(
			"only builtin types are supported not " + name);
}

Expected<Type> Parser::typeSpecifier()
{
	std::string name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	TRY(builtint, nameToBuiltin(name));
	if (current != Token::Ident)
		return Type(*builtint);

	TRY(sub, arrayDimensions());
	return Type(*builtint, move(*sub));
}

Expected<Member> Parser::element()
{
	bool parameter = accept<Token::ParameterKeyword>();
	TRY(tp, typeSpecifier());
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	if (accept<Token::Equal>())
	{
		TRY(init, expression());
		return Member(move(name), move(*tp), move(*init), parameter);
	}

	return Member(move(name), move(*tp), parameter);
}

Expected<SmallVector<Member, 3>> Parser::elementList()
{
	SmallVector<Member, 3> members;

	while (current != Token::EquationKeyword && current != Token::End)
	{
		TRY(memb, element());
		EXPECT(Token::Semicolons);
		members.emplace_back(move(*memb));
	}
	return members;
}

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
		access->insert(access->begin(), move(exp));
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

Expected<Equation> Parser::equation()
{
	TRY(l, expression());
	EXPECT(Token::Equal);
	TRY(r, expression());
	return Equation(move(*l), move(*r));
}

static bool sectionTerminator(Token current)
{
	if (current == Token::AlgorithmKeyword)
		return true;

	if (current == Token::EquationKeyword)
		return true;

	if (current == Token::PublicKeyword)
		return true;

	if (current == Token::EndKeyword)
		return true;

	if (current == Token::LPar)
		return true;

	if (current == Token::End)
		return true;
	return false;
}

Expected<SmallVector<Equation, 3>> Parser::equationSection()
{
	EXPECT(Token::EquationKeyword);

	SmallVector<Equation, 3> equations;

	while (!sectionTerminator(current))
	{
		TRY(eq, equation());
		equations.emplace_back(move(*eq));
		EXPECT(Token::Semicolons);
	}

	return equations;
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
	if (accept<Token::OperatorEqual>())
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
	TRY(l, logicalTerm());
	if (current != Token::OrKeyword)
		return move(*l);

	factors.push_back(move(*l));
	while (accept<Token::OrKeyword>())
	{
		TRY(arg, logicalTerm());
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
	args.push_back(move(first));

	while (current == Token::Minus || current == Token::Plus)
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
	if (current != Token::Multiply && current != Token::Division)
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
