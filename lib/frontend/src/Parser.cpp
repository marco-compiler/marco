#include "modelica/frontend/Parser.hpp"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "modelica/frontend/Class.hpp"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/ForEquation.hpp"
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
	if (!accept<Token::ModelKeyword>())
		if (!accept<Token::ClassKeyword>())
			EXPECT(Token::PackageKeyword);
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	Class cls(move(name), {}, {});

	while (current != Token::EndKeyword)
	{
		if (current == Token::EquationKeyword)
		{
			TRY(eq, equationSection(cls));
			continue;
		}
		TRY(mem, elementList());
		for (auto& m : *mem)
			cls.addMember(move(m));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::Ident);
	return cls;
}

Expected<SmallVector<size_t, 3>> Parser::arrayDimensions()
{
	SmallVector<size_t, 3> toReturn;
	EXPECT(Token::LSquare);
	do
	{
		toReturn.push_back(lexer.getLastInt());
		EXPECT(Token::Integer);
	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);
	return toReturn;
}

static Expected<BuiltinType> nameToBuiltin(const std::string& name)
{
	if (name == "int")
		return BuiltinType::Integer;
	if (name == "string")
		return BuiltinType::String;
	if (name == "Real")
		return BuiltinType::Float;
	if (name == "Integer")
		return BuiltinType::Integer;
	if (name == "float")
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
	if (current == Token::Ident)
		return Type(*builtint);

	TRY(sub, arrayDimensions());
	return Type(*builtint, move(*sub));
}

Expected<Member> Parser::element()
{
	accept<Token::FinalKeyword>();
	bool parameter = accept<Token::ParameterKeyword>();
	parameter |= accept<Token::ConstantKeyword>();
	TRY(tp, typeSpecifier());
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	optional<Constant> startOverload = nullopt;
	if (current == Token::LPar)
	{
		TRY(start, modification());
		if (*start != Constant(0))
			startOverload = *start;
	}
	if (accept<Token::Equal>())
	{
		TRY(init, expression());

		accept<Token::String>();
		return Member(move(name), move(*tp), move(*init), parameter);
	}
	accept<Token::String>();

	return Member(move(name), move(*tp), parameter, startOverload);
}

Expected<SmallVector<Member, 3>> Parser::elementList()
{
	SmallVector<Member, 3> members;

	while (current != Token::EquationKeyword && current != Token::EndKeyword)
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
		*exp = Expression::op<OperationKind::add>(
				makeType<int>(), move(*exp), Expression(makeType<int>(), -1));
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
Expected<SmallVector<ForEquation, 3>> Parser::forEquationBody(int nestingLevel)
{
	SmallVector<ForEquation, 3> toReturn;
	if (current != Token::ForKeyword)
	{
		TRY(innerEq, equation());
		toReturn.push_back(ForEquation({}, move(*innerEq)));
		return toReturn;
	}

	TRY(innerEq, forEquation(nestingLevel));

	for (auto& eq : *innerEq)
	{
		auto& inductions = eq.getInductions();
		toReturn.push_back(move(eq));
	}
	return toReturn;
}

Expected<SmallVector<ForEquation, 3>> Parser::forEquation(int nestingLevel)
{
	SmallVector<ForEquation, 3> toReturn;
	EXPECT(Token::ForKeyword);
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	EXPECT(Token::InKeyword);
	TRY(begin, expression());
	EXPECT(Token::Colons);
	TRY(end, expression());
	EXPECT(Token::LoopKeyword);

	Induction ind(move(name), move(*begin), move(*end));
	ind.setInductionIndex(nestingLevel);

	while (!accept<Token::EndKeyword>())
	{
		TRY(inner, forEquationBody(nestingLevel + 1));
		for (auto& eq : *inner)
		{
			auto& inds = eq.getInductions();
			inds.insert(inds.begin(), ind);
			toReturn.push_back(eq);
		}
		EXPECT(Token::Semicolons);
	}

	EXPECT(Token::ForKeyword);
	return toReturn;
}

Expected<Equation> Parser::equation()
{
	TRY(l, expression());
	EXPECT(Token::Equal);
	TRY(r, expression());
	accept<Token::String>();
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

Expected<bool> Parser::equationSection(Class& cls)
{
	EXPECT(Token::EquationKeyword);

	while (!sectionTerminator(current))
	{
		if (current == Token::ForKeyword)
		{
			TRY(equs, forEquation(0));
			for (auto& eq : *equs)
				cls.getForEquations().push_back(move(eq));
		}
		else
		{
			TRY(eq, equation());
			cls.getEquations().push_back(move(*eq));
		}
		EXPECT(Token::Semicolons);
	}

	return true;
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

Expected<Constant> Parser::modification()
{
	EXPECT(Token::LPar);

	Constant c(0);
	do
	{
		auto lastIndent = lexer.getLastIdentifier();
		EXPECT(Token::Ident);
		EXPECT(Token::Equal);
		if (lastIndent == "start")
		{
			if (current == Token::FloatingPoint)
			{
				c = Constant(lexer.getLastFloat());
				EXPECT(Token::FloatingPoint);
				continue;
			}

			if (current == Token::Integer)
			{
				c = Constant(lexer.getLastInt());
				EXPECT(Token::Integer);
				continue;
			}
			return make_error<NotImplemented>(
					"start modification must be float or integer");
		}
		if (accept<Token::FloatingPoint>())
			continue;
		if (accept<Token::Integer>())
			continue;
		if (accept<Token::String>())
			continue;

		if (accept<Token::TrueKeyword>())
			continue;

		if (accept<Token::FalseKeyword>())
			continue;

	} while (accept<Token::Comma>());

	EXPECT(Token::RPar);
	return c;
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
	Expression first = negative ? Expression::op<OperationKind::subtract>(
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
				Expression::op<OperationKind::subtract>(Type::unkown(), move(*arg));
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

		if (argumets.size() == 1)
		{
			argumets = { Expression::op<OperationKind::divide>(
					Type::unkown(), move(argumets[0]), move(*arg)) };
			continue;
		}
		auto left =
				Expression::op<OperationKind::multiply>(Type::unkown(), move(argumets));
		argumets = { Expression::op<OperationKind::divide>(
				Type::unkown(), move(left), move(*arg)) };
	}
	if (argumets.size() == 1)
		return move(argumets[0]);

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

	if (accept<Token::LPar>())
	{
		TRY(exp, expression());
		EXPECT(Token::RPar);
		return exp;
	}

	if (accept<Token::DerKeyword>())
	{
		TRY(args, functionCallArguments());
		return makeCall(
				Expression(Type::unkown(), ReferenceAccess("der")), move(*args));
	}

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
