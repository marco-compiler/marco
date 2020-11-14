#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Class.hpp>
#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/ForEquation.hpp>
#include <modelica/frontend/LexerStateMachine.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/frontend/ReferenceAccess.hpp>
#include <modelica/frontend/Type.hpp>
#include <optional>

using namespace llvm;
using namespace modelica;
using namespace std;

Parser::Parser(const string& source)
		: lexer(source), current(lexer.scan()), undo(Token::End)
{
}

Parser::Parser(const char* source)
		: lexer(source), current(lexer.scan()), undo(Token::End)
{
}

SourcePosition Parser::getPosition() const
{
	return SourcePosition(lexer.getCurrentLine(), lexer.getCurrentColumn());
}

Token Parser::getCurrentToken() const { return current; }

bool Parser::accept(Token t)
{
	if (current == t)
	{
		next();
		return true;
	}

	return false;
}

void Parser::next()
{
	if (undo != Token::End)
	{
		current = undo;
		undo = Token::End;
		return;
	}

	current = lexer.scan();
}

Expected<bool> Parser::expect(Token t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedToken>(current, t, getPosition());
}

void Parser::undoScan(Token t)
{
	undo = current;
	current = t;
}

#include <modelica/utils/ParserUtils.hpp>

/**
 * Check whether the current token leads to the termination of the current
 * section (and eventually to the start of a new one).
 */
static bool sectionTerminator(Token current)
{
	return current == Token::AlgorithmKeyword ||
				 current == Token::EquationKeyword || current == Token::PublicKeyword ||
				 current == Token::EndKeyword || current == Token::LPar ||
				 current == Token::End;
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
			"Only builtin types are supported, not " + name);
}

Expected<Class> Parser::classDefinition()
{
	ClassType type = ClassType::Class;

	if (accept(Token::BlockKeyword))
		type = ClassType::Block;
	else if (accept(Token::ConnectorKeyword))
		type = ClassType::Connector;
	else if (accept(Token::FunctionKeyword))
		type = ClassType::Function;
	else if (accept(Token::ModelKeyword))
		type = ClassType::Model;
	else if (accept(Token::PackageKeyword))
		type = ClassType::Package;
	else if (accept(Token::OperatorKeyword))
		type = ClassType::Operator;
	else if (accept(Token::RecordKeyword))
		type = ClassType::Record;
	else if (accept(Token::TypeKeyword))
		type = ClassType::Type;
	else
		EXPECT(Token::ClassKeyword);

	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	accept(Token::String);

	Class cls(type, name);

	// Whether the first elements list is allowed to be encountered or not.
	// In fact, the class definition allows a first elements list definition
	// and then others more if preceded by "public" or "protected", but no more
	// "lone" definitions are anymore allowed if any of those keywords are
	// encountered.
	bool firstElementListParsable = true;

	// absorb comments after class/model/package
	accept<Token::String>();

	while (current != Token::EndKeyword)
	{
		if (current == Token::EquationKeyword)
		{
			TRY(eq, equationSection(cls));
			continue;
		}

		if (current == Token::AlgorithmKeyword)
		{
			TRY(alg, algorithmSection());
			cls.addAlgorithm(move(*alg));
			continue;
		}

		if (current == Token::FunctionKeyword)
		{
			TRY(func, classDefinition());
			cls.addFunction(move(*func));
			EXPECT(Token::Semicolons);
			continue;
		}

		if (accept(Token::PublicKeyword))
		{
			TRY(mem, elementList());
			firstElementListParsable = false;

			for (auto& m : *mem)
				cls.addMember(move(m));

			continue;
		}

		if (accept(Token::ProtectedKeyword))
		{
			TRY(mem, elementList());
			firstElementListParsable = false;

			for (auto& m : *mem)
				cls.addMember(move(m));

			continue;
		}

		if (firstElementListParsable)
		{
			TRY(mem, elementList());

			for (auto& m : *mem)
				cls.addMember(move(m));
		}
	}

	EXPECT(Token::EndKeyword);
	auto endName = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	if (name != endName)
		return make_error<UnexpectedIdentifier>(endName, name, getPosition());

	return cls;
}

Expected<SmallVector<Member, 3>> Parser::elementList()
{
	SmallVector<Member, 3> members;

	while (
			current != Token::PublicKeyword && current != Token::ProtectedKeyword &&
			current != Token::FunctionKeyword && current != Token::EquationKeyword &&
			current != Token::AlgorithmKeyword && current != Token::EndKeyword)
	{
		TRY(memb, element());
		EXPECT(Token::Semicolons);
		members.emplace_back(move(*memb));
	}

	return members;
}

Expected<Member> Parser::element(bool publicSection)
{
	accept<Token::FinalKeyword>();
	TRY(prefix, typePrefix());
	TRY(type, typeSpecifier());
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	optional<Expression> startOverload = nullopt;

	if (current == Token::LPar)
	{
		TRY(start, modification());

		if (start->has_value())
			startOverload = move(**start);
	}

	if (accept<Token::Equal>())
	{
		TRY(init, expression());

		accept<Token::String>();
		return Member(
				move(name), move(*type), move(*prefix), move(*init), publicSection);
	}
	accept<Token::String>();

	return Member(
			move(name), move(*type), move(*prefix), publicSection, startOverload);
}

Expected<optional<Expression>> Parser::modification()
{
	EXPECT(Token::LPar);

	optional<Expression> e = nullopt;

	do
	{
		auto lastIndent = lexer.getLastIdentifier();
		EXPECT(Token::Ident);
		EXPECT(Token::Equal);
		if (lastIndent == "start")
		{
			TRY(exp, expression());
			e = move(*exp);
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
	return e;
}

Expected<TypePrefix> Parser::typePrefix()
{
	ParameterQualifier parameterQualifier = ParameterQualifier::none;

	if (accept<Token::DiscreteKeyword>())
		parameterQualifier = ParameterQualifier::discrete;
	else if (accept<Token::ParameterKeyword>())
		parameterQualifier = ParameterQualifier::parameter;
	else if (accept<Token::ConstantKeyword>())
		parameterQualifier = ParameterQualifier::constant;

	IOQualifier ioQualifier = IOQualifier::none;

	if (accept<Token::InputKeyword>())
		ioQualifier = IOQualifier::input;
	else if (accept<Token::OutputKeyword>())
		ioQualifier = IOQualifier::output;

	return TypePrefix(parameterQualifier, ioQualifier);
}

Expected<Type> Parser::typeSpecifier()
{
	string name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	TRY(builtint, nameToBuiltin(name));

	if (current == Token::Ident)
		return Type(*builtint);

	TRY(sub, arrayDimensions());
	return Type(*builtint, move(*sub));
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
				cls.addForEquation(move(eq));
		}
		else
		{
			TRY(eq, equation());
			cls.addEquation(move(*eq));
		}

		EXPECT(Token::Semicolons);
	}

	return true;
}

Expected<Algorithm> Parser::algorithmSection()
{
	EXPECT(Token::AlgorithmKeyword);
	SmallVector<Statement, 3> statements;

	while (
			current != Token::End && current != Token::PublicKeyword &&
			current != Token::ProtectedKeyword && current != Token::EquationKeyword &&
			current != Token::AlgorithmKeyword && current != Token::ExternalKeyword &&
			current != Token::AnnotationKeyword && current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(move(*stmnt));
	}

	return Algorithm(statements);
}

Expected<Equation> Parser::equation()
{
	TRY(l, expression());
	EXPECT(Token::Equal);
	TRY(r, expression());
	accept<Token::String>();
	return Equation(move(*l), move(*r));
}

Expected<Statement> Parser::statement()
{
	if (accept(Token::LPar))
	{
		vector<Expression> destinations;

		while (!accept<Token::RPar>())
		{
			if (accept<Token::Comma>())
			{
				destinations.emplace_back(Type::unknown(), ReferenceAccess::dummy());
				continue;
			}

			TRY(dest, expression());
			destinations.push_back(move(*dest));
			accept(Token::Comma);
		}

		EXPECT(Token::Assignment);
		TRY(functionName, componentReference());
		TRY(args, functionCallArguments());
		Expression call = makeCall(move(*functionName), move(*args));

		return Statement(destinations.begin(), destinations.end(), call);
	}

	TRY(component, componentReference());
	EXPECT(Token::Assignment);
	TRY(exp, expression());
	return Statement(move(*component), move(*exp));
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

	return Expression::lor(Type::unknown(), move(factors));
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

	return Expression::land(Type::unknown(), move(factors));
}

Expected<Expression> Parser::logicalFactor()
{
	bool negated = accept<Token::NotKeyword>();
	TRY(exp, relation());
	return negated ? Expression::negate(Type::unknown(), move(*exp)) : move(*exp);
}

Expected<Expression> Parser::relation()
{
	TRY(left, arithmeticExpression());
	auto op = relationalOperator();

	if (!op.has_value())
		return *left;

	TRY(right, arithmeticExpression());
	return Expression(Type::unknown(), op.value(), move(*left), move(*right));
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

Expected<Expression> Parser::arithmeticExpression()
{
	bool negative = false;

	if (accept<Token::Minus>())
		negative = true;
	else
		accept<Token::Plus>();

	TRY(left, term());
	Expression first = negative
												 ? Expression::subtract(Type::unknown(), move(*left))
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
		auto exp = Expression::subtract(Type::unknown(), move(*arg));
		args.emplace_back(move(exp));
	}

	return Expression::op<OperationKind::add>(Type::unknown(), move(args));
}

Expected<Expression> Parser::term()
{
	// we keep a list of arguments
	vector<Expression> arguments;
	TRY(toReturn, factor());

	// if we se no multiply or division sign we return.
	if (current != Token::Multiply && current != Token::Division)
		return *toReturn;

	// otherwise the first argument is placed with the others
	arguments.emplace_back(move(*toReturn));

	while (current == Token::Multiply || current == Token::Division)
	{
		// if see a multyply we add him with the others
		if (accept<Token::Multiply>())
		{
			TRY(arg, factor());
			arguments.emplace_back(move(*arg));
			continue;
		}

		// otherwise we must see a division sign
		EXPECT(Token::Division);
		TRY(arg, factor());

		// if the arguments are exactly one we collapse it in a single division
		// example a / b * c = (a/b) * c
		if (arguments.size() == 1)
		{
			arguments = { Expression::divide(
					Type::unknown(), move(arguments[0]), move(*arg)) };
			continue;
		}

		// otherwise we create a multiply from the already seen arguments
		// a * b / c * d = ((a*b)/c)*d
		auto left = Expression::multiply(Type::unknown(), move(arguments));
		arguments = { Expression::divide(Type::unknown(), move(left), move(*arg)) };
	}

	if (arguments.size() == 1)
		return move(arguments[0]);

	return Expression::multiply(Type::unknown(), move(arguments));
}

Expected<Expression> Parser::factor()
{
	TRY(l, primary());

	if (!accept<Token::Exponential>())
		return *l;

	TRY(r, primary());
	return Expression::powerOf(Type::unknown(), move(*l), move(*r));
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
				Expression(Type::unknown(), ReferenceAccess("der")), move(*args));
	}

	if (current == Token::Ident)
	{
		TRY(exp, componentReference());

		if (current != Token::LPar)
			return exp;

		TRY(args, functionCallArguments());
		return makeCall(move(*exp), move(*args));
	}

	return make_error<UnexpectedToken>(current, Token::End, getPosition());
}

Expected<Expression> Parser::componentReference()
{
	bool globalLookup = accept<Token::Dot>();
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	Expression exp(Type::unknown(), ReferenceAccess(move(name), globalLookup));

	if (current == Token::LSquare)
	{
		TRY(access, arraySubscript());
		access->insert(access->begin(), move(exp));
		exp = Expression::subscription(Type::unknown(), move(*access));
	}

	while (accept<Token::Dot>())
	{
		Expression memberName(makeType<std::string>(), lexer.getLastString());
		EXPECT(Token::String);
		exp =
				Expression::memberLookup(Type::unknown(), move(exp), move(memberName));

		if (current != Token::LSquare)
			continue;

		TRY(access, arraySubscript());
		exp = Expression::subscription(Type::unknown(), move(*access));
	}

	return exp;
}

Expected<SmallVector<Expression, 3>> Parser::functionCallArguments()
{
	EXPECT(Token::LPar);

	SmallVector<Expression, 3> expressions;
	while (!accept<Token::RPar>())
	{
		TRY(arg, expression());
		expressions.push_back(move(*arg));

		while (accept(Token::Comma))
		{
			TRY(exp, expression());
			expressions.push_back(move(*exp));
		}
	}

	return expressions;
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

Expected<vector<Expression>> Parser::arraySubscript()
{
	EXPECT(Token::LSquare);

	vector<Expression> expressions;

	do
	{
		TRY(exp, expression());
		*exp =
				Expression::add(Type::Int(), move(*exp), Expression(Type::Int(), -1));
		expressions.emplace_back(move(*exp));
	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);
	return expressions;
}
