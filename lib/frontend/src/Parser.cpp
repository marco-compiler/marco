#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/ClassContainer.hpp>
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

Parser::Parser(string filename, const string& source)
		: filename(move(filename)),
			lexer(source),
			current(lexer.scan()), undo(Token::End)
{
}

Parser::Parser(const string& source)
		: Parser("-", source)
{
}

Parser::Parser(const char* source)
		: filename("-"),
			lexer(source),
			current(lexer.scan()), undo(Token::End)
{
}

SourcePosition Parser::getPosition() const
{
	return SourcePosition(filename, lexer.getCurrentLine(), lexer.getCurrentColumn());
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

static Expected<BuiltInType> nameToBuiltin(const std::string& name)
{
	if (name == "int")
		return BuiltInType::Integer;
	if (name == "string")
		return BuiltInType::String;
	if (name == "Real")
		return BuiltInType::Float;
	if (name == "Integer")
		return BuiltInType::Integer;
	if (name == "float")
		return BuiltInType::Float;
	if (name == "bool")
		return BuiltInType::Boolean;

	return make_error<NotImplemented>(
			"Only builtin types are supported, not " + name);
}

Expected<ClassContainer> Parser::classDefinition()
{
	auto location = getPosition();

	ClassType classType = ClassType::Model;

	bool partial = accept(Token::PartialKeyword);
	bool op = accept(Token::OperatorKeyword);
	bool pure = true;

	if (op)
	{
		EXPECT(Token::FunctionKeyword);
		classType = ClassType::Function;
	}
	else if (accept(Token::ModelKeyword))
	{
		classType = ClassType::Model;
	}
	else if (accept(Token::FunctionKeyword))
	{
		classType = ClassType::Function;
	}
	else if (accept(Token::PureKeyword))
	{
		pure = true;
		op = accept(Token::OperatorKeyword);
		EXPECT(Token::FunctionKeyword);
		classType = ClassType::Function;
	}
	else if (accept(Token::ImpureKeyword))
	{
		pure = false;
		op = accept(Token::OperatorKeyword);
		EXPECT(Token::FunctionKeyword);
		classType = ClassType::Function;
	}
	else
	{
		EXPECT(Token::ClassKeyword);
	}

	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	accept(Token::String);

	SmallVector<Member, 3> members;
	SmallVector<Equation, 3> equations;
	SmallVector<ForEquation, 3> forEquations;
	SmallVector<Algorithm, 3> algorithms;
	SmallVector<ClassContainer, 3> innerClasses;

	// Whether the first elements list is allowed to be encountered or not.
	// In fact, the class definition allows a first elements list definition
	// and then others more if preceded by "public" or "protected", but no more
	// "lone" definitions are anymore allowed if any of those keywords are
	// encountered.
	bool firstElementListParsable = true;

	while (current != Token::EndKeyword)
	{
		if (current == Token::EquationKeyword)
		{
			TRY(eq, equationSection());

			for (auto& equation : eq->first)
				equations.emplace_back(move(equation));

			for (auto& forEquation : eq->second)
				forEquations.emplace_back(move(forEquation));

			continue;
		}

		if (current == Token::AlgorithmKeyword)
		{
			TRY(alg, algorithmSection());
			algorithms.emplace_back(move(*alg));
			continue;
		}

		if (current == Token::FunctionKeyword)
		{
			TRY(func, classDefinition());
			innerClasses.emplace_back(move(*func));
			EXPECT(Token::Semicolons);
			continue;
		}

		if (accept(Token::PublicKeyword))
		{
			TRY(mem, elementList());
			firstElementListParsable = false;

			for (auto& member : *mem)
				members.emplace_back(move(member));

			continue;
		}

		if (accept(Token::ProtectedKeyword))
		{
			TRY(mem, elementList(false));
			firstElementListParsable = false;

			for (auto& member : *mem)
				members.emplace_back(move(member));

			continue;
		}

		if (firstElementListParsable)
		{
			TRY(mem, elementList());

			for (auto& member : *mem)
				members.emplace_back(move(member));
		}
	}

	EXPECT(Token::EndKeyword);
	auto endName = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	if (name != endName)
		return make_error<UnexpectedIdentifier>(endName, name, getPosition());

	switch (classType)
	{
		case ClassType::Function:
			return ClassContainer(Function(location, move(name), pure, move(members), move(algorithms)));

		case ClassType::Model:
			return ClassContainer(Class(location, move(name), move(members), move(equations), move(forEquations), move(algorithms), move(innerClasses)));
	}

	assert(false && "Unreachable");
}

Expected<SmallVector<Member, 3>> Parser::elementList(bool publicSection)
{
	SmallVector<Member, 3> members;

	while (
			current != Token::PublicKeyword && current != Token::ProtectedKeyword &&
			current != Token::FunctionKeyword && current != Token::EquationKeyword &&
			current != Token::AlgorithmKeyword && current != Token::EndKeyword)
	{
		TRY(memb, element(publicSection));
		EXPECT(Token::Semicolons);
		members.emplace_back(move(*memb));
	}

	return members;
}

Expected<Member> Parser::element(bool publicSection)
{
	auto location = getPosition();
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
				location, move(name), move(*type), move(*prefix), move(*init), publicSection);
	}
	accept<Token::String>();

	return Member(
			location, move(name), move(*type), move(*prefix), publicSection, startOverload);
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

Expected<pair<SmallVector<Equation, 3>, SmallVector<ForEquation, 3>>> Parser::equationSection()
{
	EXPECT(Token::EquationKeyword);

	SmallVector<Equation, 3> equations;
	SmallVector<ForEquation, 3> forEquations;

	while (
			current != Token::End && current != Token::PublicKeyword &&
			current != Token::ProtectedKeyword && current != Token::EquationKeyword &&
			current != Token::AlgorithmKeyword && current != Token::ExternalKeyword &&
			current != Token::AnnotationKeyword && current != Token::EndKeyword)
	{
		if (current == Token::ForKeyword)
		{
			TRY(equs, forEquation(0));

			for (auto& eq : *equs)
				forEquations.emplace_back(move(eq));
		}
		else
		{
			TRY(eq, equation());
			equations.emplace_back(move(*eq));
		}

		EXPECT(Token::Semicolons);
	}

	return pair(move(equations), move(forEquations));
}

Expected<Algorithm> Parser::algorithmSection()
{
	auto location = getPosition();
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

	return Algorithm(location, move(statements));
}

Expected<Equation> Parser::equation()
{
	auto location = getPosition();
	TRY(l, expression());
	EXPECT(Token::Equal);
	TRY(r, expression());
	accept<Token::String>();
	return Equation(location, move(*l), move(*r));
}

Expected<Statement> Parser::statement()
{
	if (current == Token::IfKeyword)
	{
		TRY(statement, ifStatement());
		return Statement(move(*statement));
	}

	if (current == Token::ForKeyword)
	{
		TRY(statement, forStatement());
		return Statement(move(*statement));
	}

	if (current == Token::WhileKeyword)
	{
		TRY(statement, whileStatement());
		return Statement(move(*statement));
	}

	if (current == Token::WhenKeyword)
	{
		TRY(statement, whenStatement());
		return Statement(move(*statement));
	}

	TRY(statement, assignmentStatement());
	return Statement(move(*statement));
}

Expected<AssignmentStatement> Parser::assignmentStatement()
{
	auto location = getPosition();

	if (accept<Token::LPar>())
	{
		TRY(destinations, outputExpressionList());
		EXPECT(Token::RPar);
		EXPECT(Token::Assignment);
		TRY(functionName, componentReference());
		TRY(args, functionCallArguments());
		Expression call = Expression::call(location, Type::unknown(), move(*functionName), move(*args));

		return AssignmentStatement(location, move(*destinations), move(call));
	}

	TRY(component, componentReference());
	EXPECT(Token::Assignment);
	TRY(exp, expression());
	return AssignmentStatement(location, move(*component), move(*exp));
}

Expected<IfStatement> Parser::ifStatement()
{
	auto location = getPosition();
	SmallVector<IfStatement::Block, 3> blocks;

	EXPECT(Token::IfKeyword);
	TRY(ifCondition, expression());
	EXPECT(Token::ThenKeyword);

	SmallVector<Statement, 3> ifStatements;

	while (current != Token::ElseIfKeyword && current != Token::ElseKeyword &&
				 current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		ifStatements.push_back(move(*stmnt));
	}

	blocks.emplace_back(move(*ifCondition), move(ifStatements));

	while (current != Token::ElseKeyword && current != Token::EndKeyword)
	{
		EXPECT(Token::ElseIfKeyword);
		TRY(elseIfCondition, expression());
		EXPECT(Token::ThenKeyword);
		SmallVector<Statement, 3> elseIfStatements;

		while (current != Token::ElseIfKeyword && current != Token::ElseKeyword &&
					 current != Token::EndKeyword)
		{
			TRY(stmnt, statement());
			EXPECT(Token::Semicolons);
			elseIfStatements.push_back(move(*stmnt));
		}

		blocks.emplace_back(move(*elseIfCondition), move(elseIfStatements));
	}

	if (accept<Token::ElseKeyword>())
	{
		SmallVector<Statement, 3> elseStatements;

		while (current != Token::EndKeyword)
		{
			TRY(stmnt, statement());
			EXPECT(Token::Semicolons);
			elseStatements.push_back(move(*stmnt));
		}

		// Being the last block, it can be discarded if empty
		if (!elseStatements.empty())
			blocks.emplace_back(
					Expression::constant(location, makeType<bool>(), true),
					move(elseStatements));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::IfKeyword);

	return IfStatement(location, move(blocks));
}

Expected<ForStatement> Parser::forStatement()
{
	auto location = getPosition();

	EXPECT(Token::ForKeyword);
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	EXPECT(Token::InKeyword);
	TRY(begin, expression());
	EXPECT(Token::Colons);
	TRY(end, expression());
	EXPECT(Token::LoopKeyword);

	Induction induction(move(name), move(*begin), move(*end));
	SmallVector<Statement, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(move(*stmnt));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::ForKeyword);

	return ForStatement(location, move(induction), move(statements));
}

Expected<WhileStatement> Parser::whileStatement()
{
	auto location = getPosition();

	EXPECT(Token::WhileKeyword);
	TRY(condition, expression());
	EXPECT(Token::LoopKeyword);

	SmallVector<Statement, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(move(*stmnt));
	}

	return WhileStatement(location, move(*condition), move(statements));
}

Expected<WhenStatement> Parser::whenStatement()
{
	auto location = getPosition();

	EXPECT(Token::WhenKeyword);
	TRY(condition, expression());
	EXPECT(Token::LoopKeyword);

	SmallVector<Statement, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(move(*stmnt));
	}

	return WhenStatement(location, move(*condition), move(statements));
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

Expected<Expression> Parser::expression()
{
	TRY(l, logicalExpression());
	return move(*l);
}

Expected<Expression> Parser::logicalExpression()
{
	auto location = getPosition();
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

	return Expression::operation(location, Type::unknown(), OperationKind::lor, move(factors));
}

Expected<Expression> Parser::logicalTerm()
{
	auto location = getPosition();
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

	return Expression::operation(location, Type::unknown(), OperationKind::land, move(factors));
}

Expected<Expression> Parser::logicalFactor()
{
	auto location = getPosition();
	bool negated = accept<Token::NotKeyword>();
	TRY(exp, relation());

	if (negated)
		return Expression::operation(location, Type::unknown(), OperationKind::negate, move(*exp));

	return *exp;
}

Expected<Expression> Parser::relation()
{
	auto location = getPosition();
	TRY(left, arithmeticExpression());
	auto op = relationalOperator();

	if (!op.has_value())
		return *left;

	TRY(right, arithmeticExpression());
	return Expression::operation(location, Type::unknown(), op.value(), move(*left), move(*right));
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
	auto location = getPosition();
	bool negative = false;

	if (accept<Token::Minus>())
		negative = true;
	else
		accept<Token::Plus>();

	TRY(left, term());
	Expression first = negative
										 ? Expression::operation(location, Type::unknown(), OperationKind::subtract, move(*left))
										 : move(*left);

	if (current != Token::Minus && current != Token::Plus)
		return first;

	location = getPosition();
	vector<Expression> args;
	args.push_back(move(first));

	while (current == Token::Minus || current == Token::Plus)
	{
		location = getPosition();

		if (accept<Token::Plus>())
		{
			TRY(arg, term());
			args.push_back(move(*arg));
			continue;
		}

		EXPECT(Token::Minus);
		TRY(arg, term());
		auto exp = Expression::operation(location, Type::unknown(), OperationKind::subtract, move(*arg));
		args.emplace_back(move(exp));
	}

	return Expression::operation(move(location), Type::unknown(), OperationKind::add, move(args));
}

Expected<Expression> Parser::term()
{
	auto location = getPosition();

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
		// if see a multiply we add him with the others
		if (accept<Token::Multiply>())
		{
			TRY(arg, factor());
			arguments.emplace_back(move(*arg));
			continue;
		}

		// otherwise we must see a division sign
		auto location = getPosition();
		EXPECT(Token::Division);
		TRY(arg, factor());

		// if the arguments are exactly one we collapse it in a single division
		// example a / b * c = (a/b) * c
		if (arguments.size() == 1)
		{
			arguments = { Expression::operation(
					location, Type::unknown(), OperationKind::divide, move(arguments[0]), move(*arg)) };
			continue;
		}

		// otherwise we create a multiply from the already seen arguments
		// a * b / c * d = ((a*b)/c)*d
		auto left = Expression::operation(location, Type::unknown(), OperationKind::multiply, move(arguments));
		arguments = { Expression::operation(location, Type::unknown(), OperationKind::divide, move(left), move(*arg)) };
	}

	if (arguments.size() == 1)
		return move(arguments[0]);

	return Expression::operation(location, Type::unknown(), OperationKind::multiply, move(arguments));
}

Expected<Expression> Parser::factor()
{
	auto location = getPosition();
	TRY(l, primary());

	if (!accept<Token::Exponential>())
		return *l;

	TRY(r, primary());
	return Expression::operation(move(location), Type::unknown(), OperationKind::powerOf, move(*l), move(*r));
}

Expected<Expression> Parser::primary()
{
	auto location = getPosition();

	if (current == Token::Integer)
	{
		auto value = lexer.getLastInt();
		accept<Token::Integer>();
		return Expression::constant(location, makeType<BuiltInType::Integer>(), value);
	}

	if (current == Token::FloatingPoint)
	{
		auto value = lexer.getLastFloat();
		accept<Token::FloatingPoint>();
		return Expression::constant(location, makeType<BuiltInType::Float>(), value);
	}

	if (current == Token::String)
	{
		auto value = lexer.getLastString();
		accept<Token::String>();
		return Expression::constant(location, makeType<std::string>(), value);
	}

	if (accept<Token::TrueKeyword>())
		return Expression::constant(location, makeType<BuiltInType::Boolean>(), true);

	if (accept<Token::FalseKeyword>())
		return Expression::constant(location, makeType<BuiltInType::Boolean>(), false);

	if (accept<Token::LPar>())
	{
		TRY(exp, outputExpressionList());
		EXPECT(Token::RPar);

		if (exp->size() == 1)
			return (*exp)[0];

		return Expression(Type::unknown(), move(*exp));
	}

	if (accept<Token::DerKeyword>())
	{
		TRY(args, functionCallArguments());
		Expression function = Expression::reference(location, Type::unknown(), "der");
		return Expression::call(location, Type::unknown(), function, move(*args));
	}

	if (current == Token::Ident)
	{
		TRY(exp, componentReference());

		if (current != Token::LPar)
			return exp;

		TRY(args, functionCallArguments());
		return Expression::call(move(location), Type::unknown(), move(*exp), move(*args));
	}

	return make_error<UnexpectedToken>(current, Token::End, getPosition());
}

Expected<Expression> Parser::componentReference()
{
	auto location = getPosition();
	bool globalLookup = accept<Token::Dot>();
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	Expression expression = Expression::reference(location, Type::unknown(), name, globalLookup);

	if (current == Token::LSquare)
	{
		auto loc = getPosition();
		TRY(access, arraySubscript());
		access->insert(access->begin(), move(expression));
		expression = Expression::operation(move(loc), Type::unknown(), OperationKind::subscription, move(*access));
	}

	while (accept<Token::Dot>())
	{
		location = getPosition();
		Expression memberName = Expression::reference(location, makeType<std::string>(), lexer.getLastString());
		EXPECT(Token::String);
		expression = Expression::operation(location, Type::unknown(), OperationKind::memberLookup, move(expression), move(memberName));

		if (current != Token::LSquare)
			continue;

		location = getPosition();
		TRY(access, arraySubscript());
		expression = Expression::operation(location, Type::unknown(), OperationKind::subscription, move(*access));
	}

	return expression;
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

Expected<Tuple> Parser::outputExpressionList()
{
	auto location = getPosition();
	SmallVector<Expression, 3> expressions;

	while (current != Token::RPar)
	{
		auto location = getPosition();

		if (accept<Token::Comma>())
		{
			expressions.emplace_back(Type::unknown(), ReferenceAccess::dummy(location));
			continue;
		}

		TRY(exp, expression());
		expressions.push_back(move(*exp));
		accept(Token::Comma);
	}

	return Tuple(move(location), move(expressions));
}

Expected<vector<Expression>> Parser::arraySubscript()
{
	EXPECT(Token::LSquare);
	vector<Expression> expressions;

	do
	{
		auto location = getPosition();
		TRY(exp, expression());
		*exp =
				Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add, move(*exp), Expression::constant(location, makeType<BuiltInType::Integer>(), -1));
		expressions.emplace_back(move(*exp));
	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);
	return expressions;
}
