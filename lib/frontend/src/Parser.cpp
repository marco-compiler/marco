#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/frontend/LexerStateMachine.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/ParserErrors.hpp>
#include <optional>

using namespace modelica;
using namespace frontend;

Parser::Parser(std::string filename, const std::string& source)
		: filename(move(filename)),
			lexer(source),
			current(lexer.scan()), undo(Token::End)
{
}

Parser::Parser(const std::string& source)
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

llvm::Expected<bool> Parser::expect(Token t)
{
	if (accept(t))
		return true;

	return llvm::make_error<UnexpectedToken>(current, t, getPosition());
}

void Parser::undoScan(Token t)
{
	undo = current;
	current = t;
}

#include <modelica/utils/ParserUtils.hpp>

llvm::Expected<ClassContainer> Parser::classDefinition()
{
	llvm::SmallVector<ClassContainer, 3> classes;

	while (current != Token::End)
	{
		auto location = getPosition();

		ClassType classType = ClassType::Model;

		bool partial = accept(Token::PartialKeyword);
		bool op = accept(Token::OperatorKeyword);
		bool pure = true;

		if (op)
		{
			if (accept<Token::RecordKeyword>())
				classType = ClassType::Record;
			else
			{
				EXPECT(Token::FunctionKeyword);
				classType = ClassType::Function;
			}
		}
		else if (accept<Token::ModelKeyword>())
		{
			classType = ClassType::Model;
		}
		else if (accept<Token::RecordKeyword>())
		{
			classType = ClassType::Record;
		}
		else if (accept<Token::PackageKeyword>())
		{
			classType = ClassType::Package;
		}
		else if (accept<Token::FunctionKeyword>())
		{
			classType = ClassType::Function;
		}
		else if (accept<Token::PureKeyword>())
		{
			pure = true;
			op = accept(Token::OperatorKeyword);
			EXPECT(Token::FunctionKeyword);
			classType = ClassType::Function;
		}
		else if (accept<Token::ImpureKeyword>())
		{
			pure = false;
			op = accept<Token::OperatorKeyword>();
			EXPECT(Token::FunctionKeyword);
			classType = ClassType::Function;
		}
		else
		{
			EXPECT(Token::ClassKeyword);
		}

		std::string name = lexer.getLastIdentifier();
		EXPECT(Token::Ident);

		while (accept<Token::Dot>())
		{
			name += "." + lexer.getLastIdentifier();
			EXPECT(Token::Ident);
		}

		accept(Token::String);

		llvm::SmallVector<Member, 3> members;
		llvm::SmallVector<Equation, 3> equations;
		llvm::SmallVector<ForEquation, 3> forEquations;
		llvm::SmallVector<Algorithm, 3> algorithms;
		llvm::SmallVector<ClassContainer, 3> innerClasses;

		// Whether the first elements list is allowed to be encountered or not.
		// In fact, the class definition allows a first elements list definition
		// and then others more if preceded by "public" or "protected", but no more
		// "lone" definitions are anymore allowed if any of those keywords are
		// encountered.
		bool firstElementListParsable = true;

		while (current != Token::EndKeyword && current != Token::AnnotationKeyword)
		{
			if (current == Token::EquationKeyword)
			{
				TRY(eq, equationSection());

				for (auto& equation : eq->first)
					equations.emplace_back(std::move(equation));

				for (auto& forEquation : eq->second)
					forEquations.emplace_back(std::move(forEquation));

				continue;
			}

			if (current == Token::AlgorithmKeyword)
			{
				TRY(alg, algorithmSection());
				algorithms.emplace_back(std::move(*alg));
				continue;
			}

			if (current == Token::ClassKeyword ||
					current == Token::FunctionKeyword ||
					current == Token::ModelKeyword ||
					current == Token::RecordKeyword)
			{
				TRY(func, classDefinition());
				innerClasses.emplace_back(std::move(*func));
				continue;
			}

			if (accept(Token::PublicKeyword))
			{
				TRY(mem, elementList());
				firstElementListParsable = false;

				for (auto& member : *mem)
					members.emplace_back(std::move(member));

				continue;
			}

			if (accept(Token::ProtectedKeyword))
			{
				TRY(mem, elementList(false));
				firstElementListParsable = false;

				for (auto& member : *mem)
					members.emplace_back(std::move(member));

				continue;
			}

			if (firstElementListParsable)
			{
				TRY(mem, elementList());

				for (auto& member : *mem)
					members.emplace_back(std::move(member));
			}
		}

		Annotation clsAnnotation;

		if (current == Token::AnnotationKeyword)
		{
			TRY(ann, annotation());
			clsAnnotation = *ann;
			EXPECT(Token::Semicolons);
		}

		EXPECT(Token::EndKeyword);

		std::string endName = lexer.getLastIdentifier();
		EXPECT(Token::Ident);

		while (accept<Token::Dot>())
		{
			endName += "." + lexer.getLastIdentifier();
			EXPECT(Token::Ident);
		}

		if (name != endName)
			return llvm::make_error<UnexpectedIdentifier>(endName, name, getPosition());

		EXPECT(Token::Semicolons);

		if (classType == ClassType::Function)
			classes.emplace_back(Function(location, std::move(name), pure, std::move(members), std::move(algorithms), clsAnnotation));
		else if (classType == ClassType::Model)
			classes.emplace_back(Class(location, std::move(name), std::move(members), std::move(equations), std::move(forEquations), std::move(algorithms), std::move(innerClasses)));
		else if (classType == ClassType::Package)
			classes.emplace_back(Package(location, std::move(name), std::move(innerClasses)));
		else if (classType == ClassType::Record)
			classes.emplace_back(Record(location, std::move(name), std::move(members)));
	}

	if (classes.size() != 1)
		return ClassContainer(Package(SourcePosition::unknown(), "Main", classes));

	return classes[0];
}

llvm::Expected<llvm::SmallVector<Member, 3>> Parser::elementList(bool publicSection)
{
	llvm::SmallVector<Member, 3> members;

	while (
			current != Token::PublicKeyword && current != Token::ProtectedKeyword &&
			current != Token::FunctionKeyword && current != Token::EquationKeyword &&
			current != Token::AlgorithmKeyword && current != Token::EndKeyword &&
			current != Token::ClassKeyword && current != Token::FunctionKeyword &&
			current != Token::ModelKeyword && current != Token::PackageKeyword &&
			current != Token::RecordKeyword)
	{
		TRY(memb, element(publicSection));
		EXPECT(Token::Semicolons);
		members.emplace_back(std::move(*memb));
	}

	return members;
}

llvm::Expected<Member> Parser::element(bool publicSection)
{
	auto location = getPosition();

	accept<Token::FinalKeyword>();
	TRY(prefix, typePrefix());
	TRY(type, typeSpecifier());
	std::string name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	if (current == Token::LSquare)
	{
		TRY(dims, arrayDimensions());
		auto& postDimensions = *dims;

		llvm::SmallVector<ArrayDimension, 3> dimensions;

		if (postDimensions.size() > 1 || postDimensions[0] != 1)
			for (auto& dimension : postDimensions)
				dimensions.push_back(dimension);

		auto& preDimensions = type->getDimensions();

		if (preDimensions.size() > 1 || preDimensions[0] != 1)
			for (auto& dimension : preDimensions)
				dimensions.push_back(dimension);

		if (dimensions.empty())
			dimensions.push_back(ArrayDimension(1));

		type->setDimensions(dimensions);
	}

	std::optional<Expression> startOverload = std::nullopt;

	if (current == Token::LPar)
	{
		TRY(start, termModification());

		if (start->has_value())
			startOverload = std::move(**start);
	}

	if (accept<Token::Equal>())
	{
		TRY(init, expression());

		accept<Token::String>();
		return Member(
				location, move(name), std::move(*type), std::move(*prefix), std::move(*init), publicSection);
	}

	accept<Token::String>();

	return Member(
			location, move(name), std::move(*type), std::move(*prefix), publicSection, startOverload);
}

llvm::Expected<std::optional<Expression>> Parser::termModification()
{
	EXPECT(Token::LPar);

	std::optional<Expression> e = std::nullopt;

	do
	{
		auto lastIndent = lexer.getLastIdentifier();
		EXPECT(Token::Ident);
		EXPECT(Token::Equal);
		if (lastIndent == "start")
		{
			TRY(exp, expression());
			e = std::move(*exp);
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

llvm::Expected<TypePrefix> Parser::typePrefix()
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

llvm::Expected<Type> Parser::typeSpecifier()
{
	std::string name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	while (accept<Token::Dot>())
	{
		name += "." + lexer.getLastIdentifier();
		EXPECT(Token::Ident);
	}

	llvm::SmallVector<ArrayDimension, 3> dimensions;

	if (current != Token::Ident)
	{
		TRY(sub, arrayDimensions());

		for (const auto& dim : *sub)
			dimensions.push_back(dim);
	}

	if (dimensions.empty())
		dimensions.emplace_back(1);

	if (name == "int")
		return Type(BuiltInType::Integer, dimensions);

	if (name == "string")
		return Type(BuiltInType::String, dimensions);

	if (name == "Real")
		return Type(BuiltInType::Float, dimensions);

	if (name == "Integer")
		return Type(BuiltInType::Integer, dimensions);

	if (name == "float")
		return Type(BuiltInType::Float, dimensions);

	if (name == "bool")
		return Type(BuiltInType::Boolean, dimensions);

	return Type(UserDefinedType(name, {}), dimensions);
}

llvm::Expected<std::pair<llvm::SmallVector<Equation, 3>, llvm::SmallVector<ForEquation, 3>>> Parser::equationSection()
{
	EXPECT(Token::EquationKeyword);

	llvm::SmallVector<Equation, 3> equations;
	llvm::SmallVector<ForEquation, 3> forEquations;

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
				forEquations.emplace_back(std::move(eq));
		}
		else
		{
			TRY(eq, equation());
			equations.emplace_back(std::move(*eq));
		}

		EXPECT(Token::Semicolons);
	}

	return std::pair(std::move(equations), std::move(forEquations));
}

llvm::Expected<Algorithm> Parser::algorithmSection()
{
	auto location = getPosition();
	EXPECT(Token::AlgorithmKeyword);
	llvm::SmallVector<Statement, 3> statements;

	while (
			current != Token::End && current != Token::PublicKeyword &&
			current != Token::ProtectedKeyword && current != Token::EquationKeyword &&
			current != Token::AlgorithmKeyword && current != Token::ExternalKeyword &&
			current != Token::AnnotationKeyword && current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(std::move(*stmnt));
	}

	return Algorithm(location, std::move(statements));
}

llvm::Expected<Equation> Parser::equation()
{
	auto location = getPosition();
	TRY(l, expression());
	EXPECT(Token::Equal);
	TRY(r, expression());
	accept<Token::String>();
	return Equation(location, std::move(*l), std::move(*r));
}

llvm::Expected<Statement> Parser::statement()
{
	if (current == Token::IfKeyword)
	{
		TRY(statement, ifStatement());
		return Statement(std::move(*statement));
	}

	if (current == Token::ForKeyword)
	{
		TRY(statement, forStatement());
		return Statement(std::move(*statement));
	}

	if (current == Token::WhileKeyword)
	{
		TRY(statement, whileStatement());
		return Statement(std::move(*statement));
	}

	if (current == Token::WhenKeyword)
	{
		TRY(statement, whenStatement());
		return Statement(std::move(*statement));
	}

	if (current == Token::BreakKeyword)
	{
		TRY(statement, breakStatement());
		return Statement(std::move(*statement));
	}

	if (current == Token::ReturnKeyword)
	{
		TRY(statement, returnStatement());
		return Statement(std::move(*statement));
	}

	TRY(statement, assignmentStatement());
	return Statement(std::move(*statement));
}

llvm::Expected<AssignmentStatement> Parser::assignmentStatement()
{
	auto location = getPosition();

	if (accept<Token::LPar>())
	{
		TRY(destinations, outputExpressionList());
		EXPECT(Token::RPar);
		EXPECT(Token::Assignment);
		TRY(functionName, componentReference());
		TRY(args, functionCallArguments());
		Expression call = Expression::call(location, Type::unknown(), std::move(*functionName), std::move(*args));

		return AssignmentStatement(location, std::move(*destinations), std::move(call));
	}

	TRY(component, componentReference());
	EXPECT(Token::Assignment);
	TRY(exp, expression());
	return AssignmentStatement(location, std::move(*component), std::move(*exp));
}

llvm::Expected<IfStatement> Parser::ifStatement()
{
	auto location = getPosition();
	llvm::SmallVector<IfStatement::Block, 3> blocks;

	EXPECT(Token::IfKeyword);
	TRY(ifCondition, expression());
	EXPECT(Token::ThenKeyword);

	llvm::SmallVector<Statement, 3> ifStatements;

	while (current != Token::ElseIfKeyword && current != Token::ElseKeyword &&
				 current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		ifStatements.push_back(std::move(*stmnt));
	}

	blocks.emplace_back(std::move(*ifCondition), std::move(ifStatements));

	while (current != Token::ElseKeyword && current != Token::EndKeyword)
	{
		EXPECT(Token::ElseIfKeyword);
		TRY(elseIfCondition, expression());
		EXPECT(Token::ThenKeyword);
		llvm::SmallVector<Statement, 3> elseIfStatements;

		while (current != Token::ElseIfKeyword && current != Token::ElseKeyword &&
					 current != Token::EndKeyword)
		{
			TRY(stmnt, statement());
			EXPECT(Token::Semicolons);
			elseIfStatements.push_back(std::move(*stmnt));
		}

		blocks.emplace_back(std::move(*elseIfCondition), std::move(elseIfStatements));
	}

	if (accept<Token::ElseKeyword>())
	{
		llvm::SmallVector<Statement, 3> elseStatements;

		while (current != Token::EndKeyword)
		{
			TRY(stmnt, statement());
			EXPECT(Token::Semicolons);
			elseStatements.push_back(std::move(*stmnt));
		}

		// Being the last block, it can be discarded if empty
		if (!elseStatements.empty())
			blocks.emplace_back(
					Expression::constant(location, makeType<bool>(), true),
					std::move(elseStatements));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::IfKeyword);

	return IfStatement(location, std::move(blocks));
}

llvm::Expected<ForStatement> Parser::forStatement()
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

	Induction induction(move(name), std::move(*begin), std::move(*end));
	llvm::SmallVector<Statement, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(std::move(*stmnt));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::ForKeyword);

	return ForStatement(location, std::move(induction), std::move(statements));
}

llvm::Expected<WhileStatement> Parser::whileStatement()
{
	auto location = getPosition();

	EXPECT(Token::WhileKeyword);
	TRY(condition, expression());
	EXPECT(Token::LoopKeyword);

	llvm::SmallVector<Statement, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(std::move(*stmnt));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::WhileKeyword);

	return WhileStatement(location, std::move(*condition), std::move(statements));
}

llvm::Expected<WhenStatement> Parser::whenStatement()
{
	auto location = getPosition();

	EXPECT(Token::WhenKeyword);
	TRY(condition, expression());
	EXPECT(Token::LoopKeyword);

	llvm::SmallVector<Statement, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(std::move(*stmnt));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::WhenKeyword);

	return WhenStatement(location, std::move(*condition), std::move(statements));
}

llvm::Expected<BreakStatement> Parser::breakStatement()
{
	auto location = getPosition();
	EXPECT(Token::BreakKeyword);
	return BreakStatement(location);
}

llvm::Expected<ReturnStatement> Parser::returnStatement()
{
	auto location = getPosition();
	EXPECT(Token::ReturnKeyword);
	return ReturnStatement(location);
}

llvm::Expected<llvm::SmallVector<ForEquation, 3>> Parser::forEquation(int nestingLevel)
{
	llvm::SmallVector<ForEquation, 3> toReturn;

	EXPECT(Token::ForKeyword);
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);
	EXPECT(Token::InKeyword);
	TRY(begin, expression());
	EXPECT(Token::Colons);
	TRY(end, expression());
	EXPECT(Token::LoopKeyword);

	Induction ind(move(name), std::move(*begin), std::move(*end));
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

llvm::Expected<llvm::SmallVector<ForEquation, 3>> Parser::forEquationBody(int nestingLevel)
{
	llvm::SmallVector<ForEquation, 3> toReturn;

	if (current != Token::ForKeyword)
	{
		TRY(innerEq, equation());
		toReturn.push_back(ForEquation({}, std::move(*innerEq)));
		return toReturn;
	}

	TRY(innerEq, forEquation(nestingLevel));

	for (auto& eq : *innerEq)
	{
		auto& inductions = eq.getInductions();
		toReturn.push_back(std::move(eq));
	}

	return toReturn;
}

llvm::Expected<Expression> Parser::expression()
{
	TRY(l, logicalExpression());
	return std::move(*l);
}

llvm::Expected<Expression> Parser::logicalExpression()
{
	auto location = getPosition();
	std::vector<Expression> factors;
	TRY(l, logicalTerm());

	if (current != Token::OrKeyword)
		return std::move(*l);

	factors.push_back(std::move(*l));

	while (accept<Token::OrKeyword>())
	{
		TRY(arg, logicalTerm());
		factors.emplace_back(std::move(*arg));
	}

	return Expression::operation(location, Type::unknown(), OperationKind::lor, move(factors));
}

llvm::Expected<Expression> Parser::logicalTerm()
{
	auto location = getPosition();
	std::vector<Expression> factors;
	TRY(l, logicalFactor());

	if (current != Token::AndKeyword)
		return std::move(*l);

	factors.push_back(std::move(*l));

	while (accept<Token::AndKeyword>())
	{
		TRY(arg, logicalFactor());
		factors.emplace_back(std::move(*arg));
	}

	return Expression::operation(location, Type::unknown(), OperationKind::land, move(factors));
}

llvm::Expected<Expression> Parser::logicalFactor()
{
	auto location = getPosition();
	bool negated = accept<Token::NotKeyword>();
	TRY(exp, relation());

	if (negated)
		return Expression::operation(location, Type::unknown(), OperationKind::negate, std::move(*exp));

	return *exp;
}

llvm::Expected<Expression> Parser::relation()
{
	auto location = getPosition();
	TRY(left, arithmeticExpression());
	auto op = relationalOperator();

	if (!op.has_value())
		return *left;

	TRY(right, arithmeticExpression());
	return Expression::operation(location, Type::unknown(), op.value(), std::move(*left), std::move(*right));
}

std::optional<OperationKind> Parser::relationalOperator()
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
	return std::nullopt;
}

llvm::Expected<Expression> Parser::arithmeticExpression()
{
	auto location = getPosition();
	bool negative = false;

	if (accept<Token::Minus>())
		negative = true;
	else
		accept<Token::Plus>();

	TRY(left, term());
	Expression first = negative
										 ? Expression::operation(location, Type::unknown(), OperationKind::subtract, std::move(*left))
										 : std::move(*left);

	if (current != Token::Minus && current != Token::Plus)
		return first;

	location = getPosition();
	std::vector<Expression> args;
	args.push_back(std::move(first));

	while (current == Token::Minus || current == Token::Plus)
	{
		location = getPosition();

		if (accept<Token::Plus>())
		{
			TRY(arg, term());
			args.push_back(std::move(*arg));
			continue;
		}

		EXPECT(Token::Minus);
		TRY(arg, term());
		auto exp = Expression::operation(location, Type::unknown(), OperationKind::subtract, std::move(*arg));
		args.emplace_back(std::move(exp));
	}

	return Expression::operation(std::move(location), Type::unknown(), OperationKind::add, move(args));
}

llvm::Expected<Expression> Parser::term()
{
	auto location = getPosition();

	// we keep a list of arguments
	std::vector<Expression> arguments;
	TRY(toReturn, factor());

	// if we se no multiply or division sign we return.
	if (current != Token::Multiply && current != Token::Division)
		return *toReturn;

	// otherwise the first argument is placed with the others
	arguments.emplace_back(std::move(*toReturn));

	while (current == Token::Multiply || current == Token::Division)
	{
		// if see a multiply we add him with the others
		if (accept<Token::Multiply>())
		{
			TRY(arg, factor());
			arguments.emplace_back(std::move(*arg));
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
					location, Type::unknown(), OperationKind::divide, std::move(arguments[0]), std::move(*arg)) };
			continue;
		}

		// otherwise we create a multiply from the already seen arguments
		// a * b / c * d = ((a*b)/c)*d
		auto left = Expression::operation(location, Type::unknown(), OperationKind::multiply, move(arguments));
		arguments = { Expression::operation(location, Type::unknown(), OperationKind::divide, std::move(left), std::move(*arg)) };
	}

	if (arguments.size() == 1)
		return std::move(arguments[0]);

	return Expression::operation(location, Type::unknown(), OperationKind::multiply, move(arguments));
}

llvm::Expected<Expression> Parser::factor()
{
	auto location = getPosition();
	TRY(l, primary());

	if (!accept<Token::Exponential>())
		return *l;

	TRY(r, primary());
	return Expression::operation(std::move(location), Type::unknown(), OperationKind::powerOf, std::move(*l), std::move(*r));
}

llvm::Expected<Expression> Parser::primary()
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

		return Expression(Type::unknown(), std::move(*exp));
	}

	if (accept<Token::LCurly>())
	{
		llvm::SmallVector<Expression, 3> values;

		do
		{
			TRY(argument, expression());
			values.push_back(std::move(*argument));
		} while (accept<Token::Comma>());

		EXPECT(Token::RCurly);
		return Expression(Type::unknown(), Array(location, values));
	}

	if (accept<Token::DerKeyword>())
	{
		TRY(args, functionCallArguments());
		Expression function = Expression::reference(location, Type::unknown(), "der");
		return Expression::call(location, Type::unknown(), function, std::move(*args));
	}

	if (current == Token::Ident)
	{
		TRY(exp, componentReference());

		if (current != Token::LPar)
			return exp;

		TRY(args, functionCallArguments());
		return Expression::call(std::move(location), Type::unknown(), std::move(*exp), std::move(*args));
	}

	return llvm::make_error<UnexpectedToken>(current, Token::End, getPosition());
}

llvm::Expected<Expression> Parser::componentReference()
{
	auto location = getPosition();
	bool globalLookup = accept<Token::Dot>();

	std::string name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	while (accept<Token::Dot>())
	{
		name += "." + lexer.getLastIdentifier();
		EXPECT(Token::Ident);
	}

	Expression expression = Expression::reference(location, Type::unknown(), name, globalLookup);

	if (current == Token::LSquare)
	{
		auto loc = getPosition();
		TRY(access, arraySubscript());
		access->insert(access->begin(), std::move(expression));
		expression = Expression::operation(std::move(loc), Type::unknown(), OperationKind::subscription, move(*access));
	}

	while (accept<Token::Dot>())
	{
		location = getPosition();
		Expression memberName = Expression::reference(location, makeType<std::string>(), lexer.getLastString());
		EXPECT(Token::Ident);
		expression = Expression::operation(location, Type::unknown(), OperationKind::memberLookup, std::move(expression), std::move(memberName));

		if (current != Token::LSquare)
			continue;

		location = getPosition();
		TRY(access, arraySubscript());
		expression = Expression::operation(location, Type::unknown(), OperationKind::subscription, move(*access));
	}

	return expression;
}

llvm::Expected<llvm::SmallVector<Expression, 3>> Parser::functionCallArguments()
{
	EXPECT(Token::LPar);

	llvm::SmallVector<Expression, 3> expressions;
	while (!accept<Token::RPar>())
	{
		TRY(arg, expression());
		expressions.push_back(std::move(*arg));

		while (accept(Token::Comma))
		{
			TRY(exp, expression());
			expressions.push_back(std::move(*exp));
		}
	}

	return expressions;
}

llvm::Expected<llvm::SmallVector<ArrayDimension, 3>> Parser::arrayDimensions()
{
	llvm::SmallVector<ArrayDimension, 3> dimensions;
	EXPECT(Token::LSquare);

	do
	{
		auto location = getPosition();

		if (accept<Token::Colons>())
			dimensions.push_back(ArrayDimension(-1));
		else if (accept<Token::Integer>())
			dimensions.push_back(lexer.getLastInt());
		else
		{
			TRY(exp, expression());
			dimensions.push_back(ArrayDimension(*exp));
		}

	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);
	return dimensions;
}

llvm::Expected<Tuple> Parser::outputExpressionList()
{
	auto location = getPosition();
	llvm::SmallVector<Expression, 3> expressions;

	while (current != Token::RPar)
	{
		auto location = getPosition();

		if (accept<Token::Comma>())
		{
			expressions.emplace_back(Type::unknown(), ReferenceAccess::dummy(location));
			continue;
		}

		TRY(exp, expression());
		expressions.push_back(std::move(*exp));
		accept(Token::Comma);
	}

	return Tuple(std::move(location), std::move(expressions));
}

llvm::Expected<std::vector<Expression>> Parser::arraySubscript()
{
	EXPECT(Token::LSquare);
	std::vector<Expression> expressions;

	do
	{
		auto location = getPosition();
		TRY(exp, expression());
		*exp =
				Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add, std::move(*exp), Expression::constant(location, makeType<BuiltInType::Integer>(), -1));
		expressions.emplace_back(std::move(*exp));
	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);
	return expressions;
}

llvm::Expected<Annotation> Parser::annotation()
{
	EXPECT(Token::AnnotationKeyword);
	TRY(mod, classModification());
	return Annotation(*mod);
}

llvm::Expected<Modification> Parser::modification()
{
	if (accept<Token::Equal>() || accept<Token::Assignment>())
	{
		TRY(exp, expression());
		return Modification(*exp);
	}

	TRY(mod, classModification());

	if (accept<Token::Equal>())
	{
		TRY(exp, expression());
		return Modification(*mod, *exp);
	}

	return Modification(*mod);
}

llvm::Expected<ClassModification> Parser::classModification()
{
	EXPECT(Token::LPar);
	llvm::SmallVector<Argument, 3> arguments;

	do
	{
		TRY(arg, argument());
		arguments.push_back(*arg);
	} while (accept<Token::Comma>());

	EXPECT(Token::RPar);

	return ClassModification(arguments);
}

llvm::Expected<Argument> Parser::argument()
{
	if (current == Token::RedeclareKeyword)
	{
		TRY(el, elementRedeclaration());
		return Argument(*el);
	}

	bool each = accept<Token::EachKeyword>();
	bool final = accept<Token::FinalKeyword>();

	if (current == Token::ReplaceableKeyword)
	{
		TRY(el, elementReplaceable(each, final));
		return Argument(*el);
	}

	TRY(el, elementModification(each, final));
	return Argument(*el);
}

llvm::Expected<ElementModification> Parser::elementModification(bool each, bool final)
{
	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	if (current != Token::LPar && current != Token::Equal && current != Token::Assignment)
		return ElementModification(each, final, name);

	TRY(mod, modification());
	return ElementModification(each, final, name, *mod);
}

llvm::Expected<ElementRedeclaration> Parser::elementRedeclaration()
{
	return llvm::make_error<NotImplemented>("element-redeclaration not implemented yet");
}

llvm::Expected<ElementReplaceable> Parser::elementReplaceable(bool each, bool final)
{
	return llvm::make_error<NotImplemented>("element-replaceable not implemented yet");
}
