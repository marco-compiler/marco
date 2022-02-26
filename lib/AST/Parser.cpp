#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <marco/AST/AST.h>
#include <marco/AST/Errors.h>
#include <marco/AST/LexerStateMachine.h>
#include <marco/AST/Parser.h>
#include <optional>

using namespace marco;
using namespace marco::ast;

Parser::Parser(llvm::StringRef fileName, const char* source)
		: filename(fileName.str()),
			lexer(source),
			current(lexer.scan()),
			tokenRange(fileName, source, 1, 1, 1, 1)
{
	updateTokenSourceRange();
}

Parser::Parser(const std::string& source)
		: Parser("-", source.data())
{
}

Parser::Parser(const char* source)
		: filename("-"),
			lexer(source),
			current(lexer.scan()),
			tokenRange(filename, source, 1, 1, 1, 1)
{
	updateTokenSourceRange();
}

SourceRange Parser::getPosition() const
{
	return tokenRange;
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
	current = lexer.scan();
	updateTokenSourceRange();
}

llvm::Expected<bool> Parser::expect(Token t)
{
	if (accept(t))
		return true;

	return llvm::make_error<UnexpectedToken>(tokenRange, current);
}

void Parser::updateTokenSourceRange()
{
	tokenRange.startLine = lexer.getTokenStartLine();
	tokenRange.startColumn = lexer.getTokenStartColumn();
	tokenRange.endLine = lexer.getTokenEndLine();
	tokenRange.endColumn = lexer.getTokenEndColumn();
}

#include <marco/utils/ParserUtils.h>

llvm::Expected<Parser::ValueWrapper<bool>> Parser::getBool()
{
	auto loc = getPosition();

	if (accept<Token::TrueKeyword>())
		return ValueWrapper(loc, true);

	EXPECT(Token::FalseKeyword);
	return ValueWrapper(loc, false);
}

llvm::Expected<Parser::ValueWrapper<long>> Parser::getInt()
{
	auto loc = getPosition();
	auto value = lexer.getLastInt();
	accept<Token::Integer>();
	return ValueWrapper(loc, value);
}

llvm::Expected<Parser::ValueWrapper<double>> Parser::getFloat()
{
	auto loc = getPosition();
	auto value = lexer.getLastFloat();
	accept<Token::FloatingPoint>();
	return ValueWrapper(loc, value);
}

llvm::Expected<Parser::ValueWrapper<std::string>> Parser::getString()
{
	auto loc = getPosition();
	auto value = lexer.getLastString();
	accept<Token::String>();
	return ValueWrapper(loc, value);
}

llvm::Expected<Parser::ValueWrapper<std::string>> Parser::identifier()
{
	std::string identifier = lexer.getLastIdentifier();
	auto position = tokenRange;
	EXPECT(Token::Ident);

	while (accept<Token::Dot>())
	{
		identifier += "." + lexer.getLastIdentifier();
		position.endLine = tokenRange.endLine;
		position.endColumn = tokenRange.endColumn;
		EXPECT(Token::Ident);
	}

	return ValueWrapper<std::string>(position, std::move(identifier));
}

llvm::Expected<std::unique_ptr<Class>> Parser::classDefinition()
{
	llvm::SmallVector<std::unique_ptr<Class>, 3> classes;

	while (current != Token::End)
	{
		ClassType classType = ClassType::Model;

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

		auto loc = getPosition();
		TRY(name, identifier());

		if (accept<Token::Equal>())
		{
			// Function derivative
			assert(classType == ClassType::Function);
			EXPECT(Token::DerKeyword);
			EXPECT(Token::LPar);

			TRY(derivedFunction, expression());
			llvm::SmallVector<std::unique_ptr<Expression>, 3> independentVariables;

			while (accept<Token::Comma>())
			{
				TRY(var, expression());
				independentVariables.push_back(std::move(*var));
			}

			EXPECT(Token::RPar);
			accept(Token::String);
			EXPECT(Token::Semicolons);

			classes.push_back(Class::partialDerFunction(loc, name->getValue(), std::move(*derivedFunction), independentVariables));
			continue;
		}

		accept(Token::String);

		llvm::SmallVector<std::unique_ptr<Member>, 3> members;
		llvm::SmallVector<std::unique_ptr<Equation>, 3> equations;
		llvm::SmallVector<std::unique_ptr<ForEquation>, 3> forEquations;
		llvm::SmallVector<std::unique_ptr<Algorithm>, 3> algorithms;
		llvm::SmallVector<std::unique_ptr<Class>, 3> innerClasses;

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
				if (auto error = equationSection(equations, forEquations); error)
					return std::move(error);

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
				if (auto error = elementList(members, true); error)
					return std::move(error);

				firstElementListParsable = false;
				continue;
			}

			if (accept(Token::ProtectedKeyword))
			{
				if (auto error = elementList(members, false); error)
					return std::move(error);

				firstElementListParsable = false;
				continue;
			}

			if (firstElementListParsable)
				if (auto error = elementList(members); error)
					return std::move(error);
		}

		llvm::Optional<std::unique_ptr<Annotation>> clsAnnotation;

		if (current == Token::AnnotationKeyword)
		{
			TRY(ann, annotation());
			clsAnnotation = std::move(*ann);
			EXPECT(Token::Semicolons);
		}
		else
		{
			clsAnnotation = llvm::None;
		}

		EXPECT(Token::EndKeyword);

		TRY(endName, identifier());

		if (name->getValue() != endName->getValue())
			return llvm::make_error<UnexpectedIdentifier>(
					endName->getLocation(), endName->getValue(), name->getValue());

		EXPECT(Token::Semicolons);

		if (classType == ClassType::Function)
		{
			classes.push_back(Class::standardFunction(loc, pure, name->getValue(), members, algorithms, std::move(clsAnnotation)));
		}
		else if (classType == ClassType::Model)
		{
			classes.push_back(Class::model(
					loc, name->getValue(), members, equations, forEquations, algorithms, innerClasses));
		}
		else if (classType == ClassType::Package)
		{
			classes.push_back(Class::package(loc, name->getValue(), innerClasses));
		}
		else if (classType == ClassType::Record)
		{
			classes.push_back(Class::record(loc, name->getValue(), members));
		}
	}

	if (classes.size() != 1)
		return Class::package(SourceRange::unknown(), "Main", classes);

	assert(classes.size() == 1);
	return std::move(classes[0]);
}

llvm::Error Parser::elementList(llvm::SmallVectorImpl<std::unique_ptr<Member>>& members, bool publicSection)
{
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

	return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<Member>> Parser::element(bool publicSection)
{
	accept<Token::FinalKeyword>();
	TRY(prefix, typePrefix());
	TRY(type, typeSpecifier());
	TRY(name, identifier());

	llvm::Optional<std::unique_ptr<Expression>> startOverload;

	if (current == Token::LPar)
	{
		TRY(start, termModification());

		if (start->hasValue())
			startOverload = std::move(start->getValue());
	}

	if (accept<Token::Equal>())
	{
		TRY(init, expression());

		accept<Token::String>();
		return Member::build(
				name->getLocation(), name->getValue(), std::move(*type), std::move(*prefix), std::move(*init), publicSection);
	}

	accept<Token::String>();

	return Member::build(
			name->getLocation(), name->getValue(), std::move(*type), std::move(*prefix), llvm::None, publicSection,
			startOverload.hasValue() ? llvm::Optional(std::move(*startOverload)) : llvm::None);
}

llvm::Expected<llvm::Optional<std::unique_ptr<Expression>>> Parser::termModification()
{
	EXPECT(Token::LPar);

	llvm::Optional<std::unique_ptr<Expression>> e;

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
	return std::move(e);
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
		if (auto error = arrayDimensions(dimensions); error)
			return std::move(error);
	}

	if (dimensions.empty())
		dimensions.emplace_back(1);

	if (name == "int")
		return Type(BuiltInType::Integer, dimensions);

	if (name == "string")
		return Type(BuiltInType::String, dimensions);

	if (name == "Boolean")
		return Type(BuiltInType::Boolean, dimensions);

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

llvm::Error Parser::equationSection(llvm::SmallVectorImpl<std::unique_ptr<Equation>>& equations,
																		llvm::SmallVectorImpl<std::unique_ptr<ForEquation>>& forEquations)
{
	EXPECT(Token::EquationKeyword);

	while (
			current != Token::End && current != Token::PublicKeyword &&
			current != Token::ProtectedKeyword && current != Token::EquationKeyword &&
			current != Token::AlgorithmKeyword && current != Token::ExternalKeyword &&
			current != Token::AnnotationKeyword && current != Token::EndKeyword)
	{
		if (current == Token::ForKeyword)
		{
			if (auto error = forEquation(forEquations, 0); error)
				return error;
		}
		else
		{
			TRY(eq, equation());
			equations.push_back(std::move(*eq));
		}

		EXPECT(Token::Semicolons);
	}

	return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<Algorithm>> Parser::algorithmSection()
{
	auto location = getPosition();
	EXPECT(Token::AlgorithmKeyword);
	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;

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

	return Algorithm::build(location, std::move(statements));
}

llvm::Expected<std::unique_ptr<Induction>> Parser::induction()
{
	auto loc = getPosition();

	auto variableName = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	EXPECT(Token::InKeyword);

	TRY(begin, expression());
	EXPECT(Token::Colons);
	TRY(end, expression());

	return Induction::build(std::move(loc), variableName, std::move(*begin), std::move(*end));
}

llvm::Expected<std::unique_ptr<Equation>> Parser::equation()
{
	auto loc = getPosition();

	TRY(lhs, expression());
	EXPECT(Token::Equal);
	TRY(rhs, expression());
	accept<Token::String>();

	return Equation::build(std::move(loc), std::move(*lhs), std::move(*rhs));
}

llvm::Expected<std::unique_ptr<Statement>> Parser::statement()
{
	if (current == Token::IfKeyword)
	{
		TRY(statement, ifStatement());
		return std::move(*statement);
	}

	if (current == Token::ForKeyword)
	{
		TRY(statement, forStatement());
		return std::move(*statement);
	}

	if (current == Token::WhileKeyword)
	{
		TRY(statement, whileStatement());
		return std::move(*statement);
	}

	if (current == Token::WhenKeyword)
	{
		TRY(statement, whenStatement());
		return std::move(*statement);
	}

	if (current == Token::BreakKeyword)
	{
		TRY(statement, breakStatement());
		return std::move(*statement);
	}

	if (current == Token::ReturnKeyword)
	{
		TRY(statement, returnStatement());
		return std::move(*statement);
	}

	TRY(statement, assignmentStatement());
	return std::move(*statement);
}

llvm::Expected<std::unique_ptr<Statement>> Parser::assignmentStatement()
{
	auto loc = getPosition();

	if (accept<Token::LPar>())
	{
		llvm::SmallVector<std::unique_ptr<Expression>, 3> destinations;

		if (auto error = outputExpressionList(destinations); error)
			return std::move(error);

		auto destinationsTuple = Expression::tuple(loc, Type::unknown(), destinations);

		EXPECT(Token::RPar);
		auto assignmentSymbolLocation = getPosition();
		EXPECT(Token::Assignment);
		TRY(functionName, componentReference());

		auto argsLocation = getPosition();
		llvm::SmallVector<std::unique_ptr<Expression>, 3> args;

		if (auto error = functionCallArguments(argsLocation, args); error)
			return std::move(error);

		auto call = Expression::call(loc, Type::unknown(), std::move(*functionName), args);

		return Statement::assignmentStatement(assignmentSymbolLocation, std::move(destinationsTuple), std::move(call));
	}

	TRY(component, componentReference());
	auto assignmentSymbolLocation = getPosition();
	EXPECT(Token::Assignment);
	TRY(exp, expression());
	return Statement::assignmentStatement(assignmentSymbolLocation, std::move(*component), std::move(*exp));
}

llvm::Expected<std::unique_ptr<Statement>> Parser::ifStatement()
{
	auto location = getPosition();
	llvm::SmallVector<IfStatement::Block, 3> blocks;

	EXPECT(Token::IfKeyword);
	TRY(ifCondition, expression());
	EXPECT(Token::ThenKeyword);

	llvm::SmallVector<std::unique_ptr<Statement>, 3> ifStatements;

	while (current != Token::ElseIfKeyword && current != Token::ElseKeyword &&
				 current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		ifStatements.push_back(std::move(*stmnt));
	}

	blocks.emplace_back(std::move(*ifCondition), ifStatements);

	while (current != Token::ElseKeyword && current != Token::EndKeyword)
	{
		EXPECT(Token::ElseIfKeyword);
		TRY(elseIfCondition, expression());
		EXPECT(Token::ThenKeyword);
		llvm::SmallVector<std::unique_ptr<Statement>, 3> elseIfStatements;

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
		llvm::SmallVector<std::unique_ptr<Statement>, 3> elseStatements;

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

	return Statement::ifStatement(location, blocks);
}

llvm::Expected<std::unique_ptr<Statement>> Parser::forStatement()
{
	auto loc = getPosition();

	EXPECT(Token::ForKeyword);
	TRY(ind, induction());
	EXPECT(Token::LoopKeyword);

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(std::move(*stmnt));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::ForKeyword);

	return Statement::forStatement(loc, std::move(*ind), std::move(statements));
}

llvm::Expected<std::unique_ptr<Statement>> Parser::whileStatement()
{
	auto location = getPosition();

	EXPECT(Token::WhileKeyword);
	TRY(condition, expression());
	EXPECT(Token::LoopKeyword);

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(std::move(*stmnt));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::WhileKeyword);

	return Statement::whileStatement(location, std::move(*condition), statements);
}

llvm::Expected<std::unique_ptr<Statement>> Parser::whenStatement()
{
	auto loc = getPosition();

	EXPECT(Token::WhenKeyword);
	TRY(condition, expression());
	EXPECT(Token::LoopKeyword);

	llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;

	while (current != Token::EndKeyword)
	{
		TRY(stmnt, statement());
		EXPECT(Token::Semicolons);
		statements.push_back(std::move(*stmnt));
	}

	EXPECT(Token::EndKeyword);
	EXPECT(Token::WhenKeyword);

	return Statement::whenStatement(std::move(loc), std::move(*condition), std::move(statements));
}

llvm::Expected<std::unique_ptr<Statement>> Parser::breakStatement()
{
	auto loc = getPosition();
	EXPECT(Token::BreakKeyword);
	return Statement::breakStatement(std::move(loc));
}

llvm::Expected<std::unique_ptr<Statement>> Parser::returnStatement()
{
	auto loc = getPosition();
	EXPECT(Token::ReturnKeyword);
	return Statement::returnStatement(std::move(loc));
}

llvm::Error Parser::forEquation(llvm::SmallVectorImpl<std::unique_ptr<ForEquation>>& equations, int nestingLevel)
{
	EXPECT(Token::ForKeyword);

	TRY(ind, induction());
	(*ind)->setInductionIndex(nestingLevel);

	EXPECT(Token::LoopKeyword);

	while (!accept<Token::EndKeyword>())
	{
		llvm::SmallVector<std::unique_ptr<ForEquation>, 3> inner;

		if (auto error = forEquationBody(inner, nestingLevel + 1); error)
			return error;

		for (auto& equation : inner)
		{
			equation->addOuterInduction((*ind)->clone());
			equations.push_back(std::move(equation));
		}

		EXPECT(Token::Semicolons);
	}

	EXPECT(Token::ForKeyword);
	return llvm::Error::success();
}

llvm::Error Parser::forEquationBody(llvm::SmallVectorImpl<std::unique_ptr<ForEquation>>& equations, int nestingLevel)
{
	auto loc = getPosition();

	if (current != Token::ForKeyword)
	{
		TRY(innerEq, equation());
		equations.push_back(ForEquation::build(loc, llvm::None, std::move(*innerEq)));
		return llvm::Error::success();
	}

	if (auto error = forEquation(equations, nestingLevel); error)
		return error;

	return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<Expression>> Parser::expression()
{
	TRY(l, logicalExpression());
	return std::move(*l);
}

llvm::Expected<std::unique_ptr<Expression>> Parser::logicalExpression()
{
	auto loc = getPosition();
	std::vector<std::unique_ptr<Expression>> factors;
	TRY(l, logicalTerm());
	loc.extendEnd((*l)->getLocation());

	if (current != Token::OrKeyword)
		return std::move(*l);

	factors.push_back(std::move(*l));

	while (accept<Token::OrKeyword>())
	{
		TRY(arg, logicalTerm());
		loc.extendEnd((*arg)->getLocation());
		factors.emplace_back(std::move(*arg));
	}

	return Expression::operation(std::move(loc), Type::unknown(), OperationKind::lor, move(factors));
}

llvm::Expected<std::unique_ptr<Expression>> Parser::logicalTerm()
{
	auto loc = getPosition();
	std::vector<std::unique_ptr<Expression>> factors;
	TRY(l, logicalFactor());
	loc.extendEnd((*l)->getLocation());

	if (current != Token::AndKeyword)
		return std::move(*l);

	factors.push_back(std::move(*l));

	while (accept<Token::AndKeyword>())
	{
		TRY(arg, logicalFactor());
		loc.extendEnd((*arg)->getLocation());
		factors.emplace_back(std::move(*arg));
	}

	return Expression::operation(std::move(loc), Type::unknown(), OperationKind::land, move(factors));
}

llvm::Expected<std::unique_ptr<Expression>> Parser::logicalFactor()
{
	auto loc = getPosition();
	bool negated = accept<Token::NotKeyword>();
	TRY(exp, relation());
	loc.extendEnd((*exp)->getLocation());

	if (negated)
		return Expression::operation(std::move(loc), Type::unknown(), OperationKind::negate, std::move(*exp));

	return std::move(*exp);
}

llvm::Expected<std::unique_ptr<Expression>> Parser::relation()
{
	auto loc = getPosition();
	TRY(left, arithmeticExpression());
	auto op = relationalOperator();

	if (!op.hasValue())
		return std::move(*left);

	TRY(right, arithmeticExpression());
	loc.extendEnd((*right)->getLocation());

	return Expression::operation(
			std::move(loc), Type::unknown(), op.getValue(),
			llvm::ArrayRef({ std::move(*left), std::move(*right) }));
}

llvm::Optional<OperationKind> Parser::relationalOperator()
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

	return llvm::None;
}

llvm::Expected<std::unique_ptr<Expression>> Parser::arithmeticExpression()
{
	auto loc = getPosition();
	bool negative = false;

	if (accept<Token::Minus>())
		negative = true;
	else
		accept<Token::Plus>();

	TRY(left, term());
	loc.extendEnd((*left)->getLocation());

	auto first = negative
									 ? Expression::operation(loc, Type::unknown(), OperationKind::subtract, std::move(*left))
									 : std::move(*left);

	if (current != Token::Minus && current != Token::Plus)
		return first;

	llvm::SmallVector<std::unique_ptr<Expression>, 3> args;
	args.push_back(std::move(first));

	while (current == Token::Minus || current == Token::Plus)
	{
		if (accept<Token::Plus>())
		{
			TRY(arg, term());
			loc.extendEnd((*arg)->getLocation());
			args.push_back(std::move(*arg));
			continue;
		}

		EXPECT(Token::Minus);
		TRY(arg, term());
		loc.extendEnd((*arg)->getLocation());
		auto exp = Expression::operation(loc, Type::unknown(), OperationKind::subtract, std::move(*arg));
		args.emplace_back(std::move(exp));
	}

	return Expression::operation(std::move(loc), Type::unknown(), OperationKind::add, move(args));
}

llvm::Expected<std::unique_ptr<Expression>> Parser::term()
{
	auto loc = getPosition();

	// we keep a list of arguments
	llvm::SmallVector<std::unique_ptr<Expression>, 3> arguments;
	TRY(toReturn, factor());
	loc.extendEnd((*toReturn)->getLocation());

	// if we see no multiply or division sign we return.
	if (current != Token::Multiply && current != Token::Division)
		return std::move(*toReturn);

	// otherwise the first argument is placed with the others
	arguments.emplace_back(std::move(*toReturn));

	while (current == Token::Multiply || current == Token::Division)
	{
		// if see a multiply we add him with the others
		if (accept<Token::Multiply>())
		{
			TRY(arg, factor());
			loc.extendEnd((*arg)->getLocation());
			arguments.emplace_back(std::move(*arg));
			continue;
		}

		// otherwise we must see a division sign
		EXPECT(Token::Division);
		TRY(arg, factor());
		loc.extendEnd((*arg)->getLocation());

		// if the arguments are exactly one we collapse it in a single division
		// example a / b * c = (a/b) * c
		if (arguments.size() == 1)
		{
			auto x = std::move(arguments[0]);
			auto y = std::move(*arg);

			arguments.clear();
			arguments.push_back(Expression::operation(
					loc, Type::unknown(), OperationKind::divide,
					llvm::ArrayRef({ std::move(x), std::move(y) })));

			continue;
		}

		// otherwise we create a multiply from the already seen arguments
		// a * b / c * d = ((a*b)/c)*d
		auto left = Expression::operation(loc, Type::unknown(), OperationKind::multiply, move(arguments));

		arguments.clear();

		arguments.push_back(Expression::operation(
				loc, Type::unknown(), OperationKind::divide,
				llvm::ArrayRef({ std::move(left), std::move(*arg )})));

	}

	if (arguments.size() == 1)
		return std::move(arguments[0]);

	return Expression::operation(loc, Type::unknown(), OperationKind::multiply, move(arguments));
}

llvm::Expected<std::unique_ptr<Expression>> Parser::factor()
{
	auto loc = getPosition();
	TRY(left, primary());

	if (!accept<Token::Exponential>())
		return std::move(*left);

	TRY(right, primary());
	loc.extendEnd((*right)->getLocation());

	return Expression::operation(
			std::move(loc), Type::unknown(), OperationKind::powerOf,
			llvm::ArrayRef({ std::move(*left), std::move(*right) }));
}

llvm::Expected<std::unique_ptr<Expression>> Parser::primary()
{
	auto loc = getPosition();

	if (current == Token::Integer)
	{
		TRY(value, getInt());
		return Expression::constant(value->getLocation(), makeType<BuiltInType::Integer>(), value->getValue());
	}

	if (current == Token::FloatingPoint)
	{
		TRY(value, getFloat());
		return Expression::constant(value->getLocation(), makeType<BuiltInType::Float>(), value->getValue());
	}

	if (current == Token::String)
	{
		TRY(value, getString());
		return Expression::constant(value->getLocation(), makeType<std::string>(), value->getValue());
	}

	if (current == Token::TrueKeyword || current == Token::FalseKeyword)
	{
		TRY(value, getBool());
		return Expression::constant(value->getLocation(), makeType<BuiltInType::Boolean>(), value->getValue());
	}

	std::unique_ptr<Expression> result;

	if (accept<Token::LPar>())
	{
		llvm::SmallVector<std::unique_ptr<Expression>, 3> args;

		if (auto error = outputExpressionList(args); error)
			return std::move(error);

		loc.extendEnd(getPosition());
		EXPECT(Token::RPar);

		if (args.size() == 1)
			return std::move(args[0]);

		result = Expression::tuple(loc, Type::unknown(), args);
	}
	else if (accept<Token::LCurly>())
	{
		llvm::SmallVector<std::unique_ptr<Expression>, 3> values;

		do
		{
			TRY(argument, expression());
			values.push_back(std::move(*argument));
		} while (accept<Token::Comma>());

		loc.extendEnd(getPosition());
		EXPECT(Token::RCurly);

		result = Expression::array(loc, Type::unknown(), values);
	}
	else if (accept<Token::DerKeyword>())
	{
		auto functionLoc = loc;
		auto argsLocation = getPosition();
		llvm::SmallVector<std::unique_ptr<Expression>, 3> args;

		if (auto error = functionCallArguments(argsLocation, args); error)
			return std::move(error);

		loc.extendEnd(argsLocation);

		auto function = Expression::reference(functionLoc, Type::unknown(), "der");
		result = Expression::call(loc, Type::unknown(), std::move(function), args);
	}
	else if (current == Token::Ident)
	{
		TRY(exp, componentReference());
		loc.extendEnd((*exp)->getLocation());

		if (current != Token::LPar)
			return exp;

		auto argsLocation = getPosition();
		llvm::SmallVector<std::unique_ptr<Expression>, 3> args;

		if (auto error = functionCallArguments(argsLocation, args); error)
			return std::move(error);

		loc.extendEnd(argsLocation);
		result = Expression::call(loc, Type::unknown(), std::move(*exp), args);
	}

	if (current == Token::LSquare)
	{
		auto subscriptsLocation = getPosition();
		llvm::SmallVector<std::unique_ptr<Expression>, 3> subscripts;

		if (auto error = arraySubscript(subscriptsLocation, subscripts); error)
			return std::move(error);

		loc.extendEnd(subscriptsLocation);
		subscripts.insert(subscripts.begin(), std::move(result));
		result = Expression::operation(loc, Type::unknown(), OperationKind::subscription, subscripts);
	}

	if (result == nullptr)
		return llvm::make_error<UnexpectedToken>(tokenRange, current);

	return std::move(result);
}

llvm::Expected<std::unique_ptr<Expression>> Parser::componentReference()
{
	auto loc = getPosition();
	bool globalLookup = accept<Token::Dot>();

	TRY(name, identifier());
	loc.extendEnd(name->getLocation());

	auto expression = Expression::reference(loc, Type::unknown(), name->getValue(), globalLookup);

	if (current == Token::LSquare)
	{
		auto subscriptsLocation = getPosition();
		llvm::SmallVector<std::unique_ptr<Expression>, 3> subscripts;

		if (auto error = arraySubscript(subscriptsLocation, subscripts); error)
			return std::move(error);

		loc.extendEnd(subscriptsLocation);
		subscripts.insert(subscripts.begin(), std::move(expression));
		expression = Expression::operation(loc, Type::unknown(), OperationKind::subscription, subscripts);
	}

	while (accept<Token::Dot>())
	{
		auto memberName = Expression::reference(loc, makeType<std::string>(), lexer.getLastString());
		loc.extendEnd(getPosition());
		EXPECT(Token::Ident);

		expression = Expression::operation(
				loc, Type::unknown(), OperationKind::memberLookup,
				llvm::ArrayRef({ std::move(expression), std::move(memberName) }));

		if (current != Token::LSquare)
			continue;

		auto subscriptsLocation = getPosition();
		llvm::SmallVector<std::unique_ptr<Expression>, 3> subscripts;

		if (auto error = arraySubscript(subscriptsLocation, subscripts); error)
			return std::move(error);

		loc.extendEnd(subscriptsLocation);
		expression = Expression::operation(loc, Type::unknown(), OperationKind::subscription, subscripts);
	}

	return expression;
}

llvm::Error Parser::functionCallArguments(SourceRange& location, llvm::SmallVectorImpl<std::unique_ptr<Expression>>& args)
{
	EXPECT(Token::LPar);

	while (!accept<Token::RPar>())
	{
		TRY(arg, expression());
		location.extendEnd((*arg)->getLocation());
		args.push_back(std::move(*arg));

		while (accept(Token::Comma))
		{
			TRY(exp, expression());
			location.extendEnd((*exp)->getLocation());
			args.push_back(std::move(*exp));
		}
	}

	return llvm::Error::success();
}

llvm::Error Parser::arrayDimensions(llvm::SmallVectorImpl<ArrayDimension>& dimensions)
{
	EXPECT(Token::LSquare);

	do
	{
		auto location = getPosition();

		if (accept<Token::Colons>())
			dimensions.push_back(ArrayDimension(-1));
		else if (accept<Token::Integer>())
			dimensions.push_back(ArrayDimension(lexer.getLastInt()));
		else
		{
			TRY(exp, expression());
			dimensions.push_back(ArrayDimension(std::move(*exp)));
		}
	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);
	return llvm::Error::success();
}

llvm::Error Parser::outputExpressionList(llvm::SmallVectorImpl<std::unique_ptr<Expression>>& expressions)
{
	while (current != Token::RPar)
	{
		auto loc = getPosition();

		if (accept<Token::Comma>())
		{
			expressions.push_back(ReferenceAccess::dummy(std::move(loc), Type::unknown()));
			continue;
		}

		TRY(exp, expression());
		expressions.push_back(std::move(*exp));
		accept(Token::Comma);
	}

	return llvm::Error::success();
}

llvm::Error Parser::arraySubscript(SourceRange& location, llvm::SmallVectorImpl<std::unique_ptr<Expression>>& subscripts)
{
	EXPECT(Token::LSquare);

	do
	{
		auto loc = getPosition();
		TRY(exp, expression());
		location.extendEnd((*exp)->getLocation());

		*exp = Expression::operation(
				location, makeType<BuiltInType::Integer>(), OperationKind::add,
				llvm::ArrayRef({
						std::move(*exp),
						Expression::constant(loc, makeType<BuiltInType::Integer>(), (long) -1)
				}));

		subscripts.push_back(std::move(*exp));
	} while (accept<Token::Comma>());

	EXPECT(Token::RSquare);
	return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<Annotation>> Parser::annotation()
{
	auto loc = getPosition();
	EXPECT(Token::AnnotationKeyword);
	TRY(mod, classModification());
	return std::make_unique<Annotation>(std::move(loc), std::move(*mod));
}

llvm::Expected<std::unique_ptr<Modification>> Parser::modification()
{
	auto loc = getPosition();

	if (accept<Token::Equal>() || accept<Token::Assignment>())
	{
		TRY(exp, expression());
		return Modification::build(std::move(loc), std::move(*exp));
	}

	TRY(mod, classModification());

	if (accept<Token::Equal>())
	{
		TRY(exp, expression());
		return Modification::build(std::move(loc), std::move(*mod), std::move(*exp));
	}

	return Modification::build(std::move(loc), std::move(*mod));
}

llvm::Expected<std::unique_ptr<ClassModification>> Parser::classModification()
{
	auto location = getPosition();

	EXPECT(Token::LPar);
	llvm::SmallVector<std::unique_ptr<Argument>, 3> arguments;

	do
	{
		TRY(arg, argument());
		arguments.push_back(std::move(*arg));
	} while (accept<Token::Comma>());

	EXPECT(Token::RPar);

	return ClassModification::build(location, arguments);
}

llvm::Expected<std::unique_ptr<Argument>> Parser::argument()
{
	if (current == Token::RedeclareKeyword)
	{
		TRY(redeclaration, elementRedeclaration());
		return std::move(*redeclaration);
	}

	bool each = accept<Token::EachKeyword>();
	bool final = accept<Token::FinalKeyword>();

	if (current == Token::ReplaceableKeyword)
	{
		TRY(replaceable, elementReplaceable(each, final));
		return std::move(*replaceable);
	}

	TRY(modification, elementModification(each, final));
	return std::move(*modification);
}

llvm::Expected<std::unique_ptr<Argument>> Parser::elementModification(bool each, bool final)
{
	auto loc = getPosition();

	auto name = lexer.getLastIdentifier();
	EXPECT(Token::Ident);

	if (current != Token::LPar && current != Token::Equal && current != Token::Assignment)
		return Argument::elementModification(loc, each, final, name);

	TRY(mod, modification());
	return Argument::elementModification(loc, each, final, name, std::move(*mod));
}

llvm::Expected<std::unique_ptr<Argument>> Parser::elementRedeclaration()
{
	return llvm::make_error<NotImplemented>("element-redeclaration not implemented yet");
}

llvm::Expected<std::unique_ptr<Argument>> Parser::elementReplaceable(bool each, bool final)
{
	return llvm::make_error<NotImplemented>("element-replaceable not implemented yet");
}

namespace marco::ast::detail
{
	ParsingErrorCategory ParsingErrorCategory::category;

	std::error_condition ParsingErrorCategory::default_error_condition(int ev) const noexcept
	{
		if (ev == 1)
			return std::error_condition(ParsingErrorCode::unexpected_token);

		if (ev == 2)
			return std::error_condition(ParsingErrorCode::unexpected_identifier);

		return std::error_condition(ParsingErrorCode::success);
	}

	bool ParsingErrorCategory::equivalent(const std::error_code& code, int condition) const noexcept
	{
		bool equal = *this == code.category();
		auto v = default_error_condition(code.value()).value();
		equal = equal && static_cast<int>(v) == condition;
		return equal;
	}

	std::string ParsingErrorCategory::message(int ev) const noexcept
	{
		switch (ev)
		{
			case (0):
				return "Success";

			case (1):
				return "Unexpected Token";

			case (2):
				return "Unexpected identifier";

			default:
				return "Unknown Error";
		}
	}

	std::error_condition make_error_condition(ParsingErrorCode errc)
	{
		return std::error_condition(
				static_cast<int>(errc), detail::ParsingErrorCategory::category);
	}
}

char UnexpectedToken::ID;
char UnexpectedIdentifier::ID;

UnexpectedToken::UnexpectedToken(SourceRange location, Token token)
		: location(std::move(location)),
			token(token)
{
}

SourceRange UnexpectedToken::getLocation() const
{
	return location;
}

void UnexpectedToken::printMessage(llvm::raw_ostream& os) const
{
	os << "unexpected token [";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << token;
  os.resetColor();
	os << "]";
}

void UnexpectedToken::log(llvm::raw_ostream& os) const
{
	print(os);
}

UnexpectedIdentifier::UnexpectedIdentifier(SourceRange location,
																					 llvm::StringRef identifier,
																					 llvm::StringRef expected)
		: location(std::move(location)),
			identifier(identifier.str()),
			expected(expected.str())
{
}

SourceRange UnexpectedIdentifier::getLocation() const
{
	return location;
}

void UnexpectedIdentifier::printMessage(llvm::raw_ostream& os) const
{
	os << "unexpected identifier \"";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << identifier;
	os.resetColor();
	os << "\" (expected: \"";
	os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
	os << expected;
	os.resetColor();
	os << "\")";
}

void UnexpectedIdentifier::log(llvm::raw_ostream& os) const
{
	print(os);
}
