#include "marco/Parser/Parser.h"
#include "marco/Parser/Message.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::parser;

#define EXPECT(Token)                                                                           \
	if (!accept<Token>()) {                                                                       \
	  diagnostics->emitError<UnexpectedTokenMessage>(lexer.getTokenPosition(), current, Token);   \
    return llvm::None;                                                                          \
  }                                                                                             \
  static_assert(true)

#define TRY(outVar, expression)       \
	auto outVar = expression;           \
	if (!outVar.hasValue()) {           \
    return llvm::None;                \
  }                                   \
  static_assert(true)

namespace marco::parser
{
  Parser::Parser(diagnostic::DiagnosticEngine& diagnostics, std::shared_ptr<SourceFile> file)
      : diagnostics(&diagnostics),
        lexer(file),
        current(Token::Begin)
  {
    next();
  }

  void Parser::next()
  {
    current = lexer.scan();
  }

  llvm::Optional<std::unique_ptr<ast::Class>> Parser::parseRoot()
  {
    auto loc = lexer.getTokenPosition();
    llvm::SmallVector<std::unique_ptr<Class>, 1> classes;

    while (current != Token::EndOfFile) {
      TRY(classDefinition, parseClassDefinition());
      classes.push_back(std::move(*classDefinition));
    }

    // If multiple root classes exist, then wrap them into a package
    if (classes.size() != 1) {
      return Class::package(SourceRange::unknown(), "Main", classes);
    }

    assert(classes.size() == 1);
    return std::move(classes[0]);
  }

  llvm::Optional<Parser::ValueWrapper<bool>> Parser::parseBoolValue()
  {
    auto loc = lexer.getTokenPosition();

    if (accept<Token::True>()) {
      return ValueWrapper(loc, true);
    }

    EXPECT(Token::False);
    return ValueWrapper(loc, false);
  }

  llvm::Optional<Parser::ValueWrapper<int64_t>> Parser::parseIntValue()
  {
    auto loc = lexer.getTokenPosition();
    auto value = lexer.getInt();
    accept<Token::Integer>();
    return ValueWrapper(loc, value);
  }

  llvm::Optional<Parser::ValueWrapper<double>> Parser::parseFloatValue()
  {
    auto loc = lexer.getTokenPosition();
    auto value = lexer.getFloat();
    accept<Token::FloatingPoint>();
    return ValueWrapper(loc, value);
  }

  llvm::Optional<Parser::ValueWrapper<std::string>> Parser::parseString()
  {
    auto loc = lexer.getTokenPosition();
    auto value = lexer.getString();
    accept<Token::String>();
    return ValueWrapper(loc, value);
  }

  llvm::Optional<Parser::ValueWrapper<std::string>> Parser::parseIdentifier()
  {
    auto loc = lexer.getTokenPosition();

    std::string identifier = lexer.getIdentifier();
    EXPECT(Token::Identifier);

    while (accept<Token::Dot>()) {
      identifier += "." + lexer.getIdentifier();
      loc.end = lexer.getTokenPosition().end;
      EXPECT(Token::Identifier);
    }

    return ValueWrapper(loc, std::move(identifier));
  }

  llvm::Optional<std::unique_ptr<Class>> Parser::parseClassDefinition()
  {
    auto loc = lexer.getTokenPosition();
    ClassType classType = ClassType::Model;

    bool isOperator = accept<Token::Operator>();
    bool isPure = true;

    if (isOperator) {
      if (accept<Token::Record>()) {
        classType = ClassType::Record;
      } else {
        EXPECT(Token::Function);
        classType = ClassType::Function;
      }
    } else if (accept<Token::Model>()) {
      classType = ClassType::Model;

    } else if (accept<Token::Record>()) {
      classType = ClassType::Record;

    } else if (accept<Token::Package>()) {
      classType = ClassType::Package;

    } else if (accept<Token::Function>()) {
      classType = ClassType::Function;

    } else if (accept<Token::Pure>()) {
      isPure = true;
      isOperator = accept<Token::Operator>();
      EXPECT(Token::Function);
      classType = ClassType::Function;

    } else if (accept<Token::Impure>()) {
      isPure = false;
      isOperator = accept<Token::Operator>();
      EXPECT(Token::Function);
      classType = ClassType::Function;

    } else {
      EXPECT(Token::Class);
    }

    TRY(name, parseIdentifier());
    loc.end = name->getLocation().end;

    if (accept<Token::EqualityOperator>()) {
      // Function derivative
      assert(classType == ClassType::Function);
      EXPECT(Token::Der);
      EXPECT(Token::LPar);

      TRY(derivedFunction, parseExpression());
      llvm::SmallVector<std::unique_ptr<Expression>, 3> independentVariables;

      while (accept<Token::Comma>()) {
        TRY(var, parseExpression());
        independentVariables.push_back(std::move(*var));
      }

      EXPECT(Token::RPar);
      accept<Token::String>();
      EXPECT(Token::Semicolon);

      return Class::partialDerFunction(loc, name->getValue(), std::move(*derivedFunction), independentVariables);
    }

    accept<Token::String>();

    llvm::SmallVector<std::unique_ptr<Member>, 3> members;
    llvm::SmallVector<std::unique_ptr<EquationsBlock>, 3> equationsBlocks;
    llvm::SmallVector<std::unique_ptr<EquationsBlock>, 3> initialEquationsBlocks;
    llvm::SmallVector<std::unique_ptr<Algorithm>, 3> algorithms;
    llvm::SmallVector<std::unique_ptr<Class>, 3> innerClasses;

    // Whether the first elements list is allowed to be encountered or not.
    // In fact, the class definition allows a first elements list definition
    // and then others more if preceded by "public" or "protected", but no
    // more "lone" definitions are allowed if any of those keywords are
    // encountered.

    bool firstElementListParsable = true;

    while (current != Token::End && current != Token::Annotation) {
      if (current == Token::Equation) {
        TRY(equationsBlock, parseEquationsBlock());
        equationsBlocks.push_back(std::move(*equationsBlock));
        continue;
      }

      if (accept<Token::Initial>()) {
        TRY(equationsBlock, parseEquationsBlock());
        initialEquationsBlocks.push_back(std::move(*equationsBlock));
        continue;
      }

      if (current == Token::Algorithm) {
        TRY(algorithm, parseAlgorithmSection());
        algorithms.emplace_back(std::move(*algorithm));
        continue;
      }

      if (current == Token::Class ||
          current == Token::Function ||
          current == Token::Model ||
          current == Token::Record) {
        TRY(innerClass, parseClassDefinition());
        innerClasses.emplace_back(std::move(*innerClass));
        continue;
      }

      if (accept<Token::Public>()) {
        TRY(elementList, parseElementList(true));

        for (auto& element : *elementList) {
          members.push_back(std::move(element));
        }

        firstElementListParsable = false;
        continue;
      }

      if (accept<Token::Protected>()) {
        TRY(elementList, parseElementList(false));

        for (auto& element : *elementList) {
          members.push_back(std::move(element));
        }

        firstElementListParsable = false;
        continue;
      }

      if (firstElementListParsable) {
        TRY(elementList, parseElementList(true));

        for (auto& element : *elementList) {
          members.push_back(std::move(element));
        }
      }
    }

    // Parse an optional annotation
    llvm::Optional<std::unique_ptr<Annotation>> clsAnnotation;

    if (current == Token::Annotation) {
      TRY(annotation, parseAnnotation());
      clsAnnotation = std::move(*annotation);
      EXPECT(Token::Semicolon);
    } else {
      clsAnnotation = llvm::None;
    }

    // The class name must be present also after the 'end' keyword
    EXPECT(Token::End);
    TRY(endName, parseIdentifier());

    if (name->getValue() != endName->getValue()) {
      diagnostics->emitError<UnexpectedIdentifierMessage>(
          endName->getLocation(), endName->getValue(), name->getValue());

      return llvm::None;
    }

    EXPECT(Token::Semicolon);

    if (classType == ClassType::Function) {
      return Class::standardFunction(loc, isPure, name->getValue(), members, algorithms, std::move(clsAnnotation));

    } else if (classType == ClassType::Model) {
      return Class::model(
          loc, name->getValue(), members, equationsBlocks, initialEquationsBlocks, algorithms, innerClasses);

    } else if (classType == ClassType::Package) {
      return Class::package(loc, name->getValue(), innerClasses);

    } else if (classType == ClassType::Record) {
      return Class::record(loc, name->getValue(), members);
    }

    llvm_unreachable("Unknown class type");
    return llvm::None;
  }

  llvm::Optional<std::unique_ptr<Modification>> Parser::parseModification()
  {
    auto loc = lexer.getTokenPosition();

    if (accept<Token::EqualityOperator>() || accept<Token::AssignmentOperator>()) {
      TRY(expression, parseExpression());
      loc.end = (*expression)->getLocation().end;
      return Modification::build(std::move(loc), std::move(*expression));
    }

    TRY(classModification, parseClassModification());
    loc.end = (*classModification)->getLocation().end;

    if (accept<Token::EqualityOperator>()) {
      TRY(expression, parseExpression());
      loc.end = (*expression)->getLocation().end;
      return Modification::build(std::move(loc), std::move(*classModification), std::move(*expression));
    }

    return Modification::build(std::move(loc), std::move(*classModification));
  }

  llvm::Optional<std::unique_ptr<ClassModification>> Parser::parseClassModification()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::LPar);
    TRY(argumentList, parseArgumentList());
    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::RPar);
    return ClassModification::build(std::move(loc), std::move(*argumentList));
  }

  llvm::Optional<std::vector<std::unique_ptr<Argument>>> Parser::parseArgumentList()
  {
    std::vector<std::unique_ptr<Argument>> arguments;

    do {
      TRY(arg, parseArgument());
      arguments.push_back(std::move(*arg));
    } while (accept<Token::Comma>());

    return arguments;
  }

  llvm::Optional<std::unique_ptr<Argument>> Parser::parseArgument()
  {
    auto loc = lexer.getTokenPosition();

    if (current == Token::Redeclare) {
      TRY(elementRedeclaration, parseElementRedeclaration());
      return std::move(*elementRedeclaration);
    }

    bool each = accept<Token::Each>();
    bool final = accept<Token::Final>();

    if (current == Token::Replaceable) {
      TRY(elementReplaceable, parseElementReplaceable(each, final));
      return std::move(*elementReplaceable);
    }

    TRY(elementModification, parseElementModification(each, final));
    return std::move(*elementModification);
  }

  llvm::Optional<std::unique_ptr<Argument>> Parser::parseElementModification(bool each, bool final)
  {
    auto loc = lexer.getTokenPosition();

    auto name = lexer.getIdentifier();
    EXPECT(Token::Identifier);

    if (current != Token::LPar && current != Token::EqualityOperator && current != Token::AssignmentOperator) {
      return Argument::elementModification(loc, each, final, name);
    }

    TRY(modification, parseModification());
    return Argument::elementModification(loc, each, final, name, std::move(*modification));
  }

  llvm::Optional<std::unique_ptr<Argument>> Parser::parseElementRedeclaration()
  {
    llvm_unreachable("Not implemented");
  }

  llvm::Optional<std::unique_ptr<Argument>> Parser::parseElementReplaceable(bool each, bool final)
  {
    llvm_unreachable("Not implemented");
  }

  llvm::Optional<std::unique_ptr<Algorithm>> Parser::parseAlgorithmSection()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::Algorithm);

    llvm::SmallVector<std::unique_ptr<Statement>, 10> statements;

    while (
        current != Token::End && current != Token::Public &&
        current != Token::Protected && current != Token::Equation &&
        current != Token::Algorithm && current != Token::External &&
        current != Token::Annotation && current != Token::EndOfFile) {
      TRY(statement, parseStatement());
      loc.end = lexer.getTokenPosition().end;
      EXPECT(Token::Semicolon);
      statements.push_back(std::move(*statement));
    }

    return Algorithm::build(std::move(loc), std::move(statements));
  }

  llvm::Optional<std::unique_ptr<Equation>> Parser::parseEquation()
  {
    auto loc = lexer.getTokenPosition();

    TRY(lhs, parseExpression());
    EXPECT(Token::EqualityOperator);
    TRY(rhs, parseExpression());

    loc.end = (*rhs)->getLocation().end;
    accept<Token::String>();

    return Equation::build(std::move(loc), std::move(*lhs), std::move(*rhs));
  }

  llvm::Optional<std::unique_ptr<Statement>> Parser::parseStatement()
  {
    if (current == Token::If) {
      TRY(statement, parseIfStatement());
      return std::move(*statement);
    }

    if (current == Token::For) {
      TRY(statement, parseForStatement());
      return std::move(*statement);
    }

    if (current == Token::While) {
      TRY(statement, parseWhileStatement());
      return std::move(*statement);
    }

    if (current == Token::When) {
      TRY(statement, parseWhenStatement());
      return std::move(*statement);
    }

    if (current == Token::Break) {
      auto loc = lexer.getTokenPosition();
      EXPECT(Token::Break);
      return Statement::breakStatement(std::move(loc));
    }

    if (current == Token::Return) {
      auto loc = lexer.getTokenPosition();
      EXPECT(Token::Return);
      return Statement::returnStatement(std::move(loc));
    }

    // Assignment statement
    auto loc = lexer.getTokenPosition();

    if (accept<Token::LPar>()) {
      TRY(destinations, parseOutputExpressionList());
      loc.end = lexer.getTokenPosition().end;
      auto destinationsTuple = Expression::tuple(loc, Type::unknown(), std::move(destinations.getValue()));

      EXPECT(Token::RPar);
      EXPECT(Token::AssignmentOperator);
      TRY(function, parseComponentReference());
      TRY(functionCallArgs, parseFunctionCallArgs());

      loc.end = functionCallArgs->getLocation().end;
      auto call = Expression::call(loc, Type::unknown(), std::move(*function), std::move(functionCallArgs->getValue()));

      return Statement::assignmentStatement(std::move(loc), std::move(destinationsTuple), std::move(call));
    }

    TRY(destination, parseComponentReference());
    EXPECT(Token::AssignmentOperator);
    TRY(expression, parseExpression());
    loc.end = (*expression)->getLocation().end;

    return Statement::assignmentStatement(std::move(loc), std::move(*destination), std::move(*expression));
  }

  llvm::Optional<std::unique_ptr<Statement>> Parser::parseIfStatement()
  {
    auto loc = lexer.getTokenPosition();
    llvm::SmallVector<IfStatement::Block, 3> blocks;

    EXPECT(Token::If);
    TRY(ifCondition, parseExpression());
    EXPECT(Token::Then);

    llvm::SmallVector<std::unique_ptr<Statement>, 3> ifStatements;

    while (current != Token::ElseIf &&
           current != Token::Else &&
           current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      ifStatements.push_back(std::move(*statement));
    }

    blocks.emplace_back(std::move(*ifCondition), ifStatements);

    while (current != Token::Else && current != Token::End) {
      EXPECT(Token::ElseIf);
      TRY(elseIfCondition, parseExpression());
      EXPECT(Token::Then);
      llvm::SmallVector<std::unique_ptr<Statement>, 3> elseIfStatements;

      while (current != Token::ElseIf &&
             current != Token::Else &&
             current != Token::End) {
        TRY(statement, parseStatement());
        EXPECT(Token::Semicolon);
        elseIfStatements.push_back(std::move(*statement));
      }

      blocks.emplace_back(std::move(*elseIfCondition), std::move(elseIfStatements));
    }

    auto elseBlockLoc = lexer.getTokenPosition();

    if (accept<Token::Else>()) {
      llvm::SmallVector<std::unique_ptr<Statement>, 3> elseStatements;

      while (current != Token::End) {
        TRY(statement, parseStatement());
        EXPECT(Token::Semicolon);
        elseStatements.push_back(std::move(*statement));
      }

      // Being the last block, it can be discarded if empty
      if (!elseStatements.empty()) {
        blocks.emplace_back(
            Expression::constant(elseBlockLoc, makeType<bool>(), true),
            std::move(elseStatements));
      }
    }

    EXPECT(Token::End);

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::If);

    return Statement::ifStatement(std::move(loc), blocks);
  }

  llvm::Optional<std::unique_ptr<Statement>> Parser::parseForStatement()
  {
    auto loc = lexer.getTokenPosition();

    EXPECT(Token::For);
    TRY(induction, parseInduction());
    EXPECT(Token::Loop);

    llvm::SmallVector<std::unique_ptr<Statement>, 3> statements;

    while (current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(Token::End);

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::For);

    return Statement::forStatement(std::move(loc), std::move(*induction), std::move(statements));
  }

  llvm::Optional<std::unique_ptr<Statement>> Parser::parseWhileStatement()
  {
    auto loc = lexer.getTokenPosition();

    EXPECT(Token::While);
    TRY(condition, parseExpression());
    EXPECT(Token::Loop);

    std::vector<std::unique_ptr<Statement>> statements;

    while (current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(Token::End);

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::While);

    return Statement::whileStatement(std::move(loc), std::move(*condition), std::move(statements));
  }

  llvm::Optional<std::unique_ptr<Statement>> Parser::parseWhenStatement()
  {
    auto loc = lexer.getTokenPosition();

    EXPECT(Token::When);
    TRY(condition, parseExpression());
    EXPECT(Token::Loop);

    std::vector<std::unique_ptr<Statement>> statements;

    while (current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(Token::End);

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::When);

    return Statement::whenStatement(std::move(loc), std::move(*condition), std::move(statements));
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseExpression()
  {
    return parseSimpleExpression();
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseSimpleExpression()
  {
    auto loc = lexer.getTokenPosition();
    TRY(l1, parseLogicalExpression());
    loc.end = (*l1)->getLocation().end;

    if (accept<Token::Colon>()) {
      std::vector<std::unique_ptr<Expression>> arguments;
      TRY(l2, parseLogicalExpression());
      loc.end = (*l2)->getLocation().end;

      arguments.push_back(std::move(*l1));
      arguments.push_back(std::move(*l2));

      if (accept<Token::Colon>()) {
        TRY(l3, parseLogicalExpression());
        loc.end = (*l3)->getLocation().end;
        arguments.push_back(std::move(*l3));
      }

      return Expression::operation(std::move(loc), Type::unknown(), OperationKind::range, std::move(arguments));
    }

    return std::move(*l1);
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseLogicalExpression()
  {
    auto loc = lexer.getTokenPosition();

    std::vector<std::unique_ptr<Expression>> logicalTerms;
    TRY(logicalTerm, parseLogicalTerm());
    loc.end = (*logicalTerm)->getLocation().end;

    if (current != Token::Or) {
      return std::move(*logicalTerm);
    }

    logicalTerms.push_back(std::move(*logicalTerm));

    while (accept<Token::Or>()) {
      TRY(additionalLogicalTerm, parseLogicalTerm());
      loc.end = (*additionalLogicalTerm)->getLocation().end;
      logicalTerms.emplace_back(std::move(*additionalLogicalTerm));
    }

    return Expression::operation(std::move(loc), Type::unknown(), OperationKind::lor, std::move(logicalTerms));
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseLogicalTerm()
  {
    auto loc = lexer.getTokenPosition();

    std::vector<std::unique_ptr<Expression>> logicalFactors;
    TRY(logicalFactor, parseLogicalFactor());
    loc.end = (*logicalFactor)->getLocation().end;

    if (current != Token::And) {
      return std::move(*logicalFactor);
    }

    logicalFactors.push_back(std::move(*logicalFactor));

    while (accept<Token::And>()) {
      TRY(additionalLogicalFactor, parseLogicalFactor());
      loc.end = (*additionalLogicalFactor)->getLocation().end;
      logicalFactors.emplace_back(std::move(*additionalLogicalFactor));
    }

    return Expression::operation(std::move(loc), Type::unknown(), OperationKind::land, std::move(logicalFactors));
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseLogicalFactor()
  {
    auto loc = lexer.getTokenPosition();
    bool negated = accept<Token::Not>();

    TRY(relation, parseRelation());
    loc.end = (*relation)->getLocation().end;

    if (negated) {
      return Expression::operation(std::move(loc), Type::unknown(), OperationKind::lnot, std::move(*relation));
    }

    return std::move(*relation);
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseRelation()
  {
    auto loc = lexer.getTokenPosition();

    TRY(lhs, parseArithmeticExpression());
    loc.end = (*lhs)->getLocation().end;

    auto op = parseRelationalOperator();

    if (!op.hasValue()) {
      return std::move(*lhs);
    }

    TRY(rhs, parseArithmeticExpression());
    loc.end = (*rhs)->getLocation().end;

    return Expression::operation(
        std::move(loc), Type::unknown(), op.getValue(),
        llvm::makeArrayRef({ std::move(*lhs), std::move(*rhs) }));
  }

  llvm::Optional<OperationKind> Parser::parseRelationalOperator()
  {
    if (accept<Token::GreaterEqual>()) {
      return OperationKind::greaterEqual;
    }

    if (accept<Token::Greater>()) {
      return OperationKind::greater;
    }

    if (accept<Token::LessEqual>()) {
      return OperationKind::lessEqual;
    }

    if (accept<Token::Less>()) {
      return OperationKind::less;
    }

    if (accept<Token::Equal>()) {
      return OperationKind::equal;
    }

    if (accept<Token::NotEqual>()) {
      return OperationKind::different;
    }

    return llvm::None;
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseArithmeticExpression()
  {
    auto loc = lexer.getTokenPosition();
    bool negative = false;

    if (accept<Token::Minus>()) {
      negative = true;
    } else {
      accept<Token::Plus>();
    }

    TRY(term, parseTerm());
    loc.end = (*term)->getLocation().end;

    auto result = negative
        ? Expression::operation(loc, Type::unknown(), OperationKind::negate, std::move(*term))
        : std::move(*term);

    while (current == Token::Plus || current == Token::PlusEW || current == Token::Minus || current == Token::MinusEW) {
      TRY(addOperator, parseAddOperator());
      TRY(rhs, parseTerm());
      loc.end = (*rhs)->getLocation().end;

      std::vector<std::unique_ptr<Expression>> args;
      args.push_back(std::move(result));
      args.push_back(std::move(*rhs));
      result = Expression::operation(loc, Type::unknown(), *addOperator, args);
    }

    return std::move(result);
  }

  llvm::Optional<OperationKind> Parser::parseAddOperator()
  {
    if (accept<Token::Plus>()) {
      return OperationKind::add;
    }

    if (accept<Token::PlusEW>()) {
      return OperationKind::addEW;
    }

    if (accept<Token::Minus>()) {
      return OperationKind::subtract;
    }

    if (accept<Token::MinusEW>()) {
      return OperationKind::subtractEW;
    }

    return llvm::None;
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseTerm()
  {
    auto loc = lexer.getTokenPosition();

    TRY(factor, parseFactor());
    loc.end = (*factor)->getLocation().end;

    auto result = std::move(*factor);

    while (current == Token::Product || current == Token::ProductEW || current == Token::Division || current == Token::DivisionEW) {
      TRY(mulOperator, parseMulOperator());
      TRY(rhs, parseFactor());
      loc.end = (*rhs)->getLocation().end;

      std::vector<std::unique_ptr<Expression>> args;
      args.push_back(std::move(result));
      args.push_back(std::move(*rhs));
      result = Expression::operation(loc, Type::unknown(), *mulOperator, args);
    }

    return std::move(result);
  }

  llvm::Optional<OperationKind> Parser::parseMulOperator()
  {
    if (accept<Token::Product>()) {
      return OperationKind::multiply;
    }

    if (accept<Token::ProductEW>()) {
      return OperationKind::multiplyEW;
    }

    if (accept<Token::Division>()) {
      return OperationKind::divide;
    }

    if (accept<Token::DivisionEW>()) {
      return OperationKind::divideEW;
    }

    return llvm::None;
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseFactor()
  {
    auto loc = lexer.getTokenPosition();

    TRY(primary, parsePrimary());
    loc.end = (*primary)->getLocation().end;

    auto result = std::move(*primary);

    while (current == Token::Pow || current == Token::PowEW) {
      std::vector<std::unique_ptr<Expression>> args;
      args.push_back(std::move(result));

      if (accept<Token::Pow>()) {
        TRY(rhs, parsePrimary());
        loc.end = (*rhs)->getLocation().end;
        args.push_back(std::move(*rhs));

        result = Expression::operation(loc, Type::unknown(), OperationKind::powerOf, args);
        continue;
      }

      EXPECT(Token::PowEW);
      TRY(rhs, parsePrimary());
      loc.end = (*rhs)->getLocation().end;
      args.push_back(std::move(*rhs));

      result = Expression::operation(loc, Type::unknown(), OperationKind::powerOfEW, args);
    }

    return std::move(result);
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parsePrimary()
  {
    auto loc = lexer.getTokenPosition();

    if (current == Token::Integer) {
      TRY(value, parseIntValue());
      return Expression::constant(value->getLocation(), makeType<BuiltInType::Integer>(), value->getValue());
    }

    if (current == Token::FloatingPoint) {
      TRY(value, parseFloatValue());
      return Expression::constant(value->getLocation(), makeType<BuiltInType::Real>(), value->getValue());
    }

    if (current == Token::String) {
      TRY(value, parseString());
      return Expression::constant(value->getLocation(), makeType<std::string>(), value->getValue());
    }

    if (current == Token::True || current == Token::False) {
      TRY(value, parseBoolValue());
      return Expression::constant(value->getLocation(), makeType<BuiltInType::Boolean>(), value->getValue());
    }

    std::unique_ptr<Expression> result;

    if (accept<Token::LPar>()) {
      TRY(outputExpressionList, parseOutputExpressionList());
      loc.end = lexer.getTokenPosition().end;
      EXPECT(Token::RPar);

      if (outputExpressionList->size() == 1) {
        (*outputExpressionList)[0]->setLocation(loc);
        return std::move((*outputExpressionList)[0]);
      }

      result = Expression::tuple(loc, Type::unknown(), *outputExpressionList);

    } else if (accept<Token::LCurly>()) {
      TRY(arrayArguments, parseArrayArguments());
      loc.end = lexer.getTokenPosition().end;
      EXPECT(Token::RCurly);

      result = Expression::array(loc, Type::unknown(), *arrayArguments);

    } else if (accept<Token::Der>()) {
      auto function = Expression::reference(loc, Type::unknown(), "der");

      TRY(functionCallArgs, parseFunctionCallArgs());
      loc.end = functionCallArgs->getLocation().end;

      result = Expression::call(loc, Type::unknown(), std::move(function), std::move(functionCallArgs->getValue()));

    } else if (current == Token::Identifier) {
      TRY(identifier, parseComponentReference());
      loc.end = (*identifier)->getLocation().end;

      if (current != Token::LPar) {
        return std::move(*identifier);
      }

      TRY(functionCallArgs, parseFunctionCallArgs());
      loc.end = functionCallArgs->getLocation().end;

      result = Expression::call(std::move(loc), Type::unknown(), std::move(*identifier), functionCallArgs->getValue());
    }

    assert(result != nullptr);

    if (current == Token::LSquare) {
      TRY(arraySubscripts, parseArraySubscripts());
      loc.end = arraySubscripts.getValue().getLocation().end;

      std::vector<std::unique_ptr<Expression>> args;
      args.push_back(std::move(result));

      for (auto& subscript : arraySubscripts->getValue()) {
        args.push_back(std::move(subscript));
      }

      result = Expression::operation(loc, Type::unknown(), OperationKind::subscription, args);
    }

    return std::move(result);
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseComponentReference()
  {
    auto loc = lexer.getTokenPosition();
    bool globalLookup = accept<Token::Dot>();

    TRY(name, parseIdentifier());
    loc.end = name->getLocation().end;

    auto result = Expression::reference(loc, Type::unknown(), name->getValue(), globalLookup);

    if (current == Token::LSquare) {
      TRY(arraySubscripts, parseArraySubscripts());
      loc.end = arraySubscripts.getValue().getLocation().end;

      std::vector<std::unique_ptr<Expression>> args;
      args.push_back(std::move(result));

      for (auto& subscript : arraySubscripts->getValue()) {
        args.push_back(std::move(subscript));
      }

      result = Expression::operation(loc, Type::unknown(), OperationKind::subscription, args);
    }

    return std::move(result);
  }

  llvm::Optional<Parser::ValueWrapper<std::vector<std::unique_ptr<Expression>>>> Parser::parseFunctionCallArgs()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::LPar);

    std::vector<std::unique_ptr<Expression>> args;

    if (current != Token::RPar) {
      TRY(functionArguments, parseFunctionArguments());

      for (auto& arg : *functionArguments) {
        args.push_back(std::move(std::move(arg)));
      }
    }

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::RPar);

    return ValueWrapper(std::move(loc), std::move(args));
  }

  llvm::Optional<std::vector<std::unique_ptr<Expression>>> Parser::parseFunctionArguments()
  {
    std::vector<std::unique_ptr<Expression>> arguments;

    TRY(firstArg, parseExpression());
    arguments.push_back(std::move(*firstArg));

    if (accept<Token::Comma>()) {
      TRY(otherArgs, parseFunctionArgumentsNonFirst());

      for (auto& otherArg : *otherArgs) {
        arguments.push_back(std::move(otherArg));
      }
    }

    return arguments;
  }

  llvm::Optional<std::vector<std::unique_ptr<Expression>>> Parser::parseFunctionArgumentsNonFirst()
  {
    std::vector<std::unique_ptr<Expression>> arguments;

    do {
      TRY(argument, parseExpression());
      arguments.push_back(std::move(*argument));
    } while (accept<Token::Comma>());

    return arguments;
  }

  llvm::Optional<std::vector<std::unique_ptr<Expression>>> Parser::parseArrayArguments()
  {
    auto loc = lexer.getTokenPosition();
    std::vector<std::unique_ptr<Expression>> arguments;

    TRY(argument, parseExpression());
    loc.end = (*argument)->getLocation().end;
    arguments.push_back(std::move(*argument));

    if (accept<Token::Comma>()) {
      TRY(otherArguments, parseArrayArgumentsNonFirst());

      for (auto& otherArgument : *otherArguments) {
        loc.end = otherArgument->getLocation().end;
        arguments.push_back(std::move(otherArgument));
      }
    }

    return arguments;
  }

  llvm::Optional<std::vector<std::unique_ptr<Expression>>> Parser::parseArrayArgumentsNonFirst()
  {
    std::vector<std::unique_ptr<Expression>> arguments;

    do {
      TRY(argument, parseFunctionArgument());
      arguments.push_back(std::move(*argument));
    } while (accept<Token::Comma>());

    return arguments;
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseFunctionArgument()
  {
    return parseExpression();
  }

  llvm::Optional<std::vector<std::unique_ptr<Expression>>> Parser::parseOutputExpressionList()
  {
    std::vector<std::unique_ptr<Expression>> expressions;

    while (current != Token::RPar) {
      auto loc = lexer.getTokenPosition();

      if (accept<Token::Comma>()) {
        expressions.push_back(ReferenceAccess::dummy(std::move(loc), Type::unknown()));
        continue;
      }

      TRY(expression, parseExpression());
      expressions.push_back(std::move(*expression));
      accept<Token::Comma>();
    }

    return expressions;
  }

  llvm::Optional<Parser::ValueWrapper<std::vector<std::unique_ptr<Expression>>>> Parser::parseArraySubscripts()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::LSquare);

    std::vector<std::unique_ptr<Expression>> subscripts;

    do {
      TRY(subscript, parseSubscript());
      subscripts.push_back(std::move(*subscript));
    } while (accept<Token::Comma>());

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::RSquare);

    return ValueWrapper(std::move(loc), std::move(subscripts));
  }

  llvm::Optional<std::unique_ptr<Expression>> Parser::parseSubscript()
  {
    auto loc = lexer.getTokenPosition();

    if (accept<Token::Colon>()) {
      return Expression::constant(loc, Type(BuiltInType::Integer), -1);
    }

    TRY(expression, parseExpression());
    return std::move(*expression);
  }

  llvm::Optional<std::unique_ptr<Annotation>> Parser::parseAnnotation()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::Annotation);
    TRY(classModification, parseClassModification());
    loc.end = (*classModification)->getLocation().end;
    return std::make_unique<Annotation>(std::move(loc), std::move(*classModification));
  }

  llvm::Optional<std::unique_ptr<EquationsBlock>> Parser::parseEquationsBlock()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::Equation);

    llvm::SmallVector<std::unique_ptr<Equation>, 3> equations;
    llvm::SmallVector<std::unique_ptr<ForEquation>, 3> forEquations;

    while (
        current != Token::End &&
        current != Token::Public &&
        current != Token::Protected &&
        current != Token::Equation &&
        current != Token::Initial &&
        current != Token::Algorithm &&
        current != Token::External &&
        current != Token::Annotation &&
        current != Token::EndOfFile) {
      if (current == Token::For) {
        TRY(currentForEquations, parseForEquations());

        for (auto& forEquation : *currentForEquations) {
          forEquations.push_back(std::move(forEquation));
        }
      } else {
        TRY(equation, parseEquation());
        equations.push_back(std::move(*equation));
        EXPECT(Token::Semicolon);
      }
    }

    return std::make_unique<EquationsBlock>(std::move(loc), equations, forEquations);
  }

  llvm::Optional<std::vector<std::unique_ptr<ast::ForEquation>>> Parser::parseForEquations()
  {
    std::vector<std::unique_ptr<ForEquation>> result;

    EXPECT(Token::For);
    TRY(induction, parseInduction());
    EXPECT(Token::Loop);

    while (current != Token::End) {
      if (current == Token::For) {
        TRY(nestedForEquations, parseForEquations());

        for (auto& forEquation : *nestedForEquations) {
          std::vector<std::unique_ptr<Induction>> newInductions;
          newInductions.push_back((*induction)->clone());

          for (const auto& nestedInduction : forEquation->getInductions()) {
            newInductions.push_back(nestedInduction->clone());
          }

          result.push_back(ForEquation::build(
              forEquation->getLocation(), newInductions, forEquation->getEquation()->clone()));
        }
      } else {
        TRY(equation, parseEquation());

        result.push_back(ForEquation::build(
            (*equation)->getLocation(),
            (*induction)->clone(),
            std::move(*equation)));

        EXPECT(Token::Semicolon);
      }
    }

    EXPECT(Token::End);
    EXPECT(Token::For);
    EXPECT(Token::Semicolon);

    return result;
  }

  llvm::Optional<std::unique_ptr<ast::Induction>> Parser::parseInduction()
  {
    auto loc = lexer.getTokenPosition();

    auto variableName = lexer.getIdentifier();
    EXPECT(Token::Identifier);

    EXPECT(Token::In);

    TRY(firstExpression, parseLogicalExpression());
    EXPECT(Token::Colon);
    TRY(secondExpression, parseLogicalExpression());

    if (accept<Token::Colon>()) {
      TRY(thirdExpression, parseLogicalExpression());
      loc.end = (*thirdExpression)->getLocation().end;

      return Induction::build(
          std::move(loc),
          variableName,
          std::move(*firstExpression),
          std::move(*thirdExpression),
          std::move(*secondExpression));
    }

    loc.end = (*secondExpression)->getLocation().end;

    return Induction::build(
        std::move(loc),
        variableName,
        std::move(*firstExpression),
        std::move(*secondExpression),
        Expression::constant(loc, makeType<BuiltInType::Integer>(), 1));
  }

  llvm::Optional<std::vector<std::unique_ptr<Member>>> Parser::parseElementList(bool publicSection)
  {
    std::vector<std::unique_ptr<Member>> members;

    while (current != Token::Public &&
           current != Token::Protected &&
           current != Token::Function &&
           current != Token::Equation &&
           current != Token::Initial &&
           current != Token::Algorithm &&
           current != Token::End &&
           current != Token::Class &&
           current != Token::Function &&
           current != Token::Model &&
           current != Token::Package&&
           current != Token::Record) {
      TRY(member, parseElement(publicSection));
      EXPECT(Token::Semicolon);
      members.push_back(std::move(*member));
    }

    return members;
  }

  llvm::Optional<std::unique_ptr<ast::Member>> Parser::parseElement(bool publicSection)
  {
    accept<Token::Final>();
    TRY(typePrefix, parseTypePrefix());
    TRY(type, parseTypeSpecifier());
    TRY(name, parseIdentifier());

    std::unique_ptr<Modification> modification = nullptr;

    if (current == Token::LPar || current == Token::EqualityOperator) {
      TRY(mod, parseModification());
      modification = std::move(*mod);
    }

    // String comment. Ignore it for now.
    accept<Token::String>();

    // Annotation
    if (current == Token::Annotation) {
      TRY(annotation, parseAnnotation());
      // TODO: handle elements annotations
    }

    return Member::build(
        name->getLocation(),
        name->getValue(),
        std::move(*type),
        std::move(*typePrefix),
        publicSection,
        std::move(modification));
  }

  llvm::Optional<ast::TypePrefix> Parser::parseTypePrefix()
  {
    ParameterQualifier parameterQualifier = ParameterQualifier::none;

    if (accept<Token::Discrete>()) {
      parameterQualifier = ParameterQualifier::discrete;
    } else if (accept<Token::Parameter>()) {
      parameterQualifier = ParameterQualifier::parameter;
    } else if (accept<Token::Constant>()) {
      parameterQualifier = ParameterQualifier::constant;
    }

    IOQualifier ioQualifier = IOQualifier::none;

    if (accept<Token::Input>()) {
      ioQualifier = IOQualifier::input;
    } else if (accept<Token::Output>()) {
      ioQualifier = IOQualifier::output;
    }

    return TypePrefix(parameterQualifier, ioQualifier);
  }

  llvm::Optional<ast::Type> Parser::parseTypeSpecifier()
  {
    std::string name = lexer.getIdentifier();
    EXPECT(Token::Identifier);

    while (accept<Token::Dot>()) {
      name += "." + lexer.getIdentifier();
      EXPECT(Token::Identifier);
    }

    llvm::SmallVector<ArrayDimension, 3> dimensions;

    if (current != Token::Identifier) {
      EXPECT(Token::LSquare);

      do {
        auto loc = lexer.getTokenPosition();

        if (accept<Token::Colon>()) {
          dimensions.push_back(ArrayDimension(-1));
        } else if (accept<Token::Integer>()) {
          dimensions.push_back(ArrayDimension(lexer.getInt()));
        } else {
          TRY(expression, parseExpression());
          dimensions.push_back(ArrayDimension(std::move(*expression)));
        }
      } while (accept<Token::Comma>());

      EXPECT(Token::RSquare);
    }

    if (name == "string") {
      return Type(BuiltInType::String, dimensions);
    }

    if (name == "Boolean") {
      return Type(BuiltInType::Boolean, dimensions);
    }

    if (name == "Integer") {
      return Type(BuiltInType::Integer, dimensions);
    }

    if (name == "Real") {
      return Type(BuiltInType::Real, dimensions);
    }

    return Type(UserDefinedType(name, {}), dimensions);
  }

  llvm::Optional<std::unique_ptr<ast::Expression>> Parser::parseTermModification()
  {
    EXPECT(Token::LPar);

    std::unique_ptr<Expression> expression;

    do {
      accept<Token::Each>();
      auto lastIndentifier = lexer.getIdentifier();
      EXPECT(Token::Identifier);
      EXPECT(Token::EqualityOperator);

      if (lastIndentifier == "start") {
        TRY(exp, parseExpression());
        expression = std::move(*exp);
      }

      if (accept<Token::FloatingPoint>()) {
        continue;
      }

      if (accept<Token::Integer>()) {
        continue;
      }

      if (accept<Token::String>()) {
        continue;
      }

      if (accept<Token::True>()) {
        continue;
      }

      if (accept<Token::False>()) {
        continue;
      }
    } while (accept<Token::Comma>());

    EXPECT(Token::RPar);
    return std::move(expression);
  }
}
