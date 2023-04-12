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
	if (!outVar.has_value()) {          \
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

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseRoot()
  {
    auto loc = lexer.getTokenPosition();
    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 1> classes;

    while (current != Token::EndOfFile) {
      TRY(classDefinition, parseClassDefinition());
      classes.push_back(std::move(*classDefinition));
    }

    auto root = std::make_unique<Root>(SourceRange::unknown());
    root->setInnerClasses(classes);
    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(root));
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

    /*
    while (accept<Token::Dot>()) {
      identifier += "." + lexer.getIdentifier();
      loc.end = lexer.getTokenPosition().end;
      EXPECT(Token::Identifier);
    }
     */

    return ValueWrapper(loc, std::move(identifier));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseClassDefinition()
  {
    auto loc = lexer.getTokenPosition();
    std::unique_ptr<ASTNode> result;

    bool isOperator = accept<Token::Operator>();
    bool isPure = true;

    if (isOperator) {
      if (accept<Token::Record>()) {
        result = std::make_unique<Record>(loc);
      } else {
        EXPECT(Token::Function);
        result = std::make_unique<StandardFunction>(loc);
      }
    } else if (accept<Token::Model>()) {
      result = std::make_unique<Model>(loc);
    } else if (accept<Token::Record>()) {
      result = std::make_unique<Record>(loc);
    } else if (accept<Token::Package>()) {
      result = std::make_unique<Package>(loc);
    } else if (accept<Token::Function>()) {
      result = std::make_unique<StandardFunction>(loc);
    } else if (accept<Token::Pure>()) {
      isPure = true;
      isOperator = accept<Token::Operator>();
      EXPECT(Token::Function);
      result = std::make_unique<StandardFunction>(loc);
    } else if (accept<Token::Impure>()) {
      isPure = false;
      isOperator = accept<Token::Operator>();
      EXPECT(Token::Function);
      result = std::make_unique<StandardFunction>(loc);
    } else {
      EXPECT(Token::Class);
      result = std::make_unique<Model>(loc);
    }

    if (auto standardFunction = result->dyn_cast<StandardFunction>()) {
      standardFunction->setPure(isPure);
    }

    TRY(name, parseIdentifier());
    loc.end = name->getLocation().end;
    result->setLocation(loc);

    if (accept<Token::EqualityOperator>()) {
      // Function derivative
      assert(result->isa<StandardFunction>());
      EXPECT(Token::Der);
      EXPECT(Token::LPar);

      TRY(derivedFunction, parseExpression());
      llvm::SmallVector<std::unique_ptr<ASTNode>, 3> independentVariables;

      while (accept<Token::Comma>()) {
        TRY(var, parseExpression());
        independentVariables.push_back(std::move(*var));
      }

      EXPECT(Token::RPar);
      accept<Token::String>();
      EXPECT(Token::Semicolon);

      auto partialDerFunction = std::make_unique<PartialDerFunction>(loc);
      partialDerFunction->setName(name->getValue());
      partialDerFunction->setDerivedFunction(std::move(*derivedFunction));
      partialDerFunction->setIndependentVariables(independentVariables);
      result = std::move(partialDerFunction);

      return std::move(result);
    }

    accept<Token::String>();

    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> members;
    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> equationsBlocks;
    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> initialEquationsBlocks;
    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> algorithms;
    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> innerClasses;

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
          current == Token::Package ||
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
    llvm::Optional<std::unique_ptr<ASTNode>> clsAnnotation;

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

    result->dyn_cast<Class>()->setName(name->getValue());
    result->dyn_cast<Class>()->setVariables(members);
    result->dyn_cast<Class>()->setEquationsBlocks(equationsBlocks);
    result->dyn_cast<Class>()->setInitialEquationsBlocks(initialEquationsBlocks);
    result->dyn_cast<Class>()->setAlgorithms(algorithms);
    result->dyn_cast<Class>()->setInnerClasses(innerClasses);

    return std::move(result);
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseModification()
  {
    auto loc = lexer.getTokenPosition();

    if (accept<Token::EqualityOperator>() || accept<Token::AssignmentOperator>()) {
      TRY(expression, parseExpression());
      loc.end = (*expression)->getLocation().end;

      auto result = std::make_unique<Modification>(loc);
      result->setExpression(std::move(*expression));
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    TRY(classModification, parseClassModification());
    loc.end = (*classModification)->getLocation().end;

    if (accept<Token::EqualityOperator>()) {
      TRY(expression, parseExpression());
      loc.end = (*expression)->getLocation().end;

      auto result = std::make_unique<Modification>(loc);
      result->setClassModification(std::move(*classModification));
      result->setExpression(std::move(*expression));
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    auto result = std::make_unique<Modification>(loc);
    result->setClassModification(std::move(*classModification));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseClassModification()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::LPar);
    TRY(argumentList, parseArgumentList());
    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::RPar);

    auto result = std::make_unique<ClassModification>(loc);
    result->setArguments(*argumentList);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::vector<std::unique_ptr<ASTNode>>> Parser::parseArgumentList()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    do {
      TRY(arg, parseArgument());
      arguments.push_back(std::move(*arg));
    } while (accept<Token::Comma>());

    return arguments;
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseArgument()
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

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseElementModification(bool each, bool final)
  {
    auto loc = lexer.getTokenPosition();

    auto name = lexer.getIdentifier();
    EXPECT(Token::Identifier);

    auto result = std::make_unique<ElementModification>(loc);
    result->setName(name);
    result->setEachProperty(each);
    result->setFinalProperty(final);

    if (current != Token::LPar &&
        current != Token::EqualityOperator &&
        current != Token::AssignmentOperator) {
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    TRY(modification, parseModification());
    result->setModification(std::move(*modification));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseElementRedeclaration()
  {
    llvm_unreachable("Not implemented");
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseElementReplaceable(bool each, bool final)
  {
    llvm_unreachable("Not implemented");
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseAlgorithmSection()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::Algorithm);

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 10> statements;

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

    auto result = std::make_unique<Algorithm>(loc);
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseEquation()
  {
    auto loc = lexer.getTokenPosition();

    TRY(lhs, parseExpression());
    EXPECT(Token::EqualityOperator);
    TRY(rhs, parseExpression());

    loc.end = (*rhs)->getLocation().end;
    accept<Token::String>();

    auto result = std::make_unique<Equation>(loc);
    result->setLhsExpression(std::move(*lhs));
    result->setRhsExpression(std::move(*rhs));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseStatement()
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

      auto result = std::make_unique<BreakStatement>(loc);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    if (current == Token::Return) {
      auto loc = lexer.getTokenPosition();
      EXPECT(Token::Return);

      auto result = std::make_unique<ReturnStatement>(loc);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    // Assignment statement
    auto loc = lexer.getTokenPosition();

    if (accept<Token::LPar>()) {
      TRY(destinations, parseOutputExpressionList());
      loc.end = lexer.getTokenPosition().end;

      auto destinationsTuple = std::make_unique<Tuple>(loc);
      destinationsTuple->setExpressions(destinations.value());

      EXPECT(Token::RPar);
      EXPECT(Token::AssignmentOperator);
      TRY(function, parseComponentReference());
      TRY(functionCallArgs, parseFunctionCallArgs());

      loc.end = functionCallArgs->getLocation().end;

      auto call = std::make_unique<Call>(loc);
      call->setCallee(std::move(*function));
      call->setArguments(functionCallArgs->getValue());

      auto result = std::make_unique<AssignmentStatement>(loc);
      result->setDestinations(std::move(destinationsTuple));
      result->setExpression(std::move(call));
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    TRY(destination, parseComponentReference());
    EXPECT(Token::AssignmentOperator);
    TRY(expression, parseExpression());
    loc.end = (*expression)->getLocation().end;

    auto result = std::make_unique<AssignmentStatement>(loc);
    result->setDestinations(std::move(*destination));
    result->setExpression(std::move(*expression));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseIfStatement()
  {
    auto loc = lexer.getTokenPosition();
    auto result = std::make_unique<IfStatement>(loc);

    EXPECT(Token::If);
    TRY(ifCondition, parseExpression());
    EXPECT(Token::Then);

    auto statementsBlockLoc = lexer.getTokenPosition();
    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> ifStatements;

    while (current != Token::ElseIf &&
           current != Token::Else &&
           current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      auto& stmnt = ifStatements.emplace_back(std::move(*statement));
      statementsBlockLoc.end = stmnt->getLocation().end;
    }

    auto ifBlock = std::make_unique<StatementsBlock>(statementsBlockLoc);
    ifBlock->setBody(ifStatements);
    loc.end = ifBlock->getLocation().end;

    result->setIfCondition(std::move(*ifCondition));
    result->setIfBlock(std::move(ifBlock));

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> elseIfConditions;
    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> elseIfBlocks;

    while (current != Token::Else && current != Token::End) {
      EXPECT(Token::ElseIf);
      TRY(elseIfCondition, parseExpression());
      elseIfConditions.push_back(std::move(*elseIfCondition));
      EXPECT(Token::Then);
      llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> elseIfStatements;
      auto elseIfStatementsBlockLoc = lexer.getTokenPosition();

      while (current != Token::ElseIf &&
             current != Token::Else &&
             current != Token::End) {
        TRY(statement, parseStatement());
        EXPECT(Token::Semicolon);
        auto& stmnt = elseIfStatements.emplace_back(std::move(*statement));
        elseIfStatementsBlockLoc.end = stmnt->getLocation().end;
      }

      auto elseIfBlock = std::make_unique<StatementsBlock>(statementsBlockLoc);
      elseIfBlock->setBody(elseIfStatements);
      loc.end = elseIfBlock->getLocation().end;
      elseIfBlocks.push_back(std::move(elseIfBlock));
    }

    result->setElseIfConditions(elseIfConditions);
    result->setElseIfBlocks(elseIfBlocks);

    if (accept<Token::Else>()) {
      auto elseBlockLoc = lexer.getTokenPosition();
      llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> elseStatements;

      while (current != Token::End) {
        TRY(statement, parseStatement());
        EXPECT(Token::Semicolon);
        auto& stmnt = elseStatements.emplace_back(std::move(*statement));
        elseBlockLoc.end = stmnt->getLocation().end;
      }

      auto elseBlock = std::make_unique<StatementsBlock>(elseBlockLoc);
      elseBlock->setBody(elseStatements);
      loc.end = elseBlockLoc.end;
      result->setElseBlock(std::move(elseBlock));
    }

    EXPECT(Token::End);
    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::If);

    result->setLocation(loc);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseForStatement()
  {
    auto loc = lexer.getTokenPosition();

    EXPECT(Token::For);
    TRY(induction, parseInduction());
    EXPECT(Token::Loop);

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> statements;

    while (current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(Token::End);

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::For);

    auto result = std::make_unique<ForStatement>(loc);
    result->setInduction(std::move(*induction));
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseWhileStatement()
  {
    auto loc = lexer.getTokenPosition();

    EXPECT(Token::While);
    TRY(condition, parseExpression());
    EXPECT(Token::Loop);

    std::vector<std::unique_ptr<ast::ASTNode>> statements;

    while (current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(Token::End);

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::While);

    auto result = std::make_unique<WhileStatement>(loc);
    result->setCondition(std::move(*condition));
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseWhenStatement()
  {
    auto loc = lexer.getTokenPosition();

    EXPECT(Token::When);
    TRY(condition, parseExpression());
    EXPECT(Token::Loop);

    std::vector<std::unique_ptr<ast::ASTNode>> statements;

    while (current != Token::End) {
      TRY(statement, parseStatement());
      EXPECT(Token::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(Token::End);

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::When);

    auto result = std::make_unique<WhenStatement>(loc);
    result->setCondition(std::move(*condition));
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseExpression()
  {
    auto loc = lexer.getTokenPosition();

    if (accept<Token::If>()) {
      TRY(condition, parseExpression());
      EXPECT(Token::Then);
      TRY(trueExpression, parseExpression());
      EXPECT(Token::Else);
      TRY(falseExpression, parseExpression());

      loc.end = (*falseExpression)->getLocation().end;

      llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> args;
      args.push_back(std::move(*condition));
      args.push_back(std::move(*trueExpression));
      args.push_back(std::move(*falseExpression));

      auto result = std::make_unique<Operation>(loc);
      result->setOperationKind(OperationKind::ifelse);
      result->setArguments(args);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    return parseSimpleExpression();
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseSimpleExpression()
  {
    auto loc = lexer.getTokenPosition();
    TRY(l1, parseLogicalExpression());
    loc.end = (*l1)->getLocation().end;

    if (accept<Token::Colon>()) {
      std::vector<std::unique_ptr<ast::ASTNode>> arguments;
      TRY(l2, parseLogicalExpression());
      loc.end = (*l2)->getLocation().end;

      arguments.push_back(std::move(*l1));
      arguments.push_back(std::move(*l2));

      if (accept<Token::Colon>()) {
        TRY(l3, parseLogicalExpression());
        loc.end = (*l3)->getLocation().end;
        arguments.push_back(std::move(*l3));
      }

      auto result = std::make_unique<Operation>(loc);
      result->setOperationKind(OperationKind::range);
      result->setArguments(arguments);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    return std::move(*l1);
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseLogicalExpression()
  {
    auto loc = lexer.getTokenPosition();

    std::vector<std::unique_ptr<ast::ASTNode>> logicalTerms;
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

    auto result = std::make_unique<Operation>(loc);
    result->setOperationKind(OperationKind::lor);
    result->setArguments(logicalTerms);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseLogicalTerm()
  {
    auto loc = lexer.getTokenPosition();

    std::vector<std::unique_ptr<ast::ASTNode>> logicalFactors;
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

    auto result = std::make_unique<Operation>(loc);
    result->setOperationKind(OperationKind::land);
    result->setArguments(logicalFactors);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseLogicalFactor()
  {
    auto loc = lexer.getTokenPosition();
    bool negated = accept<Token::Not>();

    TRY(relation, parseRelation());
    loc.end = (*relation)->getLocation().end;

    if (negated) {
      auto result = std::make_unique<Operation>(loc);
      result->setOperationKind(OperationKind::lnot);
      result->setArguments(std::move(*relation));
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    return std::move(*relation);
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseRelation()
  {
    auto loc = lexer.getTokenPosition();

    TRY(lhs, parseArithmeticExpression());
    loc.end = (*lhs)->getLocation().end;

    auto op = parseRelationalOperator();

    if (!op.has_value()) {
      return std::move(*lhs);
    }

    TRY(rhs, parseArithmeticExpression());
    loc.end = (*rhs)->getLocation().end;

    auto result = std::make_unique<Operation>(loc);
    result->setOperationKind(op.value());
    result->setArguments({ std::move(*lhs), std::move(*rhs) });
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
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

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseArithmeticExpression()
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

    std::unique_ptr<ast::ASTNode> result = std::move(*term);

    if (negative) {
      auto newResult = std::make_unique<Operation>(loc);
      newResult->setOperationKind(ast::OperationKind::negate);
      newResult->setArguments(std::move(result));
      result = std::move(newResult);
    }

    while (current == Token::Plus ||
           current == Token::PlusEW ||
           current == Token::Minus ||
           current == Token::MinusEW) {
      TRY(addOperator, parseAddOperator());
      TRY(rhs, parseTerm());
      loc.end = (*rhs)->getLocation().end;

      std::vector<std::unique_ptr<ast::ASTNode>> args;
      args.push_back(std::move(result));
      args.push_back(std::move(*rhs));

      auto newResult = std::make_unique<Operation>(loc);
      newResult->setOperationKind(*addOperator);
      newResult->setArguments(args);
      result = std::move(newResult);
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

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseTerm()
  {
    auto loc = lexer.getTokenPosition();

    TRY(factor, parseFactor());
    loc.end = (*factor)->getLocation().end;

    auto result = std::move(*factor);

    while (current == Token::Product ||
           current == Token::ProductEW ||
           current == Token::Division ||
           current == Token::DivisionEW) {
      TRY(mulOperator, parseMulOperator());
      TRY(rhs, parseFactor());
      loc.end = (*rhs)->getLocation().end;

      std::vector<std::unique_ptr<ast::ASTNode>> args;
      args.push_back(std::move(result));
      args.push_back(std::move(*rhs));

      auto newResult = std::make_unique<Operation>(loc);
      newResult->setOperationKind(*mulOperator);
      newResult->setArguments(args);
      result = std::move(newResult);
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

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseFactor()
  {
    auto loc = lexer.getTokenPosition();

    TRY(primary, parsePrimary());
    loc.end = (*primary)->getLocation().end;

    auto result = std::move(*primary);

    while (current == Token::Pow || current == Token::PowEW) {
      std::vector<std::unique_ptr<ast::ASTNode>> args;
      args.push_back(std::move(result));

      if (accept<Token::Pow>()) {
        TRY(rhs, parsePrimary());
        loc.end = (*rhs)->getLocation().end;
        args.push_back(std::move(*rhs));

        auto newResult = std::make_unique<Operation>(loc);
        newResult->setOperationKind(ast::OperationKind::powerOf);
        newResult->setArguments(args);
        result = std::move(newResult);
        continue;
      }

      EXPECT(Token::PowEW);
      TRY(rhs, parsePrimary());
      loc.end = (*rhs)->getLocation().end;
      args.push_back(std::move(*rhs));

      auto newResult = std::make_unique<Operation>(loc);
      newResult->setOperationKind(ast::OperationKind::powerOfEW);
      newResult->setArguments(args);
      result = std::move(newResult);
    }

    return std::move(result);
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parsePrimary()
  {
    auto loc = lexer.getTokenPosition();

    if (current == Token::Integer) {
      TRY(value, parseIntValue());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    if (current == Token::FloatingPoint) {
      TRY(value, parseFloatValue());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    if (current == Token::String) {
      TRY(value, parseString());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    if (current == Token::True || current == Token::False) {
      TRY(value, parseBoolValue());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    std::unique_ptr<ast::ASTNode> result;

    if (accept<Token::LPar>()) {
      TRY(outputExpressionList, parseOutputExpressionList());
      loc.end = lexer.getTokenPosition().end;
      EXPECT(Token::RPar);

      if (outputExpressionList->size() == 1) {
        (*outputExpressionList)[0]->setLocation(loc);
        return std::move((*outputExpressionList)[0]);
      }

      auto newResult = std::make_unique<Tuple>(loc);
      newResult->setExpressions(*outputExpressionList);
      result = std::move(newResult);

    } else if (accept<Token::LCurly>()) {
      TRY(arrayArguments, parseArrayArguments());
      loc.end = lexer.getTokenPosition().end;
      EXPECT(Token::RCurly);

      auto newResult = std::make_unique<Array>(loc);
      newResult->setValues(*arrayArguments);
      result = std::move(newResult);

    } else if (accept<Token::Der>()) {
      auto callee = std::make_unique<ast::ComponentReference>(loc);

      llvm::SmallVector<std::unique_ptr<ASTNode>, 1> path;
      path.push_back(std::make_unique<ast::ComponentReferenceEntry>(loc));
      path[0]->cast<ast::ComponentReferenceEntry>()->setName("der");

      callee->setPath(path);

      TRY(functionCallArgs, parseFunctionCallArgs());
      loc.end = functionCallArgs->getLocation().end;

      auto newResult = std::make_unique<Call>(loc);

      newResult->setCallee(
          static_cast<std::unique_ptr<ASTNode>>(std::move(callee)));

      newResult->setArguments(functionCallArgs->getValue());
      result = std::move(newResult);

    } else if (current == Token::Identifier) {
      TRY(identifier, parseComponentReference());
      loc.end = (*identifier)->getLocation().end;

      if (current != Token::LPar) {
        return std::move(*identifier);
      }

      TRY(functionCallArgs, parseFunctionCallArgs());
      loc.end = functionCallArgs->getLocation().end;

      auto newResult = std::make_unique<Call>(loc);
      newResult->setCallee(std::move(*identifier));
      newResult->setArguments(functionCallArgs->getValue());
      result = std::move(newResult);
    }

    assert(result != nullptr);

    if (current == Token::LSquare) {
      TRY(arraySubscripts, parseArraySubscripts());
      loc.end = arraySubscripts.value().getLocation().end;

      std::vector<std::unique_ptr<ast::ASTNode>> args;
      args.push_back(std::move(result));

      for (auto& subscript : arraySubscripts->getValue()) {
        args.push_back(std::move(subscript));
      }

      auto newResult = std::make_unique<Operation>(loc);
      newResult->setOperationKind(OperationKind::subscription);
      newResult->setArguments(args);
      result = std::move(newResult);
    }

    return std::move(result);
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseComponentReference()
  {
    auto loc = lexer.getTokenPosition();
    bool globalLookup = accept<Token::Dot>();

    llvm::SmallVector<std::unique_ptr<ASTNode>> path;

    TRY(firstEntry, parseComponentReferenceEntry());
    loc.end = (*firstEntry)->getLocation().end;
    path.push_back(std::move(*firstEntry));

    while (accept<Token::Dot>()) {
      TRY(entry, parseComponentReferenceEntry());
      loc.end = (*entry)->getLocation().end;
      path.push_back(std::move(*entry));
    }

    auto result = std::make_unique<ComponentReference>(loc);

    result->setGlobalLookup(globalLookup);
    result->setPath(path);

    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>>
  Parser::parseComponentReferenceEntry()
  {
    auto loc = lexer.getTokenPosition();

    TRY(name, parseIdentifier());
    loc.end = name->getLocation().end;

    auto result = std::make_unique<ComponentReferenceEntry>(loc);
    result->setName(name->getValue());

    if (current == Token::LSquare) {
      TRY(arraySubscripts, parseArraySubscripts());
      loc.end = arraySubscripts.value().getLocation().end;

      result->setSubscripts(arraySubscripts->getValue());
      result->setLocation(loc);
    }

    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<Parser::ValueWrapper<std::vector<std::unique_ptr<ASTNode>>>> Parser::parseFunctionCallArgs()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::LPar);

    std::vector<std::unique_ptr<ast::ASTNode>> args;

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

  llvm::Optional<std::vector<std::unique_ptr<ASTNode>>> Parser::parseFunctionArguments()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

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

  llvm::Optional<std::vector<std::unique_ptr<ASTNode>>> Parser::parseFunctionArgumentsNonFirst()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    do {
      TRY(argument, parseExpression());
      arguments.push_back(std::move(*argument));
    } while (accept<Token::Comma>());

    return arguments;
  }

  llvm::Optional<std::vector<std::unique_ptr<ASTNode>>> Parser::parseArrayArguments()
  {
    auto loc = lexer.getTokenPosition();
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

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

  llvm::Optional<std::vector<std::unique_ptr<ASTNode>>> Parser::parseArrayArgumentsNonFirst()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    do {
      TRY(argument, parseFunctionArgument());
      arguments.push_back(std::move(*argument));
    } while (accept<Token::Comma>());

    return arguments;
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseFunctionArgument()
  {
    return parseExpression();
  }

  llvm::Optional<std::vector<std::unique_ptr<ASTNode>>> Parser::parseOutputExpressionList()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> expressions;

    while (current != Token::RPar) {
      auto loc = lexer.getTokenPosition();

      if (accept<Token::Comma>()) {
        auto expression = std::make_unique<ComponentReference>(loc);
        expression->setDummy(true);
        expressions.push_back(std::move(expression));
        continue;
      }

      TRY(expression, parseExpression());
      expressions.push_back(std::move(*expression));
      accept<Token::Comma>();
    }

    return expressions;
  }

  llvm::Optional<Parser::ValueWrapper<std::vector<std::unique_ptr<ASTNode>>>> Parser::parseArraySubscripts()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::LSquare);

    std::vector<std::unique_ptr<ast::ASTNode>> subscripts;

    do {
      TRY(subscript, parseSubscript());
      subscripts.push_back(std::move(*subscript));
    } while (accept<Token::Comma>());

    loc.end = lexer.getTokenPosition().end;
    EXPECT(Token::RSquare);

    return ValueWrapper(std::move(loc), std::move(subscripts));
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseSubscript()
  {
    auto loc = lexer.getTokenPosition();

    if (accept<Token::Colon>()) {
      auto result = std::make_unique<Constant>(loc);
      result->setValue(-1);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    TRY(expression, parseExpression());
    return std::move(*expression);
  }

  llvm::Optional<std::unique_ptr<ASTNode>> Parser::parseAnnotation()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::Annotation);
    TRY(classModification, parseClassModification());
    loc.end = (*classModification)->getLocation().end;

    auto result = std::make_unique<Annotation>(loc);
    result->setProperties(std::move(*classModification));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseEquationsBlock()
  {
    auto loc = lexer.getTokenPosition();
    EXPECT(Token::Equation);

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> equations;
    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> forEquations;

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

    auto result = std::make_unique<EquationsBlock>(loc);
    result->setEquations(equations);
    result->setForEquations(forEquations);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  llvm::Optional<std::vector<std::unique_ptr<ast::ASTNode>>> Parser::parseForEquations()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> result;

    EXPECT(Token::For);
    TRY(induction, parseInduction());
    EXPECT(Token::Loop);

    while (current != Token::End) {
      if (current == Token::For) {
        TRY(nestedForEquations, parseForEquations());

        for (auto& forEquation : *nestedForEquations) {
          std::vector<std::unique_ptr<ast::ASTNode>> newInductions;
          newInductions.push_back((*induction)->clone());

          size_t numOfInductions =
              forEquation->dyn_cast<ForEquation>()->getNumOfInductions();

          for (size_t i = 0; i < numOfInductions; ++i) {
            newInductions.push_back(
                forEquation->dyn_cast<ForEquation>()->getInduction(i)->clone());
          }

          auto newForEquation =
              std::make_unique<ForEquation>(forEquation->getLocation());

          newForEquation->setInductions(newInductions);

          newForEquation->setEquation(
              forEquation->dyn_cast<ForEquation>()->getEquation()->clone());

          result.push_back(std::move(newForEquation));
        }
      } else {
        TRY(equation, parseEquation());

        auto newForEquation =
            std::make_unique<ForEquation>((*equation)->getLocation());

        newForEquation->setInductions((*induction)->clone());
        newForEquation->setEquation(std::move(*equation));

        EXPECT(Token::Semicolon);

        result.push_back(std::move(newForEquation));
      }
    }

    EXPECT(Token::End);
    EXPECT(Token::For);
    EXPECT(Token::Semicolon);

    return result;
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseInduction()
  {
    auto loc = lexer.getTokenPosition();

    auto variableName = lexer.getIdentifier();
    EXPECT(Token::Identifier);

    EXPECT(Token::In);

    TRY(firstExpression, parseLogicalExpression());
    EXPECT(Token::Colon);
    TRY(secondExpression, parseLogicalExpression());

    auto result = std::make_unique<Induction>(loc);
    result->setName(variableName);
    result->setBegin(std::move(*firstExpression));

    if (accept<Token::Colon>()) {
      TRY(thirdExpression, parseLogicalExpression());
      loc.end = (*thirdExpression)->getLocation().end;

      result->setEnd(std::move(*thirdExpression));
      result->setStep(std::move(*secondExpression));
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    loc.end = (*secondExpression)->getLocation().end;

    result->setEnd(std::move(*secondExpression));

    auto oneStep = std::make_unique<Constant>(loc);
    oneStep->setValue(1);
    result->setStep(std::move(oneStep));
    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  llvm::Optional<std::vector<std::unique_ptr<ASTNode>>> Parser::parseElementList(bool publicSection)
  {
    std::vector<std::unique_ptr<ast::ASTNode>> members;

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

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseElement(bool publicSection)
  {
    accept<Token::Final>();
    TRY(typePrefix, parseTypePrefix());
    TRY(type, parseVariableType());
    TRY(name, parseIdentifier());

    std::unique_ptr<ast::ASTNode> modification;

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

    auto result = std::make_unique<Member>(name->getLocation());
    result->setName(name->getValue());
    result->setType(std::move(*type));
    result->setTypePrefix(std::move(*typePrefix));
    result->setPublic(publicSection);

    if (modification) {
      result->setModification(std::move(modification));
    }

    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseTypePrefix()
  {
    auto loc = lexer.getTokenPosition();

    VariabilityQualifier variabilityQualifier = VariabilityQualifier::none;

    if (accept<Token::Discrete>()) {
      variabilityQualifier = VariabilityQualifier::discrete;
    } else if (accept<Token::Parameter>()) {
      variabilityQualifier = VariabilityQualifier::parameter;
    } else if (accept<Token::Constant>()) {
      variabilityQualifier = VariabilityQualifier::constant;
    }

    IOQualifier ioQualifier = IOQualifier::none;

    if (accept<Token::Input>()) {
      ioQualifier = IOQualifier::input;
    } else if (accept<Token::Output>()) {
      ioQualifier = IOQualifier::output;
    }

    auto result = std::make_unique<ast::TypePrefix>(loc);
    result->setVariabilityQualifier(variabilityQualifier);
    result->setIOQualifier(ioQualifier);
    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseVariableType()
  {
    auto loc = lexer.getTokenPosition();
    std::unique_ptr<ast::ASTNode> result;

    bool globalLookup = accept<Token::Dot>();
    std::string name = lexer.getIdentifier();
    EXPECT(Token::Identifier);

    if (name == "String") {
      result = std::make_unique<BuiltInType>(loc);
      result->cast<BuiltInType>()->setBuiltInTypeKind(
          ast::BuiltInType::Kind::String);
    } else if (name == "Boolean") {
      result = std::make_unique<BuiltInType>(loc);
      result->cast<BuiltInType>()->setBuiltInTypeKind(
          ast::BuiltInType::Kind::Boolean);
    } else if (name == "Integer") {
      result = std::make_unique<BuiltInType>(loc);
      result->cast<BuiltInType>()->setBuiltInTypeKind(
          ast::BuiltInType::Kind::Integer);
    } else if (name == "Real") {
      result = std::make_unique<BuiltInType>(loc);
      result->cast<BuiltInType>()->setBuiltInTypeKind(
          ast::BuiltInType::Kind::Real);
    } else {
      llvm::SmallVector<std::string> path;
      path.push_back(name);

      while (accept<Token::Dot>()) {
        loc.end = lexer.getTokenPosition().end;
        path.push_back(lexer.getIdentifier());
        EXPECT(Token::Identifier);
      }

      result = std::make_unique<UserDefinedType>(loc);
      result->cast<UserDefinedType>()->setGlobalLookup(globalLookup);
      result->cast<UserDefinedType>()->setPath(path);
    }

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> dimensions;

    if (accept<Token::LSquare>()) {
      do {
        TRY(arrayDimension, parseArrayDimension());
        dimensions.push_back(std::move(*arrayDimension));
      } while (accept<Token::Comma>());

      EXPECT(Token::RSquare);
    }

    result->dyn_cast<VariableType>()->setDimensions(dimensions);
    return std::move(result);
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseArrayDimension()
  {
    auto loc = lexer.getTokenPosition();
    auto result = std::make_unique<ArrayDimension>(loc);

    if (accept<Token::Colon>()) {
      result->setSize(-1);
    } else if (accept<Token::Integer>()) {
      result->setSize(lexer.getInt());
    } else {
      TRY(expression, parseExpression());
      result->setSize(std::move(*expression));
    }

    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  llvm::Optional<std::unique_ptr<ast::ASTNode>> Parser::parseTermModification()
  {
    EXPECT(Token::LPar);

    std::unique_ptr<ast::ASTNode> expression;

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
