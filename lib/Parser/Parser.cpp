#include "marco/Parser/Parser.h"
#include "marco/Parser/Message.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::parser;

#define EXPECT(Token)                                           \
	if (!accept<Token>()) {                                       \
	  diagnostics->emitError<UnexpectedTokenMessage>(             \
        tokens[0].getLocation(), tokens[0].getKind(), Token);   \
    return std::nullopt;                                        \
  }                                                             \
  static_assert(true)

#define TRY(outVar, expression)       \
	auto outVar = expression;           \
	if (!outVar.has_value()) {          \
    return std::nullopt;              \
  }                                   \
  static_assert(true)

namespace marco::parser
{
  Parser::Parser(
      diagnostic::DiagnosticEngine& diagnostics,
      std::shared_ptr<SourceFile> file)
      : diagnostics(&diagnostics),
        lexer(std::move(file))
  {
    for (size_t i = 0, e = tokens.size(); i < e; ++i) {
      advance();
    }
  }

  void Parser::advance()
  {
    for (size_t i = 0, e = tokens.size(); i + 1 < e; ++i) {
      tokens[i] = tokens[i + 1];
    }
    
    tokens.back() = lexer.scan();
  }

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseRoot()
  {
    auto loc = tokens[0].getLocation();
    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 1> classes;

    while (!tokens[0].isa<TokenKind::EndOfFile>()) {
      TRY(classDefinition, parseClassDefinition());
      classes.push_back(std::move(*classDefinition));
    }

    auto root = std::make_unique<Root>(SourceRange::unknown());
    root->setInnerClasses(classes);
    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(root));
  }

  std::optional<Parser::ValueWrapper<bool>> Parser::parseBoolValue()
  {
    auto loc = tokens[0].getLocation();

    if (accept<TokenKind::True>()) {
      return ValueWrapper(loc, true);
    }

    EXPECT(TokenKind::False);
    return ValueWrapper(loc, false);
  }

  std::optional<Parser::ValueWrapper<int64_t>> Parser::parseIntValue()
  {
    auto loc = tokens[0].getLocation();
    auto value = tokens[0].getInt();
    accept<TokenKind::Integer>();
    return ValueWrapper(loc, value);
  }

  std::optional<Parser::ValueWrapper<double>> Parser::parseFloatValue()
  {
    auto loc = tokens[0].getLocation();
    auto value = tokens[0].getFloat();
    accept<TokenKind::FloatingPoint>();
    return ValueWrapper(loc, value);
  }

  std::optional<Parser::ValueWrapper<std::string>> Parser::parseString()
  {
    auto loc = tokens[0].getLocation();
    auto value = tokens[0].getString();
    accept<TokenKind::String>();
    return ValueWrapper(loc, value);
  }

  std::optional<Parser::ValueWrapper<std::string>> Parser::parseIdentifier()
  {
    auto loc = tokens[0].getLocation();

    std::string identifier = tokens[0].getString();
    EXPECT(TokenKind::Identifier);

    return ValueWrapper(loc, std::move(identifier));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseClassDefinition()
  {
    auto loc = tokens[0].getLocation();
    std::unique_ptr<ASTNode> result;

    bool isOperator = accept<TokenKind::Operator>();
    bool isPure = true;

    if (isOperator) {
      if (accept<TokenKind::Record>()) {
        result = std::make_unique<Record>(loc);
      } else {
        EXPECT(TokenKind::Function);
        result = std::make_unique<StandardFunction>(loc);
      }
    } else if (accept<TokenKind::Model>()) {
      result = std::make_unique<Model>(loc);
    } else if (accept<TokenKind::Record>()) {
      result = std::make_unique<Record>(loc);
    } else if (accept<TokenKind::Package>()) {
      result = std::make_unique<Package>(loc);
    } else if (accept<TokenKind::Function>()) {
      result = std::make_unique<StandardFunction>(loc);
    } else if (accept<TokenKind::Pure>()) {
      isPure = true;
      isOperator = accept<TokenKind::Operator>();
      EXPECT(TokenKind::Function);
      result = std::make_unique<StandardFunction>(loc);
    } else if (accept<TokenKind::Impure>()) {
      isPure = false;
      isOperator = accept<TokenKind::Operator>();
      EXPECT(TokenKind::Function);
      result = std::make_unique<StandardFunction>(loc);
    } else {
      EXPECT(TokenKind::Class);
      result = std::make_unique<Model>(loc);
    }

    if (auto standardFunction = result->dyn_cast<StandardFunction>()) {
      standardFunction->setPure(isPure);
    }

    TRY(name, parseIdentifier());
    loc.end = name->getLocation().end;
    result->setLocation(loc);

    if (accept<TokenKind::EqualityOperator>()) {
      // Function derivative
      assert(result->isa<StandardFunction>());
      EXPECT(TokenKind::Der);
      EXPECT(TokenKind::LPar);

      TRY(derivedFunction, parseExpression());
      llvm::SmallVector<std::unique_ptr<ASTNode>, 3> independentVariables;

      while (accept<TokenKind::Comma>()) {
        TRY(var, parseExpression());
        independentVariables.push_back(std::move(*var));
      }

      EXPECT(TokenKind::RPar);
      accept<TokenKind::String>();
      EXPECT(TokenKind::Semicolon);

      auto partialDerFunction = std::make_unique<PartialDerFunction>(loc);
      partialDerFunction->setName(name->getValue());
      partialDerFunction->setDerivedFunction(std::move(*derivedFunction));
      partialDerFunction->setIndependentVariables(independentVariables);
      result = std::move(partialDerFunction);

      return std::move(result);
    }

    accept<TokenKind::String>();

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

    while (!tokens[0].isa<TokenKind::End>() &&
        !tokens[0].isa<TokenKind::Annotation>()) {
      if (tokens[0].isa<TokenKind::Equation>()) {
        TRY(equationsBlock, parseEquationsBlock());
        equationsBlocks.push_back(std::move(*equationsBlock));
        continue;
      }

      if (accept<TokenKind::Initial>()) {
        TRY(equationsBlock, parseEquationsBlock());
        initialEquationsBlocks.push_back(std::move(*equationsBlock));
        continue;
      }

      if (tokens[0].isa<TokenKind::Algorithm>()) {
        TRY(algorithm, parseAlgorithmSection());
        algorithms.emplace_back(std::move(*algorithm));
        continue;
      }

      if (tokens[0].isa<TokenKind::Class>() ||
          tokens[0].isa<TokenKind::Function>() ||
          tokens[0].isa<TokenKind::Model>() ||
          tokens[0].isa<TokenKind::Package>() ||
          tokens[0].isa<TokenKind::Record>()) {
        TRY(innerClass, parseClassDefinition());
        innerClasses.emplace_back(std::move(*innerClass));
        continue;
      }

      if (accept<TokenKind::Public>()) {
        TRY(elementList, parseElementList(true));

        for (auto& element : *elementList) {
          members.push_back(std::move(element));
        }

        firstElementListParsable = false;
        continue;
      }

      if (accept<TokenKind::Protected>()) {
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

    // Parse an optional annotation.
    if (tokens[0].isa<TokenKind::Annotation>()) {
      TRY(annotation, parseAnnotation());
      EXPECT(TokenKind::Semicolon);
      result->dyn_cast<Class>()->setAnnotation(std::move(*annotation));
    }

    // The class name must be present also after the 'end' keyword
    EXPECT(TokenKind::End);
    TRY(endName, parseIdentifier());

    if (name->getValue() != endName->getValue()) {
      diagnostics->emitError<UnexpectedIdentifierMessage>(
          endName->getLocation(), endName->getValue(), name->getValue());

      return std::nullopt;
    }

    EXPECT(TokenKind::Semicolon);

    result->dyn_cast<Class>()->setName(name->getValue());
    result->dyn_cast<Class>()->setVariables(members);
    result->dyn_cast<Class>()->setEquationsBlocks(equationsBlocks);
    result->dyn_cast<Class>()->setInitialEquationsBlocks(initialEquationsBlocks);
    result->dyn_cast<Class>()->setAlgorithms(algorithms);
    result->dyn_cast<Class>()->setInnerClasses(innerClasses);

    return std::move(result);
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseModification()
  {
    auto loc = tokens[0].getLocation();

    if (accept<TokenKind::EqualityOperator>() || accept<TokenKind::AssignmentOperator>()) {
      TRY(expression, parseExpression());
      loc.end = (*expression)->getLocation().end;

      auto result = std::make_unique<Modification>(loc);
      result->setExpression(std::move(*expression));
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    TRY(classModification, parseClassModification());
    loc.end = (*classModification)->getLocation().end;

    if (accept<TokenKind::EqualityOperator>()) {
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

  std::optional<std::unique_ptr<ASTNode>> Parser::parseClassModification()
  {
    auto loc = tokens[0].getLocation();
    EXPECT(TokenKind::LPar);
    TRY(argumentList, parseArgumentList());
    loc.end = tokens[0].getLocation().end;
    EXPECT(TokenKind::RPar);

    auto result = std::make_unique<ClassModification>(loc);
    result->setArguments(*argumentList);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::vector<std::unique_ptr<ASTNode>>>
  Parser::parseArgumentList()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    do {
      TRY(arg, parseArgument());
      arguments.push_back(std::move(*arg));
    } while (accept<TokenKind::Comma>());

    return arguments;
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseArgument()
  {
    auto loc = tokens[0].getLocation();

    if (tokens[0].isa<TokenKind::Redeclare>()) {
      TRY(elementRedeclaration, parseElementRedeclaration());
      return std::move(*elementRedeclaration);
    }

    bool each = accept<TokenKind::Each>();
    bool final = accept<TokenKind::Final>();

    if (tokens[0].isa<TokenKind::Replaceable>()) {
      TRY(elementReplaceable, parseElementReplaceable(each, final));
      return std::move(*elementReplaceable);
    }

    TRY(elementModification, parseElementModification(each, final));
    return std::move(*elementModification);
  }

  std::optional<std::unique_ptr<ASTNode>>
  Parser::parseElementModification(bool each, bool final)
  {
    auto loc = tokens[0].getLocation();

    auto name = tokens[0].getString();
    EXPECT(TokenKind::Identifier);

    auto result = std::make_unique<ElementModification>(loc);
    result->setName(name);
    result->setEachProperty(each);
    result->setFinalProperty(final);

    if (!tokens[0].isa<TokenKind::LPar>() &&
        !tokens[0].isa<TokenKind::EqualityOperator>() &&
        !tokens[0].isa<TokenKind::AssignmentOperator>()) {
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    TRY(modification, parseModification());
    result->setModification(std::move(*modification));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseElementRedeclaration()
  {
    llvm_unreachable("Not implemented");
    return std::nullopt;
  }

  std::optional<std::unique_ptr<ASTNode>>
  Parser::parseElementReplaceable(bool each, bool final)
  {
    llvm_unreachable("Not implemented");
    return std::nullopt;
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseAlgorithmSection()
  {
    auto loc = tokens[0].getLocation();
    EXPECT(TokenKind::Algorithm);

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 10> statements;

    while (!tokens[0].isa<TokenKind::End>() &&
           !tokens[0].isa<TokenKind::Public>() &&
           !tokens[0].isa<TokenKind::Protected>() &&
           !tokens[0].isa<TokenKind::Equation>() &&
           !tokens[0].isa<TokenKind::Algorithm>() &&
           !tokens[0].isa<TokenKind::External>() &&
           !tokens[0].isa<TokenKind::Annotation>() &&
           !tokens[0].isa<TokenKind::EndOfFile>()) {
      TRY(statement, parseStatement());
      loc.end = tokens[0].getLocation().end;
      EXPECT(TokenKind::Semicolon);
      statements.push_back(std::move(*statement));
    }

    auto result = std::make_unique<Algorithm>(loc);
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseEquation()
  {
    auto loc = tokens[0].getLocation();

    TRY(lhs, parseExpression());
    EXPECT(TokenKind::EqualityOperator);
    TRY(rhs, parseExpression());

    loc.end = (*rhs)->getLocation().end;
    accept<TokenKind::String>();

    auto result = std::make_unique<Equation>(loc);
    result->setLhsExpression(std::move(*lhs));
    result->setRhsExpression(std::move(*rhs));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseStatement()
  {
    if (tokens[0].isa<TokenKind::If>()) {
      TRY(statement, parseIfStatement());
      return std::move(*statement);
    }

    if (tokens[0].isa<TokenKind::For>()) {
      TRY(statement, parseForStatement());
      return std::move(*statement);
    }

    if (tokens[0].isa<TokenKind::While>()) {
      TRY(statement, parseWhileStatement());
      return std::move(*statement);
    }

    if (tokens[0].isa<TokenKind::When>()) {
      TRY(statement, parseWhenStatement());
      return std::move(*statement);
    }

    if (tokens[0].isa<TokenKind::Break>()) {
      auto loc = tokens[0].getLocation();
      EXPECT(TokenKind::Break);

      auto result = std::make_unique<BreakStatement>(loc);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    if (tokens[0].isa<TokenKind::Return>()) {
      auto loc = tokens[0].getLocation();
      EXPECT(TokenKind::Return);

      auto result = std::make_unique<ReturnStatement>(loc);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    // Assignment statement
    auto loc = tokens[0].getLocation();

    if (accept<TokenKind::LPar>()) {
      TRY(destinations, parseOutputExpressionList());
      loc.end = tokens[0].getLocation().end;

      auto destinationsTuple = std::make_unique<Tuple>(loc);
      destinationsTuple->setExpressions(destinations.value());

      EXPECT(TokenKind::RPar);
      EXPECT(TokenKind::AssignmentOperator);
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
    EXPECT(TokenKind::AssignmentOperator);
    TRY(expression, parseExpression());
    loc.end = (*expression)->getLocation().end;

    auto result = std::make_unique<AssignmentStatement>(loc);
    result->setDestinations(std::move(*destination));
    result->setExpression(std::move(*expression));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseIfStatement()
  {
    auto loc = tokens[0].getLocation();
    auto result = std::make_unique<IfStatement>(loc);

    EXPECT(TokenKind::If);
    TRY(ifCondition, parseExpression());
    EXPECT(TokenKind::Then);

    auto statementsBlockLoc = tokens[0].getLocation();
    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> ifStatements;

    while (!tokens[0].isa<TokenKind::ElseIf>() &&
           !tokens[0].isa<TokenKind::Else>() &&
           !tokens[0].isa<TokenKind::End>()) {
      TRY(statement, parseStatement());
      EXPECT(TokenKind::Semicolon);
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

    while (!tokens[0].isa<TokenKind::Else>() &&
        !tokens[0].isa<TokenKind::End>()) {
      EXPECT(TokenKind::ElseIf);
      TRY(elseIfCondition, parseExpression());
      elseIfConditions.push_back(std::move(*elseIfCondition));
      EXPECT(TokenKind::Then);
      llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> elseIfStatements;
      auto elseIfStatementsBlockLoc = tokens[0].getLocation();

      while (!tokens[0].isa<TokenKind::ElseIf>() &&
             !tokens[0].isa<TokenKind::Else>() &&
             !tokens[0].isa<TokenKind::End>()) {
        TRY(statement, parseStatement());
        EXPECT(TokenKind::Semicolon);
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

    if (accept<TokenKind::Else>()) {
      auto elseBlockLoc = tokens[0].getLocation();
      llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> elseStatements;

      while (!tokens[0].isa<TokenKind::End>()) {
        TRY(statement, parseStatement());
        EXPECT(TokenKind::Semicolon);
        auto& stmnt = elseStatements.emplace_back(std::move(*statement));
        elseBlockLoc.end = stmnt->getLocation().end;
      }

      auto elseBlock = std::make_unique<StatementsBlock>(elseBlockLoc);
      elseBlock->setBody(elseStatements);
      loc.end = elseBlockLoc.end;
      result->setElseBlock(std::move(elseBlock));
    }

    EXPECT(TokenKind::End);
    loc.end = tokens[0].getLocation().end;
    EXPECT(TokenKind::If);

    result->setLocation(loc);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseForStatement()
  {
    auto loc = tokens[0].getLocation();

    EXPECT(TokenKind::For);
    TRY(induction, parseForIndexOld());
    EXPECT(TokenKind::Loop);

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> statements;

    while (!tokens[0].isa<TokenKind::End>()) {
      TRY(statement, parseStatement());
      EXPECT(TokenKind::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(TokenKind::End);

    loc.end = tokens[0].getLocation().end;
    EXPECT(TokenKind::For);

    auto result = std::make_unique<ForStatement>(loc);
    result->setInduction(std::move(*induction));
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  std::optional<Parser::ValueWrapper<
      std::vector<std::unique_ptr<ast::ASTNode>>>> Parser::parseForIndices()
  {
    auto loc = tokens[0].getLocation();
    std::vector<std::unique_ptr<ast::ASTNode>> result;

    do {
      TRY(firstIndex, parseForIndex());
      result.push_back(std::move(*firstIndex));
    } while (accept<TokenKind::Comma>());

    return ValueWrapper(loc, std::move(result));
  }

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseForIndex()
  {
    auto loc = tokens[0].getLocation();
    auto name = tokens[0].getString();
    EXPECT(TokenKind::Identifier);

    auto result = std::make_unique<ForIndex>(loc);

    if (accept<TokenKind::In>()) {
      TRY(expression, parseExpression());
      loc.end = (*expression)->getLocation().end;
      result->setLocation(loc);
      result->setExpression(std::move(*expression));
    }

    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseWhileStatement()
  {
    auto loc = tokens[0].getLocation();

    EXPECT(TokenKind::While);
    TRY(condition, parseExpression());
    EXPECT(TokenKind::Loop);

    std::vector<std::unique_ptr<ast::ASTNode>> statements;

    while (!tokens[0].isa<TokenKind::End>()) {
      TRY(statement, parseStatement());
      EXPECT(TokenKind::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(TokenKind::End);

    loc.end = tokens[0].getLocation().end;
    EXPECT(TokenKind::While);

    auto result = std::make_unique<WhileStatement>(loc);
    result->setCondition(std::move(*condition));
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseWhenStatement()
  {
    auto loc = tokens[0].getLocation();

    EXPECT(TokenKind::When);
    TRY(condition, parseExpression());
    EXPECT(TokenKind::Loop);

    std::vector<std::unique_ptr<ast::ASTNode>> statements;

    while (!tokens[0].isa<TokenKind::End>()) {
      TRY(statement, parseStatement());
      EXPECT(TokenKind::Semicolon);
      statements.push_back(std::move(*statement));
    }

    EXPECT(TokenKind::End);

    loc.end = tokens[0].getLocation().end;
    EXPECT(TokenKind::When);

    auto result = std::make_unique<WhenStatement>(loc);
    result->setCondition(std::move(*condition));
    result->setStatements(statements);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseExpression()
  {
    auto loc = tokens[0].getLocation();

    if (accept<TokenKind::If>()) {
      TRY(condition, parseExpression());
      EXPECT(TokenKind::Then);
      TRY(trueExpression, parseExpression());
      EXPECT(TokenKind::Else);
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

  std::optional<std::unique_ptr<ASTNode>> Parser::parseSimpleExpression()
  {
    auto loc = tokens[0].getLocation();
    TRY(l1, parseLogicalExpression());
    loc.end = (*l1)->getLocation().end;

    if (accept<TokenKind::Colon>()) {
      std::vector<std::unique_ptr<ast::ASTNode>> arguments;
      TRY(l2, parseLogicalExpression());
      loc.end = (*l2)->getLocation().end;

      arguments.push_back(std::move(*l1));
      arguments.push_back(std::move(*l2));

      if (accept<TokenKind::Colon>()) {
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

  std::optional<std::unique_ptr<ASTNode>> Parser::parseLogicalExpression()
  {
    auto loc = tokens[0].getLocation();

    std::vector<std::unique_ptr<ast::ASTNode>> logicalTerms;
    TRY(logicalTerm, parseLogicalTerm());
    loc.end = (*logicalTerm)->getLocation().end;

    if (!tokens[0].isa<TokenKind::Or>()) {
      return std::move(*logicalTerm);
    }

    logicalTerms.push_back(std::move(*logicalTerm));

    while (accept<TokenKind::Or>()) {
      TRY(additionalLogicalTerm, parseLogicalTerm());
      loc.end = (*additionalLogicalTerm)->getLocation().end;
      logicalTerms.emplace_back(std::move(*additionalLogicalTerm));
    }

    auto result = std::make_unique<Operation>(loc);
    result->setOperationKind(OperationKind::lor);
    result->setArguments(logicalTerms);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseLogicalTerm()
  {
    auto loc = tokens[0].getLocation();

    std::vector<std::unique_ptr<ast::ASTNode>> logicalFactors;
    TRY(logicalFactor, parseLogicalFactor());
    loc.end = (*logicalFactor)->getLocation().end;

    if (!tokens[0].isa<TokenKind::And>()) {
      return std::move(*logicalFactor);
    }

    logicalFactors.push_back(std::move(*logicalFactor));

    while (accept<TokenKind::And>()) {
      TRY(additionalLogicalFactor, parseLogicalFactor());
      loc.end = (*additionalLogicalFactor)->getLocation().end;
      logicalFactors.emplace_back(std::move(*additionalLogicalFactor));
    }

    auto result = std::make_unique<Operation>(loc);
    result->setOperationKind(OperationKind::land);
    result->setArguments(logicalFactors);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseLogicalFactor()
  {
    auto loc = tokens[0].getLocation();
    bool negated = accept<TokenKind::Not>();

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

  std::optional<std::unique_ptr<ASTNode>> Parser::parseRelation()
  {
    auto loc = tokens[0].getLocation();

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

  std::optional<OperationKind> Parser::parseRelationalOperator()
  {
    if (accept<TokenKind::GreaterEqual>()) {
      return OperationKind::greaterEqual;
    }

    if (accept<TokenKind::Greater>()) {
      return OperationKind::greater;
    }

    if (accept<TokenKind::LessEqual>()) {
      return OperationKind::lessEqual;
    }

    if (accept<TokenKind::Less>()) {
      return OperationKind::less;
    }

    if (accept<TokenKind::Equal>()) {
      return OperationKind::equal;
    }

    if (accept<TokenKind::NotEqual>()) {
      return OperationKind::different;
    }

    return std::nullopt;
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseArithmeticExpression()
  {
    auto loc = tokens[0].getLocation();
    bool negative = false;

    if (accept<TokenKind::Minus>()) {
      negative = true;
    } else {
      accept<TokenKind::Plus>();
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

    while (tokens[0].isa<TokenKind::Plus>() ||
           tokens[0].isa<TokenKind::PlusEW>() ||
           tokens[0].isa<TokenKind::Minus>() ||
           tokens[0].isa<TokenKind::MinusEW>()) {
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

  std::optional<OperationKind> Parser::parseAddOperator()
  {
    if (accept<TokenKind::Plus>()) {
      return OperationKind::add;
    }

    if (accept<TokenKind::PlusEW>()) {
      return OperationKind::addEW;
    }

    if (accept<TokenKind::Minus>()) {
      return OperationKind::subtract;
    }

    if (accept<TokenKind::MinusEW>()) {
      return OperationKind::subtractEW;
    }

    return std::nullopt;
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseTerm()
  {
    auto loc = tokens[0].getLocation();

    TRY(factor, parseFactor());
    loc.end = (*factor)->getLocation().end;

    auto result = std::move(*factor);

    while (tokens[0].isa<TokenKind::Product>() ||
           tokens[0].isa<TokenKind::ProductEW>() ||
           tokens[0].isa<TokenKind::Division>() ||
           tokens[0].isa<TokenKind::DivisionEW>()) {
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

  std::optional<OperationKind> Parser::parseMulOperator()
  {
    if (accept<TokenKind::Product>()) {
      return OperationKind::multiply;
    }

    if (accept<TokenKind::ProductEW>()) {
      return OperationKind::multiplyEW;
    }

    if (accept<TokenKind::Division>()) {
      return OperationKind::divide;
    }

    if (accept<TokenKind::DivisionEW>()) {
      return OperationKind::divideEW;
    }

    return std::nullopt;
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseFactor()
  {
    auto loc = tokens[0].getLocation();

    TRY(primary, parsePrimary());
    loc.end = (*primary)->getLocation().end;

    auto result = std::move(*primary);

    while (tokens[0].isa<TokenKind::Pow>() ||
        tokens[0].isa<TokenKind::PowEW>()) {
      std::vector<std::unique_ptr<ast::ASTNode>> args;
      args.push_back(std::move(result));

      if (accept<TokenKind::Pow>()) {
        TRY(rhs, parsePrimary());
        loc.end = (*rhs)->getLocation().end;
        args.push_back(std::move(*rhs));

        auto newResult = std::make_unique<Operation>(loc);
        newResult->setOperationKind(ast::OperationKind::powerOf);
        newResult->setArguments(args);
        result = std::move(newResult);
        continue;
      }

      EXPECT(TokenKind::PowEW);
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

  std::optional<std::unique_ptr<ASTNode>> Parser::parsePrimary()
  {
    auto loc = tokens[0].getLocation();

    if (tokens[0].isa<TokenKind::Integer>()) {
      TRY(value, parseIntValue());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    if (tokens[0].isa<TokenKind::FloatingPoint>()) {
      TRY(value, parseFloatValue());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    if (tokens[0].isa<TokenKind::String>()) {
      TRY(value, parseString());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    if (tokens[0].isa<TokenKind::True>() ||
        tokens[0].isa<TokenKind::False>()) {
      TRY(value, parseBoolValue());
      auto result = std::make_unique<Constant>(value->getLocation());
      result->setValue(value->getValue());
      return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
    }

    std::unique_ptr<ast::ASTNode> result;

    if (accept<TokenKind::LPar>()) {
      TRY(outputExpressionList, parseOutputExpressionList());
      loc.end = tokens[0].getLocation().end;
      EXPECT(TokenKind::RPar);

      if (outputExpressionList->size() == 1) {
        (*outputExpressionList)[0]->setLocation(loc);
        return std::move((*outputExpressionList)[0]);
      }

      auto newResult = std::make_unique<Tuple>(loc);
      newResult->setExpressions(*outputExpressionList);
      result = std::move(newResult);

    } else if (accept<TokenKind::LCurly>()) {
      TRY(arrayArguments, parseArrayArguments());
      loc.end = tokens[0].getLocation().end;
      EXPECT(TokenKind::RCurly);

      auto &expressions = arrayArguments->first;
      auto &forIndices = arrayArguments->second;
      if (forIndices.empty()) {
        auto newResult = std::make_unique<ArrayConstant>(loc);
        newResult->setValues(expressions);
        result = std::move(newResult);
      } else {
        assert(expressions.size() == 1);
        auto newResult = std::make_unique<ArrayForGenerator>(loc);
        newResult->setValue(std::move(expressions[0]));
        newResult->setIndices(forIndices);
        result = std::move(newResult);
      }

    } else if (accept<TokenKind::Der>()) {
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

    } else if (tokens[0].isa<TokenKind::Identifier>()) {
      TRY(identifier, parseComponentReference());
      loc.end = (*identifier)->getLocation().end;

      if (!tokens[0].isa<TokenKind::LPar>()) {
        // Identifier.
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

    if (tokens[0].isa<TokenKind::LSquare>()) {
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

  std::optional<std::unique_ptr<ASTNode>> Parser::parseComponentReference()
  {
    auto loc = tokens[0].getLocation();
    bool globalLookup = accept<TokenKind::Dot>();

    llvm::SmallVector<std::unique_ptr<ASTNode>> path;

    TRY(firstEntry, parseComponentReferenceEntry());
    loc.end = (*firstEntry)->getLocation().end;
    path.push_back(std::move(*firstEntry));

    while (accept<TokenKind::Dot>()) {
      TRY(entry, parseComponentReferenceEntry());
      loc.end = (*entry)->getLocation().end;
      path.push_back(std::move(*entry));
    }

    auto result = std::make_unique<ComponentReference>(loc);

    result->setGlobalLookup(globalLookup);
    result->setPath(path);

    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ast::ASTNode>>
  Parser::parseComponentReferenceEntry()
  {
    auto loc = tokens[0].getLocation();

    TRY(name, parseIdentifier());
    loc.end = name->getLocation().end;

    auto result = std::make_unique<ComponentReferenceEntry>(loc);
    result->setName(name->getValue());

    if (tokens[0].isa<TokenKind::LSquare>()) {
      TRY(arraySubscripts, parseArraySubscripts());
      loc.end = arraySubscripts.value().getLocation().end;

      result->setSubscripts(arraySubscripts->getValue());
      result->setLocation(loc);
    }

    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<Parser::ValueWrapper<std::vector<std::unique_ptr<ASTNode>>>>
  Parser::parseFunctionCallArgs()
  {
    auto loc = tokens[0].getLocation();
    EXPECT(TokenKind::LPar);

    std::vector<std::unique_ptr<ast::ASTNode>> args;

    if (!tokens[0].isa<TokenKind::RPar>()) {
      TRY(functionArguments, parseFunctionArguments());

      for (auto& arg : *functionArguments) {
        args.push_back(std::move(std::move(arg)));
      }
    }

    loc.end = tokens[0].getLocation().end;
    EXPECT(TokenKind::RPar);

    return ValueWrapper(std::move(loc), std::move(args));
  }

  std::optional<std::vector<std::unique_ptr<ASTNode>>>
  Parser::parseFunctionArguments()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    if (tokens[0].isa<TokenKind::Identifier>() &&
        tokens[1].isa<TokenKind::EqualityOperator>()) {
      // Named arguments.
      return parseNamedArguments();
    }

    TRY(firstArgExpression, parseExpression());

    if (tokens[0].isa<TokenKind::For>()) {
      // Reduction argument.
      TRY(forIndices, parseForIndices());

      auto reductionArg = std::make_unique<ReductionFunctionArgument>(
          (*firstArgExpression)->getLocation());

      auto loc = reductionArg->getLocation();
      loc.end = (*forIndices).getLocation().end;

      reductionArg->setForIndices(forIndices->getValue());
      return arguments;
    }

    auto firstArg = std::make_unique<ExpressionFunctionArgument>(
                        (*firstArgExpression)->getLocation());

    firstArg->setExpression(std::move(*firstArgExpression));
    arguments.push_back(std::move(firstArg));

    if (accept<TokenKind::Comma>()) {
      TRY(nonFirstArgs, parseFunctionArgumentsNonFirst());

      for (auto& otherArg : *nonFirstArgs) {
        arguments.push_back(std::move(otherArg));
      }
    }

    return arguments;
  }

  std::optional<std::vector<std::unique_ptr<ASTNode>>>
  Parser::parseFunctionArgumentsNonFirst()
  {
    if (tokens[0].isa<TokenKind::Identifier>() &&
        tokens[1].isa<TokenKind::EqualityOperator>()) {
      // Named arguments.
      return parseNamedArguments();
    }

    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    TRY(firstArg, parseFunctionArgument());
    arguments.push_back(std::move(*firstArg));

    while (accept<TokenKind::Comma>()) {
      TRY(remainingArgs, parseFunctionArgumentsNonFirst());

      for (auto& remainingArg : *remainingArgs) {
        arguments.push_back(std::move(remainingArg));
      }
    }

    return arguments;
  }

  std::optional<std::pair<
      std::vector<std::unique_ptr<ASTNode>>,
      std::vector<std::unique_ptr<ASTNode>>>>
  Parser::parseArrayArguments()
  {
    auto loc = tokens[0].getLocation();
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;
		std::vector<std::unique_ptr<ast::ASTNode>> inductions;

    TRY(argument, parseExpression());
    loc.end = (*argument)->getLocation().end;
    arguments.push_back(std::move(*argument));

    if (accept<TokenKind::Comma>()) {
      TRY(otherArguments, parseArrayArgumentsNonFirst());

      for (auto& otherArgument : *otherArguments) {
        loc.end = otherArgument->getLocation().end;
        arguments.push_back(std::move(otherArgument));
      }
    } else if (accept<TokenKind::For>()) {
      // for-indices
      do {
        TRY(induction, parseForIndexOld());
        inductions.push_back(std::move(*induction));
      } while (accept<TokenKind::Comma>());
    }

    return std::make_pair(std::move(arguments), std::move(inductions));
  }

  std::optional<std::vector<std::unique_ptr<ASTNode>>>
  Parser::parseArrayArgumentsNonFirst()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    do {
      TRY(argument, parseExpression());
      arguments.push_back(std::move(*argument));
    } while (accept<TokenKind::Comma>());

    return arguments;
  }

  std::optional<std::vector<std::unique_ptr<ast::ASTNode>>>
  Parser::parseNamedArguments()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> arguments;

    do {
      TRY(argument, parseNamedArgument());
      arguments.push_back(std::move(*argument));
    } while (accept<TokenKind::Comma>());

    return arguments;
  }

  std::optional<std::unique_ptr<ast::ASTNode>>
  Parser::parseNamedArgument()
  {
    auto loc = tokens[0].getLocation();

    auto identifier = lexer.getIdentifier();
    EXPECT(TokenKind::Identifier);
    EXPECT(TokenKind::EqualityOperator);

    TRY(expression, parseFunctionArgument());
    loc.end = (*expression)->getLocation().end;

    auto result = std::make_unique<NamedFunctionArgument>(loc);
    result->setName(identifier);
    result->setValue(std::move(*expression));

    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ast::ASTNode>>
  Parser::parseFunctionArgument()
  {
    TRY(expression, parseExpression());

    auto result = std::make_unique<ExpressionFunctionArgument>(
        (*expression)->getLocation());

    result->setExpression(std::move(*expression));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::vector<std::unique_ptr<ASTNode>>>
  Parser::parseOutputExpressionList()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> expressions;

    while (!tokens[0].isa<TokenKind::RPar>()) {
      auto loc = tokens[0].getLocation();

      if (accept<TokenKind::Comma>()) {
        auto expression = std::make_unique<ComponentReference>(loc);
        expression->setDummy(true);
        expressions.push_back(std::move(expression));
        continue;
      }

      TRY(expression, parseExpression());
      expressions.push_back(std::move(*expression));
      accept<TokenKind::Comma>();
    }

    return expressions;
  }

  std::optional<Parser::ValueWrapper<std::vector<std::unique_ptr<ASTNode>>>>
  Parser::parseArraySubscripts()
  {
    auto loc = tokens[0].getLocation();
    EXPECT(TokenKind::LSquare);

    std::vector<std::unique_ptr<ast::ASTNode>> subscripts;

    do {
      TRY(subscript, parseSubscript());
      subscripts.push_back(std::move(*subscript));
    } while (accept<TokenKind::Comma>());

    loc.end = tokens[0].getLocation().end;
    EXPECT(TokenKind::RSquare);

    return ValueWrapper(std::move(loc), std::move(subscripts));
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseSubscript()
  {
    auto loc = tokens[0].getLocation();

    if (accept<TokenKind::Colon>()) {
      auto result = std::make_unique<Constant>(loc);
      result->setValue(-1);
      return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
    }

    TRY(expression, parseExpression());
    return std::move(*expression);
  }

  std::optional<std::unique_ptr<ASTNode>> Parser::parseAnnotation()
  {
    auto loc = tokens[0].getLocation();
    EXPECT(TokenKind::Annotation);
    TRY(classModification, parseClassModification());
    loc.end = (*classModification)->getLocation().end;

    auto result = std::make_unique<Annotation>(loc);
    result->setProperties(std::move(*classModification));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseEquationsBlock()
  {
    auto loc = tokens[0].getLocation();
    EXPECT(TokenKind::Equation);

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> equations;
    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> forEquations;

    while (
        !tokens[0].isa<TokenKind::End>() &&
        !tokens[0].isa<TokenKind::Public>() &&
        !tokens[0].isa<TokenKind::Protected>() &&
        !tokens[0].isa<TokenKind::Equation>() &&
        !tokens[0].isa<TokenKind::Initial>() &&
        !tokens[0].isa<TokenKind::Algorithm>() &&
        !tokens[0].isa<TokenKind::External>() &&
        !tokens[0].isa<TokenKind::Annotation>() &&
        !tokens[0].isa<TokenKind::EndOfFile>()) {
      if (tokens[0].isa<TokenKind::For>()) {
        TRY(currentForEquations, parseForEquations());

        for (auto& forEquation : *currentForEquations) {
          forEquations.push_back(std::move(forEquation));
        }
      } else {
        TRY(equation, parseEquation());
        equations.push_back(std::move(*equation));
        EXPECT(TokenKind::Semicolon);
      }
    }

    auto result = std::make_unique<EquationsBlock>(loc);
    result->setEquations(equations);
    result->setForEquations(forEquations);
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::optional<std::vector<std::unique_ptr<ast::ASTNode>>>
  Parser::parseForEquations()
  {
    std::vector<std::unique_ptr<ast::ASTNode>> result;

    EXPECT(TokenKind::For);
    TRY(induction, parseForIndexOld());
    EXPECT(TokenKind::Loop);

    while (!tokens[0].isa<TokenKind::End>()) {
      if (tokens[0].isa<TokenKind::For>()) {
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

        EXPECT(TokenKind::Semicolon);

        result.push_back(std::move(newForEquation));
      }
    }

    EXPECT(TokenKind::End);
    EXPECT(TokenKind::For);
    EXPECT(TokenKind::Semicolon);

    return result;
  }

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseForIndexOld()
  {
    auto loc = tokens[0].getLocation();

    auto variableName = tokens[0].getString();
    EXPECT(TokenKind::Identifier);

    EXPECT(TokenKind::In);

    // Hardcode a simple-expression because any other expression, while legal,
    // does not make any semantic sense
    TRY(firstExpression, parseLogicalExpression());
    EXPECT(TokenKind::Colon);
    TRY(secondExpression, parseLogicalExpression());

    auto result = std::make_unique<Induction>(loc);
    result->setName(variableName);
    result->setBegin(std::move(*firstExpression));

    if (accept<TokenKind::Colon>()) {
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

  std::optional<std::vector<std::unique_ptr<ASTNode>>>
  Parser::parseElementList(bool publicSection)
  {
    std::vector<std::unique_ptr<ast::ASTNode>> members;

    while (!tokens[0].isa<TokenKind::Public>() &&
           !tokens[0].isa<TokenKind::Protected>() &&
           !tokens[0].isa<TokenKind::Function>() &&
           !tokens[0].isa<TokenKind::Equation>() &&
           !tokens[0].isa<TokenKind::Initial>() &&
           !tokens[0].isa<TokenKind::Algorithm>() &&
           !tokens[0].isa<TokenKind::End>() &&
           !tokens[0].isa<TokenKind::Class>() &&
           !tokens[0].isa<TokenKind::Function>() &&
           !tokens[0].isa<TokenKind::Model>() &&
           !tokens[0].isa<TokenKind::Package>() &&
           !tokens[0].isa<TokenKind::Record>()) {
      TRY(member, parseElement(publicSection));
      EXPECT(TokenKind::Semicolon);
      members.push_back(std::move(*member));
    }

    return members;
  }

  std::optional<std::unique_ptr<ast::ASTNode>>
  Parser::parseElement(bool publicSection)
  {
    accept<TokenKind::Final>();
    TRY(typePrefix, parseTypePrefix());
    TRY(type, parseVariableType());
    TRY(name, parseIdentifier());

    std::unique_ptr<ast::ASTNode> modification;

    if (tokens[0].isa<TokenKind::LPar>() ||
        tokens[0].isa<TokenKind::EqualityOperator>()) {
      TRY(mod, parseModification());
      modification = std::move(*mod);
    }

    // String comment. Ignore it for now.
    accept<TokenKind::String>();

    // Annotation
    if (tokens[0].isa<TokenKind::Annotation>()) {
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

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseTypePrefix()
  {
    auto loc = tokens[0].getLocation();

    VariabilityQualifier variabilityQualifier = VariabilityQualifier::none;

    if (accept<TokenKind::Discrete>()) {
      variabilityQualifier = VariabilityQualifier::discrete;
    } else if (accept<TokenKind::Parameter>()) {
      variabilityQualifier = VariabilityQualifier::parameter;
    } else if (accept<TokenKind::Constant>()) {
      variabilityQualifier = VariabilityQualifier::constant;
    }

    IOQualifier ioQualifier = IOQualifier::none;

    if (accept<TokenKind::Input>()) {
      ioQualifier = IOQualifier::input;
    } else if (accept<TokenKind::Output>()) {
      ioQualifier = IOQualifier::output;
    }

    auto result = std::make_unique<ast::TypePrefix>(loc);
    result->setVariabilityQualifier(variabilityQualifier);
    result->setIOQualifier(ioQualifier);
    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseVariableType()
  {
    auto loc = tokens[0].getLocation();
    std::unique_ptr<ast::ASTNode> result;

    bool globalLookup = accept<TokenKind::Dot>();
    std::string name = tokens[0].getString();
    EXPECT(TokenKind::Identifier);

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

      while (accept<TokenKind::Dot>()) {
        loc.end = tokens[0].getLocation().end;
        path.push_back(tokens[0].getString());
        EXPECT(TokenKind::Identifier);
      }

      result = std::make_unique<UserDefinedType>(loc);
      result->cast<UserDefinedType>()->setGlobalLookup(globalLookup);
      result->cast<UserDefinedType>()->setPath(path);
    }

    llvm::SmallVector<std::unique_ptr<ast::ASTNode>, 3> dimensions;

    if (accept<TokenKind::LSquare>()) {
      do {
        TRY(arrayDimension, parseArrayDimension());
        dimensions.push_back(std::move(*arrayDimension));
      } while (accept<TokenKind::Comma>());

      EXPECT(TokenKind::RSquare);
    }

    result->dyn_cast<VariableType>()->setDimensions(dimensions);
    return std::move(result);
  }

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseArrayDimension()
  {
    auto loc = tokens[0].getLocation();
    auto result = std::make_unique<ArrayDimension>(loc);

    if (accept<TokenKind::Colon>()) {
      result->setSize(-1);
    } else if (tokens[0].isa<TokenKind::Integer>()) {
      result->setSize(tokens[0].getInt());
      EXPECT(TokenKind::Integer);
    } else {
      TRY(expression, parseExpression());
      result->setSize(std::move(*expression));
    }

    return static_cast<std::unique_ptr<ast::ASTNode>>(std::move(result));
  }

  std::optional<std::unique_ptr<ast::ASTNode>> Parser::parseTermModification()
  {
    EXPECT(TokenKind::LPar);

    std::unique_ptr<ast::ASTNode> expression;

    do {
      accept<TokenKind::Each>();
      auto lastIndentifier = tokens[0].getString();
      EXPECT(TokenKind::Identifier);
      EXPECT(TokenKind::EqualityOperator);

      if (lastIndentifier == "start") {
        TRY(exp, parseExpression());
        expression = std::move(*exp);
      }

      if (accept<TokenKind::FloatingPoint>()) {
        continue;
      }

      if (accept<TokenKind::Integer>()) {
        continue;
      }

      if (accept<TokenKind::String>()) {
        continue;
      }

      if (accept<TokenKind::True>()) {
        continue;
      }

      if (accept<TokenKind::False>()) {
        continue;
      }
    } while (accept<TokenKind::Comma>());

    EXPECT(TokenKind::RPar);
    return std::move(expression);
  }
}
