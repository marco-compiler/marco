#include "marco/Parser/BaseModelica/Parser.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::ast::bmodelica;
using namespace ::marco::parser::bmodelica;

#define EXPECT(Token)                                                          \
  if (!accept<Token>()) {                                                      \
    emitUnexpectedTokenError(lookahead[0], Token);                             \
    return std::nullopt;                                                       \
  }                                                                            \
  static_assert(true)

#define TRY(outVar, expression)                                                \
  auto outVar = expression;                                                    \
  if (!outVar.has_value()) {                                                   \
    return std::nullopt;                                                       \
  }                                                                            \
  static_assert(true)

namespace marco::parser::bmodelica {
Parser::Parser(clang::DiagnosticsEngine &diagnosticsEngine,
               clang::SourceManager &sourceManager,
               std::shared_ptr<SourceFile> source)
    : diagnosticsEngine(&diagnosticsEngine), sourceManager(&sourceManager),
      lexer(std::move(source)) {
  for (size_t i = 0, e = lookahead.size(); i < e; ++i) {
    advance();
  }
}

void Parser::advance() {
  token = lookahead[0];

  for (size_t i = 0, e = lookahead.size(); i + 1 < e; ++i) {
    lookahead[i] = lookahead[i + 1];
  }

  lookahead.back() = lexer.scan();
}

SourceRange Parser::getLocation() const { return token.getLocation(); }

SourceRange Parser::getCursorLocation() const {
  return {getLocation().end, getLocation().end};
}

std::string Parser::getString() const { return token.getString(); }

int64_t Parser::getInt() const { return token.getInt(); }

double Parser::getFloat() const { return token.getFloat(); }

clang::SourceLocation
Parser::convertLocation(const SourceRange &location) const {
  auto &fileManager = sourceManager->getFileManager();

  auto fileRef =
      fileManager.getFileRef(location.begin.file->getFileName(), true);

  if (!fileRef) {
    return clang::SourceLocation();
  }

  return sourceManager->translateFileLineCol(*fileRef, location.begin.line,
                                             location.begin.column);
}

void Parser::emitUnexpectedTokenError(const Token &found, TokenKind expected) {
  auto &diags = *diagnosticsEngine;
  auto location = convertLocation(found.getLocation());

  auto diagID =
      diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                            "Unexpected token: found '%0' instead of '%1'");

  diags.Report(location, diagID) << toString(found) << toString(expected);
}

void Parser::emitUnexpectedIdentifierError(const SourceRange &location,
                                           llvm::StringRef found,
                                           llvm::StringRef expected) {
  auto &diags = *diagnosticsEngine;
  auto convertedLocation = convertLocation(location);

  auto diagID = diags.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "Unexpected identifier: found '%0' instead of '%1'");

  diags.Report(convertedLocation, diagID) << found << expected;
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseRoot() {
  llvm::SmallVector<std::unique_ptr<ASTNode>, 1> classes;

  while (!lookahead[0].isa<TokenKind::EndOfFile>()) {
    TRY(classDefinition, parseClassDefinition());
    classes.push_back(std::move(*classDefinition));
  }

  auto root = std::make_unique<Root>(SourceRange::unknown());
  root->setInnerClasses(classes);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(root));
}

WrappedParseResult<bool> Parser::parseBoolValue() {
  if (accept<TokenKind::True>()) {
    return ValueWrapper(getLocation(), true);
  }

  EXPECT(TokenKind::False);
  return ValueWrapper(getLocation(), false);
}

WrappedParseResult<int64_t> Parser::parseIntValue() {
  if (!accept<TokenKind::Integer>()) {
    return std::nullopt;
  }

  return ValueWrapper(getLocation(), getInt());
}

WrappedParseResult<double> Parser::parseFloatValue() {
  if (!accept<TokenKind::FloatingPoint>()) {
    return std::nullopt;
  }

  return ValueWrapper(getLocation(), getFloat());
}

WrappedParseResult<std::string> Parser::parseString() {
  if (!accept<TokenKind::String>()) {
    return std::nullopt;
  }

  return ValueWrapper(getLocation(), getString());
}

WrappedParseResult<std::string> Parser::parseIdentifier() {
  EXPECT(TokenKind::Identifier);
  return ValueWrapper(getLocation(), getString());
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseClassDefinition() {
  SourceRange loc = lookahead[0].getLocation();
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

  if (auto *standardFunction = result->dyn_cast<StandardFunction>()) {
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
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> equationSections;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> algorithms;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> innerClasses;

  // Whether the first elements list is allowed to be encountered or not.
  // In fact, the class definition allows a first elements list definition
  // and then others more if preceded by "public" or "protected", but no
  // more "lone" definitions are allowed if any of those keywords are
  // encountered.

  bool firstElementListParsable = true;

  while (!lookahead[0].isa<TokenKind::End>() &&
         !lookahead[0].isa<TokenKind::Annotation>() &&
         !lookahead[0].isa<TokenKind::External>()) {
    if (lookahead[0].isa<TokenKind::Initial>() &&
        lookahead[1].isa<TokenKind::Equation>()) {
      TRY(section, parseEquationSection());
      equationSections.push_back(std::move(*section));
      continue;
    }

    if (lookahead[0].isa<TokenKind::Equation>()) {
      TRY(section, parseEquationSection());
      equationSections.push_back(std::move(*section));
      continue;
    }

    if (lookahead[0].isa<TokenKind::Initial>() &&
        lookahead[1].isa<TokenKind::Algorithm>()) {
      TRY(algorithm, parseAlgorithmSection());
      algorithms.push_back(std::move(*algorithm));
      continue;
    }

    if (lookahead[0].isa<TokenKind::Algorithm>()) {
      TRY(algorithm, parseAlgorithmSection());
      algorithms.emplace_back(std::move(*algorithm));
      continue;
    }

    if (lookahead[0].isa<TokenKind::Class>() ||
        lookahead[0].isa<TokenKind::Function>() ||
        lookahead[0].isa<TokenKind::Model>() ||
        lookahead[0].isa<TokenKind::Package>() ||
        lookahead[0].isa<TokenKind::Record>()) {
      TRY(innerClass, parseClassDefinition());
      innerClasses.emplace_back(std::move(*innerClass));
      continue;
    }

    if (accept<TokenKind::Public>()) {
      TRY(elementList, parseElementList(true));

      for (auto &element : *elementList) {
        members.push_back(std::move(element));
      }

      firstElementListParsable = false;
      continue;
    }

    if (accept<TokenKind::Protected>()) {
      TRY(elementList, parseElementList(false));

      for (auto &element : *elementList) {
        members.push_back(std::move(element));
      }

      firstElementListParsable = false;
      continue;
    }

    if (firstElementListParsable) {
      TRY(elementList, parseElementList(true));

      for (auto &element : *elementList) {
        members.push_back(std::move(element));
      }
    }
  }

  // Parse the optional 'external'-related information.
  if (lookahead[0].isa<TokenKind::External>()) {
    EXPECT(TokenKind::External);
    result->cast<Class>()->setExternal(true);

    if (lookahead[0].isa<TokenKind::String>()) {
      TRY(language, parseString());

      result->cast<Class>()->setExternalLanguage(
          std::move(language->getValue()));
    }

    if (lookahead[0].isa<TokenKind::Dot>() ||
        lookahead[0].isa<TokenKind::Identifier>()) {
      TRY(externalFunctionCall, parseExternalFunctionCall());

      result->cast<Class>()->setExternalFunctionCall(
          std::move(*externalFunctionCall));
    }

    if (!lookahead[0].isa<TokenKind::Semicolon>()) {
      TRY(externalFunctionAnnotation, parseAnnotation());

      result->cast<Class>()->setExternalAnnotation(
          std::move(*externalFunctionAnnotation));
    }

    EXPECT(TokenKind::Semicolon);
  }

  // Parse an optional annotation.
  if (lookahead[0].isa<TokenKind::Annotation>()) {
    TRY(annotation, parseAnnotation());
    EXPECT(TokenKind::Semicolon);
    result->dyn_cast<Class>()->setAnnotation(std::move(*annotation));
  }

  // The class name must be present also after the 'end' keyword
  EXPECT(TokenKind::End);
  TRY(endName, parseIdentifier());

  if (name->getValue() != endName->getValue()) {
    emitUnexpectedIdentifierError(endName->getLocation(), endName->getValue(),
                                  name->getValue());

    return std::nullopt;
  }

  EXPECT(TokenKind::Semicolon);

  result->dyn_cast<Class>()->setName(name->getValue());
  result->dyn_cast<Class>()->setVariables(members);
  result->dyn_cast<Class>()->setEquationSections(equationSections);
  result->dyn_cast<Class>()->setAlgorithms(algorithms);
  result->dyn_cast<Class>()->setInnerClasses(innerClasses);

  return std::move(result);
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseExternalFunctionCall() {
  SourceRange loc = getLocation();

  auto result = std::make_unique<ExternalFunctionCall>(loc);

  // Optional destination.
  bool hasDestination = !(lookahead[0].isa<TokenKind::Identifier>() &&
                          lookahead[1].isa<TokenKind::LPar>());

  if (hasDestination) {
    TRY(destination, parseComponentReference());
    loc = (*destination)->getLocation();
    result->setDestination(std::move(*destination));
    EXPECT(TokenKind::EqualityOperator);
  }

  // Callee.
  TRY(callee, parseIdentifier());

  if (!result->hasDestination()) {
    loc = callee->getLocation();
  }

  result->setCallee(callee->getValue());

  EXPECT(TokenKind::LPar);

  // Function arguments.
  llvm::SmallVector<std::unique_ptr<ASTNode>> args;

  while (!lookahead[0].isa<TokenKind::RPar>()) {
    TRY(arg, parseExpression());
    args.push_back(std::move(*arg));

    if (!lookahead[0].isa<TokenKind::RPar>()) {
      EXPECT(TokenKind::Comma);
    }
  }

  result->setArguments(std::move(args));
  EXPECT(TokenKind::RPar);

  loc.end = getLocation().end;
  result->setLocation(loc);

  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseModification() {
  if (accept<TokenKind::EqualityOperator>() ||
      accept<TokenKind::AssignmentOperator>()) {
    SourceRange loc = getLocation();
    TRY(expression, parseExpression());
    loc.end = (*expression)->getLocation().end;

    auto result = std::make_unique<Modification>(loc);
    result->setExpression(std::move(*expression));
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  TRY(classModification, parseClassModification());
  SourceRange loc = (*classModification)->getLocation();

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

ParseResult<std::unique_ptr<ASTNode>> Parser::parseClassModification() {
  EXPECT(TokenKind::LPar);
  SourceRange loc = getLocation();
  TRY(argumentList, parseArgumentList());
  EXPECT(TokenKind::RPar);
  loc.end = getLocation().end;

  auto result = std::make_unique<ClassModification>(loc);
  result->setArguments(**argumentList);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseArgumentList() {
  std::vector<std::unique_ptr<ASTNode>> arguments;

  do {
    TRY(arg, parseArgument());
    arguments.push_back(std::move(*arg));
  } while (accept<TokenKind::Comma>());

  SourceRange loc = arguments.front()->getLocation();
  loc.end = arguments.back()->getLocation().end;

  return ValueWrapper(std::move(loc), std::move(arguments));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseArgument() {
  if (lookahead[0].isa<TokenKind::Redeclare>()) {
    TRY(elementRedeclaration, parseElementRedeclaration());
    return std::move(*elementRedeclaration);
  }

  const bool each = accept<TokenKind::Each>();
  const bool final = accept<TokenKind::Final>();

  if (lookahead[0].isa<TokenKind::Replaceable>()) {
    TRY(elementReplaceable, parseElementReplaceable(each, final));
    return std::move(*elementReplaceable);
  }

  TRY(elementModification, parseElementModification(each, final));
  return std::move(*elementModification);
}

ParseResult<std::unique_ptr<ASTNode>>
Parser::parseElementModification(bool each, bool final) {
  EXPECT(TokenKind::Identifier);
  const std::string name = getString();

  SourceRange loc = getLocation();

  auto result = std::make_unique<ElementModification>(loc);
  result->setName(name);
  result->setEachProperty(each);
  result->setFinalProperty(final);

  if (!lookahead[0].isa<TokenKind::LPar>() &&
      !lookahead[0].isa<TokenKind::EqualityOperator>() &&
      !lookahead[0].isa<TokenKind::AssignmentOperator>()) {
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  TRY(modification, parseModification());
  loc.end = (*modification)->getLocation().end;
  result->setLocation(loc);
  result->setModification(std::move(*modification));
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseElementRedeclaration() {
  llvm_unreachable("Not implemented");
  return std::nullopt;
}

ParseResult<std::unique_ptr<ASTNode>>
Parser::parseElementReplaceable(bool each, bool final) {
  llvm_unreachable("Not implemented");
  return std::nullopt;
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseAlgorithmSection() {
  EXPECT(TokenKind::Algorithm);
  SourceRange loc = getLocation();

  llvm::SmallVector<std::unique_ptr<ASTNode>, 10> statements;

  while (!lookahead[0].isa<TokenKind::End>() &&
         !lookahead[0].isa<TokenKind::Public>() &&
         !lookahead[0].isa<TokenKind::Protected>() &&
         !lookahead[0].isa<TokenKind::Equation>() &&
         !lookahead[0].isa<TokenKind::Algorithm>() &&
         !lookahead[0].isa<TokenKind::External>() &&
         !lookahead[0].isa<TokenKind::Annotation>() &&
         !lookahead[0].isa<TokenKind::EndOfFile>()) {
    TRY(statement, parseStatement());
    EXPECT(TokenKind::Semicolon);
    loc.end = getLocation().end;
    statements.push_back(std::move(*statement));
  }

  auto result = std::make_unique<Algorithm>(loc);
  result->setStatements(statements);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseEquationSection() {
  const SourceRange loc = lookahead[0].getLocation();
  const bool initial = accept<TokenKind::Initial>();
  EXPECT(TokenKind::Equation);

  llvm::SmallVector<std::unique_ptr<ASTNode>> equations;

  while (!lookahead[0].isa<TokenKind::Public>() &&
         !lookahead[0].isa<TokenKind::Protected>() &&
         !lookahead[0].isa<TokenKind::Initial>() &&
         !lookahead[0].isa<TokenKind::Equation>() &&
         !lookahead[0].isa<TokenKind::Algorithm>() &&
         !lookahead[0].isa<TokenKind::External>() &&
         !lookahead[0].isa<TokenKind::Annotation>() &&
         !lookahead[0].isa<TokenKind::End>()) {
    TRY(equation, parseEquation());
    EXPECT(TokenKind::Semicolon);
    equations.push_back(std::move(*equation));
  }

  auto result = std::make_unique<EquationSection>(loc);
  result->setInitial(initial);
  result->setEquations(equations);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseEquation() {
  if (lookahead[0].isa<TokenKind::If>()) {
    TRY(ifEquation, parseIfEquation());
    return std::move(*ifEquation);
  }

  if (lookahead[0].isa<TokenKind::For>()) {
    TRY(forEquation, parseForEquation());
    return std::move(*forEquation);
  }

  if (lookahead[0].isa<TokenKind::When>()) {
    TRY(whenEquation, parseWhenEquation());
    return std::move(*whenEquation);
  }

  TRY(lhs, parseExpression());
  EXPECT(TokenKind::EqualityOperator);
  TRY(rhs, parseExpression());
  accept<TokenKind::String>();

  SourceRange loc = (*lhs)->getLocation();
  loc.end = getLocation().end;

  auto result = std::make_unique<EqualityEquation>(loc);
  result->setLhsExpression(std::move(*lhs));
  result->setRhsExpression(std::move(*rhs));
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseStatement() {
  if (lookahead[0].isa<TokenKind::If>()) {
    TRY(statement, parseIfStatement());
    return std::move(*statement);
  }

  if (lookahead[0].isa<TokenKind::For>()) {
    TRY(statement, parseForStatement());
    return std::move(*statement);
  }

  if (lookahead[0].isa<TokenKind::While>()) {
    TRY(statement, parseWhileStatement());
    return std::move(*statement);
  }

  if (lookahead[0].isa<TokenKind::When>()) {
    TRY(statement, parseWhenStatement());
    return std::move(*statement);
  }

  if (lookahead[0].isa<TokenKind::Break>()) {
    EXPECT(TokenKind::Break);
    auto result = std::make_unique<BreakStatement>(getLocation());
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  if (lookahead[0].isa<TokenKind::Return>()) {
    EXPECT(TokenKind::Return);
    auto result = std::make_unique<ReturnStatement>(getLocation());
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  // Assignment statement.
  if (accept<TokenKind::LPar>()) {
    SourceRange loc = getLocation();
    TRY(destinations, parseOutputExpressionList());
    loc.end = destinations->getLocation().end;

    auto destinationsTuple = std::make_unique<Tuple>(loc);
    destinationsTuple->setExpressions(**destinations);

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

  if (accept<TokenKind::LPar>()) {
    SourceRange loc = (*destination)->getLocation();
    TRY(functionArguments, parseFunctionArguments());
    EXPECT(TokenKind::RPar);

    auto callee = std::make_unique<ComponentReference>(loc);
    loc.end = getLocation().end;
    auto result = std::make_unique<CallStatement>(loc);

    {
      auto call = std::make_unique<Call>(loc);
      call->setCallee(std::move(*destination));
      call->setArguments(functionArguments->getValue());
      result->setCall(std::move(call));
    }

    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  EXPECT(TokenKind::AssignmentOperator);
  TRY(expression, parseExpression());
  SourceRange loc = (*destination)->getLocation();
  loc.end = (*expression)->getLocation().end;

  auto result = std::make_unique<AssignmentStatement>(loc);
  result->setDestinations(std::move(*destination));
  result->setExpression(std::move(*expression));
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseIfEquation() {
  EXPECT(TokenKind::If);
  SourceRange loc = getLocation();
  TRY(ifCondition, parseExpression());
  EXPECT(TokenKind::Then);

  llvm::SmallVector<std::unique_ptr<ASTNode>> ifEquations;

  while (!lookahead[0].isa<TokenKind::ElseIf>() &&
         !lookahead[0].isa<TokenKind::Else>() &&
         !lookahead[0].isa<TokenKind::End>()) {
    TRY(equation, parseEquation());
    EXPECT(TokenKind::Semicolon);
    ifEquations.push_back(std::move(*equation));
  }

  llvm::SmallVector<std::unique_ptr<ASTNode>> elseIfConditions;

  llvm::SmallVector<llvm::SmallVector<std::unique_ptr<ASTNode>>>
      elseIfEquations;

  while (accept<TokenKind::ElseIf>()) {
    TRY(elseIfCondition, parseExpression());
    EXPECT(TokenKind::Then);
    elseIfConditions.push_back(std::move(*elseIfCondition));

    llvm::SmallVector<std::unique_ptr<ASTNode>> currentEquations;

    while (!lookahead[0].isa<TokenKind::ElseIf>() &&
           !lookahead[0].isa<TokenKind::Else>() &&
           !lookahead[0].isa<TokenKind::End>()) {
      TRY(equation, parseEquation());
      EXPECT(TokenKind::Semicolon);
      currentEquations.push_back(std::move(*equation));
    }

    elseIfEquations.push_back(std::move(currentEquations));
  }

  llvm::SmallVector<std::unique_ptr<ASTNode>> elseEquations;

  if (accept<TokenKind::Else>()) {
    while (!lookahead[0].isa<TokenKind::End>()) {
      TRY(equation, parseEquation());
      EXPECT(TokenKind::Semicolon);
      elseEquations.push_back(std::move(*equation));
    }
  }

  EXPECT(TokenKind::End);
  EXPECT(TokenKind::If);
  loc.end = getLocation().end;

  auto result = std::make_unique<IfEquation>(loc);

  result->setIfCondition(std::move(*ifCondition));
  result->setIfEquations(ifEquations);

  assert(elseIfConditions.size() == elseIfEquations.size());
  result->setElseIfConditions(elseIfConditions);

  for (size_t i = 0, e = elseIfEquations.size(); i < e; ++i) {
    result->setElseIfEquations(i, elseIfEquations[i]);
  }

  result->setElseEquations(elseEquations);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseIfStatement() {
  EXPECT(TokenKind::If);
  SourceRange loc = getLocation();

  TRY(ifCondition, parseExpression());
  EXPECT(TokenKind::Then);

  auto statementsBlockLoc = lookahead[0].getLocation();
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> ifStatements;

  while (!lookahead[0].isa<TokenKind::ElseIf>() &&
         !lookahead[0].isa<TokenKind::Else>() &&
         !lookahead[0].isa<TokenKind::End>()) {
    TRY(statement, parseStatement());
    EXPECT(TokenKind::Semicolon);
    auto &stmnt = ifStatements.emplace_back(std::move(*statement));
    statementsBlockLoc.end = stmnt->getLocation().end;
  }

  auto ifBlock = std::make_unique<StatementsBlock>(statementsBlockLoc);
  ifBlock->setBody(ifStatements);
  loc.end = ifBlock->getLocation().end;

  auto result = std::make_unique<IfStatement>(loc);
  result->setIfCondition(std::move(*ifCondition));
  result->setIfBlock(std::move(ifBlock));

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseIfConditions;
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseIfBlocks;

  while (!lookahead[0].isa<TokenKind::Else>() &&
         !lookahead[0].isa<TokenKind::End>()) {
    EXPECT(TokenKind::ElseIf);
    TRY(elseIfCondition, parseExpression());
    elseIfConditions.push_back(std::move(*elseIfCondition));
    EXPECT(TokenKind::Then);
    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseIfStatements;
    auto elseIfStatementsBlockLoc = lookahead[0].getLocation();

    while (!lookahead[0].isa<TokenKind::ElseIf>() &&
           !lookahead[0].isa<TokenKind::Else>() &&
           !lookahead[0].isa<TokenKind::End>()) {
      TRY(statement, parseStatement());
      EXPECT(TokenKind::Semicolon);
      auto &stmnt = elseIfStatements.emplace_back(std::move(*statement));
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
    auto elseBlockLoc = lookahead[0].getLocation();
    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> elseStatements;

    while (!lookahead[0].isa<TokenKind::End>()) {
      TRY(statement, parseStatement());
      EXPECT(TokenKind::Semicolon);
      auto &stmnt = elseStatements.emplace_back(std::move(*statement));
      elseBlockLoc.end = stmnt->getLocation().end;
    }

    auto elseBlock = std::make_unique<StatementsBlock>(elseBlockLoc);
    elseBlock->setBody(elseStatements);
    loc.end = elseBlockLoc.end;
    result->setElseBlock(std::move(elseBlock));
  }

  EXPECT(TokenKind::End);
  EXPECT(TokenKind::If);
  loc.end = getLocation().end;

  result->setLocation(loc);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseForEquation() {
  EXPECT(TokenKind::For);
  SourceRange loc = getLocation();
  TRY(forIndices, parseForIndices());
  EXPECT(TokenKind::Loop);

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> equations;

  while (!lookahead[0].isa<TokenKind::End>()) {
    TRY(equation, parseEquation());
    equations.push_back(std::move(*equation));
    EXPECT(TokenKind::Semicolon);
  }

  EXPECT(TokenKind::End);
  EXPECT(TokenKind::For);
  loc.end = getLocation().end;

  auto result = std::make_unique<ForEquation>(loc);
  result->setForIndices(forIndices->getValue());
  result->setEquations(equations);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseForStatement() {
  EXPECT(TokenKind::For);
  SourceRange loc = getLocation();
  TRY(forIndices, parseForIndices());
  EXPECT(TokenKind::Loop);

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> statements;

  while (!lookahead[0].isa<TokenKind::End>()) {
    TRY(statement, parseStatement());
    EXPECT(TokenKind::Semicolon);
    statements.push_back(std::move(*statement));
  }

  EXPECT(TokenKind::End);
  EXPECT(TokenKind::For);
  loc.end = getLocation().end;

  auto result = std::make_unique<ForStatement>(loc);
  result->setForIndices(forIndices->getValue());
  result->setStatements(statements);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseForIndices() {
  std::vector<std::unique_ptr<ASTNode>> result;
  SourceRange loc = getLocation();

  do {
    TRY(firstIndex, parseForIndex());
    result.push_back(std::move(*firstIndex));
  } while (accept<TokenKind::Comma>());

  loc.begin = result.front()->getLocation().begin;
  loc.end = result.back()->getLocation().end;
  return ValueWrapper(loc, std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseForIndex() {
  EXPECT(TokenKind::Identifier);
  SourceRange loc = getLocation();

  auto result = std::make_unique<ForIndex>(loc);
  result->setName(getString());

  if (accept<TokenKind::In>()) {
    TRY(expression, parseExpression());
    loc.end = (*expression)->getLocation().end;
    result->setExpression(std::move(*expression));
  }

  result->setLocation(loc);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseWhileStatement() {
  EXPECT(TokenKind::While);
  SourceRange loc = getLocation();

  TRY(condition, parseExpression());
  EXPECT(TokenKind::Loop);

  std::vector<std::unique_ptr<ASTNode>> statements;

  while (!lookahead[0].isa<TokenKind::End>()) {
    TRY(statement, parseStatement());
    EXPECT(TokenKind::Semicolon);
    statements.push_back(std::move(*statement));
  }

  EXPECT(TokenKind::End);
  EXPECT(TokenKind::While);
  loc.end = getLocation().end;

  auto result = std::make_unique<WhileStatement>(loc);
  result->setCondition(std::move(*condition));
  result->setStatements(statements);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseWhenEquation() {
  EXPECT(TokenKind::When);
  SourceRange loc = getLocation();
  TRY(whenCondition, parseExpression());
  EXPECT(TokenKind::Then);

  llvm::SmallVector<std::unique_ptr<ASTNode>> whenEquations;

  while (!lookahead[0].isa<TokenKind::ElseWhen>() &&
         !lookahead[0].isa<TokenKind::Else>() &&
         !lookahead[0].isa<TokenKind::End>()) {
    TRY(equation, parseEquation());
    EXPECT(TokenKind::Semicolon);
    whenEquations.push_back(std::move(*equation));
  }

  llvm::SmallVector<std::unique_ptr<ASTNode>> elseWhenConditions;

  llvm::SmallVector<llvm::SmallVector<std::unique_ptr<ASTNode>>>
      elseWhenEquations;

  while (accept<TokenKind::ElseWhen>()) {
    TRY(elseWhenCondition, parseExpression());
    EXPECT(TokenKind::Then);
    elseWhenConditions.push_back(std::move(*elseWhenCondition));

    llvm::SmallVector<std::unique_ptr<ASTNode>> currentEquations;

    while (!lookahead[0].isa<TokenKind::ElseWhen>() &&
           !lookahead[0].isa<TokenKind::Else>() &&
           !lookahead[0].isa<TokenKind::End>()) {
      TRY(equation, parseEquation());
      EXPECT(TokenKind::Semicolon);
      currentEquations.push_back(std::move(*equation));
    }

    elseWhenEquations.push_back(std::move(currentEquations));
  }

  llvm::SmallVector<std::unique_ptr<ASTNode>> elseEquations;

  if (accept<TokenKind::Else>()) {
    while (!lookahead[0].isa<TokenKind::End>()) {
      TRY(equation, parseEquation());
      EXPECT(TokenKind::Semicolon);
      elseEquations.push_back(std::move(*equation));
    }
  }

  EXPECT(TokenKind::End);
  EXPECT(TokenKind::If);
  loc.end = getLocation().end;

  auto result = std::make_unique<WhenEquation>(loc);

  result->setWhenCondition(std::move(*whenCondition));
  result->setWhenEquations(whenEquations);

  assert(elseWhenConditions.size() == elseWhenEquations.size());
  result->setElseWhenConditions(elseWhenConditions);

  for (size_t i = 0, e = elseWhenEquations.size(); i < e; ++i) {
    result->setElseWhenEquations(i, elseWhenEquations[i]);
  }

  result->setElseEquations(elseEquations);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseWhenStatement() {
  EXPECT(TokenKind::When);
  SourceRange loc = getLocation();

  TRY(condition, parseExpression());
  EXPECT(TokenKind::Loop);

  std::vector<std::unique_ptr<ASTNode>> statements;

  while (!lookahead[0].isa<TokenKind::End>()) {
    TRY(statement, parseStatement());
    EXPECT(TokenKind::Semicolon);
    statements.push_back(std::move(*statement));
  }

  EXPECT(TokenKind::End);
  EXPECT(TokenKind::When);
  loc.end = getLocation().end;

  auto result = std::make_unique<WhenStatement>(loc);
  result->setCondition(std::move(*condition));
  result->setStatements(statements);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseExpression() {
  if (accept<TokenKind::If>()) {
    SourceRange loc = getLocation();

    TRY(condition, parseExpression());
    EXPECT(TokenKind::Then);
    TRY(trueExpression, parseExpression());
    EXPECT(TokenKind::Else);
    TRY(falseExpression, parseExpression());

    loc.end = (*falseExpression)->getLocation().end;

    llvm::SmallVector<std::unique_ptr<ASTNode>, 3> args;
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

ParseResult<std::unique_ptr<ASTNode>> Parser::parseSimpleExpression() {
  TRY(l1, parseLogicalExpression());
  SourceRange loc = (*l1)->getLocation();

  if (accept<TokenKind::Colon>()) {
    std::vector<std::unique_ptr<ASTNode>> arguments;
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

ParseResult<std::unique_ptr<ASTNode>> Parser::parseLogicalExpression() {
  std::vector<std::unique_ptr<ASTNode>> logicalTerms;
  TRY(logicalTerm, parseLogicalTerm());

  if (!lookahead[0].isa<TokenKind::Or>()) {
    return std::move(*logicalTerm);
  }

  logicalTerms.push_back(std::move(*logicalTerm));

  while (accept<TokenKind::Or>()) {
    TRY(additionalLogicalTerm, parseLogicalTerm());
    logicalTerms.push_back(std::move(*additionalLogicalTerm));
  }

  SourceRange loc = logicalTerms.front()->getLocation();
  loc.end = logicalTerms.back()->getLocation().end;

  auto result = std::make_unique<Operation>(loc);
  result->setOperationKind(OperationKind::lor);
  result->setArguments(logicalTerms);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseLogicalTerm() {
  std::vector<std::unique_ptr<ASTNode>> logicalFactors;
  TRY(logicalFactor, parseLogicalFactor());

  if (!lookahead[0].isa<TokenKind::And>()) {
    return std::move(*logicalFactor);
  }

  logicalFactors.push_back(std::move(*logicalFactor));

  while (accept<TokenKind::And>()) {
    TRY(additionalLogicalFactor, parseLogicalFactor());
    logicalFactors.emplace_back(std::move(*additionalLogicalFactor));
  }

  SourceRange loc = logicalFactors.front()->getLocation();
  loc.end = logicalFactors.back()->getLocation().end;

  auto result = std::make_unique<Operation>(loc);
  result->setOperationKind(OperationKind::land);
  result->setArguments(logicalFactors);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseLogicalFactor() {
  const bool negated = accept<TokenKind::Not>();
  SourceRange loc = getLocation();

  TRY(relation, parseRelation());

  if (negated) {
    loc.end = (*relation)->getLocation().end;

    auto result = std::make_unique<Operation>(loc);
    result->setOperationKind(OperationKind::lnot);
    result->setArguments({std::move(*relation)});
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  return std::move(*relation);
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseRelation() {
  TRY(lhs, parseArithmeticExpression());

  if (!lookahead[0].isa<TokenKind::GreaterEqual>() &&
      !lookahead[0].isa<TokenKind::Greater>() &&
      !lookahead[0].isa<TokenKind::LessEqual>() &&
      !lookahead[0].isa<TokenKind::Less>() &&
      !lookahead[0].isa<TokenKind::Equal>() &&
      !lookahead[0].isa<TokenKind::NotEqual>()) {
    return std::move(*lhs);
  }

  TRY(op, parseRelationalOperator());
  TRY(rhs, parseArithmeticExpression());

  SourceRange loc = (*lhs)->getLocation();
  loc.end = (*rhs)->getLocation().end;

  auto result = std::make_unique<Operation>(loc);
  result->setOperationKind(**op);
  result->setArguments({std::move(*lhs), std::move(*rhs)});
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

WrappedParseResult<OperationKind> Parser::parseRelationalOperator() {
  if (accept<TokenKind::GreaterEqual>()) {
    return ValueWrapper(getLocation(), OperationKind::greaterEqual);
  }

  if (accept<TokenKind::Greater>()) {
    return ValueWrapper(getLocation(), OperationKind::greater);
  }

  if (accept<TokenKind::LessEqual>()) {
    return ValueWrapper(getLocation(), OperationKind::lessEqual);
  }

  if (accept<TokenKind::Less>()) {
    return ValueWrapper(getLocation(), OperationKind::less);
  }

  if (accept<TokenKind::Equal>()) {
    return ValueWrapper(getLocation(), OperationKind::equal);
  }

  EXPECT(TokenKind::NotEqual);
  return ValueWrapper(getLocation(), OperationKind::different);
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseArithmeticExpression() {
  SourceRange loc = lookahead[0].getLocation();
  bool negative = false;

  if (accept<TokenKind::Minus>()) {
    negative = true;
  } else {
    accept<TokenKind::Plus>();
  }

  TRY(term, parseTerm());
  loc.end = (*term)->getLocation().end;

  std::unique_ptr<ASTNode> result = std::move(*term);

  if (negative) {
    auto newResult = std::make_unique<Operation>(loc);
    newResult->setOperationKind(OperationKind::negate);
    newResult->setArguments({std::move(result)});
    result = std::move(newResult);
  }

  while (lookahead[0].isa<TokenKind::Plus>() ||
         lookahead[0].isa<TokenKind::PlusEW>() ||
         lookahead[0].isa<TokenKind::Minus>() ||
         lookahead[0].isa<TokenKind::MinusEW>()) {
    TRY(addOperator, parseAddOperator());
    TRY(rhs, parseTerm());
    loc.end = (*rhs)->getLocation().end;

    std::vector<std::unique_ptr<ASTNode>> args;
    args.push_back(std::move(result));
    args.push_back(std::move(*rhs));

    auto newResult = std::make_unique<Operation>(loc);
    newResult->setOperationKind(**addOperator);
    newResult->setArguments(args);
    result = std::move(newResult);
  }

  return std::move(result);
}

WrappedParseResult<OperationKind> Parser::parseAddOperator() {
  if (accept<TokenKind::Plus>()) {
    return ValueWrapper(getLocation(), OperationKind::add);
  }

  if (accept<TokenKind::PlusEW>()) {
    return ValueWrapper(getLocation(), OperationKind::addEW);
  }

  if (accept<TokenKind::Minus>()) {
    return ValueWrapper(getLocation(), OperationKind::subtract);
  }

  EXPECT(TokenKind::MinusEW);
  return ValueWrapper(getLocation(), OperationKind::subtractEW);
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseTerm() {
  SourceRange loc = lookahead[0].getLocation();

  TRY(factor, parseFactor());
  loc.end = (*factor)->getLocation().end;

  auto result = std::move(*factor);

  while (lookahead[0].isa<TokenKind::Product>() ||
         lookahead[0].isa<TokenKind::ProductEW>() ||
         lookahead[0].isa<TokenKind::Division>() ||
         lookahead[0].isa<TokenKind::DivisionEW>()) {
    TRY(mulOperator, parseMulOperator());
    TRY(rhs, parseFactor());
    loc.end = (*rhs)->getLocation().end;

    std::vector<std::unique_ptr<ASTNode>> args;
    args.push_back(std::move(result));
    args.push_back(std::move(*rhs));

    auto newResult = std::make_unique<Operation>(loc);
    newResult->setOperationKind(**mulOperator);
    newResult->setArguments(args);
    result = std::move(newResult);
  }

  return std::move(result);
}

WrappedParseResult<OperationKind> Parser::parseMulOperator() {
  if (accept<TokenKind::Product>()) {
    return ValueWrapper(getLocation(), OperationKind::multiply);
  }

  if (accept<TokenKind::ProductEW>()) {
    return ValueWrapper(getLocation(), OperationKind::multiplyEW);
  }

  if (accept<TokenKind::Division>()) {
    return ValueWrapper(getLocation(), OperationKind::divide);
  }

  EXPECT(TokenKind::DivisionEW);
  return ValueWrapper(getLocation(), OperationKind::divideEW);
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseFactor() {
  TRY(primary, parsePrimary());
  SourceRange loc = getLocation();
  loc.end = (*primary)->getLocation().end;

  auto result = std::move(*primary);

  while (lookahead[0].isa<TokenKind::Pow>() ||
         lookahead[0].isa<TokenKind::PowEW>()) {
    std::vector<std::unique_ptr<ASTNode>> args;
    args.push_back(std::move(result));

    if (accept<TokenKind::Pow>()) {
      TRY(rhs, parsePrimary());
      loc.end = (*rhs)->getLocation().end;
      args.push_back(std::move(*rhs));

      auto newResult = std::make_unique<Operation>(loc);
      newResult->setOperationKind(OperationKind::powerOf);
      newResult->setArguments(args);
      result = std::move(newResult);
      continue;
    }

    EXPECT(TokenKind::PowEW);
    TRY(rhs, parsePrimary());
    loc.end = (*rhs)->getLocation().end;
    args.push_back(std::move(*rhs));

    auto newResult = std::make_unique<Operation>(loc);
    newResult->setOperationKind(OperationKind::powerOfEW);
    newResult->setArguments(args);
    result = std::move(newResult);
  }

  return std::move(result);
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parsePrimary() {
  SourceRange loc = lookahead[0].getLocation();

  if (lookahead[0].isa<TokenKind::Integer>()) {
    TRY(value, parseIntValue());
    auto result = std::make_unique<Constant>(value->getLocation());
    result->setValue(value->getValue());
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  if (lookahead[0].isa<TokenKind::FloatingPoint>()) {
    TRY(value, parseFloatValue());
    auto result = std::make_unique<Constant>(value->getLocation());
    result->setValue(value->getValue());
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  if (lookahead[0].isa<TokenKind::String>()) {
    TRY(value, parseString());
    auto result = std::make_unique<Constant>(value->getLocation());
    result->setValue(value->getValue());
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  if (lookahead[0].isa<TokenKind::True>() ||
      lookahead[0].isa<TokenKind::False>()) {
    TRY(value, parseBoolValue());
    auto result = std::make_unique<Constant>(value->getLocation());
    result->setValue(value->getValue());
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  std::unique_ptr<ASTNode> result;

  if (accept<TokenKind::LPar>()) {
    TRY(outputExpressionList, parseOutputExpressionList());
    loc.end = lookahead[0].getLocation().end;
    EXPECT(TokenKind::RPar);

    if ((**outputExpressionList).size() == 1) {
      (**outputExpressionList)[0]->setLocation(loc);
      result = std::move((**outputExpressionList)[0]);
    } else {
      auto newResult = std::make_unique<Tuple>(loc);
      newResult->setExpressions(**outputExpressionList);
      result = std::move(newResult);
    }

  } else if (accept<TokenKind::LCurly>()) {
    TRY(arrayArguments, parseArrayArguments());
    loc.end = lookahead[0].getLocation().end;
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
    auto callee = std::make_unique<ComponentReference>(loc);

    llvm::SmallVector<std::unique_ptr<ASTNode>, 1> path;
    path.push_back(std::make_unique<ComponentReferenceEntry>(loc));
    path[0]->cast<ComponentReferenceEntry>()->setName("der");

    callee->setPath(path);

    TRY(functionCallArgs, parseFunctionCallArgs());
    loc.end = functionCallArgs->getLocation().end;

    auto newResult = std::make_unique<Call>(loc);

    newResult->setCallee(
        static_cast<std::unique_ptr<ASTNode>>(std::move(callee)));

    newResult->setArguments(functionCallArgs->getValue());
    result = std::move(newResult);

  } else if (lookahead[0].isa<TokenKind::Identifier>()) {
    TRY(identifier, parseComponentReference());
    loc.end = (*identifier)->getLocation().end;

    if (!lookahead[0].isa<TokenKind::LPar>()) {
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

  if (lookahead[0].isa<TokenKind::LSquare>()) {
    TRY(arraySubscripts, parseArraySubscripts());
    loc.end = arraySubscripts.value().getLocation().end;

    std::vector<std::unique_ptr<ASTNode>> args;
    args.push_back(std::move(result));

    for (auto &subscript : arraySubscripts->getValue()) {
      args.push_back(std::move(subscript));
    }

    auto newResult = std::make_unique<Operation>(loc);
    newResult->setOperationKind(OperationKind::subscription);
    newResult->setArguments(args);
    result = std::move(newResult);
  }

  return std::move(result);
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseComponentReference() {
  SourceRange loc = lookahead[0].getLocation();
  const bool globalLookup = accept<TokenKind::Dot>();

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

ParseResult<std::unique_ptr<ASTNode>> Parser::parseComponentReferenceEntry() {
  TRY(name, parseIdentifier());
  SourceRange loc = getLocation();
  loc.end = name->getLocation().end;

  auto result = std::make_unique<ComponentReferenceEntry>(loc);
  result->setName(name->getValue());

  if (lookahead[0].isa<TokenKind::LSquare>()) {
    TRY(arraySubscripts, parseArraySubscripts());
    loc.end = arraySubscripts.value().getLocation().end;

    result->setSubscripts(arraySubscripts->getValue());
    result->setLocation(loc);
  }

  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseFunctionCallArgs() {
  SourceRange loc = lookahead[0].getLocation();
  EXPECT(TokenKind::LPar);

  std::vector<std::unique_ptr<ASTNode>> args;

  if (!lookahead[0].isa<TokenKind::RPar>()) {
    TRY(functionArguments, parseFunctionArguments());

    for (auto &arg : **functionArguments) {
      args.push_back(std::move(std::move(arg)));
    }
  }

  EXPECT(TokenKind::RPar);
  loc.end = getLocation().end;

  return ValueWrapper(std::move(loc), std::move(args));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseFunctionArguments() {
  std::vector<std::unique_ptr<ASTNode>> arguments;

  if (lookahead[0].isa<TokenKind::Identifier>() &&
      lookahead[1].isa<TokenKind::EqualityOperator>()) {
    // Named arguments.
    return parseNamedArguments();
  }

  TRY(firstArgExpression, parseExpression());

  if (lookahead[0].isa<TokenKind::For>()) {
    // Reduction argument.
    EXPECT(TokenKind::For);
    TRY(forIndices, parseForIndices());

    auto reductionArg = std::make_unique<ReductionFunctionArgument>(
        (*firstArgExpression)->getLocation());

    SourceRange loc = reductionArg->getLocation();
    loc.end = (*forIndices).getLocation().end;

    reductionArg->setExpression(std::move(*firstArgExpression));
    reductionArg->setForIndices(forIndices->getValue());
    arguments.push_back(std::move(reductionArg));

    return ValueWrapper(std::move(loc), std::move(arguments));
  }

  auto firstArg = std::make_unique<ExpressionFunctionArgument>(
      (*firstArgExpression)->getLocation());

  firstArg->setExpression(std::move(*firstArgExpression));
  arguments.push_back(std::move(firstArg));

  if (accept<TokenKind::Comma>()) {
    TRY(nonFirstArgs, parseFunctionArgumentsNonFirst());

    for (auto &otherArg : **nonFirstArgs) {
      arguments.push_back(std::move(otherArg));
    }
  }

  SourceRange loc = arguments.front()->getLocation();
  loc.end = arguments.back()->getLocation().end;

  return ValueWrapper(std::move(loc), std::move(arguments));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseFunctionArgumentsNonFirst() {
  if (lookahead[0].isa<TokenKind::Identifier>() &&
      lookahead[1].isa<TokenKind::EqualityOperator>()) {
    // Named arguments.
    return parseNamedArguments();
  }

  std::vector<std::unique_ptr<ASTNode>> arguments;

  TRY(firstArg, parseFunctionArgument());
  arguments.push_back(std::move(*firstArg));

  while (accept<TokenKind::Comma>()) {
    TRY(remainingArgs, parseFunctionArgumentsNonFirst());

    for (auto &remainingArg : **remainingArgs) {
      arguments.push_back(std::move(remainingArg));
    }
  }

  SourceRange loc = arguments.front()->getLocation();
  loc.end = arguments.back()->getLocation().end;

  return ValueWrapper(std::move(loc), std::move(arguments));
}

ParseResult<std::pair<std::vector<std::unique_ptr<ASTNode>>,
                      std::vector<std::unique_ptr<ASTNode>>>>
Parser::parseArrayArguments() {
  SourceRange loc = lookahead[0].getLocation();
  std::vector<std::unique_ptr<ASTNode>> arguments;
  std::vector<std::unique_ptr<ASTNode>> forIndices;

  TRY(argument, parseExpression());
  loc.end = (*argument)->getLocation().end;
  arguments.push_back(std::move(*argument));

  if (accept<TokenKind::Comma>()) {
    TRY(otherArguments, parseArrayArgumentsNonFirst());

    for (auto &otherArgument : **otherArguments) {
      loc.end = otherArgument->getLocation().end;
      arguments.push_back(std::move(otherArgument));
    }
  } else if (accept<TokenKind::For>()) {
    TRY(indices, parseForIndices());

    for (auto &forIndex : indices->getValue()) {
      forIndices.push_back(std::move(forIndex));
    }
  }

  return std::make_pair(std::move(arguments), std::move(forIndices));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseArrayArgumentsNonFirst() {
  std::vector<std::unique_ptr<ASTNode>> arguments;

  do {
    TRY(argument, parseExpression());
    arguments.push_back(std::move(*argument));
  } while (accept<TokenKind::Comma>());

  SourceRange loc = arguments.front()->getLocation();
  loc.end = arguments.back()->getLocation().end;

  return ValueWrapper(std::move(loc), std::move(arguments));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseNamedArguments() {
  std::vector<std::unique_ptr<ASTNode>> arguments;

  do {
    TRY(argument, parseNamedArgument());
    arguments.push_back(std::move(*argument));
  } while (accept<TokenKind::Comma>());

  SourceRange loc = arguments.front()->getLocation();
  loc.end = arguments.back()->getLocation().end;

  return ValueWrapper(std::move(loc), std::move(arguments));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseNamedArgument() {
  EXPECT(TokenKind::Identifier);
  SourceRange loc = getLocation();
  const std::string identifier = getString();
  EXPECT(TokenKind::EqualityOperator);

  TRY(expression, parseFunctionArgument());
  loc.end = (*expression)->getLocation().end;

  auto result = std::make_unique<NamedFunctionArgument>(loc);
  result->setName(identifier);
  result->setValue(std::move(*expression));

  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseFunctionArgument() {
  TRY(expression, parseExpression());

  auto result = std::make_unique<ExpressionFunctionArgument>(
      (*expression)->getLocation());

  result->setExpression(std::move(*expression));
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseOutputExpressionList() {
  std::vector<std::unique_ptr<ASTNode>> expressions;

  while (!lookahead[0].isa<TokenKind::RPar>()) {
    const SourceRange loc = lookahead[0].getLocation();

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

  SourceRange loc = getCursorLocation();

  if (!expressions.empty()) {
    loc = expressions.front()->getLocation();
    loc.end = expressions.back()->getLocation().end;
  }

  return ValueWrapper(std::move(loc), std::move(expressions));
}

WrappedParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseArraySubscripts() {
  EXPECT(TokenKind::LSquare);
  SourceRange loc = getLocation();

  std::vector<std::unique_ptr<ASTNode>> subscripts;

  do {
    TRY(subscript, parseSubscript());
    subscripts.push_back(std::move(*subscript));
  } while (accept<TokenKind::Comma>());

  EXPECT(TokenKind::RSquare);
  loc.end = getLocation().end;

  return ValueWrapper(std::move(loc), std::move(subscripts));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseSubscript() {
  if (accept<TokenKind::Colon>()) {
    auto result = std::make_unique<Subscript>(getLocation());
    return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
  }

  TRY(expression, parseExpression());

  auto result = std::make_unique<Subscript>((*expression)->getLocation());
  result->setExpression(std::move(*expression));

  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseAnnotation() {
  EXPECT(TokenKind::Annotation);
  SourceRange loc = getLocation();

  TRY(classModification, parseClassModification());
  loc.end = (*classModification)->getLocation().end;

  auto result = std::make_unique<Annotation>(loc);
  result->setProperties(std::move(*classModification));
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::vector<std::unique_ptr<ASTNode>>>
Parser::parseElementList(bool publicSection) {
  std::vector<std::unique_ptr<ASTNode>> members;

  while (!lookahead[0].isa<TokenKind::Public>() &&
         !lookahead[0].isa<TokenKind::Protected>() &&
         !lookahead[0].isa<TokenKind::Function>() &&
         !lookahead[0].isa<TokenKind::Equation>() &&
         !lookahead[0].isa<TokenKind::Initial>() &&
         !lookahead[0].isa<TokenKind::Algorithm>() &&
         !lookahead[0].isa<TokenKind::End>() &&
         !lookahead[0].isa<TokenKind::Class>() &&
         !lookahead[0].isa<TokenKind::Function>() &&
         !lookahead[0].isa<TokenKind::Model>() &&
         !lookahead[0].isa<TokenKind::Package>() &&
         !lookahead[0].isa<TokenKind::Record>() &&
         !lookahead[0].isa<TokenKind::External>()) {
    TRY(member, parseElement(publicSection));
    EXPECT(TokenKind::Semicolon);
    members.push_back(std::move(*member));
  }

  return members;
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseElement(bool publicSection) {
  accept<TokenKind::Final>();
  TRY(typePrefix, parseTypePrefix());
  TRY(type, parseVariableType());
  TRY(name, parseIdentifier());

  std::unique_ptr<ASTNode> modification;

  if (lookahead[0].isa<TokenKind::LPar>() ||
      lookahead[0].isa<TokenKind::EqualityOperator>()) {
    TRY(mod, parseModification());
    modification = std::move(*mod);
  }

  // String comment. Ignore it for now.
  accept<TokenKind::String>();

  // Annotation
  if (lookahead[0].isa<TokenKind::Annotation>()) {
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

  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseTypePrefix() {
  SourceRange loc = lookahead[0].getLocation();
  VariabilityQualifier variabilityQualifier = VariabilityQualifier::none;

  if (accept<TokenKind::Discrete>()) {
    variabilityQualifier = VariabilityQualifier::discrete;
  } else if (accept<TokenKind::Parameter>()) {
    variabilityQualifier = VariabilityQualifier::parameter;
  } else if (accept<TokenKind::Constant>()) {
    variabilityQualifier = VariabilityQualifier::constant;
  }

  if (variabilityQualifier != VariabilityQualifier::none) {
    loc.end = getLocation().end;
  }

  IOQualifier ioQualifier = IOQualifier::none;

  if (accept<TokenKind::Input>()) {
    ioQualifier = IOQualifier::input;
  } else if (accept<TokenKind::Output>()) {
    ioQualifier = IOQualifier::output;
  }

  if (ioQualifier != IOQualifier::none) {
    loc.end = getLocation().end;
  }

  auto result = std::make_unique<TypePrefix>(loc);
  result->setVariabilityQualifier(variabilityQualifier);
  result->setIOQualifier(ioQualifier);
  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseVariableType() {
  std::unique_ptr<ASTNode> result;

  const bool globalLookup = accept<TokenKind::Dot>();
  const std::string name = lookahead[0].getString();
  EXPECT(TokenKind::Identifier);
  SourceRange loc = getLocation();

  if (name == "String") {
    result = std::make_unique<BuiltInType>(loc);
    result->cast<BuiltInType>()->setBuiltInTypeKind(BuiltInType::Kind::String);
  } else if (name == "Boolean") {
    result = std::make_unique<BuiltInType>(loc);
    result->cast<BuiltInType>()->setBuiltInTypeKind(BuiltInType::Kind::Boolean);
  } else if (name == "Integer") {
    result = std::make_unique<BuiltInType>(loc);
    result->cast<BuiltInType>()->setBuiltInTypeKind(BuiltInType::Kind::Integer);
  } else if (name == "Real") {
    result = std::make_unique<BuiltInType>(loc);
    result->cast<BuiltInType>()->setBuiltInTypeKind(BuiltInType::Kind::Real);
  } else {
    llvm::SmallVector<std::string> path;
    path.push_back(name);

    while (accept<TokenKind::Dot>()) {
      loc.end = lookahead[0].getLocation().end;
      path.push_back(lookahead[0].getString());
      EXPECT(TokenKind::Identifier);
    }

    result = std::make_unique<UserDefinedType>(loc);
    result->cast<UserDefinedType>()->setGlobalLookup(globalLookup);
    result->cast<UserDefinedType>()->setPath(path);
  }

  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> dimensions;

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

ParseResult<std::unique_ptr<ASTNode>> Parser::parseArrayDimension() {
  const SourceRange loc = lookahead[0].getLocation();
  auto result = std::make_unique<ArrayDimension>(loc);

  if (accept<TokenKind::Colon>()) {
    result->setSize(-1);
  } else if (lookahead[0].isa<TokenKind::Integer>()) {
    result->setSize(lookahead[0].getInt());
    EXPECT(TokenKind::Integer);
  } else {
    TRY(expression, parseExpression());
    result->setSize(std::move(*expression));
  }

  return static_cast<std::unique_ptr<ASTNode>>(std::move(result));
}

ParseResult<std::unique_ptr<ASTNode>> Parser::parseTermModification() {
  EXPECT(TokenKind::LPar);

  std::unique_ptr<ASTNode> expression;

  do {
    accept<TokenKind::Each>();
    const std::string lastIndentifier = lookahead[0].getString();
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
} // namespace marco::parser::bmodelica
