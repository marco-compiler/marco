#include "marco/Parser/Modelica/Parser.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::parser::modelica;

namespace marco::parser::modelica {
Parser::Parser(clang::DiagnosticsEngine &diagnosticsEngine,
               clang::SourceManager &sourceManager,
               std::shared_ptr<SourceFile> source)
    : diagnosticsEngine(&diagnosticsEngine), sourceManager(&sourceManager),
      lexer(std::move(source)) {
  for (size_t i = 0, e = lookahead.size(); i < e; ++i) {
    shiftLookahead();
  }
}

bool Parser::isAtEnd() const { return check(TokenKind::EndOfFile); }

//===---------------------------------------------------------------------===//
// Token handling
//===---------------------------------------------------------------------===//

void Parser::shiftLookahead() {
  for (size_t i = 0, e = lookahead.size(); i + 1 < e; ++i) {
    lookahead[i] = lookahead[i + 1];
  }

  Token &last = lookahead.back();

  if (!last.isa<TokenKind::EndOfFile>() && !last.isa<TokenKind::Error>()) {
    last = lexer.scan();
  }
}

const Token &Parser::peek(size_t offset) const {
  assert(offset < lookahead.size() && "Lookahead too far");
  return lookahead[offset];
}

bool Parser::check(TokenKind kind, size_t offset) const {
  return peek(offset).getKind() == kind;
}

Token Parser::advance() {
  Token token = peek();

  if (token.isa<TokenKind::EndOfFile>() || token.isa<TokenKind::Error>()) {
    return token;
  }

  shiftLookahead();
  lastConsumedLocation = token.getLocation();
  return token;
}

bool Parser::accept(TokenKind kind) {
  if (check(kind)) {
    advance();
    return true;
  }

  return false;
}

bool Parser::expect(TokenKind kind) {
  if (accept(kind)) {
    return true;
  }

  emitUnexpectedTokenError("'" + toString(kind).str() + "'");
  return false;
}

SourceRange Parser::rangeFrom(const SourceRange &start) const {
  return {start.begin, lastConsumedLocation.end};
}

//===---------------------------------------------------------------------===//
// Diagnostics
//===---------------------------------------------------------------------===//

clang::SourceLocation
Parser::convertLocation(const SourceRange &location) const {
  if (!location.begin.file) {
    return clang::SourceLocation();
  }

  auto &fileManager = sourceManager->getFileManager();

  auto fileRef =
      fileManager.getFileRef(location.begin.file->getFileName(), true);

  if (!fileRef) {
    return clang::SourceLocation();
  }

  return sourceManager->translateFileLineCol(*fileRef, location.begin.line,
                                             location.begin.column);
}

void Parser::emitError(const SourceRange &location, llvm::StringRef message) {
  auto &diags = *diagnosticsEngine;
  auto diagID = diags.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0");
  diags.Report(convertLocation(location), diagID) << message;
}

void Parser::emitUnexpectedTokenError(llvm::StringRef expected) {
  const Token &token = peek();

  if (token.isa<TokenKind::Error>()) {
    emitError(token.getLocation(), token.getString());
    return;
  }

  emitError(token.getLocation(),
            "expected " + expected.str() + ", but found " + token.describe());
}

//===---------------------------------------------------------------------===//
// Names and descriptions
//===---------------------------------------------------------------------===//

std::optional<std::string> Parser::parseIdentifier() {
  if (!check(TokenKind::Identifier)) {
    emitUnexpectedTokenError("identifier");
    return std::nullopt;
  }

  return advance().getString();
}

std::optional<Name> Parser::parseName() {
  SourceRange start = peek().getLocation();
  Name name;

  do {
    auto identifier = parseIdentifier();

    if (!identifier) {
      return std::nullopt;
    }

    name.parts.push_back(*identifier);
  } while (check(TokenKind::Dot) && check(TokenKind::Identifier, 1) &&
           accept(TokenKind::Dot));

  name.location = rangeFrom(start);
  return name;
}

std::optional<Name> Parser::parseTypeSpecifier() {
  SourceRange start = peek().getLocation();
  bool global = accept(TokenKind::Dot);
  auto name = parseName();

  if (!name) {
    return std::nullopt;
  }

  name->global = global;
  name->location = rangeFrom(start);
  return name;
}

std::optional<std::string> Parser::parseDescriptionString() {
  if (!check(TokenKind::String)) {
    return std::string();
  }

  std::string text = advance().getString();

  while (check(TokenKind::Plus) && check(TokenKind::String, 1)) {
    advance();
    text += advance().getString();
  }

  return text;
}

bool Parser::parseDescription(Description &description) {
  auto text = parseDescriptionString();

  if (!text) {
    return false;
  }

  description.text = *text;

  if (check(TokenKind::Annotation)) {
    auto annotation = parseAnnotation();

    if (!annotation) {
      return false;
    }

    description.annotation = std::move(annotation);
  }

  return true;
}

std::unique_ptr<ClassModification> Parser::parseAnnotation() {
  if (!expect(TokenKind::Annotation)) {
    return nullptr;
  }

  return parseClassModification();
}

//===---------------------------------------------------------------------===//
// Stored definition
//===---------------------------------------------------------------------===//

std::unique_ptr<StoredDefinition> Parser::parseStoredDefinition() {
  SourceRange start = peek().getLocation();
  auto definition = makeNode<StoredDefinition>(start);

  if (accept(TokenKind::Within)) {
    Name within;

    if (!check(TokenKind::Semicolon)) {
      auto name = parseName();

      if (!name) {
        return nullptr;
      }

      within = std::move(*name);
    }

    definition->setWithin(std::move(within));

    if (!expect(TokenKind::Semicolon)) {
      return nullptr;
    }
  }

  while (!isAtEnd()) {
    ElementPrefixes prefixes;
    prefixes.isFinal = accept(TokenKind::Final);

    auto classDefinition = parseClassDefinition(prefixes);

    if (!classDefinition || !expect(TokenKind::Semicolon)) {
      return nullptr;
    }

    definition->addClass(std::move(classDefinition));
  }

  finalize(*definition, start);
  definition->setParentOfChildren();
  return definition;
}

//===---------------------------------------------------------------------===//
// Classes
//===---------------------------------------------------------------------===//

std::unique_ptr<ClassDefinition> Parser::parseClassDefinition() {
  auto classDefinition = parseClassDefinition(ElementPrefixes());

  if (classDefinition) {
    classDefinition->setParentOfChildren();
  }

  return classDefinition;
}

bool Parser::isClassDefinitionStart() const {
  switch (peek().getKind()) {
  case TokenKind::Encapsulated:
  case TokenKind::Partial:
  case TokenKind::Class:
  case TokenKind::Model:
  case TokenKind::Operator:
  case TokenKind::Record:
  case TokenKind::Block:
  case TokenKind::Expandable:
  case TokenKind::Connector:
  case TokenKind::Type:
  case TokenKind::Package:
  case TokenKind::Pure:
  case TokenKind::Impure:
  case TokenKind::Function:
    return true;
  default:
    return false;
  }
}

bool Parser::parseClassPrefixes(ClassDefinition &classDefinition) {
  classDefinition.setPartial(accept(TokenKind::Partial));

  if (accept(TokenKind::Class)) {
    classDefinition.setRestriction(ClassRestriction::Class);
  } else if (accept(TokenKind::Model)) {
    classDefinition.setRestriction(ClassRestriction::Model);
  } else if (accept(TokenKind::Record)) {
    classDefinition.setRestriction(ClassRestriction::Record);
  } else if (accept(TokenKind::Block)) {
    classDefinition.setRestriction(ClassRestriction::Block);
  } else if (accept(TokenKind::Expandable)) {
    if (!expect(TokenKind::Connector)) {
      return false;
    }

    classDefinition.setRestriction(ClassRestriction::ExpandableConnector);
  } else if (accept(TokenKind::Connector)) {
    classDefinition.setRestriction(ClassRestriction::Connector);
  } else if (accept(TokenKind::Type)) {
    classDefinition.setRestriction(ClassRestriction::Type);
  } else if (accept(TokenKind::Package)) {
    classDefinition.setRestriction(ClassRestriction::Package);
  } else if (check(TokenKind::Pure) || check(TokenKind::Impure) ||
             check(TokenKind::Operator) || check(TokenKind::Function)) {
    Purity purity = Purity::Unspecified;

    if (accept(TokenKind::Pure)) {
      purity = Purity::Pure;
    } else if (accept(TokenKind::Impure)) {
      purity = Purity::Impure;
    }

    classDefinition.setPurity(purity);

    if (accept(TokenKind::Operator)) {
      if (accept(TokenKind::Function)) {
        classDefinition.setRestriction(ClassRestriction::OperatorFunction);
      } else if (purity == Purity::Unspecified && accept(TokenKind::Record)) {
        classDefinition.setRestriction(ClassRestriction::OperatorRecord);
      } else if (purity == Purity::Unspecified) {
        classDefinition.setRestriction(ClassRestriction::Operator);
      } else {
        emitUnexpectedTokenError("'function'");
        return false;
      }
    } else {
      if (!expect(TokenKind::Function)) {
        return false;
      }

      classDefinition.setRestriction(ClassRestriction::Function);
    }
  } else {
    emitUnexpectedTokenError("class kind");
    return false;
  }

  return true;
}

std::unique_ptr<ClassDefinition>
Parser::parseClassDefinition(const ElementPrefixes &prefixes) {
  SourceRange start = peek().getLocation();
  auto classDefinition = makeNode<ClassDefinition>(start);
  classDefinition->getPrefixes() = prefixes;
  classDefinition->setEncapsulated(accept(TokenKind::Encapsulated));

  if (!parseClassPrefixes(*classDefinition)) {
    return nullptr;
  }

  bool isClassExtends = accept(TokenKind::Extends);
  SourceRange nameLocation = peek().getLocation();
  auto name = parseIdentifier();

  if (!name) {
    return nullptr;
  }

  classDefinition->setName(*name);

  if (!isClassExtends && accept(TokenKind::Equal)) {
    if (accept(TokenKind::Enumeration)) {
      // Enumeration.
      classDefinition->setForm(ClassDefinition::Form::Enumeration);

      if (!expect(TokenKind::LParen)) {
        return nullptr;
      }

      if (accept(TokenKind::Colon)) {
        classDefinition->setUnspecifiedEnumeration(true);
      } else if (!check(TokenKind::RParen)) {
        do {
          SourceRange literalStart = peek().getLocation();
          auto literal = makeNode<EnumerationLiteral>(literalStart);
          auto literalName = parseIdentifier();

          if (!literalName || !parseDescription(literal->getDescription())) {
            return nullptr;
          }

          literal->setName(*literalName);
          finalize(*literal, literalStart);
          classDefinition->addEnumerationLiteral(std::move(literal));
        } while (accept(TokenKind::Comma));
      }

      if (!expect(TokenKind::RParen) ||
          !parseDescription(classDefinition->getDescription())) {
        return nullptr;
      }
    } else if (accept(TokenKind::Der)) {
      // Derivative of a function.
      classDefinition->setForm(ClassDefinition::Form::Derivative);

      if (!expect(TokenKind::LParen)) {
        return nullptr;
      }

      auto function = parseTypeSpecifier();

      if (!function || !expect(TokenKind::Comma)) {
        return nullptr;
      }

      classDefinition->setDerivedFunction(std::move(*function));

      do {
        auto arg = parseIdentifier();

        if (!arg) {
          return nullptr;
        }

        classDefinition->addDerivativeArgument(*arg);
      } while (accept(TokenKind::Comma));

      if (!expect(TokenKind::RParen) ||
          !parseDescription(classDefinition->getDescription())) {
        return nullptr;
      }
    } else {
      classDefinition->setForm(ClassDefinition::Form::Short);

      if (!parseShortClassSpecifier(*classDefinition)) {
        return nullptr;
      }
    }

    finalize(*classDefinition, start);
    return classDefinition;
  }

  // Long class specifier.
  if (isClassExtends) {
    classDefinition->setForm(ClassDefinition::Form::ClassExtends);

    if (check(TokenKind::LParen)) {
      auto modification = parseClassModification();

      if (!modification) {
        return nullptr;
      }

      classDefinition->setClassModification(std::move(modification));
    }
  }

  auto description = parseDescriptionString();

  if (!description) {
    return nullptr;
  }

  classDefinition->getDescription().text = *description;

  if (!parseComposition(*classDefinition) || !expect(TokenKind::End)) {
    return nullptr;
  }

  SourceRange endNameLocation = peek().getLocation();
  auto endName = parseIdentifier();

  if (!endName) {
    return nullptr;
  }

  if (*endName != *name) {
    emitError(endNameLocation,
              "the name '" + *endName +
                  "' doesn't match the name of the class '" + *name + "'");

    emitError(nameLocation, "class declared here");
    return nullptr;
  }

  finalize(*classDefinition, start);
  return classDefinition;
}

bool Parser::parseShortClassSpecifier(ClassDefinition &definition) {
  if (accept(TokenKind::Input)) {
    definition.setBasePrefix(Causality::Input);
  } else if (accept(TokenKind::Output)) {
    definition.setBasePrefix(Causality::Output);
  }

  auto baseClass = parseTypeSpecifier();

  if (!baseClass) {
    return false;
  }

  definition.setBaseClass(std::move(*baseClass));

  if (check(TokenKind::LSquare)) {
    Children<Expression> subscripts;

    if (!parseArraySubscripts(subscripts)) {
      return false;
    }

    for (Child<Expression> &subscript : subscripts) {
      definition.addSubscript(std::unique_ptr<Expression>(
          static_cast<Expression *>(subscript->clone().release())));
    }
  }

  if (check(TokenKind::LParen)) {
    auto modification = parseClassModification();

    if (!modification) {
      return false;
    }

    definition.setClassModification(std::move(modification));
  }

  return parseDescription(definition.getDescription());
}

bool Parser::parseComposition(ClassDefinition &definition) {
  if (!parseElementList(definition, false)) {
    return false;
  }

  while (true) {
    if (accept(TokenKind::Public)) {
      if (!parseElementList(definition, false)) {
        return false;
      }
    } else if (accept(TokenKind::Protected)) {
      if (!parseElementList(definition, true)) {
        return false;
      }
    } else if (check(TokenKind::Equation) ||
               (check(TokenKind::Initial) && check(TokenKind::Equation, 1))) {
      auto section = parseEquationSection();

      if (!section) {
        return false;
      }

      definition.addElement(std::move(section));
    } else if (check(TokenKind::Algorithm) ||
               (check(TokenKind::Initial) && check(TokenKind::Algorithm, 1))) {
      auto section = parseAlgorithmSection();

      if (!section) {
        return false;
      }

      definition.addElement(std::move(section));
    } else {
      break;
    }
  }

  if (accept(TokenKind::External)) {
    auto externalClause = parseExternalClause();

    if (!externalClause) {
      return false;
    }

    definition.setExternalClause(std::move(externalClause));
  }

  if (check(TokenKind::Annotation)) {
    auto annotation = parseAnnotation();

    if (!annotation || !expect(TokenKind::Semicolon)) {
      return false;
    }

    definition.getDescription().annotation = std::move(annotation);
  }

  return true;
}

bool Parser::isElementListEnd() const {
  switch (peek().getKind()) {
  case TokenKind::Public:
  case TokenKind::Protected:
  case TokenKind::Equation:
  case TokenKind::Algorithm:
  case TokenKind::End:
  case TokenKind::External:
  case TokenKind::Annotation:
  case TokenKind::EndOfFile:
    return true;
  case TokenKind::Initial:
    return check(TokenKind::Equation, 1) || check(TokenKind::Algorithm, 1);
  default:
    return false;
  }
}

bool Parser::parseElementList(ClassDefinition &definition, bool isProtected) {
  while (!isElementListEnd()) {
    auto element = parseElement(isProtected);

    if (!element || !expect(TokenKind::Semicolon)) {
      return false;
    }

    definition.addElement(std::move(element));
  }

  return true;
}

std::unique_ptr<Node> Parser::parseElement(bool isProtected) {
  if (check(TokenKind::Import)) {
    auto importClause = parseImportClause();

    if (importClause) {
      importClause->setProtected(isProtected);
    }

    return importClause;
  }

  if (check(TokenKind::Extends)) {
    auto extendsClause = parseExtendsClause();

    if (extendsClause) {
      extendsClause->setProtected(isProtected);
    }

    return extendsClause;
  }

  ElementPrefixes prefixes;
  prefixes.isProtected = isProtected;
  prefixes.isRedeclare = accept(TokenKind::Redeclare);
  prefixes.isFinal = accept(TokenKind::Final);
  prefixes.isInner = accept(TokenKind::Inner);
  prefixes.isOuter = accept(TokenKind::Outer);
  prefixes.isReplaceable = accept(TokenKind::Replaceable);

  std::unique_ptr<ConstrainingClause> constrainingClause;

  auto parseConstraint = [&]() -> bool {
    if (prefixes.isReplaceable && check(TokenKind::ConstrainedBy)) {
      constrainingClause = parseConstrainingClause();
      return static_cast<bool>(constrainingClause);
    }

    return true;
  };

  if (isClassDefinitionStart()) {
    auto classDefinition = parseClassDefinition(prefixes);

    if (!classDefinition || !parseConstraint()) {
      return nullptr;
    }

    classDefinition->setConstrainingClause(std::move(constrainingClause));
    return classDefinition;
  }

  auto componentClause = parseComponentClause(prefixes, false);

  if (!componentClause || !parseConstraint()) {
    return nullptr;
  }

  componentClause->setConstrainingClause(std::move(constrainingClause));
  return componentClause;
}

std::unique_ptr<ImportClause> Parser::parseImportClause() {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::Import)) {
    return nullptr;
  }

  auto importClause = makeNode<ImportClause>(start);

  if (check(TokenKind::Identifier) && check(TokenKind::Equal, 1)) {
    importClause->setForm(ImportClause::Form::Renaming);
    importClause->setAlias(advance().getString());
    advance();

    auto name = parseName();

    if (!name) {
      return nullptr;
    }

    importClause->setName(std::move(*name));
  } else {
    SourceRange nameStart = peek().getLocation();
    Name name;
    auto first = parseIdentifier();

    if (!first) {
      return nullptr;
    }

    name.parts.push_back(*first);

    while (true) {
      if (accept(TokenKind::StarEW) ||
          (check(TokenKind::Dot) && check(TokenKind::Star, 1) &&
           accept(TokenKind::Dot) && accept(TokenKind::Star))) {
        importClause->setForm(ImportClause::Form::Unqualified);
        break;
      }

      if (check(TokenKind::Dot) && check(TokenKind::LCurly, 1)) {
        advance();
        advance();
        importClause->setForm(ImportClause::Form::Multiple);

        do {
          auto importedName = parseIdentifier();

          if (!importedName) {
            return nullptr;
          }

          importClause->addImportedName(*importedName);
        } while (accept(TokenKind::Comma));

        if (!expect(TokenKind::RCurly)) {
          return nullptr;
        }

        break;
      }

      if (!accept(TokenKind::Dot)) {
        break;
      }

      auto part = parseIdentifier();

      if (!part) {
        return nullptr;
      }

      name.parts.push_back(*part);
    }

    name.location = rangeFrom(nameStart);
    importClause->setName(std::move(name));
  }

  if (!parseDescription(importClause->getDescription())) {
    return nullptr;
  }

  finalize(*importClause, start);
  return importClause;
}

std::unique_ptr<ExternalClause> Parser::parseExternalClause() {
  SourceRange start = peek().getLocation();
  auto externalClause = makeNode<ExternalClause>(start);

  if (check(TokenKind::String)) {
    externalClause->setLanguage(advance().getString());
  }

  if (check(TokenKind::Identifier) || check(TokenKind::Dot)) {
    if (!(check(TokenKind::Identifier) && check(TokenKind::LParen, 1))) {
      auto output = parseComponentReference();

      if (!output || !expect(TokenKind::Equal)) {
        return nullptr;
      }

      externalClause->setOutput(std::move(output));
    }

    auto function = parseIdentifier();

    if (!function || !expect(TokenKind::LParen)) {
      return nullptr;
    }

    externalClause->setFunction(*function);

    if (!check(TokenKind::RParen)) {
      do {
        auto argument = parseExpression();

        if (!argument) {
          return nullptr;
        }

        externalClause->addArgument(std::move(argument));
      } while (accept(TokenKind::Comma));
    }

    if (!expect(TokenKind::RParen)) {
      return nullptr;
    }
  }

  if (check(TokenKind::Annotation)) {
    auto annotation = parseAnnotation();

    if (!annotation) {
      return nullptr;
    }

    externalClause->setAnnotation(std::move(annotation));
  }

  if (!expect(TokenKind::Semicolon)) {
    return nullptr;
  }

  finalize(*externalClause, start);
  return externalClause;
}

//===---------------------------------------------------------------------===//
// Extends and components
//===---------------------------------------------------------------------===//

std::unique_ptr<ExtendsClause> Parser::parseExtendsClause() {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::Extends)) {
    return nullptr;
  }

  auto extendsClause = makeNode<ExtendsClause>(start);
  auto baseClass = parseTypeSpecifier();

  if (!baseClass) {
    return nullptr;
  }

  extendsClause->setBaseClass(std::move(*baseClass));

  if (check(TokenKind::LParen)) {
    auto modification = parseClassModification(true);

    if (!modification) {
      return nullptr;
    }

    extendsClause->setClassModification(std::move(modification));
  }

  if (check(TokenKind::Annotation)) {
    auto annotation = parseAnnotation();

    if (!annotation) {
      return nullptr;
    }

    extendsClause->setAnnotation(std::move(annotation));
  }

  finalize(*extendsClause, start);
  return extendsClause;
}

std::unique_ptr<ConstrainingClause> Parser::parseConstrainingClause() {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::ConstrainedBy)) {
    return nullptr;
  }

  auto constrainingClause = makeNode<ConstrainingClause>(start);
  auto typeSpecifier = parseTypeSpecifier();

  if (!typeSpecifier) {
    return nullptr;
  }

  constrainingClause->setTypeSpecifier(std::move(*typeSpecifier));

  if (check(TokenKind::LParen)) {
    auto modification = parseClassModification();

    if (!modification) {
      return nullptr;
    }

    constrainingClause->setClassModification(std::move(modification));
  }

  Description description;

  if (!parseDescription(description)) {
    return nullptr;
  }

  constrainingClause->setDescription(description.text);

  if (description.annotation) {
    constrainingClause->setAnnotation(
        std::unique_ptr<ClassModification>(static_cast<ClassModification *>(
            description.annotation->clone().release())));
  }

  finalize(*constrainingClause, start);
  return constrainingClause;
}

bool Parser::parseTypePrefix(TypePrefix &prefix) {
  if (accept(TokenKind::Flow)) {
    prefix.connectorKind = ConnectorKind::Flow;
  } else if (accept(TokenKind::Stream)) {
    prefix.connectorKind = ConnectorKind::Stream;
  }

  if (accept(TokenKind::Discrete)) {
    prefix.variability = Variability::Discrete;
  } else if (accept(TokenKind::Parameter)) {
    prefix.variability = Variability::Parameter;
  } else if (accept(TokenKind::Constant)) {
    prefix.variability = Variability::Constant;
  }

  if (accept(TokenKind::Input)) {
    prefix.causality = Causality::Input;
  } else if (accept(TokenKind::Output)) {
    prefix.causality = Causality::Output;
  }

  return true;
}

std::unique_ptr<ComponentClause>
Parser::parseComponentClause(const ElementPrefixes &prefixes,
                             bool singleDeclaration) {
  SourceRange start = peek().getLocation();
  auto componentClause = makeNode<ComponentClause>(start);
  componentClause->getPrefixes() = prefixes;

  if (!parseTypePrefix(componentClause->getTypePrefix())) {
    return nullptr;
  }

  auto typeSpecifier = parseTypeSpecifier();

  if (!typeSpecifier) {
    return nullptr;
  }

  componentClause->setTypeSpecifier(std::move(*typeSpecifier));

  if (check(TokenKind::LSquare)) {
    Children<Expression> subscripts;

    if (!parseArraySubscripts(subscripts)) {
      return nullptr;
    }

    for (Child<Expression> &subscript : subscripts) {
      componentClause->addSubscript(std::unique_ptr<Expression>(
          static_cast<Expression *>(subscript->clone().release())));
    }
  }

  do {
    auto declaration = parseComponentDeclaration(!singleDeclaration);

    if (!declaration) {
      return nullptr;
    }

    componentClause->addDeclaration(std::move(declaration));
  } while (!singleDeclaration && accept(TokenKind::Comma));

  finalize(*componentClause, start);
  return componentClause;
}

std::unique_ptr<ComponentDeclaration>
Parser::parseComponentDeclaration(bool allowCondition) {
  SourceRange start = peek().getLocation();
  auto declaration = makeNode<ComponentDeclaration>(start);
  auto name = parseIdentifier();

  if (!name) {
    return nullptr;
  }

  declaration->setName(*name);

  if (check(TokenKind::LSquare)) {
    Children<Expression> subscripts;

    if (!parseArraySubscripts(subscripts)) {
      return nullptr;
    }

    for (Child<Expression> &subscript : subscripts) {
      declaration->addSubscript(std::unique_ptr<Expression>(
          static_cast<Expression *>(subscript->clone().release())));
    }
  }

  if (check(TokenKind::LParen) || check(TokenKind::Equal)) {
    auto modification = parseModification();

    if (!modification) {
      return nullptr;
    }

    declaration->setModification(std::move(modification));
  }

  if (allowCondition && accept(TokenKind::If)) {
    auto condition = parseExpression();

    if (!condition) {
      return nullptr;
    }

    declaration->setCondition(std::move(condition));
  }

  if (!parseDescription(declaration->getDescription())) {
    return nullptr;
  }

  finalize(*declaration, start);
  return declaration;
}

//===---------------------------------------------------------------------===//
// Modifications
//===---------------------------------------------------------------------===//

std::unique_ptr<Modification> Parser::parseModification() {
  SourceRange start = peek().getLocation();
  auto modification = makeNode<Modification>(start);

  if (check(TokenKind::LParen)) {
    auto classModification = parseClassModification();

    if (!classModification) {
      return nullptr;
    }

    modification->setClassModification(std::move(classModification));
  }

  if (accept(TokenKind::Equal)) {
    if (accept(TokenKind::Break)) {
      modification->setBreak(true);
    } else {
      auto value = parseExpression();

      if (!value) {
        return nullptr;
      }

      modification->setValue(std::move(value));
    }
  } else if (!modification->getClassModification()) {
    emitUnexpectedTokenError("modification");
    return nullptr;
  }

  finalize(*modification, start);
  return modification;
}

std::unique_ptr<ClassModification>
Parser::parseClassModification(bool allowInheritanceModifications) {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::LParen)) {
    return nullptr;
  }

  auto classModification = makeNode<ClassModification>(start);

  if (!check(TokenKind::RParen)) {
    do {
      auto argument = parseArgument(allowInheritanceModifications);

      if (!argument) {
        return nullptr;
      }

      classModification->addArgument(std::move(argument));
    } while (accept(TokenKind::Comma));
  }

  if (!expect(TokenKind::RParen)) {
    return nullptr;
  }

  finalize(*classModification, start);
  return classModification;
}

std::unique_ptr<Argument>
Parser::parseArgument(bool allowInheritanceModifications) {
  if (allowInheritanceModifications && check(TokenKind::Break)) {
    return parseInheritanceModification();
  }

  SourceRange start = peek().getLocation();
  bool isRedeclare = accept(TokenKind::Redeclare);
  bool isEach = accept(TokenKind::Each);
  bool isFinal = accept(TokenKind::Final);

  if (isRedeclare || check(TokenKind::Replaceable)) {
    return parseElementReplaceable(isRedeclare, isEach, isFinal);
  }

  auto elementModification = makeNode<ElementModification>(start);
  elementModification->setEach(isEach);
  elementModification->setFinal(isFinal);

  auto name = parseName();

  if (!name) {
    return nullptr;
  }

  elementModification->setName(std::move(*name));

  if (check(TokenKind::LParen) || check(TokenKind::Equal)) {
    auto modification = parseModification();

    if (!modification) {
      return nullptr;
    }

    elementModification->setModification(std::move(modification));
  }

  auto description = parseDescriptionString();

  if (!description) {
    return nullptr;
  }

  elementModification->setDescription(*description);
  finalize(*elementModification, start);
  return elementModification;
}

std::unique_ptr<Argument>
Parser::parseElementReplaceable(bool isRedeclare, bool isEach, bool isFinal) {
  SourceRange start = peek().getLocation();
  auto redeclaration = makeNode<ElementRedeclaration>(start);
  redeclaration->setRedeclare(isRedeclare);
  redeclaration->setEach(isEach);
  redeclaration->setFinal(isFinal);
  redeclaration->setReplaceable(accept(TokenKind::Replaceable));

  if (isClassDefinitionStart()) {
    SourceRange classStart = peek().getLocation();
    auto classDefinition = parseClassDefinition(ElementPrefixes());

    if (!classDefinition) {
      return nullptr;
    }

    if (classDefinition->getForm() == ClassDefinition::Form::Long ||
        classDefinition->getForm() == ClassDefinition::Form::ClassExtends) {
      emitError(classStart,
                "only short class definitions are allowed within "
                "modifications");

      return nullptr;
    }

    redeclaration->setElement(std::move(classDefinition));
  } else {
    auto componentClause = parseComponentClause(ElementPrefixes(), true);

    if (!componentClause) {
      return nullptr;
    }

    redeclaration->setElement(std::move(componentClause));
  }

  if (redeclaration->isReplaceable() && check(TokenKind::ConstrainedBy)) {
    auto constrainingClause = parseConstrainingClause();

    if (!constrainingClause) {
      return nullptr;
    }

    redeclaration->setConstrainingClause(std::move(constrainingClause));
  }

  finalize(*redeclaration, start);
  return redeclaration;
}

std::unique_ptr<Argument> Parser::parseInheritanceModification() {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::Break)) {
    return nullptr;
  }

  auto modification = makeNode<InheritanceModification>(start);

  if (accept(TokenKind::Connect)) {
    if (!expect(TokenKind::LParen)) {
      return nullptr;
    }

    auto lhs = parseComponentReference();

    if (!lhs || !expect(TokenKind::Comma)) {
      return nullptr;
    }

    auto rhs = parseComponentReference();

    if (!rhs || !expect(TokenKind::RParen)) {
      return nullptr;
    }

    modification->setConnect(std::move(lhs), std::move(rhs));
  } else {
    auto component = parseIdentifier();

    if (!component) {
      return nullptr;
    }

    modification->setComponent(*component);
  }

  finalize(*modification, start);
  return modification;
}

//===---------------------------------------------------------------------===//
// Equations and statements
//===---------------------------------------------------------------------===//

bool Parser::isEquationListEnd() const { return isElementListEnd(); }

std::unique_ptr<EquationSection> Parser::parseEquationSection() {
  SourceRange start = peek().getLocation();
  bool isInitial = accept(TokenKind::Initial);

  if (!expect(TokenKind::Equation)) {
    return nullptr;
  }

  auto section = makeNode<EquationSection>(start);
  section->setInitial(isInitial);

  while (!isEquationListEnd()) {
    auto equation = parseEquation();

    if (!equation || !expect(TokenKind::Semicolon)) {
      return nullptr;
    }

    section->addEquation(std::move(equation));
  }

  finalize(*section, start);
  return section;
}

std::unique_ptr<AlgorithmSection> Parser::parseAlgorithmSection() {
  SourceRange start = peek().getLocation();
  bool isInitial = accept(TokenKind::Initial);

  if (!expect(TokenKind::Algorithm)) {
    return nullptr;
  }

  auto section = makeNode<AlgorithmSection>(start);
  section->setInitial(isInitial);

  while (!isEquationListEnd()) {
    auto statement = parseStatement();

    if (!statement || !expect(TokenKind::Semicolon)) {
      return nullptr;
    }

    section->addStatement(std::move(statement));
  }

  finalize(*section, start);
  return section;
}

bool Parser::parseEquationList(Children<Equation> &equations,
                               llvm::ArrayRef<TokenKind> terminators) {
  while (!llvm::is_contained(terminators, peek().getKind()) && !isAtEnd()) {
    auto equation = parseEquation();

    if (!equation || !expect(TokenKind::Semicolon)) {
      return false;
    }

    equations.emplace_back(std::move(equation));
  }

  return true;
}

bool Parser::parseStatementList(Children<Statement> &statements,
                                llvm::ArrayRef<TokenKind> terminators) {
  while (!llvm::is_contained(terminators, peek().getKind()) && !isAtEnd()) {
    auto statement = parseStatement();

    if (!statement || !expect(TokenKind::Semicolon)) {
      return false;
    }

    statements.emplace_back(std::move(statement));
  }

  return true;
}

std::unique_ptr<Equation> Parser::parseEquation() {
  SourceRange start = peek().getLocation();
  std::unique_ptr<Equation> equation;

  if (check(TokenKind::If)) {
    equation = parseIfEquation();
  } else if (check(TokenKind::For)) {
    equation = parseForEquation();
  } else if (check(TokenKind::When)) {
    equation = parseWhenEquation();
  } else if (check(TokenKind::Connect)) {
    equation = parseConnectEquation();
  } else {
    auto lhs = parseSimpleExpression();

    if (!lhs) {
      return nullptr;
    }

    if (accept(TokenKind::Equal)) {
      auto rhs = parseExpression();

      if (!rhs) {
        return nullptr;
      }

      auto equality = makeNode<EqualityEquation>(start);
      equality->setLhs(std::move(lhs));
      equality->setRhs(std::move(rhs));
      equation = std::move(equality);
    } else if (lhs->isa<Call>()) {
      auto callEquation = makeNode<CallEquation>(start);
      callEquation->setCall(
          std::unique_ptr<Call>(static_cast<Call *>(lhs.release())));

      equation = std::move(callEquation);
    } else {
      emitUnexpectedTokenError("'='");
      return nullptr;
    }
  }

  if (!equation || !parseDescription(equation->getDescription())) {
    return nullptr;
  }

  finalize(*equation, start);
  return equation;
}

std::unique_ptr<Equation> Parser::parseIfEquation() {
  SourceRange start = peek().getLocation();
  auto equation = makeNode<IfEquation>(start);

  if (!expect(TokenKind::If)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Equation> equations;

    if (!condition || !expect(TokenKind::Then) ||
        !parseEquationList(
            equations, {TokenKind::ElseIf, TokenKind::Else, TokenKind::End})) {
      return nullptr;
    }

    equation->addBranch(std::move(condition), std::move(equations));
  } while (accept(TokenKind::ElseIf));

  if (accept(TokenKind::Else)) {
    Children<Equation> equations;

    if (!parseEquationList(equations, {TokenKind::End})) {
      return nullptr;
    }

    equation->setElseEquations(std::move(equations));
  }

  if (!expect(TokenKind::End) || !expect(TokenKind::If)) {
    return nullptr;
  }

  return equation;
}

std::unique_ptr<Equation> Parser::parseForEquation() {
  SourceRange start = peek().getLocation();
  auto equation = makeNode<ForEquation>(start);
  Children<ForIndex> indices;
  Children<Equation> equations;

  if (!expect(TokenKind::For) || !parseForIndices(indices) ||
      !expect(TokenKind::Loop) ||
      !parseEquationList(equations, {TokenKind::End}) ||
      !expect(TokenKind::End) || !expect(TokenKind::For)) {
    return nullptr;
  }

  for (Child<ForIndex> &forIndex : indices) {
    equation->addIndex(std::unique_ptr<ForIndex>(
        static_cast<ForIndex *>(forIndex->clone().release())));
  }

  equation->setEquations(std::move(equations));
  return equation;
}

std::unique_ptr<Equation> Parser::parseWhenEquation() {
  SourceRange start = peek().getLocation();
  auto equation = makeNode<WhenEquation>(start);

  if (!expect(TokenKind::When)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Equation> equations;

    if (!condition || !expect(TokenKind::Then) ||
        !parseEquationList(equations, {TokenKind::ElseWhen, TokenKind::End})) {
      return nullptr;
    }

    equation->addBranch(std::move(condition), std::move(equations));
  } while (accept(TokenKind::ElseWhen));

  if (!expect(TokenKind::End) || !expect(TokenKind::When)) {
    return nullptr;
  }

  return equation;
}

std::unique_ptr<Equation> Parser::parseConnectEquation() {
  SourceRange start = peek().getLocation();
  auto equation = makeNode<ConnectEquation>(start);

  if (!expect(TokenKind::Connect) || !expect(TokenKind::LParen)) {
    return nullptr;
  }

  auto lhs = parseComponentReference();

  if (!lhs || !expect(TokenKind::Comma)) {
    return nullptr;
  }

  auto rhs = parseComponentReference();

  if (!rhs || !expect(TokenKind::RParen)) {
    return nullptr;
  }

  equation->setLhs(std::move(lhs));
  equation->setRhs(std::move(rhs));
  return equation;
}

std::unique_ptr<Statement> Parser::parseStatement() {
  SourceRange start = peek().getLocation();
  std::unique_ptr<Statement> statement;

  switch (peek().getKind()) {
  case TokenKind::Break:
    advance();
    statement = makeNode<BreakStatement>(start);
    break;

  case TokenKind::Return:
    advance();
    statement = makeNode<ReturnStatement>(start);
    break;

  case TokenKind::If:
    statement = parseIfStatement();
    break;

  case TokenKind::For:
    statement = parseForStatement();
    break;

  case TokenKind::While:
    statement = parseWhileStatement();
    break;

  case TokenKind::When:
    statement = parseWhenStatement();
    break;

  case TokenKind::LParen: {
    // Assignment of multiple outputs.
    auto multipleAssignment = makeNode<MultipleAssignmentStatement>(start);
    advance();

    do {
      if (check(TokenKind::Comma) || check(TokenKind::RParen)) {
        multipleAssignment->addDestination(nullptr);
        continue;
      }

      auto destination = parseExpression();

      if (!destination) {
        return nullptr;
      }

      multipleAssignment->addDestination(std::move(destination));
    } while (accept(TokenKind::Comma));

    if (!expect(TokenKind::RParen) || !expect(TokenKind::Assignment)) {
      return nullptr;
    }

    SourceRange callStart = peek().getLocation();
    auto callee = parseComponentReference();

    if (!callee) {
      return nullptr;
    }

    auto call = parseFunctionCallArgs(std::move(callee), callStart);

    if (!call) {
      return nullptr;
    }

    multipleAssignment->setCall(std::move(call));
    statement = std::move(multipleAssignment);
    break;
  }

  default: {
    auto destination = parsePrimary();

    if (!destination) {
      return nullptr;
    }

    if (accept(TokenKind::Assignment)) {
      auto value = parseExpression();

      if (!value) {
        return nullptr;
      }

      auto assignment = makeNode<AssignmentStatement>(start);
      assignment->setDestination(std::move(destination));
      assignment->setValue(std::move(value));
      statement = std::move(assignment);
    } else if (destination->isa<Call>()) {
      auto callStatement = makeNode<CallStatement>(start);
      callStatement->setCall(
          std::unique_ptr<Call>(static_cast<Call *>(destination.release())));

      statement = std::move(callStatement);
    } else {
      emitUnexpectedTokenError("':='");
      return nullptr;
    }

    break;
  }
  }

  if (!statement || !parseDescription(statement->getDescription())) {
    return nullptr;
  }

  finalize(*statement, start);
  return statement;
}

std::unique_ptr<Statement> Parser::parseIfStatement() {
  SourceRange start = peek().getLocation();
  auto statement = makeNode<IfStatement>(start);

  if (!expect(TokenKind::If)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Statement> statements;

    if (!condition || !expect(TokenKind::Then) ||
        !parseStatementList(
            statements, {TokenKind::ElseIf, TokenKind::Else, TokenKind::End})) {
      return nullptr;
    }

    statement->addBranch(std::move(condition), std::move(statements));
  } while (accept(TokenKind::ElseIf));

  if (accept(TokenKind::Else)) {
    Children<Statement> statements;

    if (!parseStatementList(statements, {TokenKind::End})) {
      return nullptr;
    }

    statement->setElseStatements(std::move(statements));
  }

  if (!expect(TokenKind::End) || !expect(TokenKind::If)) {
    return nullptr;
  }

  return statement;
}

std::unique_ptr<Statement> Parser::parseForStatement() {
  SourceRange start = peek().getLocation();
  auto statement = makeNode<ForStatement>(start);
  Children<ForIndex> indices;
  Children<Statement> statements;

  if (!expect(TokenKind::For) || !parseForIndices(indices) ||
      !expect(TokenKind::Loop) ||
      !parseStatementList(statements, {TokenKind::End}) ||
      !expect(TokenKind::End) || !expect(TokenKind::For)) {
    return nullptr;
  }

  for (Child<ForIndex> &forIndex : indices) {
    statement->addIndex(std::unique_ptr<ForIndex>(
        static_cast<ForIndex *>(forIndex->clone().release())));
  }

  statement->setStatements(std::move(statements));
  return statement;
}

std::unique_ptr<Statement> Parser::parseWhileStatement() {
  SourceRange start = peek().getLocation();
  auto statement = makeNode<WhileStatement>(start);

  if (!expect(TokenKind::While)) {
    return nullptr;
  }

  auto condition = parseExpression();
  Children<Statement> statements;

  if (!condition || !expect(TokenKind::Loop) ||
      !parseStatementList(statements, {TokenKind::End}) ||
      !expect(TokenKind::End) || !expect(TokenKind::While)) {
    return nullptr;
  }

  statement->setCondition(std::move(condition));
  statement->setStatements(std::move(statements));
  return statement;
}

std::unique_ptr<Statement> Parser::parseWhenStatement() {
  SourceRange start = peek().getLocation();
  auto statement = makeNode<WhenStatement>(start);

  if (!expect(TokenKind::When)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Statement> statements;

    if (!condition || !expect(TokenKind::Then) ||
        !parseStatementList(statements,
                            {TokenKind::ElseWhen, TokenKind::End})) {
      return nullptr;
    }

    statement->addBranch(std::move(condition), std::move(statements));
  } while (accept(TokenKind::ElseWhen));

  if (!expect(TokenKind::End) || !expect(TokenKind::When)) {
    return nullptr;
  }

  return statement;
}

bool Parser::parseForIndices(Children<ForIndex> &indices) {
  do {
    auto forIndex = parseForIndex();

    if (!forIndex) {
      return false;
    }

    indices.emplace_back(std::move(forIndex));
  } while (accept(TokenKind::Comma));

  return true;
}

std::unique_ptr<ForIndex> Parser::parseForIndex() {
  SourceRange start = peek().getLocation();
  auto forIndex = makeNode<ForIndex>(start);
  auto name = parseIdentifier();

  if (!name) {
    return nullptr;
  }

  forIndex->setName(*name);

  if (accept(TokenKind::In)) {
    auto range = parseExpression();

    if (!range) {
      return nullptr;
    }

    forIndex->setRange(std::move(range));
  }

  finalize(*forIndex, start);
  return forIndex;
}

//===---------------------------------------------------------------------===//
// Expressions
//===---------------------------------------------------------------------===//

std::unique_ptr<Expression> Parser::parseExpression() {
  SourceRange start = peek().getLocation();

  if (!accept(TokenKind::If)) {
    return parseSimpleExpression();
  }

  auto expression = makeNode<IfExpression>(start);

  do {
    auto condition = parseExpression();

    if (!condition || !expect(TokenKind::Then)) {
      return nullptr;
    }

    auto value = parseExpression();

    if (!value) {
      return nullptr;
    }

    expression->addBranch(std::move(condition), std::move(value));
  } while (accept(TokenKind::ElseIf));

  if (!expect(TokenKind::Else)) {
    return nullptr;
  }

  auto elseValue = parseExpression();

  if (!elseValue) {
    return nullptr;
  }

  expression->setElseValue(std::move(elseValue));
  finalize(*expression, start);
  return expression;
}

std::unique_ptr<Expression> Parser::parseSimpleExpression() {
  SourceRange start = peek().getLocation();
  auto first = parseLogicalExpression();

  if (!first || !accept(TokenKind::Colon)) {
    return first;
  }

  auto second = parseLogicalExpression();

  if (!second) {
    return nullptr;
  }

  auto range = makeNode<Range>(start);
  range->setStart(std::move(first));

  if (accept(TokenKind::Colon)) {
    auto third = parseLogicalExpression();

    if (!third) {
      return nullptr;
    }

    range->setStep(std::move(second));
    range->setStop(std::move(third));
  } else {
    range->setStop(std::move(second));
  }

  finalize(*range, start);
  return range;
}

namespace {
std::unique_ptr<Expression> makeBinary(SourceRange location,
                                       BinaryOperator op,
                                       std::unique_ptr<Expression> lhs,
                                       std::unique_ptr<Expression> rhs) {
  auto operation = std::make_unique<BinaryOperation>(location);
  operation->setOperator(op);
  operation->setLhs(std::move(lhs));
  operation->setRhs(std::move(rhs));
  return operation;
}
} // namespace

std::unique_ptr<Expression> Parser::parseLogicalExpression() {
  SourceRange start = peek().getLocation();
  auto result = parseLogicalTerm();

  while (result && accept(TokenKind::Or)) {
    auto rhs = parseLogicalTerm();

    if (!rhs) {
      return nullptr;
    }

    result = makeBinary(rangeFrom(start), BinaryOperator::Or,
                        std::move(result), std::move(rhs));
  }

  return result;
}

std::unique_ptr<Expression> Parser::parseLogicalTerm() {
  SourceRange start = peek().getLocation();
  auto result = parseLogicalFactor();

  while (result && accept(TokenKind::And)) {
    auto rhs = parseLogicalFactor();

    if (!rhs) {
      return nullptr;
    }

    result = makeBinary(rangeFrom(start), BinaryOperator::And,
                        std::move(result), std::move(rhs));
  }

  return result;
}

std::unique_ptr<Expression> Parser::parseLogicalFactor() {
  SourceRange start = peek().getLocation();

  if (!accept(TokenKind::Not)) {
    return parseRelation();
  }

  auto operand = parseRelation();

  if (!operand) {
    return nullptr;
  }

  auto operation = makeNode<UnaryOperation>(start);
  operation->setOperator(UnaryOperator::Not);
  operation->setOperand(std::move(operand));
  return operation;
}

std::unique_ptr<Expression> Parser::parseRelation() {
  SourceRange start = peek().getLocation();
  auto lhs = parseArithmeticExpression();

  if (!lhs) {
    return nullptr;
  }

  std::optional<BinaryOperator> op;

  switch (peek().getKind()) {
  case TokenKind::Less:
    op = BinaryOperator::Less;
    break;
  case TokenKind::LessEqual:
    op = BinaryOperator::LessEqual;
    break;
  case TokenKind::Greater:
    op = BinaryOperator::Greater;
    break;
  case TokenKind::GreaterEqual:
    op = BinaryOperator::GreaterEqual;
    break;
  case TokenKind::EqualEqual:
    op = BinaryOperator::Equal;
    break;
  case TokenKind::NotEqual:
    op = BinaryOperator::NotEqual;
    break;
  default:
    return lhs;
  }

  advance();
  auto rhs = parseArithmeticExpression();

  if (!rhs) {
    return nullptr;
  }

  return makeBinary(rangeFrom(start), *op, std::move(lhs), std::move(rhs));
}

std::unique_ptr<Expression> Parser::parseArithmeticExpression() {
  SourceRange start = peek().getLocation();
  std::optional<UnaryOperator> unaryOp;

  switch (peek().getKind()) {
  case TokenKind::Plus:
    unaryOp = UnaryOperator::Plus;
    break;
  case TokenKind::Minus:
    unaryOp = UnaryOperator::Minus;
    break;
  case TokenKind::PlusEW:
    unaryOp = UnaryOperator::PlusEW;
    break;
  case TokenKind::MinusEW:
    unaryOp = UnaryOperator::MinusEW;
    break;
  default:
    break;
  }

  if (unaryOp) {
    advance();
  }

  auto result = parseTerm();

  if (!result) {
    return nullptr;
  }

  if (unaryOp) {
    auto operation = makeNode<UnaryOperation>(start);
    operation->setOperator(*unaryOp);
    operation->setOperand(std::move(result));
    result = std::move(operation);
  }

  while (true) {
    std::optional<BinaryOperator> op;

    switch (peek().getKind()) {
    case TokenKind::Plus:
      op = BinaryOperator::Add;
      break;
    case TokenKind::Minus:
      op = BinaryOperator::Sub;
      break;
    case TokenKind::PlusEW:
      op = BinaryOperator::AddEW;
      break;
    case TokenKind::MinusEW:
      op = BinaryOperator::SubEW;
      break;
    default:
      return result;
    }

    advance();
    auto rhs = parseTerm();

    if (!rhs) {
      return nullptr;
    }

    result =
        makeBinary(rangeFrom(start), *op, std::move(result), std::move(rhs));
  }
}

std::unique_ptr<Expression> Parser::parseTerm() {
  SourceRange start = peek().getLocation();
  auto result = parseFactor();

  while (result) {
    std::optional<BinaryOperator> op;

    switch (peek().getKind()) {
    case TokenKind::Star:
      op = BinaryOperator::Mul;
      break;
    case TokenKind::Slash:
      op = BinaryOperator::Div;
      break;
    case TokenKind::StarEW:
      op = BinaryOperator::MulEW;
      break;
    case TokenKind::SlashEW:
      op = BinaryOperator::DivEW;
      break;
    default:
      return result;
    }

    advance();
    auto rhs = parseFactor();

    if (!rhs) {
      return nullptr;
    }

    result =
        makeBinary(rangeFrom(start), *op, std::move(result), std::move(rhs));
  }

  return result;
}

std::unique_ptr<Expression> Parser::parseFactor() {
  SourceRange start = peek().getLocation();
  auto base = parsePrimary();

  if (!base) {
    return nullptr;
  }

  std::optional<BinaryOperator> op;

  if (check(TokenKind::Caret)) {
    op = BinaryOperator::Pow;
  } else if (check(TokenKind::CaretEW)) {
    op = BinaryOperator::PowEW;
  } else {
    return base;
  }

  advance();
  auto exponent = parsePrimary();

  if (!exponent) {
    return nullptr;
  }

  return makeBinary(rangeFrom(start), *op, std::move(base),
                    std::move(exponent));
}

std::unique_ptr<Expression> Parser::parseNumber() {
  SourceRange start = peek().getLocation();
  Token token = advance();

  // Unitful literal: the unit is a quoted identifier
  // following the number.
  std::optional<std::string> unit;

  if (check(TokenKind::Identifier) &&
      llvm::StringRef(peek().getString()).starts_with("'")) {
    std::string quoted = advance().getString();
    unit = llvm::StringRef(quoted).drop_front().drop_back().str();
  }

  if (token.isa<TokenKind::Integer>()) {
    auto literal = makeNode<IntegerLiteral>(start);
    literal->setValue(token.getInt());
    literal->setUnit(std::move(unit));
    return literal;
  }

  auto literal = makeNode<RealLiteral>(start);
  literal->setValue(token.getFloat());
  literal->setUnit(std::move(unit));
  return literal;
}

std::unique_ptr<Expression> Parser::parsePrimary() {
  SourceRange start = peek().getLocation();

  switch (peek().getKind()) {
  case TokenKind::Integer:
  case TokenKind::Real:
    return parseNumber();

  case TokenKind::String: {
    auto literal = makeNode<StringLiteral>(start);
    literal->setValue(advance().getString());
    finalize(*literal, start);
    return literal;
  }

  case TokenKind::True:
  case TokenKind::False: {
    auto literal = makeNode<BooleanLiteral>(start);
    literal->setValue(advance().isa<TokenKind::True>());
    finalize(*literal, start);
    return literal;
  }

  case TokenKind::Time:
    advance();
    return makeNode<Time>(start);

  case TokenKind::End:
    advance();
    return makeNode<End>(start);

  case TokenKind::Der:
  case TokenKind::Initial:
  case TokenKind::Pure: {
    // Operators with function syntax.
    auto callee = makeNode<ComponentReference>(start);
    ComponentReference::Part part;
    part.name = toString(advance().getKind()).str();
    part.location = rangeFrom(start);
    callee->addPart(std::move(part));
    return parseFunctionCallArgs(std::move(callee), start);
  }

  case TokenKind::Dot:
  case TokenKind::Identifier: {
    auto reference = parseComponentReference();

    if (!reference) {
      return nullptr;
    }

    if (check(TokenKind::LParen)) {
      return parseFunctionCallArgs(std::move(reference), start);
    }

    return reference;
  }

  case TokenKind::LParen:
    return parseParenthesized();

  case TokenKind::LSquare:
    return parseArrayConcatenation();

  case TokenKind::LCurly:
    return parseArrayConstructor();

  default:
    emitUnexpectedTokenError("expression");
    return nullptr;
  }
}

std::unique_ptr<Expression> Parser::parseParenthesized() {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::LParen)) {
    return nullptr;
  }

  Children<Expression> elements;
  bool hasCommas = false;

  do {
    if (check(TokenKind::Comma) || check(TokenKind::RParen)) {
      elements.emplace_back(nullptr);
      continue;
    }

    auto element = parseExpression();

    if (!element) {
      return nullptr;
    }

    elements.emplace_back(std::move(element));
  } while (accept(TokenKind::Comma) && (hasCommas = true));

  if (!expect(TokenKind::RParen)) {
    return nullptr;
  }

  std::unique_ptr<Expression> result;

  if (!hasCommas && elements.size() == 1 && elements[0]) {
    result = std::unique_ptr<Expression>(
        static_cast<Expression *>(elements[0]->clone().release()));
  } else {
    auto outputList = makeNode<OutputList>(start);

    for (Child<Expression> &element : elements) {
      outputList->addElement(
          element ? std::unique_ptr<Expression>(
                        static_cast<Expression *>(element->clone().release()))
                  : nullptr);
    }

    result = std::move(outputList);
  }

  if (check(TokenKind::LSquare)) {
    auto subscripted = makeNode<Subscripted>(start);
    Children<Expression> subscripts;

    if (!parseArraySubscripts(subscripts)) {
      return nullptr;
    }

    subscripted->setBase(std::move(result));

    for (Child<Expression> &subscript : subscripts) {
      subscripted->addSubscript(std::unique_ptr<Expression>(
          static_cast<Expression *>(subscript->clone().release())));
    }

    finalize(*subscripted, start);
    return subscripted;
  }

  if (check(TokenKind::Dot) && check(TokenKind::Identifier, 1)) {
    advance();
    auto memberAccess = makeNode<MemberAccess>(start);
    memberAccess->setBase(std::move(result));
    memberAccess->setMember(advance().getString());
    finalize(*memberAccess, start);
    return memberAccess;
  }

  return result;
}

std::unique_ptr<Expression> Parser::parseArrayConcatenation() {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::LSquare)) {
    return nullptr;
  }

  auto concatenation = makeNode<ArrayConcatenation>(start);

  do {
    Children<Expression> row;

    do {
      auto element = parseExpression();

      if (!element) {
        return nullptr;
      }

      row.emplace_back(std::move(element));
    } while (accept(TokenKind::Comma));

    concatenation->addRow(std::move(row));
  } while (accept(TokenKind::Semicolon));

  if (!expect(TokenKind::RSquare)) {
    return nullptr;
  }

  finalize(*concatenation, start);
  return concatenation;
}

std::unique_ptr<Expression> Parser::parseArrayConstructor() {
  SourceRange start = peek().getLocation();

  if (!expect(TokenKind::LCurly)) {
    return nullptr;
  }

  auto constructor = makeNode<ArrayConstructor>(start);
  auto first = parseExpression();

  if (!first) {
    return nullptr;
  }

  constructor->addElement(std::move(first));

  if (accept(TokenKind::For)) {
    Children<ForIndex> indices;

    if (!parseForIndices(indices)) {
      return nullptr;
    }

    for (Child<ForIndex> &forIndex : indices) {
      constructor->addIterator(std::unique_ptr<ForIndex>(
          static_cast<ForIndex *>(forIndex->clone().release())));
    }
  } else {
    while (accept(TokenKind::Comma)) {
      auto element = parseExpression();

      if (!element) {
        return nullptr;
      }

      constructor->addElement(std::move(element));
    }
  }

  if (!expect(TokenKind::RCurly)) {
    return nullptr;
  }

  finalize(*constructor, start);
  return constructor;
}

std::unique_ptr<ComponentReference> Parser::parseComponentReference() {
  SourceRange start = peek().getLocation();
  auto reference = makeNode<ComponentReference>(start);
  reference->setGlobal(accept(TokenKind::Dot));

  do {
    SourceRange partStart = peek().getLocation();
    ComponentReference::Part part;
    auto name = parseIdentifier();

    if (!name) {
      return nullptr;
    }

    part.name = *name;

    if (check(TokenKind::LSquare) && !parseArraySubscripts(part.subscripts)) {
      return nullptr;
    }

    part.location = rangeFrom(partStart);
    reference->addPart(std::move(part));
  } while (check(TokenKind::Dot) && check(TokenKind::Identifier, 1) &&
           accept(TokenKind::Dot));

  finalize(*reference, start);
  return reference;
}

bool Parser::isNamedArgumentStart() const {
  return check(TokenKind::Identifier) && check(TokenKind::Equal, 1);
}

std::unique_ptr<NamedArgument> Parser::parseNamedArgument() {
  SourceRange start = peek().getLocation();
  auto argument = makeNode<NamedArgument>(start);
  auto name = parseIdentifier();

  if (!name || !expect(TokenKind::Equal)) {
    return nullptr;
  }

  auto value = parseFunctionArgument();

  if (!value) {
    return nullptr;
  }

  argument->setName(*name);
  argument->setValue(std::move(value));
  finalize(*argument, start);
  return argument;
}

std::unique_ptr<Expression> Parser::parseFunctionArgument() {
  SourceRange start = peek().getLocation();

  if (!accept(TokenKind::Function)) {
    return parseExpression();
  }

  // Function partial application.
  auto application = makeNode<FunctionPartialApplication>(start);
  auto function = parseTypeSpecifier();

  if (!function || !expect(TokenKind::LParen)) {
    return nullptr;
  }

  application->setFunction(std::move(*function));

  if (!check(TokenKind::RParen)) {
    do {
      auto argument = parseNamedArgument();

      if (!argument) {
        return nullptr;
      }

      application->addNamedArgument(std::move(argument));
    } while (accept(TokenKind::Comma));
  }

  if (!expect(TokenKind::RParen)) {
    return nullptr;
  }

  finalize(*application, start);
  return application;
}

std::unique_ptr<Call>
Parser::parseFunctionCallArgs(std::unique_ptr<ComponentReference> callee,
                              const SourceRange &start) {
  auto call = makeNode<Call>(start);
  call->setCallee(std::move(callee));

  if (!expect(TokenKind::LParen)) {
    return nullptr;
  }

  if (!check(TokenKind::RParen)) {
    bool namedArguments = isNamedArgumentStart();

    if (!namedArguments) {
      auto first = parseFunctionArgument();

      if (!first) {
        return nullptr;
      }

      bool isPartialApplication = first->isa<FunctionPartialApplication>();
      call->addArgument(std::move(first));

      if (!isPartialApplication && accept(TokenKind::For)) {
        // Reduction or array constructor with iterators.
        Children<ForIndex> indices;

        if (!parseForIndices(indices)) {
          return nullptr;
        }

        for (Child<ForIndex> &forIndex : indices) {
          call->addIterator(std::unique_ptr<ForIndex>(
              static_cast<ForIndex *>(forIndex->clone().release())));
        }
      } else {
        while (accept(TokenKind::Comma)) {
          if (isNamedArgumentStart()) {
            namedArguments = true;
            break;
          }

          auto argument = parseFunctionArgument();

          if (!argument) {
            return nullptr;
          }

          call->addArgument(std::move(argument));
        }
      }
    }

    if (namedArguments) {
      do {
        auto argument = parseNamedArgument();

        if (!argument) {
          return nullptr;
        }

        call->addNamedArgument(std::move(argument));
      } while (accept(TokenKind::Comma));
    }
  }

  if (!expect(TokenKind::RParen)) {
    return nullptr;
  }

  finalize(*call, start);
  return call;
}

bool Parser::parseArraySubscripts(Children<Expression> &subscripts) {
  if (!expect(TokenKind::LSquare)) {
    return false;
  }

  do {
    auto subscript = parseSubscript();

    if (!subscript) {
      return false;
    }

    subscripts.emplace_back(std::move(subscript));
  } while (accept(TokenKind::Comma));

  return expect(TokenKind::RSquare);
}

std::unique_ptr<Expression> Parser::parseSubscript() {
  SourceRange start = peek().getLocation();

  if (check(TokenKind::Colon) &&
      (check(TokenKind::Comma, 1) || check(TokenKind::RSquare, 1))) {
    advance();
    return makeNode<Colon>(start);
  }

  return parseExpression();
}
} // namespace marco::parser::modelica
