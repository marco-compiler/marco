#include "marco/Parser/Modelica/Parser.h"
#include "marco/Parser/Modelica/Lexer.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::parser::modelica;

namespace marco::parser::modelica {
Parser::Parser(clang::DiagnosticsEngine &diagnosticsEngine,
               clang::SourceManager &sourceManager,
               std::shared_ptr<SourceFile> source)
    : diagnosticsEngine(&diagnosticsEngine), sourceManager(&sourceManager) {
  Lexer lexer(std::move(source));
  tokens = lexer.tokenize();
}

bool Parser::isAtEnd() const { return check(TK::EndOfFile); }

//===---------------------------------------------------------------------===//
// Token handling
//===---------------------------------------------------------------------===//

const Token &Parser::peek(size_t offset) const {
  return tokens[std::min(index + offset, tokens.size() - 1)];
}

bool Parser::check(TokenKind kind, size_t offset) const {
  return peek(offset).is(kind);
}

const Token &Parser::advance() {
  const Token &token = peek();

  if (index + 1 < tokens.size()) {
    ++index;
  }

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

SourceRange Parser::rangeFrom(size_t startIndex) const {
  if (index <= startIndex) {
    return tokens[std::min(startIndex, tokens.size() - 1)].location;
  }

  return {tokens[startIndex].location.begin, tokens[index - 1].location.end};
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

  if (token.is(TK::Error)) {
    emitError(token.location, token.text);
    return;
  }

  emitError(token.location,
            "expected " + expected.str() + ", but found " + token.describe());
}

//===---------------------------------------------------------------------===//
// Names and descriptions
//===---------------------------------------------------------------------===//

std::optional<std::string> Parser::parseIdentifier() {
  if (!check(TK::Identifier)) {
    emitUnexpectedTokenError("identifier");
    return std::nullopt;
  }

  return advance().text;
}

std::optional<Name> Parser::parseName() {
  size_t start = index;
  Name name;

  do {
    auto identifier = parseIdentifier();

    if (!identifier) {
      return std::nullopt;
    }

    name.parts.push_back(*identifier);
  } while (check(TK::Dot) && check(TK::Identifier, 1) && accept(TK::Dot));

  name.location = rangeFrom(start);
  return name;
}

std::optional<Name> Parser::parseTypeSpecifier() {
  size_t start = index;
  bool global = accept(TK::Dot);
  auto name = parseName();

  if (!name) {
    return std::nullopt;
  }

  name->global = global;
  name->location = rangeFrom(start);
  return name;
}

std::optional<std::string> Parser::parseDescriptionString() {
  if (!check(TK::String)) {
    return std::string();
  }

  std::string text = advance().text;

  while (check(TK::Plus) && check(TK::String, 1)) {
    advance();
    text += advance().text;
  }

  return text;
}

bool Parser::parseDescription(Description &description) {
  auto text = parseDescriptionString();

  if (!text) {
    return false;
  }

  description.text = *text;

  if (check(TK::Annotation)) {
    auto annotation = parseAnnotation();

    if (!annotation) {
      return false;
    }

    description.annotation = std::move(annotation);
  }

  return true;
}

std::unique_ptr<ClassModification> Parser::parseAnnotation() {
  if (!expect(TK::Annotation)) {
    return nullptr;
  }

  return parseClassModification();
}

//===---------------------------------------------------------------------===//
// Stored definition
//===---------------------------------------------------------------------===//

std::unique_ptr<StoredDefinition> Parser::parseStoredDefinition() {
  size_t start = index;
  auto definition = makeNode<StoredDefinition>(start);

  if (accept(TK::Within)) {
    Name within;

    if (!check(TK::Semicolon)) {
      auto name = parseName();

      if (!name) {
        return nullptr;
      }

      within = std::move(*name);
    }

    definition->setWithin(std::move(within));

    if (!expect(TK::Semicolon)) {
      return nullptr;
    }
  }

  while (!isAtEnd()) {
    ElementPrefixes prefixes;
    prefixes.isFinal = accept(TK::Final);

    auto classDefinition = parseClassDefinition(prefixes);

    if (!classDefinition || !expect(TK::Semicolon)) {
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
  switch (peek().kind) {
  case TK::Encapsulated:
  case TK::Partial:
  case TK::Class:
  case TK::Model:
  case TK::Operator:
  case TK::Record:
  case TK::Block:
  case TK::Expandable:
  case TK::Connector:
  case TK::Type:
  case TK::Package:
  case TK::Pure:
  case TK::Impure:
  case TK::Function:
    return true;
  default:
    return false;
  }
}

bool Parser::parseClassPrefixes(ClassDefinition &classDefinition) {
  classDefinition.setPartial(accept(TK::Partial));

  if (accept(TK::Class)) {
    classDefinition.setRestriction(ClassRestriction::Class);
  } else if (accept(TK::Model)) {
    classDefinition.setRestriction(ClassRestriction::Model);
  } else if (accept(TK::Record)) {
    classDefinition.setRestriction(ClassRestriction::Record);
  } else if (accept(TK::Block)) {
    classDefinition.setRestriction(ClassRestriction::Block);
  } else if (accept(TK::Expandable)) {
    if (!expect(TK::Connector)) {
      return false;
    }

    classDefinition.setRestriction(ClassRestriction::ExpandableConnector);
  } else if (accept(TK::Connector)) {
    classDefinition.setRestriction(ClassRestriction::Connector);
  } else if (accept(TK::Type)) {
    classDefinition.setRestriction(ClassRestriction::Type);
  } else if (accept(TK::Package)) {
    classDefinition.setRestriction(ClassRestriction::Package);
  } else if (check(TK::Pure) || check(TK::Impure) || check(TK::Operator) ||
             check(TK::Function)) {
    Purity purity = Purity::Unspecified;

    if (accept(TK::Pure)) {
      purity = Purity::Pure;
    } else if (accept(TK::Impure)) {
      purity = Purity::Impure;
    }

    classDefinition.setPurity(purity);

    if (accept(TK::Operator)) {
      if (accept(TK::Function)) {
        classDefinition.setRestriction(ClassRestriction::OperatorFunction);
      } else if (purity == Purity::Unspecified && accept(TK::Record)) {
        classDefinition.setRestriction(ClassRestriction::OperatorRecord);
      } else if (purity == Purity::Unspecified) {
        classDefinition.setRestriction(ClassRestriction::Operator);
      } else {
        emitUnexpectedTokenError("'function'");
        return false;
      }
    } else {
      if (!expect(TK::Function)) {
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
  size_t start = index;
  auto classDefinition = makeNode<ClassDefinition>(start);
  classDefinition->getPrefixes() = prefixes;
  classDefinition->setEncapsulated(accept(TK::Encapsulated));

  if (!parseClassPrefixes(*classDefinition)) {
    return nullptr;
  }

  bool isClassExtends = accept(TK::Extends);
  size_t nameIndex = index;
  auto name = parseIdentifier();

  if (!name) {
    return nullptr;
  }

  classDefinition->setName(*name);

  if (!isClassExtends && accept(TK::Equal)) {
    if (accept(TK::Enumeration)) {
      // Enumeration.
      classDefinition->setForm(ClassDefinition::Form::Enumeration);

      if (!expect(TK::LParen)) {
        return nullptr;
      }

      if (accept(TK::Colon)) {
        classDefinition->setUnspecifiedEnumeration(true);
      } else if (!check(TK::RParen)) {
        do {
          size_t literalStart = index;
          auto literal = makeNode<EnumerationLiteral>(literalStart);
          auto literalName = parseIdentifier();

          if (!literalName || !parseDescription(literal->getDescription())) {
            return nullptr;
          }

          literal->setName(*literalName);
          finalize(*literal, literalStart);
          classDefinition->addEnumerationLiteral(std::move(literal));
        } while (accept(TK::Comma));
      }

      if (!expect(TK::RParen) ||
          !parseDescription(classDefinition->getDescription())) {
        return nullptr;
      }
    } else if (accept(TK::Der)) {
      // Derivative of a function.
      classDefinition->setForm(ClassDefinition::Form::Derivative);

      if (!expect(TK::LParen)) {
        return nullptr;
      }

      auto function = parseTypeSpecifier();

      if (!function || !expect(TK::Comma)) {
        return nullptr;
      }

      classDefinition->setDerivedFunction(std::move(*function));

      do {
        auto arg = parseIdentifier();

        if (!arg) {
          return nullptr;
        }

        classDefinition->addDerivativeArgument(*arg);
      } while (accept(TK::Comma));

      if (!expect(TK::RParen) ||
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

    if (check(TK::LParen)) {
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

  if (!parseComposition(*classDefinition) || !expect(TK::End)) {
    return nullptr;
  }

  size_t endNameIndex = index;
  auto endName = parseIdentifier();

  if (!endName) {
    return nullptr;
  }

  if (*endName != *name) {
    emitError(tokens[endNameIndex].location,
              "the name '" + *endName +
                  "' doesn't match the name of the class '" + *name + "'");

    emitError(tokens[nameIndex].location, "class declared here");
    return nullptr;
  }

  finalize(*classDefinition, start);
  return classDefinition;
}

bool Parser::parseShortClassSpecifier(ClassDefinition &definition) {
  if (accept(TK::Input)) {
    definition.setBasePrefix(Causality::Input);
  } else if (accept(TK::Output)) {
    definition.setBasePrefix(Causality::Output);
  }

  auto baseClass = parseTypeSpecifier();

  if (!baseClass) {
    return false;
  }

  definition.setBaseClass(std::move(*baseClass));

  if (check(TK::LSquare)) {
    Children<Expression> subscripts;

    if (!parseArraySubscripts(subscripts)) {
      return false;
    }

    for (Child<Expression> &subscript : subscripts) {
      definition.addSubscript(std::unique_ptr<Expression>(
          static_cast<Expression *>(subscript->clone().release())));
    }
  }

  if (check(TK::LParen)) {
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
    if (accept(TK::Public)) {
      if (!parseElementList(definition, false)) {
        return false;
      }
    } else if (accept(TK::Protected)) {
      if (!parseElementList(definition, true)) {
        return false;
      }
    } else if (check(TK::Equation) ||
               (check(TK::Initial) && check(TK::Equation, 1))) {
      auto section = parseEquationSection();

      if (!section) {
        return false;
      }

      definition.addElement(std::move(section));
    } else if (check(TK::Algorithm) ||
               (check(TK::Initial) && check(TK::Algorithm, 1))) {
      auto section = parseAlgorithmSection();

      if (!section) {
        return false;
      }

      definition.addElement(std::move(section));
    } else {
      break;
    }
  }

  if (accept(TK::External)) {
    auto externalClause = parseExternalClause();

    if (!externalClause) {
      return false;
    }

    definition.setExternalClause(std::move(externalClause));
  }

  if (check(TK::Annotation)) {
    auto annotation = parseAnnotation();

    if (!annotation || !expect(TK::Semicolon)) {
      return false;
    }

    definition.getDescription().annotation = std::move(annotation);
  }

  return true;
}

bool Parser::isElementListEnd() const {
  switch (peek().kind) {
  case TK::Public:
  case TK::Protected:
  case TK::Equation:
  case TK::Algorithm:
  case TK::End:
  case TK::External:
  case TK::Annotation:
  case TK::EndOfFile:
    return true;
  case TK::Initial:
    return check(TK::Equation, 1) || check(TK::Algorithm, 1);
  default:
    return false;
  }
}

bool Parser::parseElementList(ClassDefinition &definition, bool isProtected) {
  while (!isElementListEnd()) {
    auto element = parseElement(isProtected);

    if (!element || !expect(TK::Semicolon)) {
      return false;
    }

    definition.addElement(std::move(element));
  }

  return true;
}

std::unique_ptr<Node> Parser::parseElement(bool isProtected) {
  if (check(TK::Import)) {
    auto importClause = parseImportClause();

    if (importClause) {
      importClause->setProtected(isProtected);
    }

    return importClause;
  }

  if (check(TK::Extends)) {
    auto extendsClause = parseExtendsClause();

    if (extendsClause) {
      extendsClause->setProtected(isProtected);
    }

    return extendsClause;
  }

  ElementPrefixes prefixes;
  prefixes.isProtected = isProtected;
  prefixes.isRedeclare = accept(TK::Redeclare);
  prefixes.isFinal = accept(TK::Final);
  prefixes.isInner = accept(TK::Inner);
  prefixes.isOuter = accept(TK::Outer);
  prefixes.isReplaceable = accept(TK::Replaceable);

  std::unique_ptr<ConstrainingClause> constrainingClause;

  auto parseConstraint = [&]() -> bool {
    if (prefixes.isReplaceable && check(TK::ConstrainedBy)) {
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
  size_t start = index;

  if (!expect(TK::Import)) {
    return nullptr;
  }

  auto importClause = makeNode<ImportClause>(start);

  if (check(TK::Identifier) && check(TK::Equal, 1)) {
    importClause->setForm(ImportClause::Form::Renaming);
    importClause->setAlias(advance().text);
    advance();

    auto name = parseName();

    if (!name) {
      return nullptr;
    }

    importClause->setName(std::move(*name));
  } else {
    size_t nameStart = index;
    Name name;
    auto first = parseIdentifier();

    if (!first) {
      return nullptr;
    }

    name.parts.push_back(*first);

    while (true) {
      if (accept(TK::StarEW) ||
          (check(TK::Dot) && check(TK::Star, 1) && accept(TK::Dot) &&
           accept(TK::Star))) {
        importClause->setForm(ImportClause::Form::Unqualified);
        break;
      }

      if (check(TK::Dot) && check(TK::LCurly, 1)) {
        advance();
        advance();
        importClause->setForm(ImportClause::Form::Multiple);

        do {
          auto importedName = parseIdentifier();

          if (!importedName) {
            return nullptr;
          }

          importClause->addImportedName(*importedName);
        } while (accept(TK::Comma));

        if (!expect(TK::RCurly)) {
          return nullptr;
        }

        break;
      }

      if (!accept(TK::Dot)) {
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
  size_t start = index;
  auto externalClause = makeNode<ExternalClause>(start);

  if (check(TK::String)) {
    externalClause->setLanguage(advance().text);
  }

  if (check(TK::Identifier) || check(TK::Dot)) {
    if (!(check(TK::Identifier) && check(TK::LParen, 1))) {
      auto output = parseComponentReference();

      if (!output || !expect(TK::Equal)) {
        return nullptr;
      }

      externalClause->setOutput(std::move(output));
    }

    auto function = parseIdentifier();

    if (!function || !expect(TK::LParen)) {
      return nullptr;
    }

    externalClause->setFunction(*function);

    if (!check(TK::RParen)) {
      do {
        auto argument = parseExpression();

        if (!argument) {
          return nullptr;
        }

        externalClause->addArgument(std::move(argument));
      } while (accept(TK::Comma));
    }

    if (!expect(TK::RParen)) {
      return nullptr;
    }
  }

  if (check(TK::Annotation)) {
    auto annotation = parseAnnotation();

    if (!annotation) {
      return nullptr;
    }

    externalClause->setAnnotation(std::move(annotation));
  }

  if (!expect(TK::Semicolon)) {
    return nullptr;
  }

  finalize(*externalClause, start);
  return externalClause;
}

//===---------------------------------------------------------------------===//
// Extends and components
//===---------------------------------------------------------------------===//

std::unique_ptr<ExtendsClause> Parser::parseExtendsClause() {
  size_t start = index;

  if (!expect(TK::Extends)) {
    return nullptr;
  }

  auto extendsClause = makeNode<ExtendsClause>(start);
  auto baseClass = parseTypeSpecifier();

  if (!baseClass) {
    return nullptr;
  }

  extendsClause->setBaseClass(std::move(*baseClass));

  if (check(TK::LParen)) {
    auto modification = parseClassModification(true);

    if (!modification) {
      return nullptr;
    }

    extendsClause->setClassModification(std::move(modification));
  }

  if (check(TK::Annotation)) {
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
  size_t start = index;

  if (!expect(TK::ConstrainedBy)) {
    return nullptr;
  }

  auto constrainingClause = makeNode<ConstrainingClause>(start);
  auto typeSpecifier = parseTypeSpecifier();

  if (!typeSpecifier) {
    return nullptr;
  }

  constrainingClause->setTypeSpecifier(std::move(*typeSpecifier));

  if (check(TK::LParen)) {
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
  if (accept(TK::Flow)) {
    prefix.connectorKind = ConnectorKind::Flow;
  } else if (accept(TK::Stream)) {
    prefix.connectorKind = ConnectorKind::Stream;
  }

  if (accept(TK::Discrete)) {
    prefix.variability = Variability::Discrete;
  } else if (accept(TK::Parameter)) {
    prefix.variability = Variability::Parameter;
  } else if (accept(TK::Constant)) {
    prefix.variability = Variability::Constant;
  }

  if (accept(TK::Input)) {
    prefix.causality = Causality::Input;
  } else if (accept(TK::Output)) {
    prefix.causality = Causality::Output;
  }

  return true;
}

std::unique_ptr<ComponentClause>
Parser::parseComponentClause(const ElementPrefixes &prefixes,
                             bool singleDeclaration) {
  size_t start = index;
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

  if (check(TK::LSquare)) {
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
  } while (!singleDeclaration && accept(TK::Comma));

  finalize(*componentClause, start);
  return componentClause;
}

std::unique_ptr<ComponentDeclaration>
Parser::parseComponentDeclaration(bool allowCondition) {
  size_t start = index;
  auto declaration = makeNode<ComponentDeclaration>(start);
  auto name = parseIdentifier();

  if (!name) {
    return nullptr;
  }

  declaration->setName(*name);

  if (check(TK::LSquare)) {
    Children<Expression> subscripts;

    if (!parseArraySubscripts(subscripts)) {
      return nullptr;
    }

    for (Child<Expression> &subscript : subscripts) {
      declaration->addSubscript(std::unique_ptr<Expression>(
          static_cast<Expression *>(subscript->clone().release())));
    }
  }

  if (check(TK::LParen) || check(TK::Equal)) {
    auto modification = parseModification();

    if (!modification) {
      return nullptr;
    }

    declaration->setModification(std::move(modification));
  }

  if (allowCondition && accept(TK::If)) {
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
  size_t start = index;
  auto modification = makeNode<Modification>(start);

  if (check(TK::LParen)) {
    auto classModification = parseClassModification();

    if (!classModification) {
      return nullptr;
    }

    modification->setClassModification(std::move(classModification));
  }

  if (accept(TK::Equal)) {
    if (accept(TK::Break)) {
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
  size_t start = index;

  if (!expect(TK::LParen)) {
    return nullptr;
  }

  auto classModification = makeNode<ClassModification>(start);

  if (!check(TK::RParen)) {
    do {
      auto argument = parseArgument(allowInheritanceModifications);

      if (!argument) {
        return nullptr;
      }

      classModification->addArgument(std::move(argument));
    } while (accept(TK::Comma));
  }

  if (!expect(TK::RParen)) {
    return nullptr;
  }

  finalize(*classModification, start);
  return classModification;
}

std::unique_ptr<Argument>
Parser::parseArgument(bool allowInheritanceModifications) {
  if (allowInheritanceModifications && check(TK::Break)) {
    return parseInheritanceModification();
  }

  size_t start = index;
  bool isRedeclare = accept(TK::Redeclare);
  bool isEach = accept(TK::Each);
  bool isFinal = accept(TK::Final);

  if (isRedeclare || check(TK::Replaceable)) {
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

  if (check(TK::LParen) || check(TK::Equal)) {
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
  size_t start = index;
  auto redeclaration = makeNode<ElementRedeclaration>(start);
  redeclaration->setRedeclare(isRedeclare);
  redeclaration->setEach(isEach);
  redeclaration->setFinal(isFinal);
  redeclaration->setReplaceable(accept(TK::Replaceable));

  if (isClassDefinitionStart()) {
    size_t classStart = index;
    auto classDefinition = parseClassDefinition(ElementPrefixes());

    if (!classDefinition) {
      return nullptr;
    }

    if (classDefinition->getForm() == ClassDefinition::Form::Long ||
        classDefinition->getForm() == ClassDefinition::Form::ClassExtends) {
      emitError(tokens[classStart].location,
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

  if (redeclaration->isReplaceable() && check(TK::ConstrainedBy)) {
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
  size_t start = index;

  if (!expect(TK::Break)) {
    return nullptr;
  }

  auto modification = makeNode<InheritanceModification>(start);

  if (accept(TK::Connect)) {
    if (!expect(TK::LParen)) {
      return nullptr;
    }

    auto lhs = parseComponentReference();

    if (!lhs || !expect(TK::Comma)) {
      return nullptr;
    }

    auto rhs = parseComponentReference();

    if (!rhs || !expect(TK::RParen)) {
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
  size_t start = index;
  bool isInitial = accept(TK::Initial);

  if (!expect(TK::Equation)) {
    return nullptr;
  }

  auto section = makeNode<EquationSection>(start);
  section->setInitial(isInitial);

  while (!isEquationListEnd()) {
    auto equation = parseEquation();

    if (!equation || !expect(TK::Semicolon)) {
      return nullptr;
    }

    section->addEquation(std::move(equation));
  }

  finalize(*section, start);
  return section;
}

std::unique_ptr<AlgorithmSection> Parser::parseAlgorithmSection() {
  size_t start = index;
  bool isInitial = accept(TK::Initial);

  if (!expect(TK::Algorithm)) {
    return nullptr;
  }

  auto section = makeNode<AlgorithmSection>(start);
  section->setInitial(isInitial);

  while (!isEquationListEnd()) {
    auto statement = parseStatement();

    if (!statement || !expect(TK::Semicolon)) {
      return nullptr;
    }

    section->addStatement(std::move(statement));
  }

  finalize(*section, start);
  return section;
}

bool Parser::parseEquationList(Children<Equation> &equations,
                               llvm::ArrayRef<TokenKind> terminators) {
  while (!llvm::is_contained(terminators, peek().kind) && !isAtEnd()) {
    auto equation = parseEquation();

    if (!equation || !expect(TK::Semicolon)) {
      return false;
    }

    equations.emplace_back(std::move(equation));
  }

  return true;
}

bool Parser::parseStatementList(Children<Statement> &statements,
                                llvm::ArrayRef<TokenKind> terminators) {
  while (!llvm::is_contained(terminators, peek().kind) && !isAtEnd()) {
    auto statement = parseStatement();

    if (!statement || !expect(TK::Semicolon)) {
      return false;
    }

    statements.emplace_back(std::move(statement));
  }

  return true;
}

std::unique_ptr<Equation> Parser::parseEquation() {
  size_t start = index;
  std::unique_ptr<Equation> equation;

  if (check(TK::If)) {
    equation = parseIfEquation();
  } else if (check(TK::For)) {
    equation = parseForEquation();
  } else if (check(TK::When)) {
    equation = parseWhenEquation();
  } else if (check(TK::Connect)) {
    equation = parseConnectEquation();
  } else {
    auto lhs = parseSimpleExpression();

    if (!lhs) {
      return nullptr;
    }

    if (accept(TK::Equal)) {
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
  size_t start = index;
  auto equation = makeNode<IfEquation>(start);

  if (!expect(TK::If)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Equation> equations;

    if (!condition || !expect(TK::Then) ||
        !parseEquationList(equations, {TK::ElseIf, TK::Else, TK::End})) {
      return nullptr;
    }

    equation->addBranch(std::move(condition), std::move(equations));
  } while (accept(TK::ElseIf));

  if (accept(TK::Else)) {
    Children<Equation> equations;

    if (!parseEquationList(equations, {TK::End})) {
      return nullptr;
    }

    equation->setElseEquations(std::move(equations));
  }

  if (!expect(TK::End) || !expect(TK::If)) {
    return nullptr;
  }

  return equation;
}

std::unique_ptr<Equation> Parser::parseForEquation() {
  size_t start = index;
  auto equation = makeNode<ForEquation>(start);
  Children<ForIndex> indices;
  Children<Equation> equations;

  if (!expect(TK::For) || !parseForIndices(indices) || !expect(TK::Loop) ||
      !parseEquationList(equations, {TK::End}) || !expect(TK::End) ||
      !expect(TK::For)) {
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
  size_t start = index;
  auto equation = makeNode<WhenEquation>(start);

  if (!expect(TK::When)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Equation> equations;

    if (!condition || !expect(TK::Then) ||
        !parseEquationList(equations, {TK::ElseWhen, TK::End})) {
      return nullptr;
    }

    equation->addBranch(std::move(condition), std::move(equations));
  } while (accept(TK::ElseWhen));

  if (!expect(TK::End) || !expect(TK::When)) {
    return nullptr;
  }

  return equation;
}

std::unique_ptr<Equation> Parser::parseConnectEquation() {
  size_t start = index;
  auto equation = makeNode<ConnectEquation>(start);

  if (!expect(TK::Connect) || !expect(TK::LParen)) {
    return nullptr;
  }

  auto lhs = parseComponentReference();

  if (!lhs || !expect(TK::Comma)) {
    return nullptr;
  }

  auto rhs = parseComponentReference();

  if (!rhs || !expect(TK::RParen)) {
    return nullptr;
  }

  equation->setLhs(std::move(lhs));
  equation->setRhs(std::move(rhs));
  return equation;
}

std::unique_ptr<Statement> Parser::parseStatement() {
  size_t start = index;
  std::unique_ptr<Statement> statement;

  switch (peek().kind) {
  case TK::Break:
    advance();
    statement = makeNode<BreakStatement>(start);
    break;

  case TK::Return:
    advance();
    statement = makeNode<ReturnStatement>(start);
    break;

  case TK::If:
    statement = parseIfStatement();
    break;

  case TK::For:
    statement = parseForStatement();
    break;

  case TK::While:
    statement = parseWhileStatement();
    break;

  case TK::When:
    statement = parseWhenStatement();
    break;

  case TK::LParen: {
    // Assignment of multiple outputs.
    auto multipleAssignment = makeNode<MultipleAssignmentStatement>(start);
    advance();

    do {
      if (check(TK::Comma) || check(TK::RParen)) {
        multipleAssignment->addDestination(nullptr);
        continue;
      }

      auto destination = parseExpression();

      if (!destination) {
        return nullptr;
      }

      multipleAssignment->addDestination(std::move(destination));
    } while (accept(TK::Comma));

    if (!expect(TK::RParen) || !expect(TK::Assignment)) {
      return nullptr;
    }

    size_t callStart = index;
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

    if (accept(TK::Assignment)) {
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
  size_t start = index;
  auto statement = makeNode<IfStatement>(start);

  if (!expect(TK::If)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Statement> statements;

    if (!condition || !expect(TK::Then) ||
        !parseStatementList(statements, {TK::ElseIf, TK::Else, TK::End})) {
      return nullptr;
    }

    statement->addBranch(std::move(condition), std::move(statements));
  } while (accept(TK::ElseIf));

  if (accept(TK::Else)) {
    Children<Statement> statements;

    if (!parseStatementList(statements, {TK::End})) {
      return nullptr;
    }

    statement->setElseStatements(std::move(statements));
  }

  if (!expect(TK::End) || !expect(TK::If)) {
    return nullptr;
  }

  return statement;
}

std::unique_ptr<Statement> Parser::parseForStatement() {
  size_t start = index;
  auto statement = makeNode<ForStatement>(start);
  Children<ForIndex> indices;
  Children<Statement> statements;

  if (!expect(TK::For) || !parseForIndices(indices) || !expect(TK::Loop) ||
      !parseStatementList(statements, {TK::End}) || !expect(TK::End) ||
      !expect(TK::For)) {
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
  size_t start = index;
  auto statement = makeNode<WhileStatement>(start);

  if (!expect(TK::While)) {
    return nullptr;
  }

  auto condition = parseExpression();
  Children<Statement> statements;

  if (!condition || !expect(TK::Loop) ||
      !parseStatementList(statements, {TK::End}) || !expect(TK::End) ||
      !expect(TK::While)) {
    return nullptr;
  }

  statement->setCondition(std::move(condition));
  statement->setStatements(std::move(statements));
  return statement;
}

std::unique_ptr<Statement> Parser::parseWhenStatement() {
  size_t start = index;
  auto statement = makeNode<WhenStatement>(start);

  if (!expect(TK::When)) {
    return nullptr;
  }

  do {
    auto condition = parseExpression();
    Children<Statement> statements;

    if (!condition || !expect(TK::Then) ||
        !parseStatementList(statements, {TK::ElseWhen, TK::End})) {
      return nullptr;
    }

    statement->addBranch(std::move(condition), std::move(statements));
  } while (accept(TK::ElseWhen));

  if (!expect(TK::End) || !expect(TK::When)) {
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
  } while (accept(TK::Comma));

  return true;
}

std::unique_ptr<ForIndex> Parser::parseForIndex() {
  size_t start = index;
  auto forIndex = makeNode<ForIndex>(start);
  auto name = parseIdentifier();

  if (!name) {
    return nullptr;
  }

  forIndex->setName(*name);

  if (accept(TK::In)) {
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
  size_t start = index;

  if (!accept(TK::If)) {
    return parseSimpleExpression();
  }

  auto expression = makeNode<IfExpression>(start);

  do {
    auto condition = parseExpression();

    if (!condition || !expect(TK::Then)) {
      return nullptr;
    }

    auto value = parseExpression();

    if (!value) {
      return nullptr;
    }

    expression->addBranch(std::move(condition), std::move(value));
  } while (accept(TK::ElseIf));

  if (!expect(TK::Else)) {
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
  size_t start = index;
  auto first = parseLogicalExpression();

  if (!first || !accept(TK::Colon)) {
    return first;
  }

  auto second = parseLogicalExpression();

  if (!second) {
    return nullptr;
  }

  auto range = makeNode<Range>(start);
  range->setStart(std::move(first));

  if (accept(TK::Colon)) {
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
  size_t start = index;
  auto result = parseLogicalTerm();

  while (result && accept(TK::Or)) {
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
  size_t start = index;
  auto result = parseLogicalFactor();

  while (result && accept(TK::And)) {
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
  size_t start = index;

  if (!accept(TK::Not)) {
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
  size_t start = index;
  auto lhs = parseArithmeticExpression();

  if (!lhs) {
    return nullptr;
  }

  std::optional<BinaryOperator> op;

  switch (peek().kind) {
  case TK::Less:
    op = BinaryOperator::Less;
    break;
  case TK::LessEqual:
    op = BinaryOperator::LessEqual;
    break;
  case TK::Greater:
    op = BinaryOperator::Greater;
    break;
  case TK::GreaterEqual:
    op = BinaryOperator::GreaterEqual;
    break;
  case TK::EqualEqual:
    op = BinaryOperator::Equal;
    break;
  case TK::NotEqual:
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
  size_t start = index;
  std::optional<UnaryOperator> unaryOp;

  switch (peek().kind) {
  case TK::Plus:
    unaryOp = UnaryOperator::Plus;
    break;
  case TK::Minus:
    unaryOp = UnaryOperator::Minus;
    break;
  case TK::PlusEW:
    unaryOp = UnaryOperator::PlusEW;
    break;
  case TK::MinusEW:
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

    switch (peek().kind) {
    case TK::Plus:
      op = BinaryOperator::Add;
      break;
    case TK::Minus:
      op = BinaryOperator::Sub;
      break;
    case TK::PlusEW:
      op = BinaryOperator::AddEW;
      break;
    case TK::MinusEW:
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
  size_t start = index;
  auto result = parseFactor();

  while (result) {
    std::optional<BinaryOperator> op;

    switch (peek().kind) {
    case TK::Star:
      op = BinaryOperator::Mul;
      break;
    case TK::Slash:
      op = BinaryOperator::Div;
      break;
    case TK::StarEW:
      op = BinaryOperator::MulEW;
      break;
    case TK::SlashEW:
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
  size_t start = index;
  auto base = parsePrimary();

  if (!base) {
    return nullptr;
  }

  std::optional<BinaryOperator> op;

  if (check(TK::Caret)) {
    op = BinaryOperator::Pow;
  } else if (check(TK::CaretEW)) {
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
  size_t start = index;
  const Token &token = advance();

  // Unitful literal: the unit is a quoted identifier
  // following the number.
  std::optional<std::string> unit;

  if (check(TK::Identifier) && llvm::StringRef(peek().text).starts_with("'")) {
    llvm::StringRef quoted = advance().text;
    unit = quoted.drop_front().drop_back().str();
  }

  if (token.is(TK::Integer)) {
    auto literal = makeNode<IntegerLiteral>(start);
    literal->setValue(token.integerValue);
    literal->setUnit(std::move(unit));
    return literal;
  }

  auto literal = makeNode<RealLiteral>(start);
  literal->setValue(token.realValue);
  literal->setUnit(std::move(unit));
  return literal;
}

std::unique_ptr<Expression> Parser::parsePrimary() {
  size_t start = index;

  switch (peek().kind) {
  case TK::Integer:
  case TK::Real:
    return parseNumber();

  case TK::String: {
    auto literal = makeNode<StringLiteral>(start);
    literal->setValue(advance().text);
    finalize(*literal, start);
    return literal;
  }

  case TK::True:
  case TK::False: {
    auto literal = makeNode<BooleanLiteral>(start);
    literal->setValue(advance().is(TK::True));
    finalize(*literal, start);
    return literal;
  }

  case TK::Time:
    advance();
    return makeNode<Time>(start);

  case TK::End:
    if (subscriptsDepth == 0) {
      emitError(peek().location,
                "'end' can be used as expression only within array "
                "subscripts");

      return nullptr;
    }

    advance();
    return makeNode<End>(start);

  case TK::Der:
  case TK::Initial:
  case TK::Pure: {
    // Operators with function syntax.
    auto callee = makeNode<ComponentReference>(start);
    ComponentReference::Part part;
    part.name = toString(advance().kind).str();
    part.location = rangeFrom(start);
    callee->addPart(std::move(part));
    return parseFunctionCallArgs(std::move(callee), start);
  }

  case TK::Dot:
  case TK::Identifier: {
    auto reference = parseComponentReference();

    if (!reference) {
      return nullptr;
    }

    if (check(TK::LParen)) {
      return parseFunctionCallArgs(std::move(reference), start);
    }

    return reference;
  }

  case TK::LParen:
    return parseParenthesized();

  case TK::LSquare:
    return parseArrayConcatenation();

  case TK::LCurly:
    return parseArrayConstructor();

  default:
    emitUnexpectedTokenError("expression");
    return nullptr;
  }
}

std::unique_ptr<Expression> Parser::parseParenthesized() {
  size_t start = index;

  if (!expect(TK::LParen)) {
    return nullptr;
  }

  Children<Expression> elements;
  bool hasCommas = false;

  do {
    if (check(TK::Comma) || check(TK::RParen)) {
      elements.emplace_back(nullptr);
      continue;
    }

    auto element = parseExpression();

    if (!element) {
      return nullptr;
    }

    elements.emplace_back(std::move(element));
  } while (accept(TK::Comma) && (hasCommas = true));

  if (!expect(TK::RParen)) {
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

  if (check(TK::LSquare)) {
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

  if (check(TK::Dot) && check(TK::Identifier, 1)) {
    advance();
    auto memberAccess = makeNode<MemberAccess>(start);
    memberAccess->setBase(std::move(result));
    memberAccess->setMember(advance().text);
    finalize(*memberAccess, start);
    return memberAccess;
  }

  return result;
}

std::unique_ptr<Expression> Parser::parseArrayConcatenation() {
  size_t start = index;

  if (!expect(TK::LSquare)) {
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
    } while (accept(TK::Comma));

    concatenation->addRow(std::move(row));
  } while (accept(TK::Semicolon));

  if (!expect(TK::RSquare)) {
    return nullptr;
  }

  finalize(*concatenation, start);
  return concatenation;
}

std::unique_ptr<Expression> Parser::parseArrayConstructor() {
  size_t start = index;

  if (!expect(TK::LCurly)) {
    return nullptr;
  }

  auto constructor = makeNode<ArrayConstructor>(start);
  auto first = parseExpression();

  if (!first) {
    return nullptr;
  }

  constructor->addElement(std::move(first));

  if (accept(TK::For)) {
    Children<ForIndex> indices;

    if (!parseForIndices(indices)) {
      return nullptr;
    }

    for (Child<ForIndex> &forIndex : indices) {
      constructor->addIterator(std::unique_ptr<ForIndex>(
          static_cast<ForIndex *>(forIndex->clone().release())));
    }
  } else {
    while (accept(TK::Comma)) {
      auto element = parseExpression();

      if (!element) {
        return nullptr;
      }

      constructor->addElement(std::move(element));
    }
  }

  if (!expect(TK::RCurly)) {
    return nullptr;
  }

  finalize(*constructor, start);
  return constructor;
}

std::unique_ptr<ComponentReference> Parser::parseComponentReference() {
  size_t start = index;
  auto reference = makeNode<ComponentReference>(start);
  reference->setGlobal(accept(TK::Dot));

  do {
    size_t partStart = index;
    ComponentReference::Part part;
    auto name = parseIdentifier();

    if (!name) {
      return nullptr;
    }

    part.name = *name;

    if (check(TK::LSquare) && !parseArraySubscripts(part.subscripts)) {
      return nullptr;
    }

    part.location = rangeFrom(partStart);
    reference->addPart(std::move(part));
  } while (check(TK::Dot) && check(TK::Identifier, 1) && accept(TK::Dot));

  finalize(*reference, start);
  return reference;
}

bool Parser::isNamedArgumentStart() const {
  return check(TK::Identifier) && check(TK::Equal, 1);
}

std::unique_ptr<NamedArgument> Parser::parseNamedArgument() {
  size_t start = index;
  auto argument = makeNode<NamedArgument>(start);
  auto name = parseIdentifier();

  if (!name || !expect(TK::Equal)) {
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
  size_t start = index;

  if (!accept(TK::Function)) {
    return parseExpression();
  }

  // Function partial application.
  auto application = makeNode<FunctionPartialApplication>(start);
  auto function = parseTypeSpecifier();

  if (!function || !expect(TK::LParen)) {
    return nullptr;
  }

  application->setFunction(std::move(*function));

  if (!check(TK::RParen)) {
    do {
      auto argument = parseNamedArgument();

      if (!argument) {
        return nullptr;
      }

      application->addNamedArgument(std::move(argument));
    } while (accept(TK::Comma));
  }

  if (!expect(TK::RParen)) {
    return nullptr;
  }

  finalize(*application, start);
  return application;
}

std::unique_ptr<Call>
Parser::parseFunctionCallArgs(std::unique_ptr<ComponentReference> callee,
                              size_t startIndex) {
  auto call = makeNode<Call>(startIndex);
  call->setCallee(std::move(callee));

  if (!expect(TK::LParen)) {
    return nullptr;
  }

  if (!check(TK::RParen)) {
    bool namedArguments = isNamedArgumentStart();

    if (!namedArguments) {
      auto first = parseFunctionArgument();

      if (!first) {
        return nullptr;
      }

      bool isPartialApplication = first->isa<FunctionPartialApplication>();
      call->addArgument(std::move(first));

      if (!isPartialApplication && accept(TK::For)) {
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
        while (accept(TK::Comma)) {
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
      } while (accept(TK::Comma));
    }
  }

  if (!expect(TK::RParen)) {
    return nullptr;
  }

  finalize(*call, startIndex);
  return call;
}

bool Parser::parseArraySubscripts(Children<Expression> &subscripts) {
  if (!expect(TK::LSquare)) {
    return false;
  }

  ++subscriptsDepth;

  do {
    auto subscript = parseSubscript();

    if (!subscript) {
      --subscriptsDepth;
      return false;
    }

    subscripts.emplace_back(std::move(subscript));
  } while (accept(TK::Comma));

  --subscriptsDepth;
  return expect(TK::RSquare);
}

std::unique_ptr<Expression> Parser::parseSubscript() {
  size_t start = index;

  if (check(TK::Colon) && (check(TK::Comma, 1) || check(TK::RSquare, 1))) {
    advance();
    return makeNode<Colon>(start);
  }

  return parseExpression();
}
} // namespace marco::parser::modelica
