#include "Common.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::ast::modelica;
using namespace ::marco::parser::modelica;
using namespace ::marco::parser::modelica::test;

namespace {
/// Parse an expression, checking that the whole input is consumed.
std::unique_ptr<Expression> parseExpression(llvm::StringRef source) {
  StringParser parser(source);
  auto expression = parser->parseExpression();

  if (!expression || !parser->isAtEnd()) {
    return nullptr;
  }

  return expression;
}

/// Parse a file, checking that no error is reported.
std::unique_ptr<StoredDefinition> parseFile(llvm::StringRef source) {
  StringParser parser(source);
  auto result = parser->parseStoredDefinition();

  if (!parser.getErrors().empty()) {
    return nullptr;
  }

  return result;
}

/// Parse a single class.
std::unique_ptr<ClassDefinition> parseClass(llvm::StringRef source) {
  auto file = parseFile(source);

  if (!file || file->getClasses().size() != 1) {
    return nullptr;
  }

  return std::unique_ptr<ClassDefinition>(static_cast<ClassDefinition *>(
      file->getClasses()[0]->clone().release()));
}

/// Get the errors reported while parsing a file.
std::vector<std::string> getErrors(llvm::StringRef source) {
  StringParser parser(source);
  parser->parseStoredDefinition();
  return parser.getErrors();
}

/// Get the name of a component reference made of a single part.
std::string getName(const Expression *expression) {
  auto reference = expression->dyn_cast<ComponentReference>();

  if (!reference) {
    return "";
  }

  return reference->getName().toString();
}

/// Get the elements of a given kind of a class.
template <typename T>
std::vector<const T *> getElements(const ClassDefinition &classDefinition) {
  std::vector<const T *> result;

  for (const Child<Node> &element : classDefinition.getElements()) {
    if (auto casted = element->dyn_cast<T>()) {
      result.push_back(casted);
    }
  }

  return result;
}
} // namespace

//===---------------------------------------------------------------------===//
// Expressions
//===---------------------------------------------------------------------===//

TEST(ModelicaParser, literals) {
  auto integer = parseExpression("42");
  ASSERT_TRUE(integer && integer->isa<IntegerLiteral>());
  EXPECT_EQ(integer->cast<IntegerLiteral>()->getValue(), 42);

  auto real = parseExpression("1.5e2");
  ASSERT_TRUE(real && real->isa<RealLiteral>());
  EXPECT_DOUBLE_EQ(real->cast<RealLiteral>()->getValue(), 150);

  auto string = parseExpression(R"("text")");
  ASSERT_TRUE(string && string->isa<StringLiteral>());
  EXPECT_EQ(string->cast<StringLiteral>()->getValue(), "text");

  auto boolean = parseExpression("false");
  ASSERT_TRUE(boolean && boolean->isa<BooleanLiteral>());
  EXPECT_FALSE(boolean->cast<BooleanLiteral>()->getValue());

  auto time = parseExpression("time");
  ASSERT_TRUE(time && time->isa<Time>());
}

TEST(ModelicaParser, unitfulLiterals) {
  auto real = parseExpression("9.8'm/s2'");
  ASSERT_TRUE(real && real->isa<RealLiteral>());
  EXPECT_EQ(real->cast<RealLiteral>()->getUnit(), "m/s2");

  auto negated = parseExpression("-5 'cm'");
  ASSERT_TRUE(negated && negated->isa<UnaryOperation>());

  auto literal =
      negated->cast<UnaryOperation>()->getOperand()->dyn_cast<IntegerLiteral>();

  ASSERT_TRUE(literal);
  EXPECT_EQ(literal->getUnit(), "cm");
}

TEST(ModelicaParser, operatorPrecedence) {
  // a + b * c ^ d < e and not f or g
  auto expression = parseExpression("a + b * c ^ d < e and not f or g");
  ASSERT_TRUE(expression);

  auto orOp = expression->dyn_cast<BinaryOperation>();
  ASSERT_TRUE(orOp && orOp->getOperator() == BinaryOperator::Or);

  auto andOp = orOp->getLhs()->dyn_cast<BinaryOperation>();
  ASSERT_TRUE(andOp && andOp->getOperator() == BinaryOperator::And);

  auto notOp = andOp->getRhs()->dyn_cast<UnaryOperation>();
  ASSERT_TRUE(notOp && notOp->getOperator() == UnaryOperator::Not);

  auto less = andOp->getLhs()->dyn_cast<BinaryOperation>();
  ASSERT_TRUE(less && less->getOperator() == BinaryOperator::Less);

  auto add = less->getLhs()->dyn_cast<BinaryOperation>();
  ASSERT_TRUE(add && add->getOperator() == BinaryOperator::Add);

  auto mul = add->getRhs()->dyn_cast<BinaryOperation>();
  ASSERT_TRUE(mul && mul->getOperator() == BinaryOperator::Mul);

  auto pow = mul->getRhs()->dyn_cast<BinaryOperation>();
  ASSERT_TRUE(pow && pow->getOperator() == BinaryOperator::Pow);
}

TEST(ModelicaParser, unaryMinusAppliesToFirstTerm) {
  // -a^2 + b is (-(a^2)) + b.
  auto expression = parseExpression("-a^2 + b");
  ASSERT_TRUE(expression);

  auto add = expression->dyn_cast<BinaryOperation>();
  ASSERT_TRUE(add && add->getOperator() == BinaryOperator::Add);

  auto minus = add->getLhs()->dyn_cast<UnaryOperation>();
  ASSERT_TRUE(minus && minus->getOperator() == UnaryOperator::Minus);
  EXPECT_TRUE(minus->getOperand()->isa<BinaryOperation>());
}

TEST(ModelicaParser, leftAssociativity) {
  // a - b - c is (a - b) - c.
  auto expression = parseExpression("a - b - c");
  ASSERT_TRUE(expression);

  auto outer = expression->cast<BinaryOperation>();
  EXPECT_EQ(getName(outer->getRhs()), "c");
  EXPECT_TRUE(outer->getLhs()->isa<BinaryOperation>());
}

TEST(ModelicaParser, elementwiseOperators) {
  auto expression = parseExpression("a .* b ./ c .+ d .^ e");
  ASSERT_TRUE(expression);

  auto addEW = expression->cast<BinaryOperation>();
  EXPECT_EQ(addEW->getOperator(), BinaryOperator::AddEW);
  EXPECT_EQ(addEW->getRhs()->cast<BinaryOperation>()->getOperator(),
            BinaryOperator::PowEW);
}

TEST(ModelicaParser, ranges) {
  auto simple = parseExpression("1:n");
  ASSERT_TRUE(simple && simple->isa<Range>());
  EXPECT_FALSE(simple->cast<Range>()->getStep());

  // 1:2:n has start 1, step 2 and stop n.
  auto stepped = parseExpression("1:2:n");
  ASSERT_TRUE(stepped && stepped->isa<Range>());
  auto range = stepped->cast<Range>();
  ASSERT_TRUE(range->getStep());
  EXPECT_EQ(range->getStep()->cast<IntegerLiteral>()->getValue(), 2);
  EXPECT_EQ(getName(range->getStop()), "n");
}

TEST(ModelicaParser, ifExpression) {
  auto expression = parseExpression("if a then 1 elseif b then 2 else 3");
  ASSERT_TRUE(expression && expression->isa<IfExpression>());

  auto ifExpression = expression->cast<IfExpression>();
  EXPECT_EQ(ifExpression->getConditions().size(), 2u);
  EXPECT_EQ(ifExpression->getElseValue()->cast<IntegerLiteral>()->getValue(),
            3);
}

TEST(ModelicaParser, componentReferences) {
  auto expression = parseExpression(".a.b[1, :].c[end - 1]");
  ASSERT_TRUE(expression && expression->isa<ComponentReference>());

  auto reference = expression->cast<ComponentReference>();
  EXPECT_TRUE(reference->isGlobal());
  ASSERT_EQ(reference->getParts().size(), 3u);
  EXPECT_EQ(reference->getName().toString(), ".a.b.c");

  const auto &subscripts = reference->getParts()[1].subscripts;
  ASSERT_EQ(subscripts.size(), 2u);
  EXPECT_TRUE(subscripts[0]->isa<IntegerLiteral>());
  EXPECT_TRUE(subscripts[1]->isa<Colon>());

  auto endMinusOne =
      reference->getParts()[2].subscripts[0]->dyn_cast<BinaryOperation>();

  ASSERT_TRUE(endMinusOne);
  EXPECT_TRUE(endMinusOne->getLhs()->isa<End>());
}

TEST(ModelicaParser, colonVersusRange) {
  // A colon followed by an expression is a range, not a colon subscript.
  auto expression = parseExpression("a[2:end, :]");
  ASSERT_TRUE(expression);

  const auto &subscripts =
      expression->cast<ComponentReference>()->getParts()[0].subscripts;

  ASSERT_EQ(subscripts.size(), 2u);
  EXPECT_TRUE(subscripts[0]->isa<Range>());
  EXPECT_TRUE(subscripts[1]->isa<Colon>());
}

TEST(ModelicaParser, endOutsideSubscripts) {
  StringParser parser("end");
  EXPECT_FALSE(parser->parseExpression());
  ASSERT_EQ(parser.getErrors().size(), 1u);
  EXPECT_EQ(parser.getErrors()[0],
            "'end' can be used as expression only within array subscripts");
}

TEST(ModelicaParser, calls) {
  auto expression = parseExpression("Modelica.Math.sin(x, y + 1, scale = 2)");
  ASSERT_TRUE(expression && expression->isa<Call>());

  auto call = expression->cast<Call>();
  EXPECT_EQ(call->getCallee()->getName().toString(), "Modelica.Math.sin");
  EXPECT_EQ(call->getArguments().size(), 2u);
  ASSERT_EQ(call->getNamedArguments().size(), 1u);
  EXPECT_EQ(call->getNamedArguments()[0]->getName(), "scale");
}

TEST(ModelicaParser, operatorsWithFunctionSyntax) {
  for (llvm::StringRef name : {"der", "initial", "pure"}) {
    auto expression = parseExpression((name + "(x)").str());
    ASSERT_TRUE(expression && expression->isa<Call>()) << name.str();
    EXPECT_EQ(expression->cast<Call>()->getCallee()->getName().toString(),
              name);
  }
}

TEST(ModelicaParser, reductionWithIterators) {
  auto expression = parseExpression("sum(x[i] * y[j] for i in 1:n, j)");
  ASSERT_TRUE(expression && expression->isa<Call>());

  auto call = expression->cast<Call>();
  EXPECT_EQ(call->getArguments().size(), 1u);
  ASSERT_EQ(call->getIterators().size(), 2u);
  EXPECT_EQ(call->getIterators()[0]->getName(), "i");
  EXPECT_TRUE(call->getIterators()[0]->getRange());

  // Implicit iteration range.
  EXPECT_EQ(call->getIterators()[1]->getName(), "j");
  EXPECT_FALSE(call->getIterators()[1]->getRange());
}

TEST(ModelicaParser, functionPartialApplication) {
  auto expression = parseExpression("quadrature(0, 1, function f(a = 2))");
  ASSERT_TRUE(expression && expression->isa<Call>());

  const auto &arguments = expression->cast<Call>()->getArguments();
  ASSERT_EQ(arguments.size(), 3u);

  auto application = arguments[2]->dyn_cast<FunctionPartialApplication>();
  ASSERT_TRUE(application);
  EXPECT_EQ(application->getFunction().toString(), "f");
  EXPECT_EQ(application->getNamedArguments().size(), 1u);
}

TEST(ModelicaParser, arrayConstructors) {
  auto elements = parseExpression("{1, 2, 3}");
  ASSERT_TRUE(elements && elements->isa<ArrayConstructor>());
  EXPECT_EQ(elements->cast<ArrayConstructor>()->getElements().size(), 3u);

  auto iterators = parseExpression("{i * j for i in 1:3, j in 1:2}");
  ASSERT_TRUE(iterators && iterators->isa<ArrayConstructor>());
  EXPECT_EQ(iterators->cast<ArrayConstructor>()->getIterators().size(), 2u);
}

TEST(ModelicaParser, arrayConcatenation) {
  auto expression = parseExpression("[1, 2; 3, 4; 5, 6]");
  ASSERT_TRUE(expression && expression->isa<ArrayConcatenation>());

  const auto &rows = expression->cast<ArrayConcatenation>()->getRows();
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows[0].size(), 2u);
}

TEST(ModelicaParser, parenthesizedExpressions) {
  // A single parenthesized expression is not an output list.
  auto parenthesized = parseExpression("(a + b) * c");
  ASSERT_TRUE(parenthesized && parenthesized->isa<BinaryOperation>());
  EXPECT_EQ(parenthesized->cast<BinaryOperation>()->getOperator(),
            BinaryOperator::Mul);

  auto subscripted = parseExpression("(a * b)[1, :]");
  ASSERT_TRUE(subscripted && subscripted->isa<Subscripted>());
  EXPECT_EQ(subscripted->cast<Subscripted>()->getSubscripts().size(), 2u);

  auto member = parseExpression("(f(x)).re");
  ASSERT_TRUE(member && member->isa<MemberAccess>());
  EXPECT_EQ(member->cast<MemberAccess>()->getMember(), "re");

  auto outputList = parseExpression("(a, , b)");
  ASSERT_TRUE(outputList && outputList->isa<OutputList>());

  const auto &outputs = outputList->cast<OutputList>()->getElements();
  ASSERT_EQ(outputs.size(), 3u);
  EXPECT_TRUE(outputs[0]);
  EXPECT_FALSE(outputs[1]);
  EXPECT_TRUE(outputs[2]);
}

//===---------------------------------------------------------------------===//
// Classes
//===---------------------------------------------------------------------===//

TEST(ModelicaParser, withinClause) {
  auto file = parseFile("within Modelica.Blocks; model M end M;");
  ASSERT_TRUE(file && file->getWithin());
  EXPECT_EQ(file->getWithin()->toString(), "Modelica.Blocks");

  auto topLevel = parseFile("within; model M end M;");
  ASSERT_TRUE(topLevel && topLevel->getWithin());
  EXPECT_TRUE(topLevel->getWithin()->parts.empty());

  auto none = parseFile("model M end M;");
  ASSERT_TRUE(none);
  EXPECT_FALSE(none->getWithin());
}

TEST(ModelicaParser, classRestrictions) {
  struct Case {
    llvm::StringRef prefixes;
    ClassRestriction restriction;
    Purity purity;
  };

  Case cases[] = {
      {"class", ClassRestriction::Class, Purity::Unspecified},
      {"model", ClassRestriction::Model, Purity::Unspecified},
      {"record", ClassRestriction::Record, Purity::Unspecified},
      {"operator record", ClassRestriction::OperatorRecord,
       Purity::Unspecified},
      {"block", ClassRestriction::Block, Purity::Unspecified},
      {"connector", ClassRestriction::Connector, Purity::Unspecified},
      {"expandable connector", ClassRestriction::ExpandableConnector,
       Purity::Unspecified},
      {"type", ClassRestriction::Type, Purity::Unspecified},
      {"package", ClassRestriction::Package, Purity::Unspecified},
      {"function", ClassRestriction::Function, Purity::Unspecified},
      {"pure function", ClassRestriction::Function, Purity::Pure},
      {"impure function", ClassRestriction::Function, Purity::Impure},
      {"operator", ClassRestriction::Operator, Purity::Unspecified},
      {"operator function", ClassRestriction::OperatorFunction,
       Purity::Unspecified},
      {"impure operator function", ClassRestriction::OperatorFunction,
       Purity::Impure}};

  for (const Case &c : cases) {
    std::string source = (c.prefixes + " C end C;").str();
    auto classDefinition = parseClass(source);
    ASSERT_TRUE(classDefinition) << source;
    EXPECT_EQ(classDefinition->getRestriction(), c.restriction) << source;
    EXPECT_EQ(classDefinition->getPurity(), c.purity) << source;
  }
}

TEST(ModelicaParser, classPrefixes) {
  auto file = parseFile("final encapsulated partial model M end M;");
  ASSERT_TRUE(file);

  const ClassDefinition &classDefinition = *file->getClasses()[0];
  EXPECT_TRUE(classDefinition.getPrefixes().isFinal);
  EXPECT_TRUE(classDefinition.isEncapsulated());
  EXPECT_TRUE(classDefinition.isPartial());
}

TEST(ModelicaParser, longClassDefinition) {
  auto classDefinition = parseClass(R"(
    model M "A model"
      Real x;
    protected
      Real y;
    public
      Real z;
    equation
      x = 1;
    initial equation
      y = 0;
    algorithm
      z := 1;
    initial algorithm
      z := 0;
      annotation(experiment(StopTime = 1));
    end M;)");

  ASSERT_TRUE(classDefinition);
  EXPECT_EQ(classDefinition->getForm(), ClassDefinition::Form::Long);
  EXPECT_EQ(classDefinition->getDescription().text, "A model");
  EXPECT_TRUE(classDefinition->getDescription().annotation);

  auto components = getElements<ComponentClause>(*classDefinition);
  ASSERT_EQ(components.size(), 3u);
  EXPECT_FALSE(components[0]->getPrefixes().isProtected);
  EXPECT_TRUE(components[1]->getPrefixes().isProtected);
  EXPECT_FALSE(components[2]->getPrefixes().isProtected);

  auto equationSections = getElements<EquationSection>(*classDefinition);
  ASSERT_EQ(equationSections.size(), 2u);
  EXPECT_FALSE(equationSections[0]->isInitial());
  EXPECT_TRUE(equationSections[1]->isInitial());

  auto algorithmSections = getElements<AlgorithmSection>(*classDefinition);
  ASSERT_EQ(algorithmSections.size(), 2u);
  EXPECT_TRUE(algorithmSections[1]->isInitial());
}

TEST(ModelicaParser, mismatchingEndName) {
  auto errors = getErrors("model M end N;");
  ASSERT_FALSE(errors.empty());
  EXPECT_EQ(errors[0],
            "the name 'N' doesn't match the name of the class 'M'");
}

TEST(ModelicaParser, shortClassDefinition) {
  auto classDefinition =
      parseClass(R"(connector RealInput = input Real[3](unit = "V") "Input";)");

  ASSERT_TRUE(classDefinition);
  EXPECT_EQ(classDefinition->getForm(), ClassDefinition::Form::Short);
  EXPECT_EQ(classDefinition->getBasePrefix(), Causality::Input);
  EXPECT_EQ(classDefinition->getBaseClass().toString(), "Real");
  EXPECT_EQ(classDefinition->getSubscripts().size(), 1u);
  EXPECT_TRUE(classDefinition->getClassModification());
  EXPECT_EQ(classDefinition->getDescription().text, "Input");
}

TEST(ModelicaParser, enumerations) {
  auto classDefinition =
      parseClass(R"(type E = enumeration(a "First", b, c) "Enum";)");

  ASSERT_TRUE(classDefinition);
  EXPECT_EQ(classDefinition->getForm(), ClassDefinition::Form::Enumeration);

  const auto &literals = classDefinition->getEnumerationLiterals();
  ASSERT_EQ(literals.size(), 3u);
  EXPECT_EQ(literals[0]->getName(), "a");
  EXPECT_EQ(literals[0]->getDescription().text, "First");

  auto unspecified = parseClass("type F = enumeration(:);");
  ASSERT_TRUE(unspecified);
  EXPECT_TRUE(unspecified->isUnspecifiedEnumeration());
}

TEST(ModelicaParser, derivativeDefinition) {
  auto classDefinition = parseClass("function df = der(f, x, y);");
  ASSERT_TRUE(classDefinition);
  EXPECT_EQ(classDefinition->getForm(), ClassDefinition::Form::Derivative);
  EXPECT_EQ(classDefinition->getDerivedFunction().toString(), "f");

  auto args = classDefinition->getDerivativeArguments();
  ASSERT_EQ(args.size(), 2u);
  EXPECT_EQ(args[0], "x");
  EXPECT_EQ(args[1], "y");
}

TEST(ModelicaParser, classExtends) {
  auto classDefinition = parseClass(R"(
    package P
      redeclare replaceable model extends Medium(nX = 2) "Extended"
        Real x;
      end Medium;
    end P;)");

  ASSERT_TRUE(classDefinition);
  auto nested = getElements<ClassDefinition>(*classDefinition);
  ASSERT_EQ(nested.size(), 1u);

  EXPECT_EQ(nested[0]->getForm(), ClassDefinition::Form::ClassExtends);
  EXPECT_TRUE(nested[0]->getPrefixes().isRedeclare);
  EXPECT_TRUE(nested[0]->getPrefixes().isReplaceable);
  EXPECT_TRUE(nested[0]->getClassModification());
  EXPECT_EQ(nested[0]->getElements().size(), 1u);
}

TEST(ModelicaParser, componentClauses) {
  auto classDefinition = parseClass(R"(
    model M
      inner outer flow discrete input Real[3] a, b[2](start = 1) = c if d "B";
      replaceable Water medium constrainedby PartialMedium(p = 1) "Medium";
    end M;)");

  ASSERT_TRUE(classDefinition);
  auto components = getElements<ComponentClause>(*classDefinition);
  ASSERT_EQ(components.size(), 2u);

  const ComponentClause &clause = *components[0];
  EXPECT_TRUE(clause.getPrefixes().isInner);
  EXPECT_TRUE(clause.getPrefixes().isOuter);
  EXPECT_EQ(clause.getTypePrefix().connectorKind, ConnectorKind::Flow);
  EXPECT_EQ(clause.getTypePrefix().variability, Variability::Discrete);
  EXPECT_EQ(clause.getTypePrefix().causality, Causality::Input);
  EXPECT_EQ(clause.getTypeSpecifier().toString(), "Real");
  EXPECT_EQ(clause.getSubscripts().size(), 1u);
  ASSERT_EQ(clause.getDeclarations().size(), 2u);

  const ComponentDeclaration &b = *clause.getDeclarations()[1];
  EXPECT_EQ(b.getName(), "b");
  EXPECT_EQ(b.getSubscripts().size(), 1u);
  ASSERT_TRUE(b.getModification());
  EXPECT_TRUE(b.getModification()->getClassModification());
  EXPECT_TRUE(b.getModification()->getValue());
  EXPECT_TRUE(b.getCondition());
  EXPECT_EQ(b.getDescription().text, "B");

  const ComponentClause &medium = *components[1];
  EXPECT_TRUE(medium.getPrefixes().isReplaceable);
  ASSERT_TRUE(medium.getConstrainingClause());
  EXPECT_EQ(medium.getConstrainingClause()->getTypeSpecifier().toString(),
            "PartialMedium");
  EXPECT_EQ(medium.getConstrainingClause()->getDescription(), "Medium");
}

TEST(ModelicaParser, modifications) {
  auto classDefinition = parseClass(R"(
    model M
      R r(each final x = 1, a.b(c = 2), d = break,
          redeclare Water medium, redeclare each replaceable model C = D,
          replaceable Real y constrainedby Real);
    end M;)");

  ASSERT_TRUE(classDefinition);
  auto components = getElements<ComponentClause>(*classDefinition);
  ASSERT_EQ(components.size(), 1u);

  const auto &arguments = components[0]
                              ->getDeclarations()[0]
                              ->getModification()
                              ->getClassModification()
                              ->getArguments();

  ASSERT_EQ(arguments.size(), 6u);

  auto x = arguments[0]->dyn_cast<ElementModification>();
  ASSERT_TRUE(x);
  EXPECT_TRUE(x->isEach());
  EXPECT_TRUE(x->isFinal());

  auto ab = arguments[1]->dyn_cast<ElementModification>();
  ASSERT_TRUE(ab);
  EXPECT_EQ(ab->getName().toString(), "a.b");

  auto d = arguments[2]->dyn_cast<ElementModification>();
  ASSERT_TRUE(d && d->getModification());
  EXPECT_TRUE(d->getModification()->isBreak());

  auto medium = arguments[3]->dyn_cast<ElementRedeclaration>();
  ASSERT_TRUE(medium);
  EXPECT_TRUE(medium->isRedeclare());
  EXPECT_FALSE(medium->isReplaceable());
  EXPECT_TRUE(medium->getElement()->isa<ComponentClause>());

  auto c = arguments[4]->dyn_cast<ElementRedeclaration>();
  ASSERT_TRUE(c);
  EXPECT_TRUE(c->isRedeclare());
  EXPECT_TRUE(c->isEach());
  EXPECT_TRUE(c->isReplaceable());
  ASSERT_TRUE(c->getElement()->isa<ClassDefinition>());
  EXPECT_EQ(c->getElement()->cast<ClassDefinition>()->getForm(),
            ClassDefinition::Form::Short);

  auto y = arguments[5]->dyn_cast<ElementRedeclaration>();
  ASSERT_TRUE(y);
  EXPECT_FALSE(y->isRedeclare());
  EXPECT_TRUE(y->isReplaceable());
  EXPECT_TRUE(y->getConstrainingClause());
}

TEST(ModelicaParser, longClassWithinModification) {
  auto errors =
      getErrors("model M R r(redeclare model C Real x; end C); end M;");

  ASSERT_FALSE(errors.empty());
  EXPECT_EQ(errors[0],
            "only short class definitions are allowed within modifications");
}

TEST(ModelicaParser, extendsClauses) {
  auto classDefinition = parseClass(R"(
    model M
      extends Base(x = 1, break friction, break connect(a.b, c))
        annotation(IconMap(primitivesVisible = false));
    protected
      extends .Other;
    end M;)");

  ASSERT_TRUE(classDefinition);
  auto extendsClauses = getElements<ExtendsClause>(*classDefinition);
  ASSERT_EQ(extendsClauses.size(), 2u);

  const ExtendsClause &base = *extendsClauses[0];
  EXPECT_EQ(base.getBaseClass().toString(), "Base");
  EXPECT_TRUE(base.getAnnotation());

  const auto &arguments = base.getClassModification()->getArguments();
  ASSERT_EQ(arguments.size(), 3u);

  // Selective model extension.
  auto friction = arguments[1]->dyn_cast<InheritanceModification>();
  ASSERT_TRUE(friction);
  EXPECT_FALSE(friction->isConnect());
  EXPECT_EQ(friction->getComponent(), "friction");

  auto connection = arguments[2]->dyn_cast<InheritanceModification>();
  ASSERT_TRUE(connection);
  EXPECT_TRUE(connection->isConnect());
  EXPECT_EQ(connection->getConnectLhs()->getName().toString(), "a.b");

  EXPECT_TRUE(extendsClauses[1]->isProtected());
  EXPECT_TRUE(extendsClauses[1]->getBaseClass().global);
}

TEST(ModelicaParser, breakOutsideExtends) {
  auto errors = getErrors("model M R r(break x); end M;");
  ASSERT_FALSE(errors.empty());
}

TEST(ModelicaParser, importClauses) {
  auto classDefinition = parseClass(R"(
    package P
      import A.B.C;
      import X = A.B;
      import A.B.*;
      import A.B.{C, D};
    end P;)");

  ASSERT_TRUE(classDefinition);
  auto imports = getElements<ImportClause>(*classDefinition);
  ASSERT_EQ(imports.size(), 4u);

  EXPECT_EQ(imports[0]->getForm(), ImportClause::Form::Qualified);
  EXPECT_EQ(imports[0]->getName().toString(), "A.B.C");

  EXPECT_EQ(imports[1]->getForm(), ImportClause::Form::Renaming);
  EXPECT_EQ(imports[1]->getAlias(), "X");
  EXPECT_EQ(imports[1]->getName().toString(), "A.B");

  EXPECT_EQ(imports[2]->getForm(), ImportClause::Form::Unqualified);
  EXPECT_EQ(imports[2]->getName().toString(), "A.B");

  EXPECT_EQ(imports[3]->getForm(), ImportClause::Form::Multiple);
  EXPECT_EQ(imports[3]->getName().toString(), "A.B");
  ASSERT_EQ(imports[3]->getImportedNames().size(), 2u);
  EXPECT_EQ(imports[3]->getImportedNames()[1], "D");
}

TEST(ModelicaParser, externalClauses) {
  auto withCall = parseClass(R"(
    function f
      input Real x[:];
      output Real y;
    external "C" y = f_impl(x, size(x, 1)) annotation(Library = "lib");
    end f;)");

  ASSERT_TRUE(withCall && withCall->getExternalClause());

  const ExternalClause &clause = *withCall->getExternalClause();
  EXPECT_EQ(clause.getLanguage(), "C");
  EXPECT_EQ(clause.getFunction(), "f_impl");
  ASSERT_TRUE(clause.getOutput());
  EXPECT_EQ(clause.getOutput()->getName().toString(), "y");
  EXPECT_EQ(clause.getArguments().size(), 2u);
  EXPECT_TRUE(clause.getAnnotation());

  auto withoutOutput = parseClass("function g external \"C\" g_impl(); end g;");
  ASSERT_TRUE(withoutOutput && withoutOutput->getExternalClause());
  EXPECT_FALSE(withoutOutput->getExternalClause()->getOutput());
  EXPECT_TRUE(withoutOutput->getExternalClause()->hasFunctionCall());

  auto defaultMapping = parseClass("function h external; end h;");
  ASSERT_TRUE(defaultMapping && defaultMapping->getExternalClause());
  EXPECT_FALSE(defaultMapping->getExternalClause()->getLanguage());
  EXPECT_FALSE(defaultMapping->getExternalClause()->hasFunctionCall());
}

//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

namespace {
/// Parse the equations of a model.
std::vector<const Equation *> parseEquations(llvm::StringRef equations,
                                             std::unique_ptr<ClassDefinition> &owner) {
  owner = parseClass(("model M equation " + equations + " end M;").str());

  if (!owner) {
    return {};
  }

  auto sections = getElements<EquationSection>(*owner);
  std::vector<const Equation *> result;

  for (const Child<Equation> &equation : sections[0]->getEquations()) {
    result.push_back(equation.get());
  }

  return result;
}
} // namespace

TEST(ModelicaParser, simpleEquations) {
  std::unique_ptr<ClassDefinition> owner;
  auto equations = parseEquations(R"(
    x = 1 "description";
    (a, , b) = f(x);
    assert(x > 0, "message");
    der(x) = -x;)",
                                   owner);

  ASSERT_EQ(equations.size(), 4u);

  auto first = equations[0]->dyn_cast<EqualityEquation>();
  ASSERT_TRUE(first);
  EXPECT_EQ(getName(first->getLhs()), "x");
  EXPECT_EQ(first->getDescription().text, "description");

  auto tuple = equations[1]->dyn_cast<EqualityEquation>();
  ASSERT_TRUE(tuple && tuple->getLhs()->isa<OutputList>());

  auto call = equations[2]->dyn_cast<CallEquation>();
  ASSERT_TRUE(call);
  EXPECT_EQ(call->getCall()->getCallee()->getName().toString(), "assert");

  auto derivative = equations[3]->dyn_cast<EqualityEquation>();
  ASSERT_TRUE(derivative && derivative->getLhs()->isa<Call>());
}

TEST(ModelicaParser, structuredEquations) {
  std::unique_ptr<ClassDefinition> owner;
  auto equations = parseEquations(R"(
    for i in 1:n, j in 1:m loop
      connect(a[i].p, b[j].n) annotation(Line(points = {{0, 0}, {1, 1}}));
    end for;
    if c then x = 1; elseif d then x = 2; else x = 3; end if;
    when sample(0, 1) then reinit(x, 0); elsewhen initial() then y = 1; end when;)",
                                   owner);

  ASSERT_EQ(equations.size(), 3u);

  auto forEquation = equations[0]->dyn_cast<ForEquation>();
  ASSERT_TRUE(forEquation);
  EXPECT_EQ(forEquation->getIndices().size(), 2u);
  ASSERT_EQ(forEquation->getEquations().size(), 1u);

  auto connect = forEquation->getEquations()[0]->dyn_cast<ConnectEquation>();
  ASSERT_TRUE(connect);
  EXPECT_EQ(connect->getLhs()->getName().toString(), "a.p");
  EXPECT_TRUE(connect->getDescription().annotation);

  auto ifEquation = equations[1]->dyn_cast<IfEquation>();
  ASSERT_TRUE(ifEquation);
  EXPECT_EQ(ifEquation->getConditions().size(), 2u);
  EXPECT_EQ(ifEquation->getElseEquations().size(), 1u);

  auto whenEquation = equations[2]->dyn_cast<WhenEquation>();
  ASSERT_TRUE(whenEquation);
  EXPECT_EQ(whenEquation->getConditions().size(), 2u);
  EXPECT_TRUE(whenEquation->getBranches()[0][0]->isa<CallEquation>());
}

TEST(ModelicaParser, invalidEquation) {
  auto errors = getErrors("model M equation x; end M;");
  ASSERT_FALSE(errors.empty());
  EXPECT_EQ(errors[0], "expected '=', but found ';'");
}

//===---------------------------------------------------------------------===//
// Statements
//===---------------------------------------------------------------------===//

TEST(ModelicaParser, statements) {
  auto classDefinition = parseClass(R"(
    function f
    algorithm
      x := 1;
      x[i] := 2;
      der(y) := 3;
      (a, , b) := g(x);
      print("message");
      for i in 1:10 loop
        if i > 5 then break; elseif i < 0 then return; end if;
      end for;
      while x < 10 loop x := x + 1; end while;
      when x > 2 then y := 1; elsewhen x < 0 then y := 2; end when;
    end f;)");

  ASSERT_TRUE(classDefinition);
  auto sections = getElements<AlgorithmSection>(*classDefinition);
  ASSERT_EQ(sections.size(), 1u);

  const auto &statements = sections[0]->getStatements();
  ASSERT_EQ(statements.size(), 8u);

  EXPECT_TRUE(statements[0]->isa<AssignmentStatement>());
  EXPECT_TRUE(statements[1]
                  ->cast<AssignmentStatement>()
                  ->getDestination()
                  ->isa<ComponentReference>());

  EXPECT_TRUE(
      statements[2]->cast<AssignmentStatement>()->getDestination()->isa<Call>());

  auto multiple = statements[3]->dyn_cast<MultipleAssignmentStatement>();
  ASSERT_TRUE(multiple);
  ASSERT_EQ(multiple->getDestinations().size(), 3u);
  EXPECT_FALSE(multiple->getDestinations()[1]);
  EXPECT_EQ(multiple->getCall()->getCallee()->getName().toString(), "g");

  EXPECT_TRUE(statements[4]->isa<CallStatement>());

  auto forStatement = statements[5]->dyn_cast<ForStatement>();
  ASSERT_TRUE(forStatement);

  auto ifStatement = forStatement->getStatements()[0]->dyn_cast<IfStatement>();
  ASSERT_TRUE(ifStatement);
  EXPECT_TRUE(ifStatement->getBranches()[0][0]->isa<BreakStatement>());
  EXPECT_TRUE(ifStatement->getBranches()[1][0]->isa<ReturnStatement>());

  EXPECT_TRUE(statements[6]->isa<WhileStatement>());

  auto whenStatement = statements[7]->dyn_cast<WhenStatement>();
  ASSERT_TRUE(whenStatement);
  EXPECT_EQ(whenStatement->getBranches().size(), 2u);
}

TEST(ModelicaParser, invalidStatement) {
  auto errors = getErrors("function f algorithm x = 1; end f;");
  ASSERT_FALSE(errors.empty());
  EXPECT_EQ(errors[0], "expected ':=', but found '='");
}

//===---------------------------------------------------------------------===//
// Errors and AST utilities
//===---------------------------------------------------------------------===//

TEST(ModelicaParser, missingSemicolon) {
  auto errors = getErrors("model M Real x end M;");
  ASSERT_FALSE(errors.empty());
  EXPECT_EQ(errors[0], "expected ';', but found 'end'");
}

TEST(ModelicaParser, lexerErrorsAreReported) {
  auto errors = getErrors("model M Real x = \"abc; end M;");
  ASSERT_FALSE(errors.empty());
  EXPECT_EQ(errors[0], "unterminated string");
}

TEST(ModelicaParser, parentsAndClone) {
  auto file = parseFile("model M Real x = 1 + 2; end M;");
  ASSERT_TRUE(file);

  const ClassDefinition &classDefinition = *file->getClasses()[0];
  EXPECT_EQ(classDefinition.getParent(), file.get());

  auto clone = std::unique_ptr<ClassDefinition>(
      static_cast<ClassDefinition *>(classDefinition.clone().release()));

  // The children of the clone refer to the clone as parent.
  ASSERT_EQ(clone->getElements().size(), 1u);
  EXPECT_EQ(clone->getElements()[0]->getParent(), clone.get());
  EXPECT_EQ(llvm::json::Value(clone->toJSON()),
            llvm::json::Value(classDefinition.toJSON()));
}

TEST(ModelicaParser, locations) {
  auto file = parseFile("model M\n  Real x = 1 + 2;\nend M;");
  ASSERT_TRUE(file);

  auto components = getElements<ComponentClause>(*file->getClasses()[0]);
  ASSERT_EQ(components.size(), 1u);

  SourceRange location = components[0]->getLocation();
  EXPECT_EQ(location.begin.line, 2);
  EXPECT_EQ(location.begin.column, 3);
  EXPECT_EQ(location.end.line, 2);
  EXPECT_EQ(location.end.column, 16);
}
