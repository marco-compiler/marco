#include "gtest/gtest.h"

#include "modelica/Parser.hpp"

using namespace modelica;
TEST(ParserTest, longClassDefinition)
{
	auto parser = Parser("encapsulated class Test\"comment\" public end Test");

	auto eq = parser.classDefinition();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<LongClassDecl>(ptr));
	auto casted = llvm::cast<LongClassDecl>(ptr);
	EXPECT_EQ(true, llvm::isa<Composition>(casted->getComposition()));
	EXPECT_EQ(false, casted->isExtend());
	EXPECT_EQ(false, casted->isPartial());
	EXPECT_EQ(true, casted->isEncapsulated());
	EXPECT_EQ(ClassDecl::SubType::Class, casted->subType());
	EXPECT_EQ("comment", casted->getComment());
}

TEST(ParserTest, extendClassDefinition)
{
	auto parser = Parser("package extends Test () \"comment\" public end Test");

	auto eq = parser.classDefinition();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<LongClassDecl>(ptr));
	auto casted = llvm::cast<LongClassDecl>(ptr);
	EXPECT_EQ(true, llvm::isa<Composition>(casted->getComposition()));
	EXPECT_EQ(true, casted->isExtend());
	EXPECT_EQ(false, casted->isPartial());
	EXPECT_EQ(false, casted->isEncapsulated());

	EXPECT_EQ(true, llvm::isa<ClassModification>(casted->getModification()));
	EXPECT_EQ(ClassDecl::SubType::Package, casted->subType());
	EXPECT_EQ("comment", casted->getComment());
}

TEST(ParserTest, derClassDefinition)
{
	auto parser = Parser("record Test = der (.tst, id1, id2) ");

	auto eq = parser.classDefinition();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<DerClassDecl>(ptr));
	auto casted = llvm::cast<DerClassDecl>(ptr);
	EXPECT_EQ(false, casted->isPartial());
	EXPECT_EQ(false, casted->isEncapsulated());

	EXPECT_EQ(2, casted->getIdents().size());
	EXPECT_EQ(true, casted->getTypeSpecifier().second);
	EXPECT_EQ(ClassDecl::SubType::Record, casted->subType());
}

TEST(ParserTest, enumerationClassDefinition)
{
	auto parser = Parser("type Test = enumeration (id1, id2) ");

	auto eq = parser.classDefinition();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<EnumerationClass>(ptr));
	auto casted = llvm::cast<EnumerationClass>(ptr);
	EXPECT_EQ(false, casted->isPartial());
	EXPECT_EQ(false, casted->isEncapsulated());

	EXPECT_EQ(true, llvm::isa<EnumerationLiteral>(casted->getEnumLiteratl(0)));
	EXPECT_EQ(2, casted->enumsCount());
	EXPECT_EQ(ClassDecl::SubType::Type, casted->subType());
}

TEST(ParserTest, compositionTest)
{
	auto parser =
			Parser("block t = .t; public block t2 = .t2; protected block t3 = .t3; "
						 "initial equation x = 1; algorithm external \"C\" call();");

	auto eq = parser.composition();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<Composition>(ptr));
	auto casted = llvm::cast<Composition>(ptr);
	EXPECT_EQ(true, llvm::isa<CompositionSection>(casted->getPublicSection()));
	EXPECT_EQ(true, llvm::isa<CompositionSection>(casted->getPrivateSection()));
	EXPECT_EQ(true, llvm::isa<CompositionSection>(casted->getProtectedSection()));
	EXPECT_EQ(
			true, llvm::isa<ExternalFunctionCall>(casted->getExternalFunctionCall()));
}

TEST(ParserTest, extendClause)
{
	auto parser = Parser("extends .t");

	auto eq = parser.extendClause();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ExtendClause>(ptr));
}

TEST(ParserTest, constrainClause)
{
	auto parser = Parser("constrainedby .t");

	auto eq = parser.constrainingClause();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ConstrainingClause>(ptr));
}
TEST(ParserTest, componentClause)
{
	auto parser = Parser("flow discrete input .t [1] nm, nm2");

	auto eq = parser.componentClause();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ComponentClause>(ptr));
	auto casted = llvm::cast<ComponentClause>(ptr);
	EXPECT_EQ(true, llvm::isa<ComponentDeclaration>(casted->getComponent(0)));
	EXPECT_EQ(
			ComponentClause::FlowStream::flow, casted->getPrefix().getFlowStream());
	EXPECT_EQ(ComponentClause::IO::input, casted->getPrefix().getIOType());
	EXPECT_EQ(ComponentClause::Type::discrete, casted->getPrefix().getType());
}

TEST(ParserTest, emptyClassModification)
{
	auto parser = Parser("()");

	auto eq = parser.classModification();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ClassModification>(ptr));
}

TEST(ParserTest, elementReplaceable)
{
	auto parser = Parser("each final replaceable .t t2");

	auto eq = parser.argument();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ReplecableModification>(ptr));
	auto casted = llvm::cast<ReplecableModification>(ptr);
	EXPECT_EQ(true, casted->isFinal());
	EXPECT_EQ(true, casted->hasEach());
}

TEST(ParserTest, simpleModification)
{
	auto parser = Parser("= 1");

	auto eq = parser.modification();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<SimpleModification>(ptr));
}

TEST(ParserTest, modification)
{
	auto parser = Parser("(nm, nm2)");

	auto eq = parser.modification();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ClassModification>(ptr));
}

TEST(ParserTest, importClause)
{
	auto parser = Parser("import name.{ls1,ls2}");

	auto eq = parser.importClause();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ImportClause>(ptr));
}

TEST(ParserTest, importClauseAll)
{
	auto parser = Parser("import name.*");

	auto eq = parser.importClause();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ImportClause>(ptr));
}

TEST(ParserTest, importClauseNamed)
{
	auto parser = Parser("import name = asd");

	auto eq = parser.importClause();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ImportClause>(ptr));
}
TEST(ParserTest, shortClassDefinition)
{
	auto parser =
			Parser("replaceable expandable connector Test = enumeration(:)");

	auto eq = parser.argument();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ReplecableModification>(ptr));
	auto casted = llvm::cast<ReplecableModification>(ptr);
	EXPECT_EQ(false, casted->isFinal());
	EXPECT_EQ(false, casted->hasEach());
}

TEST(ParserTest, elementRedeclaration)
{
	auto parser = Parser("redeclare each final .t t2");

	auto eq = parser.argument();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<Redeclaration>(ptr));
	auto casted = llvm::cast<Redeclaration>(ptr);
	EXPECT_EQ(true, casted->isFinal());
	EXPECT_EQ(true, casted->hasEach());
}

TEST(ParserTest, shortClassSpecifier)
{
	auto parser = Parser("block Test = input output .test[1]");

	auto eq = parser.classDefinition();
	if (!eq)
		FAIL();

	auto ptr = eq.get().get();

	EXPECT_EQ(true, llvm::isa<ShortClassDecl>(ptr));
	auto casted = llvm::cast<ShortClassDecl>(ptr);
	EXPECT_EQ(false, casted->isPartial());
	EXPECT_EQ(false, casted->isEncapsulated());

	EXPECT_EQ(true, casted->isInput());
	EXPECT_EQ(true, casted->isOutput());
	EXPECT_EQ(ClassDecl::SubType::Block, casted->subType());
}

TEST(ParserTest, classPrefixes)
{
	auto parser = Parser("partial expandable connector");

	auto pref = parser.classPrefixes();
	if (!pref)
		FAIL();

	auto [partial, pure, subType] = *pref;
	EXPECT_EQ(partial, true);
	EXPECT_EQ(pure, true);
	EXPECT_EQ(subType, ClassDecl::SubType::ExpandableConnector);
}

TEST(ParserTest, classPrefixesImpure)
{
	auto parser = Parser("impure operator function");

	auto pref = parser.classPrefixes();
	if (!pref)
		FAIL();

	auto [partial, pure, subType] = *pref;
	EXPECT_EQ(partial, false);
	EXPECT_EQ(pure, false);
	EXPECT_EQ(subType, ClassDecl::SubType::OperatorFunction);
}

TEST(ParserTest, emptyComposition)
{
	auto parser = Parser("public");

	auto decl = parser.composition();
	if (!decl)
		FAIL();

	auto ptr = decl.get().get();
	EXPECT_EQ(true, llvm::isa<Composition>(ptr));
	auto casted = llvm::cast<Composition>(ptr);
	EXPECT_EQ(true, llvm::isa<CompositionSection>(casted->getPrivateSection()));
	EXPECT_EQ(true, llvm::isa<CompositionSection>(casted->getPublicSection()));
	EXPECT_EQ(true, llvm::isa<CompositionSection>(casted->getProtectedSection()));
	EXPECT_EQ(nullptr, casted->getAnnotation());
	EXPECT_EQ(nullptr, casted->getExternalCallAnnotation());
	EXPECT_EQ("", casted->getLanguageSpec());
}

TEST(ParserTest, compositionWithString)
{
	auto parser = Parser("lambda = 148 \"Thermal "
											 "conductivity of silicon\" annotation(Evaluate = true)");

	auto decl = parser.componentDeclaration();
	if (!decl)
		FAIL();

	EXPECT_EQ(Token::End, parser.getCurrentToken());

	// auto ptr = decl.get().get();
}

TEST(ParserTest, equationSectionWithComments)
{
	auto parser =
			Parser("equation der(T[1,1,1]) = 1/C*(Gx*((-T[1,1,1]) + T[2,1,1]) + "
						 "Gy*((-T[1,1,1]) + T[1,2,1]) + Gz*(2*Tt-3*T[1,1,1] + T[1,1,2])) "
						 "\"Upper left top corner\";");

	auto decl = parser.composition();
	if (!decl)
		FAIL();
	EXPECT_EQ(Token::End, parser.getCurrentToken());
}

TEST(ParserTest, connectorClass)
{
	auto parser = Parser("connector HeatPort Types.Temperature T; flow "
											 "Types.Power Q; end HeatPort;");

	auto decl = parser.classDefinition();
	if (!decl)
		FAIL();

	auto ptr = decl.get().get();
	EXPECT_EQ(true, llvm::isa<LongClassDecl>(ptr));
	auto casted = llvm::cast<LongClassDecl>(ptr);

	auto comp = llvm::cast<Composition>(casted->getComposition());
	EXPECT_EQ(true, llvm::isa<CompositionSection>(comp->getPrivateSection()));
	auto publicSection =
			llvm::cast<CompositionSection>(comp->getPrivateSection());
	EXPECT_EQ(2, publicSection->size());
	EXPECT_EQ(true, llvm::isa<Element>(publicSection->getChild(0)));
}
