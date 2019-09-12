#include <algorithm>

#include "modelica/Parser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

Expected<tuple<bool, bool, ClassDecl::SubType>> Parser::classPrefixes()
{
	bool partial = false;
	if (accept<Token::PartialKeyword>())
		partial = true;

	if (accept<Token::ClassKeyword>())
		return make_tuple(partial, true, ClassDecl::SubType::Class);

	if (accept<Token::ModelKeyword>())
		return make_tuple(partial, true, ClassDecl::SubType::Model);

	if (accept<Token::BlockKeyword>())
		return make_tuple(partial, true, ClassDecl::SubType::Block);

	if (accept<Token::TypeKeyword>())
		return make_tuple(partial, true, ClassDecl::SubType::Type);

	if (accept<Token::PackageKeyword>())
		return make_tuple(partial, true, ClassDecl::SubType::Package);

	if (accept<Token::FunctionKeyword>())
		return make_tuple(partial, true, ClassDecl::SubType::Function);

	if (accept<Token::RecordKeyword>())
		return make_tuple(partial, true, ClassDecl::SubType::Record);

	if (accept<Token::OperatorKeyword>())
	{
		if (accept<Token::RecordKeyword>())
			return make_tuple(partial, true, ClassDecl::SubType::OperatorRecord);

		if (accept<Token::FunctionKeyword>())
			return make_tuple(partial, true, ClassDecl::SubType::OperatorFunction);

		return make_tuple(partial, true, ClassDecl::SubType::Operator);
	}

	if (accept<Token::ExpandableKeyword>())
	{
		if (auto e = expect(Token::ConnectorKeyword); !e)
			return e.takeError();

		return make_tuple(partial, true, ClassDecl::SubType::ExpandableConnector);
	}

	// purity is by default
	bool pure = true;
	if (accept<Token::ImpureKeyword>())
		pure = false;

	if (accept<Token::PureKeyword>() && !pure)
		return make_error<UnexpectedToken>(Token::InputKeyword, Token::PureKeyword);

	if (accept<Token::OperatorKeyword>())
	{
		if (auto e = expect(Token::FunctionKeyword); !e)
			return e.takeError();

		return make_tuple(partial, pure, ClassDecl::SubType::OperatorFunction);
	}

	if (accept<Token::FunctionKeyword>())
		return make_tuple(partial, pure, ClassDecl::SubType::Function);

	return make_error<UnexpectedToken>(current, Token::ClassKeyword);
}

Expected<string> Parser::stringComment()
{
	string comment = lexer.getLastString();
	if (!accept<Token::String>())
		return "";

	while (accept<Token::Plus>())
	{
		comment += lexer.getLastString();
		if (auto e = expect(Token::String); !e)
			return e.takeError();
	}
	return comment;
}

ExpectedUnique<ClassDecl> Parser::extendLongClassSpecifier()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::ExtendsKeyword); !e)
		return e.takeError();

	string name = lexer.getLastIdentifier();
	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	UniqueDecl modification = nullptr;
	if (current == Token::LPar)
	{
		auto mod = classModification();
		if (!mod)
			return mod.takeError();

		modification = move(*mod);
	}

	auto str = stringComment();
	if (!str)
		return str.takeError();

	auto comp = composition();
	if (!comp)
		return comp.takeError();

	if (auto e = expect(Token::EndKeyword); !e)
		return e.takeError();

	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	auto node = makeNode<LongClassDecl>(
			currentPos, move(*comp), move(modification), true);
	if (!node)
		return node.takeError();

	node.get()->setName(move(name));
	node.get()->setComment(move(*str));

	return node;
}

// notice that the extend clause is handled in extendLongClassSpecifier
// on top of that notice that the first ident has been processed by class
// specifier since it was needed to tell apart the various cases
///
ExpectedUnique<ClassDecl> Parser::longClassSpecifier()
{
	SourcePosition currentPos = getPosition();

	auto comment = stringComment();
	if (!comment)
		return comment.takeError();

	auto comp = composition();
	if (!comp)
		return comp.takeError();

	if (auto e = expect(Token::EndKeyword); !e)
		return e.takeError();

	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	auto cl = makeNode<LongClassDecl>(currentPos, move(*comp));
	if (!cl)
		return cl;
	cl.get()->setComment(move(*comment));
	return cl;
}

// notice that the enumeration sub case is handled in
// enumerationClassSpecifier
ExpectedUnique<ClassDecl> Parser::shortClassSpecifier()
{
	SourcePosition currentPos = getPosition();
	bool input = accept<Token::InputKeyword>();
	bool output = accept<Token::OutputKeyword>();

	auto tp = typeSpecifier();
	if (!tp)
		return tp.takeError();

	UniqueExpr subScript = nullptr;
	if (accept<Token::LSquare>())
	{
		auto subScriptVector = arraySubscript();
		if (!subScriptVector)
			return subScriptVector.takeError();

		auto newNode = makeNode<ArraySubscriptionExpr>(
				currentPos, nullptr, move(*subScriptVector));
		if (!newNode)
			return newNode.takeError();

		subScript = move(*newNode);

		if (auto e = expect(Token::RSquare); !e)
			return e.takeError();
	}
	auto s = makeNode<ArraySubscriptionDecl>(currentPos, move(subScript));
	if (!s)
		return s.takeError();

	UniqueDecl mod = nullptr;
	if (current == Token ::LPar)
	{
		auto mod = modification();
		if (!mod)
			return mod.takeError();
	}

	auto ann = comment();
	if (!ann)
		return ann.takeError();

	return makeNode<ShortClassDecl>(
			currentPos, input, output, move(*tp), move(*s), move(mod), move(*ann));
}

// The Ident = der part of the rule must be handled by the caller, since it's
// used to tell apart the kind of declarations
ExpectedUnique<ClassDecl> Parser::derClassSpecifier()
{
	SourcePosition currentPos = getPosition();

	if (auto e = expect(Token::LPar); !e)
		return e.takeError();

	auto tp = typeSpecifier();
	if (!tp)
		return tp.takeError();

	if (auto e = expect(Token::Comma); !e)
		return e.takeError();

	vector<string> idents;

	do
	{
		idents.push_back(lexer.getLastIdentifier());
		if (auto e = expect(Token::Ident); !e)
			return e.takeError();

	} while (accept<Token::Comma>());

	if (auto e = expect(Token::RPar); !e)
		return e.takeError();

	auto ann = comment();
	if (!ann)
		return ann.takeError();

	return makeNode<DerClassDecl>(
			currentPos, move(idents), move(*tp), move(*ann));
}

ExpectedUnique<Declaration> Parser::enumerationLiteral()
{
	SourcePosition currentPos = getPosition();
	string name = lexer.getLastIdentifier();

	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	auto ann = comment();
	if (!ann)
		return ann.takeError();

	return makeNode<EnumerationLiteral>(currentPos, move(name), move(*ann));
}

// the IDENT = enumeration part of the rule is handled in
// the caller, since it's needed to tell apart the two cases
ExpectedUnique<ClassDecl> Parser::enumerationClassSpecifier()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::LPar); !e)
		return e.takeError();

	bool colons = accept<Token::Colons>();
	vectorUnique<Declaration> literals;

	if (!colons)
	{
		do
		{
			auto lit = enumerationLiteral();
			if (!lit)
				return lit.takeError();

			literals.push_back(move(*lit));
		} while (accept<Token::Comma>());
	}

	auto ann = comment();
	if (!ann)
		return ann.takeError();

	return makeNode<EnumerationClass>(
			currentPos, colons, move(literals), move(*ann));
}

// select class specifier is used to tell apart the 4 possibile class
// definitions that start with IDENT =
ExpectedUnique<ClassDecl> Parser::selectClassSpecifier()
{
	if (!accept<Token::Equal>())
		return longClassSpecifier();

	if (accept<Token::DerKeyword>())
		return derClassSpecifier();

	if (accept<Token::EnumerationKeyword>())
		return enumerationClassSpecifier();

	return shortClassSpecifier();
}

ExpectedUnique<ClassDecl> Parser::shortClassDefinition()
{
	auto classPref = classPrefixes();
	if (!classPref)
		return classPref.takeError();

	auto [partial, pure, subType] = *classPref;

	string name = lexer.getLastIdentifier();
	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	if (auto e = expect(Token::Equal); !e)
		return e.takeError();

	unique_ptr<ClassDecl> decl = nullptr;
	if (accept<Token::EnumerationKeyword>())
	{
		auto d = enumerationClassSpecifier();
		if (!d)
			return d.takeError();
		decl = move(*d);
	}
	else
	{
		auto d = shortClassSpecifier();
		if (!d)
			return d.takeError();
		decl = move(*d);
	}

	decl->setName(move(name));

	decl->setType(subType);
	decl->setPure(pure);
	decl->setPartial(partial);

	return move(decl);
}

ExpectedUnique<Declaration> Parser::externalFunctionCall()
{
	// page 266
	//
	// the optional [component-reference "="]
	// is hard to tell apart from the rule without
	// the optional part.
	//
	// The way we solve this is by getting the component
	// reference, and then checking it the next is a "="
	//
	// if it's not then the component reference should have been
	// an ident and we steal the data from there.

	SourcePosition currentPos = getPosition();
	string ident;

	bool compRefFound = true;
	auto compRef = componentReference();
	if (!compRef)
		return compRef.takeError();

	if (accept<Token::Equal>())
	{
		compRefFound = true;
		ident = lexer.getLastIdentifier();
		if (auto e = expect(Token::Ident); !e)
			return e.takeError();
	}
	else
	{
		compRefFound = false;
		if (!isa<ComponentReferenceExpr>(compRef.get().get()))
			return make_error<UnexpectedToken>(current, Token::LSquare);

		auto ref = dyn_cast<ComponentReferenceExpr>(compRef.get().get());

		// make sure that there was not a dot at the start of the ident
		if (ref->hasGlobalLookup())
			return make_error<UnexpectedToken>(Token::Dot, Token::Ident);

		if (ref->getPreviousLookUp() != nullptr)
			return make_error<UnexpectedToken>(Token::Dot, Token::Ident);

		ident = ref->getName();
	}

	UniqueExpr args;

	if (auto e = expect(Token::LPar); !e)
		return e.takeError();

	if (!accept<Token::RPar>())
	{
		auto ls = expressionList();
		if (!ls)
			return ls.takeError();

		args = move(*ls);
	}

	if (auto e = expect(Token::RPar); e)
		return e.takeError();

	return makeNode<ExternalFunctionCall>(
			currentPos,
			move(ident),
			move(args),
			compRefFound ? move(*compRef) : nullptr);
}

// The check for the initial keyword is up to the caller,
// since seeing the InitialKeyword token is not enough to tell
// apart this rule from the algorithmSection rule
ExpectedUnique<Declaration> Parser::equationSection(
		const vector<Token>& stopTokens, bool initial)
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::EquationKeyword); !e)
		return e.takeError();

	vectorUnique<Equation> equs;

	while (find(stopTokens.begin(), stopTokens.end(), current) ==
				 stopTokens.end())
	{
		auto eq = equation();
		if (!eq)
			return eq.takeError();

		equs.push_back(move(*eq));

		if (auto e = expect(Token::Semicolons); !e)
			return e.takeError();
	}

	return makeNode<EquationSection>(currentPos, move(equs), initial);
}

// The check for the initial keyword is up to the caller,
// since seeing the InitialKeyword token is not enough to tell
// apart this rule from the euqtionSection rule
ExpectedUnique<Declaration> Parser::algorithmSection(
		const vector<Token>& stopTokens, bool initial)
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::AlgorithmKeyword); !e)
		return e.takeError();

	vectorUnique<Statement> equs;

	while (find(stopTokens.begin(), stopTokens.end(), current) ==
				 stopTokens.end())
	{
		auto eq = statement();
		if (!eq)
			return eq.takeError();

		equs.push_back(move(*eq));

		if (auto e = expect(Token::Semicolons); !e)
			return e.takeError();
	}

	return makeNode<AlgorithmSection>(currentPos, move(equs), initial);
}

// this rule (page 266 ) is huge and has some tricky parts inside it
// it's divided in 4 parts, the first is the mandatory element list, that can be
// empty then there is a while that collects all the sections and then there are
// 2 optional parts at the end.
ExpectedUnique<Declaration> Parser::composition()
{
	SourcePosition currentPos = getPosition();
	const vector<Token> stopTokens = {
		Token::PublicKeyword,		Token::ProtectedKeyword,
		Token::EquationKeyword, Token::AlgorithmKeyword,
		Token::ExternalKeyword, Token::AnnotationKeyword,
		Token::EndKeyword,			Token::End,
		Token::InitialKeyword
	};

	vectorUnique<Declaration> privates;
	vectorUnique<Declaration> publics;
	vectorUnique<Declaration> protecteds;
	vectorUnique<Declaration> equations;
	vectorUnique<Declaration> alghoritms;

	// collect the first element list
	auto firstList = elementList(stopTokens);
	if (!firstList)
		return firstList.takeError();

	for (auto& p : *firstList)
		privates.push_back(move(p));

	// initial can signal both equations or algoritms (page 268)
	// so if we found it we keep track here
	bool initialFound = false;

	// while we are not seeing a token that signal the next parts or the end of
	// the rule we keep reading sections
	while (current != Token::AnnotationKeyword &&
				 current != Token::ExternalKeyword && current != Token::EndKeyword &&
				 current != Token::End)
	{
		switch (current)
		{
			case Token::PublicKeyword:
			{
				accept<Token::PublicKeyword>();
				auto firstList = elementList(stopTokens);
				if (!firstList)
					return firstList.takeError();

				for (auto& p : *firstList)
					publics.push_back(move(p));
				initialFound = false;
				continue;
			}
			case Token::ProtectedKeyword:
			{
				accept<Token::ProtectedKeyword>();
				auto firstList = elementList(stopTokens);
				if (!firstList)
					return firstList.takeError();

				for (auto& p : *firstList)
					publics.push_back(move(p));
				initialFound = false;
				continue;
			}
			// notice the absence of continue
			// if there is a "Initial" we eat that and we keep reading and we
			// because the next keyword will tell us wich rule to invoke
			case Token::InitialKeyword:
			{
				accept<Token::InitialKeyword>();
				initialFound = true;
				continue;
			}
			case Token::EquationKeyword:
			{
				auto eqSection = equationSection(stopTokens, initialFound);
				if (!eqSection)
					return eqSection.takeError();

				equations.emplace_back(move(*eqSection));
				initialFound = false;
				continue;
			}
			case Token::AlgorithmKeyword:
			{
				auto eqSection = algorithmSection(stopTokens, initialFound);
				if (!eqSection)
					return eqSection.takeError();

				alghoritms.push_back(move(*eqSection));
				initialFound = false;
				continue;
			}
			default:
				return make_error<UnexpectedToken>(current, Token::PublicKeyword);
		}
	}

	// external part of the rule
	//[external [language-spec] ....]
	string languageSpec;
	UniqueDecl externalCall = nullptr;
	UniqueDecl annotationExternal = nullptr;

	if (accept<Token::ExternalKeyword>())
	{
		if (current == Token::String)
		{
			languageSpec = lexer.getLastString();
			accept<Token::String>();
		}

		// dot and ident are the possible starting tokens of component reference
		// this allows us to tell if there is a exteral function call or not
		if (current == Token::Dot || current == Token::Ident)
		{
			auto call = externalFunctionCall();
			if (!call)
				return call.takeError();

			externalCall = move(*call);
		}

		if (current == Token::AnnotationKeyword)
		{
			auto ann = annotation();
			if (!ann)
				return ann.takeError();

			annotationExternal = move(*ann);
		}

		if (auto e = expect(Token::Semicolons); !e)
			return e.takeError();
	}

	// last optional part of the rule:
	// [annotation ";"]
	UniqueDecl annot;
	if (current == Token::AnnotationKeyword)
	{
		auto ann = annotation();
		if (!ann)
			return ann.takeError();

		annot = move(*ann);

		if (auto e = expect(Token::Semicolons); !e)
			return e.takeError();
	}

	// finally we build the CompositionSections and the the composition it self

	auto publicSection = makeNode<CompositionSection>(currentPos, move(publics));
	if (!publicSection)
		return publicSection.takeError();
	auto privateSection =
			makeNode<CompositionSection>(currentPos, move(privates));
	if (!privateSection)
		return privateSection.takeError();
	auto protectedSection =
			makeNode<CompositionSection>(currentPos, move(protecteds));
	if (!protectedSection)
		return protectedSection.takeError();

	return makeNode<Composition>(
			currentPos,
			move(*privateSection),
			move(*publicSection),
			move(*protectedSection),
			move(equations),
			move(alghoritms),
			move(externalCall),
			move(annotationExternal),
			move(annot),
			move(languageSpec));
}

/**
 * Page 267, rather complex rule.
 * The simpler form import IDENT "=" name comment is covered in the first if.
 */
ExpectedUnique<ImportClause> Parser::importClause()
{
	SourcePosition currentPos = getPosition();

	if (auto e = expect(Token::ImportKeyword); !e)
		return e.takeError();

	string ident = lexer.getLastIdentifier();
	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	if (accept<Token::Equal>())
	{
		auto nm = name();
		if (!nm)
			return nm.takeError();

		auto cmnt = comment();
		if (!cmnt)
			return cmnt.takeError();

		return makeNode<ImportClause>(
				currentPos, move(*nm), move(ident), move(*cmnt));
	}

	// if there is not a equal it means that there is a name and the rule is in
	// the second form, the problem is that it's not a standard name since it
	// might be in the form {IDENT.}"*" that means we don't know when to stop.
	vector<string> nm;
	nm.push_back(move(ident));

	// we keep accepting dots and identifiers until
	// one of the two stops being there. We keep track if the last cycle ended
	// with a dot. if it didn't it means that it is a single import and not a
	// import all or a list of imports.
	bool seenDot = (current == Token::Dot);
	while (accept<Token::Dot>() && current == Token::Ident)
	{
		nm.push_back(lexer.getLastIdentifier());
		if (auto e = expect(Token::Ident); !e)
			return e.takeError();
		seenDot = (current == Token::Dot);
	}

	bool importAll = false;
	vector<string> impList;
	// if it did not ended with a dot it means that it is a single and import and
	// we continue, else we check for a .* or for a import list.
	if (seenDot)
	{
		if (accept<Token::Multiply>())
			importAll = true;
		else if (accept<Token::LCurly>())
		{
			auto ls = importList();
			if (!ls)
				return ls.takeError();

			if (auto e = expect(Token::RCurly); !e)
				return e.takeError();
			impList = move(*ls);
		}
	}

	auto cmnt = comment();
	if (!cmnt)
		return cmnt.takeError();

	return makeNode<ImportClause>(
			currentPos, move(nm), "", move(*cmnt), importAll, move(impList));
}

// page 266, since the 3 options all might be start with
// IDENT = then we need to read that part to tell them apart.
//
// The way we do it is to save the name of the class
// and set it later when the declaration in being returned.
ExpectedUnique<ClassDecl> Parser::classSpecifier()
{
	if (current == Token::ExtendsKeyword)
		return extendLongClassSpecifier();

	string lastIdent = lexer.getLastIdentifier();
	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	ExpectedUnique<ClassDecl> decl = selectClassSpecifier();

	if (!decl)
		return decl.takeError();

	decl.get()->setName(move(lastIdent));
	return move(*decl);
}

Expected<vector<string>> Parser::importList()
{
	vector<string> toReturn;
	do
	{
		toReturn.push_back(lexer.getLastIdentifier());
		if (auto e = expect(Token::Ident); !e)
			return e.takeError();

	} while (accept<Token::Comma>());

	return move(toReturn);
}

Expected<TypeSpecifier> Parser::typeSpecifier()
{
	bool globalLookUp = false;
	if (accept<Token::Dot>())
		globalLookUp = true;

	auto nm = name();
	if (!nm)
		return nm.takeError();

	return make_pair(move(*nm), globalLookUp);
}

ExpectedUnique<Declaration> Parser::comment()
{
	SourcePosition currentPos = getPosition();
	auto strCmnt = stringComment();
	if (!strCmnt)
		return strCmnt.takeError();

	ExpectedUnique<Declaration> mod = nullptr;
	if (current == Token::Equal || current == Token::Colons ||
			current == Token::LPar)
		mod = modification();

	if (!mod)
		return mod;

	auto ann = makeNode<Annotation>(currentPos, move(*mod));
	if (!ann)
		return ann;
	ann.get()->setComment(move(*strCmnt));
	return ann;
}

ExpectedUnique<Declaration> Parser::constrainingClause()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::ConstraynedByKeyword))
		return e.takeError();

	auto tpSpec = typeSpecifier();
	if (!tpSpec)
		return tpSpec.takeError();

	ExpectedUnique<Declaration> mod = nullptr;
	if (current == Token::LPar)
		mod = classModification();

	if (!mod)
		return mod;

	return makeNode<ConstrainingClause>(currentPos, move(*tpSpec), move(*mod));
}

ExpectedUnique<Declaration> Parser::annotation()
{
	if (auto e = expect(Token::AnnotationKeyword); !e)
		return e.takeError();

	return classModification();
}

ExpectedUnique<Declaration> Parser::extendClause()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::ExtendsKeyword); !e)
		return e.takeError();

	auto tpSpec = typeSpecifier();
	if (!tpSpec)
		return tpSpec.takeError();

	ExpectedUnique<Declaration> mod = nullptr;
	if (current == Token::LPar)
		mod = classModification();

	if (!mod)
		return mod;

	ExpectedUnique<Declaration> ann = nullptr;

	if (current == Token::AnnotationKeyword)
	{
		ann = annotation();
		if (!ann)
			return ann.takeError();
	}

	return makeNode<ExtendClause>(
			currentPos, move(*tpSpec), move(*mod), move(*ann));
}

ExpectedUnique<Declaration> Parser::argument()
{
	if (current == Token::RedeclareKeyword)
		return elementRedeclaration();

	bool each = accept<Token::EachKeyword>();
	bool fnal = accept<Token::FinalKeyword>();

	if (current == Token::ReplacableKeyword)
		return elementReplaceable(each, fnal);

	return elementModification(each, fnal);
}

ExpectedUnique<Declaration> Parser::classModification()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::LPar); !e)
		return e.takeError();

	vectorUnique<Declaration> arguments;

	if (accept<Token::RPar>())
		return makeNode<ClassModification>(currentPos, move(arguments));

	auto arg = argument();
	if (!arg)
		return arg.takeError();
	arguments.push_back(move(*arg));

	while (accept<Token::Comma>())
	{
		arg = argument();
		if (!arg)
			return arg.takeError();
		arguments.push_back(move(*arg));
	}

	if (auto e = expect(Token::RPar); !e)
		return e.takeError();

	return makeNode<ClassModification>(currentPos, move(arguments));
}

/**
 * page 267, element rule, we need to tell apart class definition and component
 * clause. the easier way is to check if the next token is a type-prefix or a
 * type specifier
 */
bool isCompClauseNext(Token next)
{
	switch (next)
	{
		case Token::FlowKeyword:
			return true;
		case Token::StremKeyword:
			return true;
		case Token::DiscreteKeyword:
			return true;
		case Token::ParameterKeyword:
			return true;
		case Token::ConstantKeyword:
			return true;
		case Token::InputKeyword:
			return true;
		case Token::OutputKeyword:
			return true;
		case Token::Dot:
			return true;
		case Token::Ident:
			return true;
		default:
			return false;
	}
}

/**
 * an element list in the grammar can be empty but it's a pain to check it here,
 * the optionality is delegated to the users that will provide the tokens that
 * will signal the end of the elementList
 */
Expected<vectorUnique<Declaration>> Parser::elementList(
		const vector<Token>& stopTokens)
{
	vectorUnique<Declaration> toReturn;
	while (find(stopTokens.begin(), stopTokens.end(), current) ==
				 stopTokens.end())
	{
		auto elem = element();
		if (!elem)
			return elem.takeError();

		toReturn.push_back(move(*elem));

		if (auto e = expect(Token::Semicolons); !e)
			return e.takeError();
	}
	return move(toReturn);
}

ExpectedUnique<Declaration> Parser::element()
{
	SourcePosition currentPos = getPosition();
	if (current == Token::ImportKeyword)
		return importClause();

	if (current == Token::ExtendsKeyword)
		return extendClause();

	// collects the first part of the instructions rules.
	// we will use repleacable later to tell apart the two cases
	bool redeclare = accept<Token::RedeclareKeyword>();
	bool fnl = accept<Token::FinalKeyword>();
	bool inner = accept<Token::InnerKeyword>();
	bool outer = accept<Token::OuterKeyword>();
	bool repleacable = accept<Token::ReplacableKeyword>();

	// child is the part of the ast we are building now
	UniqueDecl child = nullptr;

	// we look the current token to tell if it's a component clause or
	// a class definition, in both cases we store the node in child, so that we
	// can use it later.
	if (isCompClauseNext(current))
	{
		auto compClause = componentClause();
		if (!compClause)
			return compClause.takeError();

		child = move(*compClause);
	}
	else
	{
		auto clsDef = classDefinition();
		if (!clsDef)
			return clsDef.takeError();

		child = move(*clsDef);
	}

	// if the repleacable keyword was seen we don't return use child direcly
	// we must place that in a replacable clause.
	//
	if (repleacable)
	{
		// if we seen a repleacable keyword we could see a constraining clause as
		// well, either cases we put the repleacable modification into child once we
		// are done building it.
		if (current == Token::ConstraynedByKeyword)
		{
			auto clause = constrainingClause();
			if (!clause)
				return clause;

			auto cmnt = comment();
			if (!cmnt)
				return cmnt;

			auto node = makeNode<ReplecableModification>(
					currentPos, move(child), false, fnl, move(*clause), move(*cmnt));
			if (!node)
				return node.takeError();

			child = move(*node);
		}
		else
		{
			auto node =
					makeNode<ReplecableModification>(currentPos, move(child), false, fnl);
			if (!node)
				return node.takeError();
			child = move(*node);
		}
	}

	// if we have seen a redeclare keyword we build a redeclaration and we store
	// that in child.
	if (redeclare)
	{
		auto node = makeNode<Redeclaration>(currentPos, move(child), false, fnl);
		if (!node)
			return node.takeError();

		child = move(*node);
	}

	// final no matter what is in child we encapsulate that into element and we
	// return it.
	return makeNode<Element>(currentPos, move(child), inner, outer);
}

ExpectedUnique<Declaration> Parser::elementRedeclaration()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::RedeclareKeyword); !e)
		return e.takeError();

	bool each = accept<Token::EachKeyword>();
	bool fnal = accept<Token::FinalKeyword>();

	ExpectedUnique<Declaration> child = nullptr;

	if (current == Token::ReplacableKeyword)
		child = elementReplaceable(each, fnal);
	else
		child = componentClause1();

	if (!child)
	{
		child = shortClassDefinition();
		if (!child)
			return child.takeError();
	}

	return makeNode<Redeclaration>(currentPos, move(*child), each, fnal);
}

ExpectedUnique<Declaration> Parser::elementModification(bool each, bool finl)
{
	SourcePosition currentPos = getPosition();
	auto nm = name();
	if (!nm)
		return nm.takeError();

	ExpectedUnique<Declaration> decl = nullptr;
	if (current == Token::LPar || current == Token::Colons ||
			current == Token::Equal)
	{
		decl = modification();
		if (!decl)
			return decl.takeError();
	}

	auto commnt = stringComment();
	if (!commnt)
		return commnt.takeError();

	auto node = makeNode<ElementModification>(
			currentPos, move(*decl), move(*nm), each, finl);
	if (!node)
		return node;

	node->get()->setComment(move(*commnt));
	return move(node);
}

ExpectedUnique<Declaration> Parser::elementReplaceable(bool each, bool fnl)
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::ReplacableKeyword); !e)
		return e.takeError();

	auto clause = componentClause1();
	if (!clause)
	{
		clause = shortClassDefinition();

		if (!clause)
			return clause;
	}

	ExpectedUnique<Declaration> constrClause = nullptr;
	if (current == Token::ConstraynedByKeyword)
		constrClause = constrainingClause();
	if (!constrClause)
		return constrClause;

	return makeNode<ReplecableModification>(
			currentPos, move(*clause), each, fnl, move(*constrClause));
}

ExpectedUnique<Declaration> Parser::modification()
{
	SourcePosition currentPos = getPosition();
	if (accept<Token::Colons>())
	{
		if (auto e = expect(Token::Equal); !e)
			return e.takeError();

		auto expr = expression();
		if (!expr)
			return expr.takeError();

		return makeNode<SimpleModification>(currentPos, move(*expr));
	}

	if (accept<Token::Equal>())
	{
		auto expr = expression();
		if (!expr)
			return expr.takeError();

		return makeNode<SimpleModification>(currentPos, move(*expr));
	}

	auto mod = classModification();
	if (!mod)
		return mod.takeError();

	if (!accept<Token::Equal>())
		return mod;

	SourcePosition currentPos2 = getPosition();
	auto expr = expression();
	if (!expr)
		return expr.takeError();

	auto simpleMod = makeNode<SimpleModification>(currentPos2, move(*expr));
	if (!simpleMod)
		return simpleMod;

	return makeNode<OverridingClassModification>(
			currentPos, move(*mod), move(*simpleMod));
}

ExpectedUnique<Declaration> Parser::componentClause1()
{
	SourcePosition currentPos = getPosition();
	auto prefix = typePrefix();
	if (!prefix)
		return prefix.takeError();

	auto typeSpec = typeSpecifier();
	if (!typeSpec)
		return typeSpec.takeError();

	auto compDecl = componentDeclaration1();
	if (!compDecl)
		return compDecl.takeError();

	vectorUnique<Declaration> vector;
	vector.push_back(move(*compDecl));

	return makeNode<ComponentClause>(
			currentPos,
			*prefix,
			typeSpec.get().second,			 // global look up
			move(typeSpec.get().first),	// name to lookup
			move(vector));
}

ExpectedUnique<Declaration> Parser::componentClause()
{
	SourcePosition currentPos = getPosition();
	auto prefix = typePrefix();
	if (!prefix)
		return prefix.takeError();

	auto typeSpec = typeSpecifier();
	if (!typeSpec)
		return typeSpec.takeError();

	UniqueExpr subScript = nullptr;
	if (accept<Token::LSquare>())
	{
		auto subScriptVector = arraySubscript();
		if (!subScriptVector)
			return subScriptVector.takeError();

		auto newNode = makeNode<ArraySubscriptionExpr>(
				currentPos, nullptr, move(*subScriptVector));
		if (!newNode)
			return newNode.takeError();

		subScript = move(*newNode);

		if (auto e = expect(Token::RSquare); !e)
			return e.takeError();
	}

	auto s = makeNode<ArraySubscriptionDecl>(currentPos, move(subScript));
	if (!s)
		return s.takeError();

	auto compDecl = componentList();
	if (!compDecl)
		return compDecl.takeError();

	return makeNode<ComponentClause>(
			currentPos,
			*prefix,
			typeSpec.get().second,			 // global look up
			move(typeSpec.get().first),	// name to lookup
			move(*compDecl),
			move(*s));
}

ExpectedUnique<Declaration> Parser::conditionAttribute()
{
	SourcePosition currentPos = getPosition();
	auto exp = expression();
	if (!exp)
		return exp.takeError();

	return makeNode<ConditionAttribute>(currentPos, move(*exp));
}

Expected<vectorUnique<Declaration>> Parser::componentList()
{
	vectorUnique<Declaration> toReturn;

	auto component = componentDeclaration();
	if (!component)
		return component.takeError();

	toReturn.push_back(move(*component));

	while (accept<Token::Comma>())
	{
		component = componentDeclaration();
		if (!component)
			return component.takeError();

		toReturn.push_back(move(*component));
	}

	return move(toReturn);
}

ExpectedUnique<Declaration> Parser::componentDeclaration()
{
	SourcePosition currentPos = getPosition();
	auto decl = declaration();
	if (!decl)
		return decl.takeError();

	UniqueDecl expr = nullptr;
	if (accept<Token::IfKeyword>())
	{
		auto exp = conditionAttribute();
		if (!exp)
			return exp.takeError();

		expr = move(*exp);
	}

	auto cmnt = comment();
	if (!cmnt)
		return cmnt.takeError();

	return makeNode<ComponentDeclaration>(
			currentPos,
			move(get<0>(*decl)),
			move(get<1>(*decl)),
			move(get<2>(*decl)),
			move(*cmnt),
			move(expr));
}

ExpectedUnique<Declaration> Parser::componentDeclaration1()
{
	SourcePosition currentPos = getPosition();
	auto decl = declaration();
	if (!decl)
		return decl.takeError();

	auto cmnt = comment();
	if (!cmnt)
		return cmnt.takeError();

	return makeNode<ComponentDeclaration>(
			currentPos,
			move(get<0>(*decl)),
			move(get<1>(*decl)),
			move(get<2>(*decl)),
			move(*cmnt));
}

Expected<DeclarationName> Parser::declaration()
{
	SourcePosition currentPos = getPosition();
	std::string ident = lexer.getLastIdentifier();
	if (auto e = expect(Token::Ident); !e)
		return e.takeError();

	UniqueExpr subScript = nullptr;
	if (accept<Token::LSquare>())
	{
		auto subScriptVector = arraySubscript();
		if (!subScriptVector)
			return subScriptVector.takeError();

		auto newNode = makeNode<ArraySubscriptionExpr>(
				currentPos, nullptr, move(*subScriptVector));
		if (!newNode)
			return newNode.takeError();

		subScript = move(*newNode);

		if (auto e = expect(Token::RSquare); !e)
			return e.takeError();
	}

	auto s = makeNode<ArraySubscriptionDecl>(currentPos, move(subScript));
	if (!s)
		return s.takeError();

	ExpectedUnique<Declaration> dec = nullptr;
	if (current == Token::LPar || current == Token::Assignment ||
			current == Token::Colons)
		dec = modification();

	if (!dec)
		return dec.takeError();

	return make_tuple(move(ident), move(*s), move(*dec));
}

Expected<ComponentClause::Prefix> Parser::typePrefix()
{
	auto fl = ComponentClause::FlowStream::none;
	auto IO = ComponentClause::IO::none;
	auto type = ComponentClause::Type::none;

	if (accept<Token::FlowKeyword>())
		fl = ComponentClause::FlowStream::flow;
	else if (accept<Token::StremKeyword>())
		fl = ComponentClause::FlowStream::stream;

	if (accept<Token::DiscreteKeyword>())
		type = ComponentClause::Type::discrete;
	else if (accept<Token::ParameterKeyword>())
		type = ComponentClause::Type::parameter;
	else if (accept<Token::ConstantKeyword>())
		type = ComponentClause::Type::consant;

	if (accept<Token::InputKeyword>())
		IO = ComponentClause::IO::input;
	else if (accept<Token::OutputKeyword>())
		IO = ComponentClause::IO::output;

	return ComponentClause::Prefix(fl, IO, type);
}

ExpectedUnique<ClassDecl> Parser::classDefinition()
{
	bool encapsulated = false;
	if (accept<Token::EncapsulatedKeyword>())
		encapsulated = true;

	auto pref = classPrefixes();
	if (!pref)
		return pref.takeError();

	auto [partial, pure, subType] = *pref;

	auto specifier = classSpecifier();
	if (!specifier)
		return specifier.takeError();

	specifier.get()->setEncapsulated(encapsulated);
	specifier.get()->setType(subType);
	specifier.get()->setPure(pure);
	specifier.get()->setPartial(partial);
	return move(*specifier);
}
