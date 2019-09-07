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

	if (accept<Token::OperaptorKeyword>())
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

	bool pure = true;
	if (accept<Token::ImpureKeyword>())
		pure = false;

	if (accept<Token::PureKeyword>() && !pure)
		return make_error<UnexpectedToken>(current);

	if (accept<Token::OperaptorKeyword>())
	{
		if (auto e = expect(Token::FunctionKeyword); !e)
			return e.takeError();

		return make_tuple(partial, pure, ClassDecl::SubType::OperatorFunction);
	}

	if (accept<Token::FunctionKeyword>())
		return make_tuple(partial, pure, ClassDecl::SubType::Function);

	return make_error<UnexpectedToken>(current);
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

ExpectedUnique<ClassDecl> Parser::longClassSpecifier()
{
	return make_error<NotImplemented>("notImplemented");
	// SourcePosition currentPos = getPosition();
	// auto cl = makeNode<LongClassDecl>(currentPos);

	// auto comment = stringComment();
	// if (!comment)
	// return comment.takeError();

	// cl.get()->setComment(move(*comment));

	// auto comp = composition();
	// if (!comp)
	// return comp.takeError();

	// if (auto e = expect(Token::EndKeyword); !e)
	// return e.takeError();

	// if (auto e = expect(Token::Ident); !e)
	// return e.takeError();

	// return cl;
}

ExpectedUnique<ClassDecl> Parser::selectClassSpecifier()
{
	return make_error<NotImplemented>("notImplemented");
	// if (!accept<Token::Assignment>())
	// return longClassSpecifier();

	// if (accept<Token::DerKeyword>())
	// return derClassSpecifier();

	// if (accept<Token::EnumerationKeyword>())
	// return enumerationClassSpecifier();

	// return shortClassSpecifier();
}

ExpectedUnique<ImportClause> Parser::importClause()
{
	return make_error<NotImplemented>("notImplemented");
	// SourcePosition currentPos = getPosition();

	// if (auto e = expect(Token::ImportKeyword); !e)
	// return e.takeError();

	// if (current != Token::Ident)
	// return make_error<UnexpectedToken>(Token::Ident);

	// auto leftHand = name();
	// if (!leftHand)
	// return leftHand.takeError();

	// if (accept<Token::Equal>())
	//{
	// if (leftHand->size() != 1)
	// return make_error<UnexpectedToken>(current);

	// auto rightHand = name();
	// if (!rightHand)
	// return rightHand.takeError();

	// auto cmnt = comment();
	//}
}

ExpectedUnique<ClassDecl> Parser::classSpecifier()
{
	return make_error<NotImplemented>("notImplemented");
	// if (accept<Token::ExtendsKeyword>())
	// return extendLongClassSpecifier();

	// string lastIdent = lexer.getLastIdentifier();
	// if (auto e = expect(Token::Ident); !e)
	// return e.takeError();

	// ExpectedUnique<ClassDecl> decl = selectClassSpecifier();

	// if (!decl)
	// return decl.takeError();

	// decl.get()->setName(move(lastIdent));
	// return move(*decl);
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
	if (auto e = expect(Token::ExtendsKeyword))
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

ExpectedUnique<Declaration> Parser::elementRedeclaration()
{
	SourcePosition currentPos = getPosition();
	if (auto e = expect(Token::RedeclareKeyword); !e)
		return e.takeError();

	bool each = accept<Token::EachKeyword>();
	bool fnal = accept<Token::FinalKeyword>();

	ExpectedUnique<Declaration> child = nullptr;

	if (current == Token::RedeclareKeyword)
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

		if (!expect(Token::RSquare))
			return llvm::make_error<UnexpectedToken>(current);
	}

	auto s = makeNode<ArraySubscriptionDecl>(currentPos, move(subScript));

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
	if (decl)
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
	if (decl)
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

		if (!expect(Token::RSquare))
			return llvm::make_error<UnexpectedToken>(current);
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
