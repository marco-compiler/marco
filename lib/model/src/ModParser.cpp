#include "modelica/model/ModParser.hpp"

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "modelica/model/ModEqTemplate.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModLexerStateMachine.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/Interval.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

llvm::Expected<bool> ModParser::expect(ModToken t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedModToken>(current, t, getPosition());
}

#include "modelica/utils/ParserUtils.hpp"

Expected<string> ModParser::reference()
{
	auto s = lexer.getLastIdentifier();
	EXPECT(ModToken::Ident);
	return s;
}

Expected<int> ModParser::integer()
{
	bool minus = accept<ModToken::Minus>();

	int b = lexer.getLastInt();
	EXPECT(ModToken::Integer);

	return minus ? -b : b;
}

Expected<float> ModParser::floatingPoint()
{
	bool minus = accept<ModToken::Minus>();

	float b = lexer.getLastFloat();
	EXPECT(ModToken::Float);

	return minus ? -b : b;
}

Expected<ModConst> ModParser::intVector()
{
	SmallVector<int, 3> args;
	EXPECT(ModToken::LCurly);
	TRY(val, integer());
	args.push_back(*val);

	while (accept(ModToken::Comma))
	{
		TRY(val, integer());
		args.push_back(*val);
	}

	EXPECT(ModToken::RCurly);
	return ModConst(move(args));
}

Expected<ModConst> ModParser::floatVector()
{
	SmallVector<float, 3> args;
	EXPECT(ModToken::LCurly);
	TRY(val, floatingPoint());
	args.push_back(*val);

	while (accept(ModToken::Comma))
	{
		TRY(val, floatingPoint());
		args.push_back(*val);
	}

	EXPECT(ModToken::RCurly);
	return ModConst(move(args));
}

Expected<ModConst> ModParser::boolVector()
{
	SmallVector<bool, 3> args;
	EXPECT(ModToken::LCurly);

	bool b = static_cast<bool>(lexer.getLastInt());
	args.push_back(b);
	EXPECT(ModToken::Integer);

	while (accept(ModToken::Comma))
	{
		bool b = static_cast<bool>(lexer.getLastInt());
		args.push_back(b);
		EXPECT(ModToken::Integer);
	}

	EXPECT(ModToken::RCurly);
	return ModConst(move(args));
}

Expected<SmallVector<size_t, 3>> ModParser::typeDimensions()
{
	SmallVector<size_t, 3> v;
	EXPECT(ModToken::LSquare);

	if (accept<ModToken::RSquare>())
		return v;

	v.emplace_back(lexer.getLastInt());
	EXPECT(ModToken::Integer);

	while (accept(ModToken::Comma))
	{
		v.emplace_back(lexer.getLastInt());
		EXPECT(ModToken::Integer);
	}

	EXPECT(ModToken::RSquare);

	return v;
}

Expected<tuple<ModExpKind, vector<ModExp>>> ModParser::operation()
{
	EXPECT(ModToken::LPar);

	ModExpKind kind;

	if (accept<ModToken::Minus>())
		kind = ModExpKind::sub;
	else if (accept<ModToken::Multiply>())
		kind = ModExpKind::mult;
	else if (accept<ModToken::Division>())
		kind = ModExpKind::divide;
	else if (accept<ModToken::GreaterEqual>())
		kind = ModExpKind::greaterEqual;
	else if (accept<ModToken::GreaterThan>())
		kind = ModExpKind::greaterThan;
	else if (accept<ModToken::LessEqual>())
		kind = ModExpKind::lessEqual;
	else if (accept<ModToken::LessThan>())
		kind = ModExpKind::less;
	else if (accept<ModToken::OperatorEqual>())
		kind = ModExpKind::equal;
	else if (accept<ModToken::Modulo>())
		kind = ModExpKind::module;
	else if (accept<ModToken::Exponential>())
		kind = ModExpKind::elevation;
	else if (accept<ModToken::Ternary>())
		kind = ModExpKind::conditional;
	else if (accept<ModToken::IndKeyword>())
		kind = ModExpKind::induction;
	else if (accept<ModToken::Not>())
		kind = ModExpKind::negate;
	else if (accept<ModToken::AtKeyword>())
		kind = ModExpKind::at;
	else if (auto e = expect(ModToken::Plus); !e)
		return e.takeError();
	else
		kind = ModExpKind::add;

	vector<ModExp> args;

	size_t arity = ModExp::Operation::arityOfOp(kind);
	for (size_t a = 0; a < arity; a++)
	{
		TRY(arg, expression());
		args.emplace_back(move(*arg));
		if (a != arity - 1)
			EXPECT(ModToken::Comma);
	}

	EXPECT(ModToken::RPar);
	return tuple(kind, move(args));
}

Expected<ModType> ModParser::type()
{
	BultinModTypes type;
	if (accept<ModToken::BoolKeyword>())
		type = BultinModTypes::BOOL;
	else if (accept<ModToken::IntKeyword>())
		type = BultinModTypes::INT;
	else if (auto e = expect(ModToken::FloatKeyword); e)
		type = BultinModTypes::FLOAT;
	else
		return e.takeError();

	TRY(dim, typeDimensions());
	return ModType(type, std::move(*dim));
}

Expected<vector<ModExp>> ModParser::args()
{
	vector<ModExp> args;
	EXPECT(ModToken::LPar);
	if (accept<ModToken::RPar>())
		return args;

	TRY(exp, expression());
	args.emplace_back(move(*exp));

	while (accept(ModToken::Comma))
	{
		TRY(exp, expression());
		args.emplace_back(move(*exp));
	}

	EXPECT(ModToken::RPar);
	return args;
}

Expected<ModCall> ModParser::call()
{
	EXPECT(ModToken::CallKeyword);
	auto fname = lexer.getLastIdentifier();
	EXPECT(ModToken::Ident);

	TRY(t, type());
	TRY(argVec, args());

	SmallVector<unique_ptr<ModExp>, 3> vec;
	for (auto& exp : *argVec)
		vec.emplace_back(std::make_unique<ModExp>(move(exp)));

	return ModCall(move(fname), move(vec), move(*t));
}

Expected<StringMap<ModVariable>> ModParser::initSection()
{
	StringMap<ModVariable> map;
	EXPECT(ModToken::InitKeyword);

	while (current != ModToken::End && current != ModToken::UpdateKeyword &&
				 current != ModToken::TemplateKeyword)
	{
		bool constant = accept<ModToken::ConstantKeyword>();
		bool state = accept<ModToken::StateKeyword>();
		TRY(stat, statement());

		auto [name, exp] = move(*stat);
		ModVariable var(name, move(exp), state, constant);

		if (map.find(var.getName()) != map.end())
			return make_error<UnexpectedModToken>(
					ModToken::Ident, ModToken::Ident, getPosition());
		map.try_emplace(name, move(var));
	}

	return map;
}

Expected<ModEqTemplate> ModParser::singleTemplate()
{
	auto name = lexer.getLastIdentifier();
	EXPECT(ModToken::Ident);
	TRY(leftHand, expression());
	EXPECT(ModToken::Assign);
	TRY(rightHand, expression());

	return ModEqTemplate(move(*leftHand), move(*rightHand), move(name));
}

Expected<ModParser::TemplatesMap> ModParser::templates()
{
	TemplatesMap toReturn;

	if (!accept<ModToken::TemplateKeyword>())
		return toReturn;
	while (current != ModToken::UpdateKeyword && current != ModToken::End)
	{
		TRY(temp, singleTemplate());
		toReturn.try_emplace(temp->getName(), make_shared<ModEqTemplate>(*temp));
	}

	return toReturn;
}

Expected<SmallVector<ModEquation, 0>> ModParser::updateSection(
		const TemplatesMap& templs)
{
	SmallVector<ModEquation, 0> map;
	EXPECT(ModToken::UpdateKeyword);

	while (current != ModToken::End)
	{
		TRY(stat, matchedUpdateStatement(templs));
		map.push_back(move(*stat));
	}

	return map;
}

Expected<Model> ModParser::simulation()
{
	TRY(initSect, initSection());
	TRY(templ, templates());
	TRY(updateSect, updateSection(*templ));
	return Model(move(*updateSect), move(*initSect));
}

Expected<tuple<string, ModExp>> ModParser::statement()
{
	auto name = lexer.getLastIdentifier();
	EXPECT(ModToken::Ident);
	EXPECT(ModToken::Assign);
	TRY(exp, expression());

	return tuple(move(name), move(*exp));
}
Expected<Interval> ModParser::singleInduction()
{
	EXPECT(ModToken::LSquare);

	size_t begin = lexer.getLastInt();

	EXPECT(ModToken::Integer);
	EXPECT(ModToken::Comma);
	size_t end = lexer.getLastInt();
	EXPECT(ModToken::Integer);
	EXPECT(ModToken::RSquare);

	return Interval(begin, end);
}
Expected<MultiDimInterval> ModParser::inductions()
{
	SmallVector<Interval, 2> inductions;
	if (!accept<ModToken::ForKeyword>())
		return inductions;
	while (current == ModToken::LSquare)
	{
		TRY(ind, singleInduction());
		inductions.push_back(move(*ind));
	}
	return inductions;
}

Expected<ModEquation> ModParser::matchedUpdateStatement(const TemplatesMap& map)
{
	TRY(eq, updateStatement(map));
	if (accept<ModToken::MatchedKeyword>())
	{
		TRY(e, matchingPath());
		eq->setMatchedExp(move(*e));
	}

	return eq;
}

Expected<ModEquation> ModParser::updateStatement(const TemplatesMap& map)
{
	bool backward = accept<ModToken::BackwardKeyword>();
	TRY(inductionsV, inductions());
	MultiDimInterval ind = move(*inductionsV);

	if (accept<ModToken::TemplateKeyword>())
	{
		auto templateName = lexer.getLastIdentifier();
		EXPECT(ModToken::Ident);
		assert(map.find(templateName) != map.end());
		auto templ = map.find(templateName)->second;
		return ModEquation(std::move(templ), std::move(ind), !backward);
	}

	if (current == ModToken::Ident)
	{
		auto name = lexer.getLastIdentifier();
		EXPECT(ModToken::Ident);
		EXPECT(ModToken::Assign);
		TRY(exp, expression());
		auto tp = exp->getModType();
		auto leftRef = ModExp(move(name), move(tp));

		return ModEquation(leftRef, move(*exp), "", move(ind), !backward);
	}
	TRY(leftHand, expression());
	EXPECT(ModToken::Assign);
	TRY(exp, expression());

	return ModEquation(move(*leftHand), move(*exp), "", move(ind), !backward);
}

Expected<ModExp> ModParser::expression()
{
	TRY(tp, type());
	if (current == ModToken::Ident)
	{
		TRY(ref, reference());
		return ModExp(move(*ref), move(*tp));
	}

	if (current == ModToken::LCurly)
	{
		if (tp->getBuiltin() == BultinModTypes::BOOL)
		{
			TRY(cont, boolVector());
			return ModExp(move(*cont), move(*tp));
		}
		if (tp->getBuiltin() == BultinModTypes::INT)
		{
			TRY(cont, intVector());
			return ModExp(move(*cont), move(*tp));
		}
		if (tp->getBuiltin() == BultinModTypes::FLOAT)
		{
			TRY(cont, floatVector());
			return ModExp(move(*cont), move(*tp));
		}
	}

	if (current == ModToken::CallKeyword)
	{
		TRY(c, call());
		return ModExp(move(*c), move(*tp));
	}

	if (current == ModToken::LPar)
	{
		TRY(op, operation());
		auto [kind, args] = move(*op);

		SmallVector<unique_ptr<ModExp>, 3> vec;
		for (auto& exp : args)
			vec.push_back(std::make_unique<ModExp>(move(exp)));

		return ModExp(kind, move(vec), move(*tp));
	}

	return make_error<UnexpectedModToken>(current, ModToken::LPar, getPosition());
}

Expected<EquationPath> ModParser::matchingPath()
{
	EXPECT(ModToken::LSquare);
	auto isLeft = lexer.getLastInt() == 0;
	EXPECT(ModToken::Integer);
	SmallVector<size_t, 3> pos;

	if (accept<ModToken::Comma>())
	{
		pos.emplace_back(lexer.getLastInt());
		EXPECT(ModToken::Integer);
	}

	EXPECT(ModToken::RSquare);
	return EquationPath(move(pos), isLeft);
}
