#include "modelica/model/ModParser.hpp"

#include "modelica/model/ModErrors.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

llvm::Expected<bool> ModParser::expect(ModToken t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedModToken>(current, t, getPosition());
}

Expected<string> ModParser::reference()
{
	auto s = lexer.getLastIdentifier();
	if (auto e = expect(ModToken::Ident); !e)
		return e.takeError();
	return s;
}

Expected<int> ModParser::integer()
{
	bool minus = accept<ModToken::Minus>();

	int b = lexer.getLastInt();
	if (auto e = expect(ModToken::Integer); !e)
		return e.takeError();

	return minus ? -b : b;
}

Expected<float> ModParser::floatingPoint()
{
	bool minus = accept<ModToken::Minus>();

	float b = lexer.getLastFloat();
	if (auto e = expect(ModToken::Float); !e)
		return e.takeError();

	return minus ? -b : b;
}

Expected<ModConst<int>> ModParser::intVector()
{
	SmallVector<int, 3> args;
	if (auto e = expect(ModToken::LCurly); !e)
		return e.takeError();

	auto val = integer();
	if (!val)
		return val.takeError();

	args.push_back(*val);

	while (accept(ModToken::Comma))
	{
		auto val = integer();
		if (!val)
			return val.takeError();

		args.push_back(*val);
	}

	if (auto e = expect(ModToken::RCurly); !e)
		return e.takeError();

	return ModConst<int>(move(args));
}

Expected<ModConst<float>> ModParser::floatVector()
{
	SmallVector<float, 3> args;
	if (auto e = expect(ModToken::LCurly); !e)
		return e.takeError();

	auto val = floatingPoint();
	if (!val)
		return val.takeError();

	args.push_back(*val);

	while (accept(ModToken::Comma))
	{
		auto val = floatingPoint();
		if (!val)
			return val.takeError();

		args.push_back(*val);
	}

	if (auto e = expect(ModToken::RCurly); !e)
		return e.takeError();

	return ModConst<float>(move(args));
}

Expected<ModConst<bool>> ModParser::boolVector()
{
	SmallVector<bool, 3> args;
	if (auto e = expect(ModToken::LCurly); !e)
		return e.takeError();

	bool b = static_cast<bool>(lexer.getLastInt());
	args.push_back(b);

	if (auto e = expect(ModToken::Integer); !e)
		return e.takeError();

	while (accept(ModToken::Comma))
	{
		bool b = static_cast<bool>(lexer.getLastInt());
		args.push_back(b);

		if (auto e = expect(ModToken::Integer); !e)
			return e.takeError();
	}

	if (auto e = expect(ModToken::RCurly); !e)
		return e.takeError();

	return ModConst<bool>(move(args));
}

Expected<vector<size_t>> ModParser::typeDimensions()
{
	vector<size_t> v;
	if (auto e = expect(ModToken::LSquare); !e)
		return e.takeError();

	if (accept<ModToken::RSquare>())
		return v;

	v.emplace_back(lexer.getLastInt());
	if (auto e = expect(ModToken::Integer); !e)
		return e.takeError();

	while (accept(ModToken::Comma))
	{
		v.emplace_back(lexer.getLastInt());
		if (auto e = expect(ModToken::Integer); !e)
			return e.takeError();
	}

	if (auto e = expect(ModToken::RSquare); !e)
		return e.takeError();

	return v;
}

Expected<tuple<ModExpKind, vector<ModExp>>> ModParser::operation()
{
	if (auto e = expect(ModToken::LPar); !e)
		return e.takeError();

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
		auto arg = expression();
		if (!arg)
			return arg.takeError();
		args.emplace_back(move(*arg));
		if (a != arity - 1)
			if (auto e = expect(ModToken::Comma); !e)
				return e.takeError();
	}

	if (auto e = expect(ModToken::RPar); !e)
		return e.takeError();

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

	auto dim = typeDimensions();
	if (!dim)
		return dim.takeError();

	return ModType(type, std::move(*dim));
}

Expected<vector<ModExp>> ModParser::args()
{
	vector<ModExp> args;
	if (auto e = expect(ModToken::LPar); !e)
		return e.takeError();
	if (accept<ModToken::RPar>())
		return args;

	auto exp = expression();
	if (!exp)
		return exp.takeError();
	args.emplace_back(move(*exp));

	while (accept(ModToken::Comma))
	{
		auto exp = expression();
		if (!exp)
			return exp.takeError();
		args.emplace_back(move(*exp));
	}

	if (auto e = expect(ModToken::RPar); !e)
		return e.takeError();

	return args;
}

Expected<ModCall> ModParser::call()
{
	if (auto e = expect(ModToken::CallKeyword); !e)
		return e.takeError();

	auto fname = lexer.getLastIdentifier();
	if (auto e = expect(ModToken::Ident); !e)
		return e.takeError();

	auto t = type();
	if (!t)
		return t.takeError();

	auto argVec = args();
	if (!argVec)
		return argVec.takeError();

	SmallVector<unique_ptr<ModExp>, 3> vec;
	for (auto& exp : *argVec)
		vec.emplace_back(std::make_unique<ModExp>(move(exp)));

	return ModCall(move(fname), move(vec), move(*t));
}

Expected<StringMap<ModExp>> ModParser::initSection()
{
	StringMap<ModExp> map;
	if (auto e = expect(ModToken::InitKeyword); !e)
		return e.takeError();

	while (current != ModToken::End && current != ModToken::UpdateKeyword)
	{
		auto stat = statement();
		if (!stat)
			return stat.takeError();

		auto [name, exp] = move(*stat);
		if (map.find(name) != map.end())
			return make_error<UnexpectedModToken>(
					ModToken::Ident, ModToken::Ident, getPosition());
		map.try_emplace(move(name), move(exp));
	}

	return map;
}

Expected<SmallVector<Assigment, 0>> ModParser::updateSection()
{
	SmallVector<Assigment, 0> map;
	if (auto e = expect(ModToken::UpdateKeyword); !e)
		return e.takeError();

	while (current != ModToken::End)
	{
		auto stat = updateStatement();
		if (!stat)
			return stat.takeError();

		map.push_back(move(*stat));
	}

	return map;
}

Expected<tuple<StringMap<ModExp>, SmallVector<Assigment, 0>>>
ModParser::simulation()
{
	auto initSect = initSection();
	if (!initSect)
		return initSect.takeError();

	auto updateSect = updateSection();
	if (!updateSect)
		return updateSect.takeError();

	return tuple(move(*initSect), move(*updateSect));
}

Expected<tuple<string, ModExp>> ModParser::statement()
{
	auto name = lexer.getLastIdentifier();
	if (auto e = expect(ModToken::Ident); !e)
		return e.takeError();

	if (auto e = expect(ModToken::Assign); !e)
		return e.takeError();

	auto exp = expression();
	if (!exp)
		return exp.takeError();

	return tuple(move(name), move(*exp));
}
Expected<InductionVar> ModParser::singleInduction()
{
	if (auto e = expect(ModToken::LSquare); !e)
		return e.takeError();

	size_t begin = lexer.getLastInt();

	if (auto e = expect(ModToken::Integer); !e)
		return e.takeError();

	if (auto e = expect(ModToken::Comma); !e)
		return e.takeError();
	size_t end = lexer.getLastInt();

	if (auto e = expect(ModToken::Integer); !e)
		return e.takeError();

	if (auto e = expect(ModToken::RSquare); !e)
		return e.takeError();

	return InductionVar(begin, end);
}
Expected<SmallVector<InductionVar, 3>> ModParser::inductions()
{
	SmallVector<InductionVar, 3> inductions;
	if (!accept<ModToken::ForKeyword>())
		return inductions;
	while (current == ModToken::LSquare)
	{
		auto ind = singleInduction();
		if (!ind)
			return ind.takeError();
		inductions.push_back(move(*ind));
	}
	return inductions;
}

Expected<Assigment> ModParser::updateStatement()
{
	SmallVector<InductionVar, 3> ind;

	auto inductionsV = inductions();
	if (!inductionsV)
		return inductionsV.takeError();

	ind = move(*inductionsV);

	if (current == ModToken::Ident)
	{
		auto name = lexer.getLastIdentifier();
		if (auto e = expect(ModToken::Ident); !e)
			return e.takeError();

		if (auto e = expect(ModToken::Assign); !e)
			return e.takeError();

		auto exp = expression();
		if (!exp)
			return exp.takeError();

		return Assigment(move(name), move(*exp), move(ind));
	}
	auto leftHand = expression();
	if (!leftHand)
		return leftHand.takeError();

	if (auto e = expect(ModToken::Assign); !e)
		return e.takeError();

	auto exp = expression();
	if (!exp)
		return exp.takeError();

	return Assigment(move(*leftHand), move(*exp), move(ind));
}

Expected<ModExp> ModParser::expression()
{
	auto tp = type();
	if (!tp)
		return tp.takeError();
	if (current == ModToken::Ident)
	{
		auto ref = reference();
		if (!ref)
			return ref.takeError();

		return ModExp(move(*ref), move(*tp));
	}

	if (current == ModToken::LCurly)
	{
		if (tp->getBuiltin() == BultinModTypes::BOOL)
		{
			auto cont = boolVector();
			if (!cont)
				return cont.takeError();

			return ModExp(move(*cont), move(*tp));
		}
		if (tp->getBuiltin() == BultinModTypes::INT)
		{
			auto cont = intVector();
			if (!cont)
				return cont.takeError();

			return ModExp(move(*cont), move(*tp));
		}
		if (tp->getBuiltin() == BultinModTypes::FLOAT)
		{
			auto cont = floatVector();
			if (!cont)
				return cont.takeError();

			return ModExp(move(*cont), move(*tp));
		}
	}

	if (current == ModToken::CallKeyword)
	{
		auto c = call();
		if (!c)
			return c;

		return ModExp(move(*c), move(*tp));
	}

	if (current == ModToken::LPar)
	{
		auto op = operation();
		if (!op)
			return op.takeError();

		auto [kind, args] = move(*op);

		SmallVector<unique_ptr<ModExp>, 3> vec;
		for (auto& exp : args)
			vec.push_back(std::make_unique<ModExp>(move(exp)));

		return ModExp(kind, move(vec), move(*tp));
	}

	return make_error<UnexpectedModToken>(current, ModToken::LPar, getPosition());
}
