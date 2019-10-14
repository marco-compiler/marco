#include "modelica/simulation/SimParser.hpp"

#include "modelica/simulation/SimErrors.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

llvm::Expected<bool> SimParser::expect(SimToken t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedSimToken>(current, t, getPosition());
}

Expected<string> SimParser::reference()
{
	auto s = lexer.getLastIdentifier();
	if (auto e = expect(SimToken::Ident); !e)
		return e.takeError();
	return s;
}

Expected<int> SimParser::integer()
{
	bool minus = accept<SimToken::Minus>();

	int b = lexer.getLastInt();
	if (auto e = expect(SimToken::Integer); !e)
		return e.takeError();

	return minus ? -b : b;
}

Expected<float> SimParser::floatingPoint()
{
	bool minus = accept<SimToken::Minus>();

	float b = lexer.getLastFloat();
	if (auto e = expect(SimToken::Float); !e)
		return e.takeError();

	return minus ? -b : b;
}

Expected<SimConst<int>> SimParser::intVector()
{
	SmallVector<int, 3> args;
	if (auto e = expect(SimToken::LCurly); !e)
		return e.takeError();

	auto val = integer();
	if (!val)
		return val.takeError();

	args.push_back(*val);

	while (accept(SimToken::Comma))
	{
		auto val = integer();
		if (!val)
			return val.takeError();

		args.push_back(*val);
	}

	if (auto e = expect(SimToken::RCurly); !e)
		return e.takeError();

	return SimConst<int>(move(args));
}

Expected<SimConst<float>> SimParser::floatVector()
{
	SmallVector<float, 3> args;
	if (auto e = expect(SimToken::LCurly); !e)
		return e.takeError();

	auto val = floatingPoint();
	if (!val)
		return val.takeError();

	args.push_back(*val);

	while (accept(SimToken::Comma))
	{
		auto val = floatingPoint();
		if (!val)
			return val.takeError();

		args.push_back(*val);
	}

	if (auto e = expect(SimToken::RCurly); !e)
		return e.takeError();

	return SimConst<float>(move(args));
}

Expected<SimConst<bool>> SimParser::boolVector()
{
	SmallVector<bool, 3> args;
	if (auto e = expect(SimToken::LCurly); !e)
		return e.takeError();

	bool b = static_cast<bool>(lexer.getLastInt());
	args.push_back(b);

	if (auto e = expect(SimToken::Integer); !e)
		return e.takeError();

	while (accept(SimToken::Comma))
	{
		bool b = static_cast<bool>(lexer.getLastInt());
		args.push_back(b);

		if (auto e = expect(SimToken::Integer); !e)
			return e.takeError();
	}

	if (auto e = expect(SimToken::RCurly); !e)
		return e.takeError();

	return SimConst<bool>(move(args));
}

Expected<vector<size_t>> SimParser::typeDimensions()
{
	vector<size_t> v;
	if (auto e = expect(SimToken::LSquare); !e)
		return e.takeError();

	if (accept<SimToken::RSquare>())
		return v;

	v.emplace_back(lexer.getLastInt());
	if (auto e = expect(SimToken::Integer); !e)
		return e.takeError();

	while (accept(SimToken::Comma))
	{
		v.emplace_back(lexer.getLastInt());
		if (auto e = expect(SimToken::Integer); !e)
			return e.takeError();
	}

	if (auto e = expect(SimToken::RSquare); !e)
		return e.takeError();

	return v;
}

Expected<tuple<SimExpKind, vector<SimExp>>> SimParser::operation()
{
	if (auto e = expect(SimToken::LPar); !e)
		return e.takeError();

	SimExpKind kind;

	if (accept<SimToken::Minus>())
		kind = SimExpKind::sub;
	else if (accept<SimToken::Multiply>())
		kind = SimExpKind::mult;
	else if (accept<SimToken::Division>())
		kind = SimExpKind::divide;
	else if (accept<SimToken::GreaterEqual>())
		kind = SimExpKind::greaterEqual;
	else if (accept<SimToken::GreaterThan>())
		kind = SimExpKind::greaterThan;
	else if (accept<SimToken::LessEqual>())
		kind = SimExpKind::lessEqual;
	else if (accept<SimToken::LessThan>())
		kind = SimExpKind::less;
	else if (accept<SimToken::OperatorEqual>())
		kind = SimExpKind::equal;
	else if (accept<SimToken::Modulo>())
		kind = SimExpKind::module;
	else if (accept<SimToken::Exponential>())
		kind = SimExpKind::elevation;
	else if (accept<SimToken::Ternary>())
		kind = SimExpKind::conditional;
	else if (accept<SimToken::Not>())
		kind = SimExpKind::negate;
	else if (auto e = expect(SimToken::Plus); !e)
		return e.takeError();
	else
		kind = SimExpKind::add;

	vector<SimExp> args;

	size_t arity = SimExp::Operation::arityOfOp(kind);
	for (size_t a = 0; a < arity; a++)
	{
		auto arg = expression();
		if (!arg)
			return arg.takeError();
		args.emplace_back(move(*arg));
		if (a != arity - 1)
			if (auto e = expect(SimToken::Comma); !e)
				return e.takeError();
	}

	if (auto e = expect(SimToken::RPar); !e)
		return e.takeError();

	return tuple(kind, move(args));
}

Expected<SimType> SimParser::type()
{
	BultinSimTypes type;
	if (accept<SimToken::BoolKeyword>())
		type = BultinSimTypes::BOOL;
	else if (accept<SimToken::IntKeyword>())
		type = BultinSimTypes::INT;
	else if (auto e = expect(SimToken::FloatKeyword); e)
		type = BultinSimTypes::FLOAT;
	else
		return e.takeError();

	auto dim = typeDimensions();
	if (!dim)
		return dim.takeError();

	return SimType(type, std::move(*dim));
}

Expected<vector<SimExp>> SimParser::args()
{
	vector<SimExp> args;
	if (auto e = expect(SimToken::LPar); !e)
		return e.takeError();
	if (accept<SimToken::RPar>())
		return args;

	auto exp = expression();
	if (!exp)
		return exp.takeError();
	args.emplace_back(move(*exp));

	while (accept(SimToken::Comma))
	{
		auto exp = expression();
		if (!exp)
			return exp.takeError();
		args.emplace_back(move(*exp));
	}

	if (auto e = expect(SimToken::RPar); !e)
		return e.takeError();

	return args;
}

Expected<SimCall> SimParser::call()
{
	if (auto e = expect(SimToken::CallKeyword); !e)
		return e.takeError();

	auto fname = lexer.getLastIdentifier();
	if (auto e = expect(SimToken::Ident); !e)
		return e.takeError();

	auto t = type();
	if (!t)
		return t.takeError();

	auto argVec = args();
	if (!argVec)
		return argVec.takeError();

	SmallVector<unique_ptr<SimExp>, 3> vec;
	for (auto& exp : *argVec)
		vec.emplace_back(std::make_unique<SimExp>(move(exp)));

	return SimCall(move(fname), move(vec), move(*t));
}

Expected<StringMap<SimExp>> SimParser::initSection()
{
	StringMap<SimExp> map;
	if (auto e = expect(SimToken::InitKeyword); !e)
		return e.takeError();

	while (current != SimToken::End && current != SimToken::UpdateKeyword)
	{
		auto stat = statement();
		if (!stat)
			return stat.takeError();

		auto [name, exp] = move(*stat);
		if (map.find(name) != map.end())
			return make_error<UnexpectedSimToken>(
					SimToken::Ident, SimToken::Ident, getPosition());
		map.try_emplace(move(name), move(exp));
	}

	return map;
}

Expected<StringMap<SimExp>> SimParser::updateSection()
{
	StringMap<SimExp> map;
	if (auto e = expect(SimToken::UpdateKeyword); !e)
		return e.takeError();

	while (current != SimToken::End)
	{
		auto stat = statement();
		if (!stat)
			return stat.takeError();

		auto [name, exp] = move(*stat);
		if (map.find(name) != map.end())
			return make_error<UnexpectedSimToken>(
					SimToken::Ident, SimToken::Ident, getPosition());
		map.try_emplace(move(name), move(exp));
	}

	return map;
}

Expected<tuple<StringMap<SimExp>, StringMap<SimExp>>> SimParser::simulation()
{
	auto initSect = initSection();
	if (!initSect)
		return initSect.takeError();

	auto updateSect = updateSection();
	if (!updateSect)
		return updateSect.takeError();

	return tuple(move(*initSect), move(*updateSect));
}

Expected<tuple<string, SimExp>> SimParser::statement()
{
	auto name = lexer.getLastIdentifier();
	if (auto e = expect(SimToken::Ident); !e)
		return e.takeError();

	if (auto e = expect(SimToken::Assign); !e)
		return e.takeError();

	auto exp = expression();
	if (!exp)
		return exp.takeError();

	return tuple(move(name), move(*exp));
}

Expected<SimExp> SimParser::expression()
{
	auto tp = type();
	if (!tp)
		return tp.takeError();
	if (current == SimToken::Ident)
	{
		auto ref = reference();
		if (!ref)
			return ref.takeError();

		return SimExp(move(*ref), move(*tp));
	}

	if (current == SimToken::LCurly)
	{
		if (tp->getBuiltin() == BultinSimTypes::BOOL)
		{
			auto cont = boolVector();
			if (!cont)
				return cont.takeError();

			return SimExp(move(*cont), move(*tp));
		}
		if (tp->getBuiltin() == BultinSimTypes::INT)
		{
			auto cont = intVector();
			if (!cont)
				return cont.takeError();

			return SimExp(move(*cont), move(*tp));
		}
		if (tp->getBuiltin() == BultinSimTypes::FLOAT)
		{
			auto cont = floatVector();
			if (!cont)
				return cont.takeError();

			return SimExp(move(*cont), move(*tp));
		}
	}

	if (current == SimToken::CallKeyword)
	{
		auto c = call();
		if (!c)
			return c;

		return SimExp(move(*c), move(*tp));
	}

	if (current == SimToken::LPar)
	{
		auto op = operation();
		if (!op)
			return op.takeError();

		auto [kind, args] = move(*op);

		SmallVector<unique_ptr<SimExp>, 3> vec;
		for (auto& exp : args)
			vec.push_back(std::make_unique<SimExp>(move(exp)));

		return SimExp(kind, move(vec), move(*tp));
	}

	return make_error<UnexpectedSimToken>(current, SimToken::LPar, getPosition());
}
