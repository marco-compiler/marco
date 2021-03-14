#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

struct LValueVisitor
{
	bool operator()(const Constant& obj) const { return false; }
	bool operator()(const ReferenceAccess& obj) const { return true; }
	bool operator()(const Operation& obj) const { return obj.isLValue(); }
	bool operator()(const Call& obj) const { return false; }
	bool operator()(const Tuple& obj) const { return false; }
};

Expression::Expression(Type type, Constant constant)
		: content(move(constant)),
			type(move(type))
{
}

Expression::Expression(Type type, ReferenceAccess access)
		: content(move(access)),
			type(move(type))
{
}

Expression::Expression(Type type, Operation operation)
		: content(move(operation)),
			type(move(type))
{
}

Expression::Expression(Type type, Call call)
		: content(move(call)),
			type(move(type))
{
}

Expression::Expression(Type type, Tuple tuple)
		: content(move(tuple)),
			type(move(type))
{
}

bool Expression::operator==(const Expression& other) const
{
	return type == other.type && content == other.content;
}

bool Expression::operator!=(const Expression& other) const
{
	return !(*this == other);
}

void Expression::dump() const { dump(outs(), 0); }

void Expression::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "type: ";
	getType().dump(os);
	os << "\n";

	visit([&](const auto& exp) { exp.dump(os, indents + 1); });
}

SourcePosition Expression::getLocation() const
{
	return visit([](const auto& value) { return value.getLocation(); });
}

bool Expression::isLValue() const { return visit(LValueVisitor()); }

Type& Expression::getType() { return type; }

const Type& Expression::getType() const { return type; }

void Expression::setType(Type tp) { type = move(tp); }

llvm::raw_ostream& modelica::operator<<(llvm::raw_ostream& stream, const Expression& obj)
{
	return stream << toString(obj);
}

std::string modelica::toString(const Expression& obj)
{
	return obj.visit([](const auto& obj) { return toString(obj); });
}
