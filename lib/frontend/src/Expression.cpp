#include <llvm/ADT/StringRef.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Expression::Expression(Type type, Constant constant)
		: content(move(constant)), type(move(type))
{
}

Expression::Expression(Type type, ReferenceAccess access)
		: content(move(access)), type(move(type))
{
}

Expression::Expression(Type type, Call call)
		: content(move(call)), type(move(type))
{
}

Expression::Expression(Tuple tuple): content(move(tuple)), type(Type::tuple())
{
}

Expression::Expression(Type type, OperationKind kind, Operation::Container args)
		: content(Operation(kind, move(args))), type(move(type))
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

	if (isA<Operation>())
	{
		get<Operation>().dump(os, indents + 1);
		return;
	}

	if (isA<Constant>())
	{
		get<Constant>().dump(os, indents + 1);
		return;
	}

	if (isA<ReferenceAccess>())
	{
		get<ReferenceAccess>().dump(os, indents + 1);
		return;
	}

	if (isA<Call>())
	{
		get<Call>().dump(os, indents + 1);
		return;
	}

	os << "\n";

	assert(false && "Unreachable");
}

bool Expression::isLValue() const
{
	if (isA<Operation>())
		return get<Operation>().isLValue();

	if (isA<ReferenceAccess>())
		return true;

	return false;
}

Type& Expression::getType() { return type; }

const Type& Expression::getType() const { return type; }

void Expression::setType(Type tp) { type = move(tp); }

Expression modelica::makeCall(Expression fun, llvm::ArrayRef<Expression> args)
{
	return Expression(Type::unknown(), Call(move(fun), move(args)));
}
