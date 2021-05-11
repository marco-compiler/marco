#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

// TODO type
ReferenceAccess::ReferenceAccess(SourcePosition location,
																 Type type,
																 llvm::StringRef name,
																 bool globalLookup,
																 bool dummy)
		: ExpressionCRTP<ReferenceAccess>(
					ASTNodeKind::EXPRESSION_REFERENCE_ACCESS, std::move(location), std::move(type)),
			name(name.str()),
			globalLookup(globalLookup),
			dummyVar(dummy)
{
}

ReferenceAccess::ReferenceAccess(const ReferenceAccess& other)
		: ExpressionCRTP<ReferenceAccess>(static_cast<ExpressionCRTP&>(*this)),
			name(other.name),
			globalLookup(other.globalLookup),
			dummyVar(other.dummyVar)
{
}

ReferenceAccess::ReferenceAccess(ReferenceAccess&& other) = default;

ReferenceAccess::~ReferenceAccess() = default;

ReferenceAccess& ReferenceAccess::operator=(const ReferenceAccess& other)
{
	ReferenceAccess result(other);
	swap(*this, result);
	return *this;
}

ReferenceAccess& ReferenceAccess::operator=(ReferenceAccess&& other) = default;

namespace modelica::frontend
{
	void swap(ReferenceAccess& first, ReferenceAccess& second)
	{
		swap(static_cast<impl::ExpressionCRTP<ReferenceAccess>&>(first),
				 static_cast<impl::ExpressionCRTP<ReferenceAccess>&>(second));

		using std::swap;
		swap(first.name, second.name);
		swap(first.globalLookup, second.globalLookup);
		swap(first.dummyVar, second.dummyVar);
	}
}

void ReferenceAccess::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "reference access: " << (hasGlobalLookup() ? "." : "") << getName() << "\n";

	os.indent(indents);
	os << "type: ";
	getType().dump(os);
	os << "\n";
}

bool ReferenceAccess::isLValue() const
{
	return true;
}

bool ReferenceAccess::operator==(const ReferenceAccess& other) const
{
	return globalLookup == other.globalLookup &&
				 name == other.name &&
				 dummyVar == other.dummyVar;
}

bool ReferenceAccess::operator!=(const ReferenceAccess& other) const
{
	return !(*this == other);
}

llvm::StringRef ReferenceAccess::getName() const
{
	return name;
}

void ReferenceAccess::setName(llvm::StringRef newName)
{
	this->name = newName.str();
}

bool ReferenceAccess::hasGlobalLookup() const
{
	return globalLookup;
}

bool ReferenceAccess::isDummy() const
{
	return dummyVar;
}

ReferenceAccess ReferenceAccess::dummy(SourcePosition location)
{
	return ReferenceAccess(location, "", false, true);
}

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ReferenceAccess& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const ReferenceAccess& obj)
	{
		return obj.getName().str();
	}
}
