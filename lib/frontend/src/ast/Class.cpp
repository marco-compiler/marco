#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Class::Class(ASTNodeKind kind,
						 SourcePosition location,
						 llvm::StringRef name)
		: ASTNodeCRTP<Class>(kind, std::move(location)),
			name(name.str())
{
	assert(!this->name.empty());
}

Class::Class(const Class& other)
		: ASTNodeCRTP<Class>(static_cast<ASTNodeCRTP<Class>&>(*this)),
			name(other.name)
{
	assert(!this->name.empty());
}

Class::Class(Class&& other) = default;

Class::~Class() = default;

Class& Class::operator=(const Class& other)
{
	if (this != &other)
	{
		static_cast<ASTNodeCRTP<Class>&>(*this) =
				static_cast<const ASTNodeCRTP<Class>&>(other);

		this->name = other.name;
	}

	return *this;
}

Class& Class::operator=(Class&& other) = default;

namespace modelica::frontend
{
	void swap(Class& first, Class& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Class>&>(first),
				 static_cast<impl::ASTNodeCRTP<Class>&>(second));

		using std::swap;
		swap(first.name, second.name);
	}
}

llvm::StringRef Class::getName() const
{
	return name;
}
