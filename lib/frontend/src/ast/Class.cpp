#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Class::Class(DerFunction content)
		: content(std::move(content))
{
}

Class::Class(StandardFunction content)
		: content(std::move(content))
{
}

Class::Class(Model content)
		: content(std::move(content))
{
}

Class::Class(Package content)
		: content(std::move(content))
{
}

Class::Class(Record content)
		: content(std::move(content))
{
}

Class::Class(const Class& other)
		: content(other.content)
{
}

Class::Class(Class&& other) = default;

Class::~Class() = default;

Class& Class::operator=(const Class& other)
{
	return *this;
}

Class& Class::operator=(Class&& other) = default;

namespace modelica::frontend
{
	void swap(Class& first, Class& second)
	{
		using std::swap;
		swap(first.content, second.content);
	}
}

void Class::print(llvm::raw_ostream& os, size_t indents) const
{
	visit([&os, indents](const auto& value) {
		value.dump(os, indents + 1);
	});
}
