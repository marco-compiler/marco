#include <modelica/frontend/ClassContainer.hpp>

using namespace modelica;

ClassContainer::ClassContainer(Function function) : content(function) {}

ClassContainer::ClassContainer(Class model) : content(model) {}

void ClassContainer::dump() const { dump(llvm::outs(), 0); }

void ClassContainer::dump(llvm::raw_ostream& os, size_t indents) const
{
	visit([&](const auto& obj) { obj.dump(os, indents); });
}
