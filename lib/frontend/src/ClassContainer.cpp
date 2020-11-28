#include <modelica/frontend/ClassContainer.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

ClassContainer::ClassContainer(Function function) : content(function) {}

ClassContainer::ClassContainer(Class model) : content(model) {}

void ClassContainer::dump() const { dump(outs(), 0); }

void ClassContainer::dump(raw_ostream& os, size_t indents) const
{
	visit([&](const auto& obj) { obj.dump(os, indents); });
}
