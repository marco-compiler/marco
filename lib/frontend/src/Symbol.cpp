#include <modelica/frontend/Symbol.hpp>

using namespace modelica;

Symbol::Symbol() = default;

Symbol::Symbol(Function& function): content(&function) {}

Symbol::Symbol(Class& model): content(&model) {}

Symbol::Symbol(Member& mem): content(&mem) {}

Symbol::Symbol(Induction& mem): content(&mem) {}
