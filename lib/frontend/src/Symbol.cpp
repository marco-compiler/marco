#include <modelica/frontend/Symbol.hpp>

using namespace modelica;

Symbol::Symbol() = default;

Symbol::Symbol(Function& function): content(&function) {}

Symbol::Symbol(Class& model): content(&model) {}

Symbol::Symbol(Package& package): content(&package) {}

Symbol::Symbol(Record& record): content(&record) {}

Symbol::Symbol(Member& member): content(&member) {}

Symbol::Symbol(Induction& induction): content(&induction) {}
