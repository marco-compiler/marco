#include <modelica/frontend/Symbol.hpp>

using namespace modelica::frontend;

Symbol::Symbol() = default;

Symbol::Symbol(DerFunction& function): content(&function) {}

Symbol::Symbol(StandardFunction& function): content(&function) {}

Symbol::Symbol(Model& model): content(&model) {}

Symbol::Symbol(Package& package): content(&package) {}

Symbol::Symbol(Record& record): content(&record) {}

Symbol::Symbol(Member& member): content(&member) {}

Symbol::Symbol(Induction& induction): content(&induction) {}
