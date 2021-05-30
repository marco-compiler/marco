#include <modelica/frontend/Symbol.hpp>

using namespace modelica::frontend;

Symbol::Symbol() = default;

Symbol::Symbol(Class& cls): content(&cls) {}

Symbol::Symbol(Member& member): content(&member) {}

Symbol::Symbol(Induction& induction): content(&induction) {}
