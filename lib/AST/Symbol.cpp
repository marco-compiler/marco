#include <marco/AST/Symbol.h>

using namespace marco::ast;

Symbol::Symbol() = default;

Symbol::Symbol(Class& cls): content(&cls) {}

Symbol::Symbol(Member& member): content(&member) {}

Symbol::Symbol(Induction& induction): content(&induction) {}
