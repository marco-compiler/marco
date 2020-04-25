#include "modelica/frontend/Parser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

Expected<bool> Parser::expect(Token t)
{
	if (accept(t))
		return true;

	return make_error<UnexpectedToken>(current, t, getPosition());
}

#include "modelica/utils/ParserUtils.hpp"
