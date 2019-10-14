#include "modelica/simulation/SimLexerStateMachine.hpp"

using namespace modelica;
using State = SimLexerStateMachine::State;
using namespace llvm;
using namespace std;

std::string modelica::tokenToString(SimToken token)
{
	switch (token)
	{
		case SimToken::Comma:
			return "comma";
		case SimToken::BoolKeyword:
			return "BOOL";
		case SimToken::FloatKeyword:
			return "FLOAT";
		case SimToken::IntKeyword:
			return "INT";
		case SimToken::None:
			return "None";
		case SimToken::Begin:
			return "Begin";
		case SimToken::Ident:
			return "Ident";
		case SimToken::Bool:
			return "Bool";
		case SimToken::Integer:
			return "Integer";
		case SimToken::Float:
			return "Float";
		case SimToken::Error:
			return "Error";
		case SimToken::CallKeyword:
			return "call";
		case SimToken::InitKeyword:
			return "Init";
		case SimToken::UpdateKeyword:
			return "Update";

		case SimToken::Modulo:
			return "module";
		case SimToken::Multiply:
			return "Multiply";
		case SimToken::Division:
			return "Division";
		case SimToken::Plus:
			return "Plus";
		case SimToken::Minus:
			return "Minus";
		case SimToken::OperatorEqual:
			return "OperatorEqual";
		case SimToken::LessThan:
			return "LessThan";
		case SimToken::LessEqual:
			return "LessEqual";
		case SimToken::GreaterThan:
			return "GreaterThan";
		case SimToken::GreaterEqual:
			return "GreaterEqual";
		case SimToken::LPar:
			return "LPar";
		case SimToken::RPar:
			return "RPar";
		case SimToken::LSquare:
			return "LSquare";
		case SimToken::RSquare:
			return "RSquare";
		case SimToken::LCurly:
			return "LCurly";
		case SimToken::RCurly:
			return "RCurly";
		case SimToken::Exponential:
			return "Exponential";
		case SimToken::Assign:
			return "Assign";
		case SimToken::Ternary:
			return "Ternary";
		case SimToken::Not:
			return "Not";

		case SimToken::End:
			return "End";
	}
	assert(false && "Unreachable");	 // NOLINT
	return "Unkown SimToken";
}

SimLexerStateMachine::SimLexerStateMachine(char first)
		: state(State::Normal),
			current('\0'),
			next(first),
			currentToken(Token::Begin),
			lastIdentifier(""),
			lineNumber(1),
			columnNumber(0)
{
	keywordMap["call"] = SimToken::CallKeyword;
	keywordMap["init"] = SimToken::InitKeyword;
	keywordMap["update"] = SimToken::UpdateKeyword;
	keywordMap["BOOL"] = SimToken::BoolKeyword;
	keywordMap["FLOAT"] = SimToken::FloatKeyword;
	keywordMap["INT"] = SimToken::IntKeyword;

	symbols['*'] = Token::Multiply;
	symbols[','] = Token::Comma;
	symbols['-'] = Token::Minus;
	symbols['+'] = Token::Plus;
	symbols['/'] = Token::Division;
	symbols['['] = Token::LSquare;
	symbols[']'] = Token::RSquare;
	symbols['{'] = Token::LCurly;
	symbols['}'] = Token::RCurly;
	symbols['('] = Token::LPar;
	symbols[')'] = Token::RPar;
	symbols['='] = Token::Assign;
	symbols['<'] = Token::LessThan;
	symbols['>'] = Token::GreaterThan;
	symbols['^'] = Token::Exponential;
	symbols['?'] = Token::Ternary;
	symbols['!'] = Token::Not;
}

SimToken SimLexerStateMachine::charToToken(char c) const
{
	if (auto iter = symbols.find(c); iter != symbols.end())
		return iter->second;

	return Token::Error;
}

SimToken SimLexerStateMachine::tryScanSymbol()
{
	state = State::IgnoreNextChar;

	if (current == '<' && next == '=')
		return Token::LessEqual;

	if (current == '>' && next == '=')
		return Token::GreaterEqual;

	if (next == '=' && current == '=')
		return Token::OperatorEqual;

	state = State::Normal;
	Token token = charToToken(current);
	if (token == Token::Error)
	{
		error = "Unexpeted character ";
		error.push_back(current);
	}

	return token;
}
static bool isDigit(char c) { return ('0' <= c && c <= '9'); }
static bool isNonDigit(char c)
{
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

template<>
SimToken SimLexerStateMachine::scan<State::ParsingNum>()
{
	if (isDigit(current))
		lastNum.addUpper(current - '0');

	if (current == '.')
	{
		state = State::ParsingFloat;
		return Token::None;
	}

	if (current == 'E' || current == 'e')
	{
		state = State::ParsingFloatExponentialSign;
		return Token::None;
	}

	auto isAccetable = [](char c) {
		return isDigit(c) || c == '.' || c == 'e' || c == 'E';
	};

	if (!isAccetable(next))
	{
		state = State::Normal;
		return Token::Integer;
	}

	return Token::None;
}

template<>
SimToken SimLexerStateMachine::scan<State::ParsingId>()
{
	lastIdentifier.push_back(current);

	if (!isDigit(next) && !isNonDigit(next))
	{
		state = State::Normal;
		return stringToToken(lastIdentifier);
	}

	return Token::None;
}

template<>
SimToken SimLexerStateMachine::scan<State::ParsingComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '*' && next == '/')
		state = State::EndOfComment;

	return Token::None;
}

template<>
SimToken SimLexerStateMachine::scan<State::EndOfComment>()
{
	state = State::Normal;
	return Token::None;
}

template<>
SimToken SimLexerStateMachine::scan<State::ParsingLineComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '\n')
		state = State::Normal;

	return Token::None;
}

template<>
SimToken SimLexerStateMachine::scan<State::Normal>()
{
	if (std::isspace(current) != 0)
		return Token::None;

	if (isNonDigit(current))
	{
		state = State::ParsingId;
		lastIdentifier = "";

		return scan<State::ParsingId>();
	}

	if (isDigit(current))
	{
		state = State::ParsingNum;
		lastNum = FloatLexer<defaultBase>();
		return scan<State::ParsingNum>();
	}

	if (current == '/' && next == '/')
	{
		state = State::ParsingLineComment;
		return Token::None;
	}

	if (current == '/' && next == '*')
	{
		state = State::ParsingComment;
		return Token::None;
	}

	if (current == '\0')
	{
		state = State::End;
		return Token::End;
	}

	return tryScanSymbol();
}

template<>
SimToken SimLexerStateMachine::scan<State::ParsingFloatExponent>()
{
	if (isDigit(current))
		lastNum.addExponential(current - '0');

	if (isDigit(next))
		return Token::None;

	state = State::Normal;
	return Token::Float;
}

template<>
SimToken SimLexerStateMachine::scan<State::ParsingFloatExponentialSign>()
{
	if (current == '-' || current == '+')
	{
		if (!isDigit(next))
		{
			error = "Exp sign must be followed by a number";
			state = State::Normal;
			return Token::Error;
		}
		state = State::ParsingFloatExponent;
		lastNum.setSign(current == '+');
		return Token::None;
	}

	if (isDigit(current))
	{
		state = State::ParsingFloatExponent;
		lastNum.addExponential(current - '0');
		return Token::None;
	}

	error = "Error unexpected char " + std::to_string(current) +
					" in floating number scan";
	state = State::Normal;
	return Token::Error;
}

template<>
SimToken SimLexerStateMachine::scan<State::ParsingFloat>()
{
	if (isDigit(current))
		lastNum.addLower(current - '0');

	if (current == 'E' || current == 'e')
	{
		state = State::ParsingFloatExponentialSign;
		return Token::None;
	}

	if (isDigit(next) || next == 'E' || next == 'e')
		return Token::None;

	state = State::Normal;
	return Token::Float;
}

SimToken SimLexerStateMachine::stringToToken(const std::string& lookUp) const
{
	if (auto iter = keywordMap.find(lookUp); iter != keywordMap.end())
		return iter->getValue();

	return Token::Ident;
}

SimToken SimLexerStateMachine::step(char c)
{
	advance(c);
	switch (state)
	{
		case (State::Normal):
			return scan<State::Normal>();
		case (State::ParsingComment):
			return scan<State::ParsingComment>();
		case (State::ParsingLineComment):
			return scan<State::ParsingLineComment>();
		case (State::EndOfComment):
			return scan<State::EndOfComment>();
		case (State::ParsingNum):
			return scan<State::ParsingNum>();
		case (State::ParsingFloat):
			return scan<State::ParsingFloat>();
		case (State::ParsingFloatExponentialSign):
			return scan<State::ParsingFloatExponentialSign>();
		case (State::ParsingFloatExponent):
			return scan<State::ParsingFloatExponent>();
		case (State::ParsingId):
			return scan<State::ParsingId>();
		case (State::End):
			return Token::End;
		case (State::IgnoreNextChar):
			state = State::Normal;
			return Token::None;
	}

	error = "Unandled Lexer State";
	return Token::Error;
}
