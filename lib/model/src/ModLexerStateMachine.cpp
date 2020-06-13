#include "modelica/model/ModLexerStateMachine.hpp"

using namespace modelica;
using State = ModLexerStateMachine::State;
using namespace llvm;
using namespace std;

std::string modelica::tokenToString(ModToken token)
{
	switch (token)
	{
		case ModToken::Comma:
			return "comma";
		case ModToken::BoolKeyword:
			return "BOOL";
		case ModToken::FloatKeyword:
			return "FLOAT";
		case ModToken::IntKeyword:
			return "INT";
		case ModToken::None:
			return "None";
		case ModToken::Begin:
			return "Begin";
		case ModToken::Ident:
			return "Ident";
		case ModToken::Bool:
			return "Bool";
		case ModToken::Integer:
			return "Integer";
		case ModToken::Float:
			return "Float";
		case ModToken::Error:
			return "Error";
		case ModToken::CallKeyword:
			return "call";
		case ModToken::InitKeyword:
			return "Init";
		case ModToken::UpdateKeyword:
			return "Update";
		case ModToken::AtKeyword:
			return "At";
		case ModToken::ConstantKeyword:
			return "const";
		case ModToken::StateKeyword:
			return "state";

		case ModToken::Modulo:
			return "module";
		case ModToken::Multiply:
			return "Multiply";
		case ModToken::Division:
			return "Division";
		case ModToken::Plus:
			return "Plus";
		case ModToken::Minus:
			return "Minus";
		case ModToken::OperatorEqual:
			return "OperatorEqual";
		case ModToken::LessThan:
			return "LessThan";
		case ModToken::LessEqual:
			return "LessEqual";
		case ModToken::GreaterThan:
			return "GreaterThan";
		case ModToken::GreaterEqual:
			return "GreaterEqual";
		case ModToken::LPar:
			return "LPar";
		case ModToken::RPar:
			return "RPar";
		case ModToken::LSquare:
			return "LSquare";
		case ModToken::RSquare:
			return "RSquare";
		case ModToken::LCurly:
			return "LCurly";
		case ModToken::RCurly:
			return "RCurly";
		case ModToken::Exponential:
			return "Exponential";
		case ModToken::Assign:
			return "Assign";
		case ModToken::Ternary:
			return "Ternary";
		case ModToken::Not:
			return "Not";
		case ModToken::ForKeyword:
			return "For";
		case ModToken::IndKeyword:
			return "Ind";
		case ModToken::BackwardKeyword:
			return "Backward";
		case ModToken::TemplateKeyword:
			return "Template";
		case ModToken::MatchedKeyword:
			return "Matched";

		case ModToken::End:
			return "End";
	}
	assert(false && "Unreachable");	 // NOLINT
	return "Unkown ModToken";
}

ModLexerStateMachine::ModLexerStateMachine(char first)
		: state(State::Normal),
			current('\0'),
			next(first),
			currentToken(Token::Begin),
			lastIdentifier(""),
			lineNumber(1),
			columnNumber(0)
{
	keywordMap["call"] = ModToken::CallKeyword;
	keywordMap["init"] = ModToken::InitKeyword;
	keywordMap["update"] = ModToken::UpdateKeyword;
	keywordMap["BOOL"] = ModToken::BoolKeyword;
	keywordMap["FLOAT"] = ModToken::FloatKeyword;
	keywordMap["INT"] = ModToken::IntKeyword;
	keywordMap["for"] = ModToken::ForKeyword;
	keywordMap["ind"] = ModToken::IndKeyword;
	keywordMap["at"] = ModToken::AtKeyword;
	keywordMap["const"] = ModToken::ConstantKeyword;
	keywordMap["backward"] = ModToken::BackwardKeyword;
	keywordMap["template"] = ModToken::TemplateKeyword;
	keywordMap["matched"] = ModToken::MatchedKeyword;
	keywordMap["state"] = ModToken::StateKeyword;

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

ModToken ModLexerStateMachine::charToToken(char c) const
{
	if (auto iter = symbols.find(c); iter != symbols.end())
		return iter->second;

	return Token::Error;
}

ModToken ModLexerStateMachine::tryScanSymbol()
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
ModToken ModLexerStateMachine::scan<State::ParsingNum>()
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
ModToken ModLexerStateMachine::scan<State::ParsingId>()
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
ModToken ModLexerStateMachine::scan<State::ParsingComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '*' && next == '/')
		state = State::EndOfComment;

	return Token::None;
}

template<>
ModToken ModLexerStateMachine::scan<State::EndOfComment>()
{
	state = State::Normal;
	return Token::None;
}

template<>
ModToken ModLexerStateMachine::scan<State::ParsingLineComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '\n')
		state = State::Normal;

	return Token::None;
}

template<>
ModToken ModLexerStateMachine::scan<State::Normal>()
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
ModToken ModLexerStateMachine::scan<State::ParsingFloatExponent>()
{
	if (isDigit(current))
		lastNum.addExponential(current - '0');

	if (isDigit(next))
		return Token::None;

	state = State::Normal;
	return Token::Float;
}

template<>
ModToken ModLexerStateMachine::scan<State::ParsingFloatExponentialSign>()
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
ModToken ModLexerStateMachine::scan<State::ParsingFloat>()
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

ModToken ModLexerStateMachine::stringToToken(const std::string& lookUp) const
{
	if (auto iter = keywordMap.find(lookUp); iter != keywordMap.end())
		return iter->getValue();

	return Token::Ident;
}

ModToken ModLexerStateMachine::step(char c)
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
