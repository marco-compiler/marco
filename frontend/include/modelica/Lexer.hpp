#pragma once

#include <cctype>
#include <functional>
#include <iterator>

//#include "modelica/LexerStateMachine.hpp"

namespace modelica
{
	/**
	 * Iterator over lexer, the iterator is a input iterator
	 * so if it is advanced it will modify the state of the lexer
	 * by scanning for the next token.
	 *
	 */
	template<typename Lexer>
	class IteratorLexer
	{
		public:
		using iterator_category = std::input_iterator_tag;
		using value_type = typename Lexer::Token;
		using difference_type = std::ptrdiff_t;
		using pointer = typename Lexer::Token*;
		using reference = typename Lexer::Token&;

		/**
		 * Creates a new iterator operating on lex that has the starting value of
		 * tok.
		 *
		 * IteratorLexer(lex, lex.getCurret()) is equal to lex.begin() the second
		 * version is preferred.
		 */
		IteratorLexer(Lexer& lex, value_type tok): lexer(lex), token(tok) {}

		/**
		 * Two iterator are the same if when deferenced have the same value and if
		 * they are referring to the same lexer.
		 */
		bool operator==(const IteratorLexer& other) const
		{
			return (other.token == token) && (&lexer == &other.lexer);
		}

		/**
		 * Logical not of operator==
		 */
		bool operator!=(const IteratorLexer& other) const
		{
			return !(*this == other);
		}

		/**
		 * Once a iterator has been constructed or adnvaced it will keep the same
		 * value until it is adanved again.
		 */
		value_type operator*() const { return token; }

		/**
		 * When advanced the state of the underlaying lexer will be adanved as well,
		 * by scanning for the next token. So it two different iterators are
		 * operating on the same lexer they will have an effect on each other.
		 *
		 * It's fine two have multiple iterators if only one of them is ever
		 * advanced, since they will keep the values until ++ is invoked.
		 */
		IteratorLexer operator++()
		{
			token = lexer.scan();
			return *this;
		}

		private:
		Lexer& lexer;
		value_type token;
	};

	/**
	 * The lexer does NOT handle the life scope of the source string.
	 * A lexer is a wrapper around a finite state machine that keeps
	 * feeding the machine a character at the time util a token is
	 * returned.
	 *
	 * Lexer extends the state machine so every public method there is
	 * still accessible.
	 *
	 * The second template argument is the iterator type from wich the
	 * lexer will read from.
	 *
	 * The state machine must implement a constructor that accepts a
	 * char, a inner type named Token that is the returned type of iterators over
	 * the lexer, the getCurrent() that returns a token. Token::End must exist.
	 * Token::None must exist.
	 */
	template<typename StateMachine>
	class Lexer: public StateMachine
	{
		public:
		/**
		 * The type of objects that the state machine is allowed to return.
		 */
		using Token = typename StateMachine::Token;

		/**
		 * Makes a lexer out of a iteratable type.
		 * The iteratable type will be copied, so it's better if it is cheap to
		 * copy.
		 */
		template<typename Iterator>
		Lexer(const Iterator&& iter)
				: StateMachine(*iter),
					getNext([iter = iter]() mutable -> char {
						iter++;
						return *iter;
					}),
					lastChar(*iter)
		{
		}

		/**
		 * Makes a lexer out of a string taken by reference. the string is not
		 * copied, it is just used to extract a iterator.
		 *
		 * DO NOT CHANGE THIS TO A llvm::stringRef, stringRef is not null terminated
		 */
		Lexer(const std::string& str)
				: StateMachine(str[0]),
					getNext([iter = str.begin()]() mutable -> char {
						iter++;
						return *iter;
					}),
					lastChar(str[0])
		{
		}

		/**
		 * Advances the reading of the state machine by one character until the
		 * state machine provides a token.
		 *
		 * it will not be advanced if '\0' was already reached, but the same '\0'
		 * will be forwarded to the state machine, it's up to the machine to return
		 * a token.
		 */
		Token scan()
		{
			auto tok = Token::None;
			while (tok == Token::None)
			{
				if (lastChar != '\0')
					lastChar = getNext();

				tok = StateMachine::step(lastChar);
			}

			return tok;
		}

		/**
		 * Returns a iterator operating on this lexer that has as value
		 * the current value of the state machine.
		 */
		IteratorLexer<Lexer> begin()
		{
			return IteratorLexer<Lexer>(*this, StateMachine::getCurrent());
		}

		/**
		 * Retunts a iterator operating on this lexer loaded with Token::End.
		 */
		IteratorLexer<Lexer> end()
		{
			return IteratorLexer<Lexer>(*this, Token::End);
		}

		private:
		std::function<char()> getNext;
		char lastChar;
	};

}	// namespace modelica
