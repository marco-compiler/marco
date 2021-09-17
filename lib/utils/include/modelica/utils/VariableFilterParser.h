//
// Created by Ale on 26/06/2021.
//

#ifndef PARSER_LEXER_M_VF_VARIABLEFILTERPARSER_H
#define PARSER_LEXER_M_VF_VARIABLEFILTERPARSER_H

#include <regex>
#include <iostream>
#include <cstring>
#include <list>
#include <llvm/ADT/StringRef.h>
#include "VariableFilter.h"
#include "llvm/Support/Regex.h"

using namespace std;
namespace modelica {

    class VariableFilterParser {

        [[maybe_unused]] static void throwGenericError(void);

        static void throwError(string msg);

        //	LEXER 	//
        //============================================================================================================//
        // The lexer returns tokens [0-255] if it is an unknown character, otherwise
        // one
        // of these for known things.
        enum Token {
            tok_eof = -1,

            // commands
            tok_der = -2,
            tok_der_close = -3,

            // primary
            tok_identifier = -4,

            //array support
            tok_sq_bracket_l = -5,
            tok_sq_bracket_r = -6,
            tok_comma = -7,
            tok_colon = -8,
            tok_rng_value = -9,

            //regex support
            tok_regex_expr = -10


        };

        std::string IdentifierStr;     // Filled in if tok_identifier
        std::string RegexStr; //filled in if tok_regex_exp
        const char *inputString;
        unsigned short i = 0;
        bool parsingArray;
        string rangeValue;

        bool isValidIDChar(char c);

        /// gettok - Return the next token from the input string. T O K E N I Z E R
        int gettok();

        // === AST / PARSER === //
        //============================================================================================================//

        /// ExprAST - Base class for all expression nodes.
        class ExprAST {
        public:
            virtual ~ExprAST() {}
        };

        class VariableExprAST : public ExprAST {
            string identifier;

        public:
            VariableExprAST(const string &identifier) : identifier(identifier) {}

            string getIdentifier() const { return identifier; }
        };

        class ArrayRangeAST : public ExprAST {
            const int lvalue, rvalue;

        public:
            ArrayRangeAST(const int lvalue, const int rvalue) : lvalue(lvalue), rvalue(rvalue) {}

            int getLvalue() const {
                return lvalue;
            }

            int getRvalue() const {
                return rvalue;
            }


        };

        class ArrayExprAST : public ExprAST {

        public:

            ArrayExprAST(const VariableExprAST &arrayVariableIdentifier, uint16_t dimension,
                         list<ArrayRangeAST> &ranges)
                    : arrayVariableIdentifier(arrayVariableIdentifier), dimension(dimension), _ranges(ranges) {}

            const VariableExprAST getArrayVariableIdentifier() const {
                return arrayVariableIdentifier;
            }

            uint16_t getDimension() const {
                return dimension;
            }

            list<ArrayRangeAST> getRanges() const {
                return _ranges;
            }


        private:

            VariableExprAST arrayVariableIdentifier;
            uint16_t dimension;
            list<ArrayRangeAST> _ranges;
        };

        class DerivativeExprAST : public ExprAST {
            VariableExprAST derivedVariable;


        public:
            explicit DerivativeExprAST(VariableExprAST derivedVariable)
                    : derivedVariable(std::move(derivedVariable)) {
            }

            const VariableExprAST &getDerivedVariable() const {
                return derivedVariable;
            }
        };

        class RegexExprAST : public ExprAST {
            string regex;

        public:
            explicit RegexExprAST(string regex) : regex(std::move(regex)) {}

            const string &getRegex() const {
                return regex;
            }
        };

        /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the
        /// current
        /// token the parser is looking at.  getNextToken reads another token from
        /// the lexer and updates CurTok with its results.
        int CurTok;

        int getNextToken() {
            return CurTok = gettok();
        }

        //===========================================GRAMMAR PRODUCTIONS=================================================//
        //===============================================================================================================//
        /// var ::= string aAzZ09_
        std::unique_ptr<VariableExprAST> ParseVariableExpr() {
            std::string idName = IdentifierStr;
            auto result = std::make_unique<VariableExprAST>(idName);
            // getNextToken();	 // consume the number
            return std::move(result);
        }

        std::unique_ptr<DerivativeExprAST> ParseDerivative() {
            getNextToken();     // eat "der("

            if (CurTok >= 0) throwError("expecting derivative identifier");
            else if (CurTok == tok_der) throwError("nested derivatives are not allowed");
            // create derivative
            std::string IdName = IdentifierStr;
            auto Result = std::make_unique<DerivativeExprAST>(IdName);
            getNextToken();     // consume )
            return std::move(Result);

        }

        std::unique_ptr<ArrayExprAST> ParseArray() {

            //indices begin
            if (getNextToken() != tok_sq_bracket_l) { //[
                throwError("expecting an array opening bracket '['");
            }
            //cout << "\n opening bracket [";
            std::string arrayName = IdentifierStr;
            uint16_t dimension = 0;

            std::list<ArrayRangeAST> local_ranges;

            int rtemp, ltemp;
            do {
                //cout << "\n\tnew range:: ";
                if (getNextToken() != tok_rng_value) throwError("expecting a range left value");           //left-value
                // cout << "lval=" << rangeValue;
                ltemp = 0 == strncmp(rangeValue.c_str(), "$", 1) ? -1 : atoi(rangeValue.c_str());

                if (getNextToken() != tok_colon) throwError("expecting a separating colon ':'");      // :
                // cout << " : ";
                if (getNextToken() != tok_rng_value) throwError("expecting a range right value");    //right value
                // cout << "r-val=" << rangeValue;
                rtemp = 0 == strncmp(rangeValue.c_str(), "$", 1) ? -1 : atoi(rangeValue.c_str());

                ArrayRangeAST rang3(ltemp, rtemp);
                local_ranges.push_back(rang3);
                dimension++;

            } while (getNextToken() == tok_comma);

            //indices end
            if (CurTok != tok_sq_bracket_r) throwError("expecting a closing bracket ']'");
            //cout << " \nclosing bracket ]" << endl;

            auto result = std::make_unique<ArrayExprAST>(arrayName, dimension, local_ranges);
            //list<ArrayRangeAST> rntest = result->getRanges(); OK
            return result;

        }

        std::unique_ptr<RegexExprAST> ParseRegex() {

            std::string regex = RegexStr;
            llvm::StringRef regexRef(regex);
            if (regex.empty()) throwError("provided regex is empty");
            llvm::Regex regexObj(regexRef); //try to build a regex to see if it's correct
            if (!regexObj.isValid()) throwError("provided regex is not valid");
            auto result = std::make_unique<RegexExprAST>(regex);
            return result;
        }


    private:

        string inputStringReference;

        void lexerReset();

        void displayWarning(std::string msg);


    public:
        explicit VariableFilterParser(string commandLineString)
                : inputStringReference(std::move(commandLineString)) {
        }

        explicit VariableFilterParser() {}

        void parseCommandLine(string commandLineArguments, VariableFilter &vf);

        /** parses the current input string and carries out parsing into var-regex-array-derivative by
         * adding a tracker in the VariableFilter vf*/
        void parseExpressionElement(VariableFilter &vf);;
    };

}


#endif //PARSER_LEXER_M_VF_VARIABLEFILTERPARSER_H
