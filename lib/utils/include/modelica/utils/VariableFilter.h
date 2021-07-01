//
// Created by Ale on 21/06/2021.
//

#ifndef PARSER_LEXER_M_VF_VARIABLEFILTER_H
#define PARSER_LEXER_M_VF_VARIABLEFILTER_H

#include "VariableTracker.h"
#include <string>
#include <list>

#include <regex>
#include <iostream>

using namespace std;

/**
 *  Keeps tracks of a variables, arrays, derivatives (and regex for matching) we want to print.
 */
namespace modelica {
    class VariableFilter {
    public:
        void addVariable(VariableTracker var);

        void addRegexString(string regex);

        void dump();

        /**
         *
         * @param identifier the string that will be matched with all the regexes
         * @return true if there is a stored regular expression that matches the received identifier
         */
        bool matchesRegex(const string &identifier) {

            for (const auto &regexString : _regex) {
                std::regex regex(regexString, std::regex::ECMAScript);
                if (regex_match(identifier, regex)) {
                    return true;
                }
            }
            return false;
        }

        /**
         *
         * @param identifier the variable identifier we want to query
         * @return the variable tracker associated with that variable
         */
        VariableTracker lookupByIdentifier(const string &identifier) {
            if (std::any_of(_variables.begin(), _variables.end(), [&identifier](VariableTracker i) {
                                return i.getName().compare(identifier) ;
                            }
            )) {
                for (const auto &varTracker : _variables) {
                    if (varTracker.getName() == (identifier)) return varTracker;
                }
            }
        }

    private:
        list<VariableTracker> _variables;
        list<string> _regex;
    };
}


#endif //PARSER_LEXER_M_VF_VARIABLEFILTER_H
