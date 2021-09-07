//
// Created by Ale on 21/06/2021.
//

#ifndef PARSER_LEXER_M_VF_VARIABLETRACKER_H
#define PARSER_LEXER_M_VF_VARIABLETRACKER_H

#include <string>
#include <list>
#include <iostream>

using namespace std;

/**
 * represents an array range, $ special character is '-1'
 */
class Range {

public:
    Range(int leftValue, int rightValue);
    bool noUpperBound() const; //-1 represents "unbounded"
    bool noLowerBound() const;
    int leftValue, rightValue;
};

/**
 * Keeps tracks of a single variable, array or derivative that has been specified by command line arguments.
 */
class VariableTracker {
public:

    VariableTracker(const string &name, const bool isArray, const bool isDerivative, const unsigned int dim);

    void setRanges(const list <Range> &ranges);

    const string &getName() const;

    const bool getIsArray() const;

    const bool getIsDerivative() const;

    const uint16_t getDim() const;

    const list <Range> &getRanges() const;

    const Range getRangeOfDimensionN(unsigned int N) {
        int i = 0;
        for (auto &range : _ranges){
            if (i == N) {
                return range;
            }
            i++;
        }
    }

    void dump(void) const;

private:

    string _name;
    bool _isArray;
    bool _isDerivative;
    unsigned int _dim;

    //-1 means "all"
    list <Range> _ranges;


};


#endif //PARSER_LEXER_M_VF_VARIABLETRACKER_H
