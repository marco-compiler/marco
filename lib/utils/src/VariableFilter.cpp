//
// Created by Ale on 21/06/2021.
//

#include "../include/modelica/utils/VariableFilter.h"

void modelica::VariableFilter::dump() {
    for (int s = 0; s < 12; ++s) printf("#");
    cout << "\n *** TRACKED VARIABLES *** : " << endl;

    for (const auto &vt : _variables) {
        vt.dump();
    }
    for (int s = 0; s < 12; ++s) printf("#");
    cout << "\n *** TRACKED REGEX(s) *** : " << endl;
    for (const auto &regex : _regex) {
        printf("Regex: /%s/", regex.c_str());
    }

}

void modelica::VariableFilter::addRegexString(string regex) {
    _regex.push_back(regex);
}

void modelica::VariableFilter::addVariable(VariableTracker var) {
    _variables.push_back(var);
}
