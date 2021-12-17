#ifndef MARCO_MATCHING_TEST_REALSCENARIOS_H
#define MARCO_MATCHING_TEST_REALSCENARIOS_H

#include <marco/modeling/Matching.h>

#include "MatchingCommon.h"

namespace marco::modeling::matching::test
{
  /**
   * var
   * x[3]
   *
   * equ
   * eq1 ; x[i] + x[i + 1] = 0 ; i[0:2)
   * eq2 ; x[2] = 0
   */
  MatchingGraph<Variable, Equation> testCase1();

  /**
   * var
   * x[4]
   *
   * equ
   * eq1 ; x[i] = 0 ; i[0:2)
   * eq2 ; x[i] = 0 ; i[2,4)
   */
  MatchingGraph<Variable, Equation> testCase2();

  /**
   * var
   * l[1]
   * h[1]
   * fl[1]
   * fh[1]
   * x[5]
   * y[5]
   * f[5]
   *
   * equ
   * eq1 ; l[0] + fl[0] = 0
   * eq2 ; fl[0] = 0
   * eq3 ; h[0] + fh[0] = 0
   * eq4 ; fh[0] = 0
   * eq5 ; fl[0] + f[i] + x[i] = 0 ; i[0:5)
   * eq6 ; fh[0] + f[i] + y[i] = 0 ; i[0:5)
   * eq7 ; f[i] = 0 ; i[0:5)
   */
  MatchingGraph<Variable, Equation> testCase3();

  /**
   * var
   * l[1]
   * h[1]
   * x[5]
   * f[6]
   *
   * equ
   * e1 ; l[0] + f[0] = 0
   * e2 ; f[0] = 0
   * e3 ; x[i] + f[i] + f[i+1] = 0 ; i[0:5)
   * e4 ; f[i] = 0                 ; i[1:5)
   * e5 ; h[0] + f[5] = 0
   * e6 ; f[5] = 0
   */
  MatchingGraph<Variable, Equation> testCase4();

  /**
   * var
   * x[5]
   * y[4]
   * z[5]
   *
   * equ
   * e1 ; x[i] = 10          ; i[0:5)
   * e2 ; y[i] = x[i+1]      ; i[0:4)
   * e3 ; z[i] = x[i] + y[i] ; i[0:4)
   * e4 ; z[4] = x[4]        ;
   */
  MatchingGraph<Variable, Equation> testCase5();

  /**
   * var
   * x[6]
   * y[3]
   *
   * equ
   * e1 ; x[i] + y[i] = 0 ; i[0:3)
   * e2 ; x[i] + y[1] = 0 ; i[0:6)
   */
  MatchingGraph<Variable, Equation> testCase6();

  /**
   * var
   * x[2]
   * y[1]
   * z[1]
   *
   * equ
   * e1 ; x[0] = 0
   * e2 ; x[1] + y[0] = 0
   * e3 ; y[0] + z[0] = 0
   * e4 ; y[0] + z[0] = 0
   */
  MatchingGraph<Variable, Equation> testCase7();

  /**
   * var
   * x[9]
   * y[3]
   *
   * equ
   * e1 ; x[i] + y[0] = 0 ; i[0:3)
   * e2 ; x[i] + y[1] = 0 ; i[3:7)
   * e3 ; x[i] + y[2] = 0 ; i[7:9)
   * e4 ; y[i] = 12       ; i[0:3)
   */
  MatchingGraph<Variable, Equation> testCase8();

  /**
   * var
   * x[5]
   * y[5]
   *
   * equ
   * e1 ; x[i] - y[i] = 0                      ; i[0:5)
   * e2 ; x[0] + x[1] + x[2] + x[3] + x[4] = 2 ;
   * e3 ; y[0] + y[1] + y[2] + y[3] + y[4] = 3 ;
   * e4 ; x[0] - x[1] + x[2] + x[3] + x[4] = 2 ;
   * e5 ; y[0] + y[1] - y[2] + y[3] + y[4] = 3 ;
   * e6 ; x[0] + x[1] + x[2] - x[3] + x[4] = 2 ;
   */
  MatchingGraph<Variable, Equation> testCase9();

  /**
   * var
   * x[2]
   * y[2]
   *
   * equ
   * e1 ; x[i] - y[i] = 0 ; i[0:2)
   * e2 ; x[0] + x[1] = 2 ;
   * e3 ; y[0] + y[1] = 3 ;
   */
  MatchingGraph<Variable, Equation> testCase10();
}

#endif //MARCO_MATCHING_TEST_REALSCENARIOS_H
