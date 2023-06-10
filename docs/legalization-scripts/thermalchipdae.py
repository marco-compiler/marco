#!/usr/bin/python3

import sys

def process_lines(lines, N, M, P):
    count = 0

    for line in lines:
        count += 1

        if count == 20:
            new_line = "Real[" + str(N) + "," + str(M) + "," + str(P) + "] 'T'(nominal = fill(500.0, 4, 5, 6), fixed = true, start = 313.15);"
            print(new_line)
        elif count == 36:
            print("for 'i' in 1:" + str(N) + " loop")
            print("for 'j' in 1:" + str(int(M/2)) + " loop")
            rhs = line.split(" = ")[1]
            rhs = rhs[5:]
            rhs = rhs[:-1]
            rhs = rhs.split(",")
            print("'Qb'['i','j'] = " + rhs[0] + ";");
            print("end for;")
            print("end for;")
        elif count == 37:
            print("for 'i' in 1:" + str(N) + " loop")
            print("for 'j' in " + str(int(M/2) + 1) + ":" + str(M) + " loop")
            rhs = line.split(" = ")[1]
            rhs = rhs[5:]
            rhs = rhs[:-1]
            rhs = rhs.split(",")
            print("'Qb'['i','j'] = " + rhs[0] + ";");
            print("end for;")
            print("end for;")
        else:
            print(line)

def main(argv):
    if len(sys.argv) != 5:
        print("Usage:", sys.argv[0], " file.mo N M P")
        exit(-1)

    with open(sys.argv[1], 'r') as input_file:
        lines = input_file.readlines()
        process_lines(lines, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))


if __name__ == "__main__":
    main(sys.argv)
