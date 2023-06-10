#!/usr/bin/python3

import sys

def process_lines(lines, Nu, Nh, Nv):
    count = 0

    for line in lines:
        count += 1

        if count == 72:
            new_line = "Real[" + str(Nu) + ", " + str(Nh) + ", " + str(Nv) + "] 'T_tilde'(nominal = fill(500.0, 3, 4, 6), fixed = true, start = 493.15);"
            print(new_line)
        elif count == 77:
            new_line = "Real[" + str(Nu) + ", " + str(Nh) + ", " + str(Nv) + "] 'T_w'(nominal = fill(500.0, 3, 4, 6), fixed = true, start = 493.15);"
            print(new_line)
        elif count == 78:
            new_line = "output Real[" + str(Nu) + "] 'T_m'(fixed = true, start = 493.15);"
            print(new_line)
        elif count == 97:
            print("for 'i' in 1:" + str(Nu) + " loop")
            print("for 'j' in 1:" + str(Nh) + " loop")
            print("for 'k' in 1:" + str(Nv) + " loop")
            print("'T_tilde'['i','j','k'] = 'T'['i','j','k' + 1];")
            print("end for;")
            print("end for;")
            print("end for;")
        elif count == 100:
            separator = "sum('h'['i','j'," + str(Nv + 1) + "] for 'j' in 1:" + str(Nh) + ")"
            substrings = line.split(separator)
            new_line = substrings[0] + "("

            for j in range(1,Nh + 1):
                if j != 1:
                    new_line += " + "
                new_line += "'h'['i'," + str(j) + "," + str(Nv + 1) + "]"

            new_line += ")" + substrings[1]
            print(new_line)
        else:
            print(line)

def main(argv):
    if len(sys.argv) != 5:
        print("Usage:", sys.argv[0], " file.mo Nu Nh Nv")
        exit(-1)

    with open(sys.argv[1], 'r') as input_file:
        lines = input_file.readlines()
        process_lines(lines, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))


if __name__ == "__main__":
    main(sys.argv)
