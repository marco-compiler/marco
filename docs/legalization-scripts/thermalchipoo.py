#!/usr/bin/python3

import sys

def process_lines(lines, N, M, P):
    count = 0

    for line in lines:
        count += 1

        if count == 20:
            #			new_line = "Real[" + str(N) + "," + str(M) + "," + str(P) + "] 'vol.T'(each fixed = true, each start = 313.15);"
            new_line = "Real[" + str(N) + "," + str(M) + "," + str(P) + "] 'vol.T';"
            print(new_line)
        elif count == 44:
            print("Real[" + str(N) + ", " + str(M) + "] 'Tsource.T';")
        elif count == 57:
            rhs = line.split(" = {")[1]
            rhs = rhs.split(" ")
            qsourceq_value = rhs[0]
            qsourceq_dim0 = rhs[4].split(":")[1][:-1]
            qsourceq_dim1 = rhs[7].split(":")[1][:-1]
            print("Real[" + str(qsourceq_dim0) + ", " + str(qsourceq_dim1) + "] 'Qsource.Q';")
        elif count == 117:
            print("for 'i' in 1:" + str(N) + " loop")
            print("for 'j' in 1:" + str(M) + " loop")
            print("for 'k' in 1:" + str(P) + " loop")
            print("'vol.center.T'['i','j','k'] = 'vol.T'['i','j','k'];")
            print("end for;")
            print("end for;")
            print("end for;")
        elif count == 118:
            print("for 'i' in 1:" + str(N) + " loop")
            print("for 'j' in 1:" + str(M) + " loop")
            print("'Tsource.port.T'['i','j'] = 'Tsource.T'['i','j'];")
            print("end for;")
            print("end for;")
        elif count == 243:
            print("for 'i' in 1:" + str(N) + " loop")
            print("for 'j' in 1:" + str(M) + " loop")
            print("'Tsource.T'['i','j'] = 313.15;")
            print("end for;")
            print("end for;")

            print("for 'i' in 1:" + str(qsourceq_dim0) + " loop")
            print("for 'j' in 1:" + str(qsourceq_dim1) + " loop")
            print("'Qsource.Q'['i','j'] = " + str(qsourceq_value) + ";")
            print("end for;")
            print("end for;")

            print("initial equation")
            print("for 'i' in 1:" + str(N) + " loop")
            print("for 'j' in 1:" + str(M) + " loop")
            print("for 'k' in 1:" + str(P) + " loop")
            print("'vol.T'['i','j','k'] = 313.15;")
            print("end for;")
            print("end for;")
            print("end for;")

            print("equation")

            print(line)
        elif count >= 35 and count <= 40:
            continue
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
