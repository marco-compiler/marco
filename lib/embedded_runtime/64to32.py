with open("test/example.ll") as file:
    result = open("test/nomain.ll","a")
    for l in file:
        string = l
        if "@printf" in string:
            result.write(string.replace(" i64 ", " i32 "))
        else:
            result.write(string)
    result.close()