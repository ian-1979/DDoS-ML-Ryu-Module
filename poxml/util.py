
# <editor-fold desc="PRINT UTIL FUNCTIONS">

def print_err(string):
    return "\033[1;31;40m"+ string + "\033[0m"

def print_success(test_name):
    print(test_name, ":", "\033[1;32;40m all tests passed \033[0m")

# </editor-fold>